from __future__ import annotations

import re as stdlib_re
from typing import Final, NamedTuple

import mypy.errorcodes as codes
import mypy.plugin
from mypy.nodes import (
    AssignmentStmt,
    BytesExpr,
    CallExpr,
    IndexExpr,
    MemberExpr,
    NameExpr,
    RefExpr,
    StrExpr,
    Var,
)
from mypy.plugin import CheckerPluginInterface, FunctionContext, MethodContext
from mypy.typeops import try_getting_int_literals_from_type, try_getting_str_literals
from mypy.types import (
    AnyType,
    Instance,
    NoneType,
    TupleType,
    Type,
    TypeOfAny,
    UnionType,
    get_proper_type,
)


_ESCAPED_BACKSLASH_RE: Final = stdlib_re.compile(r"\\\\")
_NUMERIC_BACKREF_RE: Final = stdlib_re.compile(r"\\(\d+)")
_NAMED_BACKREF_RE: Final = stdlib_re.compile(r"\\g<([^>]+)>")

_COMPILE_PATTERN_REGISTRY: dict[int, str] = {}

_RE_COMPILE_FULLNAMES: Final = frozenset({
    "re.compile",
})

_RE_FUNC_FULLNAMES: Final = frozenset({
    "re.compile",
    "re.match",
    "re.search",
    "re.fullmatch",
    "re.findall",
    "re.finditer",
    "re.sub",
    "re.subn",
    "re.split",
})


class PatternInfo(NamedTuple):
    group_count: int
    named_groups: dict[str, int]
    is_valid: bool
    error_msg: str | None


_PATTERN_ANALYSIS_CACHE: dict[str, PatternInfo] = {}


def analyze_pattern(pattern: str) -> PatternInfo:
    cached = _PATTERN_ANALYSIS_CACHE.get(pattern)
    if cached is not None:
        return cached
    try:
        compiled = stdlib_re.compile(pattern)
        info = PatternInfo(
            group_count=compiled.groups,
            named_groups=dict(compiled.groupindex),
            is_valid=True,
            error_msg=None,
        )
    except stdlib_re.error as e:
        info = PatternInfo(
            group_count=0,
            named_groups={},
            is_valid=False,
            error_msg=str(e),
        )
    _PATTERN_ANALYSIS_CACHE[pattern] = info
    return info


def _get_pattern_from_expr(ctx: FunctionContext | MethodContext, arg_index: int) -> str | None:
    if arg_index >= len(ctx.args) or not ctx.args[arg_index]:
        return None

    expr = ctx.args[arg_index][0]
    if isinstance(expr, StrExpr):
        return expr.value

    if isinstance(expr, BytesExpr):
        return expr.value

    if arg_index < len(ctx.arg_types) and ctx.arg_types[arg_index]:
        literals = try_getting_str_literals(expr, ctx.arg_types[arg_index][0])
        if literals and len(literals) == 1:
            return literals[0]

    return None


def _is_bytes_pattern(ctx: FunctionContext | MethodContext, arg_index: int) -> bool:
    if arg_index >= len(ctx.args) or not ctx.args[arg_index]:
        return False

    expr = ctx.args[arg_index][0]
    if isinstance(expr, BytesExpr):
        return True

    if arg_index < len(ctx.arg_types) and ctx.arg_types[arg_index]:
        arg_type = get_proper_type(ctx.arg_types[arg_index][0])
        if isinstance(arg_type, Instance) and arg_type.type.fullname == "builtins.bytes":
            return True

    return False


def _is_bytes_type_from_instance(obj_type: Instance) -> bool:
    if obj_type.args:
        arg = get_proper_type(obj_type.args[0])
        if isinstance(arg, Instance) and arg.type.fullname == "builtins.bytes":
            return True
    return False


def _get_element_type(ctx: FunctionContext | MethodContext, is_bytes: bool) -> Type:
    if is_bytes:
        return ctx.api.named_generic_type("builtins.bytes", [])
    return ctx.api.named_generic_type("builtins.str", [])


def _report_invalid_pattern(
    ctx: FunctionContext | MethodContext, info: PatternInfo, arg_index: int
) -> None:
    if not info.is_valid and arg_index < len(ctx.args) and ctx.args[arg_index]:
        ctx.api.fail(
            f"Invalid regex pattern: {info.error_msg}",
            ctx.args[arg_index][0],
            code=codes.RE_PATTERN,
        )


def _strip_escaped_backslashes(repl: str) -> str:
    return _ESCAPED_BACKSLASH_RE.sub("", repl)


def _validate_replacement_refs(
    ctx: FunctionContext | MethodContext,
    repl: str,
    info: PatternInfo,
    repl_arg_index: int,
) -> None:
    cleaned = _strip_escaped_backslashes(repl)

    for m in _NUMERIC_BACKREF_RE.finditer(cleaned):
        group_num = int(m.group(1))
        if group_num == 0:
            continue
        if group_num > info.group_count:
            ctx.api.fail(
                f"Replacement references group {group_num}, "
                f"but pattern only has {info.group_count} group(s)",
                ctx.args[repl_arg_index][0],
                code=codes.RE_GROUP,
            )

    for m in _NAMED_BACKREF_RE.finditer(cleaned):
        name = m.group(1)
        if name.isdigit():
            group_num = int(name)
            if group_num == 0:
                continue
            if group_num > info.group_count:
                ctx.api.fail(
                    f"Replacement references group {group_num}, "
                    f"but pattern only has {info.group_count} group(s)",
                    ctx.args[repl_arg_index][0],
                    code=codes.RE_GROUP,
                )
        elif name not in info.named_groups:
            ctx.api.fail(
                f'Replacement references unknown group name "{name}"; '
                f"pattern has groups: {sorted(info.named_groups)}",
                ctx.args[repl_arg_index][0],
                code=codes.RE_GROUP,
            )


def _build_findall_type(
    ctx: FunctionContext | MethodContext, info: PatternInfo, element_type: Type
) -> Type:
    if info.group_count == 0 or info.group_count == 1:
        return ctx.api.named_generic_type("builtins.list", [element_type])
    items = [element_type] * info.group_count
    tuple_fallback = ctx.api.named_generic_type("builtins.tuple", [element_type])
    tuple_type = TupleType(items, tuple_fallback)
    return ctx.api.named_generic_type("builtins.list", [tuple_type])


def _build_split_type(
    ctx: FunctionContext | MethodContext, info: PatternInfo, element_type: Type
) -> Type:
    if info.group_count == 0:
        return ctx.api.named_generic_type("builtins.list", [element_type])
    none_type = NoneType()
    union_type = UnionType.make_union([element_type, none_type])
    return ctx.api.named_generic_type("builtins.list", [union_type])


def _get_repl_string(
    ctx: FunctionContext | MethodContext, repl_arg_index: int
) -> list[str] | None:
    if repl_arg_index >= len(ctx.args) or not ctx.args[repl_arg_index]:
        return None

    repl_expr = ctx.args[repl_arg_index][0]
    if isinstance(repl_expr, StrExpr):
        return [repl_expr.value]

    if repl_arg_index < len(ctx.arg_types) and ctx.arg_types[repl_arg_index]:
        return try_getting_str_literals(repl_expr, ctx.arg_types[repl_arg_index][0])

    return None


def _validate_repl_for_pattern(
    ctx: FunctionContext | MethodContext,
    info: PatternInfo,
    repl_arg_index: int,
) -> None:
    repl_literals = _get_repl_string(ctx, repl_arg_index)
    if repl_literals:
        for repl_str in repl_literals:
            _validate_replacement_refs(ctx, repl_str, info, repl_arg_index)


def _find_var_from_method_context(ctx: MethodContext) -> Var | None:
    context = ctx.context
    if isinstance(context, CallExpr):
        callee = context.callee
        if not isinstance(callee, MemberExpr):
            return None
        receiver = callee.expr
        if not isinstance(receiver, NameExpr):
            return None
        node = receiver.node
        if isinstance(node, Var):
            return node
    elif isinstance(context, IndexExpr):
        base = context.base
        if not isinstance(base, NameExpr):
            return None
        node = base.node
        if isinstance(node, Var):
            return node
    return None


def _find_assignment_rvalue_call(
    api: CheckerPluginInterface, var: Var
) -> CallExpr | None:
    from mypy.checker import TypeChecker

    if not isinstance(api, TypeChecker):
        return None

    tree = api.tree
    for stmt in tree.defs:
        if not isinstance(stmt, AssignmentStmt):
            continue
        if not isinstance(stmt.rvalue, CallExpr):
            continue
        for lval in stmt.lvalues:
            if isinstance(lval, NameExpr) and lval.node is var:
                return stmt.rvalue
    return None


def _extract_pattern_from_call(call: CallExpr) -> str | None:
    callee = call.callee
    fullname: str | None = None
    if isinstance(callee, RefExpr):
        fullname = callee.fullname
    if fullname is None:
        return None
    if fullname not in _RE_FUNC_FULLNAMES:
        return None
    if call.args and isinstance(call.args[0], StrExpr):
        return call.args[0].value
    return None


def _resolve_pattern_for_var(ctx: MethodContext, var: Var) -> PatternInfo | None:
    var_id = id(var)
    cached_pat = _COMPILE_PATTERN_REGISTRY.get(var_id)
    if cached_pat is not None:
        return analyze_pattern(cached_pat)

    call = _find_assignment_rvalue_call(ctx.api, var)
    if call is None:
        return None

    pattern_str = _extract_pattern_from_call(call)
    if pattern_str is None:
        return None

    _COMPILE_PATTERN_REGISTRY[var_id] = pattern_str
    return analyze_pattern(pattern_str)


def _resolve_pattern_for_method(ctx: MethodContext) -> PatternInfo | None:
    var = _find_var_from_method_context(ctx)
    if var is None:
        return None
    return _resolve_pattern_for_var(ctx, var)


def _validate_group_args(ctx: MethodContext, info: PatternInfo) -> None:
    if not ctx.args or not ctx.args[0]:
        return

    for i, arg_expr in enumerate(ctx.args[0]):
        if i >= len(ctx.arg_types[0]):
            break
        arg_type = ctx.arg_types[0][i]

        str_literals = try_getting_str_literals(arg_expr, arg_type)
        if str_literals is not None:
            for name in str_literals:
                if name not in info.named_groups:
                    ctx.api.fail(
                        f'Regex pattern has no group named "{name}"; '
                        f"pattern has groups: {sorted(info.named_groups)}",
                        arg_expr,
                        code=codes.RE_GROUP,
                    )
            continue

        int_literals = try_getting_int_literals_from_type(arg_type)
        if int_literals is not None:
            for idx in int_literals:
                if idx == 0:
                    continue
                if idx < 0 or idx > info.group_count:
                    ctx.api.fail(
                        f"Regex group index {idx} is out of range "
                        f"(pattern has {info.group_count} groups)",
                        arg_expr,
                        code=codes.RE_GROUP,
                    )


def re_compile_callback(ctx: FunctionContext) -> Type:
    pattern_str = _get_pattern_from_expr(ctx, arg_index=0)
    if pattern_str is not None:
        info = analyze_pattern(pattern_str)
        _report_invalid_pattern(ctx, info, arg_index=0)
        if info.is_valid and isinstance(ctx.context, CallExpr):
            _COMPILE_PATTERN_REGISTRY[id(ctx.context)] = pattern_str
    return ctx.default_return_type


def re_match_callback(ctx: FunctionContext | MethodContext) -> Type:
    pattern_str = _get_pattern_from_expr(ctx, arg_index=0)
    if pattern_str is not None:
        info = analyze_pattern(pattern_str)
        _report_invalid_pattern(ctx, info, arg_index=0)
    return ctx.default_return_type


def re_findall_callback(ctx: FunctionContext) -> Type:
    pattern_str = _get_pattern_from_expr(ctx, arg_index=0)
    if pattern_str is None:
        return ctx.default_return_type

    info = analyze_pattern(pattern_str)
    if not info.is_valid:
        _report_invalid_pattern(ctx, info, arg_index=0)
        return ctx.default_return_type

    is_bytes = _is_bytes_pattern(ctx, arg_index=0)
    element_type = _get_element_type(ctx, is_bytes)
    return _build_findall_type(ctx, info, element_type)


def re_split_callback(ctx: FunctionContext) -> Type:
    pattern_str = _get_pattern_from_expr(ctx, arg_index=0)
    if pattern_str is None:
        return ctx.default_return_type

    info = analyze_pattern(pattern_str)
    if not info.is_valid:
        _report_invalid_pattern(ctx, info, arg_index=0)
        return ctx.default_return_type

    is_bytes = _is_bytes_pattern(ctx, arg_index=0)
    element_type = _get_element_type(ctx, is_bytes)
    return _build_split_type(ctx, info, element_type)


def re_sub_callback(ctx: FunctionContext | MethodContext) -> Type:
    pattern_str = _get_pattern_from_expr(ctx, arg_index=0)
    if pattern_str is not None:
        info = analyze_pattern(pattern_str)
        if not info.is_valid:
            _report_invalid_pattern(ctx, info, arg_index=0)
            return ctx.default_return_type
        _validate_repl_for_pattern(ctx, info, repl_arg_index=1)
    return ctx.default_return_type


def re_subn_callback(ctx: FunctionContext | MethodContext) -> Type:
    return re_sub_callback(ctx)


def re_pattern_findall_callback(ctx: MethodContext) -> Type:
    obj_type = get_proper_type(ctx.type)
    if not isinstance(obj_type, Instance):
        return ctx.default_return_type

    is_bytes = _is_bytes_type_from_instance(obj_type)
    element_type = _get_element_type(ctx, is_bytes)

    info = _resolve_pattern_for_method(ctx)
    if info is not None and info.is_valid:
        return _build_findall_type(ctx, info, element_type)

    return ctx.default_return_type


def re_pattern_sub_callback(ctx: MethodContext) -> Type:
    info = _resolve_pattern_for_method(ctx)
    if info is not None and info.is_valid:
        _validate_repl_for_pattern(ctx, info, repl_arg_index=0)
    return ctx.default_return_type


def re_pattern_subn_callback(ctx: MethodContext) -> Type:
    return re_pattern_sub_callback(ctx)


def match_group_callback(ctx: MethodContext) -> Type:
    if not ctx.args or not ctx.args[0]:
        return ctx.default_return_type

    info = _resolve_match_pattern(ctx)
    if info is not None and info.is_valid:
        _validate_group_args(ctx, info)

    return ctx.default_return_type


def match_getitem_callback(ctx: MethodContext) -> Type:
    if not ctx.args or not ctx.args[0]:
        return ctx.default_return_type

    info = _resolve_match_pattern(ctx)
    if info is not None and info.is_valid:
        _validate_group_args(ctx, info)

    return ctx.default_return_type


def _resolve_match_pattern(ctx: MethodContext) -> PatternInfo | None:
    var = _find_var_from_method_context(ctx)
    if var is None:
        return None

    call = _find_assignment_rvalue_call(ctx.api, var)
    if call is None:
        return None

    callee = call.callee
    fullname: str | None = None
    if isinstance(callee, RefExpr):
        fullname = callee.fullname

    if fullname is None:
        return None

    if fullname in ("re.match", "re.search", "re.fullmatch"):
        if call.args and isinstance(call.args[0], StrExpr):
            return analyze_pattern(call.args[0].value)
        return None

    if fullname == "re.compile":
        if call.args and isinstance(call.args[0], StrExpr):
            pattern_str = call.args[0].value
            info = analyze_pattern(pattern_str)
            if not info.is_valid:
                return None
            return _resolve_match_from_pattern_var(ctx, call)
        return None

    return None


def _resolve_match_from_pattern_var(
    ctx: MethodContext, compile_call: CallExpr
) -> PatternInfo | None:
    if not compile_call.args or not isinstance(compile_call.args[0], StrExpr):
        return None

    from mypy.checker import TypeChecker

    if not isinstance(ctx.api, TypeChecker):
        return None

    var = _find_var_from_method_context(ctx)
    if var is None:
        return None

    tree = ctx.api.tree
    for stmt in tree.defs:
        if not isinstance(stmt, AssignmentStmt):
            continue
        if not isinstance(stmt.rvalue, CallExpr):
            continue
        call = stmt.rvalue
        call_callee = call.callee
        if not isinstance(call_callee, MemberExpr):
            continue
        receiver = call_callee.expr
        if not isinstance(receiver, NameExpr):
            continue
        receiver_node = receiver.node
        if not isinstance(receiver_node, Var):
            continue
        pat_str = _COMPILE_PATTERN_REGISTRY.get(id(receiver_node))
        if pat_str is not None:
            for lval in stmt.lvalues:
                if isinstance(lval, NameExpr) and lval.node is var:
                    return analyze_pattern(pat_str)

    return None
