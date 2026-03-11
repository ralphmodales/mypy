from __future__ import annotations

import re as stdlib_re
from typing import Final, NamedTuple

from mypy import message_registry
from mypy.nodes import (
    AssignmentStmt,
    BytesExpr,
    CallExpr,
    ClassDef,
    IndexExpr,
    MemberExpr,
    NameExpr,
    RefExpr,
    StrExpr,
    Var,
)
from mypy.plugin import CheckerPluginInterface, FunctionContext, MethodContext
from mypy.typeops import try_getting_int_literals_from_type, try_getting_str_literals
from mypy.types import Instance, NoneType, TupleType, Type, UnionType, get_proper_type


_ESCAPED_BACKSLASH_RE: Final = stdlib_re.compile(r"\\\\")
_NUMERIC_BACKREF_RE: Final = stdlib_re.compile(r"\\(\d+)")
_NAMED_BACKREF_RE: Final = stdlib_re.compile(r"\\g<([^>]+)>")

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


_PATTERN_ANALYSIS_CACHE: dict[str | bytes, PatternInfo] = {}


def analyze_pattern(pattern: str | bytes) -> PatternInfo:
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


def _get_pattern_from_expr(
    ctx: FunctionContext | MethodContext, arg_index: int
) -> str | bytes | None:
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
            message_registry.INVALID_RE_PATTERN.format(info.error_msg),
            ctx.args[arg_index][0],
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
                message_registry.INVALID_RE_REPLACEMENT_REF.format(
                    group_num, info.group_count
                ),
                ctx.args[repl_arg_index][0],
            )

    for m in _NAMED_BACKREF_RE.finditer(cleaned):
        name = m.group(1)
        if name.isdigit():
            group_num = int(name)
            if group_num == 0:
                continue
            if group_num > info.group_count:
                ctx.api.fail(
                    message_registry.INVALID_RE_REPLACEMENT_REF.format(
                        group_num, info.group_count
                    ),
                    ctx.args[repl_arg_index][0],
                )
        elif name not in info.named_groups:
            ctx.api.fail(
                message_registry.INVALID_RE_REPLACEMENT_NAME.format(
                    name, sorted(info.named_groups)
                ),
                ctx.args[repl_arg_index][0],
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
    api: CheckerPluginInterface, var: Var, context_line: int | None
) -> CallExpr | None:
    from mypy.checker import TypeChecker

    if not isinstance(api, TypeChecker):
        return None

    def statement_lists() -> list[list[object]]:
        out: list[list[object]] = []
        func = api.scope.current_function()
        if func is not None and getattr(func, "body", None) is not None:
            body = func.body
            if getattr(body, "body", None) is not None:
                out.append(body.body)
        info = api.scope.active_class()
        if info is not None and getattr(info, "defn", None) is not None:
            defn = info.defn
            if isinstance(defn, ClassDef):
                defs = getattr(defn, "defs", None)
                if defs is not None and getattr(defs, "body", None) is not None:
                    out.append(defs.body)
        out.append(api.tree.defs)
        return out

    best: CallExpr | None = None
    best_line = -1
    for stmts in statement_lists():
        for stmt in stmts:
            if not isinstance(stmt, AssignmentStmt):
                continue
            if not isinstance(stmt.rvalue, CallExpr):
                continue
            stmt_line = getattr(stmt, "line", -1)
            if context_line is not None and stmt_line > context_line:
                continue
            for lval in stmt.lvalues:
                if isinstance(lval, NameExpr) and lval.node is var:
                    if stmt_line >= best_line:
                        best = stmt.rvalue
                        best_line = stmt_line
    return best


def _extract_pattern_from_call(call: CallExpr) -> str | bytes | None:
    callee = call.callee
    fullname: str | None = None
    if isinstance(callee, RefExpr):
        fullname = callee.fullname
    if fullname is None:
        return None
    if fullname not in _RE_FUNC_FULLNAMES:
        return None
    if call.args and isinstance(call.args[0], (StrExpr, BytesExpr)):
        return call.args[0].value
    return None


def _resolve_pattern_for_var(ctx: MethodContext, var: Var) -> PatternInfo | None:
    context_line = getattr(ctx.context, "line", None)
    call = _find_assignment_rvalue_call(ctx.api, var, context_line)
    if call is None:
        return None

    pattern_str = _extract_pattern_from_call(call)
    if pattern_str is None:
        return None

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
                        message_registry.INVALID_RE_GROUP_NAME.format(
                            name, sorted(info.named_groups)
                        ),
                        arg_expr,
                    )
            continue

        int_literals = try_getting_int_literals_from_type(arg_type)
        if int_literals is not None:
            for idx in int_literals:
                if idx == 0:
                    continue
                if idx < 0 or idx > info.group_count:
                    ctx.api.fail(
                        message_registry.INVALID_RE_GROUP_INDEX.format(
                            idx, info.group_count
                        ),
                        arg_expr,
                    )


def re_compile_callback(ctx: FunctionContext) -> Type:
    pattern_str = _get_pattern_from_expr(ctx, arg_index=0)
    if pattern_str is not None:
        info = analyze_pattern(pattern_str)
        _report_invalid_pattern(ctx, info, arg_index=0)
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

    context_line = getattr(ctx.context, "line", None)
    call = _find_assignment_rvalue_call(ctx.api, var, context_line)
    if call is None:
        return None

    callee = call.callee

    if isinstance(callee, MemberExpr) and callee.name in ("match", "search", "fullmatch"):
        if callee.fullname in ("re.match", "re.search", "re.fullmatch"):
            if call.args and isinstance(call.args[0], (StrExpr, BytesExpr)):
                return analyze_pattern(call.args[0].value)
            return None

        receiver = callee.expr
        if isinstance(receiver, NameExpr) and isinstance(receiver.node, Var):
            info = _resolve_pattern_for_var(ctx, receiver.node)
            if info is not None and info.is_valid:
                return info
            return None
        if isinstance(receiver, CallExpr):
            recv_callee = receiver.callee
            if isinstance(recv_callee, RefExpr) and recv_callee.fullname == "re.compile":
                if receiver.args and isinstance(receiver.args[0], (StrExpr, BytesExpr)):
                    info = analyze_pattern(receiver.args[0].value)
                    return info if info.is_valid else None
        return None

    if isinstance(callee, RefExpr):
        fullname = callee.fullname
        if fullname in ("re.match", "re.search", "re.fullmatch"):
            if call.args and isinstance(call.args[0], (StrExpr, BytesExpr)):
                return analyze_pattern(call.args[0].value)
        return None

    return None
