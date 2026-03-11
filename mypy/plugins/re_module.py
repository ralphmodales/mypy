from __future__ import annotations

import re as stdlib_re
from typing import Final, NamedTuple

from mypy import message_registry
from mypy.nodes import (
    AssignmentStmt,
    BytesExpr,
    CallExpr,
    ClassDef,
    Expression,
    IndexExpr,
    IntExpr,
    MemberExpr,
    NameExpr,
    OpExpr,
    RefExpr,
    StrExpr,
    UnaryExpr,
    Var,
)
from mypy.plugin import CheckerPluginInterface, FunctionContext, MethodContext
from mypy.typeops import try_getting_int_literals_from_type, try_getting_str_literals
from mypy.types import Instance, NoneType, TupleType, Type, UnionType, get_proper_type


_RE_FUNC_FULLNAMES: Final = frozenset(
    {
        "re.compile",
        "re.match",
        "re.search",
        "re.fullmatch",
        "re.findall",
        "re.finditer",
        "re.sub",
        "re.subn",
        "re.split",
    }
)

_FLAG_FULLNAMES: Final[dict[str, int]] = {
    "re.A": stdlib_re.A,
    "re.ASCII": stdlib_re.ASCII,
    "re.DEBUG": stdlib_re.DEBUG,
    "re.I": stdlib_re.I,
    "re.IGNORECASE": stdlib_re.IGNORECASE,
    "re.L": stdlib_re.L,
    "re.LOCALE": stdlib_re.LOCALE,
    "re.M": stdlib_re.M,
    "re.MULTILINE": stdlib_re.MULTILINE,
    "re.S": stdlib_re.S,
    "re.DOTALL": stdlib_re.DOTALL,
    "re.U": stdlib_re.U,
    "re.UNICODE": stdlib_re.UNICODE,
    "re.X": stdlib_re.X,
    "re.VERBOSE": stdlib_re.VERBOSE,
    "re.RegexFlag.A": stdlib_re.A,
    "re.RegexFlag.ASCII": stdlib_re.ASCII,
    "re.RegexFlag.DEBUG": stdlib_re.DEBUG,
    "re.RegexFlag.I": stdlib_re.I,
    "re.RegexFlag.IGNORECASE": stdlib_re.IGNORECASE,
    "re.RegexFlag.L": stdlib_re.L,
    "re.RegexFlag.LOCALE": stdlib_re.LOCALE,
    "re.RegexFlag.M": stdlib_re.M,
    "re.RegexFlag.MULTILINE": stdlib_re.MULTILINE,
    "re.RegexFlag.S": stdlib_re.S,
    "re.RegexFlag.DOTALL": stdlib_re.DOTALL,
    "re.RegexFlag.U": stdlib_re.U,
    "re.RegexFlag.UNICODE": stdlib_re.UNICODE,
    "re.RegexFlag.X": stdlib_re.X,
    "re.RegexFlag.VERBOSE": stdlib_re.VERBOSE,
}


class PatternInfo(NamedTuple):
    group_count: int
    named_groups: dict[str, int]
    is_valid: bool
    error_msg: str | None


_PATTERN_ANALYSIS_CACHE: dict[tuple[str | bytes, int], PatternInfo] = {}


def analyze_pattern(pattern: str | bytes, flags: int) -> PatternInfo:
    key = (pattern, flags)
    cached = _PATTERN_ANALYSIS_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        compiled = stdlib_re.compile(pattern, flags)
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
    _PATTERN_ANALYSIS_CACHE[key] = info
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


def _formal_index(ctx: FunctionContext, name: str) -> int | None:
    for i, n in enumerate(ctx.callee_arg_names):
        if n == name:
            return i
    return None


def _try_eval_flags_expr(expr: Expression) -> int | None:
    if isinstance(expr, IntExpr):
        return expr.value

    if isinstance(expr, UnaryExpr) and expr.op == "-" and isinstance(expr.expr, IntExpr):
        return -expr.expr.value

    if isinstance(expr, OpExpr) and expr.op in {"|", "&", "^"}:
        left = _try_eval_flags_expr(expr.left)
        right = _try_eval_flags_expr(expr.right)
        if left is None or right is None:
            return None
        if expr.op == "|":
            return left | right
        if expr.op == "&":
            return left & right
        return left ^ right

    if isinstance(expr, RefExpr):
        if expr.fullname in _FLAG_FULLNAMES:
            return _FLAG_FULLNAMES[expr.fullname]

    if isinstance(expr, MemberExpr):
        if expr.fullname in _FLAG_FULLNAMES:
            return _FLAG_FULLNAMES[expr.fullname]

    return None


def _ref_fullname(expr: Expression) -> str | None:
    if isinstance(expr, RefExpr) and expr.fullname:
        return expr.fullname
    if isinstance(expr, MemberExpr) and expr.fullname:
        return expr.fullname
    return None


def _is_re_compile_callee(expr: Expression) -> bool:
    fullname = _ref_fullname(expr)
    if fullname == "re.compile":
        return True
    if isinstance(expr, MemberExpr) and expr.name == "compile":
        if isinstance(expr.expr, NameExpr) and expr.expr.fullname == "re":
            return True
    return False


def _flags_from_ctx(ctx: FunctionContext) -> int | None:
    idx = _formal_index(ctx, "flags")
    if idx is None:
        return 0
    if idx >= len(ctx.args) or not ctx.args[idx]:
        return 0
    flags_expr = ctx.args[idx][0]
    v = _try_eval_flags_expr(flags_expr)
    return v


def _report_invalid_pattern(
    ctx: FunctionContext | MethodContext,
    info: PatternInfo,
    arg_index: int,
) -> None:
    if not info.is_valid and arg_index < len(ctx.args) and ctx.args[arg_index]:
        ctx.api.fail(
            message_registry.INVALID_RE_PATTERN.format(info.error_msg),
            ctx.args[arg_index][0],
        )


def _iter_replacement_refs(repl: str) -> tuple[list[int], list[str]]:
    numeric: list[int] = []
    named: list[str] = []
    i = 0
    n = len(repl)
    while i < n:
        c = repl[i]
        if c != "\\":
            i += 1
            continue
        if i + 1 >= n:
            break
        nxt = repl[i + 1]
        if nxt == "\\":
            i += 2
            continue
        if nxt == "g" and i + 2 < n and repl[i + 2] == "<":
            j = i + 3
            while j < n and repl[j] != ">":
                j += 1
            if j >= n:
                i += 2
                continue
            name = repl[i + 3 : j]
            if name.isdigit():
                numeric.append(int(name))
            else:
                named.append(name)
            i = j + 1
            continue
        if nxt.isdigit():
            if nxt == "0":
                j = i + 2
                k = 0
                while j < n and k < 2 and repl[j] in "01234567":
                    j += 1
                    k += 1
                i = j
                continue
            j = i + 1
            while j < n and repl[j].isdigit():
                j += 1
            numeric.append(int(repl[i + 1 : j]))
            i = j
            continue
        i += 2
    return numeric, named


def _validate_replacement_refs(
    ctx: FunctionContext | MethodContext,
    repl: str,
    info: PatternInfo,
    repl_arg_index: int,
) -> None:
    nums, names = _iter_replacement_refs(repl)

    for group_num in nums:
        if group_num == 0:
            continue
        if group_num > info.group_count:
            ctx.api.fail(
                message_registry.INVALID_RE_REPLACEMENT_REF.format(
                    group_num, info.group_count
                ),
                ctx.args[repl_arg_index][0],
            )

    for name in names:
        if name not in info.named_groups:
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


def _find_name_from_method_context(ctx: MethodContext) -> str | None:
    context = ctx.context
    if isinstance(context, CallExpr):
        callee = context.callee
        if isinstance(callee, MemberExpr) and isinstance(callee.expr, NameExpr):
            return callee.expr.name
    elif isinstance(context, IndexExpr):
        if isinstance(context.base, NameExpr):
            return context.base.name
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


def _find_assignment_rvalue_call_by_name(
    api: CheckerPluginInterface, name: str, context_line: int | None
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
                if isinstance(lval, NameExpr) and lval.name == name:
                    if stmt_line >= best_line:
                        best = stmt.rvalue
                        best_line = stmt_line
    return best


def _extract_pattern_flags_from_call(call: CallExpr) -> tuple[str | bytes, int] | None:
    callee_fullname = _ref_fullname(call.callee)
    if callee_fullname is None or callee_fullname not in _RE_FUNC_FULLNAMES:
        return None

    if not call.args or not isinstance(call.args[0], (StrExpr, BytesExpr)):
        return None
    pattern = call.args[0].value

    flags_expr: Expression | None = None
    for arg, name in zip(call.args, call.arg_names):
        if name == "flags":
            flags_expr = arg
            break
    if flags_expr is None:
        if len(call.args) >= 3 and call.arg_names[2] is None and callee_fullname in {
            "re.match",
            "re.search",
            "re.fullmatch",
            "re.findall",
            "re.finditer",
            "re.split",
        }:
            flags_expr = call.args[2]
        elif len(call.args) >= 5 and call.arg_names[4] is None and callee_fullname in {
            "re.sub",
            "re.subn",
        }:
            flags_expr = call.args[4]
        elif len(call.args) >= 2 and call.arg_names[1] is None and callee_fullname == "re.compile":
            flags_expr = call.args[1]

    if flags_expr is None:
        flags = 0
    else:
        flags_val = _try_eval_flags_expr(flags_expr)
        if flags_val is None:
            return None
        flags = flags_val

    return pattern, flags


def _receiver_expr_from_method_context(ctx: MethodContext) -> Expression | None:
    if isinstance(ctx.context, CallExpr) and isinstance(ctx.context.callee, MemberExpr):
        return ctx.context.callee.expr
    return None


def _resolve_compiled_pattern_from_expr(
    ctx: MethodContext, receiver: Expression
) -> tuple[PatternInfo, int] | None:
    context_line = getattr(ctx.context, "line", None)

    if isinstance(receiver, NameExpr) and isinstance(receiver.node, Var):
        call = _find_assignment_rvalue_call(ctx.api, receiver.node, context_line)
        if call is None:
            return None
        if not _is_re_compile_callee(call.callee):
            return None
        if not call.args or not isinstance(call.args[0], (StrExpr, BytesExpr)):
            return None
        pattern = call.args[0].value
        flags = 0
        flags_expr: Expression | None = None
        for arg, name in zip(call.args, call.arg_names):
            if name == "flags":
                flags_expr = arg
                break
        if flags_expr is None and len(call.args) >= 2 and call.arg_names[1] is None:
            flags_expr = call.args[1]
        if flags_expr is not None:
            flags_val = _try_eval_flags_expr(flags_expr)
            if flags_val is None:
                return None
            flags = flags_val
        info = analyze_pattern(pattern, flags)
        return info, flags

    if isinstance(receiver, NameExpr) and receiver.node is None:
        call = _find_assignment_rvalue_call_by_name(ctx.api, receiver.name, context_line)
        if call is None:
            return None
        if not _is_re_compile_callee(call.callee):
            return None
        if not call.args or not isinstance(call.args[0], (StrExpr, BytesExpr)):
            return None
        pattern = call.args[0].value
        flags = 0
        flags_expr: Expression | None = None
        for arg, name in zip(call.args, call.arg_names):
            if name == "flags":
                flags_expr = arg
                break
        if flags_expr is None and len(call.args) >= 2 and call.arg_names[1] is None:
            flags_expr = call.args[1]
        if flags_expr is not None:
            flags_val = _try_eval_flags_expr(flags_expr)
            if flags_val is None:
                return None
            flags = flags_val
        info = analyze_pattern(pattern, flags)
        return info, flags

    if isinstance(receiver, CallExpr):
        if not _is_re_compile_callee(receiver.callee):
            return None
        if not receiver.args or not isinstance(receiver.args[0], (StrExpr, BytesExpr)):
            return None
        pattern = receiver.args[0].value
        flags = 0
        flags_expr: Expression | None = None
        for arg, name in zip(receiver.args, receiver.arg_names):
            if name == "flags":
                flags_expr = arg
                break
        if flags_expr is None and len(receiver.args) >= 2 and receiver.arg_names[1] is None:
            flags_expr = receiver.args[1]
        if flags_expr is not None:
            flags_val = _try_eval_flags_expr(flags_expr)
            if flags_val is None:
                return None
            flags = flags_val
        info = analyze_pattern(pattern, flags)
        return info, flags

    return None


def _resolve_match_pattern(ctx: MethodContext) -> PatternInfo | None:
    context_line = getattr(ctx.context, "line", None)
    var = _find_var_from_method_context(ctx)
    if var is not None:
        call = _find_assignment_rvalue_call(ctx.api, var, context_line)
    else:
        name = _find_name_from_method_context(ctx)
        if name is None:
            return None
        call = _find_assignment_rvalue_call_by_name(ctx.api, name, context_line)
    if call is None:
        return None

    if isinstance(call.callee, MemberExpr) and call.callee.name in {"match", "search", "fullmatch"}:
        fullname = call.callee.fullname
        if fullname in {"re.match", "re.search", "re.fullmatch"}:
            pf = _extract_pattern_flags_from_call(call)
            if pf is None:
                return None
            pattern, flags = pf
            info = analyze_pattern(pattern, flags)
            return info if info.is_valid else None

        receiver = call.callee.expr
        resolved = _resolve_compiled_pattern_from_expr(ctx, receiver)
        if resolved is None:
            return None
        info, _ = resolved
        return info if info.is_valid else None

    if isinstance(call.callee, RefExpr):
        if call.callee.fullname in {"re.match", "re.search", "re.fullmatch"}:
            pf = _extract_pattern_flags_from_call(call)
            if pf is None:
                return None
            pattern, flags = pf
            info = analyze_pattern(pattern, flags)
            return info if info.is_valid else None
        return None

    return None


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
                        message_registry.INVALID_RE_GROUP_INDEX.format(idx, info.group_count),
                        arg_expr,
                    )


def re_compile_callback(ctx: FunctionContext) -> Type:
    pattern = _get_pattern_from_expr(ctx, arg_index=0)
    if pattern is None:
        return ctx.default_return_type

    flags = _flags_from_ctx(ctx)
    if flags is None:
        return ctx.default_return_type

    info = analyze_pattern(pattern, flags)
    if not info.is_valid:
        _report_invalid_pattern(ctx, info, arg_index=0)
    return ctx.default_return_type


def re_match_callback(ctx: FunctionContext) -> Type:
    pattern = _get_pattern_from_expr(ctx, arg_index=0)
    if pattern is None:
        return ctx.default_return_type

    flags = _flags_from_ctx(ctx)
    if flags is None:
        return ctx.default_return_type

    info = analyze_pattern(pattern, flags)
    if not info.is_valid:
        _report_invalid_pattern(ctx, info, arg_index=0)
    return ctx.default_return_type


def re_findall_callback(ctx: FunctionContext) -> Type:
    pattern = _get_pattern_from_expr(ctx, arg_index=0)
    if pattern is None:
        return ctx.default_return_type

    flags = _flags_from_ctx(ctx)
    if flags is None:
        return ctx.default_return_type

    info = analyze_pattern(pattern, flags)
    if not info.is_valid:
        _report_invalid_pattern(ctx, info, arg_index=0)
        return ctx.default_return_type

    element_type = _get_element_type(ctx, _is_bytes_pattern(ctx, arg_index=0))
    return _build_findall_type(ctx, info, element_type)


def re_split_callback(ctx: FunctionContext) -> Type:
    pattern = _get_pattern_from_expr(ctx, arg_index=0)
    if pattern is None:
        return ctx.default_return_type

    flags = _flags_from_ctx(ctx)
    if flags is None:
        return ctx.default_return_type

    info = analyze_pattern(pattern, flags)
    if not info.is_valid:
        _report_invalid_pattern(ctx, info, arg_index=0)
        return ctx.default_return_type

    element_type = _get_element_type(ctx, _is_bytes_pattern(ctx, arg_index=0))
    return _build_split_type(ctx, info, element_type)


def re_sub_callback(ctx: FunctionContext) -> Type:
    pattern = _get_pattern_from_expr(ctx, arg_index=0)
    if pattern is None:
        return ctx.default_return_type

    flags = _flags_from_ctx(ctx)
    if flags is None:
        return ctx.default_return_type

    info = analyze_pattern(pattern, flags)
    if not info.is_valid:
        _report_invalid_pattern(ctx, info, arg_index=0)
        return ctx.default_return_type

    _validate_repl_for_pattern(ctx, info, repl_arg_index=1)
    return ctx.default_return_type


def re_subn_callback(ctx: FunctionContext) -> Type:
    return re_sub_callback(ctx)


def re_pattern_findall_callback(ctx: MethodContext) -> Type:
    obj_type = get_proper_type(ctx.type)
    if not isinstance(obj_type, Instance):
        return ctx.default_return_type

    receiver = _receiver_expr_from_method_context(ctx)
    if receiver is None:
        return ctx.default_return_type

    resolved = _resolve_compiled_pattern_from_expr(ctx, receiver)
    if resolved is None:
        return ctx.default_return_type

    info, _flags = resolved
    if not info.is_valid:
        return ctx.default_return_type

    element_type = _get_element_type(ctx, _is_bytes_type_from_instance(obj_type))
    return _build_findall_type(ctx, info, element_type)


def re_pattern_sub_callback(ctx: MethodContext) -> Type:
    receiver = _receiver_expr_from_method_context(ctx)
    if receiver is None:
        return ctx.default_return_type

    resolved = _resolve_compiled_pattern_from_expr(ctx, receiver)
    if resolved is None:
        return ctx.default_return_type

    info, _flags = resolved
    if info.is_valid:
        _validate_repl_for_pattern(ctx, info, repl_arg_index=0)
    return ctx.default_return_type


def re_pattern_subn_callback(ctx: MethodContext) -> Type:
    return re_pattern_sub_callback(ctx)


def re_pattern_match_callback(ctx: MethodContext) -> Type:
    receiver = _receiver_expr_from_method_context(ctx)
    if receiver is None:
        return ctx.default_return_type

    resolved = _resolve_compiled_pattern_from_expr(ctx, receiver)
    if resolved is None:
        return ctx.default_return_type

    info, _flags = resolved
    if info.is_valid:
        return ctx.default_return_type

    if isinstance(receiver, CallExpr) and receiver.args:
        if isinstance(receiver.args[0], (StrExpr, BytesExpr)):
            ctx.api.fail(
                message_registry.INVALID_RE_PATTERN.format(info.error_msg),
                receiver.args[0],
            )
    return ctx.default_return_type


def match_group_callback(ctx: MethodContext) -> Type:
    if not ctx.args or not ctx.args[0]:
        return ctx.default_return_type

    info = _resolve_match_pattern(ctx)
    if info is not None:
        _validate_group_args(ctx, info)

    return ctx.default_return_type


def match_getitem_callback(ctx: MethodContext) -> Type:
    if not ctx.args or not ctx.args[0]:
        return ctx.default_return_type

    info = _resolve_match_pattern(ctx)
    if info is not None:
        _validate_group_args(ctx, info)

    return ctx.default_return_type
