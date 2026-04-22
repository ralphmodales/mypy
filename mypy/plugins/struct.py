from __future__ import annotations

from typing import Final

from mypy.messages import format_type_bare
from mypy.nodes import ARG_POS, ARG_STAR, BytesExpr, Context, Expression, StrExpr
from mypy.plugin import FunctionContext, MethodContext
from mypy.subtypes import is_subtype
from mypy.typeops import make_simplified_union, try_getting_str_literals_from_type
from mypy.types import (
    AnyType,
    Instance,
    LiteralType,
    TupleType,
    Type,
    TypeOfAny,
    get_proper_type,
)

_BYTE_ORDER_PREFIX_CHARS: Final = frozenset("@=<>!")
_ASCII_WHITESPACE_CHARS: Final = frozenset(" \t\n\r\v\f")
_PAD_FORMAT_CHAR: Final = "x"
_BYTE_LENGTH_FORMAT_CHARS: Final = frozenset("sp")
_MAX_REPEAT: Final = 32

_INTEGER_FORMAT_CHARS: Final = frozenset("bBhHiIlLqQnNP")
_FLOAT_FORMAT_CHARS: Final = frozenset("efd")
_BOOL_FORMAT_CHARS: Final = frozenset("?")
_BYTES_FORMAT_CHARS: Final = frozenset("csp")

_FORMAT_CHAR_TO_FULLNAME: Final[dict[str, str]] = {
    "b": "builtins.int",
    "B": "builtins.int",
    "h": "builtins.int",
    "H": "builtins.int",
    "i": "builtins.int",
    "I": "builtins.int",
    "l": "builtins.int",
    "L": "builtins.int",
    "q": "builtins.int",
    "Q": "builtins.int",
    "n": "builtins.int",
    "N": "builtins.int",
    "P": "builtins.int",
    "e": "builtins.float",
    "f": "builtins.float",
    "d": "builtins.float",
    "?": "builtins.bool",
    "c": "builtins.bytes",
    "s": "builtins.bytes",
    "p": "builtins.bytes",
}

STRUCT_UNPACK_FULLNAME: Final = "_struct.unpack"
STRUCT_UNPACK_FROM_FULLNAME: Final = "_struct.unpack_from"
STRUCT_ITER_UNPACK_FULLNAME: Final = "_struct.iter_unpack"
STRUCT_PACK_FULLNAME: Final = "_struct.pack"
STRUCT_PACK_INTO_FULLNAME: Final = "_struct.pack_into"
STRUCT_CLASS_FULLNAME: Final = "_struct.Struct"
STRUCT_CLASS_UNPACK_FULLNAME: Final = "_struct.Struct.unpack"
STRUCT_CLASS_UNPACK_FROM_FULLNAME: Final = "_struct.Struct.unpack_from"
STRUCT_CLASS_ITER_UNPACK_FULLNAME: Final = "_struct.Struct.iter_unpack"
STRUCT_CLASS_PACK_FULLNAME: Final = "_struct.Struct.pack"
STRUCT_CLASS_PACK_INTO_FULLNAME: Final = "_struct.Struct.pack_into"

_STRUCT_FMT_ATTR: Final = "__mypy_struct_fmt"


def _read_byte_order_prefix(format_string: str, start: int) -> int:
    if start < len(format_string) and format_string[start] in _BYTE_ORDER_PREFIX_CHARS:
        return start + 1
    return start


def _read_decimal_repeat(format_string: str, start: int) -> tuple[int | None, int]:
    position = start
    length = len(format_string)
    while position < length and format_string[position].isdigit():
        position += 1
    if position == start:
        return None, position
    digits = format_string[start:position]
    return int(digits), position


def _parse_struct_format(format_string: str) -> list[str] | None:
    length = len(format_string)
    position = _read_byte_order_prefix(format_string, 0)
    element_chars: list[str] = []
    while position < length:
        current_char = format_string[position]
        if current_char in _ASCII_WHITESPACE_CHARS:
            position += 1
            continue
        repeat_count, position_after_digits = _read_decimal_repeat(format_string, position)
        if repeat_count is not None:
            if position_after_digits >= length:
                return None
            position = position_after_digits
            current_char = format_string[position]
        if current_char == _PAD_FORMAT_CHAR:
            position += 1
            continue
        if current_char not in _FORMAT_CHAR_TO_FULLNAME:
            return None
        if current_char in _BYTE_LENGTH_FORMAT_CHARS:
            element_chars.append(current_char)
        else:
            effective_count = repeat_count if repeat_count is not None else 1
            if effective_count > _MAX_REPEAT:
                return None
            element_chars.extend([current_char] * effective_count)
        if len(element_chars) > _MAX_REPEAT:
            return None
        position += 1
    if len(element_chars) > _MAX_REPEAT:
        return None
    return element_chars


def _extract_literal_formats_from_expr(
    expression: Expression, arg_type: Type | None
) -> list[str] | None:
    if isinstance(expression, StrExpr):
        return [expression.value]
    if isinstance(expression, BytesExpr):
        return [expression.value]
    if arg_type is None:
        return None
    return try_getting_str_literals_from_type(arg_type)


def _extract_formats_from_func_ctx(ctx: FunctionContext) -> list[str] | None:
    if not ctx.args or not ctx.args[0]:
        return None
    expression = ctx.args[0][0]
    arg_type: Type | None = None
    if ctx.arg_types and ctx.arg_types[0]:
        arg_type = ctx.arg_types[0][0]
    return _extract_literal_formats_from_expr(expression, arg_type)


def _parse_all_formats(format_strings: list[str]) -> list[list[str]] | None:
    parsed: list[list[str]] = []
    for fmt in format_strings:
        chars = _parse_struct_format(fmt)
        if chars is None:
            return None
        parsed.append(chars)
    return parsed


def _build_tuple_type_from_chars(
    ctx: FunctionContext | MethodContext, chars: list[str]
) -> TupleType:
    fallback_instance = ctx.api.named_generic_type(
        "builtins.tuple", [AnyType(TypeOfAny.special_form)]
    )
    items: list[Type] = [
        ctx.api.named_generic_type(_FORMAT_CHAR_TO_FULLNAME[c], []) for c in chars
    ]
    return TupleType(items, fallback_instance)


def _build_unpack_return_type(
    ctx: FunctionContext | MethodContext, all_formats: list[list[str]]
) -> Type:
    tuple_types: list[Type] = [
        _build_tuple_type_from_chars(ctx, fmt_chars) for fmt_chars in all_formats
    ]
    if len(tuple_types) == 1:
        return tuple_types[0]
    return make_simplified_union(tuple_types)


def _build_fmt_marker_type(ctx: FunctionContext, chars: list[str]) -> TupleType:
    str_type = ctx.api.named_generic_type("builtins.str", [])
    fallback = ctx.api.named_generic_type("builtins.tuple", [str_type])
    items: list[Type] = [LiteralType(c, str_type) for c in chars]
    return TupleType(items, fallback)


def _get_stored_fmt_chars(ctx: MethodContext) -> list[str] | None:
    if not isinstance(ctx.type, Instance):
        return None
    if not ctx.type.extra_attrs:
        return None
    if _STRUCT_FMT_ATTR not in ctx.type.extra_attrs.attrs:
        return None
    stored = get_proper_type(ctx.type.extra_attrs.attrs[_STRUCT_FMT_ATTR])
    if not isinstance(stored, TupleType):
        return None
    chars: list[str] = []
    for item in stored.items:
        item_proper = get_proper_type(item)
        if not isinstance(item_proper, LiteralType):
            return None
        if not isinstance(item_proper.value, str):
            return None
        chars.append(item_proper.value)
    return chars


def _infer_unpack_tuple_from_func_ctx(ctx: FunctionContext) -> Type | None:
    formats = _extract_formats_from_func_ctx(ctx)
    if formats is None or not formats:
        return None
    parsed = _parse_all_formats(formats)
    if parsed is None:
        return None
    return _build_unpack_return_type(ctx, parsed)


def struct_unpack_callback(ctx: FunctionContext) -> Type:
    result = _infer_unpack_tuple_from_func_ctx(ctx)
    if result is None:
        return ctx.default_return_type
    return result


def struct_unpack_from_callback(ctx: FunctionContext) -> Type:
    result = _infer_unpack_tuple_from_func_ctx(ctx)
    if result is None:
        return ctx.default_return_type
    return result


def struct_iter_unpack_callback(ctx: FunctionContext) -> Type:
    result = _infer_unpack_tuple_from_func_ctx(ctx)
    if result is None:
        return ctx.default_return_type
    return ctx.api.named_generic_type("typing.Iterator", [result])


def struct_class_callback(ctx: FunctionContext) -> Type:
    default = ctx.default_return_type
    default_proper = get_proper_type(default)
    if not isinstance(default_proper, Instance):
        return default
    formats = _extract_formats_from_func_ctx(ctx)
    if formats is None or len(formats) != 1:
        return default
    chars = _parse_struct_format(formats[0])
    if chars is None:
        return default
    marker = _build_fmt_marker_type(ctx, chars)
    return default_proper.copy_with_extra_attr(_STRUCT_FMT_ATTR, marker)


def _infer_unpack_tuple_from_stored(ctx: MethodContext) -> Type | None:
    chars = _get_stored_fmt_chars(ctx)
    if chars is None:
        return None
    return _build_tuple_type_from_chars(ctx, chars)


def struct_class_unpack_callback(ctx: MethodContext) -> Type:
    result = _infer_unpack_tuple_from_stored(ctx)
    if result is None:
        return ctx.default_return_type
    return result


def struct_class_unpack_from_callback(ctx: MethodContext) -> Type:
    result = _infer_unpack_tuple_from_stored(ctx)
    if result is None:
        return ctx.default_return_type
    return result


def struct_class_iter_unpack_callback(ctx: MethodContext) -> Type:
    result = _infer_unpack_tuple_from_stored(ctx)
    if result is None:
        return ctx.default_return_type
    return ctx.api.named_generic_type("typing.Iterator", [result])


def _expected_type_name_for_char(char: str) -> str:
    if char in _BOOL_FORMAT_CHARS:
        return "bool"
    if char in _INTEGER_FORMAT_CHARS:
        return "int"
    if char in _FLOAT_FORMAT_CHARS:
        return "float"
    return "bytes"


def _value_type_matches_char(
    ctx: FunctionContext | MethodContext, value_type: Type, char: str
) -> bool:
    if char in _BOOL_FORMAT_CHARS:
        bool_type = ctx.api.named_generic_type("builtins.bool", [])
        int_type = ctx.api.named_generic_type("builtins.int", [])
        return is_subtype(value_type, bool_type) or is_subtype(value_type, int_type)
    if char in _INTEGER_FORMAT_CHARS:
        int_type = ctx.api.named_generic_type("builtins.int", [])
        return is_subtype(value_type, int_type)
    if char in _FLOAT_FORMAT_CHARS:
        float_type = ctx.api.named_generic_type("builtins.float", [])
        return is_subtype(value_type, float_type)
    bytes_type = ctx.api.named_generic_type("builtins.bytes", [])
    return is_subtype(value_type, bytes_type)


def _collect_positional_values(
    ctx: FunctionContext | MethodContext, start_group: int
) -> tuple[list[Expression], list[Type], bool]:
    values: list[Expression] = []
    types: list[Type] = []
    saw_star = False
    for i in range(start_group, len(ctx.arg_kinds)):
        for j, kind in enumerate(ctx.arg_kinds[i]):
            if kind == ARG_STAR:
                saw_star = True
                continue
            if kind != ARG_POS:
                continue
            values.append(ctx.args[i][j])
            types.append(ctx.arg_types[i][j])
    return values, types, saw_star


_PackError = tuple[str, Context]


def _validate_pack_for_chars(
    ctx: FunctionContext | MethodContext,
    chars: list[str],
    method_name: str,
    values: list[Expression],
    value_types: list[Type],
    fmt_string: str,
) -> list[_PackError]:
    errors: list[_PackError] = []
    expected_count = len(chars)
    got_count = len(value_types)
    if expected_count != got_count:
        errors.append(
            (
                f'Wrong number of values for format "{fmt_string}":'
                f" expected {expected_count}, got {got_count}",
                ctx.context,
            )
        )
        return errors
    for index in range(got_count):
        value_type = value_types[index]
        char = chars[index]
        if not _value_type_matches_char(ctx, value_type, char):
            expected_name = _expected_type_name_for_char(char)
            actual_name = format_type_bare(value_type, ctx.api.options)
            errors.append(
                (
                    f'Argument {index + 1} to "{method_name}" has incompatible type'
                    f' "{actual_name}"; expected "{expected_name}"'
                    f' (for format char "{char}")',
                    values[index],
                )
            )
    return errors


def _emit_pack_errors(
    ctx: FunctionContext | MethodContext,
    all_formats: list[list[str]],
    format_strings: list[str],
    method_name: str,
    values: list[Expression],
    value_types: list[Type],
    saw_star: bool,
) -> None:
    if saw_star:
        return
    first_errors: list[_PackError] | None = None
    for chars, fmt_str in zip(all_formats, format_strings):
        errors = _validate_pack_for_chars(
            ctx, chars, method_name, values, value_types, fmt_str
        )
        if not errors:
            return
        if first_errors is None:
            first_errors = errors
    if first_errors is None:
        return
    for message, context in first_errors:
        ctx.api.msg.fail(message, context)


def _gather_func_formats(
    ctx: FunctionContext,
) -> tuple[list[str], list[list[str]]] | None:
    format_strings = _extract_formats_from_func_ctx(ctx)
    if format_strings is None or not format_strings:
        return None
    parsed = _parse_all_formats(format_strings)
    if parsed is None:
        return None
    return format_strings, parsed


def struct_pack_callback(ctx: FunctionContext) -> Type:
    result = _gather_func_formats(ctx)
    if result is None:
        return ctx.default_return_type
    format_strings, all_formats = result
    values, value_types, saw_star = _collect_positional_values(ctx, start_group=1)
    _emit_pack_errors(
        ctx,
        all_formats,
        format_strings,
        "struct.pack",
        values,
        value_types,
        saw_star,
    )
    return ctx.default_return_type


def struct_pack_into_callback(ctx: FunctionContext) -> Type:
    result = _gather_func_formats(ctx)
    if result is None:
        return ctx.default_return_type
    format_strings, all_formats = result
    values, value_types, saw_star = _collect_positional_values(ctx, start_group=3)
    _emit_pack_errors(
        ctx,
        all_formats,
        format_strings,
        "struct.pack_into",
        values,
        value_types,
        saw_star,
    )
    return ctx.default_return_type


def struct_class_pack_callback(ctx: MethodContext) -> Type:
    chars = _get_stored_fmt_chars(ctx)
    if chars is None:
        return ctx.default_return_type
    fmt_string = "".join(chars)
    values, value_types, saw_star = _collect_positional_values(ctx, start_group=0)
    _emit_pack_errors(
        ctx,
        [chars],
        [fmt_string],
        "Struct.pack",
        values,
        value_types,
        saw_star,
    )
    return ctx.default_return_type


def struct_class_pack_into_callback(ctx: MethodContext) -> Type:
    chars = _get_stored_fmt_chars(ctx)
    if chars is None:
        return ctx.default_return_type
    fmt_string = "".join(chars)
    values, value_types, saw_star = _collect_positional_values(ctx, start_group=2)
    _emit_pack_errors(
        ctx,
        [chars],
        [fmt_string],
        "Struct.pack_into",
        values,
        value_types,
        saw_star,
    )
    return ctx.default_return_type
