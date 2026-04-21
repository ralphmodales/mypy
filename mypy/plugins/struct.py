from __future__ import annotations

from typing import Final

from mypy.nodes import BytesExpr, Expression, StrExpr
from mypy.plugin import FunctionContext
from mypy.typeops import try_getting_str_literals
from mypy.types import AnyType, Instance, TupleType, Type, TypeOfAny

_BYTE_ORDER_PREFIX_CHARS: Final = frozenset("@=<>!")
_ASCII_WHITESPACE_CHARS: Final = frozenset(" \t\n\r\v\f")
_PAD_FORMAT_CHAR: Final = "x"
_BYTE_LENGTH_FORMAT_CHARS: Final = frozenset("sp")
_MAX_REPEAT: Final = 32

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
    element_fullnames: list[str] = []
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
        element_fullname = _FORMAT_CHAR_TO_FULLNAME.get(current_char)
        if element_fullname is None:
            return None
        if current_char in _BYTE_LENGTH_FORMAT_CHARS:
            element_fullnames.append(element_fullname)
        else:
            effective_count = repeat_count if repeat_count is not None else 1
            if effective_count > _MAX_REPEAT:
                return None
            element_fullnames.extend([element_fullname] * effective_count)
        if len(element_fullnames) > _MAX_REPEAT:
            return None
        position += 1
    if len(element_fullnames) > _MAX_REPEAT:
        return None
    return element_fullnames


def _extract_literal_format(ctx: FunctionContext) -> str | None:
    if not ctx.args or not ctx.args[0]:
        return None
    first_expression: Expression = ctx.args[0][0]
    if isinstance(first_expression, StrExpr):
        return first_expression.value
    if isinstance(first_expression, BytesExpr):
        return first_expression.value
    if not ctx.arg_types or not ctx.arg_types[0]:
        return None
    literal_values = try_getting_str_literals(first_expression, ctx.arg_types[0][0])
    if literal_values is None or len(literal_values) != 1:
        return None
    return literal_values[0]


def _build_tuple_type(ctx: FunctionContext, element_fullnames: list[str]) -> TupleType:
    fallback_instance: Instance = ctx.api.named_generic_type(
        "builtins.tuple", [AnyType(TypeOfAny.special_form)]
    )
    tuple_items: list[Type] = [
        ctx.api.named_generic_type(fullname, []) for fullname in element_fullnames
    ]
    return TupleType(tuple_items, fallback_instance)


def _infer_unpack_tuple(ctx: FunctionContext) -> TupleType | None:
    raw_format = _extract_literal_format(ctx)
    if raw_format is None:
        return None
    parsed_elements = _parse_struct_format(raw_format)
    if parsed_elements is None:
        return None
    return _build_tuple_type(ctx, parsed_elements)


def struct_unpack_callback(ctx: FunctionContext) -> Type:
    inferred_tuple = _infer_unpack_tuple(ctx)
    if inferred_tuple is None:
        return ctx.default_return_type
    return inferred_tuple


def struct_unpack_from_callback(ctx: FunctionContext) -> Type:
    inferred_tuple = _infer_unpack_tuple(ctx)
    if inferred_tuple is None:
        return ctx.default_return_type
    return inferred_tuple


def struct_iter_unpack_callback(ctx: FunctionContext) -> Type:
    inferred_tuple = _infer_unpack_tuple(ctx)
    if inferred_tuple is None:
        return ctx.default_return_type
    return ctx.api.named_generic_type("typing.Iterator", [inferred_tuple])
