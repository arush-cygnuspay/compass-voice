# app/core/quantity_formatter.py
from __future__ import annotations

from decimal import Decimal
from typing import Union

_QuantityInput = Union[int, float, Decimal, str, None]


def format_item_quantity(quantity: _QuantityInput) -> str:
    """
    Normalise a quantity value to a clean integer string for TTS/display.

    Handles the case where quantities are stored or deserialised as floats
    (e.g. 1.0, 0.1 due to scaling bugs) by rounding to the nearest integer,
    with a floor of 1 so we never display "0 items".

    Examples:
      1        -> "1"
      1.0      -> "1"
      Decimal("1.00") -> "1"
      0.1      -> "1"   (rounds, then floor-clamped to 1)
      2        -> "2"
    """
    try:
        rounded = round(float(quantity))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        rounded = 1
    return str(max(1, rounded))


def parse_item_quantity(quantity: _QuantityInput) -> int:
    """
    Return quantity as a safe int (≥ 1).  Mirrors format_item_quantity's
    rounding logic but returns int rather than str, for use in cart/domain
    normalization at deserialization boundaries.
    """
    try:
        rounded = round(float(quantity))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        rounded = 1
    return max(1, rounded)
