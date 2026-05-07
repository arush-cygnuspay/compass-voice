# app/core/quantity_formatter.py
from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Union

_QuantityInput = Union[int, float, Decimal, str, None]


def normalize_food_quantity(value: _QuantityInput) -> int | None:
    """Decode a raw quantity value to a positive int, or None if ambiguous.

    Handles the ASR/NLU decimal-encoding convention where 0.N (a zero integer
    part with a single non-zero decimal digit) means N items — e.g. 0.2 → 2,
    0.1 → 1.  Uses ``Decimal`` arithmetic throughout so that binary float
    imprecision (0.1 → 0.100000000000000055511…) cannot corrupt the result.

    Rules (evaluated in priority order):

    1. ``None`` → ``None``.
    2. ``int >= 1`` → same int.
    3. ``int < 1`` (includes 0 and negatives) → ``None``.
    4. Numeric value with integer-part == 0 and a single non-zero fractional
       digit → that digit (0.2 → 2, 0.9 → 9, 0.1 → 1).
       Multi-digit fractions (0.15, 0.09) and a leading-zero fraction (0.01)
       are rejected → ``None``.
    5. Numeric value with integer-part >= 1 and zero fractional part → that
       integer (1.0 → 1, 3.0 → 3).
    6. Numeric value with non-zero fractional part and integer-part >= 1
       (e.g. 1.5, 2.7) → ``None`` (ambiguous — could be a weight or price).
    7. ``str`` containing only digits → int parse.
    8. ``str`` representing a decimal number (e.g. "0.2", "1.0") → Decimal
       parse then apply rules 4-6.
    9. Any other ``str`` (word like "two") or unparseable value → ``None``.
       Callers that need word-string parsing should use
       :func:`app.nlu.matching.quantity_parser.normalize_quantity` first.

    Does NOT use ``round()`` — uses exact Decimal arithmetic.
    """
    if value is None:
        return None

    # ── Rule 2-3: bare Python int ─────────────────────────────────────────
    if isinstance(value, int):
        return value if value >= 1 else None

    # ── Rules 7-8: string — try cheap digit check first, then Decimal ─────
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.lstrip("-").isdigit():
            n = int(text)
            return n if n >= 1 else None
        # Fall through to Decimal parsing for "0.2", "1.0", etc.
        # Word strings ("two") will raise InvalidOperation → return None.

    # ── Rules 4-6: float / Decimal / numeric str ──────────────────────────
    try:
        d = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None

    if d.is_nan() or d.is_infinite():
        return None

    if d < 0:
        return None

    int_part = int(d.to_integral_value(rounding="ROUND_FLOOR"))
    frac = d - Decimal(int_part)

    if int_part == 0:
        if frac == Decimal("0"):
            return None  # exactly 0 — not a valid quantity

        # 0.N encoding: valid only when the fractional part has exactly one
        # non-zero digit.
        # 0.2  → first_digit=2, remainder=0  → 2   ✓
        # 0.1  → first_digit=1, remainder=0  → 1   ✓
        # 0.09 → first_digit=0               → None (leading-zero fraction)
        # 0.15 → first_digit=1, remainder≠0  → None (two-digit fraction)
        first_digit_d = frac * Decimal("10")
        first_digit = int(first_digit_d.to_integral_value(rounding="ROUND_FLOOR"))
        remainder = first_digit_d - Decimal(first_digit)

        if first_digit == 0:
            return None  # leading-zero fraction (0.0X)
        if remainder != Decimal("0"):
            return None  # more than one fractional digit (0.NM)
        return first_digit

    # int_part >= 1: only valid when fractional part is exactly zero.
    if frac != Decimal("0"):
        return None  # e.g. 1.5, 2.7 → ambiguous

    return int_part


def format_item_quantity(quantity: _QuantityInput) -> str:
    """Normalise a quantity value to a clean integer string for TTS/display.

    Delegates to :func:`normalize_food_quantity` so that the ASR decimal
    encoding (0.N → N items) is handled correctly.  Falls back to
    ``round(float(...))`` for inputs that cannot be decoded by
    ``normalize_food_quantity`` (e.g. 2.5, 0.01) with a floor of 1.

    Examples::

        format_item_quantity(1)            → "1"
        format_item_quantity(1.0)          → "1"
        format_item_quantity(0.2)          → "2"   # ASR decimal encoding
        format_item_quantity(0.1)          → "1"   # ASR decimal encoding
        format_item_quantity(0.01)         → "1"   # fallback, floor-clamped
        format_item_quantity(None)         → "1"   # fallback default
        format_item_quantity(Decimal("2")) → "2"
    """
    result = normalize_food_quantity(quantity)
    if result is not None:
        return str(max(1, result))
    # Fallback for values normalize_food_quantity cannot decode (0.01, 1.5, …)
    try:
        rounded = round(float(quantity))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        rounded = 1
    return str(max(1, rounded))


def parse_item_quantity(quantity: _QuantityInput) -> int:
    """Return *quantity* as a safe int (≥ 1).

    Mirrors :func:`format_item_quantity`'s logic but returns ``int`` rather
    than ``str``, for use at cart/domain deserialization boundaries.

    Examples::

        parse_item_quantity(1)     → 1
        parse_item_quantity(0.2)   → 2   # ASR decimal encoding
        parse_item_quantity(0.1)   → 1   # ASR decimal encoding
        parse_item_quantity(1.0)   → 1
        parse_item_quantity(None)  → 1   # fallback default
    """
    result = normalize_food_quantity(quantity)
    if result is not None:
        return max(1, result)
    # Fallback
    try:
        rounded = round(float(quantity))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        rounded = 1
    return max(1, rounded)
