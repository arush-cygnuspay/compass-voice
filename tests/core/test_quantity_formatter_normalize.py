# tests/core/test_quantity_formatter_normalize.py
"""Unit tests for normalize_food_quantity (Phase 1) and the updated
format_item_quantity / parse_item_quantity (Phase 2).

Verifies:
- int passthrough (≥ 1) and rejection (< 1)
- 0.N ASR decimal encoding (0.2 → 2, 0.1 → 1, 0.9 → 9)
- Exact-integer floats (1.0 → 1, 3.0 → 3)
- Ambiguous / multi-decimal → None  (1.5, 2.7, 0.15, 0.09)
- Negative / zero → None
- Numeric strings ("2" → 2, "0.2" → 2, "1.0" → 1)
- Word strings → None (callers use normalize_quantity for those)
- format_item_quantity floors to 1 for None-producing inputs
- parse_item_quantity floors to 1 for None-producing inputs
- No use of round() — confirmed by Decimal precision cases
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from app.core.quantity_formatter import (
    format_item_quantity,
    normalize_food_quantity,
    parse_item_quantity,
)


# ---------------------------------------------------------------------------
# normalize_food_quantity — int inputs
# ---------------------------------------------------------------------------

class TestNormalizeFoodQuantityInt:
    @pytest.mark.parametrize("value,expected", [
        (1, 1),
        (2, 2),
        (10, 10),
        (99, 99),
    ])
    def test_positive_int_passthrough(self, value, expected):
        assert normalize_food_quantity(value) == expected

    @pytest.mark.parametrize("value", [0, -1, -5, -100])
    def test_non_positive_int_returns_none(self, value):
        assert normalize_food_quantity(value) is None


# ---------------------------------------------------------------------------
# normalize_food_quantity — 0.N ASR decimal encoding
# ---------------------------------------------------------------------------

class TestNormalizeFoodQuantityZeroDotN:
    @pytest.mark.parametrize("value,expected", [
        (0.1, 1),
        (0.2, 2),
        (0.3, 3),
        (0.4, 4),
        (0.5, 5),
        (0.6, 6),
        (0.7, 7),
        (0.8, 8),
        (0.9, 9),
    ])
    def test_float_zero_dot_n(self, value, expected):
        """0.N float → N items (single decimal digit, non-zero)."""
        assert normalize_food_quantity(value) == expected

    @pytest.mark.parametrize("value,expected", [
        (Decimal("0.1"), 1),
        (Decimal("0.2"), 2),
        (Decimal("0.9"), 9),
    ])
    def test_decimal_zero_dot_n(self, value, expected):
        assert normalize_food_quantity(value) == expected

    @pytest.mark.parametrize("value", [
        0.01, 0.09,         # leading-zero fraction: 0.0X
        Decimal("0.09"),
    ])
    def test_leading_zero_fraction_returns_none(self, value):
        """0.0X is not a valid single-digit encoding."""
        assert normalize_food_quantity(value) is None

    @pytest.mark.parametrize("value", [
        0.15, 0.25, 0.99,   # two-digit fractions
        Decimal("0.15"),
        Decimal("0.99"),
    ])
    def test_multi_digit_fraction_returns_none(self, value):
        """0.NM (two+ fractional digits) is ambiguous → None."""
        assert normalize_food_quantity(value) is None


# ---------------------------------------------------------------------------
# normalize_food_quantity — exact-integer floats / Decimals
# ---------------------------------------------------------------------------

class TestNormalizeFoodQuantityExactInteger:
    @pytest.mark.parametrize("value,expected", [
        (1.0, 1),
        (2.0, 2),
        (5.0, 5),
        (Decimal("1.00"), 1),
        (Decimal("3.0"), 3),
        (Decimal("10"), 10),
    ])
    def test_exact_integer_float_or_decimal(self, value, expected):
        assert normalize_food_quantity(value) == expected


# ---------------------------------------------------------------------------
# normalize_food_quantity — ambiguous non-integer floats
# ---------------------------------------------------------------------------

class TestNormalizeFoodQuantityAmbiguous:
    @pytest.mark.parametrize("value", [
        1.5, 2.7, 1.1, 9.9,
        Decimal("1.5"),
        Decimal("2.25"),
    ])
    def test_non_integer_float_returns_none(self, value):
        """Floats with non-zero fractional part and int_part ≥ 1 → None."""
        assert normalize_food_quantity(value) is None


# ---------------------------------------------------------------------------
# normalize_food_quantity — None input
# ---------------------------------------------------------------------------

class TestNormalizeFoodQuantityNone:
    def test_none_returns_none(self):
        assert normalize_food_quantity(None) is None


# ---------------------------------------------------------------------------
# normalize_food_quantity — string inputs
# ---------------------------------------------------------------------------

class TestNormalizeFoodQuantityStrings:
    @pytest.mark.parametrize("value,expected", [
        ("1", 1),
        ("2", 2),
        ("10", 10),
    ])
    def test_digit_string(self, value, expected):
        assert normalize_food_quantity(value) == expected

    @pytest.mark.parametrize("value,expected", [
        ("0.2", 2),
        ("0.1", 1),
        ("0.9", 9),
        ("1.0", 1),
        ("2.0", 2),
    ])
    def test_numeric_decimal_string(self, value, expected):
        assert normalize_food_quantity(value) == expected

    @pytest.mark.parametrize("value", ["two", "three", "a dozen", "some", ""])
    def test_word_strings_return_none(self, value):
        """Word strings are not in the domain of normalize_food_quantity."""
        assert normalize_food_quantity(value) is None

    @pytest.mark.parametrize("value", ["0", "-1", "-2"])
    def test_non_positive_numeric_strings_return_none(self, value):
        assert normalize_food_quantity(value) is None

    @pytest.mark.parametrize("value", ["1.5", "0.15", "0.09", "garbage", "1x"])
    def test_ambiguous_or_invalid_strings_return_none(self, value):
        assert normalize_food_quantity(value) is None


# ---------------------------------------------------------------------------
# normalize_food_quantity — no round() precision bugs
# ---------------------------------------------------------------------------

class TestNormalizeFoodQuantityNoPrecisionBug:
    def test_float_0_2_is_2_not_0(self):
        """Binary float 0.2 must decode to 2, not 0 (the round() bug)."""
        result = normalize_food_quantity(0.2)
        assert result == 2, f"Expected 2, got {result}"

    def test_float_0_7_is_7_not_1(self):
        """0.7 must decode to 7, not 1 (round(0.7) = 1 is wrong here)."""
        result = normalize_food_quantity(0.7)
        assert result == 7, f"Expected 7, got {result}"

    def test_decimal_str_0_2_is_2(self):
        assert normalize_food_quantity("0.2") == 2

    def test_does_not_use_round_for_zero_dot_n(self):
        """round(0.5) == 0 in banker's rounding; normalize_food_quantity(0.5) must be 5."""
        assert normalize_food_quantity(0.5) == 5


# ---------------------------------------------------------------------------
# format_item_quantity — updated behaviour
# ---------------------------------------------------------------------------

class TestFormatItemQuantityUpdated:
    @pytest.mark.parametrize("value,expected", [
        (0.2, "2"),
        (0.3, "3"),
        (0.9, "9"),
        (Decimal("0.2"), "2"),
        ("0.2", "2"),
        (2.0, "2"),
        (Decimal("2.0"), "2"),
    ])
    def test_new_decoding_cases(self, value, expected):
        assert format_item_quantity(value) == expected

    @pytest.mark.parametrize("value,expected", [
        (1, "1"),
        (0.1, "1"),
        (0.01, "1"),
        (None, "1"),
        (0, "1"),
        (-3, "1"),
    ])
    def test_floor_clamped_cases_unchanged(self, value, expected):
        """Values that can't decode cleanly still floor-clamp to 1."""
        assert format_item_quantity(value) == expected


# ---------------------------------------------------------------------------
# parse_item_quantity — updated behaviour
# ---------------------------------------------------------------------------

class TestParseItemQuantityUpdated:
    @pytest.mark.parametrize("value,expected", [
        (0.2, 2),
        (0.3, 3),
        (0.9, 9),
        (Decimal("0.2"), 2),
        ("0.2", 2),
        (2.0, 2),
    ])
    def test_new_decoding_cases(self, value, expected):
        assert parse_item_quantity(value) == expected

    @pytest.mark.parametrize("value,expected", [
        (1, 1),
        (0.1, 1),
        (0.01, 1),
        (None, 1),
        (0, 1),
    ])
    def test_floor_clamped_cases_unchanged(self, value, expected):
        assert parse_item_quantity(value) == expected
