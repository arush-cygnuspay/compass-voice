# tests/core/test_quantity_formatter.py
"""
Regression tests for format_item_quantity / parse_item_quantity.

These guard against the class of bug where item quantities stored as floats
(e.g. 0.1, 1.0, 0.01) flow into TTS/order-review renderers and produce
strings like "0.1 Coke" or "0 BBQ Chicken Pizza".
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from app.core.quantity_formatter import format_item_quantity, parse_item_quantity
from app.responses.cart_responses import render_cart_summary, render_checkout_review_summary


# ---------------------------------------------------------------------------
# format_item_quantity
# ---------------------------------------------------------------------------

class TestFormatItemQuantity:
    def test_plain_int_1(self):
        assert format_item_quantity(1) == "1"

    def test_plain_int_2(self):
        assert format_item_quantity(2) == "2"

    def test_float_1_0(self):
        assert format_item_quantity(1.0) == "1"

    def test_float_0_1_rounds_to_1(self):
        # Internally-scaled float that represents "1 item" in some broken
        # serialization should round to 1, not truncate to 0.
        assert format_item_quantity(0.1) == "1"

    def test_float_0_01_rounds_to_1(self):
        assert format_item_quantity(0.01) == "1"

    def test_decimal_1_00(self):
        assert format_item_quantity(Decimal("1.00")) == "1"

    def test_decimal_0_1(self):
        assert format_item_quantity(Decimal("0.1")) == "1"

    def test_float_2_0(self):
        assert format_item_quantity(2.0) == "2"

    def test_string_int(self):
        assert format_item_quantity("3") == "3"

    def test_none_defaults_to_1(self):
        assert format_item_quantity(None) == "1"

    def test_floor_clamp_zero(self):
        # Even if rounding gives 0 somehow, result must be at least 1.
        assert format_item_quantity(0) == "1"

    def test_floor_clamp_negative(self):
        assert format_item_quantity(-5) == "1"


# ---------------------------------------------------------------------------
# parse_item_quantity
# ---------------------------------------------------------------------------

class TestParseItemQuantity:
    def test_int(self):
        assert parse_item_quantity(1) == 1

    def test_float_1_0(self):
        assert parse_item_quantity(1.0) == 1

    def test_float_0_1_rounds_to_1(self):
        assert parse_item_quantity(0.1) == 1

    def test_float_2_5_banker_rounds_to_2(self):
        # Python round() uses banker's rounding (round-half-to-even).
        assert parse_item_quantity(2.5) == 2

    def test_none_returns_1(self):
        assert parse_item_quantity(None) == 1

    def test_zero_clamped_to_1(self):
        assert parse_item_quantity(0) == 1


# ---------------------------------------------------------------------------
# render_cart_summary — TTS path, most critical
# ---------------------------------------------------------------------------

def _cart_payload(quantity, name="Coke", total="$5.00"):
    return {
        "items": [{"quantity": quantity, "name": name}],
        "total": total,
        "item_count": None,
    }


class TestRenderCartSummaryQuantity:
    def test_integer_quantity(self):
        text = render_cart_summary(_cart_payload(1, "Coke"))
        assert "1 Coke" in text
        assert "0.1" not in text

    def test_float_1_0(self):
        text = render_cart_summary(_cart_payload(1.0, "Coke"))
        assert "1 Coke" in text

    def test_float_0_1(self):
        text = render_cart_summary(_cart_payload(0.1, "Coke"))
        assert "1 Coke" in text
        assert "0.1" not in text

    def test_float_0_01(self):
        text = render_cart_summary(_cart_payload(0.01, "BBQ Chicken Pizza"))
        assert "1 BBQ Chicken Pizza" in text
        assert "0.01" not in text

    def test_price_preserved(self):
        text = render_cart_summary(_cart_payload(1, "Coke", total="$17.20"))
        assert "$17.20" in text

    def test_multi_item_count_uses_int(self):
        payload = {
            "items": [
                {"quantity": 0.1, "name": "Coke"},
                {"quantity": 0.1, "name": "Burger"},
            ],
            "total": "$10.00",
            "item_count": None,
        }
        text = render_cart_summary(payload)
        # item_count should be 2 (1+1 after rounding), not 0
        assert "2 items" in text


# ---------------------------------------------------------------------------
# render_checkout_review_summary — order confirmation read-back
# ---------------------------------------------------------------------------

class TestRenderCheckoutReviewSummaryQuantity:
    def test_integer_quantity(self):
        """spoken_quantity_label(1, name) omits the leading '1' — TTS says 'Burger'."""
        payload = {"items": [{"quantity": 1, "name": "Burger"}], "total": "$8.00"}
        text = render_checkout_review_summary(payload)
        # qty=1 → spoken_quantity_label returns just the name (no leading "1")
        assert "Burger" in text
        # Must never contain x-notation
        assert "1 x Burger" not in text
        assert " x " not in text

    def test_float_0_1_does_not_produce_zero(self):
        payload = {"items": [{"quantity": 0.1, "name": "Coke"}], "total": "$5.00"}
        text = render_checkout_review_summary(payload)
        # Old bare int() cast gives "0 Coke"; formatter must give "Coke" (qty=1, no prefix)
        assert "0 Coke" not in text
        # spoken_quantity_label(1, "Coke") → "Coke" (no leading 1 for singular)
        assert "Coke" in text
        assert " x " not in text

    def test_float_0_01_bbq_chicken_pizza(self):
        payload = {"items": [{"quantity": 0.01, "name": "BBQ Chicken Pizza"}], "total": "$17.20"}
        text = render_checkout_review_summary(payload)
        assert "0 BBQ Chicken Pizza" not in text
        # qty decodes to 1 → spoken_quantity_label → name only (no leading 1)
        assert "BBQ Chicken Pizza" in text
        assert "$17.20" in text
        assert " x " not in text
