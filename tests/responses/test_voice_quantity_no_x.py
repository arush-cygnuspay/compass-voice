# tests/responses/test_voice_quantity_no_x.py
"""Tests for spoken_quantity_label and compact_quantity_label (Phase 5).

Also guards the existing voice response paths (success.py, cart_responses.py)
against accidental introduction of 'x' notation (Phase 6).

Voice TTS must never produce "2x Coke", "Coke x2", "× 2 Coke" etc. as these
read out literally on Deepgram TTS ("two ex coke").
"""
from __future__ import annotations

import pytest

from app.responses.item.format_utils import compact_quantity_label, spoken_quantity_label
from app.responses.cart_responses import render_cart_summary, render_checkout_review_summary


# ---------------------------------------------------------------------------
# spoken_quantity_label
# ---------------------------------------------------------------------------

class TestSpokenQuantityLabel:
    def test_quantity_1_returns_name_only(self):
        assert spoken_quantity_label(1, "Coke") == "Coke"

    def test_quantity_2_no_explicit_plural_returns_2_name(self):
        assert spoken_quantity_label(2, "Coke") == "2 Coke"

    def test_quantity_3_no_explicit_plural_returns_3_name(self):
        assert spoken_quantity_label(3, "Burger") == "3 Burger"

    def test_quantity_2_with_explicit_plural(self):
        assert spoken_quantity_label(2, "Coke", plural="Cokes") == "2 Cokes"

    def test_quantity_3_with_explicit_plural(self):
        assert spoken_quantity_label(3, "French Fry", plural="French Fries") == "3 French Fries"

    def test_no_x_notation_in_output(self):
        label = spoken_quantity_label(2, "Coke")
        assert "x" not in label.lower() or label == "2 Coke"
        # More precisely: no standalone "x" multiplier
        assert " x " not in label
        assert "x2" not in label
        assert "2x" not in label

    def test_empty_name(self):
        assert spoken_quantity_label(1, "") == ""
        assert spoken_quantity_label(2, "") == "2 "

    def test_quantity_below_1_floors_to_1(self):
        """Even if a caller passes 0, it floors to 1 and returns name only."""
        assert spoken_quantity_label(0, "Coke") == "Coke"

    @pytest.mark.parametrize("q,name", [
        (2, "French Fries"),
        (3, "BBQ Chicken Pizza"),
        (4, "Onion Ring"),
        (5, "Dr Pepper"),
    ])
    def test_complex_names_not_mangled(self, q, name):
        """Multi-word item names must appear verbatim in the label."""
        label = spoken_quantity_label(q, name)
        assert name in label
        assert "x" not in label.split()[0]  # no "x" prefix on the number


# ---------------------------------------------------------------------------
# compact_quantity_label
# ---------------------------------------------------------------------------

class TestCompactQuantityLabel:
    def test_quantity_1_returns_name_only(self):
        assert compact_quantity_label(1, "Coke") == "Coke"

    def test_quantity_2_returns_2_name(self):
        assert compact_quantity_label(2, "Coke") == "2 Coke"

    def test_no_x_notation(self):
        label = compact_quantity_label(3, "Burger")
        assert " x " not in label
        assert "x3" not in label
        assert "3x" not in label

    def test_quantity_below_1_floors_to_1(self):
        assert compact_quantity_label(0, "Coke") == "Coke"

    @pytest.mark.parametrize("q,name,expected", [
        (1, "Coke", "Coke"),
        (2, "Coke", "2 Coke"),
        (3, "Large Fries", "3 Large Fries"),
        (10, "Wing", "10 Wing"),
    ])
    def test_parametrized(self, q, name, expected):
        assert compact_quantity_label(q, name) == expected


# ---------------------------------------------------------------------------
# Phase 6: Voice response paths must not contain "x" notation
# ---------------------------------------------------------------------------

def _cart_payload(quantity, name="Coke", total="$5.00"):
    return {
        "items": [{"quantity": quantity, "name": name}],
        "total": total,
        "item_count": None,
    }


class TestVoicePathsNoXNotation:
    @pytest.mark.parametrize("quantity", [1, 2, 3, 5])
    def test_render_cart_summary_no_x(self, quantity):
        text = render_cart_summary(_cart_payload(quantity, "Coke"))
        assert " x " not in text
        assert "x2" not in text
        assert "2x" not in text

    @pytest.mark.parametrize("quantity", [1, 2, 3])
    def test_render_checkout_review_no_x(self, quantity):
        payload = {"items": [{"quantity": quantity, "name": "Burger"}], "total": "$8.00"}
        text = render_checkout_review_summary(payload)
        assert " x " not in text

    def test_render_cart_summary_decimal_quantity_no_x(self):
        """Decimal-encoded quantity 0.2 → renders as '2 Coke', never '0.2 x Coke'."""
        text = render_cart_summary(_cart_payload(0.2, "Coke"))
        assert "x" not in text.lower() or "2 Coke" in text
        assert "0.2" not in text
