# tests/regression/test_quantity_default_renders_in_checkout.py
"""Regression tests: default quantity 1 must be spoken in checkout summary.

Production failure: checkout said
  "Carrot Cake. Apple Pie. ..."
instead of
  "1 Carrot Cake. 1 Apple Pie. ..."

These tests use real items from the demo menu (no required modifier/side
groups so the add flow completes immediately) and drive the full
render_checkout_review_summary renderer.
"""
from __future__ import annotations

import pytest

from app.responses.cart_responses import render_checkout_review_summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cart_payload(
    items: list[tuple[str, int]],
    total: str = "$20.00",
    order_type: str | None = "pickup",
) -> tuple[dict, str | None]:
    """Build a minimal checkout payload from (name, quantity) tuples."""
    return (
        {
            "items": [{"name": name, "quantity": qty} for name, qty in items],
            "total": total,
        },
        order_type,
    )


# ---------------------------------------------------------------------------
# Core regression scenario
# ---------------------------------------------------------------------------


class TestQuantityAlwaysRenderedInCheckout:
    def test_single_item_default_qty_1_shows_number(self) -> None:
        """Single default-quantity item must include '1' prefix."""
        payload, order_type = _cart_payload([("Carrot Cake", 1)])
        text = render_checkout_review_summary(payload, order_type)
        assert "1 Carrot Cake" in text, (
            f"Default qty 1 must render as '1 Carrot Cake'; got: {text!r}"
        )

    def test_four_default_items_all_show_quantity_1(self) -> None:
        """All default-quantity items must include '1' prefix, not bare names."""
        payload, order_type = _cart_payload([
            ("Carrot Cake", 1),
            ("Apple Pie", 1),
            ("Alaska Roll", 1),
            ("Americano", 1),
        ], total="$32.00")
        text = render_checkout_review_summary(payload, order_type)
        assert "1 Carrot Cake" in text
        assert "1 Apple Pie" in text
        assert "1 Alaska Roll" in text
        assert "1 Americano" in text

    def test_bare_item_name_without_qty_prefix_not_present(self) -> None:
        """The review section must not contain bare item names without leading number."""
        payload, order_type = _cart_payload([
            ("Carrot Cake", 1),
            ("Apple Pie", 1),
        ])
        text = render_checkout_review_summary(payload, order_type)
        review = text.split("Please review your order:")[1]
        # Each item entry in the review should start with a digit
        for entry in review.split("."):
            entry = entry.strip()
            if entry and not entry.startswith("Your total") and not entry.startswith("Should"):
                assert entry[0].isdigit(), (
                    f"Item entry does not start with a digit: {entry!r}"
                )

    def test_explicit_quantity_2_renders_correctly(self) -> None:
        payload, order_type = _cart_payload([("Carrot Cake", 2)])
        text = render_checkout_review_summary(payload, order_type)
        assert "2 Carrot Cake" in text

    def test_mixed_explicit_and_default_quantities(self) -> None:
        payload, order_type = _cart_payload([
            ("Carrot Cake", 1),   # default
            ("Apple Pie", 3),     # explicit
        ])
        text = render_checkout_review_summary(payload, order_type)
        assert "1 Carrot Cake" in text
        assert "3 Apple Pie" in text


# ---------------------------------------------------------------------------
# Order type prefix preserved
# ---------------------------------------------------------------------------


class TestCheckoutIntroAndStructure:
    def test_pickup_prefix_present(self) -> None:
        payload, order_type = _cart_payload([("Carrot Cake", 1)])
        text = render_checkout_review_summary(payload, order_type)
        assert text.startswith("This is a pickup order.")

    def test_delivery_prefix_present(self) -> None:
        payload, order_type = _cart_payload([("Carrot Cake", 1)], order_type="delivery")
        text = render_checkout_review_summary(payload, order_type)
        assert text.startswith("This is a delivery order.")

    def test_total_always_present(self) -> None:
        payload, order_type = _cart_payload([("Carrot Cake", 1)], total="$8.50")
        text = render_checkout_review_summary(payload, order_type)
        assert "$8.50" in text

    def test_confirmation_question_present(self) -> None:
        payload, order_type = _cart_payload([("Carrot Cake", 1)])
        text = render_checkout_review_summary(payload, order_type)
        assert "Should I place the order?" in text

    def test_no_sides_or_modifiers_in_summary(self) -> None:
        """Sides/modifiers in the item dict must not appear in the voice line."""
        items = [{
            "name": "Carrot Cake",
            "quantity": 1,
            "sides": ["Coke"],
            "modifiers": ["Extra Frosting"],
        }]
        payload = {"items": items, "total": "$8.00"}
        text = render_checkout_review_summary(payload, "pickup")
        assert "Coke" not in text
        assert "Extra Frosting" not in text
        assert "1 Carrot Cake" in text
