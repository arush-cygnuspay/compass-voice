# tests/responses/test_checkout_summary_quantity_contract.py
"""Contract tests for checkout review summary quantity rendering.

Business rule: checkout summary always speaks the quantity for every main item,
including default quantity 1.  "1 Chicken Burger" not "Chicken Burger".

Quantity only — sides, modifiers, and variants are intentionally omitted from
the checkout voice summary.
"""
from __future__ import annotations

import pytest

from app.responses.cart_responses import render_checkout_review_summary
from app.responses.item.format_utils import checkout_quantity_label


# ---------------------------------------------------------------------------
# checkout_quantity_label unit tests
# ---------------------------------------------------------------------------


class TestCheckoutQuantityLabel:
    def test_qty_1_includes_leading_1(self) -> None:
        assert checkout_quantity_label(1, "Chicken Burger") == "1 Chicken Burger"

    def test_qty_2_includes_leading_2(self) -> None:
        assert checkout_quantity_label(2, "Coke") == "2 Coke"

    def test_qty_3_with_plural(self) -> None:
        assert checkout_quantity_label(3, "Coke", plural="Cokes") == "3 Cokes"

    def test_qty_1_plural_not_used(self) -> None:
        # plural is only for qty > 1
        assert checkout_quantity_label(1, "Coke", plural="Cokes") == "1 Coke"

    def test_qty_none_normalizes_to_1(self) -> None:
        assert checkout_quantity_label(None, "Burger") == "1 Burger"

    def test_qty_zero_normalizes_to_1(self) -> None:
        assert checkout_quantity_label(0, "Burger") == "1 Burger"

    def test_qty_negative_normalizes_to_1(self) -> None:
        assert checkout_quantity_label(-3, "Burger") == "1 Burger"

    def test_no_naive_pluralization_without_plural_param(self) -> None:
        """Without an explicit plural, no 's' is appended to avoid 'French Friess'."""
        label = checkout_quantity_label(2, "French Fries")
        assert label == "2 French Fries"
        assert label.endswith("ss") is False


# ---------------------------------------------------------------------------
# render_checkout_review_summary — quantity contract
# ---------------------------------------------------------------------------


def _checkout_payload(
    items: list[dict],
    total: str = "$35.45",
    order_type: str | None = None,
) -> tuple[dict, str | None]:
    return {"items": items, "total": total}, order_type


class TestSingleDefaultQuantityRendersAs1:
    def test_single_default_quantity_renders_as_1_item_name(self) -> None:
        payload, order_type = _checkout_payload(
            [{"name": "Chicken Burger", "quantity": 1}]
        )
        text = render_checkout_review_summary(payload, order_type)
        assert "1 Chicken Burger" in text

    def test_single_item_no_leading_bare_name(self) -> None:
        """The bare item name without quantity prefix must not appear before 'Please review'."""
        payload, order_type = _checkout_payload(
            [{"name": "Burger", "quantity": 1}]
        )
        text = render_checkout_review_summary(payload, order_type)
        review_idx = text.index("Please review your order:")
        after_review = text[review_idx:]
        assert "1 Burger" in after_review


class TestMultipleDefaultQuantities:
    def test_multiple_default_quantities_all_render_with_1(self) -> None:
        items = [
            {"name": "Chicken Burger", "quantity": 1},
            {"name": "Chicken Taco", "quantity": 1},
            {"name": "Bourbon Chicken", "quantity": 1},
            {"name": "Cheese Burger", "quantity": 1},
        ]
        payload, order_type = _checkout_payload(items, total="$35.45")
        text = render_checkout_review_summary(payload, order_type)
        assert "1 Chicken Burger" in text
        assert "1 Chicken Taco" in text
        assert "1 Bourbon Chicken" in text
        assert "1 Cheese Burger" in text


class TestExplicitQuantity:
    def test_explicit_quantity_renders_unchanged(self) -> None:
        payload, order_type = _checkout_payload(
            [{"name": "Coke", "quantity": 3}]
        )
        text = render_checkout_review_summary(payload, order_type)
        assert "3 Coke" in text

    def test_mixed_default_and_explicit_quantities(self) -> None:
        items = [
            {"name": "Burger", "quantity": 1},
            {"name": "Coke", "quantity": 2},
        ]
        payload, order_type = _checkout_payload(items)
        text = render_checkout_review_summary(payload, order_type)
        assert "1 Burger" in text
        assert "2 Coke" in text


class TestEdgeQuantities:
    def test_qty_zero_is_normalized_to_1(self) -> None:
        payload, order_type = _checkout_payload(
            [{"name": "Burger", "quantity": 0}]
        )
        text = render_checkout_review_summary(payload, order_type)
        assert "1 Burger" in text
        assert "0 Burger" not in text

    def test_qty_none_is_normalized_to_1(self) -> None:
        payload, order_type = _checkout_payload(
            [{"name": "Coke", "quantity": None}]
        )
        text = render_checkout_review_summary(payload, order_type)
        assert "1 Coke" in text
        assert "None" not in text

    def test_qty_float_0_1_normalized_to_1(self) -> None:
        payload, order_type = _checkout_payload(
            [{"name": "Coke", "quantity": 0.1}]
        )
        text = render_checkout_review_summary(payload, order_type)
        assert "1 Coke" in text
        assert "0 Coke" not in text


class TestPluralBehavior:
    def test_plural_name_absent_no_naive_pluralization(self) -> None:
        """Without explicit plural, the name is never modified."""
        payload, order_type = _checkout_payload(
            [{"name": "French Fries", "quantity": 2}]
        )
        text = render_checkout_review_summary(payload, order_type)
        assert "2 French Fries" in text
        assert "French Friess" not in text


class TestSidesAndModifiersNotIncluded:
    def test_sides_not_in_checkout_summary(self) -> None:
        """render_checkout_review_summary uses compact line renderer (no sides)."""
        items = [{"name": "Chicken Burger", "quantity": 1, "sides": ["Coke"]}]
        payload, order_type = _checkout_payload(items)
        text = render_checkout_review_summary(payload, order_type)
        assert "Coke" not in text
        assert "1 Chicken Burger" in text

    def test_modifiers_not_in_checkout_summary(self) -> None:
        items = [{"name": "Chicken Burger", "quantity": 1, "modifiers": ["American Cheese"]}]
        payload, order_type = _checkout_payload(items)
        text = render_checkout_review_summary(payload, order_type)
        assert "American Cheese" not in text
        assert "1 Chicken Burger" in text

    def test_with_keyword_not_in_checkout_summary(self) -> None:
        items = [{"name": "Chicken Burger", "quantity": 1}]
        payload, order_type = _checkout_payload(items)
        text = render_checkout_review_summary(payload, order_type)
        review_section = text.split("Please review your order:")[1]
        assert " with " not in review_section.lower()


class TestOrderTypeIntro:
    def test_order_type_pickup_intro_preserved(self) -> None:
        payload = {"items": [{"name": "Burger", "quantity": 1}], "total": "$8.00"}
        text = render_checkout_review_summary(payload, "pickup")
        assert text.startswith("This is a pickup order.")

    def test_order_type_delivery_intro_preserved(self) -> None:
        payload = {"items": [{"name": "Burger", "quantity": 1}], "total": "$8.00"}
        text = render_checkout_review_summary(payload, "delivery")
        assert text.startswith("This is a delivery order.")

    def test_no_order_type_no_intro(self) -> None:
        payload = {"items": [{"name": "Burger", "quantity": 1}], "total": "$8.00"}
        text = render_checkout_review_summary(payload)
        assert text.startswith("Please review your order:")

    def test_exact_production_case(self) -> None:
        """Exact reproduction of the reported production failure."""
        items = [
            {"name": "Chicken Burger", "quantity": 1},
            {"name": "Chicken Taco", "quantity": 1},
            {"name": "Bourbon Chicken", "quantity": 1},
            {"name": "Cheese Burger", "quantity": 1},
        ]
        payload = {"items": items, "total": "$35.45"}
        text = render_checkout_review_summary(payload, "pickup")
        assert text == (
            "This is a pickup order. Please review your order: "
            "1 Chicken Burger. 1 Chicken Taco. 1 Bourbon Chicken. 1 Cheese Burger."
            " Your total is $35.45. "
            "Should I place the order?"
        )
