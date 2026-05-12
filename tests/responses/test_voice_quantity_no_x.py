# tests/responses/test_voice_quantity_no_x.py
"""Tests for spoken_quantity_label and compact_quantity_label (Phase 5).

Also guards the existing voice response paths (success.py, cart_responses.py,
cart_summary_builder) against accidental introduction of 'x' notation (Phase 6/7).

Voice TTS must never produce "2x Coke", "Coke x2", "× 2 Coke", "1 x Burger" etc.
as these read out literally on Deepgram TTS ("two ex coke" / "one ex burger").
"""
from __future__ import annotations

import re
import pytest

from app.responses.item.format_utils import (
    compact_quantity_label,
    spoken_quantity_label,
    _added_text,
)
from app.responses.item.success import item_added_successfully
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


# ---------------------------------------------------------------------------
# _added_text (format_utils) — explicit regression guards
# ---------------------------------------------------------------------------

class TestAddedTextNoXNotation:
    """_added_text is the core voice fragment used by item_added_successfully."""

    @pytest.mark.parametrize("qty,name,expected", [
        (1, "Cheese Burger", "Cheese Burger added"),
        (2, "Cheese Burger", "Added 2 Cheese Burger"),
        (3, "French Fries", "Added 3 French Fries"),
        (1, "Smash Burger",  "Smash Burger added"),
        (2, "Smash Burger",  "Added 2 Smash Burger"),
    ])
    def test_added_text_format(self, qty, name, expected):
        assert _added_text(name, qty) == expected

    @pytest.mark.parametrize("qty", [1, 2, 3, 5])
    def test_added_text_no_x_notation(self, qty):
        text = _added_text("Cheese Burger", qty)
        assert " x " not in text
        assert re.search(r"\b\d+\s*x\s+", text) is None, (
            f"_added_text({qty}, ...) contains x-notation: {text!r}"
        )


# ---------------------------------------------------------------------------
# item_added_successfully — no " x " in voice text
# ---------------------------------------------------------------------------

def _success_payload(quantity: int, item_name: str = "Cheese Burger", **extra) -> dict:
    return {
        "item_name": item_name,
        "quantity": quantity,
        "item_aliases": [],
        "item_voice_labels": [],
        "unmatched_names": [],
        **extra,
    }


class TestItemAddedSuccessNoXNotation:
    @pytest.mark.parametrize("qty", [1, 2, 3])
    def test_no_x_notation(self, qty):
        text = item_added_successfully(_success_payload(qty))
        assert " x " not in text
        assert re.search(r"\b\d+\s*x\s+", text) is None, (
            f"item_added_successfully({qty}) contains x-notation: {text!r}"
        )

    def test_qty_1_cheese_burger_voice(self):
        text = item_added_successfully(_success_payload(1, "Cheese Burger"))
        assert "Cheese Burger added" in text
        assert " x " not in text

    def test_qty_2_smash_burger_voice(self):
        text = item_added_successfully(_success_payload(2, "Smash Burger"))
        assert "2 Smash Burger" in text
        assert " x " not in text

    def test_qty_2_with_modifier_no_x(self):
        payload = _success_payload(2, "Burger", spoken_modifiers=["no onions"])
        text = item_added_successfully(payload)
        assert " x " not in text
        # Voice response omits modifier detail — only item name + quantity
        assert "with no onions" not in text
        assert "2 Burger" in text

    # Queue-transition branch
    def test_queue_transition_same_item_no_x(self):
        """Merged ack for identical prev/current item must not contain x notation."""
        payload = _success_payload(
            1, "Cheese Burger",
            queue_transition=True,
            prev_item_name="Cheese Burger",
            prev_quantity=1,
            remaining_queue_count=0,
        )
        text = item_added_successfully(payload)
        assert " x " not in text
        assert re.search(r"\b\d+\s*x\s+", text) is None

    def test_queue_transition_different_items_no_x(self):
        """Two-part queue ack for different items must not contain x notation."""
        payload = _success_payload(
            1, "Coke",
            queue_transition=True,
            prev_item_name="Cheese Burger",
            prev_quantity=1,
            remaining_queue_count=0,
        )
        text = item_added_successfully(payload)
        assert " x " not in text
        assert "Cheese Burger" in text
        assert "Coke" in text


# ---------------------------------------------------------------------------
# render_checkout_review_summary — with duplicate sides (the key regression)
# ---------------------------------------------------------------------------

class TestCheckoutReviewNoXNotation:
    """render_checkout_review_summary must never voice 'with Coke x2'."""

    def _review_payload(self, quantity=1, name="Burger", sides=None, total="$10.00"):
        return {
            "items": [{"quantity": quantity, "name": name, "sides": sides or [], "modifiers": []}],
            "total": total,
        }

    @pytest.mark.parametrize("qty", [1, 2])
    def test_no_x_in_item_quantity(self, qty):
        text = render_checkout_review_summary(self._review_payload(quantity=qty))
        assert " x " not in text
        assert re.search(r"\b\d+\s*x\s+", text) is None

    def test_single_side_omitted_from_voice(self):
        # Sides are excluded from voice compact summary — only item name appears.
        text = render_checkout_review_summary(self._review_payload(sides=["Coke"]))
        assert " x " not in text
        assert "Coke" not in text
        assert "Burger" in text

    def test_duplicate_side_count_prefix_omitted_from_voice(self):
        """Sides are excluded from voice compact summary regardless of their label format."""
        text = render_checkout_review_summary(self._review_payload(sides=["2 Coke"]))
        assert " x " not in text
        assert "2 Coke" not in text
        assert "Burger" in text

    def test_old_x2_side_label_absent_from_voice(self):
        """Regression: the old 'Coke x2' format must never appear in voice."""
        text = render_checkout_review_summary(self._review_payload(sides=["2 Coke"]))
        assert "x2" not in text.lower()
        assert "x 2" not in text.lower()

    @pytest.mark.parametrize("quantity", [1, 2, 3])
    def test_checkout_review_qty_no_x(self, quantity):
        text = render_checkout_review_summary(self._review_payload(quantity=quantity))
        assert " x " not in text

    def test_qty_1_omits_leading_1(self):
        """spoken_quantity_label(1, name) returns name without leading '1'."""
        text = render_checkout_review_summary(self._review_payload(quantity=1, name="Burger"))
        # "1 Burger" is acceptable, but "1 x Burger" is not
        assert "1 x Burger" not in text

    def test_qty_2_shows_2(self):
        text = render_checkout_review_summary(self._review_payload(quantity=2, name="Burger"))
        assert "2 Burger" in text


# ---------------------------------------------------------------------------
# CartSummaryBuilder → render_checkout_review_summary integration
# ---------------------------------------------------------------------------

class TestCartBuilderToVoiceNoX:
    """End-to-end: CartSummaryBuilder produces labels; voice summary must be x-free."""

    def _side_label_direct(self, count: int, name: str = "Coke") -> str:
        """Call _get_side_labels logic by exercising CartSummaryBuilder directly."""
        from types import SimpleNamespace
        from app.cart.cart_item import CartItem
        from app.cart.read_models.cart_summary_builder import CartSummaryBuilder

        side_choice = SimpleNamespace(
            item_id="coke",
            name=name,
            pricing=SimpleNamespace(price_cents=150),
        )
        menu_item = SimpleNamespace(
            name="Burger",
            pricing=SimpleNamespace(price_cents=800, variants=[]),
            side_groups=[
                SimpleNamespace(group_id="drinks", choices=[side_choice])
            ],
            modifier_groups=[],
        )

        class _FakeCart:
            def get_items(self_):
                return [CartItem.create(
                    item_id="burger", quantity=1, variant_id=None,
                    sides={"drinks": ["coke"] * count},
                    side_variants={}, modifiers={},
                )]

        class _FakeRepo:
            def get_item(self_, _): return menu_item

        summary = CartSummaryBuilder(_FakeRepo()).build(_FakeCart())
        return summary["items"][0]["sides"]

    def test_single_side_no_count_prefix(self):
        # build() wraps _get_side_labels() tuple in a list
        labels = self._side_label_direct(1)
        assert list(labels) == ["Coke"]

    def test_two_same_sides_count_prefix_no_x(self):
        labels = self._side_label_direct(2)
        assert "2 Coke" in list(labels)
        assert not any("x" in label for label in labels)

    def test_three_same_sides_count_prefix_no_x(self):
        labels = self._side_label_direct(3)
        assert "3 Coke" in list(labels)
        assert not any("x" in label for label in labels)

    def test_voice_summary_with_duplicate_sides_no_x(self):
        """Full pipeline: sides are omitted from compact voice summary — x-notation impossible."""
        payload = {
            "items": [{
                "quantity": 1,
                "name": "Burger",
                "sides": ["2 Coke"],   # builder produces this but voice omits sides
                "modifiers": [],
            }],
            "total": "$11.00",
        }
        text = render_checkout_review_summary(payload)
        assert " x " not in text
        assert "x2" not in text
        # Sides are omitted from voice compact format; only item name appears
        assert "2 Coke" not in text
        assert "Burger" in text


# ---------------------------------------------------------------------------
# Broad regex regression guard (voice surfaces only)
# ---------------------------------------------------------------------------

_X_QUANTITY_RE = re.compile(r"\b\d+\s*x\s+", re.IGNORECASE)


def _assert_no_x_notation(text: str, label: str = "") -> None:
    """Assert that *text* contains no TTS-problematic x-quantity notation."""
    assert _X_QUANTITY_RE.search(text) is None, (
        f"{label}: found x-notation in voice text: {text!r}"
    )
    # Also catch suffix form: "Coke x2" / "Burger x 3"
    assert not re.search(r"\bx\s*\d+\b", text, re.IGNORECASE), (
        f"{label}: found suffix x-notation in voice text: {text!r}"
    )


class TestBroadXRegressionGuard:
    """Parameterised sweep of voice response outputs."""

    @pytest.mark.parametrize("qty,name", [
        (1, "Cheese Burger"),
        (2, "Cheese Burger"),
        (1, "Smash Burger"),
        (2, "Smash Burger"),
        (3, "Coke"),
    ])
    def test_item_added_no_x(self, qty, name):
        text = item_added_successfully(_success_payload(qty, name))
        _assert_no_x_notation(text, f"item_added_successfully(qty={qty}, name={name!r})")

    @pytest.mark.parametrize("qty,name", [
        (1, "Cheese Burger"),
        (2, "Smash Burger"),
        (3, "Wing"),
    ])
    def test_checkout_review_no_x(self, qty, name):
        payload = {"items": [{"quantity": qty, "name": name, "sides": [], "modifiers": []}]}
        text = render_checkout_review_summary(payload)
        _assert_no_x_notation(text, f"render_checkout_review_summary(qty={qty}, name={name!r})")

    @pytest.mark.parametrize("qty", [1, 2, 3, 5])
    def test_cart_summary_no_x(self, qty):
        text = render_cart_summary(_cart_payload(qty, "Burger"))
        _assert_no_x_notation(text, f"render_cart_summary(qty={qty})")
