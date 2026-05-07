# tests/cart/test_cart_item_decimal_quantity.py
"""Tests for CartItem deserialization with decimal-encoded quantities (Phase 2).

When a cart is serialized/deserialized and the quantity field contains a float
like 0.2 (ASR decimal encoding for 2 items), CartItem.from_dict must decode
it correctly to 2, not round it to 0 or clamp it to 1.
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from app.cart.cart_item import CartItem
from app.core.quantity_formatter import parse_item_quantity


# ---------------------------------------------------------------------------
# parse_item_quantity (the function CartItem.from_dict uses)
# ---------------------------------------------------------------------------

class TestParseItemQuantityAtCartBoundary:
    @pytest.mark.parametrize("raw,expected", [
        # Normal integer quantities
        (1, 1),
        (2, 2),
        (3, 3),
        # ASR decimal encoding: 0.N → N
        (0.2, 2),
        (0.1, 1),
        (0.3, 3),
        (0.9, 9),
        # Exact-integer floats
        (1.0, 1),
        (2.0, 2),
        # Decimal type (from JSON Decimal coercion)
        (Decimal("0.2"), 2),
        (Decimal("1.0"), 1),
        # String representations
        ("1", 1),
        ("2", 2),
        ("0.2", 2),
        # Fallback cases (floor-clamped to 1)
        (0, 1),
        (None, 1),
        (0.01, 1),    # leading-zero fraction → None → fallback → 1
    ])
    def test_parse_item_quantity(self, raw, expected):
        assert parse_item_quantity(raw) == expected, (
            f"parse_item_quantity({raw!r}) expected {expected}, "
            f"got {parse_item_quantity(raw)}"
        )


# ---------------------------------------------------------------------------
# CartItem.from_dict — the actual deserialization path
# ---------------------------------------------------------------------------

def _base_item_dict(quantity) -> dict:
    return {
        "cart_item_id": "test-id-001",
        "item_id": "item-coke",
        "quantity": quantity,
        "variant_id": None,
        "sides": {},
        "side_variants": {},
        "modifiers": {},
    }


class TestCartItemFromDictDecimalQuantity:
    def test_integer_quantity_deserialized(self):
        item = CartItem.from_dict(_base_item_dict(2))
        assert item.quantity == 2

    def test_float_0_2_deserialized_as_2(self):
        """ASR decimal encoding: stored as 0.2 → quantity is 2."""
        item = CartItem.from_dict(_base_item_dict(0.2))
        assert item.quantity == 2, (
            f"Expected quantity 2, got {item.quantity}. "
            "The old round() bug would have produced 1."
        )

    def test_float_0_1_deserialized_as_1(self):
        item = CartItem.from_dict(_base_item_dict(0.1))
        assert item.quantity == 1

    def test_float_0_3_deserialized_as_3(self):
        item = CartItem.from_dict(_base_item_dict(0.3))
        assert item.quantity == 3

    def test_float_1_0_deserialized_as_1(self):
        item = CartItem.from_dict(_base_item_dict(1.0))
        assert item.quantity == 1

    def test_float_0_01_falls_back_to_1(self):
        """0.01 is not a valid single-digit encoding → falls back to 1."""
        item = CartItem.from_dict(_base_item_dict(0.01))
        assert item.quantity == 1

    def test_quantity_is_always_int(self):
        """CartItem.quantity must be a Python int, never float."""
        item = CartItem.from_dict(_base_item_dict(0.2))
        assert isinstance(item.quantity, int)

    def test_quantity_never_zero(self):
        """Even a malformed 0 quantity floors to 1."""
        item = CartItem.from_dict(_base_item_dict(0))
        assert item.quantity >= 1
