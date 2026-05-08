# tests/cart/test_cart_summary_builder_duplicate_sides.py
"""Duplicate-side pricing and display tests for CartSummaryBuilder.

Verifies that sides stored as repeated IDs in CartItem.sides are:
1. Priced once per occurrence (not once per unique ID).
2. Displayed with "{count} {name}" labels (e.g. "2 Coke") — never "Coke x2"
   so the TTS engine reads "two Coke" rather than "Coke ex two".
3. Grouped correctly with other cart items that have the same duplicate pattern.
"""
from types import SimpleNamespace

from app.cart.cart_item import CartItem
from app.cart.read_models.cart_summary_builder import CartSummaryBuilder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeCart:
    def __init__(self, items):
        self._items = items

    def get_items(self):
        return list(self._items)


class _FakeMenuRepo:
    def __init__(self, item):
        self._item = item

    def get_item(self, _item_id):
        return self._item


def _make_menu_item(
    *,
    name="Burger",
    base_price=800,
    side_group_id="drinks",
    side_choices=None,
):
    if side_choices is None:
        side_choices = [
            SimpleNamespace(
                item_id="coke",
                name="Coke",
                pricing=SimpleNamespace(price_cents=150),
            ),
            SimpleNamespace(
                item_id="sprite",
                name="Sprite",
                pricing=SimpleNamespace(price_cents=150),
            ),
        ]
    return SimpleNamespace(
        name=name,
        pricing=SimpleNamespace(price_cents=base_price, variants=[]),
        side_groups=[
            SimpleNamespace(
                group_id=side_group_id,
                choices=side_choices,
            )
        ],
        modifier_groups=[],
    )


def _make_cart_item(
    *,
    item_id="burger",
    quantity=1,
    sides=None,
):
    return CartItem.create(
        item_id=item_id,
        quantity=quantity,
        variant_id=None,
        sides=sides or {},
        side_variants={},
        modifiers={},
    )


# ---------------------------------------------------------------------------
# Pricing tests
# ---------------------------------------------------------------------------

class TestDuplicateSidePricing:
    def test_single_side_charges_once(self):
        menu_item = _make_menu_item()
        cart_item = _make_cart_item(sides={"drinks": ["coke"]})
        summary = CartSummaryBuilder(_FakeMenuRepo(menu_item)).build(_FakeCart([cart_item]))

        # base 800 + coke 150 = 950
        assert summary["items"][0]["unit_price"] == "$9.50"

    def test_two_same_sides_charges_twice(self):
        menu_item = _make_menu_item()
        cart_item = _make_cart_item(sides={"drinks": ["coke", "coke"]})
        summary = CartSummaryBuilder(_FakeMenuRepo(menu_item)).build(_FakeCart([cart_item]))

        # base 800 + coke 150 + coke 150 = 1100
        assert summary["items"][0]["unit_price"] == "$11.00"

    def test_three_same_sides_charges_three_times(self):
        menu_item = _make_menu_item()
        cart_item = _make_cart_item(sides={"drinks": ["coke", "coke", "coke"]})
        summary = CartSummaryBuilder(_FakeMenuRepo(menu_item)).build(_FakeCart([cart_item]))

        # base 800 + 3 * 150 = 1250
        assert summary["items"][0]["unit_price"] == "$12.50"

    def test_mixed_sides_each_charged_by_count(self):
        menu_item = _make_menu_item()
        # 2 cokes + 1 sprite
        cart_item = _make_cart_item(sides={"drinks": ["coke", "coke", "sprite"]})
        summary = CartSummaryBuilder(_FakeMenuRepo(menu_item)).build(_FakeCart([cart_item]))

        # base 800 + 2*150 + 1*150 = 1250
        assert summary["items"][0]["unit_price"] == "$12.50"

    def test_zero_price_side_does_not_inflate(self):
        menu_item = _make_menu_item(
            side_choices=[
                SimpleNamespace(
                    item_id="water",
                    name="Water",
                    pricing=SimpleNamespace(price_cents=0),
                )
            ]
        )
        cart_item = _make_cart_item(sides={"drinks": ["water", "water", "water"]})
        summary = CartSummaryBuilder(_FakeMenuRepo(menu_item)).build(_FakeCart([cart_item]))

        # base 800 + 0*3 = 800
        assert summary["items"][0]["unit_price"] == "$8.00"

    def test_line_total_reflects_quantity_and_duplicate_sides(self):
        menu_item = _make_menu_item()
        # 2 cokes per burger, ordering 2 burgers
        cart_item = _make_cart_item(quantity=2, sides={"drinks": ["coke", "coke"]})
        summary = CartSummaryBuilder(_FakeMenuRepo(menu_item)).build(_FakeCart([cart_item]))

        # unit = 800 + 300 = 1100; line = 1100 * 2 = 2200
        assert summary["items"][0]["unit_price"] == "$11.00"
        assert summary["items"][0]["line_total"] == "$22.00"


# ---------------------------------------------------------------------------
# Display label tests
# ---------------------------------------------------------------------------

class TestDuplicateSideDisplayLabels:
    def test_single_side_no_multiplier(self):
        menu_item = _make_menu_item()
        cart_item = _make_cart_item(sides={"drinks": ["coke"]})
        summary = CartSummaryBuilder(_FakeMenuRepo(menu_item)).build(_FakeCart([cart_item]))

        assert summary["items"][0]["sides"] == ["Coke"]

    def test_two_same_sides_shows_count_prefix(self):
        """Duplicate sides render as '2 Coke', never 'Coke x2' (bad for TTS)."""
        menu_item = _make_menu_item()
        cart_item = _make_cart_item(sides={"drinks": ["coke", "coke"]})
        summary = CartSummaryBuilder(_FakeMenuRepo(menu_item)).build(_FakeCart([cart_item]))

        sides = summary["items"][0]["sides"]
        assert sides == ["2 Coke"]
        # Regression guard: no "x" notation must appear
        assert all("x" not in s for s in sides)

    def test_three_same_sides_shows_count_prefix(self):
        """Three duplicate sides render as '3 Coke', never 'Coke x3'."""
        menu_item = _make_menu_item()
        cart_item = _make_cart_item(sides={"drinks": ["coke", "coke", "coke"]})
        summary = CartSummaryBuilder(_FakeMenuRepo(menu_item)).build(_FakeCart([cart_item]))

        sides = summary["items"][0]["sides"]
        assert sides == ["3 Coke"]
        assert all("x" not in s for s in sides)

    def test_mixed_sides_correct_labels(self):
        menu_item = _make_menu_item()
        cart_item = _make_cart_item(sides={"drinks": ["coke", "coke", "sprite"]})
        summary = CartSummaryBuilder(_FakeMenuRepo(menu_item)).build(_FakeCart([cart_item]))

        # Sprite appears once → no prefix; Coke twice → "2 Coke"
        sides = summary["items"][0]["sides"]
        assert "2 Coke" in sides
        assert "Sprite" in sides
        # No "x" notation anywhere — regression guard
        assert not any("x" in s for s in sides)
        assert "1 Sprite" not in sides  # single items carry no count prefix

    def test_two_different_sides_no_multiplier(self):
        menu_item = _make_menu_item()
        cart_item = _make_cart_item(sides={"drinks": ["coke", "sprite"]})
        summary = CartSummaryBuilder(_FakeMenuRepo(menu_item)).build(_FakeCart([cart_item]))

        sides = summary["items"][0]["sides"]
        assert "Coke" in sides
        assert "Sprite" in sides
        assert all("x" not in s for s in sides)


# ---------------------------------------------------------------------------
# Cart total tests
# ---------------------------------------------------------------------------

class TestDuplicateSideCartTotal:
    def test_cart_total_includes_duplicate_sides(self):
        menu_item = _make_menu_item()
        cart_item = _make_cart_item(sides={"drinks": ["coke", "coke"]})
        summary = CartSummaryBuilder(_FakeMenuRepo(menu_item)).build(_FakeCart([cart_item]))

        # 800 + 300 = 1100
        assert summary["total"] == "$11.00"

    def test_two_cart_items_totalled_correctly(self):
        menu_item = _make_menu_item()
        item1 = _make_cart_item(item_id="b1", sides={"drinks": ["coke", "coke"]})
        item2 = _make_cart_item(item_id="b2", sides={"drinks": ["sprite"]})

        # item1: 800 + 300 = 1100; item2: 800 + 150 = 950; total 2050
        summary = CartSummaryBuilder(_FakeMenuRepo(menu_item)).build(_FakeCart([item1, item2]))
        assert summary["total"] == "$20.50"


# ---------------------------------------------------------------------------
# Phase 7: Group key tests — duplicates produce distinct keys from singles
# ---------------------------------------------------------------------------

class TestGroupKeyWithDuplicates:
    """_build_group_key uses sorted(item_ids) so tuple length encodes count."""

    def _builder(self):
        return CartSummaryBuilder(_FakeMenuRepo(_make_menu_item()))

    def test_single_vs_double_produce_different_keys(self):
        item_one = _make_cart_item(sides={"drinks": ["coke"]})
        item_two = _make_cart_item(sides={"drinks": ["coke", "coke"]})
        builder = self._builder()
        assert builder._build_group_key(item_one) != builder._build_group_key(item_two)

    def test_two_items_with_same_double_share_key(self):
        item_a = _make_cart_item(item_id="a", sides={"drinks": ["coke", "coke"]})
        item_b = _make_cart_item(item_id="b", sides={"drinks": ["coke", "coke"]})
        builder = self._builder()
        # item_id is also part of the key — so these will differ by item_id.
        # But the sides portion must be identical.
        key_a = builder._build_group_key(item_a)
        key_b = builder._build_group_key(item_b)
        # sides component is at index 2
        assert key_a[2] == key_b[2]

    def test_two_cart_items_with_same_duplicate_sides_grouped_together(self):
        """Two CartItems with identical item_id AND identical duplicate sides collapse into one line."""
        menu_item = _make_menu_item()
        # Same item_id "burger" and same sides ["coke","coke"] → same group key → merged
        item_a = _make_cart_item(sides={"drinks": ["coke", "coke"]})
        item_b = _make_cart_item(sides={"drinks": ["coke", "coke"]})
        summary = CartSummaryBuilder(_FakeMenuRepo(menu_item)).build(_FakeCart([item_a, item_b]))
        # Should be 1 line item with quantity=2
        assert len(summary["items"]) == 1
        assert summary["items"][0]["quantity"] == 2

    def test_single_and_double_not_grouped(self):
        """item with ["coke"] and item with ["coke","coke"] must NOT merge."""
        menu_item = _make_menu_item()
        item_one = _make_cart_item(sides={"drinks": ["coke"]})
        item_two = _make_cart_item(sides={"drinks": ["coke", "coke"]})
        summary = CartSummaryBuilder(_FakeMenuRepo(menu_item)).build(_FakeCart([item_one, item_two]))
        assert len(summary["items"]) == 2

    def test_order_within_group_does_not_affect_key(self):
        """["coke","sprite"] and ["sprite","coke"] have same sorted tuple → same key."""
        item_a = _make_cart_item(sides={"drinks": ["coke", "sprite"]})
        item_b = _make_cart_item(sides={"drinks": ["sprite", "coke"]})
        builder = self._builder()
        assert builder._build_group_key(item_a) == builder._build_group_key(item_b)
