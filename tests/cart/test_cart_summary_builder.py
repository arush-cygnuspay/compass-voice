from types import SimpleNamespace

from app.cart.cart_item import CartItem
from app.cart.read_models.cart_summary_builder import CartSummaryBuilder


class FakeCart:
    def __init__(self, items):
        self._items = items

    def get_items(self):
        return list(self._items)


class FakeMenuRepo:
    def __init__(self, item):
        self._item = item

    def get_item(self, item_id):
        return self._item


def test_cart_summary_builder_handles_rich_modifier_entries():
    menu_item = SimpleNamespace(
        name="Bourbon Chicken",
        pricing=SimpleNamespace(price_cents=1000, variants=[]),
        side_groups=[],
        modifier_groups=[
            SimpleNamespace(
                group_id="mods",
                choices=[
                    SimpleNamespace(modifier_id="pepper", name="Green Bell Pepper", price_cents=150),
                    SimpleNamespace(modifier_id="onion", name="Onion", price_cents=100),
                ],
            )
        ],
    )

    cart_item = CartItem.create(
        item_id="bourbon",
        quantity=1,
        variant_id=None,
        sides={},
        side_variants={},
        modifiers={
            "mods": [
                {
                    "modifier_id": "pepper",
                    "name": "Green Bell Pepper",
                    "action": "add",
                    "instruction": None,
                },
                {
                    "modifier_id": "onion",
                    "name": "Onion",
                    "action": "remove",
                    "instruction": None,
                },
            ]
        },
    )

    summary = CartSummaryBuilder(FakeMenuRepo(menu_item)).build(FakeCart([cart_item]))

    assert summary["items"][0]["modifiers"] == ["Green Bell Pepper", "no Onion"]
    assert summary["items"][0]["unit_price"] == "$11.50"


def test_cart_summary_builder_renders_on_side_modifier_instruction():
    menu_item = SimpleNamespace(
        name="Wings",
        pricing=SimpleNamespace(price_cents=1200, variants=[]),
        side_groups=[],
        modifier_groups=[
            SimpleNamespace(
                group_id="mods",
                choices=[
                    SimpleNamespace(modifier_id="ranch", name="Ranch", price_cents=50),
                ],
            )
        ],
    )

    cart_item = CartItem.create(
        item_id="wings",
        quantity=1,
        variant_id=None,
        sides={},
        side_variants={},
        modifiers={
            "mods": [
                {
                    "modifier_id": "ranch",
                    "name": "Ranch",
                    "action": "add",
                    "instruction": "on_side",
                }
            ]
        },
    )

    summary = CartSummaryBuilder(FakeMenuRepo(menu_item)).build(FakeCart([cart_item]))

    assert summary["items"][0]["modifiers"] == ["Ranch on the side"]
    assert summary["items"][0]["unit_price"] == "$12.50"
