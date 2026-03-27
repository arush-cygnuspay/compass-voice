import pytest


class DummyContext:
    def __init__(self):
        self.cart = []
        self.waiting_for = None
        self.pending_item = None
        self.pending_side = None
        self.pending_modifiers = []
        self.pending_variant = None


def add_item_to_cart(ctx, item):
    ctx.cart.append(item)


def test_simple_item_no_options():
    ctx = DummyContext()

    item = {
        "name": "Chicken Taco",
        "price": 499
    }

    add_item_to_cart(ctx, item)

    assert len(ctx.cart) == 1
    assert ctx.cart[0]["name"] == "Chicken Taco"


def test_variant_item_requires_size():
    ctx = DummyContext()

    item = {
        "name": "Coke",
        "variants": ["small", "medium", "large"]
    }

    ctx.pending_item = item
    ctx.waiting_for = "size"

    assert ctx.waiting_for == "size"


def test_variant_selected():
    ctx = DummyContext()

    item = {
        "name": "Coke",
        "variants": ["small", "medium", "large"]
    }

    ctx.pending_item = item
    ctx.waiting_for = "size"

    ctx.pending_variant = "medium"
    ctx.waiting_for = None

    add_item_to_cart(ctx, {
        "name": "Coke",
        "size": "medium"
    })

    assert ctx.cart[0]["size"] == "medium"


def test_combo_requires_side():
    ctx = DummyContext()

    combo = {
        "name": "Crabcake Combo",
        "side_groups": ["drink"]
    }

    ctx.pending_item = combo
    ctx.waiting_for = "side"

    assert ctx.waiting_for == "side"


def test_side_selected():
    ctx = DummyContext()

    ctx.pending_item = {"name": "Crabcake Combo"}
    ctx.waiting_for = "side"

    ctx.pending_side = "Coke"

    add_item_to_cart(ctx, {
        "name": "Crabcake Combo",
        "side": "Coke"
    })

    assert ctx.cart[0]["side"] == "Coke"


def test_optional_modifier():
    ctx = DummyContext()

    ctx.pending_item = {"name": "Chicken Taco"}
    ctx.waiting_for = "modifier"

    ctx.pending_modifiers.append("Cheese")

    add_item_to_cart(ctx, {
        "name": "Chicken Taco",
        "modifiers": ["Cheese"]
    })

    assert ctx.cart[0]["modifiers"] == ["Cheese"]


def test_user_skips_modifier():
    ctx = DummyContext()

    ctx.pending_item = {"name": "Chicken Taco"}
    ctx.waiting_for = "modifier"

    ctx.waiting_for = None

    add_item_to_cart(ctx, {
        "name": "Chicken Taco",
        "modifiers": []
    })

    assert ctx.cart[0]["modifiers"] == []