# tests/state_machine/handlers/item/add_item/test_group_classification.py
"""Phase 1 — group_classification unit tests."""
from __future__ import annotations

import pytest

from app.state_machine.handlers.item.add_item.group_classification import (
    is_drink_like_group,
    ordinal_word,
    speech_noun_for_side_group,
)


class TestIsDrinkLikeGroup:
    # ── positive examples ──────────────────────────────────────────────────────
    @pytest.mark.parametrize("name", [
        "Choose Drink",
        "Family Meal Drinks",
        "Burger Meal Drinks",
        "Can Drinks",
        "Drinks",
        "Hot Beverages",
        "Soft Drinks",
        "Fountain Drinks",
        "Soda",
        "Sodas",
        "Beverages",
        "Beverage",
        "DRINKS",                 # case insensitive
        "Can drinks option",
    ])
    def test_positive(self, name):
        assert is_drink_like_group(name), f"Expected {name!r} to be drink-like"

    # ── negative examples ──────────────────────────────────────────────────────
    @pytest.mark.parametrize("name", [
        "Family Wing Sauces",
        "Choose Side",
        "Choose Your Side",
        "Platter Sides",
        "Sandwich Sides",
        "Wing Choice",
        "Add-ons",
        "Drink Modifications",    # blocklisted
        "Can Modifications",      # blocklisted
        "Drink modification",     # blocklisted (singular)
        "",
        "Toppings",
        "Protein Choice",
        "Sauce",
    ])
    def test_negative(self, name):
        assert not is_drink_like_group(name), f"Expected {name!r} to NOT be drink-like"


class TestSpeechNounForSideGroup:
    def test_drink_group_returns_drink(self):
        assert speech_noun_for_side_group("Can Drinks") == "drink"

    def test_non_drink_group_returns_side(self):
        assert speech_noun_for_side_group("Choose Side") == "side"

    def test_empty_returns_side(self):
        assert speech_noun_for_side_group("") == "side"

    def test_beverages_returns_drink(self):
        assert speech_noun_for_side_group("Hot Beverages") == "drink"

    def test_platter_sides_returns_side(self):
        assert speech_noun_for_side_group("Platter Sides") == "side"


class TestOrdinalWord:
    @pytest.mark.parametrize("n,expected", [
        (1, "first"),
        (2, "second"),
        (3, "third"),
        (4, "fourth"),
        (5, "fifth"),
        (10, "tenth"),
    ])
    def test_known_ordinals(self, n, expected):
        assert ordinal_word(n) == expected

    def test_beyond_list_falls_back_to_nth(self):
        assert ordinal_word(11) == "11th"
        assert ordinal_word(99) == "99th"
