# app/state_machine/handlers/item/add_item/group_classification.py
"""Classifier helpers for side and modifier group voice UX.

Used by:
- pending_add_item_factory  → sort drinks last among side groups
- add_item_flow             → attach prompt metadata to side/modifier-step payload
- sides.py                  → build progressive/ordinal wording
- modifiers.py              → build progressive modifier wording
"""
from __future__ import annotations

import re

# Tokens whose presence in a normalised group name signals a drink-like group.
DRINK_TOKENS: frozenset[str] = frozenset({
    "drink", "drinks", "beverage", "beverages",
    "soda", "sodas", "fountain", "soft", "can",
})

# Tokens that veto drink classification even when DRINK_TOKENS match.
DRINK_HARD_BLOCKLIST: frozenset[str] = frozenset({
    "modification", "modifications",
})

_ORDINALS = (
    "first", "second", "third", "fourth",
    "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
)


def _tokenize(name: str) -> list[str]:
    """Lowercase alpha-only tokens from a group name."""
    return re.findall(r"[a-z]+", (name or "").lower())


def is_drink_like_group(name: str) -> bool:
    """Return True when *name* describes a drink/beverage side group.

    Positive: "Can Drinks", "Burger Meal Drinks", "Drinks", "Hot Beverages"
    Negative: "Choose Side", "Family Wing Sauces", "Drink Modifications"
    """
    tokens = set(_tokenize(name))
    if tokens & DRINK_HARD_BLOCKLIST:
        return False
    return bool(tokens & DRINK_TOKENS)


def speech_noun_for_side_group(name: str) -> str:
    """Return "drink" for drink-like groups, "side" for everything else."""
    return "drink" if is_drink_like_group(name) else "side"


_MODIFIER_NOUN_MAP: dict[str, str] = {
    "sauce": "sauce",
    "sauces": "sauce",
    "topping": "topping",
    "toppings": "topping",
    "cheese": "cheese",
    "cheeses": "cheese",
    "protein": "protein",
    "proteins": "protein",
}


def speech_noun_for_modifier_group(name: str, prompt_noun: str | None = None) -> str:
    """Return a singular spoken noun for a modifier group.

    First checks group-name tokens against known noun mappings.
    Falls back to prompt_noun (from menu config) then "add-on".
    """
    tokens = _tokenize(name)
    for token in tokens:
        if token in _MODIFIER_NOUN_MAP:
            return _MODIFIER_NOUN_MAP[token]
    if prompt_noun and prompt_noun.strip():
        return prompt_noun.strip()
    return "add-on"


def ordinal_word(n: int) -> str:
    """Return the English ordinal word for 1-based position *n*.

    Falls back to "{n}th" for values beyond the precomputed list.
    """
    if 1 <= n <= len(_ORDINALS):
        return _ORDINALS[n - 1]
    return f"{n}th"
