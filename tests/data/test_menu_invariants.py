# tests/data/test_menu_invariants.py
"""Menu data invariant tests for PR-1 corrections.

Validates:
- Sandwich condiments are optional on non-combo sandwiches
- Crabcake Combo has both required Choose Side and Choose Drink
- Side of Tenderloin has no required Breakfast Special Sides
- Wing sauce groups carry prompt_noun='sauce'
- Typo-renamed items preserve old aliases and item_ids
"""
from __future__ import annotations

import pytest

from tests.support.voice_test_harness import build_menu_repo

# ─────────────────────────────────────────────────────────────────
# Session-scoped fixture (mirrors conftest.py pattern)
# ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def store():
    return build_menu_repo().store


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _modifier_groups_named(item, fragment: str):
    return [g for g in item.modifier_groups if fragment.lower() in g.name.lower()]


def _side_groups_named(item, fragment: str):
    return [g for g in item.side_groups if fragment.lower() in g.name.lower()]


# ─────────────────────────────────────────────────────────────────
# 1. Sandwich condiments optional on non-combo sandwiches
# ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("item_id,label", [
    ("bc32db05-38d5-45fe-a3f4-3d74ce9bb56a", "Steak & Cheese Sub"),
    ("f2f320ce-df79-42c1-9ff6-1990d5aac025", "Grilled Chicken Sandwich"),
    ("ea7f7a98-8f5b-42f8-a9d7-5fda57230dd5", "Jumbo Lump Crabck Sand"),
])
def test_no_required_condiments_on_non_combo_sandwiches(store, item_id, label):
    item = store.items[item_id]
    groups = _modifier_groups_named(item, "sandwich condiments")
    assert groups, f"{label}: expected a Sandwich Condiments modifier_group"
    for g in groups:
        assert not g.is_required, f"{label}: Sandwich Condiments must not be required"
        assert g.min_selector == 0, f"{label}: Sandwich Condiments min_selector must be 0, got {g.min_selector}"


# ─────────────────────────────────────────────────────────────────
# 2. Crabcake Combo has required Choose Side AND Choose Drink
# ─────────────────────────────────────────────────────────────────

CRABCAKE_COMBO_ID = "89daa1f7-99b1-455d-8efa-f50df9528db7"


def test_crabcake_combo_has_required_choose_side_and_choose_drink(store):
    item = store.items[CRABCAKE_COMBO_ID]
    side_groups_by_name = {g.name: g for g in item.side_groups}

    assert "Choose Drink" in side_groups_by_name, "Crabcake Combo missing Choose Drink side_group"
    assert "Choose Side" in side_groups_by_name, "Crabcake Combo missing Choose Side side_group"

    drink = side_groups_by_name["Choose Drink"]
    assert drink.is_required, "Choose Drink must be required"
    assert drink.min_selector == 1, f"Choose Drink min_selector expected 1, got {drink.min_selector}"
    assert drink.max_selector == 1, f"Choose Drink max_selector expected 1, got {drink.max_selector}"

    side = side_groups_by_name["Choose Side"]
    assert side.is_required, "Choose Side must be required"
    assert side.min_selector == 1, f"Choose Side min_selector expected 1, got {side.min_selector}"
    assert side.max_selector == 1, f"Choose Side max_selector expected 1, got {side.max_selector}"


# ─────────────────────────────────────────────────────────────────
# 3. Side of Tenderloin has no required Breakfast Special Sides
# ─────────────────────────────────────────────────────────────────

TENDERLOIN_ID = "fa26abcd-2a74-43c6-bb06-ac5f651a7849"


def test_side_of_tenderloin_has_no_required_breakfast_special_sides(store):
    item = store.items[TENDERLOIN_ID]
    blocking = [
        g for g in item.side_groups
        if "breakfast special sides" in g.name.lower() and g.is_required
    ]
    assert not blocking, (
        f"Side of Tenderloin must not have a required Breakfast Special Sides group; "
        f"found: {[g.name for g in blocking]}"
    )


# ─────────────────────────────────────────────────────────────────
# 4. Wing sauce groups carry prompt_noun = "sauce"
# ─────────────────────────────────────────────────────────────────

SAUCE_GROUP_NAMES = {"family wing sauces", "wing sauce", "wing flavors"}


def test_wing_sauce_groups_speak_sauce_noun(store):
    missing = []
    for item in store.items.values():
        for g in list(item.side_groups) + list(item.modifier_groups):
            if g.name.lower() in SAUCE_GROUP_NAMES:
                noun = getattr(g, "prompt_noun", None)
                if noun != "sauce":
                    missing.append(f"{item.name} / {g.name}: prompt_noun={noun!r}")

    assert not missing, (
        "These wing sauce groups are missing prompt_noun='sauce':\n  " + "\n  ".join(missing)
    )


# ─────────────────────────────────────────────────────────────────
# 5. Renamed items still resolve from old alias / entity_index key
# ─────────────────────────────────────────────────────────────────

TYPO_RENAMES = [
    # (item_id, new_canonical_name, old_typo_alias, entity_index_old_key)
    (
        "170ba7e6-26b4-4fee-9ea5-159ad78e9800",
        "24 Bone in Wings",
        "24 bone in wigns",
        "24 bone in wings",        # new canonical entity_index key added in PR-1
    ),
    (
        "d83a28ea-450c-4bd5-829e-b5d9cb2dc973",
        "Jumbo Lump Crabcake Platter",
        "jumbo lump crabcake platte",
        "jumbo lump crabcake platter",
    ),
    (
        "9175332e-cc89-4eec-af69-581103ec6840",
        "Mandarin Orange Salad",
        "mandirin orange salad",
        "mandarin orange salad",
    ),
    (
        "6130eb1e-3945-41a1-a6ab-95f71757e275",
        "Crab Ball Snack",
        "crab ball sanck",
        "crab ball snack",
    ),
    (
        "998b1962-b495-4cdb-8bae-4e4912039f2f",
        "Sriracha",
        "siriracha",
        "sriracha",
    ),
]


@pytest.mark.parametrize("item_id,new_name,old_alias,ei_key", TYPO_RENAMES)
def test_renamed_items_resolve_from_old_alias(store, item_id, new_name, old_alias, ei_key):
    item = store.items[item_id]
    assert item.name == new_name, f"Expected name {new_name!r}, got {item.name!r}"

    assert old_alias in item.aliases, (
        f"{new_name}: old alias {old_alias!r} must be preserved in aliases; "
        f"got {list(item.aliases)}"
    )

    entries = store.entity_index.get(ei_key, [])
    assert entries, f"entity_index key {ei_key!r} not found"
    matched_ids = [e["item_id"] for e in entries if e.get("type") == "item"]
    assert item_id in matched_ids, (
        f"entity_index[{ei_key!r}] does not resolve to {item_id}; got {matched_ids}"
    )


# ─────────────────────────────────────────────────────────────────
# 6. Typo-renamed items retain their original item_ids
# ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("item_id,new_name,old_alias,ei_key", TYPO_RENAMES)
def test_item_ids_preserved_across_typo_renames(store, item_id, new_name, old_alias, ei_key):
    item = store.items.get(item_id)
    assert item is not None, f"item_id {item_id} not found in store after rename to {new_name!r}"
    assert item.item_id == item_id, (
        f"{new_name}: item_id changed; expected {item_id}, got {item.item_id}"
    )
    # No duplicate entries: only one item should resolve from the new canonical key
    entries = store.entity_index.get(ei_key, [])
    item_entries = [e for e in entries if e.get("type") == "item"]
    ids = [e["item_id"] for e in item_entries]
    assert ids.count(item_id) == 1, (
        f"entity_index[{ei_key!r}] has duplicate item_id entries: {ids}"
    )
