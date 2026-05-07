# tests/state_machine/handlers/item/add_item/test_pending_add_item_sort.py
"""Phase 2 — pending side-group sort: drinks last, stable within each bucket."""
from __future__ import annotations

import types

import pytest

from app.state_machine.handlers.item.add_item.pending_add_item_factory import build_pending_add_item


def _fake_pricing(mode="fixed"):
    return types.SimpleNamespace(mode=mode, variants=[], price_cents=0)


def _fake_choice(item_id, name):
    return types.SimpleNamespace(
        item_id=item_id,
        name=name,
        pricing=_fake_pricing(),
        aliases=[],
        voice_labels=[],
    )


def _fake_side_group(group_id, name, is_required=False, choices=None):
    choices = choices or [_fake_choice(f"{group_id}_c1", "Option A")]
    return types.SimpleNamespace(
        group_id=group_id,
        name=name,
        is_required=is_required,
        min_selector=1,
        max_selector=1,
        choices=choices,
        prompt_noun=None,
        prompt_verb=None,
    )


def _fake_item(side_groups):
    return types.SimpleNamespace(
        item_id="test_item",
        name="Test Item",
        aliases=[],
        voice_labels=[],
        pricing=_fake_pricing(),
        side_groups=side_groups,
        modifier_groups=[],
    )


def _group_names(pending) -> list[str]:
    return [g.name for g in pending.side_groups]


class TestSideGroupSort:
    def test_drink_first_menu_order_becomes_food_first(self):
        """[Choose Drink, Choose Side] → [Choose Side, Choose Drink]."""
        item = _fake_item([
            _fake_side_group("drink_g", "Choose Drink"),
            _fake_side_group("side_g", "Choose Side"),
        ])
        pending = build_pending_add_item(item)
        assert _group_names(pending) == ["Choose Side", "Choose Drink"]

    def test_drink_a_side_drink_b_sorts_to_side_drink_a_drink_b(self):
        """[Drink A, Side, Drink B] → [Side, Drink A, Drink B] — relative order preserved."""
        item = _fake_item([
            _fake_side_group("da", "Family Meal Drinks"),
            _fake_side_group("side", "Choose Side"),
            _fake_side_group("db", "Can Drinks"),
        ])
        pending = build_pending_add_item(item)
        assert _group_names(pending) == ["Choose Side", "Family Meal Drinks", "Can Drinks"]

    def test_only_drink_unchanged(self):
        """[Choose Drink] remains [Choose Drink]."""
        item = _fake_item([_fake_side_group("d", "Choose Drink")])
        pending = build_pending_add_item(item)
        assert _group_names(pending) == ["Choose Drink"]

    def test_no_drink_unchanged(self):
        """[Side, Wing Sauce, Side2] remains unchanged (all non-drink)."""
        item = _fake_item([
            _fake_side_group("s1", "Choose Side"),
            _fake_side_group("ws", "Wing Sauce"),
            _fake_side_group("s2", "Platter Sides"),
        ])
        pending = build_pending_add_item(item)
        assert _group_names(pending) == ["Choose Side", "Wing Sauce", "Platter Sides"]

    def test_group_ids_unchanged_after_sort(self):
        """Sorting must not alter group_id values."""
        item = _fake_item([
            _fake_side_group("drink_id", "Can Drinks"),
            _fake_side_group("side_id", "Choose Side"),
        ])
        pending = build_pending_add_item(item)
        ids = [g.group_id for g in pending.side_groups]
        assert "drink_id" in ids
        assert "side_id" in ids
        assert ids.index("side_id") < ids.index("drink_id")

    def test_side_groups_by_id_still_keyed_correctly(self):
        """side_groups_by_id lookup must work after sorting."""
        item = _fake_item([
            _fake_side_group("drink_id", "Drinks"),
            _fake_side_group("side_id", "Choose Side"),
        ])
        pending = build_pending_add_item(item)
        assert "drink_id" in pending.side_groups_by_id
        assert "side_id" in pending.side_groups_by_id

    def test_required_semantics_preserved(self):
        """Required/optional flags survive sorting."""
        item = _fake_item([
            _fake_side_group("d", "Drinks", is_required=True),
            _fake_side_group("s", "Choose Side", is_required=False),
        ])
        pending = build_pending_add_item(item)
        by_id = pending.side_groups_by_id
        assert by_id["d"].is_required is True
        assert by_id["s"].is_required is False

    def test_top_choice_names_capped_at_six(self):
        """Phase 3: top_choice_names must hold up to 6 names."""
        choices = [_fake_choice(f"c{i}", f"Choice {i}") for i in range(10)]
        item = _fake_item([_fake_side_group("s", "Choose Side", choices=choices)])
        pending = build_pending_add_item(item)
        assert len(pending.side_groups[0].top_choice_names) == 6
