# tests/state_machine/handlers/item/add_item/test_side_modifier_fuzzy_consumption.py
"""Tests that consumed_phrases suppression applies equally to side/modifier candidates.

The fix in MultiGroupPrefillEngine._build_candidate_phrases uses consumed_phrases to
suppress candidates in both step 2a (slot values) and step 2c (connector splits).
This file verifies parity: a consumed item raw query does not resurface as an
unresolved side or modifier candidate.
"""
from __future__ import annotations

from app.menu.models import MenuItem, ModifierChoice, ModifierGroup, Pricing, SideChoice, SideGroup
from app.nlu.nlu_result import SlotValue
from app.nlu.query_normalization.text_preprocessor import normalize_text
from app.state_machine.handlers.item.add_item.multi_group_prefill import MultiGroupPrefillEngine
from app.state_machine.handlers.item.add_item.pending_add_item_factory import build_pending_add_item


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _item_with_side_and_modifier(
    item_name: str = "Burger",
    side_choices: list[tuple[str, str]] | None = None,
    mod_choices: list[tuple[str, str]] | None = None,
) -> MenuItem:
    side_choices = side_choices or [("coke", "Coke"), ("sprite", "Sprite")]
    mod_choices = mod_choices or [("cheese", "Cheese"), ("bacon", "Bacon")]
    return MenuItem(
        item_id="burger_1",
        name=item_name,
        normalized_name=normalize_text(item_name),
        aliases=(item_name.lower(),),
        normalized_aliases=(normalize_text(item_name),),
        voice_labels=(item_name.lower(),),
        pricing=Pricing(mode="fixed", price_cents=700),
        side_groups=[
            SideGroup(
                group_id="drinks",
                name="Choose a drink",
                normalized_name="choose a drink",
                is_required=False,
                min_selector=0,
                max_selector=1,
                choices=[
                    SideChoice(
                        item_id=sid,
                        name=sname,
                        normalized_name=normalize_text(sname),
                        pricing=Pricing(mode="fixed", price_cents=0),
                    )
                    for sid, sname in side_choices
                ],
            )
        ],
        modifier_groups=[
            ModifierGroup(
                group_id="mods",
                name="Add-ons",
                normalized_name="add-ons",
                is_required=False,
                min_selector=0,
                max_selector=1,
                choices=[
                    ModifierChoice(
                        modifier_id=mid,
                        name=mname,
                        normalized_name=normalize_text(mname),
                        price_cents=0,
                    )
                    for mid, mname in mod_choices
                ],
            )
        ],
        available=True,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSideModifierFuzzyConsumption:
    def test_consumed_item_slot_value_not_in_unresolved_with_side_group(self) -> None:
        """When raw item query is consumed, it must not appear in unresolved phrases
        even when there is a side group on the item."""
        item = _item_with_side_and_modifier("Pot Stickers")
        pending = build_pending_add_item(item)
        engine = MultiGroupPrefillEngine()

        result = engine.prefill(
            pending=pending,
            segment_text="port stickers with coke",
            slots=(
                SlotValue(name="ITEM", value="port stickers"),
                SlotValue(name="ITEM", value="Coke"),
            ),
            consumed_tokens=frozenset({"port", "stickers", "pot"}),
            consumed_phrases=frozenset({"port stickers"}),
        )

        for phrase in result.unresolved_phrases:
            assert "port" not in phrase.lower(), (
                f"'port stickers' leaked into unresolved: {result.unresolved_phrases}"
            )

    def test_legitimate_side_still_binds_when_item_query_consumed(self) -> None:
        """With consumed item query, genuine side choices must still bind correctly."""
        item = _item_with_side_and_modifier("Pot Stickers")
        pending = build_pending_add_item(item)
        engine = MultiGroupPrefillEngine()

        result = engine.prefill(
            pending=pending,
            segment_text="port stickers with coke",
            slots=(
                SlotValue(name="ITEM", value="port stickers"),
                SlotValue(name="ITEM", value="Coke"),
            ),
            consumed_tokens=frozenset({"port", "stickers", "pot"}),
            consumed_phrases=frozenset({"port stickers"}),
        )

        assert "drinks" in result.side_selections, (
            f"Coke side binding was lost; side_selections={result.side_selections!r}"
        )
        assert "coke" in result.side_selections.get("drinks", []), (
            f"Expected 'coke' in drinks; side_selections={result.side_selections!r}"
        )

    def test_consumed_item_slot_not_matched_as_modifier(self) -> None:
        """Consumed item raw query must not be attempted as a modifier candidate."""
        item = _item_with_side_and_modifier(
            "Pot Stickers",
            mod_choices=[("port", "Port"), ("stickers", "Stickers")],
        )
        pending = build_pending_add_item(item)
        engine = MultiGroupPrefillEngine()

        result = engine.prefill(
            pending=pending,
            segment_text="port stickers",
            slots=(SlotValue(name="ITEM", value="port stickers"),),
            consumed_tokens=frozenset({"port", "stickers", "pot"}),
            consumed_phrases=frozenset({"port stickers"}),
        )

        mods = result.modifier_selections.get("mods", [])
        mod_ids = [m.modifier_id for m in mods]
        assert "port" not in mod_ids and "stickers" not in mod_ids, (
            f"Consumed item tokens bound as modifiers: {mod_ids}"
        )
