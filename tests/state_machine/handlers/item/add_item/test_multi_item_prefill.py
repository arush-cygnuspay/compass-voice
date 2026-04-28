# tests/state_machine/handlers/item/add_item/test_multi_item_prefill.py
"""
Tests for multi-item prefill: every option phrase in a segment must be
resolved against ALL valid groups for that item before the FSM decides
which question to ask next.

Covers the cases from QUICK_FIX_CHECKLIST and the recent NLU log:

  a) "i want a chicken taco with coke steak and chicken
        and a chicken burger with american cheese red onions and fresh mushrooms"
  b) "a chicken taco with coke and steak jelly
        and a chicken burger with american cheese"
  c) "chicken taco with coke"  → Can Drinks must be satisfied, no re-prompt.
  d) Slot label noise: ITEM=coke inside the chicken-taco segment must still
     bind to Can Drinks → Coke (12 oz.).
  e) Boundary regression from production NLU log:
     "a chicken taco with coke and steak chicken cheese and jelly
        a chicken burger with red onions and fresh mushrooms"
     The bare "a" before "chicken burger" must split the utterance into
     two items.
"""
from __future__ import annotations

from pathlib import Path

from app.menu.repository import MenuRepository
from app.menu.store import MenuStore
from app.nlu.intent_resolution.intent import Intent
from app.nlu.nlu_result import SlotValue
from app.state_machine.handlers.item.add_item.add_item_handler import AddItemHandler
from app.state_machine.handlers.item.add_item.multi_group_prefill import (
    MultiGroupPrefillEngine,
)
from app.state_machine.handlers.item.add_item.pending_add_item_factory import (
    build_pending_add_item,
)
from app.state_machine.models.conversation_context import ConversationContext


# ─── Fixtures ────────────────────────────────────────────────────────────────
def _demo_repo() -> MenuRepository:
    data_root = (
        Path(__file__).resolve().parents[5]
        / "app"
        / "data"
        / "restaurants"
        / "demo"
    )
    store = MenuStore(
        menu_path=data_root / "menu.json",
        entity_index_path=data_root / "entity_index.json",
    )
    return MenuRepository(store)


def _resolve(repo: MenuRepository, name: str):
    """Tiny helper: pull a MenuItem out of the demo menu by exact name."""
    item_id = repo.store.find_item_ids_by_alias(name)[0]
    return repo.store.items[item_id]


def _slot(name: str, value: str, start: int | None = None, end: int | None = None) -> SlotValue:
    return SlotValue(
        name=name,
        value=value,
        raw=value,
        start=start,
        end=end,
        confidence=1.0,
    )


# ─── Engine-level tests (multi-group prefill in isolation) ───────────────────
class TestMultiGroupPrefillEngine:
    def test_coke_attaches_to_can_drinks_even_when_slot_is_item(self):
        """ITEM=coke inside a Chicken Taco segment must still bind to Can Drinks."""
        repo = _demo_repo()
        item = _resolve(repo, "chicken taco")
        pending = build_pending_add_item(item)
        engine = MultiGroupPrefillEngine()

        result = engine.prefill(
            pending=pending,
            segment_text="chicken taco with coke steak and chicken",
            slots=(
                _slot("ITEM", "chicken taco", 0, 12),
                # NLU mis-labels "coke" as ITEM — engine must not gate on label.
                _slot("ITEM", "coke", 18, 22),
            ),
        )

        # Coke (12 oz.) bound to Can Drinks side group.
        assert result.side_selections, result.debug
        bound_side_names = {
            choice.name
            for group in pending.side_groups
            for choice_id in result.side_selections.get(group.group_id, [])
            for choice in [group.choices_by_item_id[choice_id]]
        }
        assert any("Coke" in n for n in bound_side_names), bound_side_names

        # Steak + Chicken bound to Additional Meat for Plates.
        bound_modifier_names = {
            sel.name
            for sels in result.modifier_selections.values()
            for sel in sels
        }
        assert "Steak" in bound_modifier_names, bound_modifier_names
        assert "Chicken" in bound_modifier_names, bound_modifier_names

    def test_chicken_burger_segment_prefills_required_groups(self):
        """Chicken Burger segment carries American Cheese + Red Onions + Fresh Mushroom."""
        repo = _demo_repo()
        item = _resolve(repo, "chicken burger")
        pending = build_pending_add_item(item)
        engine = MultiGroupPrefillEngine()

        result = engine.prefill(
            pending=pending,
            segment_text="chicken burger with american cheese red onions and fresh mushroom",
            slots=(
                _slot("ITEM", "chicken burger", 0, 14),
                _slot("MODIFIER", "american cheese", 20, 35),
                _slot("MODIFIER", "red onions", 36, 46),
                _slot("MODIFIER", "fresh mushroom", 51, 65),
            ),
        )

        bound_side_names = {
            choice.name
            for group in pending.side_groups
            for choice_id in result.side_selections.get(group.group_id, [])
            for choice in [group.choices_by_item_id[choice_id]]
        }
        assert "American Cheese" in bound_side_names, bound_side_names

        bound_modifier_names = {
            sel.name
            for sels in result.modifier_selections.values()
            for sel in sels
        }
        assert "Red Onions" in bound_modifier_names, bound_modifier_names
        assert "Fresh Mushroom" in bound_modifier_names, bound_modifier_names

    def test_single_segment_coke_satisfies_can_drinks(self):
        """`chicken taco with coke` — Can Drinks satisfied; engine returns Coke."""
        repo = _demo_repo()
        item = _resolve(repo, "chicken taco")
        pending = build_pending_add_item(item)
        engine = MultiGroupPrefillEngine()

        result = engine.prefill(
            pending=pending,
            segment_text="chicken taco with coke",
            slots=(
                _slot("ITEM", "chicken taco", 0, 12),
                _slot("SIDE", "coke", 18, 22),
            ),
        )

        bound_side_names = {
            choice.name
            for group in pending.side_groups
            for choice_id in result.side_selections.get(group.group_id, [])
            for choice in [group.choices_by_item_id[choice_id]]
        }
        assert any("Coke" in n for n in bound_side_names), bound_side_names

    def test_unknown_phrase_leaves_unresolved(self):
        """Phrases that don't match any group on the item land in unresolved."""
        repo = _demo_repo()
        item = _resolve(repo, "chicken taco")
        pending = build_pending_add_item(item)
        engine = MultiGroupPrefillEngine()

        result = engine.prefill(
            pending=pending,
            segment_text="chicken taco with quokka and starfish",
            slots=(_slot("ITEM", "chicken taco", 0, 12),),
        )
        joined = " ".join(result.unresolved_phrases)
        assert "quokka" in joined or "starfish" in joined, result.unresolved_phrases

    def test_no_business_logic_in_engine_for_state_transitions(self):
        """The engine never touches ConversationContext directly."""
        repo = _demo_repo()
        item = _resolve(repo, "chicken taco")
        pending = build_pending_add_item(item)
        engine = MultiGroupPrefillEngine()

        result = engine.prefill(
            pending=pending,
            segment_text="chicken taco with coke",
            slots=(),
        )
        # The result is a pure data structure. No FSM/state mutation.
        assert isinstance(result.side_selections, dict)
        assert isinstance(result.modifier_selections, dict)


# ─── Handler-level tests (end to end) ────────────────────────────────────────
class TestAddItemHandlerMultiItemPrefill:
    """Exercise AddItemHandler.handle on real multi-item utterances."""

    def _handler(self) -> AddItemHandler:
        return AddItemHandler(menu_repo=_demo_repo())

    def test_chicken_taco_with_coke_steak_chicken_does_not_reask_can_drinks(self):
        """The original failing case — Can Drinks must NOT be re-asked."""
        handler = self._handler()
        ctx = ConversationContext()
        text = (
            "i want a chicken taco with coke steak and chicken "
            "and a chicken burger with american cheese red onions "
            "and fresh mushroom"
        )
        # Approximation of the slot set the production NLU emits for this
        # utterance (item slots only; the menu store discovers "coke").
        slots = (
            _slot("ITEM", "chicken taco", 9, 21),
            _slot("ITEM", "chicken burger", 56, 70),
            _slot("MODIFIER", "american cheese", 76, 91),
            _slot("MODIFIER", "red onions", 92, 102),
            _slot("MODIFIER", "fresh mushroom", 107, 121),
        )
        ctx.last_slots = slots
        ctx.last_user_text = text

        result = handler.handle(
            intent=Intent.ADD_ITEM,
            context=ctx,
            user_text=text,
            session=None,
        )

        # We are NOT in WAITING_FOR_SIDE for "Can Drinks" — Coke was prefilled.
        payload = result.response_payload or {}
        if result.response_key == "ask_for_side":
            assert "Can Drink" not in (payload.get("group_name") or ""), payload
        # Chicken Taco context now has Coke selected.
        pending = ctx.pending_add_item
        assert pending is not None
        coke_in_sides = False
        for group in pending.side_groups:
            for sid in ctx.selected_side_groups.get(group.group_id, []):
                choice = group.choices_by_item_id.get(sid)
                if choice and "Coke" in choice.name:
                    coke_in_sides = True
        assert coke_in_sides, ctx.selected_side_groups

        # Both Steak and Chicken are in Additional Meat for Plates.
        modifier_names = {
            sel.name
            for sels in ctx.selected_modifier_groups.values()
            for sel in sels
        }
        assert "Steak" in modifier_names, modifier_names
        assert "Chicken" in modifier_names, modifier_names

        # Chicken Burger is queued, not lost.
        queued_names = [q.item_slot_value for q in ctx.pending_item_queue]
        assert any("burger" in (n or "").lower() for n in queued_names), queued_names

    def test_chicken_taco_with_jelly_then_bare_a_chicken_burger_splits(self):
        """Production-NLU regression: bare 'a' (no 'and') still splits items."""
        handler = self._handler()
        ctx = ConversationContext()
        text = (
            "a chicken taco with coke and steak chicken cheese and jelly "
            "a chicken burger with red onions and fresh mushroom"
        )
        slots = (
            _slot("ITEM", "chicken taco", 2, 14),
            _slot("SIDE", "coke", 20, 24),
            _slot("MODIFIER", "steak chicken cheese", 29, 49),
            _slot("MODIFIER", "jelly", 54, 59),
            _slot("ITEM", "chicken burger", 62, 76),
            _slot("SIDE", "red onions", 82, 92),
            _slot("MODIFIER", "fresh mushroom", 97, 111),
        )
        ctx.last_slots = slots
        ctx.last_user_text = text

        result = handler.handle(
            intent=Intent.ADD_ITEM,
            context=ctx,
            user_text=text,
            session=None,
        )

        # Chicken Burger MUST be queued — i.e. parser produced 2 segments.
        queued = [q.item_slot_value for q in ctx.pending_item_queue]
        assert any("burger" in (n or "").lower() for n in queued), (
            f"chicken burger was not queued; got {queued!r}"
        )

        # Chicken Taco's Jelly stays attached to Chicken Taco (not orphaned
        # into the burger segment).
        pending = ctx.pending_add_item
        assert pending is not None
        modifier_names = {
            sel.name
            for sels in ctx.selected_modifier_groups.values()
            for sel in sels
        }
        assert "Jelly" in modifier_names, modifier_names
        assert "Cheese" in modifier_names, modifier_names

        # And the unresolved-feedback should not blame the Chicken Taco for
        # red onions / fresh mushroom — those belong to the burger segment.
        feedback = (result.response_payload or {}).get("prefill_feedback", "")
        assert "red onion" not in feedback.lower(), feedback
        assert "fresh mushroom" not in feedback.lower(), feedback

    def test_simple_chicken_taco_with_coke_does_not_reask_drinks(self):
        """Smallest case — Can Drinks satisfied, FSM moves past it."""
        handler = self._handler()
        ctx = ConversationContext()
        text = "a chicken taco with coke"
        slots = (
            _slot("ITEM", "chicken taco", 2, 14),
            _slot("SIDE", "coke", 20, 24),
        )
        ctx.last_slots = slots
        ctx.last_user_text = text

        result = handler.handle(
            intent=Intent.ADD_ITEM,
            context=ctx,
            user_text=text,
            session=None,
        )
        payload = result.response_payload or {}
        if result.response_key == "ask_for_side":
            assert "Can Drink" not in (payload.get("group_name") or "")

        coke_bound = any(
            "Coke" in choice.name
            for group in ctx.pending_add_item.side_groups
            for sid in ctx.selected_side_groups.get(group.group_id, [])
            for choice in [group.choices_by_item_id[sid]]
        )
        assert coke_bound, ctx.selected_side_groups

    def test_prefill_debug_payload_carries_engine_diagnostics(self):
        """Debug payload must expose the keys the brief asked for."""
        handler = self._handler()
        ctx = ConversationContext()
        text = "a chicken taco with coke and steak"
        slots = (
            _slot("ITEM", "chicken taco", 2, 14),
            _slot("SIDE", "coke", 20, 24),
            _slot("MODIFIER", "steak", 29, 34),
        )
        ctx.last_slots = slots
        ctx.last_user_text = text

        result = handler.handle(
            intent=Intent.ADD_ITEM,
            context=ctx,
            user_text=text,
            session=None,
        )
        debug = (result.response_payload or {}).get("prefill_debug") or {}
        for key in (
            "segment_text",
            "candidate_phrases",
            "resolved_group_values",
            "missing_groups_after_prefill",
            "skipped_groups_because_prefilled",
            "bindings",
        ):
            assert key in debug, f"missing debug key: {key} (got {list(debug)})"


# ─── Parser regression: bare 'a' between items should split ─────────────────
class TestMultiItemParserBareQuantitySplit:
    def test_bare_a_chicken_burger_after_modifier_list_splits(self):
        from app.nlu.multi_item_parser import parse_multi_item_utterance

        repo = _demo_repo()
        text = (
            "a chicken taco with coke and steak chicken cheese and jelly "
            "a chicken burger with red onions and fresh mushroom"
        )
        slots = (
            _slot("ITEM", "chicken taco", 2, 14),
            _slot("ITEM", "chicken burger", 62, 76),
        )
        segments = parse_multi_item_utterance(text, slots, menu_store=repo.store)
        assert len(segments) == 2, [s.raw_text for s in segments]
        assert "chicken taco" in segments[0].raw_text
        assert "chicken burger" in segments[1].raw_text
        # And the trailing modifiers ("...and jelly") stay on segment 1.
        assert "jelly" in segments[0].raw_text, segments[0].raw_text
        # While red onions / fresh mushroom belong to segment 2.
        assert "red onion" in segments[1].raw_text.lower()
        assert "fresh mushroom" in segments[1].raw_text.lower()

    def test_with_in_segment_does_not_attach_next_item(self):
        """The historical bug: ' with ' anywhere in lookback over-attached."""
        from app.nlu.multi_item_parser import _slot_looks_attached

        # "...taco with coke and a chicken burger" — "with" is far upstream.
        text = "a chicken taco with coke and a chicken burger"
        slot = _slot("ITEM", "chicken burger", 31, 45)
        # The historical heuristic returned True here. The fix returns False.
        assert _slot_looks_attached(text, slot) is False
