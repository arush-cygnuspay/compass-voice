# tests/state_machine/models/test_conversation_context_serde_duplicates.py
"""Verify that duplicate side IDs survive ConversationContext to_dict/from_dict round-trips.

selected_side_groups stores list[str] per group where repeated item_ids represent multiple
selections (e.g. ["coke","coke"] = two Cokes selected). The list must survive unchanged —
no deduplication, no sorting.
"""
from app.state_machine.models.conversation_context import ConversationContext
from app.state_machine.models.conversation_context_serde import (
    _pending_side_group_to_dict,
    _pending_side_group_from_dict,
)
from app.state_machine.models.pending_item_models import PendingSideGroup


# ---------------------------------------------------------------------------
# ConversationContext.selected_side_groups round-trip
# ---------------------------------------------------------------------------

class TestSelectedSideGroupsRoundTrip:
    def _ctx_with_sides(self, sides: dict) -> ConversationContext:
        ctx = ConversationContext()
        ctx.selected_side_groups = sides
        return ctx

    def test_single_id_unchanged(self):
        ctx = self._ctx_with_sides({"drinks": ["coke"]})
        restored = ConversationContext.from_dict(ctx.to_dict())
        assert restored.selected_side_groups == {"drinks": ["coke"]}

    def test_two_duplicate_ids_preserved(self):
        ctx = self._ctx_with_sides({"drinks": ["coke", "coke"]})
        restored = ConversationContext.from_dict(ctx.to_dict())
        assert restored.selected_side_groups["drinks"] == ["coke", "coke"]

    def test_three_duplicate_ids_preserved(self):
        ctx = self._ctx_with_sides({"drinks": ["coke", "coke", "coke"]})
        restored = ConversationContext.from_dict(ctx.to_dict())
        assert restored.selected_side_groups["drinks"] == ["coke", "coke", "coke"]

    def test_mixed_ids_order_preserved(self):
        ctx = self._ctx_with_sides({"drinks": ["coke", "sprite", "coke"]})
        restored = ConversationContext.from_dict(ctx.to_dict())
        assert restored.selected_side_groups["drinks"] == ["coke", "sprite", "coke"]

    def test_multiple_groups_each_preserved(self):
        ctx = self._ctx_with_sides({
            "drinks": ["coke", "coke"],
            "sauces": ["ranch", "ranch", "bbq"],
        })
        restored = ConversationContext.from_dict(ctx.to_dict())
        assert restored.selected_side_groups["drinks"] == ["coke", "coke"]
        assert restored.selected_side_groups["sauces"] == ["ranch", "ranch", "bbq"]

    def test_empty_list_preserved(self):
        ctx = self._ctx_with_sides({"drinks": []})
        restored = ConversationContext.from_dict(ctx.to_dict())
        assert restored.selected_side_groups["drinks"] == []

    def test_empty_dict_preserved(self):
        ctx = self._ctx_with_sides({})
        restored = ConversationContext.from_dict(ctx.to_dict())
        assert restored.selected_side_groups == {}


# ---------------------------------------------------------------------------
# PendingSideGroup serde — allow_duplicate_selections field
# ---------------------------------------------------------------------------

def _make_pending_side_group(*, allow_duplicate_selections: bool) -> PendingSideGroup:
    return PendingSideGroup(
        group_id="g1",
        name="Drinks",
        is_required=False,
        min_selector=0,
        max_selector=3,
        choices=[],
        choices_by_item_id={},
        choices_by_normalized_name={},
        choice_names=(),
        normalized_choice_names=(),
        top_choice_names=(),
        allow_duplicate_selections=allow_duplicate_selections,
    )


class TestPendingSideGroupSerdeAllowDuplicates:
    def test_true_survives_round_trip(self):
        group = _make_pending_side_group(allow_duplicate_selections=True)
        d = _pending_side_group_to_dict(group)
        restored = _pending_side_group_from_dict(d)
        assert restored.allow_duplicate_selections is True

    def test_false_survives_round_trip(self):
        group = _make_pending_side_group(allow_duplicate_selections=False)
        d = _pending_side_group_to_dict(group)
        restored = _pending_side_group_from_dict(d)
        assert restored.allow_duplicate_selections is False

    def test_missing_key_defaults_to_true(self):
        """Older serialized contexts without the field should default to True."""
        d = {
            "group_id": "g1",
            "name": "Drinks",
            "is_required": False,
            "min_selector": 0,
            "max_selector": 3,
            "choices": [],
            # allow_duplicate_selections intentionally absent
        }
        restored = _pending_side_group_from_dict(d)
        assert restored.allow_duplicate_selections is True
