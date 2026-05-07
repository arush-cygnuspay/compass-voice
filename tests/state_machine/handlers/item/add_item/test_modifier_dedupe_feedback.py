# tests/state_machine/handlers/item/add_item/test_modifier_dedupe_feedback.py
"""Phase F — duplicate modifier detection and repeat feedback."""
from __future__ import annotations

import types
from unittest.mock import MagicMock

import pytest

from app.state_machine.handlers.item.add_item.modifier_group_resolver import (
    ModifierGroupResolver,
    ModifierGroupMatch,
)
from app.state_machine.models.pending_item_models import (
    PendingModifierChoice,
    PendingModifierGroup,
)
from app.responses.item.modifiers import repeat_modifier_options


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_group(choices: list[PendingModifierChoice], min_sel: int = 1) -> PendingModifierGroup:
    return PendingModifierGroup(
        group_id="grp",
        name="Choose Sauce",
        is_required=min_sel > 0,
        min_selector=min_sel,
        max_selector=1,
        choices=choices,
        choices_by_modifier_id={c.modifier_id: c for c in choices},
        choices_by_normalized_name={c.normalized_name: [c] for c in choices},
        choice_names=tuple(c.name for c in choices),
        normalized_choice_names=tuple(c.normalized_name for c in choices),
        top_choice_names=tuple(c.name for c in choices),
    )


def _choice(modifier_id: str, name: str) -> PendingModifierChoice:
    return PendingModifierChoice(
        modifier_id=modifier_id,
        name=name,
        group_id="grp",
        normalized_name=name.lower(),
    )


# ---------------------------------------------------------------------------
# ModifierGroupResolver — duplicate_names capture
# ---------------------------------------------------------------------------

class TestModifierGroupResolverDuplicates:
    def test_duplicate_name_recorded_when_already_selected(self):
        ranch = _choice("ranch", "Ranch")
        bbq = _choice("bbq", "BBQ")
        group = _make_group([ranch, bbq])

        result = ModifierGroupResolver().resolve(
            group=group,
            normalized_user_text="ranch",
            normalized_slot_values=["ranch"],
            already_selected_ids=["ranch"],
        )

        assert result.duplicate_names == ["Ranch"]
        assert result.selections == []

    def test_no_duplicate_when_not_already_selected(self):
        ranch = _choice("ranch", "Ranch")
        bbq = _choice("bbq", "BBQ")
        group = _make_group([ranch, bbq])

        result = ModifierGroupResolver().resolve(
            group=group,
            normalized_user_text="ranch",
            normalized_slot_values=["ranch"],
            already_selected_ids=[],
        )

        assert result.duplicate_names == []
        assert len(result.selections) == 1
        assert result.selections[0].modifier_id == "ranch"

    def test_duplicate_names_list_deduped(self):
        """Saying the same modifier twice should only record one duplicate entry."""
        ranch = _choice("ranch", "Ranch")
        group = _make_group([ranch])

        result = ModifierGroupResolver().resolve(
            group=group,
            normalized_user_text="ranch ranch",
            normalized_slot_values=["ranch", "ranch"],
            already_selected_ids=["ranch"],
        )

        assert result.duplicate_names == ["Ranch"]

    def test_non_duplicate_selection_clears_duplicate_names(self):
        """A valid new selection alongside a duplicate should still work."""
        ranch = _choice("ranch", "Ranch")
        bbq = _choice("bbq", "BBQ")
        group = _make_group([ranch, bbq], min_sel=0)

        # ranch already selected, user says "bbq"
        result = ModifierGroupResolver().resolve(
            group=group,
            normalized_user_text="bbq",
            normalized_slot_values=["bbq"],
            already_selected_ids=["ranch"],
        )

        assert result.duplicate_names == []
        assert len(result.selections) == 1
        assert result.selections[0].modifier_id == "bbq"


# ---------------------------------------------------------------------------
# ModifierGroupMatch — dataclass field defaults
# ---------------------------------------------------------------------------

class TestModifierGroupMatchDefaults:
    def test_duplicate_names_defaults_to_empty_list(self):
        match = ModifierGroupMatch(selections=[], unmatched_values=[])
        assert match.duplicate_names == []

    def test_duplicate_names_stored_correctly(self):
        match = ModifierGroupMatch(
            selections=[],
            unmatched_values=[],
            duplicate_names=["Ranch", "BBQ"],
        )
        assert match.duplicate_names == ["Ranch", "BBQ"]


# ---------------------------------------------------------------------------
# repeat_modifier_options — duplicate feedback wording
# ---------------------------------------------------------------------------

class TestRepeatModifierOptionsDuplicateFeedback:
    def _ctx(self):
        return types.SimpleNamespace(
            current_item_id="item_1",
            current_modifier_group_index=0,
            selected_modifier_groups={},
        )

    def _repo(self):
        choice = types.SimpleNamespace(modifier_id="ranch", name="Ranch")
        group = types.SimpleNamespace(
            group_id="grp", name="Choose Sauce",
            choices=[choice], min_selector=1, max_selector=1,
        )
        item = types.SimpleNamespace(modifier_groups=[group])
        return types.SimpleNamespace(store=types.SimpleNamespace(get_item=lambda _: item))

    def test_duplicate_single_name_feedback(self):
        ctx = self._ctx()
        repo = self._repo()
        payload = {
            "repeat_reason": "duplicate",
            "duplicate_names": ["Ranch"],
            "speech_noun": "sauce",
        }
        result = repeat_modifier_options(ctx, repo, payload)
        assert "Ranch" in result
        assert "already" in result.lower()
        assert "sauce" in result

    def test_duplicate_multiple_names_feedback(self):
        ctx = self._ctx()
        repo = self._repo()
        payload = {
            "repeat_reason": "duplicate",
            "duplicate_names": ["Ranch", "BBQ"],
            "speech_noun": "sauce",
        }
        result = repeat_modifier_options(ctx, repo, payload)
        assert "Ranch" in result
        assert "BBQ" in result
        assert "sauce" in result

    def test_duplicate_no_names_fallback(self):
        ctx = self._ctx()
        repo = self._repo()
        payload = {
            "repeat_reason": "duplicate",
            "duplicate_names": [],
            "speech_noun": "topping",
        }
        result = repeat_modifier_options(ctx, repo, payload)
        assert "topping" in result
        assert "already" in result.lower() or "selected" in result.lower()

    def test_duplicate_reason_not_confused_with_invalid(self):
        ctx = self._ctx()
        repo = self._repo()
        payload = {
            "repeat_reason": "duplicate",
            "duplicate_names": ["Cheese"],
            "speech_noun": "cheese",
        }
        result = repeat_modifier_options(ctx, repo, payload)
        # Should not say "I didn't catch" or use the invalid-input path
        assert "didn't catch" not in result.lower()
        assert "I already have Cheese" in result

    def test_non_duplicate_reason_unaffected(self):
        ctx = self._ctx()
        repo = self._repo()
        payload = {
            "repeat_reason": "invalid",
            "reprompt_count": 0,
        }
        result = repeat_modifier_options(ctx, repo, payload)
        assert "already" not in result.lower() or "I didn" in result
