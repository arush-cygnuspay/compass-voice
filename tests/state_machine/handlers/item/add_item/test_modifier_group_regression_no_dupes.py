# tests/state_machine/handlers/item/add_item/test_modifier_group_regression_no_dupes.py
"""Regression guard: modifier groups must NOT allow duplicate selections.

Unlike side groups (allow_duplicate_selections=True by default), modifier groups
always deduplicate. Saying "Tomato Sauce and Tomato Sauce" should produce exactly
one Tomato Sauce selection, and the duplicate_names list should record the repeat.

This file is a Phase 11 guard against accidental drift.
"""
from types import SimpleNamespace

from app.state_machine.handlers.item.add_item.modifier_group_resolver import ModifierGroupResolver
from app.state_machine.handlers.item.add_item.option_matching import OptionCandidate
from app.nlu.query_normalization.text_preprocessor import normalize_text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_choice(modifier_id: str, name: str):
    return SimpleNamespace(
        modifier_id=modifier_id,
        name=name,
        normalized_name=normalize_text(name),
        match_texts=(normalize_text(name),),
    )


def _make_group(choices):
    return SimpleNamespace(
        group_id="sauces",
        name="Sauces",
        choices=choices,
        is_required=False,
        min_selector=0,
        max_selector=2,
    )


def _resolver():
    return ModifierGroupResolver()


def _candidates(*texts, source="slot_value"):
    return [OptionCandidate(text=t, source=source) for t in texts]


# ---------------------------------------------------------------------------
# Deduplication guarantee
# ---------------------------------------------------------------------------

class TestModifierNeverDuplicates:
    def test_same_modifier_twice_via_slots_yields_one_selection(self):
        """Two MODIFIER slots for ranch → one selection, duplicate_names records it."""
        group = _make_group([_make_choice("ranch", "Ranch")])
        result = _resolver().resolve(
            group=group,
            normalized_user_text="ranch ranch",
            option_candidates=_candidates("ranch", "ranch"),
            already_selected_ids=[],
        )
        matched_ids = [sel.modifier_id for sel in result.selections]
        assert matched_ids.count("ranch") == 1

    def test_duplicate_already_selected_blocked(self):
        """Modifier already in already_selected_ids is skipped (added to duplicate_names)."""
        group = _make_group([_make_choice("ranch", "Ranch")])
        result = _resolver().resolve(
            group=group,
            normalized_user_text="ranch",
            option_candidates=_candidates("ranch"),
            already_selected_ids=["ranch"],
        )
        matched_ids = [sel.modifier_id for sel in result.selections]
        assert "ranch" not in matched_ids
        assert "Ranch" in result.duplicate_names

    def test_distinct_modifiers_both_accepted(self):
        group = _make_group([
            _make_choice("ranch", "Ranch"),
            _make_choice("bbq", "BBQ Sauce"),
        ])
        result = _resolver().resolve(
            group=group,
            normalized_user_text="ranch bbq",
            option_candidates=_candidates("ranch", "bbq sauce"),
            already_selected_ids=[],
        )
        matched_ids = [sel.modifier_id for sel in result.selections]
        assert "ranch" in matched_ids
        assert "bbq" in matched_ids

    def test_new_modifier_accepted_when_other_already_selected(self):
        group = _make_group([
            _make_choice("ranch", "Ranch"),
            _make_choice("bbq", "BBQ Sauce"),
        ])
        result = _resolver().resolve(
            group=group,
            normalized_user_text="bbq",
            option_candidates=_candidates("bbq sauce"),
            already_selected_ids=["ranch"],
        )
        matched_ids = [sel.modifier_id for sel in result.selections]
        assert "bbq" in matched_ids
        assert "ranch" not in matched_ids

    def test_no_allow_duplicate_selections_attribute_on_modifier_group(self):
        """PendingModifierGroup has no allow_duplicate_selections — does not exist."""
        from app.state_machine.models.pending_item_models import PendingModifierGroup
        group = PendingModifierGroup(
            group_id="g",
            name="Sauces",
            is_required=False,
            min_selector=0,
            max_selector=2,
            choices=[],
            choices_by_modifier_id={},
            choices_by_normalized_name={},
            choice_names=(),
            normalized_choice_names=(),
            top_choice_names=(),
        )
        assert not hasattr(group, "allow_duplicate_selections")

    def test_same_utterance_candidates_deduplicated_within_pass(self):
        """Even without already_selected_ids, same candidate twice → one match."""
        group = _make_group([_make_choice("mayo", "Mayo")])
        result = _resolver().resolve(
            group=group,
            normalized_user_text="mayo mayo",
            option_candidates=_candidates("mayo", "mayo"),
        )
        matched_ids = [sel.modifier_id for sel in result.selections]
        assert matched_ids.count("mayo") == 1
