# tests/state_machine/handlers/item/add_item/test_side_group_resolver_duplicates.py
"""Tests for SideGroupResolver duplicate-side behavior.

Verifies that when allow_duplicate_selections=True, the resolver:
- allows re-selecting an item that is already in already_selected_ids
- expands repeated slot values via slot_value_counts
- prevents double-match of the same candidate within a single resolve() pass
- still deduplicates when allow_duplicate_selections=False
"""
from types import SimpleNamespace

from app.state_machine.handlers.item.add_item.side_group_resolver import SideGroupResolver
from app.state_machine.handlers.item.add_item.option_matching import OptionCandidate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_choice(item_id: str, name: str):
    from app.nlu.query_normalization.text_preprocessor import normalize_text
    return SimpleNamespace(
        item_id=item_id,
        name=name,
        normalized_name=normalize_text(name),
        match_texts=(normalize_text(name),),
    )


def _make_group(choices, *, allow_duplicate_selections=True):
    return SimpleNamespace(
        group_id="drinks",
        name="Drinks",
        choices=choices,
        allow_duplicate_selections=allow_duplicate_selections,
    )


def _resolver():
    return SideGroupResolver()


def _candidates(*texts, source="slot_value"):
    return [OptionCandidate(text=t, source=source) for t in texts]


# ---------------------------------------------------------------------------
# allow_duplicate_selections=True (default)
# ---------------------------------------------------------------------------

class TestAllowDuplicates:
    def test_selects_new_item_when_already_selected_allows_dupes(self):
        group = _make_group([
            _make_choice("coke", "Coke"),
            _make_choice("sprite", "Sprite"),
        ])
        result = _resolver().resolve(
            group=group,
            normalized_user_text="coke",
            option_candidates=_candidates("coke"),
            already_selected_ids=["coke"],  # already picked once
        )
        assert "coke" in result.matched_item_ids

    def test_slot_value_counts_repeats_match(self):
        """slot_value_counts={"coke": 3} → coke appears 3x in results."""
        group = _make_group([_make_choice("coke", "Coke")])
        result = _resolver().resolve(
            group=group,
            normalized_user_text="coke",
            option_candidates=_candidates("coke"),
            slot_value_counts={"coke": 3},
        )
        assert result.matched_item_ids.count("coke") == 3
        assert result.matched_names.count("Coke") == 3

    def test_slot_value_counts_two_repeats(self):
        group = _make_group([_make_choice("coke", "Coke")])
        result = _resolver().resolve(
            group=group,
            normalized_user_text="coke",
            option_candidates=_candidates("coke"),
            slot_value_counts={"coke": 2},
        )
        assert result.matched_item_ids.count("coke") == 2

    def test_single_candidate_does_not_double_count_without_slot_counts(self):
        """Without slot_value_counts, one candidate → one match (no implicit doubling)."""
        group = _make_group([_make_choice("coke", "Coke")])
        result = _resolver().resolve(
            group=group,
            normalized_user_text="coke",
            option_candidates=_candidates("coke"),
        )
        assert result.matched_item_ids.count("coke") == 1

    def test_multiple_distinct_candidates_each_matched_once(self):
        group = _make_group([
            _make_choice("coke", "Coke"),
            _make_choice("sprite", "Sprite"),
        ])
        result = _resolver().resolve(
            group=group,
            normalized_user_text="coke sprite",
            option_candidates=_candidates("coke", "sprite"),
        )
        assert "coke" in result.matched_item_ids
        assert "sprite" in result.matched_item_ids
        # Each appears exactly once (no duplication without slot_value_counts)
        assert result.matched_item_ids.count("coke") == 1
        assert result.matched_item_ids.count("sprite") == 1

    def test_already_selected_plus_slot_counts_combined(self):
        """User already has 1 coke; says 'coke coke' → slot_value_counts=2 → adds 2 more."""
        group = _make_group([_make_choice("coke", "Coke")])
        result = _resolver().resolve(
            group=group,
            normalized_user_text="coke coke",
            option_candidates=_candidates("coke"),
            already_selected_ids=["coke"],
            slot_value_counts={"coke": 2},
        )
        assert result.matched_item_ids.count("coke") == 2


# ---------------------------------------------------------------------------
# allow_duplicate_selections=False
# ---------------------------------------------------------------------------

class TestNoDuplicates:
    def test_already_selected_item_is_skipped(self):
        group = _make_group(
            [_make_choice("coke", "Coke"), _make_choice("sprite", "Sprite")],
            allow_duplicate_selections=False,
        )
        result = _resolver().resolve(
            group=group,
            normalized_user_text="coke",
            option_candidates=_candidates("coke"),
            already_selected_ids=["coke"],
        )
        assert "coke" not in result.matched_item_ids

    def test_new_item_still_matched_when_other_selected(self):
        group = _make_group(
            [_make_choice("coke", "Coke"), _make_choice("sprite", "Sprite")],
            allow_duplicate_selections=False,
        )
        result = _resolver().resolve(
            group=group,
            normalized_user_text="sprite",
            option_candidates=_candidates("sprite"),
            already_selected_ids=["coke"],
        )
        assert "sprite" in result.matched_item_ids

    def test_slot_value_counts_ignored_when_no_dupes_allowed(self):
        """slot_value_counts is only meaningful when allow_duplicate_selections=True."""
        group = _make_group(
            [_make_choice("coke", "Coke")],
            allow_duplicate_selections=False,
        )
        result = _resolver().resolve(
            group=group,
            normalized_user_text="coke",
            option_candidates=_candidates("coke"),
            slot_value_counts={"coke": 3},
        )
        # Only one match because duplicates are disallowed
        assert result.matched_item_ids.count("coke") == 1


# ---------------------------------------------------------------------------
# Unmatched values not polluted by duplicate intent
# ---------------------------------------------------------------------------

class TestUnmatchedCleanup:
    def test_no_unmatched_for_fully_resolved_duplicate_request(self):
        group = _make_group([_make_choice("coke", "Coke")])
        result = _resolver().resolve(
            group=group,
            normalized_user_text="coke",
            option_candidates=_candidates("coke"),
            slot_value_counts={"coke": 2},
        )
        assert result.unmatched_values == []
