# tests/responses/test_side_response_ux_duplicates.py
"""Tests for voice response formatting when duplicate sides are selected.

Phase 10: _format_names_with_counts collapses repeated names into natural
spoken phrases so the bot says "Got Coke twice." instead of "Got Coke and Coke."
"""
from app.responses.item.format_utils import _format_names_with_counts, _build_entity_feedback


class TestFormatNamesWithCounts:
    def test_single_name_unchanged(self):
        assert _format_names_with_counts(["Coke"]) == "Coke"

    def test_two_distinct_names_uses_and(self):
        assert _format_names_with_counts(["Coke", "Sprite"]) == "Coke and Sprite"

    def test_two_same_names_says_twice(self):
        assert _format_names_with_counts(["Coke", "Coke"]) == "Coke twice"

    def test_three_same_names_says_n_times(self):
        assert _format_names_with_counts(["Coke", "Coke", "Coke"]) == "Coke 3 times"

    def test_four_same_names_says_n_times(self):
        assert _format_names_with_counts(["Coke", "Coke", "Coke", "Coke"]) == "Coke 4 times"

    def test_two_same_plus_one_distinct(self):
        assert _format_names_with_counts(["Coke", "Coke", "Sprite"]) == "Coke twice and Sprite"

    def test_one_plus_three_same(self):
        result = _format_names_with_counts(["Sprite", "Coke", "Coke", "Coke"])
        assert result == "Sprite and Coke 3 times"

    def test_all_distinct_three_names(self):
        result = _format_names_with_counts(["Coke", "Sprite", "Water"])
        assert result == "Coke, Sprite, and Water"

    def test_mixed_duplicate_groups(self):
        result = _format_names_with_counts(["Coke", "Coke", "Sprite", "Sprite"])
        assert result == "Coke twice and Sprite twice"

    def test_order_of_first_appearance_preserved(self):
        result = _format_names_with_counts(["Sprite", "Sprite", "Coke"])
        assert result == "Sprite twice and Coke"

    def test_empty_list_returns_empty_string(self):
        assert _format_names_with_counts([]) == ""

    def test_whitespace_only_items_filtered(self):
        assert _format_names_with_counts(["  ", "Coke", ""]) == "Coke"

    def test_single_item_two_groups_three_items(self):
        result = _format_names_with_counts(["Coke", "Sprite", "Coke"])
        # First appearance order: Coke, Sprite
        assert result == "Coke twice and Sprite"


class TestBuildEntityFeedbackWithDuplicates:
    def test_single_match_no_duplicate(self):
        payload = {"matched_names": ["Coke"], "unmatched_names": []}
        result = _build_entity_feedback(payload)
        assert result == "Got Coke. "

    def test_two_duplicate_matches_says_twice(self):
        payload = {"matched_names": ["Coke", "Coke"], "unmatched_names": []}
        result = _build_entity_feedback(payload)
        assert result == "Got Coke twice. "

    def test_three_duplicate_matches_says_n_times(self):
        payload = {"matched_names": ["Coke", "Coke", "Coke"], "unmatched_names": []}
        result = _build_entity_feedback(payload)
        assert result == "Got Coke 3 times. "

    def test_mixed_duplicates_in_feedback(self):
        payload = {"matched_names": ["Coke", "Coke", "Sprite"], "unmatched_names": []}
        result = _build_entity_feedback(payload)
        assert result == "Got Coke twice and Sprite. "

    def test_distinct_names_still_use_and(self):
        payload = {"matched_names": ["Coke", "Sprite"], "unmatched_names": []}
        result = _build_entity_feedback(payload)
        assert result == "Got Coke and Sprite. "

    def test_unmatched_names_still_echoed(self):
        payload = {
            "matched_names": ["Coke", "Coke"],
            "unmatched_names": ["mystery drink"],
        }
        result = _build_entity_feedback(payload)
        assert "Got Coke twice." in result
        assert "mystery drink" in result
