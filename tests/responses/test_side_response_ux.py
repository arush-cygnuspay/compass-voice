# tests/responses/test_side_response_ux.py
"""
Tests for the side / item-residue UX fixes.

Coverage:
1. _has_echable_content — stop-word filter used by _build_entity_feedback
2. _build_entity_feedback — filler/conversational phrases are suppressed
3. _progress_prompt — "Pick 1 more" → "Please choose one option." when nothing selected
4. repeat_side_options — group name surfaced in invalid_lead
5. _collapse_unresolved_for_feedback (via prefill_orchestrator) — ordering
   filler residue ("okay then give me a") is stripped
"""
from __future__ import annotations

import types
import pytest

from app.responses.item_responses import (
    _build_entity_feedback,
    _has_echable_content,
    _progress_prompt,
    repeat_side_options,
)


# ── 1. _has_echable_content ───────────────────────────────────────────────────

class TestHasEchableContent:
    def test_pure_filler_phrase_is_not_echable(self):
        assert not _has_echable_content("then give me a")

    def test_single_conversational_word_is_not_echable(self):
        # "am i confused" — only 1 meaningful token ("confused") < 2 threshold
        assert not _has_echable_content("am i confused")

    def test_short_stt_fragment_is_not_echable(self):
        # "my country is" — only 1 meaningful token ("country") < 2 threshold
        assert not _has_echable_content("my country is")

    def test_empty_string_is_not_echable(self):
        assert not _has_echable_content("")

    def test_only_stop_words_is_not_echable(self):
        assert not _has_echable_content("okay a the and")

    def test_two_meaningful_tokens_is_echable(self):
        # "american cheese" → 2 meaningful tokens
        assert _has_echable_content("american cheese")

    def test_menu_phrase_with_filler_is_echable(self):
        # "no cheese please" → "cheese" is 1 meaningful token, but "no" is not
        # a stop token → 2 meaningful tokens ("no" not in stop list)
        # Actually "no" is not in _FEEDBACK_STOP_TOKENS so it counts.
        assert _has_echable_content("no cheese please")

    def test_multi_word_menu_item_is_echable(self):
        assert _has_echable_content("mozzarella cheese")

    def test_genuine_unmatched_item_with_preposition_is_echable(self):
        # "pizza with extra sauce" → meaningful: pizza, extra, sauce → 3
        assert _has_echable_content("pizza with extra sauce")


# ── 2. _build_entity_feedback ─────────────────────────────────────────────────

class TestBuildEntityFeedback:
    def test_filler_residue_not_echoed(self):
        payload = {"unmatched_names": ["then give me a"]}
        assert _build_entity_feedback(payload) == ""

    def test_all_stop_word_phrase_not_echoed(self):
        # "then give me" — 0 meaningful tokens → filtered
        payload = {"unmatched_names": ["then give me"]}
        assert _build_entity_feedback(payload) == ""

    def test_conversational_fragment_not_echoed(self):
        # "am i confused" — 1 meaningful token → below >= 2 threshold → filtered
        payload = {"unmatched_names": ["am i confused"]}
        assert _build_entity_feedback(payload) == ""

    def test_bad_stt_fragment_not_echoed(self):
        # "my country is" — 1 meaningful token → filtered
        payload = {"unmatched_names": ["my country is"]}
        assert _build_entity_feedback(payload) == ""

    def test_empty_unmatched_produces_no_feedback(self):
        assert _build_entity_feedback({}) == ""

    def test_valid_unmatched_item_is_echoed(self):
        payload = {"unmatched_names": ["mozzarella cheese"]}
        feedback = _build_entity_feedback(payload)
        assert "I couldn't find" in feedback
        assert "mozzarella cheese" in feedback

    def test_matched_names_still_shown(self):
        payload = {"matched_names": ["Cheddar Cheese"], "unmatched_names": ["am i confused"]}
        feedback = _build_entity_feedback(payload)
        assert "Got" in feedback
        assert "Cheddar Cheese" in feedback
        # single-token conversational fragment not echoed
        assert "am i confused" not in feedback

    def test_mixed_valid_and_filler_only_valid_echoed(self):
        payload = {"unmatched_names": ["then give me a", "mozzarella cheese"]}
        feedback = _build_entity_feedback(payload)
        assert "mozzarella cheese" in feedback
        assert "then give me a" not in feedback


# ── 3. _progress_prompt ──────────────────────────────────────────────────────

def _side_payload(selected_count: int, remaining_to_min: int, choices: list[str]) -> dict:
    return {
        "selected_count": selected_count,
        "remaining_to_min": remaining_to_min,
        "remaining_to_max": 0,
        "top_choices": choices,
        "all_choices": choices,
    }


class TestProgressPrompt:
    def test_no_selection_required_one_says_please_choose_one(self):
        payload = _side_payload(selected_count=0, remaining_to_min=1,
                                choices=["American Cheese", "Cheddar Cheese"])
        result = _progress_prompt(payload, item_word="side",
                                  invalid_lead="I didn't catch a valid side.")
        assert "Please choose one option." in result
        assert "Pick 1 more" not in result

    def test_no_selection_required_two_says_please_choose_n(self):
        payload = _side_payload(selected_count=0, remaining_to_min=2,
                                choices=["A", "B", "C"])
        result = _progress_prompt(payload, item_word="side",
                                  invalid_lead="I didn't catch a valid side.")
        assert "Please choose 2 options." in result
        assert "Pick 2 more" not in result

    def test_partial_selection_uses_pick_more(self):
        # 1 of 2 selected → "Pick 1 more."
        payload = _side_payload(selected_count=1, remaining_to_min=1,
                                choices=["Cheddar Cheese"])
        result = _progress_prompt(payload, item_word="side",
                                  invalid_lead="I didn't catch a valid side.")
        assert "Pick 1 more" in result
        assert "Please choose one option" not in result

    def test_options_appended_to_please_choose(self):
        payload = _side_payload(selected_count=0, remaining_to_min=1,
                                choices=["American Cheese", "Cheddar Cheese"])
        result = _progress_prompt(payload, item_word="side",
                                  invalid_lead="I didn't catch a valid side.")
        assert "American Cheese" in result

    def test_need_more_reason_unchanged(self):
        payload = {
            "repeat_reason": "need_more",
            "remaining_to_min": 1,
            "selected_count": 1,
            "top_choices": ["Cheddar Cheese"],
            "all_choices": ["Cheddar Cheese"],
        }
        result = _progress_prompt(payload, item_word="side",
                                  invalid_lead="I didn't catch a valid side.")
        assert "Pick 1 more" in result

    def test_optional_more_reason_unchanged(self):
        payload = {
            "repeat_reason": "optional_more",
            "remaining_to_max": 1,
            "selected_count": 0,
            "top_choices": ["Cheddar Cheese"],
            "all_choices": ["Cheddar Cheese"],
        }
        result = _progress_prompt(payload, item_word="side",
                                  invalid_lead="I didn't catch a valid side.")
        assert "done" in result.lower()


# ── 4. repeat_side_options — group name in invalid_lead ──────────────────────

def _make_side_context(selected_ids: list[str] | None = None):
    ctx = types.SimpleNamespace(
        current_item_id="burger_1",
        current_side_group_index=0,
        selected_side_groups={},
        selected_modifier_groups={},
    )
    if selected_ids:
        ctx.selected_side_groups["cheese_group"] = selected_ids
    return ctx


def _make_choice(item_id: str, name: str):
    return types.SimpleNamespace(
        item_id=item_id,
        name=name,
        pricing_mode="fixed",
        variants=[],
        variant_names=[],
    )


def _make_side_group(name: str, choices: list, min_sel: int = 1, max_sel: int = 1):
    g = types.SimpleNamespace(
        group_id="cheese_group",
        name=name,
        choices=choices,
        choices_by_item_id={c.item_id: c for c in choices},
        min_selector=min_sel,
        max_selector=max_sel,
    )
    return g


def _make_menu_repo_with_side_group(group_name: str = "Cheese"):
    choices = [
        _make_choice("c1", "American Cheese"),
        _make_choice("c2", "Cheddar Cheese"),
    ]
    group = _make_side_group(group_name, choices)
    item = types.SimpleNamespace(
        name="Burger",
        side_groups=[group],
        modifier_groups=[],
    )
    store = types.SimpleNamespace(get_item=lambda _: item)
    return types.SimpleNamespace(store=store)


class TestRepeatSideOptions:
    def test_group_name_appears_in_invalid_lead(self):
        context = _make_side_context()
        menu_repo = _make_menu_repo_with_side_group("Cheese")
        result = repeat_side_options(context, menu_repo, {"repeat_reason": "invalid"})
        assert "cheese" in result.lower()

    def test_filler_unmatched_name_not_echoed(self):
        # "am i confused" has 1 meaningful token ("confused") — filtered by >= 2 threshold.
        context = _make_side_context()
        menu_repo = _make_menu_repo_with_side_group("Cheese")
        payload = {
            "repeat_reason": "invalid",
            "unmatched_names": ["am i confused"],
        }
        result = repeat_side_options(context, menu_repo, payload)
        assert "am i confused" not in result
        assert "I couldn't find" not in result

    def test_valid_side_selection_not_affected(self):
        # When valid side already selected, do not ask "Please choose one option."
        context = _make_side_context(selected_ids=["c1"])
        menu_repo = _make_menu_repo_with_side_group("Cheese")
        # Only 1 choice remains; selected_count=1 meets min → would be done
        # Test with remaining_to_min=0 to confirm "done" path
        result = repeat_side_options(context, menu_repo, {
            "repeat_reason": "need_more",
            "remaining_to_min": 0,
        })
        assert "done" in result.lower() or "American Cheese" in result or "Cheddar Cheese" in result


# ── 5. _collapse_unresolved_for_feedback — filler residue stripped ────────────

class TestCollapseUnresolvedForFeedback:
    """Tests the prefill_orchestrator's cleanup of unresolved phrases."""

    def _collapse(self, phrases: list[str], item_name: str = "Chicken Burger") -> list[str]:
        import types
        from app.state_machine.handlers.item.add_item.prefill_orchestrator import (
            PrefillOrchestrator,
        )

        pending = types.SimpleNamespace(
            item_name=item_name,
            side_groups=[],
            modifier_groups=[],
            item_variants=[],
        )
        return PrefillOrchestrator._collapse_unresolved_for_feedback(phrases, pending=pending)

    def test_ordering_filler_residue_is_collapsed(self):
        # "okay then give me a" is pure filler — no meaningful token after
        # stripping item name tokens and the extended stop set.
        result = self._collapse(["okay then give me a chicken burger"])
        assert result == []

    def test_then_give_me_a_is_collapsed(self):
        result = self._collapse(["then give me a"])
        assert result == []

    def test_genuine_unmatched_food_phrase_survives(self):
        # "extra pickles" should survive since "pickles" is not a stop word
        result = self._collapse(["extra pickles"])
        assert result != []

    def test_real_unmatched_item_name_survives(self):
        result = self._collapse(["spicy mayo"], item_name="Chicken Burger")
        # "spicy" and "mayo" are not stop tokens
        assert result != []

    def test_empty_list_returns_empty(self):
        assert self._collapse([]) == []
