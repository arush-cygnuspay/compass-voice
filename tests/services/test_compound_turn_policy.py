# tests/services/test_compound_turn_policy.py
"""Unit tests for app/services/compound_turn_policy.py.

Coverage:
  - All 8 decision rules in _decide()
  - looks_like_item_with_option() public helper
  - decide_compound_fallback() never raises (exception guard)
  - _count_gpt_items() fallback paths
  - All CompoundFallbackDecision enum values reachable
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest

from app.services.compound_turn_policy import (
    CompoundFallbackDecision,
    decide_compound_fallback,
    looks_like_item_with_option,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class _GptItem:
    item_name: str
    quantity: int = 1


@dataclass
class _ValidatedPlan:
    items: tuple[_GptItem, ...]


@dataclass
class _PlannerResult:
    """Minimal GPT planner result (GptAddItemPlanResult-shaped)."""
    safe_to_apply: bool = True
    validated_plan: _ValidatedPlan | None = None

    def with_items(self, *names: str) -> "_PlannerResult":
        self.validated_plan = _ValidatedPlan(
            items=tuple(_GptItem(n) for n in names)
        )
        return self


@dataclass
class _LocalPlannerResult:
    """Minimal ParsedMultiItemPlan-shaped stub."""
    is_compound: bool = False
    items: list = field(default_factory=list)


def _gpt_plan_with(*names: str) -> _PlannerResult:
    return _PlannerResult().with_items(*names)


def _call(
    transcript: str = "test",
    planner_result: Any = None,
    local_planner_result: Any = None,
    unsafe_slot_reason: "str | None" = None,
    valid_candidates_count: int = 0,
    unresolved_spans: list[str] | None = None,
    reprompt_count: int = 0,
) -> CompoundFallbackDecision:
    return decide_compound_fallback(
        transcript=transcript,
        planner_result=planner_result,
        local_planner_result=local_planner_result,
        unsafe_slot_reason=unsafe_slot_reason,
        valid_candidates_count=valid_candidates_count,
        unresolved_spans=unresolved_spans or [],
        reprompt_count=reprompt_count,
    )


# ---------------------------------------------------------------------------
# Rule 1: GPT planner has ≥1 valid item → EXECUTE_VALID_PLAN
# ---------------------------------------------------------------------------


class TestRule1GptPlannerValid:
    """GPT planner result with items → always EXECUTE_VALID_PLAN."""

    def test_gpt_one_item_execute(self) -> None:
        result = _call(
            transcript="tuna melt",
            planner_result=_gpt_plan_with("tuna melt"),
            unsafe_slot_reason="multi_item_slots",
        )
        assert result == CompoundFallbackDecision.EXECUTE_VALID_PLAN

    def test_gpt_two_items_execute(self) -> None:
        result = _call(
            transcript="burger and fries",
            planner_result=_gpt_plan_with("burger", "fries"),
            unsafe_slot_reason="multi_item_slots",
        )
        assert result == CompoundFallbackDecision.EXECUTE_VALID_PLAN

    def test_gpt_overrides_broken_reason(self) -> None:
        """GPT plan wins even when unsafe_slot_reason is set."""
        result = _call(
            planner_result=_gpt_plan_with("wings"),
            unsafe_slot_reason="long_compound_add_item",
            valid_candidates_count=0,
        )
        assert result == CompoundFallbackDecision.EXECUTE_VALID_PLAN

    def test_gpt_zero_items_falls_through_to_later_rules(self) -> None:
        """Empty validated_plan.items must NOT match Rule 1."""
        plan = _PlannerResult(
            safe_to_apply=True,
            validated_plan=_ValidatedPlan(items=()),
        )
        result = _call(
            planner_result=plan,
            unsafe_slot_reason=None,
        )
        # Rule 4 fires: no broken reason → EXECUTE_VALID_PLAN for a different reason
        assert result == CompoundFallbackDecision.EXECUTE_VALID_PLAN

    def test_gpt_plan_none_falls_through(self) -> None:
        """planner_result=None → Rule 1 skipped.

        At reprompt_count=0 (first encounter) the policy falls through to the
        single-item path (Rule 9), NOT to a clarification prompt.
        """
        result = _call(
            planner_result=None,
            unsafe_slot_reason="multi_item_slots",
        )
        # Rule 9: first encounter → EXECUTE_VALID_PLAN (let single-item path try)
        assert result == CompoundFallbackDecision.EXECUTE_VALID_PLAN

    def test_gpt_plan_direct_items_attribute(self) -> None:
        """SmartTurnPlan-shaped result with direct .items attribute."""
        plan = MagicMock()
        plan.validated_plan = None  # no validated_plan
        plan.items = [_GptItem("burger"), _GptItem("coke")]
        result = _call(planner_result=plan)
        assert result == CompoundFallbackDecision.EXECUTE_VALID_PLAN

    def test_gpt_plan_validated_plan_overrides_items(self) -> None:
        """When validated_plan exists, items on the outer object is ignored."""
        plan = MagicMock()
        plan.validated_plan.items = [_GptItem("burger")]
        result = _call(planner_result=plan)
        assert result == CompoundFallbackDecision.EXECUTE_VALID_PLAN


# ---------------------------------------------------------------------------
# Rule 2: Local planner ≥1 item + unresolved → EXECUTE_PARTIAL_AND_CLARIFY
# ---------------------------------------------------------------------------


class TestRule2LocalPlannerPartial:

    def test_one_valid_with_unresolved_spans(self) -> None:
        result = _call(
            valid_candidates_count=1,
            unresolved_spans=["dragon pasta"],
        )
        assert result == CompoundFallbackDecision.EXECUTE_PARTIAL_AND_CLARIFY

    def test_two_valid_with_unresolved(self) -> None:
        result = _call(
            valid_candidates_count=2,
            unresolved_spans=["mystery item"],
        )
        assert result == CompoundFallbackDecision.EXECUTE_PARTIAL_AND_CLARIFY

    def test_unresolved_spans_none_treated_as_empty(self) -> None:
        """None unresolved_spans must not raise and must not trigger Rule 2."""
        result = _call(
            valid_candidates_count=1,
            unresolved_spans=None,
        )
        # Rule 3 fires
        assert result == CompoundFallbackDecision.EXECUTE_VALID_PLAN


# ---------------------------------------------------------------------------
# Rule 3: Local planner ≥1 item, all resolved → EXECUTE_VALID_PLAN
# ---------------------------------------------------------------------------


class TestRule3LocalPlannerAllResolved:

    def test_one_valid_no_unresolved(self) -> None:
        result = _call(
            valid_candidates_count=1,
            unresolved_spans=[],
        )
        assert result == CompoundFallbackDecision.EXECUTE_VALID_PLAN

    def test_three_valid_no_unresolved(self) -> None:
        result = _call(
            valid_candidates_count=3,
            unresolved_spans=[],
            unsafe_slot_reason="multi_item_slots",
        )
        assert result == CompoundFallbackDecision.EXECUTE_VALID_PLAN


# ---------------------------------------------------------------------------
# Rule 4: Slots are safe (no broken reason) → EXECUTE_VALID_PLAN
# ---------------------------------------------------------------------------


class TestRule4SlotsSafe:

    def test_no_broken_reason_executes(self) -> None:
        result = _call(unsafe_slot_reason=None)
        assert result == CompoundFallbackDecision.EXECUTE_VALID_PLAN

    def test_empty_string_broken_reason_is_falsy(self) -> None:
        result = _call(unsafe_slot_reason="")
        assert result == CompoundFallbackDecision.EXECUTE_VALID_PLAN


# ---------------------------------------------------------------------------
# Rule 5: Recoverable broken-slot reason → EXECUTE_VALID_PLAN
# ---------------------------------------------------------------------------


class TestRule5RecoverableReason:

    @pytest.mark.parametrize("reason", [
        "low_confidence_add_item",
        "size_word_inside_item",
        "numeric_piece_variant",
        "multi_variant_slots",
    ])
    def test_recoverable_reason_executes(self, reason: str) -> None:
        result = _call(unsafe_slot_reason=reason)
        assert result == CompoundFallbackDecision.EXECUTE_VALID_PLAN

    def test_unknown_reason_not_recoverable(self) -> None:
        """A reason not in _RECOVERABLE_REASONS falls through to later rules.

        At reprompt_count=0 (default) Rule 9 fires: single-item path gets a
        first attempt before we show a clarification prompt.
        """
        result = _call(unsafe_slot_reason="totally_unknown_reason")
        # No "with" marker → Rule 9 (first encounter) → EXECUTE_VALID_PLAN
        assert result == CompoundFallbackDecision.EXECUTE_VALID_PLAN


# ---------------------------------------------------------------------------
# Rule 6: "with/no/extra/…" marker in multi-item reason → EXECUTE_VALID_PLAN
# ---------------------------------------------------------------------------


class TestRule6ItemOptionMarker:

    @pytest.mark.parametrize("multi_reason", [
        "multi_item_slots",
        "long_compound_add_item",
        "merged_item_slot",
    ])
    @pytest.mark.parametrize("transcript", [
        "burger with fries",
        "grilled chicken sandwich with small coke",
        "tuna melt with mayo",
        "6 piece wings with buffalo sauce",
        "cheeseburger no onions",
        "large fries without salt",
        "wings extra crispy",
        "fries on the side",
        "burger hold the onions",          # " hold the " must appear mid-utterance
        "burger add bacon",                # " add " mid-utterance
        "salad light dressing",            # " light " mid-utterance
        "chicken sandwich easy on the mayo",  # " easy on the " mid-utterance
    ])
    def test_item_option_marker_executes(self, multi_reason: str, transcript: str) -> None:
        result = _call(
            transcript=transcript,
            unsafe_slot_reason=multi_reason,
        )
        assert result == CompoundFallbackDecision.EXECUTE_VALID_PLAN, (
            f"Expected EXECUTE_VALID_PLAN for {transcript!r} / {multi_reason!r}, "
            f"got {result!r}"
        )

    def test_no_marker_no_execute(self) -> None:
        """Transcript without item-option marker → Rule 6 does NOT fire.

        At reprompt_count=0 (first encounter) Rule 9 applies: fall through to
        the single-item path instead of asking for the first item.
        """
        result = _call(
            transcript="burger fries rings",
            unsafe_slot_reason="multi_item_slots",
        )
        # Rule 9 (first encounter, reprompt=0) → EXECUTE_VALID_PLAN
        assert result == CompoundFallbackDecision.EXECUTE_VALID_PLAN

    def test_marker_only_for_multi_item_reasons(self) -> None:
        """Rule 6 only fires for multi-item family, not arbitrary reasons."""
        # "low_confidence_add_item" is already caught by Rule 5
        result = _call(
            transcript="burger with fries",
            unsafe_slot_reason="low_confidence_add_item",
        )
        assert result == CompoundFallbackDecision.EXECUTE_VALID_PLAN  # Rule 5


# ---------------------------------------------------------------------------
# Rule 7: reprompt_count ≥ 2 → FALLBACK_ONE_AT_A_TIME
# ---------------------------------------------------------------------------


class TestRule7RepeatedFailure:

    def test_reprompt_count_2_escalates(self) -> None:
        result = _call(
            transcript="stuff and things",
            unsafe_slot_reason="multi_item_slots",
            reprompt_count=2,
        )
        assert result == CompoundFallbackDecision.FALLBACK_ONE_AT_A_TIME

    def test_reprompt_count_5_escalates(self) -> None:
        result = _call(
            transcript="foo bar baz",
            unsafe_slot_reason="long_compound_add_item",
            reprompt_count=5,
        )
        assert result == CompoundFallbackDecision.FALLBACK_ONE_AT_A_TIME

    def test_reprompt_count_1_does_not_escalate(self) -> None:
        result = _call(
            unsafe_slot_reason="multi_item_slots",
            reprompt_count=1,
        )
        assert result == CompoundFallbackDecision.FALLBACK_REPEAT_FIRST_ITEM

    def test_reprompt_count_0_falls_to_single_item(self) -> None:
        """First encounter (reprompt=0): falls through to single-item path, not clarification."""
        result = _call(
            unsafe_slot_reason="multi_item_slots",
            reprompt_count=0,
        )
        assert result == CompoundFallbackDecision.EXECUTE_VALID_PLAN


# ---------------------------------------------------------------------------
# Rule 8/9: Default — ask on second encounter; fall through on first
# ---------------------------------------------------------------------------


class TestRule8Default:

    def test_reprompt_1_asks_for_first_item(self) -> None:
        """Second encounter (reprompt=1) → FALLBACK_REPEAT_FIRST_ITEM."""
        result = _call(
            transcript="something unclear",
            unsafe_slot_reason="multi_item_slots",
            reprompt_count=1,
        )
        assert result == CompoundFallbackDecision.FALLBACK_REPEAT_FIRST_ITEM

    def test_first_encounter_falls_through(self) -> None:
        """First encounter (reprompt=0) → EXECUTE_VALID_PLAN (single-item path gets a try)."""
        result = _call(
            transcript="something unclear",
            unsafe_slot_reason="multi_item_slots",
            reprompt_count=0,
        )
        assert result == CompoundFallbackDecision.EXECUTE_VALID_PLAN

    def test_merged_item_slot_no_marker_second_encounter(self) -> None:
        """merged_item_slot + no marker + reprompt=1 → FALLBACK_REPEAT_FIRST_ITEM."""
        result = _call(
            transcript="largefriessmallcoke",  # no spaces → no marker
            unsafe_slot_reason="merged_item_slot",
            reprompt_count=1,
        )
        assert result == CompoundFallbackDecision.FALLBACK_REPEAT_FIRST_ITEM

    def test_merged_item_slot_no_marker_first_encounter(self) -> None:
        """merged_item_slot + no marker + reprompt=0 → EXECUTE_VALID_PLAN."""
        result = _call(
            transcript="largefriessmallcoke",
            unsafe_slot_reason="merged_item_slot",
        )
        assert result == CompoundFallbackDecision.EXECUTE_VALID_PLAN


# ---------------------------------------------------------------------------
# looks_like_item_with_option()
# ---------------------------------------------------------------------------


class TestLooksLikeItemWithOption:

    @pytest.mark.parametrize("text", [
        "burger with fries",
        "tuna melt with mayo",
        "chicken sandwich no onions",
        "6 piece wings with buffalo sauce",
        "cheeseburger no onions",
        "fries without salt",
        "wings extra crispy",
        "fries on the side",
        "burger hold the onions",         # " hold the " mid-utterance
        "burger add bacon",               # " add " mid-utterance
        "salad light dressing",           # " light " mid-utterance
        "chicken sandwich easy on the mayo",  # " easy on the " mid-utterance
    ])
    def test_returns_true_for_option_markers(self, text: str) -> None:
        assert looks_like_item_with_option(text) is True, (
            f"Expected True for {text!r}"
        )

    @pytest.mark.parametrize("text", [
        "burger fries rings",
        "i want a burger and a coke",
        "burger",
        "",
        "   ",
    ])
    def test_returns_false_for_no_markers(self, text: str) -> None:
        # Note: "i want a burger and a coke" has no _ITEM_OPTION_MARKERS
        # It uses "and" which is NOT in the marker list (it signals two items)
        assert looks_like_item_with_option(text) is False, (
            f"Expected False for {text!r}"
        )

    def test_case_insensitive(self) -> None:
        assert looks_like_item_with_option("Burger WITH Fries") is True

    def test_none_like_empty_string(self) -> None:
        assert looks_like_item_with_option("") is False


# ---------------------------------------------------------------------------
# decide_compound_fallback() never raises
# ---------------------------------------------------------------------------


class TestNeverRaises:

    def test_none_transcript(self) -> None:
        result = decide_compound_fallback(
            transcript=None,  # type: ignore[arg-type]
            planner_result=None,
            local_planner_result=None,
            unsafe_slot_reason=None,
            valid_candidates_count=0,
            unresolved_spans=[],
        )
        assert isinstance(result, CompoundFallbackDecision)

    def test_bad_planner_result_type(self) -> None:
        """Garbage planner_result must not raise — _count_gpt_items handles it."""
        result = decide_compound_fallback(
            transcript="burger",
            planner_result="not_a_real_plan",
            local_planner_result=None,
            unsafe_slot_reason=None,
            valid_candidates_count=0,
            unresolved_spans=[],
        )
        assert isinstance(result, CompoundFallbackDecision)

    def test_exception_in_decide_falls_back(self) -> None:
        """If _decide raises internally, the outer guard returns FALLBACK_REPEAT_FIRST_ITEM."""
        class _Evil:
            @property
            def validated_plan(self):
                raise RuntimeError("boom")
            @property
            def items(self):
                raise RuntimeError("boom")

        result = decide_compound_fallback(
            transcript="test",
            planner_result=_Evil(),
            local_planner_result=None,
            unsafe_slot_reason=None,
            valid_candidates_count=0,
            unresolved_spans=[],
        )
        # Exception in _count_gpt_items is caught, falls through to Rule 4
        assert isinstance(result, CompoundFallbackDecision)

    def test_unresolved_spans_none_ok(self) -> None:
        result = decide_compound_fallback(
            transcript="burger",
            planner_result=None,
            local_planner_result=None,
            unsafe_slot_reason=None,
            valid_candidates_count=0,
            unresolved_spans=None,  # type: ignore[arg-type]
        )
        assert isinstance(result, CompoundFallbackDecision)

    def test_negative_reprompt_count(self) -> None:
        """Negative reprompt count is treated as < 1 → EXECUTE_VALID_PLAN (single-item path)."""
        result = decide_compound_fallback(
            transcript="burger",
            planner_result=None,
            local_planner_result=None,
            unsafe_slot_reason="multi_item_slots",
            valid_candidates_count=0,
            unresolved_spans=[],
            reprompt_count=-1,
        )
        assert result == CompoundFallbackDecision.EXECUTE_VALID_PLAN


# ---------------------------------------------------------------------------
# Integration: real-world utterances that must NOT trigger "one at a time"
# ---------------------------------------------------------------------------


class TestRealWorldUtterances:
    """Regression tests for the original bug report."""

    @pytest.mark.parametrize("transcript", [
        "grilled chicken sandwich with small coke",
        "burger with fries",
        "tuna melt with mayo",
        "6 piece wings with buffalo sauce",
        "cheeseburger no onions",
    ])
    def test_item_with_option_never_one_at_a_time(self, transcript: str) -> None:
        """Multi-item reason + 'with/no' marker → EXECUTE_VALID_PLAN, not clarification."""
        result = _call(
            transcript=transcript,
            planner_result=None,
            unsafe_slot_reason="multi_item_slots",
            valid_candidates_count=0,
        )
        assert result == CompoundFallbackDecision.EXECUTE_VALID_PLAN
        assert result != CompoundFallbackDecision.FALLBACK_ONE_AT_A_TIME
        assert result != CompoundFallbackDecision.FALLBACK_REPEAT_FIRST_ITEM

    def test_truly_ambiguous_triggers_repeat_first(self) -> None:
        """Genuinely unclear compound without item-option marker → ask for first item.

        On first encounter (reprompt=0) the policy falls through to single-item
        path.  On the second encounter (reprompt=1) it asks "what's the first item?".
        """
        first_encounter = _call(
            transcript="a burger fries rings shake",
            unsafe_slot_reason="multi_item_slots",
            valid_candidates_count=0,
            reprompt_count=0,
        )
        assert first_encounter == CompoundFallbackDecision.EXECUTE_VALID_PLAN

        second_encounter = _call(
            transcript="a burger fries rings shake",
            unsafe_slot_reason="multi_item_slots",
            valid_candidates_count=0,
            reprompt_count=1,
        )
        assert second_encounter == CompoundFallbackDecision.FALLBACK_REPEAT_FIRST_ITEM

    def test_repeated_failure_escalates(self) -> None:
        """Same unclear compound after 2 failures → one at a time."""
        result = _call(
            transcript="a burger fries rings shake",
            unsafe_slot_reason="multi_item_slots",
            valid_candidates_count=0,
            reprompt_count=2,
        )
        assert result == CompoundFallbackDecision.FALLBACK_ONE_AT_A_TIME

    def test_partial_success_preserved(self) -> None:
        """One item resolved + unresolved span → partial execute."""
        result = _call(
            transcript="a chicken sandwich and dragon pasta",
            valid_candidates_count=1,
            unresolved_spans=["dragon pasta"],
        )
        assert result == CompoundFallbackDecision.EXECUTE_PARTIAL_AND_CLARIFY

    def test_large_fries_small_coke_safe_slots_execute(self) -> None:
        """Clean compound with safe slots → EXECUTE_VALID_PLAN even without "with"."""
        result = _call(
            transcript="large fries and small coke",
            unsafe_slot_reason=None,  # safe slots
        )
        assert result == CompoundFallbackDecision.EXECUTE_VALID_PLAN
