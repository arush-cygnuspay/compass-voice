# app/services/compound_turn_policy.py
"""CompoundTurnPolicy — decide the fallback action for compound/multi-item turns.

Fixes the over-eager "repeat one at a time" message that was firing for valid
compound requests such as "burger with fries" or "chicken sandwich with small coke".

The policy is consulted AFTER the GPT and local planners have had their first
chance, and AFTER the unsafe-slot guard has determined whether the NLU slots
look broken.  It makes the final call on whether to:

  EXECUTE_VALID_PLAN         — planners or single-item path can handle this.
  EXECUTE_PARTIAL_AND_CLARIFY — partial success; keep valid items, note gaps.
  ASK_ABOUT_UNRESOLVED       — one item resolved; ask about the unresolved span.
  FALLBACK_REPEAT_FIRST_ITEM — couldn't parse safely; ask "what's the first item?"
  FALLBACK_ONE_AT_A_TIME     — repeated failure; escalate to one-at-a-time prompt.

Decision rules (in priority order)
------------------------------------
1. GPT planner has ≥1 valid item                  → EXECUTE_VALID_PLAN
2. Local planner has ≥1 valid item + unresolved   → EXECUTE_PARTIAL_AND_CLARIFY
3. Local planner has ≥1 valid item, all resolved  → EXECUTE_VALID_PLAN
4. Slots are safe (no broken reason)              → EXECUTE_VALID_PLAN
5. Broken reason is recoverable
   (low_confidence, size_word_inside_item, etc.)  → EXECUTE_VALID_PLAN
6. Transcript has item+option marker ("with",
   "no", "extra", …) and broken reason is
   multi-item family                              → EXECUTE_VALID_PLAN
7. reprompt_count ≥ 2 (repeated failure)          → FALLBACK_ONE_AT_A_TIME
8. reprompt_count ≥ 1 (second encounter)          → FALLBACK_REPEAT_FIRST_ITEM
9. Default (first encounter, reprompt_count=0)    → EXECUTE_VALID_PLAN (let
   single-item path try before asking for first item)

Design
------
* Pure function — no I/O, no side effects, never raises.
* Conservative: prefers falling through to single-item path over blocking.
* "one at a time" is a last resort reserved for repeated failures on
  clearly ambiguous utterances.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Sequence


class CompoundFallbackDecision(str, Enum):
    """Action the AddItemHandler should take when a compound turn is ambiguous."""

    EXECUTE_VALID_PLAN = "execute_valid_plan"
    """Planner succeeded or slots are safe — proceed normally."""

    EXECUTE_PARTIAL_AND_CLARIFY = "execute_partial_and_clarify"
    """Some items resolved; execute valid items, note gaps."""

    ASK_ABOUT_UNRESOLVED = "ask_about_unresolved"
    """One item resolved; ask specifically about the unresolved span."""

    FALLBACK_REPEAT_FIRST_ITEM = "fallback_repeat_first_item"
    """Couldn't parse safely — ask: 'What's the first item?'"""

    FALLBACK_ONE_AT_A_TIME = "fallback_one_at_a_time"
    """Escalated after repeated failure — 'Please say one item at a time.'"""


# ---------------------------------------------------------------------------
# Item-option markers: phrases that suggest "item + modifier/side", not
# two separate items.
# ---------------------------------------------------------------------------
_ITEM_OPTION_MARKERS: tuple[str, ...] = (
    " with ",
    " no ",
    " without ",
    " extra ",
    " on the side",
    " hold the ",
    " add ",
    " light ",
    " easy on the ",
)

# Modifier-only prefixes at the start of an utterance (e.g. "no onions")
_MODIFIER_PREFIX_PATTERNS: tuple[str, ...] = (
    "no ",
    "without ",
    "extra ",
    "hold the ",
    "light ",
)

# Broken-slot reasons that are typically recoverable: the single-item path
# handles them correctly and produces a sensible response.
_RECOVERABLE_REASONS: frozenset[str] = frozenset({
    "low_confidence_add_item",
    "size_word_inside_item",
    "numeric_piece_variant",
    "multi_variant_slots",
})

# Broken-slot reasons that signal genuinely separate items in the utterance.
_MULTI_ITEM_REASONS: frozenset[str] = frozenset({
    "multi_item_slots",
    "long_compound_add_item",
    "merged_item_slot",
})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def decide_compound_fallback(
    transcript: str,
    planner_result: Any,
    local_planner_result: Any,
    unsafe_slot_reason: "str | None",
    valid_candidates_count: int,
    unresolved_spans: "Sequence[str]",
    *,
    reprompt_count: int = 0,
) -> CompoundFallbackDecision:
    """Determine the fallback action for a compound/multi-item ADD_ITEM turn.

    Parameters
    ----------
    transcript:
        Normalized user utterance text.
    planner_result:
        GPT planner result object (None if unavailable or disabled).
        Expected to have `validated_plan.items` when populated.
    local_planner_result:
        ParsedMultiItemPlan from plan_multi_item_order().  May be None.
    unsafe_slot_reason:
        Reason code from slot_pairing_looks_broken(), or None when safe.
    valid_candidates_count:
        Number of items the local planner successfully resolved (0 or 1
        when the handler calls this — the ≥2 case is handled upstream).
    unresolved_spans:
        Spans the local planner could not match to any menu item.
    reprompt_count:
        How many times the "what's the first item?" prompt has already
        been shown for compound failures in this session.  At ≥2 the
        policy escalates to FALLBACK_ONE_AT_A_TIME.

    Returns
    -------
    CompoundFallbackDecision

    Never raises.
    """
    try:
        return _decide(
            transcript=transcript,
            planner_result=planner_result,
            local_planner_result=local_planner_result,
            unsafe_slot_reason=unsafe_slot_reason,
            valid_candidates_count=valid_candidates_count,
            unresolved_spans=list(unresolved_spans or ()),
            reprompt_count=reprompt_count,
        )
    except Exception:
        return CompoundFallbackDecision.FALLBACK_REPEAT_FIRST_ITEM


def looks_like_item_with_option(transcript: str) -> bool:
    """Return True when the utterance likely describes one item + modifier/side.

    Examples that return True
    -------------------------
    - "burger with fries"
    - "tuna melt with mayo"
    - "chicken sandwich no onions"
    - "6 piece wings with buffalo sauce"
    - "cheeseburger no onions"

    Examples that return False
    --------------------------
    - "burger fries rings" (no connector)
    - "i want a burger and a coke" (hard compound — two items)
    - "burger" (single item)
    """
    text = (transcript or "").strip().lower()
    return _has_item_option_marker(text)


# ---------------------------------------------------------------------------
# Internal implementation
# ---------------------------------------------------------------------------


def _decide(
    transcript: str,
    planner_result: Any,
    local_planner_result: Any,
    unsafe_slot_reason: "str | None",
    valid_candidates_count: int,
    unresolved_spans: list[str],
    reprompt_count: int,
) -> CompoundFallbackDecision:
    text = (transcript or "").strip().lower()

    # ── Rule 1: GPT planner has valid items → execute them ────────────────
    if planner_result is not None and _count_gpt_items(planner_result) >= 1:
        return CompoundFallbackDecision.EXECUTE_VALID_PLAN

    # ── Rule 2/3: Local planner resolved items ────────────────────────────
    if valid_candidates_count >= 1:
        if unresolved_spans:
            return CompoundFallbackDecision.EXECUTE_PARTIAL_AND_CLARIFY
        return CompoundFallbackDecision.EXECUTE_VALID_PLAN

    # ── Rule 4: Slots are safe → single-item path will handle it ─────────
    if not unsafe_slot_reason:
        return CompoundFallbackDecision.EXECUTE_VALID_PLAN

    # ── Rule 5: Recoverable broken-slot reason ────────────────────────────
    # low_confidence, size_word_inside_item, numeric_piece_variant, etc. are
    # all handled better by the single-item path than by a clarification prompt.
    if unsafe_slot_reason in _RECOVERABLE_REASONS:
        return CompoundFallbackDecision.EXECUTE_VALID_PLAN

    # ── Rule 6: "with/no/extra/…" marker → item + option, not two items ──
    if unsafe_slot_reason in _MULTI_ITEM_REASONS and _has_item_option_marker(text):
        return CompoundFallbackDecision.EXECUTE_VALID_PLAN

    # ── Rule 7: Repeated compound failure → escalate to one-at-a-time ────
    if reprompt_count >= 2:
        return CompoundFallbackDecision.FALLBACK_ONE_AT_A_TIME

    # ── Rule 8/9: Default — ask on second encounter; fall through on first ─
    # On the very first encounter (reprompt_count=0) let the single-item path
    # try before showing a clarification prompt.  On the second encounter
    # (reprompt_count=1) ask "what's the first item?".
    if reprompt_count >= 1:
        return CompoundFallbackDecision.FALLBACK_REPEAT_FIRST_ITEM
    return CompoundFallbackDecision.EXECUTE_VALID_PLAN


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _has_item_option_marker(text: str) -> bool:
    """Return True when text contains a marker suggesting item + modifier/side."""
    return any(m in text for m in _ITEM_OPTION_MARKERS)


def _count_gpt_items(planner_result: Any) -> int:
    """Count validated items in a GPT planner result object.

    Compatible with GptAddItemPlanResult, SmartTurnPlan, and any other
    plan object that has a validated_plan.items or items attribute.
    """
    try:
        # Try validated_plan.items (GptAddItemPlanResult pattern)
        validated_plan = getattr(planner_result, "validated_plan", None)
        if validated_plan is not None:
            items = getattr(validated_plan, "items", ()) or ()
            return len(items)
        # Fallback: direct items attribute (SmartTurnPlan pattern)
        items = getattr(planner_result, "items", ()) or ()
        return len(items)
    except Exception:
        return 0
