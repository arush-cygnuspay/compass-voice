# app/services/smart_turn_policy.py
"""SmartTurnPolicy — trigger detection and plan validation for SmartTurnPlanner.

Answers two questions:
  1. should_use_smart_planner() → (bool, reason_str)
     Should the SmartTurnPlanner be invoked for this turn?

  2. validate_smart_plan() → ValidationResult
     Is the plan safe to apply to the live FSM path?

Design principles
-----------------
* Policy is purely function-based — no class state, no side effects.
* All functions return immediately and never raise.
* The planner is triggered conservatively: local FSM handles everything;
  SmartTurnPlanner is consulted only when the local path is likely to fail
  AND the cost of a wrong decision is high (repeated reprompt, wrong item).
* validate_smart_plan() is strict: when in doubt it returns is_safe=False
  so the local FSM path runs unchanged.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.smart_turn_planner import SmartTurnPlan

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Trigger thresholds
# ---------------------------------------------------------------------------

# Minimum local-NLU confidence below which we consider the prediction risky.
_LOW_CONFIDENCE_THRESHOLD = 0.55

# Minimum confidence required in the SmartTurnPlan for safe application.
_MIN_PLAN_CONFIDENCE = 0.75

# Maximum menu items to pass as context (prevents oversized payloads).
MAX_MENU_CONTEXT_ITEMS = 12

# ---------------------------------------------------------------------------
# Patterns that signal a self-correction utterance
# ---------------------------------------------------------------------------

_CORRECTION_PREFIXES: tuple[str, ...] = (
    "no i said",
    "no i meant",
    "no i want",
    "actually",
    "scratch that",
    "cancel that",
    "i meant",
    "i mean",
    "not that",
    "change that to",
    "change it to",
    "make it",
    "instead",
    "wait",
)

_COMPOUND_SIGNALS: tuple[str, ...] = (
    " with ",
    " and ",
    " plus ",
    " also ",
    " as well",
)

# States where SmartTurnPlanner adds value.
# Values must match ConversationState.value (all lowercase).
_ELIGIBLE_STATES: frozenset[str] = frozenset({
    "idle",
    "waiting_for_modifier",
    "waiting_for_side",
    "waiting_for_side_size",
    "confirming_item",
})


# ---------------------------------------------------------------------------
# should_use_smart_planner
# ---------------------------------------------------------------------------

def should_use_smart_planner(
    transcript: str,
    state: str,
    local_intent: str,
    local_confidence: float,
) -> tuple[bool, str]:
    """Return (should_invoke: bool, reason: str) for the current turn.

    Parameters
    ----------
    transcript:
        Normalized customer utterance.
    state:
        Current FSM state name.
    local_intent:
        Intent predicted by local NLU.
    local_confidence:
        Confidence score (0.0–1.0) from local NLU model.

    Returns
    -------
    (True, reason) when the SmartTurnPlanner should be consulted.
    (False, reason) when local FSM is sufficient.
    """
    # Normalize state to lowercase so callers may pass either
    # ConversationState.value ("idle") or the legacy uppercase form ("IDLE").
    state = (state or "").lower()

    # Always skip for terminal / payment states.
    # Values must match ConversationState.value (all lowercase).
    if state in {
        "completed",
        "error_recovery",
        "transferring_to_human_agent",
        "waiting_for_payment",
        "waiting_for_checkout_completion",
        "confirming_order",
        "cancellation_confirmation",
    }:
        return False, "terminal_or_payment_state"

    # Only eligible states
    if state not in _ELIGIBLE_STATES:
        return False, "state_not_eligible"

    text = (transcript or "").strip().lower()
    if not text:
        return False, "empty_transcript"

    # 1. Self-correction signal
    if _has_correction_signal(text):
        return True, "correction_phrase"

    # 2. Compound utterance in IDLE / CONFIRMING_ITEM (add-item with modifier + side
    #    in one breath — the local parser may drop modifiers or sides)
    if state in {"idle", "confirming_item"} and _is_compound_utterance(text):
        return True, "compound_utterance"

    # 3. In WAITING_FOR_MODIFIER or WAITING_FOR_SIDE: the customer says something
    #    that doesn't look like a valid option answer AND confidence is low.
    if state in {
        "waiting_for_modifier",
        "waiting_for_side",
        "waiting_for_side_size",
    } and local_confidence < _LOW_CONFIDENCE_THRESHOLD:
        return True, "low_confidence_waiting_state"

    # 4. Local NLU confidence is very low on an actionable intent in IDLE
    if (
        state == "idle"
        and local_intent == "ADD_ITEM"
        and local_confidence < _LOW_CONFIDENCE_THRESHOLD
    ):
        return True, "low_confidence_add_item"

    return False, "local_path_sufficient"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Result of validate_smart_plan()."""

    is_safe: bool
    reason: str
    block_reason: str | None = None   # machine-readable when is_safe=False


def validate_smart_plan(
    plan: "SmartTurnPlan",
    *,
    menu_context: list[str],
    cart_snapshot: list[str],
    state: str,
    local_intent: str,
    trigger_reason: str = "",
) -> ValidationResult:
    """Return whether the plan is safe to apply to the live FSM flow.

    Parameters
    ----------
    plan:
        The SmartTurnPlan returned by plan_smart_turn().
    menu_context:
        List of relevant menu item names (compact, not full menu).
    cart_snapshot:
        Current cart item names.
    state:
        FSM state at the time the plan was produced.
    local_intent:
        Local NLU predicted intent.
    trigger_reason:
        Why the planner was invoked (for logging).

    Safety gates (all must pass for is_safe=True)
    -----------------------------------------------
    1. plan.gpt_called must be True (not a skipped/error placeholder)
    2. plan.decision must be actionable ("add_items" or "correction")
    3. plan.confidence ≥ _MIN_PLAN_CONFIDENCE
    4. For "add_items": items[] must be non-empty
    5. For "add_items": each item_name must appear in menu_context
       (fuzzy: substring match, case-insensitive)
    6. For "correction": correction field must be set
    7. State must still be in _ELIGIBLE_STATES
    """
    # Normalize state to lowercase so callers may pass either
    # ConversationState.value ("idle") or the legacy uppercase form ("IDLE").
    state = (state or "").lower()

    if not plan.gpt_called:
        return ValidationResult(
            is_safe=False,
            reason="gpt_not_called",
            block_reason="gpt_not_called",
        )

    if plan.decision not in {"add_items", "correction"}:
        return ValidationResult(
            is_safe=False,
            reason=f"non_actionable_decision:{plan.decision}",
            block_reason="non_actionable_decision",
        )

    if plan.confidence < _MIN_PLAN_CONFIDENCE:
        return ValidationResult(
            is_safe=False,
            reason=f"low_confidence:{plan.confidence:.2f}",
            block_reason="low_confidence",
        )

    # Validate that side size fields are safe strings (not injected values)
    for item in getattr(plan, "items", ()):
        for side in getattr(item, "sides", ()):
            size_val = getattr(side, "size", None)
            variant_val = getattr(side, "variant", None)
            if size_val is not None and not isinstance(size_val, str):
                return ValidationResult(is_safe=False, reason="side_size_not_string", block_reason="side_size_not_string")
            if variant_val is not None and not isinstance(variant_val, str):
                return ValidationResult(is_safe=False, reason="side_variant_not_string", block_reason="side_variant_not_string")
            # Clamp side size to safe length
            if size_val and len(size_val) > 50:
                return ValidationResult(is_safe=False, reason="side_size_too_long", block_reason="side_size_too_long")

    if state not in _ELIGIBLE_STATES:
        return ValidationResult(
            is_safe=False,
            reason="state_changed_to_ineligible",
            block_reason="state_changed_to_ineligible",
        )

    if plan.decision == "add_items":
        if not plan.items:
            return ValidationResult(
                is_safe=False,
                reason="add_items_but_no_items",
                block_reason="add_items_but_no_items",
            )
        # Gate 5: every resolved item_name must fuzzy-match something in menu_context.
        # If menu_context is empty we skip this gate — the caller didn't provide context.
        if menu_context:
            menu_lower = [n.lower() for n in menu_context]
            for item in plan.items:
                item_lower = item.item_name.lower()
                if not any(
                    item_lower in m or m in item_lower
                    for m in menu_lower
                ):
                    return ValidationResult(
                        is_safe=False,
                        reason=f"item_not_in_menu_context:{item.item_name!r}",
                        block_reason="item_not_in_menu_context",
                    )

    elif plan.decision == "correction":
        if plan.correction is None:
            return ValidationResult(
                is_safe=False,
                reason="correction_decision_but_no_correction_field",
                block_reason="missing_correction_field",
            )

    return ValidationResult(
        is_safe=True,
        reason="all_gates_passed",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CHECKOUT_PHRASES: tuple[str, ...] = (
    "that's it",
    "that's all",
    "i'm done",
    "i am done",
    "done ordering",
    "nothing else",
    "no more",
    "checkout",
    "check out",
    "place my order",
    "submit my order",
)


def _has_correction_signal(text: str) -> bool:
    """Return True when the utterance starts with a known correction prefix."""
    stripped = text.strip()
    for prefix in _CORRECTION_PREFIXES:
        if stripped.startswith(prefix):
            return True
    return False


def _is_checkout_phrase(text: str) -> bool:
    """Return True when the utterance contains a checkout/done phrase."""
    stripped = text.strip()
    return any(phrase in stripped for phrase in _CHECKOUT_PHRASES)


def _is_compound_utterance(text: str) -> bool:
    """Return True when the utterance contains compound markers (with/and/plus)
    combined with food-related context — heuristic, not NLU-based."""
    for signal in _COMPOUND_SIGNALS:
        if signal in text:
            return True
    # Also detect commas used as list separators between food items
    # ("burger, fries, coke")
    if text.count(",") >= 1:
        return True
    return False


def determine_smart_task_mode(
    transcript: str,
    state: str,
    local_intent: str,
    local_confidence: float,
) -> str | None:
    """Return the best task_mode string for this turn, or None if not applicable.

    Task modes and their priorities
    --------------------------------
    1. "correction"         — correction phrase detected (highest priority)
    2. "modifier_selection" — state is WAITING_FOR_MODIFIER
    3. "side_selection"     — state is WAITING_FOR_SIDE or WAITING_FOR_SIDE_SIZE
    4. "checkout"           — checkout phrase detected in IDLE/CONFIRMING
    5. "compound_add_item"  — compound markers in IDLE/CONFIRMING
    6. "generic_repair"     — low-confidence catch-all for active ordering states

    Returns None when no task mode is appropriate (policy returns False).
    """
    # Normalize state to lowercase so callers may pass either
    # ConversationState.value ("idle") or the legacy uppercase form ("IDLE").
    state = (state or "").lower()
    text = (transcript or "").strip().lower()

    # Correction takes priority over everything else
    if _has_correction_signal(text):
        return "correction"

    # State-specific waiting modes
    if state == "waiting_for_modifier":
        return "modifier_selection"
    if state in {"waiting_for_side", "waiting_for_side_size"}:
        return "side_selection"

    # Checkout phrases in IDLE / CONFIRMING_ITEM
    if state in {"idle", "confirming_item"} and _is_checkout_phrase(text):
        return "checkout"

    # Compound add-item in IDLE / CONFIRMING_ITEM
    if state in {"idle", "confirming_item"} and _is_compound_utterance(text):
        return "compound_add_item"

    # Low-confidence catch-all for active ordering states
    if local_confidence < _LOW_CONFIDENCE_THRESHOLD:
        return "generic_repair"

    return None


def build_menu_context_for_turn(
    *,
    item_names: list[str],
    max_items: int = MAX_MENU_CONTEXT_ITEMS,
) -> list[str]:
    """Return a clamped list of item names safe to include in the GPT payload.

    Caller should pass the resolved candidate item names from the menu query,
    NOT the full menu.  This function clips to *max_items* to keep payload
    size bounded.
    """
    return list(item_names)[:max_items]
