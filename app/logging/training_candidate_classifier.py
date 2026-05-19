# app/logging/training_candidate_classifier.py
"""Classify whether a turn is a useful training candidate for future NLU models.

A turn is a training candidate when one or more quality signals indicate the
local NLU model might have made a mistake, an edge-case was hit, or a human
reviewer could extract a ground-truth label.

Design principles
-----------------
* Pure functions — no I/O, no side effects, never raises.
* Returns a frozen TrainingClassification dataclass.
* Operates on TurnEvent + FinalDecision (both from the logging layer).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.diagnostics.turn_event import TurnEvent
    from app.logging.final_decision_resolver import FinalDecision


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

# Turns with local NLU confidence below this are candidate labelling targets.
_LOW_CONFIDENCE_THRESHOLD: float = 0.60

# Response keys that indicate a generic or repeated-prompt fallback
_FALLBACK_RESPONSE_KEYS: frozenset[str] = frozenset({
    "repeat_side_options",
    "repeat_modifier_options",
    "repeat_size_options",
    "repeat_side_size_options",
    "invalid_size_option",
    "invalid_side_size_option",
    "invalid_quantity_option",
    "item_not_found",
    "intent_not_allowed",
    "item_context_missing",
    "invalid_input",
    "generic_fallback",
    "ask_for_clarification",
    "list_side_options",
    "list_modifier_options",
})

# Response keys that represent reprompt escalation (user repeated after reprompt)
_REPROMPT_RESPONSE_KEYS: frozenset[str] = frozenset({
    "repeat_side_options",
    "repeat_modifier_options",
    "repeat_size_options",
    "repeat_side_size_options",
    "invalid_size_option",
    "invalid_side_size_option",
    "required_side_cannot_skip",
    "required_modifier_cannot_skip",
    "required_size_cannot_skip",
    "required_side_size_cannot_skip",
})

# Correction/cancel phrases that indicate intent ambiguity
_CORRECTION_PHRASE_KEYS: frozenset[str] = frozenset({
    "correction_applied",
    "correction",
    "intent_correction",
    "cancel",
    "CANCEL",
})


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class TrainingClassification:
    """Result of the training candidate classifier for one turn."""

    candidate: bool
    """True when this turn should be considered for training/labelling."""

    candidate_reasons: list[str]
    """Non-empty list of signal codes when candidate=True.

    Codes:
      local_low_confidence    — local NLU confidence < threshold.
      gpt_changed_intent      — GPT/repair changed the intent.
      gpt_changed_slots       — GPT/repair changed slots.
      planner_applied         — SmartTurnPlanner result was applied.
      validator_rejected      — plan validator rejected the GPT/planner output.
      fallback_used           — generic fallback response emitted.
      item_not_found          — item was off-menu or not resolved.
      correction_turn         — utterance contained a self-correction phrase.
      repeated_reprompt       — user was reprompted more than once for same field.
      no_state_progress       — state did not change after a meaningful utterance.
    """

    label_status: str
    """One of: unlabeled | auto_labeled | needs_human_review."""

    needs_human_review: bool
    """True when the turn is complex enough to warrant human annotation."""


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

def classify_training_candidate(
    event: "TurnEvent",
    final_decision: "FinalDecision",
) -> TrainingClassification:
    """Return a TrainingClassification for *event*.

    Never raises.  Returns a safe default if any field is missing.
    """
    try:
        return _classify(event, final_decision)
    except Exception:
        return TrainingClassification(
            candidate=False,
            candidate_reasons=[],
            label_status="unlabeled",
            needs_human_review=False,
        )


def _classify(
    event: "TurnEvent",
    final_decision: "FinalDecision",
) -> TrainingClassification:
    reasons: list[str] = []

    # 1. Low local NLU confidence
    confidence: float | None = (
        getattr(event, "local_intent_confidence_before_gpt", None)
        or getattr(event, "pred_intent_confidence", None)
    )
    if confidence is not None and confidence < _LOW_CONFIDENCE_THRESHOLD:
        reasons.append("local_low_confidence")

    # 2. GPT / SmartTurnPlanner changed intent
    if final_decision.intent_changed:
        reasons.append("gpt_changed_intent")

    # 3. GPT changed slots
    if final_decision.slots_changed:
        reasons.append("gpt_changed_slots")

    # 4. Planner was applied (future hook; currently no planner_applied flag in TurnEvent)
    # NOTE: When SmartTurnPlanner result is wired to TurnEvent, add check here.

    # 5. Validator rejected output
    add_item_has_blocking_warnings: bool = bool(
        getattr(event, "add_item_has_blocking_warnings", False)
    )
    if add_item_has_blocking_warnings:
        reasons.append("validator_rejected")

    # 6. Fallback response used
    response_key: str = getattr(event, "response_key", "") or ""
    fallback_triggered: bool = bool(getattr(event, "fallback_triggered", False))
    if response_key in _FALLBACK_RESPONSE_KEYS or fallback_triggered:
        reasons.append("fallback_used")

    # 7. Item not found / off-menu
    if response_key == "item_not_found" or (
        response_key in {"item_context_missing", "intent_not_allowed"}
        and getattr(event, "state_before", "") in ("idle", "confirming_item")
    ):
        reasons.append("item_not_found")

    # 8. Correction / cancel turn
    pred_intent: str = getattr(event, "pred_intent", "") or ""
    normalized_text: str = getattr(event, "normalized_text", "") or ""
    _correction_signals = (
        "actually", "scratch that", "cancel that", "i meant", "no i said",
        "change that", "make it", "instead",
    )
    if pred_intent in _CORRECTION_PHRASE_KEYS or any(
        normalized_text.lower().startswith(sig) for sig in _correction_signals
    ):
        reasons.append("correction_turn")

    # 9. Repeated reprompt (reprompt_count > 1)
    reprompt_count: int = int(getattr(event, "reprompt_count", 0) or 0)
    if reprompt_count > 1 or response_key in _REPROMPT_RESPONSE_KEYS:
        reasons.append("repeated_reprompt")

    # 10. No state progress after meaningful input
    state_before: str = getattr(event, "state_before", "") or ""
    state_after: str = getattr(event, "state_after", "") or ""
    meaningful_text = bool(normalized_text and len(normalized_text.split()) >= 2)
    if meaningful_text and state_before == state_after and response_key in _FALLBACK_RESPONSE_KEYS:
        reasons.append("no_state_progress")

    candidate = bool(reasons)
    needs_human_review = len(reasons) >= 2 or "validator_rejected" in reasons

    return TrainingClassification(
        candidate=candidate,
        candidate_reasons=reasons,
        label_status="unlabeled",
        needs_human_review=needs_human_review,
    )
