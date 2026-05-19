# app/logging/final_decision_resolver.py
"""Compute the final-decision section of a canonical TurnEvent JSONL record.

Answers:
  - Which system made the final call? (final_source)
  - Was a repair actually applied?     (repair_applied, repair_type)
  - Did the intent change from local?  (intent_changed)
  - Did any slot change from local?    (slots_changed)
  - Why was this decision made?        (decision_reason)

Design principles
-----------------
* Pure functions — no I/O, no side effects, never raises.
* Operates on the existing TurnEvent dataclass from app/diagnostics/turn_event.py.
* import-safe: all imports are TYPE_CHECKING-guarded.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.diagnostics.turn_event import TurnEvent


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class FinalDecision:
    """Computed final-decision metadata for one turn."""

    final_intent: str | None
    """The intent that was actually used to drive the response."""

    final_source: str
    """Which system provided the final decision.

    Values (mutually exclusive, priority order):
      smart_planner  — SmartTurnPlanner result was applied.
      gpt_repair     — GptRepairService result was applied.
      local_nlu      — Local FSM / NLU output used unchanged.
      fallback       — A generic fallback response was emitted.
    """

    repair_applied: bool
    """True when any external system (GPT / planner) changed the live outcome."""

    repair_type: str
    """Fine-grained repair classification.

    Values:
      not_invoked           — no repair system was called.
      no_repair             — called but outcome unchanged (same intent, same slots).
      intent_repair         — intent was changed by repair system.
      slot_repair           — slots were changed, intent same.
      intent_and_slot_repair — both intent and slots changed.
      cart_action_repair    — cart action (add/remove/modify) was injected.
      fallback_repair       — fallback response was injected by repair system.
    """

    intent_changed: bool
    """True when the final intent differs from the local NLU intent."""

    slots_changed: bool
    """True when any slot correction was applied by the repair system."""

    response_key: str
    """Final response_key emitted to the caller."""

    decision_reason: str
    """Human-readable one-liner explaining why this decision was made."""


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------

# Response keys that indicate a pure fallback turn
_FALLBACK_RESPONSE_KEYS: frozenset[str] = frozenset({
    "intent_not_allowed",
    "item_not_found",
    "item_context_missing",
    "invalid_input",
    "generic_fallback",
    "ask_for_clarification",
    "payment_not_confirmed_yet",
    "payment_verification_error",
    "checkout_link_send_failed",
    "payment_link_send_failed",
    "payment_link_unavailable_now",
    "confirmation_state_error",
})


def resolve_final_decision(event: "TurnEvent") -> FinalDecision:
    """Return the computed FinalDecision for *event*.

    Never raises.  Returns a safe default if any field is missing.
    """
    try:
        return _resolve(event)
    except Exception:
        return FinalDecision(
            final_intent=None,
            final_source="local_nlu",
            repair_applied=False,
            repair_type="not_invoked",
            intent_changed=False,
            slots_changed=False,
            response_key=getattr(event, "response_key", "") or "",
            decision_reason="resolver_error",
        )


def _resolve(event: "TurnEvent") -> FinalDecision:
    response_key: str = getattr(event, "response_key", "") or ""
    gpt_applied: bool = bool(getattr(event, "gpt_applied", False))
    gpt_called: bool = bool(getattr(event, "gpt_called", False))
    gpt_decision: str | None = getattr(event, "gpt_decision", None)
    fallback_triggered: bool = bool(getattr(event, "fallback_triggered", False))

    # Local NLU intent (before any repair)
    local_intent: str | None = (
        getattr(event, "local_intent_before_gpt", None)
        or getattr(event, "pred_intent", None)
    )

    # Final intent after any repair
    final_intent_raw: str | None = getattr(event, "final_intent_after_gpt", None)
    if final_intent_raw:
        final_intent = final_intent_raw
    elif gpt_applied and getattr(event, "gpt_selected_intent", None):
        final_intent = event.gpt_selected_intent  # type: ignore[union-attr]
    else:
        final_intent = local_intent

    # Slot corrections
    slot_corrections_json: str | None = getattr(event, "gpt_slot_corrections_json", None)
    slots_changed = bool(
        slot_corrections_json and slot_corrections_json.strip() not in ("", "[]", "null")
    )

    # Intent changed?
    intent_changed = bool(
        final_intent
        and local_intent
        and final_intent != local_intent
    )

    # ── Determine final_source ────────────────────────────────────────────
    # SmartTurnPlanner is future; we check gpt_repair first.
    # When slot has smart_planner fields wired, extend here.
    if gpt_applied:
        final_source = "gpt_repair"
    elif response_key in _FALLBACK_RESPONSE_KEYS or (
        fallback_triggered and not gpt_called
    ):
        final_source = "fallback"
    else:
        final_source = "local_nlu"

    repair_applied = gpt_applied

    # ── Determine repair_type ─────────────────────────────────────────────
    if not gpt_called and not repair_applied:
        repair_type = "not_invoked"
    elif gpt_decision == "fallback" and (
        getattr(event, "gpt_fallback_type", "none") != "none"
    ):
        repair_type = "fallback_repair"
    elif not repair_applied:
        # GPT was called but not applied → no_repair
        repair_type = "no_repair"
    elif intent_changed and slots_changed:
        repair_type = "intent_and_slot_repair"
    elif intent_changed:
        repair_type = "intent_repair"
    elif slots_changed:
        repair_type = "slot_repair"
    else:
        # Applied but nothing visibly different — treat as no_repair
        repair_type = "no_repair"

    # ── Decision reason ───────────────────────────────────────────────────
    apply_reason: str | None = getattr(event, "gpt_apply_reason", None)
    if apply_reason:
        decision_reason = apply_reason
    elif gpt_called and not repair_applied:
        decision_reason = "gpt_called_not_applied"
    elif not gpt_called:
        decision_reason = "local_nlu_only"
    else:
        decision_reason = "repair_applied"

    return FinalDecision(
        final_intent=final_intent,
        final_source=final_source,
        repair_applied=repair_applied,
        repair_type=repair_type,
        intent_changed=intent_changed,
        slots_changed=slots_changed,
        response_key=response_key,
        decision_reason=decision_reason,
    )
