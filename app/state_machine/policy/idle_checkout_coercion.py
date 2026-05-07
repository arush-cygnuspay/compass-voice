# app/state_machine/policy/idle_checkout_coercion.py
"""Idle-state checkout coercion policy.

When the NLU model fires UNKNOWN (intent confidence below INTENT_MIN_CONF)
while the FSM is IDLE and the cart is non-empty, and the user's text is a
recognisable checkout / done / payment phrase, this policy coerces the
intent to Intent.CHECKOUT so the normal IDLE → CONFIRMING_ORDER path
activates (StartOrderHandler → confirm_order_summary).

Critical invariant
------------------
Payment phrases coerced here land on ``confirm_order_summary`` **first**.
The payment flow starts only after the user explicitly confirms the order
summary in CONFIRMING_ORDER state.  This policy never routes directly to
WAITING_FOR_PAYMENT — that decision belongs to the confirming-order handler.

Design
------
- Pure function ``coerce_idle_to_checkout`` — no side effects, no shared
  state, deterministic, cheap (one classifier call + one semantic check).
- Only fires in IDLE with a non-empty cart.
- Never touches WAITING_FOR_SIDE / WAITING_FOR_MODIFIER / WAITING_FOR_SIZE;
  those states must see their own handler logic, not checkout coercion.
- Never coerces already-resolved checkout-family intents (handled upstream
  by FlowGate._apply_idle_shortcuts).
- Bare yes/no utterances are not coerced (classifier excludes them).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.nlu.control_phrase_classifier import DEFAULT_CLASSIFIER
from app.nlu.intent_resolution.intent import Intent
from app.nlu.intent_resolution.intent_result import IntentResult
from app.state_machine.models.conversation_state import ConversationState
from app.state_machine.semantic_signals import is_done_like_response


# Intents already handled correctly by FlowGate._apply_idle_shortcuts.
# Coercion must skip these to avoid double-processing.
_ALREADY_HANDLED_INTENTS: frozenset[Intent] = frozenset({
    Intent.CHECKOUT,
    Intent.CONFIRM_ORDER,
    Intent.FINISH_ORDER,
    Intent.END_ADDING,
    Intent.PAYMENT_REQUEST,
    Intent.REVIEW_ORDER,
    Intent.START_ORDER,
    Intent.DENY,
})


@dataclass(frozen=True, slots=True)
class IdleCoercionResult:
    """Return type of :func:`coerce_idle_to_checkout`.

    ``coercion_reason`` is ``None`` when no coercion was applied.
    """
    intent_result: IntentResult
    coercion_reason: str | None


def coerce_idle_to_checkout(
    *,
    state: ConversationState,
    intent_result: IntentResult,
    nlu: Any,
    cart: Any,
) -> IdleCoercionResult:
    """Coerce UNKNOWN checkout-phrases to :attr:`Intent.CHECKOUT` in IDLE.

    Args:
        state:         Current FSM state.
        intent_result: Resolved intent (may be UNKNOWN after confidence gate).
        nlu:           :class:`~app.nlu.nlu_result.NLUResult`; only
                       ``normalized_text`` and ``intent_confidence`` are read.
        cart:          Session cart object; only ``is_empty()`` is called.

    Returns:
        :class:`IdleCoercionResult` with ``coercion_reason=None`` when no
        coercion was applied.
    """
    _no_op = IdleCoercionResult(intent_result=intent_result, coercion_reason=None)

    # ── Rule 1: Only IDLE state ───────────────────────────────────────────
    if state != ConversationState.IDLE:
        return _no_op

    # ── Rule 2: Nothing to check out — leave it alone ─────────────────────
    if cart.is_empty():
        return _no_op

    intent = intent_result.intent

    # ── Rule 3: Existing shortcuts already handle these intents ───────────
    # FlowGate._apply_idle_shortcuts converts them to START_ORDER when cart
    # is non-empty, which then routes to confirm_order_summary.  No need to
    # double-process.
    if intent in _ALREADY_HANDLED_INTENTS:
        return _no_op

    # ── Rule 4: Only coerce UNKNOWN — other resolved intents own their flow ─
    if intent != Intent.UNKNOWN:
        return _no_op

    # ─── From here: state=IDLE, cart non-empty, intent=UNKNOWN ────────────

    text = (getattr(nlu, "normalized_text", "") or "").strip()
    if not text:
        return _no_op

    # ── Rule 5: Control-phrase classifier says this is a checkout phrase ───
    # The classifier was trained with exact-match sets for IDLE state; it
    # excludes bare yes/no and ambiguous single words.
    cpc_result = DEFAULT_CLASSIFIER.classify(text, state="idle")
    if cpc_result.action == "checkout":
        return IdleCoercionResult(
            intent_result=IntentResult(
                intent=Intent.CHECKOUT,
                raw_text=intent_result.raw_text,
            ),
            coercion_reason=f"idle_checkout_classifier:{cpc_result.reason}",
        )

    # ── Rule 6: Semantic done-like signal ─────────────────────────────────
    # Catches ASR phrase variations (e.g. "go ahead and check out") that
    # semantic_signals.DONE_WORDS covers but may not yet be in the
    # classifier's exact-match set.  The classifier (Rule 5) is preferred;
    # this is a belt-and-braces fallback.
    if is_done_like_response(Intent.UNKNOWN, text):
        return IdleCoercionResult(
            intent_result=IntentResult(
                intent=Intent.CHECKOUT,
                raw_text=intent_result.raw_text,
            ),
            coercion_reason="idle_done_phrase_with_cart",
        )

    return _no_op
