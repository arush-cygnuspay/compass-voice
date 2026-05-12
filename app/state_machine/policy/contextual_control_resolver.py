# app/state_machine/policy/contextual_control_resolver.py
"""Contextual control-intent resolver.

Resolves ambiguous user turns whose correct interpretation depends on *what
the bot just said* (last_prompt_type) rather than purely on the NLU output.

Four production failures this fixes
-------------------------------------
1. IDLE + "No. That's it for now."  → UNKNOWN  → intent_not_allowed
   Now: coerces to CHECKOUT → confirm_order_summary

2. IDLE + "No. I don't want anything else."  → CANCEL_ORDER  → intent_not_allowed
   Now: coerces to CHECKOUT → confirm_order_summary

3. CONFIRMING_ORDER + "You got your card." → UNKNOWN → confirm_order_summary_unclear
   Now: coerces to PAYMENT_STATUS → payment_not_started

4. CONFIRMING_ORDER + "The code." → UNKNOWN → confirm_order_summary_unclear
   Now: coerces to PAYMENT_STATUS → payment_not_started

Design
------
- Pure function — no side effects, no shared state.
- Only fires when feature flag COMPASS_CONTEXTUAL_CONTROL_V2=1 (default: on).
- Required-item states are guarded: resolver returns NONE for
  WAITING_FOR_SIDE / MODIFIER / SIZE / SIDE_SIZE / QUANTITY / CONFIRMING_ITEM.
- Result carries kind + source + reason for structured logging.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from app.nlu.intent_resolution.intent import Intent
from app.nlu.prompt_type import PromptType
from app.nlu.query_normalization.text_preprocessor import normalize_text
from app.state_machine.models.conversation_state import ConversationState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

class ContextualControlKind(str, Enum):
    NONE = "none"
    FINISH_ADDING = "finish_adding"        # done adding items → proceed to checkout
    CHECKOUT_REQUEST = "checkout_request"  # explicit checkout/payment phrase
    PAYMENT_STATUS_QUERY = "payment_status_query"  # user thinks they already paid


@dataclass(frozen=True, slots=True)
class ContextualControlDecision:
    kind: ContextualControlKind
    source: str
    confidence: float = 1.0
    reason: Optional[str] = None


_DECISION_NONE = ContextualControlDecision(
    kind=ContextualControlKind.NONE,
    source="none",
    confidence=0.0,
)


# ---------------------------------------------------------------------------
# Guard: states where an active required item step is in progress.
# The resolver must not interfere with these — checkout phrases in
# WAITING_FOR_MODIFIER etc. are handled by their own handlers (as DONE).
# ---------------------------------------------------------------------------

_REQUIRED_ITEM_STEP_STATES: frozenset[ConversationState] = frozenset({
    ConversationState.WAITING_FOR_SIDE,
    ConversationState.WAITING_FOR_MODIFIER,
    ConversationState.WAITING_FOR_SIZE,
    ConversationState.WAITING_FOR_SIDE_SIZE,
    ConversationState.WAITING_FOR_QUANTITY,
    ConversationState.CONFIRMING_ITEM,
})


# ---------------------------------------------------------------------------
# Intents that look like "cancel" but are frequently misfired on
# "done adding" phrases by the NLU model.
# ---------------------------------------------------------------------------

_CANCEL_MISFIRE_INTENTS: frozenset[Intent] = frozenset({
    Intent.CANCEL_ORDER,
    Intent.DENY,
    Intent.CANCEL,
})


# ---------------------------------------------------------------------------
# Exact-match phrase set for "I'm done adding items" utterances.
#
# These phrases reliably mean "please proceed to checkout / order review"
# even when NLU fires CANCEL_ORDER, DENY, or UNKNOWN.  The guard condition
# (IDLE + cart non-empty + last_prompt_type = ANYTHING_ELSE) ensures we
# only apply this interpretation when the bot just asked "Anything else?".
#
# All entries are in normalize_text() form (lowercase, stripped punctuation).
# ---------------------------------------------------------------------------

_FINISH_ADDING_EXACT: frozenset[str] = frozenset({
    # "no" prefix variants (NLU fires DENY or UNKNOWN for these)
    "no thats it",
    "no that is it",
    "no thats all",
    "no that is all",
    "no thats it for now",
    "no that is it for now",
    "no thats all for now",
    "no that is all for now",
    "no nothing else",
    "no no more",
    "no more items",
    "no additional items",
    "no i dont want anything else",
    "no i do not want anything else",
    "no i dont want more",
    "no i do not want more",
    # "I don't want" variants (NLU fires CANCEL_ORDER for these)
    "i dont want anything else",
    "i do not want anything else",
    "dont want anything else",
    "do not want anything else",
    "i dont want more",
    "i do not want more",
    "i dont want anything more",
    "i do not want anything more",
    "dont want more",
    # "that's it for now" — not in existing _IDLE_CHECKOUT_EXACT
    "thats it for now",
    "that is it for now",
    "thats all for now",
    "that is all for now",
    # "I think that's all" etc.
    "i think thats all",
    "i think that is all",
    "i think thats it",
    "i think that is it",
    "i think im good",
    "i think i am good",
    "i think were good",
    "i think we are good",
})


# ---------------------------------------------------------------------------
# Payment-status phrases in CONFIRMING_ORDER context.
#
# These signal "I believe I already paid / have you received payment?" —
# they should map to payment_not_started (order hasn't started yet).
# All entries are in normalize_text() form.
# ---------------------------------------------------------------------------

_PAYMENT_STATUS_CONFIRMING_ORDER: frozenset[str] = frozenset({
    # Card / payment received queries
    "you got your card",
    "you got the card",
    "got your card",
    "got the card",
    "got your payment",
    "you got payment",
    "did you get my payment",
    "did you get the payment",
    "did you receive payment",
    "have you received payment",
    "you received payment",
    "received payment",
    "payment received",
    # Already paid
    "i paid",
    "i have paid",
    "already paid",
    "i already paid",
    "payment is done",
    "payment done",
    "i paid already",
    # Payment code / QR — "the code" is ambiguous but in CONFIRMING_ORDER
    # context it means "I have the payment code"
    "the code",
    "qr code",
    "qr",
    "the qr",
    "the qr code",
    "the payment code",
    "payment code",
    "i have the code",
    "i got the code",
    "here is the code",
    "heres the code",
    "here the code",
})


# ---------------------------------------------------------------------------
# Public resolver
# ---------------------------------------------------------------------------

def resolve_contextual_control(
    *,
    state: ConversationState,
    last_prompt_type: str | None,
    cart_has_items: bool,
    normalized_text: str,
    intent: Intent,
) -> ContextualControlDecision:
    """Resolve contextual control intent.

    Args:
        state:            Current FSM state.
        last_prompt_type: Value of session.last_prompt_type (PromptType.value).
        cart_has_items:   True when the session cart is non-empty.
        normalized_text:  Normalized user utterance from NLU.
        intent:           Effective intent after all upstream coercions.

    Returns:
        :class:`ContextualControlDecision` with kind=NONE when no contextual
        override applies.
    """
    # ── Guard: mid-item required-step states — never interfere ────────────
    if state in _REQUIRED_ITEM_STEP_STATES:
        return _DECISION_NONE

    text = (normalize_text(normalized_text) or "").strip()
    if not text:
        return _DECISION_NONE

    # ── Rule A: IDLE + ANYTHING_ELSE context + finish-adding phrase ────────
    #
    # When the bot just asked "Anything else?" (after item_added_successfully)
    # and the user says something that means "no thanks I'm done", NLU often
    # fires CANCEL_ORDER, DENY, or UNKNOWN instead of END_ADDING/CHECKOUT.
    # Coerce to CHECKOUT so the normal IDLE → CONFIRMING_ORDER flow activates.
    if (
        state == ConversationState.IDLE
        and cart_has_items
        and last_prompt_type == PromptType.ANYTHING_ELSE.value
    ):
        # Sub-case A1: exact finish-adding phrase
        if text in _FINISH_ADDING_EXACT:
            _log("finish_adding_exact_match", state=state, intent=intent, text=text)
            return ContextualControlDecision(
                kind=ContextualControlKind.FINISH_ADDING,
                source="finish_adding_exact",
                confidence=1.0,
                reason=f"exact_match:{text!r}",
            )

        # Sub-case A2: CANCEL_ORDER misfire on "done adding" intent.
        # If the NLU fires CANCEL_ORDER and there is no explicit cancel
        # signal in the text, treat it as FINISH_ADDING.  Explicit cancel
        # phrases ("cancel my order", "cancel everything") have content words
        # that prevent them from landing here.
        if intent in _CANCEL_MISFIRE_INTENTS and _is_done_like_not_cancel(text):
            _log("cancel_misfire_coerced", state=state, intent=intent, text=text)
            return ContextualControlDecision(
                kind=ContextualControlKind.FINISH_ADDING,
                source="cancel_misfire_guard",
                confidence=0.85,
                reason="cancel_intent_on_done_like_text",
            )

    # ── Rule B: CONFIRMING_ORDER + payment-status phrase ──────────────────
    #
    # "You got your card", "The code", "I paid" in CONFIRMING_ORDER context
    # means the user believes payment is done / is asking about payment status.
    # Coerce to PAYMENT_STATUS so ConfirmOrderHandler returns payment_not_started.
    if state == ConversationState.CONFIRMING_ORDER:
        if text in _PAYMENT_STATUS_CONFIRMING_ORDER:
            _log("payment_status_exact_match", state=state, intent=intent, text=text)
            return ContextualControlDecision(
                kind=ContextualControlKind.PAYMENT_STATUS_QUERY,
                source="payment_status_exact",
                confidence=1.0,
                reason=f"exact_match:{text!r}",
            )

    return _DECISION_NONE


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

# Words that only appear in explicit cancel requests, never in finish-adding.
_EXPLICIT_CANCEL_WORDS: frozenset[str] = frozenset({
    "cancel",
    "everything",
    "entire",
    "whole",
    "order",
    "start",
    "over",
    "begin",
    "again",
    "restart",
})


def _is_done_like_not_cancel(text: str) -> bool:
    """True when *text* looks like "I'm done" not "cancel my order".

    The CANCEL_ORDER misfire guard uses this to distinguish:
    - "I don't want anything else" (finish-adding) → True
    - "cancel my order" (explicit cancel) → False
    - "cancel everything" → False
    """
    words = set(text.lower().split())
    # If any hard-cancel word is present, this is a real cancel request.
    if words & _EXPLICIT_CANCEL_WORDS:
        return False
    # Short utterances without cancel words are finish-adding signals.
    return True


def _log(event: str, *, state: ConversationState, intent: Intent, text: str) -> None:
    logger.debug(
        "contextual_control_resolved",
        extra={
            "event": event,
            "state": state.value,
            "intent": intent.value if hasattr(intent, "value") else str(intent),
            "text": text,
        },
    )
