# app/nlu/turn_resolver/control_flow_policy.py
"""Trigger policy for Priority-6 control-flow GPT buckets.

Decides whether to invoke one of three checkout/payment/order-type GPT
buckets for a given turn.  Returns (should_call, task_mode, reason).

Buckets covered
---------------
* Bucket 5 — checkout_confirmation_resolution
  Triggers when state is idle/confirming_order and the utterance looks
  like a checkout-intent phrase or is ambiguous after order review.

* Bucket 6 — order_type_change / pickup_delivery_initial
  Triggers when state is waiting_for_order_type (initial) or any active
  ordering state and the utterance contains a pickup/delivery phrase.

* Bucket payment — payment_permission_resolution
  Triggers when state is waiting_for_pickup_sms_permission and the
  utterance looks like a payment preference phrase.

Safety contract
---------------
* Pure functions only — no GPT calls, no state mutation.
* Never raises.
* Returns (False, "", ...) in all no-trigger cases.
"""
from __future__ import annotations

import re

from app.nlu.turn_resolver.prompt_registry import (
    TASK_CHECKOUT_CONFIRMATION_RESOLUTION,
    TASK_PAYMENT_PERMISSION_RESOLUTION,
    TASK_PICKUP_DELIVERY_INITIAL,
    TASK_ORDER_TYPE_CHANGE,
)

# ── States by bucket ──────────────────────────────────────────────────────────

# States where checkout confirmation resolution applies.
_CHECKOUT_STATES: frozenset[str] = frozenset({
    "idle",
    "confirming_order",
})

# States where payment permission resolution applies.
_PAYMENT_PERMISSION_STATES: frozenset[str] = frozenset({
    "waiting_for_pickup_sms_permission",
    "waiting_for_payment_permission",
})

# States where a mid-order type switch may be requested.
# (Mirrors _ALLOWED_STATES in order_type_service.py.)
_ORDER_TYPE_CHANGEABLE_STATES: frozenset[str] = frozenset({
    "idle",
    "greeting",
    "confirming_item",
    "waiting_for_size",
    "waiting_for_side",
    "waiting_for_side_size",
    "waiting_for_modifier",
    "waiting_for_quantity",
    "confirming_order",
    "modifying_item",
    "removing_item",
    "waiting_for_delivery_eligibility",
    "waiting_for_delivery_address_collection",
    "cancellation_confirmation",
})

# State for initial pickup-vs-delivery selection.
_INITIAL_ORDER_TYPE_STATE: str = "waiting_for_order_type"

# Terminal/payment-completed states — never trigger any bucket.
_TERMINAL_STATES: frozenset[str] = frozenset({
    "completed",
    "transferring_to_human_agent",
    "error_recovery",
    "waiting_for_payment",
    "waiting_for_checkout_completion",
})

# ── Phrase patterns ───────────────────────────────────────────────────────────

# Checkout / order-confirmation phrases.
_CHECKOUT_PHRASES_RE = re.compile(
    r"\b("
    r"that'?s?\s+it"
    r"|that\s+is\s+it"
    r"|\ball\s+done\b"
    r"|\bdone\b"
    r"|\bcheckout\b"
    r"|check\s+out"
    r"|place\s+(the\s+)?order"
    r"|submit\s+(the\s+)?order"
    r"|yeah\s+do\s+it"
    r"|yes\s+do\s+it"
    r"|go\s+ahead"
    r"|confirm\s+(the\s+)?order"
    r"|finish(\s+up)?"
    r"|i'?m\s+done"
    r"|i'?m\s+ready"
    r"|keep\s+ordering"
    r"|add\s+more"
    r"|not\s+yet"
    r")\b",
    re.IGNORECASE,
)

# Payment-link permission phrases.
_PAYMENT_PHRASES_RE = re.compile(
    r"\b("
    r"send\s+(the\s+)?link"
    r"|text\s+me(\s+the\s+link)?"
    r"|send\s+(it|me)\b"
    r"|no\s+payment\s+link"
    r"|no\s+link"
    r"|pay\s+(at|when|on|there|in|cash)\b"
    r"|pay\s+there"
    r"|pay\s+on\s+arrival"
    r"|i'?ll?\s+pay\s+when"
    r"|no\s+i'?ll?\s+pay"
    r"|yeah\s+do\s+it"
    r"|yes(\s+please)?"
    r"|\bno(\s+thanks?)?\b"
    r")\b",
    re.IGNORECASE,
)

# Order-type change phrases (mid-order pickup/delivery switch).
_ORDER_TYPE_CHANGE_RE = re.compile(
    r"\b("
    r"make\s+it\s+(delivery|pickup|a\s+pickup|a\s+delivery)"
    r"|switch\s+to\s+(delivery|pickup)"
    r"|change\s+to\s+(delivery|pickup)"
    r"|i\s+want\s+(delivery|pickup)"
    r"|want\s+(delivery|pickup)"
    r"|i'?ll?\s+come\s+get\s+it"
    r"|i'?ll?\s+pick\s+it\s+up"
    r"|i'?ll?\s+pick\s+up"
    r"|actually\s+pickup"
    r"|actually\s+pick\s+up"
    r"|for\s+pickup"
    r"|for\s+delivery"
    r"|delivery\s+please"
    r"|pickup\s+please"
    r"|bring\s+it\s+to\s+me"
    r"|send\s+it\s+to\s+me"
    r"|to\s+my\s+(house|home|door|address|apartment|place)"
    r"|don'?t\s+deliver"
    r"|no\s+delivery"
    r")\b"
    r"|\b(delivery|pickup|pick\s+up)\b",
    re.IGNORECASE,
)

# Initial order-type selection phrases (for waiting_for_order_type).
_INITIAL_ORDER_TYPE_RE = re.compile(
    r"\b("
    r"pickup|pick\s+up|pick\s+it\s+up"
    r"|delivery|deliver"
    r"|i'?ll?\s+come\s+get\s+it"
    r"|bring\s+it"
    r"|send\s+it"
    r"|i'?ll?\s+pick"
    r"|for\s+(pickup|delivery)"
    r"|carryout|carry\s+out"
    r"|takeout|take\s+out"
    r")\b",
    re.IGNORECASE,
)

# ── Policy threshold ──────────────────────────────────────────────────────────

_LOW_CONFIDENCE_THRESHOLD: float = 0.70


# ── Public API ────────────────────────────────────────────────────────────────


def should_call_control_flow_gpt(
    *,
    state: str,
    user_text: str,
    normalized_text: str,
    local_intent: str | None,
    local_confidence: float | None,
    previous_assistant_prompt: str | None = None,
    context: object = None,
) -> tuple[bool, str, str]:
    """Decide whether to invoke a control-flow GPT bucket.

    Parameters
    ----------
    state:
        Current FSM state string.
    user_text:
        Raw customer utterance.
    normalized_text:
        Lowercase / punctuation-stripped utterance.
    local_intent:
        NLU-detected intent (may be None).
    local_confidence:
        NLU confidence score (may be None).
    previous_assistant_prompt:
        Last bot message — used to disambiguate "yeah do it" context.
    context:
        ConversationContext (duck-typed, read-only; reserved for future use).

    Returns
    -------
    (True, task_mode, reason)  — bucket should fire; task_mode is a PromptRegistry constant.
    (False, "", reason)         — bucket must NOT fire.

    Decision order (first match wins)
    ----------------------------------
    1.  Terminal state                     → False
    2.  Empty text                         → False
    3.  Initial order type selection       → pickup_delivery_initial
    4.  Payment permission state + phrase  → payment_permission_resolution
    5.  Order type change phrase           → order_type_change
    6.  Checkout state + checkout phrase   → checkout_confirmation_resolution
    7.  Confirming_order + low confidence  → checkout_confirmation_resolution
    8.  Otherwise                          → False
    """
    state_lower = (state or "").lower().strip()
    text = (normalized_text or user_text or "").strip()

    # 1. Never trigger in terminal states.
    if state_lower in _TERMINAL_STATES:
        return False, "", "terminal_state"

    # 2. Never trigger on empty / silence.
    if not text:
        return False, "", "empty_text"

    # 3. Initial order type selection.
    if state_lower == _INITIAL_ORDER_TYPE_STATE:
        if _INITIAL_ORDER_TYPE_RE.search(text):
            return True, TASK_PICKUP_DELIVERY_INITIAL, "initial_order_type_phrase"
        if _is_low_confidence(local_confidence):
            return True, TASK_PICKUP_DELIVERY_INITIAL, "low_confidence_order_type"
        return False, "", "no_trigger"

    # 4. Payment permission resolution.
    if state_lower in _PAYMENT_PERMISSION_STATES:
        if _PAYMENT_PHRASES_RE.search(text):
            return True, TASK_PAYMENT_PERMISSION_RESOLUTION, "payment_phrase"
        if _is_low_confidence(local_confidence):
            return True, TASK_PAYMENT_PERMISSION_RESOLUTION, "low_confidence_payment"
        return False, "", "no_trigger"

    # 5. Order type change phrase (any changeable state).
    if state_lower in _ORDER_TYPE_CHANGEABLE_STATES:
        if _ORDER_TYPE_CHANGE_RE.search(text):
            return True, TASK_ORDER_TYPE_CHANGE, "order_type_phrase"

    # 6. Checkout confirmation phrase in idle / confirming_order.
    if state_lower in _CHECKOUT_STATES:
        if _CHECKOUT_PHRASES_RE.search(text):
            return True, TASK_CHECKOUT_CONFIRMATION_RESOLUTION, "checkout_phrase"
        # 7. Low confidence in confirming_order — intent is ambiguous.
        if state_lower == "confirming_order" and _is_low_confidence(local_confidence):
            return True, TASK_CHECKOUT_CONFIRMATION_RESOLUTION, "low_confidence_confirming"

    return False, "", "no_trigger"


# ── Private helpers ───────────────────────────────────────────────────────────


def _is_low_confidence(local_confidence: float | None) -> bool:
    """Return True when local NLU confidence is below the trigger threshold."""
    return float(local_confidence or 0.0) < _LOW_CONFIDENCE_THRESHOLD
