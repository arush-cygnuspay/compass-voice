# app/services/order_type_service.py
"""Order type change service — detect, validate, and apply pickup/delivery switching.

Allows customers to switch between pickup and delivery at any point before payment
is sent or the order is submitted.

Detection priority
------------------
1. OrderTypeResolver (existing deterministic lexical matcher) — fast, O(k) phrases.
   Includes a negation check: "don't deliver it" → pickup rather than delivery.
2. Supplementary substring patterns — phrases not covered by OrderTypeResolver
   (e.g. "to my house", "switch to delivery").
3. SmartTurnPlan semantic hint — checks `reason` / `response` fields of an
   optional SmartTurnPlan for pickup/delivery keywords.

Public API
----------
detect_order_type_change(transcript, state, context, smart_plan=None)
    → OrderTypeChangeResult | None

set_order_type(order_type, context) → None          [mutating; call after detection]
validate_order_type_change(order_type, context, *, state, ...) → OrderTypeChangeResult
build_order_type_response(result) → FlowDecision

Design principles
-----------------
* Pure validation — no LLM calls, no I/O.
* set_order_type() is the only mutating function; it touches only order-type fields.
* Existing pending item / side / modifier state is never touched.
* Always returns a safe result; never raises.
* cart data, full menu, PII are not sent to logs.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, TYPE_CHECKING

from app.state_machine.common.order_type_resolver import OrderTypeResolver

if TYPE_CHECKING:
    from app.services.conversation_flow_policy import FlowDecision

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result codes
# ---------------------------------------------------------------------------

class OrderTypeChangeCode(str, Enum):
    """Outcome code for an order type change request."""

    OK                      = "ok"
    DELIVERY_ADDRESS_REQUIRED = "delivery_address_required"
    DELIVERY_NOT_AVAILABLE  = "delivery_not_available"
    DELIVERY_OUT_OF_RADIUS  = "delivery_out_of_radius"
    PAYMENT_ALREADY_SENT    = "payment_already_sent"
    ORDER_ALREADY_SUBMITTED = "order_already_submitted"
    INVALID_ORDER_TYPE      = "invalid_order_type"


@dataclass(frozen=True, slots=True)
class OrderTypeChangeResult:
    """Immutable result of an order type detection + validation pass.

    All fields are populated and safe to log directly.

    Logging fields (for turn_events.jsonl)
    ----------------------------------------
    order_type_before         Previous order_type value.
    order_type_after          New order_type value (None when blocked).
    change_detected           True when a change request was found in the utterance.
    change_source             "phrase_match" | "smart_planner" | "local_intent" | "none".
    order_type_change_result  code.value — short machine-readable outcome.
    delivery_address_required True when delivery was chosen but address is not yet collected.
    delivery_available        False when the store does not support delivery.
    blocked_reason            Non-empty string when code is not OK.
    """

    code: OrderTypeChangeCode
    detected_order_type: str | None    # "pickup" | "delivery" | None
    order_type_before: str | None
    order_type_after: str | None
    change_detected: bool
    change_source: str                  # phrase_match | smart_planner | local_intent | none
    delivery_address_required: bool
    delivery_available: bool
    blocked_reason: str
    response_text: str
    response_key: str
    target_state: str | None           # Suggested next state; None = no state change


# ---------------------------------------------------------------------------
# Phrase supplementary patterns
# (OrderTypeResolver already covers the core vocabulary — see that module)
# ---------------------------------------------------------------------------

# Extra PICKUP substrings not in OrderTypeResolver
_EXTRA_PICKUP_SUBSTRINGS: tuple[str, ...] = (
    "make it pickup",
    "make it a pickup",
    "switch to pickup",
    "change to pickup",
    "i want pickup",
    "want pickup",
    "ill come get it",      # "I'll come get it" after apostrophe removal
    "actually pick it up",
    "gonna pick it up",
    "just pick it up",
)

# Extra DELIVERY substrings not in OrderTypeResolver
_EXTRA_DELIVERY_SUBSTRINGS: tuple[str, ...] = (
    "to my house",
    "to my home",
    "to my place",
    "to my door",
    "to my address",
    "to my apartment",
    "make it delivery",
    "make it a delivery",
    "switch to delivery",
    "change to delivery",
    "i want delivery",
    "want delivery",
)

# Words that negate the following order-type phrase.
# "don't deliver it" → resolver sees "deliver" (delivery) → negation flips to pickup.
_NEGATION_WORDS: frozenset[str] = frozenset({
    "dont", "no", "not", "never", "without", "cancel", "stop",
})

_VALID_ORDER_TYPES: frozenset[str] = frozenset({"pickup", "delivery"})

# States where a mid-order switch is accepted
_ALLOWED_STATES: frozenset[str] = frozenset({
    "idle", "greeting",
    "confirming_item",
    "waiting_for_size", "waiting_for_side", "waiting_for_side_size",
    "waiting_for_modifier", "waiting_for_quantity",
    "confirming_order",
    "modifying_item", "removing_item",
    "waiting_for_order_type",
    "waiting_for_delivery_eligibility",
    "waiting_for_delivery_address_collection",
    "cancellation_confirmation",
})

# States blocked because payment is already in flight
_PAYMENT_STATES: frozenset[str] = frozenset({
    "waiting_for_payment",
    "waiting_for_checkout_completion",
    "waiting_for_pickup_sms_permission",
})

# Terminal states where the order cannot be modified at all
_TERMINAL_STATES: frozenset[str] = frozenset({
    "completed",
    "transferring_to_human_agent",
    "error_recovery",
})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_order_type_change(
    transcript: str,
    state: str,
    context: Any,
    smart_plan: Any = None,
) -> OrderTypeChangeResult | None:
    """Detect a mid-order pickup/delivery switch in a customer utterance.

    Returns None when no order type change is detected.
    Returns a full, validated OrderTypeChangeResult when detected.

    The result tells the caller what to do next; the caller is responsible
    for calling set_order_type() to apply the change to the context.

    Detection runs in three priority tiers:
    1. OrderTypeResolver (primary — fastest, most reliable)
       with negation check ("don't deliver it" → pickup).
    2. Supplementary substring patterns.
    3. SmartTurnPlan semantic hint (``smart_plan.reason`` / ``.response``).
    """
    try:
        return _detect_and_validate(
            transcript=transcript,
            state=(state or "").lower().strip(),
            context=context,
            smart_plan=smart_plan,
        )
    except Exception as exc:
        logger.warning(
            "order_type_service.detect_error",
            extra={"error": f"{type(exc).__name__}: {exc}"},
        )
        return None


def set_order_type(order_type: str, context: Any) -> None:
    """Mutate the context to reflect the new order type.

    Pickup
        Sets delivery_address_required = False.
        Resets delivery_address (clears collected/confirmed flags).
        Sets onboarding_complete = True.

    Delivery
        Sets delivery_address_required = True.
        Resets delivery_address only if not already collected (preserves re-entry).
        Sets onboarding_complete = False (address collection still needed).

    Pending item / side / modifier / queue state is NOT touched.
    """
    try:
        order_type = (order_type or "").lower().strip()
        if order_type not in _VALID_ORDER_TYPES:
            logger.warning(
                "order_type_service.set_order_type_invalid",
                extra={"order_type": order_type},
            )
            return

        context.order_type = order_type
        context.delivery_address_confirmed = False

        if order_type == "pickup":
            context.delivery_address_required = False
            context.onboarding_complete = True
            try:
                context.delivery_address.reset_for_new_delivery()
            except AttributeError:
                pass

        else:  # "delivery"
            context.delivery_address_required = True
            addr = getattr(context, "delivery_address", None)
            already_collected = bool(getattr(addr, "collected", False))
            if not already_collected:
                context.onboarding_complete = False
                try:
                    context.delivery_address.reset_for_new_delivery()
                except AttributeError:
                    pass

    except Exception as exc:
        logger.warning(
            "order_type_service.set_order_type_error",
            extra={"error": f"{type(exc).__name__}: {exc}"},
        )


def validate_order_type_change(
    order_type: str,
    context: Any,
    *,
    state: str = "",
    change_source: str = "none",
    order_type_before: str | None = None,
) -> OrderTypeChangeResult:
    """Validate whether switching to the requested order type is currently allowed.

    Can be called independently of detect_order_type_change when the caller
    already knows the requested order type (e.g. from a local NLU intent).

    Parameters
    ----------
    order_type : str
        Requested order type ("pickup" or "delivery").
    context : ConversationContext
        Current conversation context (read-only in this call).
    state : str
        Current ConversationState value string (lowercase).
    change_source : str
        How the request was detected ("phrase_match" | "smart_planner" | ...).
    order_type_before : str | None
        Previous order_type for logging; derived from context if None.
    """
    try:
        return _validate(
            order_type=(order_type or "").lower().strip(),
            context=context,
            state=(state or "").lower().strip(),
            change_source=change_source,
            order_type_before=order_type_before,
        )
    except Exception as exc:
        logger.warning(
            "order_type_service.validate_error",
            extra={"error": f"{type(exc).__name__}: {exc}"},
        )
        return _error_result(
            detected=order_type,
            before=order_type_before,
            source=change_source,
            reason="validation_exception",
        )


def build_order_type_response(result: OrderTypeChangeResult) -> FlowDecision:
    """Convert an OrderTypeChangeResult into a FlowDecision for the handler layer.

    Maps OrderTypeChangeCode → FlowAction and populates the response fields
    with logging metadata.  Never raises.
    """
    try:
        from app.services.conversation_flow_policy import FlowAction, FlowDecision

        action, target_state = _code_to_action(result)

        return FlowDecision(
            action=action,
            reason=f"order_type_{result.code.value}",
            response_text=result.response_text,
            response_key=result.response_key,
            target_state=target_state or result.target_state,
            metadata={
                "order_type_before": result.order_type_before,
                "order_type_after": result.order_type_after,
                "order_type_change_detected": result.change_detected,
                "order_type_change_source": result.change_source,
                "order_type_change_result": result.code.value,
                "delivery_address_required": result.delivery_address_required,
                "delivery_available": result.delivery_available,
                "blocked_reason": result.blocked_reason,
            },
        )
    except Exception as exc:
        logger.warning(
            "order_type_service.build_response_error",
            extra={"error": f"{type(exc).__name__}: {exc}"},
        )
        from app.services.conversation_flow_policy import FlowAction, FlowDecision
        return FlowDecision(
            action=FlowAction.FALLBACK_LOCAL,
            reason="order_type_response_error",
        )


# ---------------------------------------------------------------------------
# Internal — detection
# ---------------------------------------------------------------------------

def _detect_and_validate(
    *,
    transcript: str,
    state: str,
    context: Any,
    smart_plan: Any,
) -> OrderTypeChangeResult | None:
    """Run the three-tier detection pipeline."""
    order_type_before = getattr(context, "order_type", None)

    # ── Tier 1: OrderTypeResolver + negation check ────────────────────────
    resolver_match = OrderTypeResolver.resolve(transcript)
    if resolver_match is not None:
        effective_type = _apply_negation_check(
            transcript, resolver_match.matched_phrase, resolver_match.order_type
        )
        if effective_type is not None:
            return _validate(
                order_type=effective_type,
                context=context,
                state=state,
                change_source="phrase_match",
                order_type_before=order_type_before,
            )

    # ── Tier 2: Supplementary substring patterns ──────────────────────────
    detected = _extra_phrase_detect(transcript)
    if detected is not None:
        return _validate(
            order_type=detected,
            context=context,
            state=state,
            change_source="phrase_match",
            order_type_before=order_type_before,
        )

    # ── Tier 3: SmartTurnPlan semantic hint ───────────────────────────────
    if smart_plan is not None:
        semantic = _detect_from_smart_plan(smart_plan)
        if semantic is not None:
            return _validate(
                order_type=semantic,
                context=context,
                state=state,
                change_source="smart_planner",
                order_type_before=order_type_before,
            )

    return None


def _apply_negation_check(
    transcript: str,
    matched_phrase: str,
    detected_type: str,
) -> str | None:
    """If a negation word immediately precedes the matched phrase, flip the type.

    "don't deliver it" → resolver matches "deliver" (delivery) →
    "dont" is a negation before "deliver" → return "pickup".

    Returns None to signal "don't use this match" (extremely rare edge case).
    """
    # Normalize: lowercase + strip apostrophes
    norm = re.sub(r"['’‘]", "", transcript.lower())
    norm = re.sub(r"\s+", " ", norm).strip()

    phrase = re.sub(r"['’‘]", "", matched_phrase.lower())
    idx = norm.find(phrase)
    if idx <= 0:
        return detected_type  # nothing precedes → no negation

    before_words = norm[:idx].split()[-3:]  # up to 3 words before phrase
    if any(w in _NEGATION_WORDS for w in before_words):
        # Flip delivery → pickup and vice versa
        return "pickup" if detected_type == "delivery" else "delivery"

    return detected_type


def _extra_phrase_detect(transcript: str) -> str | None:
    """Check supplementary substrings not in OrderTypeResolver."""
    norm = re.sub(r"['’‘]", "", transcript.lower())
    norm = re.sub(r"\s+", " ", norm).strip()

    # Check delivery first (longer patterns, fewer false positives)
    for phrase in _EXTRA_DELIVERY_SUBSTRINGS:
        if phrase in norm:
            return "delivery"

    for phrase in _EXTRA_PICKUP_SUBSTRINGS:
        if phrase in norm:
            return "pickup"

    return None


def _detect_from_smart_plan(smart_plan: Any) -> str | None:
    """Extract pickup/delivery intent from SmartTurnPlan reason / response fields."""
    reason = str(getattr(smart_plan, "reason", "") or "").lower()
    response = str(getattr(smart_plan, "response", "") or "").lower()
    combined = f"{reason} {response}"

    _DELIVERY_KW = {"delivery", "deliver", "send it", "bring it", "to my house"}
    _PICKUP_KW = {"pickup", "pick up", "come get", "pick it up", "carryout", "carry out"}

    if any(kw in combined for kw in _DELIVERY_KW):
        return "delivery"
    if any(kw in combined for kw in _PICKUP_KW):
        return "pickup"

    return None


# ---------------------------------------------------------------------------
# Internal — validation
# ---------------------------------------------------------------------------

def _validate(
    *,
    order_type: str,
    context: Any,
    state: str,
    change_source: str,
    order_type_before: str | None,
) -> OrderTypeChangeResult:
    """Core validation gate.  Returns a fully-populated OrderTypeChangeResult."""

    # ── Invalid type ──────────────────────────────────────────────────────
    if order_type not in _VALID_ORDER_TYPES:
        return _blocked_result(
            code=OrderTypeChangeCode.INVALID_ORDER_TYPE,
            detected=order_type,
            before=order_type_before,
            source=change_source,
            reason="invalid_order_type",
            response_text="Sorry, I didn't catch that. Would you like pickup or delivery?",
            response_key="invalid_order_type",
        )

    # ── Terminal states ───────────────────────────────────────────────────
    if state in _TERMINAL_STATES:
        return _blocked_result(
            code=OrderTypeChangeCode.ORDER_ALREADY_SUBMITTED,
            detected=order_type,
            before=order_type_before,
            source=change_source,
            reason=f"terminal_state:{state}",
            response_text=(
                "Sorry, I can't change the order type — your order has already been completed. "
                "I can transfer you to our team if you need help."
            ),
            response_key="order_type_change_blocked_submitted",
        )

    # ── Payment-in-progress states ────────────────────────────────────────
    if state in _PAYMENT_STATES:
        return _blocked_result(
            code=OrderTypeChangeCode.PAYMENT_ALREADY_SENT,
            detected=order_type,
            before=order_type_before,
            source=change_source,
            reason=f"payment_state:{state}",
            response_text=(
                "Sorry, the payment link has already been sent. "
                "I can transfer you to our team if you need to change the order type."
            ),
            response_key="order_type_change_blocked_payment_sent",
        )

    # ── Context-level payment / submission flags ──────────────────────────
    if _payment_already_sent(context):
        return _blocked_result(
            code=OrderTypeChangeCode.PAYMENT_ALREADY_SENT,
            detected=order_type,
            before=order_type_before,
            source=change_source,
            reason="payment_link_sent_context",
            response_text=(
                "Sorry, the payment link has already been sent. "
                "I can transfer you to our team if you need to change the order type."
            ),
            response_key="order_type_change_blocked_payment_sent",
        )

    if _order_already_submitted(context):
        return _blocked_result(
            code=OrderTypeChangeCode.ORDER_ALREADY_SUBMITTED,
            detected=order_type,
            before=order_type_before,
            source=change_source,
            reason="order_submitted_context",
            response_text=(
                "Sorry, I can't change the order type — your order has already been placed."
            ),
            response_key="order_type_change_blocked_submitted",
        )

    # ── Delivery-specific checks ──────────────────────────────────────────
    delivery_available = bool(getattr(context, "delivery_available", True))

    if order_type == "delivery":
        if not delivery_available:
            return _blocked_result(
                code=OrderTypeChangeCode.DELIVERY_NOT_AVAILABLE,
                detected=order_type,
                before=order_type_before,
                source=change_source,
                reason="delivery_not_available",
                response_text=(
                    "Sorry, delivery isn't available for this store. "
                    "I can keep it for pickup."
                ),
                response_key="delivery_not_available",
                delivery_available=False,
            )

        addr = getattr(context, "delivery_address", None)

        # Address collected but area flagged as unserviceable
        area_serviceable = getattr(addr, "area_serviceable", None)
        if area_serviceable is False:
            return _blocked_result(
                code=OrderTypeChangeCode.DELIVERY_OUT_OF_RADIUS,
                detected=order_type,
                before=order_type_before,
                source=change_source,
                reason="delivery_out_of_radius",
                response_text=(
                    "Sorry, we don't deliver to that area. I can keep it as pickup."
                ),
                response_key="delivery_out_of_radius",
                delivery_address_required=True,
            )

        # Address not yet collected → ask for it
        address_collected = bool(getattr(addr, "collected", False))
        if not address_collected:
            return OrderTypeChangeResult(
                code=OrderTypeChangeCode.DELIVERY_ADDRESS_REQUIRED,
                detected_order_type=order_type,
                order_type_before=order_type_before,
                order_type_after="delivery",
                change_detected=True,
                change_source=change_source,
                delivery_address_required=True,
                delivery_available=True,
                blocked_reason="",
                response_text="Sure, I'll make it delivery. What's the delivery address?",
                response_key="ask_delivery_address",
                target_state="waiting_for_delivery_eligibility",
            )

        # Address already on file → straight switch
        return OrderTypeChangeResult(
            code=OrderTypeChangeCode.OK,
            detected_order_type=order_type,
            order_type_before=order_type_before,
            order_type_after="delivery",
            change_detected=True,
            change_source=change_source,
            delivery_address_required=False,
            delivery_available=True,
            blocked_reason="",
            response_text="Sure, I'll switch it to delivery.",
            response_key="order_type_changed_delivery",
            target_state=None,
        )

    # ── Pickup ────────────────────────────────────────────────────────────
    # If currently in address-collection flow, we must exit it.
    address_collection_states = {
        "waiting_for_delivery_address_collection",
        "waiting_for_delivery_eligibility",
    }
    target = "idle" if state in address_collection_states else None

    return OrderTypeChangeResult(
        code=OrderTypeChangeCode.OK,
        detected_order_type=order_type,
        order_type_before=order_type_before,
        order_type_after="pickup",
        change_detected=True,
        change_source=change_source,
        delivery_address_required=False,
        delivery_available=delivery_available,
        blocked_reason="",
        response_text="Got it, I'll make it pickup.",
        response_key="order_type_changed_pickup",
        target_state=target,
    )


# ---------------------------------------------------------------------------
# Internal — helpers
# ---------------------------------------------------------------------------

def _code_to_action(result: OrderTypeChangeResult) -> tuple[Any, str | None]:
    """Map OrderTypeChangeCode → (FlowAction, target_state_override)."""
    from app.services.conversation_flow_policy import FlowAction

    code = result.code
    if code == OrderTypeChangeCode.OK:
        return FlowAction.CHANGE_ORDER_TYPE, result.target_state
    if code == OrderTypeChangeCode.DELIVERY_ADDRESS_REQUIRED:
        return FlowAction.ASK_DELIVERY_ADDRESS, result.target_state
    # All blocking codes
    return FlowAction.REJECT_ORDER_TYPE_CHANGE, None


def _blocked_result(
    *,
    code: OrderTypeChangeCode,
    detected: str | None,
    before: str | None,
    source: str,
    reason: str,
    response_text: str,
    response_key: str,
    delivery_address_required: bool = False,
    delivery_available: bool = True,
) -> OrderTypeChangeResult:
    return OrderTypeChangeResult(
        code=code,
        detected_order_type=detected,
        order_type_before=before,
        order_type_after=None,
        change_detected=True,
        change_source=source,
        delivery_address_required=delivery_address_required,
        delivery_available=delivery_available,
        blocked_reason=reason,
        response_text=response_text,
        response_key=response_key,
        target_state=None,
    )


def _error_result(
    *,
    detected: str | None,
    before: str | None,
    source: str,
    reason: str,
) -> OrderTypeChangeResult:
    return _blocked_result(
        code=OrderTypeChangeCode.INVALID_ORDER_TYPE,
        detected=detected,
        before=before,
        source=source,
        reason=reason,
        response_text="I'm sorry, something went wrong. What would you like to do?",
        response_key="error_fallback",
    )


def _payment_already_sent(context: Any) -> bool:
    """Duck-typed check: has a payment link been sent?"""
    addr = getattr(context, "delivery_address", None)
    if addr is not None:
        if getattr(addr, "payment_link", None):
            return True
        if getattr(addr, "payment_link_send_attempts", 0) > 0:
            return True
    return bool(getattr(context, "payment_link_sent", False))


def _order_already_submitted(context: Any) -> bool:
    """Duck-typed check: has the order been submitted?"""
    return bool(getattr(context, "order_submitted", False))
