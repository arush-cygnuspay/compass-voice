# app/nlu/turn_resolver/gpt_fallback_policy.py
"""State-specific fallback response builder for GPT failure paths.

When GPT fails (timeout, invalid JSON, provider error, etc.) AND the local
NLU result is not safe to execute, the FSM must respond with a natural
clarification prompt rather than executing corrupt or ambiguous local slots.

Rules
-----
* Never say "GPT failed", "AI failed", or "OpenAI".
* Response keys are the same customer-facing templates used in normal flow.
* Cart mutation MUST NOT happen on any fallback path from this module.
* Only build responses — do not call handlers, modify context, or advance FSM.

Fallback matrix
---------------
  IDLE + local valid ADD_ITEM        → continue local add_item path (no fallback needed)
  IDLE + local unsafe multi-item     → "I heard a few items but couldn't separate them.
                                        What's the first item?"
  WAITING_FOR_MODIFIER + GPT failure → "Please choose one of: {options}"
  WAITING_FOR_SIDE + GPT failure     → "Please choose one of: {sides}"
  WAITING_FOR_SIZE + GPT failure     → "What size would you like?"
  CONFIRMING_ORDER + GPT failure     → "Please say yes to confirm, or no to keep ordering."
  WAITING_FOR_PAYMENT + GPT failure  → continue deterministic payment flow
  any other + GPT failure + local valid → use local
  any other + GPT failure + local unsafe → generic neutral clarification
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

from app.state_machine.models.conversation_state import ConversationState

if TYPE_CHECKING:
    from app.nlu.turn_resolver.gpt_safe_client import GptSafeResult
    from app.state_machine.models.conversation_context import ConversationContext

# ---------------------------------------------------------------------------
# Fallback descriptors
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GptFallbackDecision:
    """Rich fallback decision for GPT failure paths.

    Returned by ``decide_gpt_failure_fallback()``.  The caller uses this to
    decide whether to continue with the local deterministic path or issue a
    state-specific clarification prompt.

    Fields
    ------
    use_local:
        True when the local NLU result is safe and the caller should proceed
        as if GPT was not involved.
    use_state_clarification:
        True when use_local=False and the caller should issue the
        response_key clarification to the customer.
    fallback_source:
        One of: ``"local"``, ``"state_clarification"``, ``"handoff"``,
        ``"no_action"``.
    response_key:
        The response template key to use when use_state_clarification=True.
    response_text:
        Reserved for future inline response text.  None in current impl.
    reason:
        Short code explaining why this decision was made (for logging).
    local_safe:
        Whether the local NLU result was deemed safe to use.
    unsafe_reason:
        Short code from is_local_result_safe() when local is not safe.
    """

    use_local: bool
    use_state_clarification: bool
    fallback_source: str
    response_key: str | None
    response_text: str | None
    reason: str
    local_safe: bool
    unsafe_reason: str | None


@dataclass(frozen=True, slots=True)
class GptFallbackResponse:
    """A response key + payload to issue when GPT fails and local is unsafe.

    The calling handler or TurnEngine converts this to a ``HandlerResult``
    with the given response_key and payload.

    ``use_local`` signals that the local deterministic path should be used
    instead of issuing a clarification response.
    """

    response_key: str
    response_payload: dict
    use_local: bool = False
    fallback_source: str = "state_clarification"  # "local" | "state_clarification" | "handoff"


# ---------------------------------------------------------------------------
# State-specific builders
# ---------------------------------------------------------------------------

# Maps waiting states to their fallback response key
_WAITING_STATE_FALLBACK_KEYS: dict[ConversationState, str] = {
    ConversationState.WAITING_FOR_SIZE: "ask_for_size",
    ConversationState.WAITING_FOR_SIDE_SIZE: "ask_for_size",
    ConversationState.WAITING_FOR_SIDE: "repeat_side_options",
    ConversationState.WAITING_FOR_MODIFIER: "repeat_modifier_options",
}

# Response key for compound-utterance GPT failure in idle state
_COMPOUND_FALLBACK_KEY = "compound_unclear_ask_first"

# Response key for generic neutral clarification
_GENERIC_CLARIFY_KEY = "ask_to_repeat"


def _side_choice_names(context: "ConversationContext", max_count: int = 6) -> list[str]:
    """Extract the top side choice names from the current pending item."""
    try:
        pending = context.pending_add_item
        if pending is None:
            return []
        idx = context.current_side_group_index
        groups = pending.side_groups
        if idx >= len(groups):
            return []
        group = groups[idx]
        return [c.name for c in group.choices[:max_count]]
    except Exception:
        return []


def _modifier_choice_names(context: "ConversationContext", max_count: int = 6) -> list[str]:
    """Extract the top modifier choice names from the current pending item."""
    try:
        pending = context.pending_add_item
        if pending is None:
            return []
        idx = context.current_modifier_group_index
        groups = pending.modifier_groups
        if idx >= len(groups):
            return []
        group = groups[idx]
        return [c.name for c in group.choices[:max_count]]
    except Exception:
        return []


def _modifier_group_name(context: "ConversationContext") -> str | None:
    try:
        pending = context.pending_add_item
        if pending is None:
            return None
        idx = context.current_modifier_group_index
        groups = pending.modifier_groups
        if idx >= len(groups):
            return None
        return groups[idx].name
    except Exception:
        return None


def _side_group_name(context: "ConversationContext") -> str | None:
    try:
        pending = context.pending_add_item
        if pending is None:
            return None
        idx = context.current_side_group_index
        groups = pending.side_groups
        if idx >= len(groups):
            return None
        return groups[idx].name
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def is_local_result_safe(
    state: ConversationState,
    local_intent_value: str,
    local_slots: Sequence[object],
    local_confidence: float,
    *,
    low_confidence_threshold: float = 0.55,
) -> tuple[bool, str | None]:
    """Return (is_safe, unsafe_reason) for the local NLU result.

    A result is unsafe when:
    - Multiple ITEM slots are present (compound utterance — multi-item ambiguity)
    - Confidence is below threshold and the state is IDLE
    - The local intent is not allowed in a waiting state

    Parameters
    ----------
    state:
        Current conversation state.
    local_intent_value:
        The ``Intent.value`` string (e.g. "add_item", "unknown").
    local_slots:
        Slot values from local NLU (each must have a .name attribute).
    local_confidence:
        Intent confidence (0.0–1.0).
    low_confidence_threshold:
        Below this in IDLE state with UNKNOWN intent → unsafe.
    """
    # Count ITEM slots
    item_slot_count = sum(
        1 for s in local_slots
        if getattr(s, "name", "").upper() in {"ITEM", "MENU_ITEM"}
    )

    # Multiple ITEM slots → compound ambiguity → unsafe
    if item_slot_count >= 2:
        return False, "multi_item_slots"

    # IDLE with UNKNOWN + very low confidence → unsafe
    if (
        state == ConversationState.IDLE
        and local_intent_value.upper() == "UNKNOWN"
        and local_confidence < low_confidence_threshold
    ):
        return False, "unknown_intent_low_confidence"

    # Waiting state but intent is not add_item / control → potentially unsafe
    _waiting = {
        ConversationState.WAITING_FOR_MODIFIER,
        ConversationState.WAITING_FOR_SIDE,
        ConversationState.WAITING_FOR_SIDE_SIZE,
        ConversationState.WAITING_FOR_SIZE,
    }
    if state in _waiting and local_intent_value.upper() not in {
        "ADD_ITEM", "CONFIRM", "DENY", "CANCEL", "AFFIRM", "UNKNOWN"
    }:
        return False, f"wrong_intent_for_state:{local_intent_value}"

    return True, None


def build_gpt_failure_fallback_response(
    state: ConversationState,
    context: "ConversationContext",
    local_intent_value: str,
    local_slots: Sequence[object],
    local_confidence: float,
    gpt_status: str,
    unsafe_reason: str | None = None,
) -> GptFallbackResponse:
    """Build the state-specific response for when GPT fails and local is unsafe.

    Parameters
    ----------
    state:
        Current conversation state at the time of the GPT failure.
    context:
        Current conversation context (read-only — not mutated).
    local_intent_value:
        The ``Intent.value`` string from local NLU.
    local_slots:
        Slot values from local NLU.
    local_confidence:
        Local NLU confidence.
    gpt_status:
        The ``GptCallStatus`` constant describing why GPT failed.
    unsafe_reason:
        Short code from ``is_local_result_safe()`` (or None when caller
        already determined local is safe).

    Returns
    -------
    ``GptFallbackResponse`` — response key + payload the caller uses to
    build a ``HandlerResult``.  Never raises.
    """
    try:
        return _build_fallback(
            state=state,
            context=context,
            local_intent_value=local_intent_value,
            local_slots=local_slots,
            local_confidence=local_confidence,
            gpt_status=gpt_status,
            unsafe_reason=unsafe_reason,
        )
    except Exception:
        # Absolute last resort — never raise into the call flow
        return GptFallbackResponse(
            response_key=_GENERIC_CLARIFY_KEY,
            response_payload={"repeat_reason": "gpt_failure"},
            fallback_source="state_clarification",
        )


def _build_fallback(
    state: ConversationState,
    context: "ConversationContext",
    local_intent_value: str,
    local_slots: Sequence[object],
    local_confidence: float,
    gpt_status: str,
    unsafe_reason: str | None,
) -> GptFallbackResponse:
    """Internal implementation — may raise; wrapped by build_gpt_failure_fallback_response."""

    local_safe, detected_reason = is_local_result_safe(
        state=state,
        local_intent_value=local_intent_value,
        local_slots=local_slots,
        local_confidence=local_confidence,
    )
    reason = unsafe_reason or detected_reason

    # ── Payment / checkout states: always continue deterministic flow ────────
    if state in {
        ConversationState.WAITING_FOR_PAYMENT,
        ConversationState.WAITING_FOR_CHECKOUT_COMPLETION,
    }:
        return GptFallbackResponse(
            response_key="payment_continue_deterministic",
            response_payload={"gpt_status": gpt_status},
            use_local=True,
            fallback_source="local",
        )

    # ── CONFIRMING_ORDER: deterministic yes/no ──────────────────────────────
    if state == ConversationState.CONFIRMING_ORDER:
        return GptFallbackResponse(
            response_key="confirm_order_repeat",
            response_payload={"repeat_reason": "gpt_failure"},
            use_local=False,
            fallback_source="state_clarification",
        )

    # ── Waiting states: repeat current group options ─────────────────────────
    if state == ConversationState.WAITING_FOR_MODIFIER:
        choices = _modifier_choice_names(context)
        group_name = _modifier_group_name(context) or "modifier"
        return GptFallbackResponse(
            response_key="repeat_modifier_options",
            response_payload={
                "group_name": group_name,
                "top_choices": choices,
                "all_choices": choices,
                "total_choices": len(choices),
                "selected_names": [],
                "selected_count": 0,
                "repeat_reason": "need_choice",
            },
            fallback_source="state_clarification",
        )

    if state in {ConversationState.WAITING_FOR_SIDE, ConversationState.WAITING_FOR_SIDE_SIZE}:
        choices = _side_choice_names(context)
        group_name = _side_group_name(context) or "side"
        return GptFallbackResponse(
            response_key="repeat_side_options",
            response_payload={
                "group_name": group_name,
                "top_choices": choices,
                "all_choices": choices,
                "total_choices": len(choices),
                "selected_names": [],
                "selected_count": 0,
                "repeat_reason": "need_choice",
            },
            fallback_source="state_clarification",
        )

    if state in {ConversationState.WAITING_FOR_SIZE, ConversationState.WAITING_FOR_SIDE_SIZE}:
        return GptFallbackResponse(
            response_key="ask_for_size",
            response_payload={"repeat_reason": "gpt_failure"},
            fallback_source="state_clarification",
        )

    # ── IDLE state fallback ──────────────────────────────────────────────────
    if state == ConversationState.IDLE:
        if local_safe:
            # Local is safe: continue normal add-item / local path
            return GptFallbackResponse(
                response_key="",  # no special response — caller uses local path
                response_payload={},
                use_local=True,
                fallback_source="local",
            )

        if reason == "multi_item_slots":
            # Compound utterance: ask for first item only (safe degradation)
            return GptFallbackResponse(
                response_key=_COMPOUND_FALLBACK_KEY,
                response_payload={
                    "repeat_reason": "compound_unclear",
                },
                fallback_source="state_clarification",
            )

        # Other unsafe: generic clarify
        return GptFallbackResponse(
            response_key=_GENERIC_CLARIFY_KEY,
            response_payload={"repeat_reason": "gpt_failure"},
            fallback_source="state_clarification",
        )

    # ── All other states: if local is safe, use it; else generic clarify ─────
    if local_safe:
        return GptFallbackResponse(
            response_key="",
            response_payload={},
            use_local=True,
            fallback_source="local",
        )

    return GptFallbackResponse(
        response_key=_GENERIC_CLARIFY_KEY,
        response_payload={"repeat_reason": "gpt_failure"},
        fallback_source="state_clarification",
    )


# ---------------------------------------------------------------------------
# High-level decision helper (string-state API for Priority 2 layer)
# ---------------------------------------------------------------------------


def decide_gpt_failure_fallback(
    *,
    gpt_result: "GptSafeResult",
    state: str,
    local_intent: str | None,
    local_confidence: float | None,
    local_slots: list | tuple | None,
    allowed_intents: list | tuple,
    context: "ConversationContext",
    unsafe_local_reason: str | None = None,
) -> GptFallbackDecision:
    """Return a structured fallback decision for a GPT failure.

    Accepts a plain string *state* (as produced by GptContextBuilder).
    Never raises — on any internal error returns a safe state_clarification.

    Parameters
    ----------
    gpt_result:
        The failed GptSafeResult (for status extraction).
    state:
        Current conversation state as a lowercase string (e.g. ``"idle"``).
    local_intent:
        Intent string from local NLU, or None.
    local_confidence:
        Local NLU confidence, or None.
    local_slots:
        Slot objects from local NLU (each must have a ``.name`` attribute).
    allowed_intents:
        Allowed intents for this state (AllowedIntent objects or plain strings).
    context:
        Current ConversationContext (read-only).
    unsafe_local_reason:
        Caller-provided override for the unsafe reason code.
    """
    try:
        return _decide_fallback_impl(
            gpt_result=gpt_result,
            state=state,
            local_intent=local_intent,
            local_confidence=local_confidence,
            local_slots=local_slots,
            allowed_intents=allowed_intents,
            context=context,
            unsafe_local_reason=unsafe_local_reason,
        )
    except Exception:
        return GptFallbackDecision(
            use_local=False,
            use_state_clarification=True,
            fallback_source="state_clarification",
            response_key=_GENERIC_CLARIFY_KEY,
            response_text=None,
            reason="fallback_error",
            local_safe=False,
            unsafe_reason="internal_error",
        )


def _decide_fallback_impl(
    *,
    gpt_result: "GptSafeResult",
    state: str,
    local_intent: str | None,
    local_confidence: float | None,
    local_slots: list | tuple | None,
    allowed_intents: list | tuple,
    context: "ConversationContext",
    unsafe_local_reason: str | None,
) -> GptFallbackDecision:
    gpt_status: str = getattr(gpt_result, "status", "unknown_error") if gpt_result else "unknown_error"
    state_str = (state or "").strip().lower()
    local_intent_val = (local_intent or "unknown").strip()
    slots: Sequence[Any] = local_slots or ()
    confidence = float(local_confidence) if local_confidence is not None else 0.0

    # Map string state → ConversationState (safe fallback to IDLE)
    try:
        conv_state = ConversationState(state_str)
    except (ValueError, AttributeError):
        conv_state = ConversationState.IDLE

    # Check whether local intent is in the allowed list
    allowed_names: set[str] = set()
    for ai in (allowed_intents or []):
        name = getattr(ai, "name", None) or str(ai)
        allowed_names.add(name.lower())

    # Local safety check
    local_safe, detected_unsafe = is_local_result_safe(
        state=conv_state,
        local_intent_value=local_intent_val,
        local_slots=slots,
        local_confidence=confidence,
    )
    unsafe_reason_final = unsafe_local_reason or detected_unsafe

    # Caller-provided override takes priority
    if unsafe_local_reason:
        local_safe = False

    # Build underlying fallback response using existing state-aware logic
    fb = build_gpt_failure_fallback_response(
        state=conv_state,
        context=context,
        local_intent_value=local_intent_val,
        local_slots=slots,
        local_confidence=confidence,
        gpt_status=gpt_status,
        unsafe_reason=unsafe_reason_final,
    )

    use_local = fb.use_local
    use_clarification = not use_local

    if use_local:
        reason = f"gpt_{gpt_status}_local_safe"
        fallback_source = "local"
    else:
        reason = f"gpt_{gpt_status}_state_clarification"
        fallback_source = fb.fallback_source

    response_key = fb.response_key if not use_local else None

    return GptFallbackDecision(
        use_local=use_local,
        use_state_clarification=use_clarification,
        fallback_source=fallback_source,
        response_key=response_key or None,
        response_text=None,
        reason=reason,
        local_safe=local_safe,
        unsafe_reason=unsafe_reason_final,
    )
