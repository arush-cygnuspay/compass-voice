# app/state_machine/phase3_controls.py
from __future__ import annotations

from app.nlu.control_decision_service import DEFAULT_SERVICE as _control_service
from app.nlu.intent_resolution.intent import Intent
from app.nlu.nlu_result import NLUResult
from app.nlu.query_normalization.text_preprocessor import normalize_text


_STAY_ON_CALL_PHRASES: tuple[str, ...] = (
    "stay on the line",
    "stay on line",
    "stay with me",
    "stay here",
    "stay on the call",
    "wait on the line",
    "hold on the line",
)

_AFTER_CALL_PHRASES: tuple[str, ...] = (
    "after the call",
    "after this call",
    "i will do it after the call",
    "ill do it after the call",
    "i will complete it after the call",
    "ill complete it after the call",
    "i will do it later",
    "ill do it later",
    "i'll do it later",
    "later",
    "not now",
)

_CANNOT_OPEN_LINK_PHRASES: tuple[str, ...] = (
    "cant open the link",
    "cannot open the link",
    "cant open the message",
    "cannot open the message",
    "cant open the sms",
    "cannot open the sms",
    "i cant open the link",
    "i cannot open the link",
    "i cant open the message",
    "i cannot open the message",
    "phone is to my ear",
    "while im on the phone",
    "while i am on the phone",
)

_RESEND_LINK_PHRASES: tuple[str, ...] = (
    "resend the link",
    "send it again",
    "send the link again",
    "text it again",
    "resend it",
)

_RESTART_ITEM_PHRASES: tuple[str, ...] = (
    "start over",
    "restart this item",
    "restart the item",
    "start this item over",
    "cancel that item",
)

_RESTART_ORDER_PHRASES: tuple[str, ...] = (
    "start the order over",
    "restart the order",
    "start over with the order",
    "cancel the whole order",
)

_ORDER_SUMMARY_REQUEST_PHRASES: tuple[str, ...] = (
    "what did i order",
    "what is my order",
    "read back my order",
    "repeat my order",
    "order summary",
)


def _contains_any(normalized_text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in normalized_text for phrase in phrases)


def normalized_control_text(text: str) -> str:
    return normalize_text(text or "")


def detect_payment_wait_mode_choice(text: str) -> str | None:
    normalized = normalized_control_text(text)
    if not normalized:
        return None

    if _contains_any(normalized, _STAY_ON_CALL_PHRASES):
        return "stay_on_call"
    if _contains_any(normalized, _AFTER_CALL_PHRASES):
        return "after_call"
    if _contains_any(normalized, _CANNOT_OPEN_LINK_PHRASES):
        return "cannot_open_link"
    if _contains_any(normalized, _RESEND_LINK_PHRASES):
        return "resend_link"
    return None


def is_live_agent_request(
    text: str,
    nlu_result: NLUResult | None = None,
) -> bool:
    """Return True when *text* (optionally with *nlu_result*) signals a live-agent transfer.

    Resolution order:
    1. NLU intent Intent.REQUEST_AGENT above confidence threshold.
    2. Phrase fallback via FallbackPhraseMatcher (logs fallback_hit for retraining).

    The *nlu_result* parameter is optional for backward-compatibility with
    call sites that do not yet pass an NLU result.
    """
    decision = _control_service.resolve_agent_request(text, nlu_result)
    return decision.intent == Intent.REQUEST_AGENT


def is_restart_item_request(text: str) -> bool:
    normalized = normalized_control_text(text)
    if not normalized:
        return False
    return _contains_any(normalized, _RESTART_ITEM_PHRASES)


def is_restart_order_request(text: str) -> bool:
    normalized = normalized_control_text(text)
    if not normalized:
        return False
    return _contains_any(normalized, _RESTART_ORDER_PHRASES)


def is_repeat_order_summary_request(text: str) -> bool:
    normalized = normalized_control_text(text)
    if not normalized:
        return False
    return _contains_any(normalized, _ORDER_SUMMARY_REQUEST_PHRASES)
