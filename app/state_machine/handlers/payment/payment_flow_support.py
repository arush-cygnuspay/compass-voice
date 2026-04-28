from __future__ import annotations

import logging
import os
from typing import Any

from app.services.checkout_service import CheckoutService, PAYMENT_FAILURE_STATUSES
from app.session.session import Session
from app.state_machine.handler_result import HandlerResult
from app.state_machine.models.conversation_state import ConversationState

logger = logging.getLogger(__name__)

SEND_ORDER_SUMMARY_SMS_BEFORE_PAYMENT = (
    os.getenv("SEND_ORDER_SUMMARY_SMS_BEFORE_PAYMENT", "1") != "0"
)
PAYMENT_LINK_RESEND_COOLDOWN_SECONDS = float(
    os.getenv("COMPASS_PAYMENT_LINK_RESEND_COOLDOWN_SECONDS", "60")
)
VOICE_PAYMENT_FALLBACK_AVAILABLE = (
    os.getenv("COMPASS_VOICE_PAYMENT_FALLBACK_AVAILABLE", "0") == "1"
)


def _set_pending_payment_tracking(
    *,
    delivery,
    pending_state: ConversationState,
    status: str | None,
    reference: str | None = None,
) -> None:
    if delivery is None:
        return

    normalized_status = str(status or "").strip().lower()
    if pending_state == ConversationState.WAITING_FOR_CHECKOUT_COMPLETION:
        delivery.checkout_status = (
            "checkout_opened" if getattr(delivery, "form_completed", False) else "checkout_sent"
        )

    if normalized_status in PAYMENT_FAILURE_STATUSES:
        delivery.payment_status = "payment_failed"
        delivery.payment_session_state = "payment_failed_or_expired"
    else:
        delivery.payment_status = "payment_pending"
        if delivery.payment_wait_mode == "after_call":
            delivery.payment_session_state = "waiting_payment_after_call"
        else:
            delivery.payment_session_state = "waiting_payment_stay_on_call"

    if reference:
        delivery.payment_reference = reference


def _set_completed_payment_tracking(*, delivery, reference: str | None = None) -> None:
    if delivery is None:
        return

    if getattr(delivery, "address_form_link", None):
        delivery.checkout_status = "checkout_opened"
    delivery.payment_status = "payment_confirmed"
    delivery.payment_session_state = "payment_confirmed"
    if reference:
        delivery.payment_reference = reference


def format_order_summary_sms(order_summary: dict[str, Any] | None) -> str:
    summary = order_summary or {}
    items = list(summary.get("items") or [])
    if not items:
        return "View full order details in the checkout link."

    lines: list[str] = []
    for index, item in enumerate(items):
        if index >= 3:
            lines.append("View full order details in the checkout link.")
            break
        quantity = int(item.get("quantity", 1) or 1)
        name = str(item.get("name") or "Item").strip()
        detail_parts: list[str] = []
        variant = str(item.get("variant") or "").strip()
        if variant:
            detail_parts.append(variant)
        modifiers = [str(x).strip() for x in (item.get("modifiers") or []) if str(x).strip()]
        sides = [str(x).strip() for x in (item.get("sides") or []) if str(x).strip()]
        if modifiers:
            detail_parts.extend(modifiers[:3])
        if sides:
            detail_parts.extend(sides[:2])
        line = f"{quantity}x {name}"
        if detail_parts:
            line = f"{line} - {', '.join(detail_parts[:4])}"
        lines.append(line)

    total = str(summary.get("total") or summary.get("grand_total") or "").strip()
    if total:
        lines.append(f"Total: {total}")
    return "\n".join(lines)


def build_payment_sms_payload(
    *,
    template: str,
    phone_number: str,
    order_number: str,
    link: str,
    order_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "template": template,
        "phone_number": phone_number,
        "order_number": str(order_number or ""),
        "link": link,
    }
    if SEND_ORDER_SUMMARY_SMS_BEFORE_PAYMENT:
        payload["summary_text"] = format_order_summary_sms(order_summary)
    return payload


def _detect_phone_surface(session: Session | None) -> str:
    if session is None:
        return "unknown"
    context = getattr(session, "conversation_context", None)
    device_type = getattr(context, "caller_device_type", None) if context else None
    if device_type == "chat":
        return "chat_ui"
    return "twilio"


def log_phone_number_unavailable(
    *,
    session: Session | None,
    consumer: str,
) -> None:
    """Structured log for sites that skipped phone-dependent flows.

    Emitted whenever the chat (or any other) surface lacks a usable phone
    number so we can observe in prod how often the SMS/phone-dependent
    branches are skipped.
    """
    surface = _detect_phone_surface(session)
    logger.info(
        "phone_number_unavailable",
        extra={
            "event_name": "phone_number_unavailable",
            "surface": surface,
            "consumer": consumer,
        },
    )


def append_payment_event(
    payload: dict[str, Any] | None,
    *,
    event_name: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    updated = dict(payload or {})
    events = list(updated.get("_payment_events") or [])
    events.append({"event_name": event_name, "metadata": metadata or {}})
    updated["_payment_events"] = events
    return updated


def ensure_payment_link_for_voice_session(
    *,
    checkout_service: CheckoutService,
    session: Session,
    order_summary: dict,
    address_source: str | None = None,
) -> dict:
    delivery = session.conversation_context.delivery_address
    context = session.conversation_context

    return checkout_service.ensure_payment_link(
        restaurant_id=session.restaurant_id,
        call_sid=session.session_id,
        order_number=delivery.order_number,
        customer_phone_number=delivery.customer_phone_number,
        address_required=context.order_type == "delivery",
        area=delivery.area,
        postal_code=delivery.postal_code,
        order_summary=order_summary,
        house_number=delivery.house_number,
        street=delivery.street,
        secondary_address=delivery.secondary_address,
        city=delivery.city,
        state=delivery.state,
        full_address_raw=delivery.full_address_raw,
        address_source=address_source or delivery.source or "voice",
    )


def verify_payment_for_order(
    *,
    checkout_service: CheckoutService,
    order_number: str | None,
    pending_state: ConversationState,
    pending_response_key: str,
    delivery=None,
) -> HandlerResult:
    if not order_number:
        logger.warning("verify_payment_for_order: missing order_number")
        _set_pending_payment_tracking(
            delivery=delivery,
            pending_state=pending_state,
            status="pending",
        )
        return HandlerResult(
            next_state=pending_state,
            response_key="payment_not_confirmed_yet",
            response_payload=append_payment_event(
                None,
                event_name="payment_reminder_sent",
                metadata={"pending_response_key": "payment_not_confirmed_yet"},
            ),
        )

    try:
        result = checkout_service.verify_payment_by_order_number(order_number)
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "verify_payment_for_order: error verifying order %s: %s",
            order_number,
            exc,
        )
        _set_pending_payment_tracking(
            delivery=delivery,
            pending_state=pending_state,
            status="pending",
        )
        return HandlerResult(
            next_state=pending_state,
            response_key="payment_verification_error",
            response_payload=append_payment_event(
                None,
                event_name="payment_reminder_sent",
                metadata={"pending_response_key": "payment_verification_error"},
            ),
        )

    if result.get("payment_completed") or result.get("paid"):
        _set_completed_payment_tracking(
            delivery=delivery,
            reference=result.get("reference"),
        )
        return HandlerResult(
            next_state=ConversationState.COMPLETED,
            response_key="order_completed",
            response_payload=append_payment_event(
                {"order_number": order_number},
                event_name="payment_confirmed",
            ),
            reset_context=True,
            command={"type": "CLEAR_CART"},
        )

    status_lower = str(result.get("status") or "").lower()
    _set_pending_payment_tracking(
        delivery=delivery,
        pending_state=pending_state,
        status=status_lower,
        reference=result.get("reference"),
    )
    if status_lower in PAYMENT_FAILURE_STATUSES:
        return HandlerResult(
            next_state=ConversationState.CONFIRMING_ORDER,
            response_key="payment_draft_saved_retry_later",
            response_payload=append_payment_event(
                {"order_number": order_number},
                event_name="payment_failed",
                metadata={"status": status_lower},
            ),
        )

    return HandlerResult(
        next_state=pending_state,
        response_key=pending_response_key,
        response_payload=append_payment_event(
            None,
            event_name="payment_reminder_sent",
            metadata={"pending_response_key": pending_response_key},
        ),
    )
