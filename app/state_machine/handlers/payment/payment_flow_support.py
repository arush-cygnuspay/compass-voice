from __future__ import annotations

import logging

from app.services.checkout_service import CheckoutService, PAYMENT_FAILURE_STATUSES
from app.session.session import Session
from app.state_machine.handler_result import HandlerResult
from app.state_machine.models.conversation_state import ConversationState

logger = logging.getLogger(__name__)


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
) -> HandlerResult:
    if not order_number:
        logger.warning("verify_payment_for_order: missing order_number")
        return HandlerResult(
            next_state=pending_state,
            response_key="payment_not_confirmed_yet",
        )

    try:
        result = checkout_service.verify_payment_by_order_number(order_number)
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "verify_payment_for_order: error verifying order %s: %s",
            order_number,
            exc,
        )
        return HandlerResult(
            next_state=pending_state,
            response_key="payment_verification_error",
        )

    if result.get("payment_completed") or result.get("paid"):
        return HandlerResult(
            next_state=ConversationState.COMPLETED,
            response_key="order_completed",
            response_payload={"order_number": order_number},
            reset_context=True,
            command={"type": "CLEAR_CART"},
        )

    status_lower = str(result.get("status") or "").lower()
    if status_lower in PAYMENT_FAILURE_STATUSES:
        return HandlerResult(
            next_state=ConversationState.CONFIRMING_ORDER,
            response_key="payment_draft_saved_retry_later",
            response_payload={"order_number": order_number},
        )

    return HandlerResult(
        next_state=pending_state,
        response_key=pending_response_key,
    )
