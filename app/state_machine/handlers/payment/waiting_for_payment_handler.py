from __future__ import annotations

from app.nlu.intent_resolution.intent import Intent
from app.services.checkout_service import CheckoutService
from app.state_machine.handler_result import HandlerResult
from app.state_machine.handlers.base_handler import BaseHandler
from app.state_machine.handlers.payment.payment_flow_support import (
    ensure_payment_link_for_voice_session,
    verify_payment_for_order,
)
from app.state_machine.models.conversation_state import ConversationState


class WaitingForPaymentHandler(BaseHandler):
    RESEND_PAYMENT_INTENTS = {
        Intent.PAYMENT_REQUEST,
        Intent.PAYMENT_METHODS_QUERY,
    }

    STATUS_INTENTS = {
        Intent.PAYMENT_STATUS,
        Intent.SHOW_CART,
        Intent.SHOW_TOTAL,
        Intent.REVIEW_ORDER,
        Intent.ORDER_STATUS_GENERAL,
        Intent.ORDER_PROCESSING_STATUS,
        Intent.ORDER_PLACEMENT_STATUS,
        Intent.ORDER_ERROR_STATUS,
    }

    COMPLETE_PAYMENT_INTENTS = {
        Intent.PAYMENT_DONE,
    }

    VERIFY_STATUS_INTENTS = {
        Intent.PAYMENT_STATUS,
        Intent.ORDER_STATUS_GENERAL,
        Intent.ORDER_PROCESSING_STATUS,
        Intent.ORDER_PLACEMENT_STATUS,
    }

    def __init__(self, cart_summary_builder, checkout_service: CheckoutService):
        self.cart_summary_builder = cart_summary_builder
        self.checkout_service = checkout_service

    def handle(self, intent, context, user_text, session=None):
        if session is None:
            return HandlerResult(
                next_state=ConversationState.ERROR_RECOVERY,
                response_key="confirmation_state_error",
            )

        if session.conversation_state != ConversationState.WAITING_FOR_PAYMENT:
            return HandlerResult(
                next_state=ConversationState.ERROR_RECOVERY,
                response_key="confirmation_state_error",
            )

        if intent in self.COMPLETE_PAYMENT_INTENTS:
            delivery = context.delivery_address if context else None
            order_number = getattr(delivery, "order_number", None) if delivery else None
            return verify_payment_for_order(
                checkout_service=self.checkout_service,
                order_number=order_number,
                pending_state=ConversationState.WAITING_FOR_PAYMENT,
                pending_response_key="payment_not_confirmed_yet",
            )

        if intent == Intent.CANCEL_ORDER:
            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="order_cancelled",
            )

        if intent in self.RESEND_PAYMENT_INTENTS:
            delivery = context.delivery_address
            phone_number = delivery.customer_phone_number
            if not phone_number:
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_PAYMENT,
                    response_key="payment_link_send_failed",
                    response_payload={"reason": "missing_customer_phone_number"},
                )

            order_summary = self.cart_summary_builder.build(session.cart)
            try:
                payment_info = ensure_payment_link_for_voice_session(
                    checkout_service=self.checkout_service,
                    session=session,
                    order_summary=order_summary,
                    address_source=delivery.source or "voice",
                )
            except Exception:
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_PAYMENT,
                    response_key="payment_link_send_failed",
                )
            delivery.order_number = payment_info.get("order_number") or delivery.order_number
            delivery.payment_link = payment_info.get("redirect_url") or delivery.payment_link
            delivery.confirmation_link = (
                payment_info.get("confirmation_link") or delivery.confirmation_link
            )

            if payment_info.get("payment_completed"):
                return HandlerResult(
                    next_state=ConversationState.COMPLETED,
                    response_key="order_completed",
                    response_payload={"order_number": delivery.order_number},
                    reset_context=True,
                    command={"type": "CLEAR_CART"},
                )

            if not delivery.payment_link:
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_PAYMENT,
                    response_key="payment_link_send_failed",
                )

            delivery.payment_link_send_attempts = 0

            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_PAYMENT,
                response_key="payment_link_sent",
                response_payload={"order_number": delivery.order_number},
                command={
                    "type": "SEND_SMS",
                    "payload": {
                        "template": "payment_link",
                        "phone_number": phone_number,
                        "order_number": str(delivery.order_number),
                        "link": delivery.payment_link,
                    },
                },
            )

        if intent in self.VERIFY_STATUS_INTENTS:
            delivery = context.delivery_address if context else None
            order_number = getattr(delivery, "order_number", None) if delivery else None
            return verify_payment_for_order(
                checkout_service=self.checkout_service,
                order_number=order_number,
                pending_state=ConversationState.WAITING_FOR_PAYMENT,
                pending_response_key="waiting_for_payment",
            )

        if intent in self.STATUS_INTENTS or intent in {Intent.DENY, Intent.CANCEL}:
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_PAYMENT,
                response_key="waiting_for_payment",
            )

        return HandlerResult(
            next_state=ConversationState.WAITING_FOR_PAYMENT,
            response_key="waiting_for_payment",
        )
