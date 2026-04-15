# app/state_machine/handlers/order/confirm_order_handler.py
from __future__ import annotations

import os

from app.nlu.intent_resolution.intent import Intent
from app.state_machine.handlers.base_handler import BaseHandler
from app.state_machine.handlers.payment.payment_flow_support import (
    ensure_payment_link_for_voice_session,
)
from app.state_machine.models.conversation_state import ConversationState
from app.state_machine.handler_result import HandlerResult


FORCE_VOICE_ADDRESS_FALLBACK = os.getenv("COMPASS_FORCE_VOICE_ADDRESS_FALLBACK", "0") == "1"


class ConfirmOrderHandler(BaseHandler):
    CHECKOUT_CONFIRM_INTENTS = {
        Intent.CONFIRM,
        Intent.CHECKOUT,
        Intent.CONFIRM_ORDER,
        Intent.FINISH_ORDER,
        Intent.PAYMENT_REQUEST,
    }

    REVIEW_INTENTS = {
        Intent.SHOW_CART,
        Intent.SHOW_TOTAL,
        Intent.REVIEW_ORDER,
    }

    def __init__(self, cart_summary_builder, sms_service, checkout_service):
        self.cart_summary_builder = cart_summary_builder
        self.sms_service = sms_service
        self.checkout_service = checkout_service

    def _build_payment_link_result(self, session, context) -> HandlerResult:
        delivery = context.delivery_address
        phone_number = delivery.customer_phone_number
        if not phone_number:
            return HandlerResult(
                next_state=ConversationState.CONFIRMING_ORDER,
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
                next_state=ConversationState.CONFIRMING_ORDER,
                response_key="payment_link_send_failed",
            )

        delivery.order_number = payment_info.get("order_number") or delivery.order_number
        delivery.payment_link = payment_info.get("redirect_url") or delivery.payment_link
        delivery.confirmation_link = (
            payment_info.get("confirmation_link") or delivery.confirmation_link
        )
        delivery.payment_link_send_attempts = 0

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
                next_state=ConversationState.CONFIRMING_ORDER,
                response_key="payment_link_send_failed",
            )

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

    def handle(self, intent, context, user_text, session=None):
        if session is None:
            return HandlerResult(
                next_state=ConversationState.ERROR_RECOVERY,
                response_key="confirmation_state_error",
            )

        if session.conversation_state != ConversationState.CONFIRMING_ORDER:
            return HandlerResult(
                next_state=ConversationState.ERROR_RECOVERY,
                response_key="confirmation_state_error",
            )

        if session.cart.is_empty():
            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="cart_empty",
            )

        if intent in self.CHECKOUT_CONFIRM_INTENTS:
            delivery = context.delivery_address

            if context.order_type == "delivery":
                if context.delivery_address_confirmed or delivery.collected or delivery.confirmed:
                    delivery.source = "voice"
                    return self._build_payment_link_result(session, context)

                if not delivery.area or not delivery.postal_code:
                    context.current_prompt_field = "delivery_area"
                    return HandlerResult(
                        next_state=ConversationState.WAITING_FOR_DELIVERY_ELIGIBILITY,
                        response_key="ask_for_delivery_area",
                    )

                can_use_checkout_link = (
                        not FORCE_VOICE_ADDRESS_FALLBACK
                        and self.sms_service.is_configured()
                        and bool(delivery.customer_phone_number)
                )

                if can_use_checkout_link:
                    delivery.checkout_link_send_attempts = 0

                    checkout_session = self.checkout_service.create_session(
                        restaurant_id=session.restaurant_id,
                        call_sid=getattr(session, "call_sid", None)
                        or getattr(session, "session_id", None),
                        order_number=delivery.order_number,
                        customer_phone_number=delivery.customer_phone_number,
                        address_required=True,
                        area=delivery.area,
                        postal_code=delivery.postal_code,
                        order_summary=self.cart_summary_builder.build(session.cart),
                    )
                    delivery.order_number = checkout_session.order_number
                    delivery.address_form_link = self.checkout_service.build_checkout_url(
                        checkout_session.token
                    )
                    delivery.source = "sms_form"

                    return HandlerResult(
                        next_state=ConversationState.WAITING_FOR_CHECKOUT_COMPLETION,
                        response_key="checkout_link_sent",
                        response_payload={"order_number": delivery.order_number},
                        command={
                            "type": "SEND_SMS",
                            "payload": {
                                "template": "checkout_link",
                                "phone_number": delivery.customer_phone_number,
                                "order_number": str(delivery.order_number),
                                "link": delivery.address_form_link,
                            },
                        },
                    )

                delivery.source = "voice"
                context.current_prompt_field = "delivery_house_number"
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_DELIVERY_ADDRESS_COLLECTION,
                    response_key="checkout_link_unavailable_fallback_voice",
                    response_payload={
                        "area": delivery.area,
                        "postal_code": delivery.postal_code,
                    },
                )

            delivery.source = delivery.source or "voice"
            return self._build_payment_link_result(session, context)

        if intent == Intent.PAYMENT_STATUS:
            return HandlerResult(
                next_state=ConversationState.CONFIRMING_ORDER,
                response_key="payment_not_started",
            )

        if intent in {Intent.CANCEL_ORDER, Intent.DENY, Intent.CANCEL}:
            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="order_cancelled",
            )

        payload = self.cart_summary_builder.build(session.cart)
        return HandlerResult(
            next_state=ConversationState.CONFIRMING_ORDER,
            response_key="confirm_order_summary",
            response_payload=payload,
        )
