# app/state_machine/handlers/order/confirm_order_handler.py
from __future__ import annotations

import os

from app.nlu.intent_resolution.intent import Intent
from app.state_machine.handlers.base_handler import BaseHandler
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

    def __init__(self, cart_summary_builder, sms_service):
        self.cart_summary_builder = cart_summary_builder
        self.sms_service = sms_service

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
                # If voice fallback already collected the delivery address,
                # do NOT try checkout link again. Send payment link now.
                if context.delivery_address_confirmed or delivery.collected or delivery.confirmed:
                    phone_number = delivery.customer_phone_number
                    if not phone_number:
                        return HandlerResult(
                            next_state=ConversationState.CONFIRMING_ORDER,
                            response_key="payment_link_send_failed",
                            response_payload={"reason": "missing_customer_phone_number"},
                        )

                    delivery.order_number = delivery.order_number or "TEST123"
                    delivery.payment_link = delivery.payment_link or "https://www.cygnuspay.com"

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

                delivery.order_number = delivery.order_number or "TEST123"
                delivery.address_form_link = delivery.address_form_link or "https://www.cygnuspay.com"

                can_use_checkout_link = (
                        not FORCE_VOICE_ADDRESS_FALLBACK
                        and self.sms_service.is_configured()
                        and bool(delivery.customer_phone_number)
                        and bool(delivery.address_form_link)
                )

                if can_use_checkout_link:
                    delivery.order_number = delivery.order_number or "TEST123"
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