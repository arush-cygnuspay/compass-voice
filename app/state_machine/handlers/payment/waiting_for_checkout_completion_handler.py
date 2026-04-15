from __future__ import annotations

from app.nlu.intent_resolution.intent import Intent
from app.services.checkout_service import CheckoutService
from app.state_machine.handler_result import HandlerResult
from app.state_machine.handlers.base_handler import BaseHandler
from app.state_machine.handlers.payment.payment_flow_support import verify_payment_for_order
from app.state_machine.models.conversation_state import ConversationState


class WaitingForCheckoutCompletionHandler(BaseHandler):
    COMPLETE_INTENTS = {Intent.PAYMENT_DONE}
    RESEND_INTENTS = {Intent.PAYMENT_REQUEST, Intent.PAYMENT_METHODS_QUERY}
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

    VERIFY_STATUS_INTENTS = {
        Intent.PAYMENT_STATUS,
        Intent.ORDER_STATUS_GENERAL,
        Intent.ORDER_PROCESSING_STATUS,
        Intent.ORDER_PLACEMENT_STATUS,
    }

    def __init__(self, checkout_service: CheckoutService):
        self.checkout_service = checkout_service

    def handle(self, intent, context, user_text, session=None):
        if session is None:
            return HandlerResult(
                next_state=ConversationState.ERROR_RECOVERY,
                response_key="confirmation_state_error",
            )

        if session.conversation_state != ConversationState.WAITING_FOR_CHECKOUT_COMPLETION:
            return HandlerResult(
                next_state=ConversationState.ERROR_RECOVERY,
                response_key="confirmation_state_error",
            )

        delivery = context.delivery_address

        if intent in self.COMPLETE_INTENTS:
            return verify_payment_for_order(
                checkout_service=self.checkout_service,
                order_number=delivery.order_number,
                pending_state=ConversationState.WAITING_FOR_CHECKOUT_COMPLETION,
                pending_response_key="payment_not_confirmed_yet",
            )

        if intent == Intent.CANCEL_ORDER:
            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="order_cancelled",
            )

        if intent in self.RESEND_INTENTS:
            if not delivery.customer_phone_number or not delivery.address_form_link:
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_CHECKOUT_COMPLETION,
                    response_key="checkout_link_send_failed",
                )

            delivery.checkout_link_send_attempts = 0

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

        if intent in self.VERIFY_STATUS_INTENTS:
            return verify_payment_for_order(
                checkout_service=self.checkout_service,
                order_number=delivery.order_number,
                pending_state=ConversationState.WAITING_FOR_CHECKOUT_COMPLETION,
                pending_response_key="waiting_for_checkout_completion",
            )

        if intent in self.STATUS_INTENTS or intent in {Intent.DENY, Intent.CANCEL}:
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_CHECKOUT_COMPLETION,
                response_key="waiting_for_checkout_completion",
            )

        return HandlerResult(
            next_state=ConversationState.WAITING_FOR_CHECKOUT_COMPLETION,
            response_key="waiting_for_checkout_completion",
        )
