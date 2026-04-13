# app/state_machine/handlers/payment/waiting_for_checkout_completion_handler.py
from app.nlu.intent_resolution.intent import Intent
from app.state_machine.handlers.base_handler import BaseHandler
from app.state_machine.models.conversation_state import ConversationState
from app.state_machine.handler_result import HandlerResult


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
            delivery.form_completed = True
            delivery.collected = True
            delivery.confirmed = True
            context.delivery_address_confirmed = True

            return HandlerResult(
                next_state=ConversationState.COMPLETED,
                response_key="order_completed",
                reset_context=True,
                command={"type": "CLEAR_CART"},
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

            delivery.order_number = delivery.order_number or "TEST123"

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

        if intent in self.STATUS_INTENTS or intent in {Intent.DENY, Intent.CANCEL}:
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_CHECKOUT_COMPLETION,
                response_key="waiting_for_checkout_completion",
            )

        return HandlerResult(
            next_state=ConversationState.WAITING_FOR_CHECKOUT_COMPLETION,
            response_key="waiting_for_checkout_completion",
        )