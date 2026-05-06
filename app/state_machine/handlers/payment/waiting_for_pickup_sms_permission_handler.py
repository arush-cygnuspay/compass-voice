# app/state_machine/handlers/payment/waiting_for_pickup_sms_permission_handler.py
from __future__ import annotations

from app.state_machine.control_intent_resolver import resolve_control_intent
from app.state_machine.handler_result import HandlerResult
from app.state_machine.handlers.base_handler import BaseHandler
from app.state_machine.handlers.payment.payment_flow_support import build_payment_sms_payload
from app.state_machine.handlers.payment.pickup_sms_resolver import (
    PickupSmsDecision,
    resolve_pickup_sms_decision,
)
from app.state_machine.models.conversation_state import ConversationState


class WaitingForPickupSmsPermissionHandler(BaseHandler):
    """Handles the SMS-permission step in the pickup checkout flow.

    After the order is placed and the payment link is generated, the bot
    offers to text a payment link or let the customer pay on arrival.
    Resolution is delegated entirely to resolve_pickup_sms_decision().
    """

    def __init__(self, cart_summary_builder, sms_service, checkout_service) -> None:
        self.cart_summary_builder = cart_summary_builder
        self.sms_service = sms_service
        self.checkout_service = checkout_service

    def handle(self, intent, context, user_text, session=None):
        if session is None:
            return HandlerResult(
                next_state=ConversationState.ERROR_RECOVERY,
                response_key="confirmation_state_error",
            )

        if session.conversation_state != ConversationState.WAITING_FOR_PICKUP_SMS_PERMISSION:
            return HandlerResult(
                next_state=ConversationState.ERROR_RECOVERY,
                response_key="confirmation_state_error",
            )

        control_intent = resolve_control_intent(
            user_text,
            intent,
            getattr(context.last_nlu, "model_sub_intent", None),
            ConversationState.WAITING_FOR_PICKUP_SMS_PERMISSION,
            context,
            nlu_result=context.last_nlu,
            intent_confidence=context.last_intent_confidence,
        )

        decision = resolve_pickup_sms_decision(user_text, intent, control_intent)
        delivery = context.delivery_address

        if decision == PickupSmsDecision.SEND_SMS:
            if delivery.has_phone_number and delivery.payment_link:
                order_summary = self.cart_summary_builder.build(session.cart)
                return HandlerResult(
                    next_state=ConversationState.COMPLETED,
                    response_key="pickup_sms_sent_end_call",
                    response_payload={"order_number": delivery.order_number},
                    reset_context=True,
                    command={
                        "type": "SEND_SMS",
                        "payload": build_payment_sms_payload(
                            template="payment_link",
                            phone_number=delivery.customer_phone_number,
                            order_number=str(delivery.order_number or ""),
                            link=delivery.payment_link,
                            order_summary=order_summary,
                        ),
                    },
                )
            # Order is placed; phone/link missing — end gracefully without SMS.
            return HandlerResult(
                next_state=ConversationState.COMPLETED,
                response_key="pickup_end_call",
                response_payload={"order_number": delivery.order_number},
                reset_context=True,
                command={"type": "CLEAR_CART"},
            )

        if decision == PickupSmsDecision.PAY_ON_PICKUP:
            return HandlerResult(
                next_state=ConversationState.COMPLETED,
                response_key="pickup_no_sms_end_call",
                response_payload={"order_number": delivery.order_number},
                reset_context=True,
                command={"type": "CLEAR_CART"},
            )

        # UNKNOWN — re-prompt with the natural two-option prompt.
        return HandlerResult(
            next_state=ConversationState.WAITING_FOR_PICKUP_SMS_PERMISSION,
            response_key="pickup_repeat_sms_permission",
        )
