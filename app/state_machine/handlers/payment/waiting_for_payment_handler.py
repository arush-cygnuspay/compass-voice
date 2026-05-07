from __future__ import annotations

import time

from app.nlu.intent_resolution.intent import Intent
from app.services.checkout_service import CheckoutService
from app.state_machine.control_intent_resolver import (
    ControlIntentKind,
    log_control_intent_event,
    resolve_control_intent,
)
from app.state_machine.handler_result import HandlerResult
from app.state_machine.handlers.base_handler import BaseHandler
from app.state_machine.handlers.payment.payment_flow_support import (
    PAYMENT_LINK_RESEND_COOLDOWN_SECONDS,
    VOICE_PAYMENT_FALLBACK_AVAILABLE,
    append_payment_event,
    build_payment_sms_payload,
    ensure_payment_link_for_voice_session,
    log_phone_number_unavailable,
    verify_payment_for_order,
)
from app.state_machine.models.conversation_state import ConversationState
from app.state_machine.phase3_controls import is_live_agent_request
from app.config.voice_transfer import HUMAN_AGENT_TRANSFER_NUMBER


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

        delivery = context.delivery_address if context else None

        if is_live_agent_request(user_text):
            return HandlerResult(
                next_state=ConversationState.TRANSFERRING_TO_HUMAN_AGENT,
                response_key="transferring_to_human_agent",
                response_payload=append_payment_event(
                    {
                        "transfer_number": HUMAN_AGENT_TRANSFER_NUMBER,
                        "reason": "user_requested_agent",
                    },
                    event_name="user_requested_agent",
                    metadata={"reason": "payment_wait"},
                ),
                command={
                    "type": "transfer_call",
                    "transfer_number": HUMAN_AGENT_TRANSFER_NUMBER,
                },
            )

        control_intent = resolve_control_intent(
            user_text,
            intent,
            getattr(context.last_nlu, "model_sub_intent", None),
            ConversationState.WAITING_FOR_PAYMENT,
            context,
            nlu_result=context.last_nlu,
            intent_confidence=context.last_intent_confidence,
        )

        if control_intent is not None:
            if control_intent.kind == ControlIntentKind.PAYMENT_STAY_ON_CALL:
                delivery.payment_wait_mode = "stay_on_call"
                delivery.payment_session_state = "waiting_payment_stay_on_call"
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_PAYMENT,
                    response_key="payment_wait_stay_on_call",
                    response_payload=append_payment_event(
                        None,
                        event_name="payment_wait_mode_selected",
                        metadata={"mode": "stay_on_call"},
                    ),
                )

            if control_intent.kind == ControlIntentKind.PAYMENT_AFTER_CALL:
                delivery.payment_wait_mode = "after_call"
                delivery.payment_session_state = "waiting_payment_after_call"
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_PAYMENT,
                    response_key="payment_after_call_selected",
                    response_payload=append_payment_event(
                        None,
                        event_name="payment_after_call_selected",
                        metadata={"mode": "after_call"},
                    ),
                )

            if control_intent.kind == ControlIntentKind.PAYMENT_CANNOT_OPEN_LINK:
                delivery.payment_wait_mode = "after_call"
                delivery.payment_session_state = "waiting_payment_after_call"
                response_key = (
                    "payment_voice_fallback_available"
                    if VOICE_PAYMENT_FALLBACK_AVAILABLE
                    else "payment_link_after_call_fallback"
                )
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_PAYMENT,
                    response_key=response_key,
                    response_payload=append_payment_event(
                        None,
                        event_name="payment_after_call_selected",
                        metadata={"reason": "cannot_open_link"},
                    ),
                )

        # Defensive guard: checkout / finalize / done phrases while payment is
        # already in progress.  These cannot re-trigger the checkout flow;
        # acknowledge the current state instead.  This prevents accidental
        # re-routing if a coercion upstream fires before the payment handler
        # can assert its own state.
        _CHECKOUT_LIKE_INTENTS = {
            Intent.CHECKOUT,
            Intent.FINISH_ORDER,
            Intent.CONFIRM_ORDER,
            Intent.END_ADDING,
            Intent.START_ORDER,
        }
        if intent in _CHECKOUT_LIKE_INTENTS:
            if getattr(delivery, "payment_wait_mode", None) == "after_call":
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_PAYMENT,
                    response_key="payment_after_call_selected",
                )
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_PAYMENT,
                response_key="waiting_for_payment",
            )

        if intent in self.COMPLETE_PAYMENT_INTENTS or (
            control_intent is not None and control_intent.kind == ControlIntentKind.AFFIRM
        ):
            order_number = getattr(delivery, "order_number", None) if delivery else None
            return verify_payment_for_order(
                checkout_service=self.checkout_service,
                order_number=order_number,
                pending_state=ConversationState.WAITING_FOR_PAYMENT,
                pending_response_key="payment_not_confirmed_yet",
                delivery=delivery,
            )

        if intent in self.RESEND_PAYMENT_INTENTS:
            now = time.time()
            last_resend_at = float(getattr(delivery, "last_payment_link_resend_at_epoch", 0.0) or 0.0)
            if last_resend_at and (now - last_resend_at) < PAYMENT_LINK_RESEND_COOLDOWN_SECONDS:
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_PAYMENT,
                    response_key="payment_link_resend_cooldown",
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
            delivery.payment_status = "payment_pending"
            delivery.checkout_status = None

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
            delivery.last_payment_link_resend_at_epoch = now

            if not delivery.has_phone_number:
                log_phone_number_unavailable(
                    session=session,
                    consumer="waiting_for_payment_handler.resend",
                )
                delivery.payment_link_delivery_channel = "in_session"
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_PAYMENT,
                    response_key="payment_link_resent",
                    response_payload={
                        "order_number": delivery.order_number,
                        "payment_link": delivery.payment_link,
                        "payment_link_delivery_channel": "in_session",
                    },
                )

            delivery.payment_link_delivery_channel = "sms"
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_PAYMENT,
                response_key="payment_link_resent",
                response_payload={
                    "order_number": delivery.order_number,
                    "payment_link_delivery_channel": "sms",
                },
                command={
                    "type": "SEND_SMS",
                    "payload": build_payment_sms_payload(
                        template="payment_link",
                        phone_number=delivery.customer_phone_number,
                        order_number=str(delivery.order_number),
                        link=delivery.payment_link,
                        order_summary=order_summary,
                    ),
                },
            )

        if intent in self.VERIFY_STATUS_INTENTS:
            order_number = getattr(delivery, "order_number", None) if delivery else None
            return verify_payment_for_order(
                checkout_service=self.checkout_service,
                order_number=order_number,
                pending_state=ConversationState.WAITING_FOR_PAYMENT,
                pending_response_key="waiting_for_payment",
                delivery=delivery,
            )

        if control_intent is not None and control_intent.kind == ControlIntentKind.CANCEL:
            log_control_intent_event(
                "control_intent_action",
                state=ConversationState.WAITING_FOR_PAYMENT.value,
                action="payment_cancel_blocked",
                kind=control_intent.kind.value,
            )
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_PAYMENT,
                response_key="cannot_cancel_during_checkout",
            )

        if control_intent is not None and control_intent.kind == ControlIntentKind.META_CLARIFY:
            log_control_intent_event(
                "meta_clarify_repeated",
                state=ConversationState.WAITING_FOR_PAYMENT.value,
                field_name="payment_wait",
            )

        if control_intent is not None and control_intent.kind == ControlIntentKind.OPTIONS_REQUEST:
            log_control_intent_event(
                "options_requested",
                state=ConversationState.WAITING_FOR_PAYMENT.value,
                field_name="payment_wait",
            )

        if (
            intent in self.STATUS_INTENTS
            or (control_intent is not None and control_intent.kind in {
                ControlIntentKind.DENY,
                ControlIntentKind.DONE,
                ControlIntentKind.META_CLARIFY,
                ControlIntentKind.OPTIONS_REQUEST,
            })
        ):
            if delivery.payment_wait_mode == "after_call":
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_PAYMENT,
                    response_key="payment_after_call_selected",
                )
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_PAYMENT,
                response_key="waiting_for_payment",
            )

        if delivery.payment_wait_mode == "after_call":
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_PAYMENT,
                response_key="payment_after_call_selected",
            )

        return HandlerResult(
            next_state=ConversationState.WAITING_FOR_PAYMENT,
            response_key="waiting_for_payment",
        )
