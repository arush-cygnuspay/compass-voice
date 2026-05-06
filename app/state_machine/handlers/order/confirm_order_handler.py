from __future__ import annotations

import os

from app.nlu.control_phrase_classifier import DEFAULT_CLASSIFIER
from app.nlu.intent_resolution.intent import Intent
from app.state_machine.control_intent_resolver import (
    ControlIntentKind,
    log_control_intent_event,
    resolve_control_intent,
)
from app.state_machine.handlers.base_handler import BaseHandler
from app.state_machine.handlers.payment.payment_flow_support import (
    build_payment_sms_payload,
    ensure_payment_link_for_voice_session,
    log_phone_number_unavailable,
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

    EDIT_INTENTS = {
        Intent.REMOVE_ITEM,
        Intent.MODIFY_ITEM,
        Intent.REPLACE_ITEM,
        Intent.UNDO_LAST,
    }

    def __init__(self, cart_summary_builder, sms_service, checkout_service):
        self.cart_summary_builder = cart_summary_builder
        self.sms_service = sms_service
        self.checkout_service = checkout_service

    def _build_pickup_checkout_result(self, session, context) -> HandlerResult:
        """Pickup-specific checkout path.

        Places the order (creates payment link) then asks the customer for
        permission to send an SMS confirmation.  The call ends after that
        single yes/no — no live payment waiting loop is started.
        """
        delivery = context.delivery_address
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
        delivery.payment_status = "payment_pending"
        delivery.payment_wait_mode = None
        delivery.payment_session_state = None
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
                next_state=ConversationState.CONFIRMING_ORDER,
                response_key="payment_link_send_failed",
            )

        if not delivery.has_phone_number:
            log_phone_number_unavailable(
                session=session,
                consumer="confirm_order_handler.pickup_no_phone",
            )
            return HandlerResult(
                next_state=ConversationState.COMPLETED,
                response_key="pickup_end_call",
                response_payload={"order_number": delivery.order_number},
                reset_context=True,
                command={"type": "CLEAR_CART"},
            )

        return HandlerResult(
            next_state=ConversationState.WAITING_FOR_PICKUP_SMS_PERMISSION,
            response_key="pickup_ask_sms_permission",
            response_payload={"order_number": delivery.order_number},
        )

    def _build_payment_link_result(self, session, context) -> HandlerResult:
        delivery = context.delivery_address
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
        delivery.payment_status = "payment_pending"
        delivery.payment_wait_mode = "undecided"
        delivery.payment_session_state = "waiting_payment_stay_on_call"
        delivery.checkout_status = None
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

        if not delivery.has_phone_number:
            log_phone_number_unavailable(
                session=session,
                consumer="confirm_order_handler.payment_link",
            )
            delivery.payment_link_delivery_channel = "in_session"
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_PAYMENT,
                response_key="payment_link_sent",
                response_payload={
                    "order_number": delivery.order_number,
                    "payment_link": delivery.payment_link,
                    "payment_link_delivery_channel": "in_session",
                },
            )

        delivery.payment_link_delivery_channel = "sms"
        return HandlerResult(
            next_state=ConversationState.WAITING_FOR_PAYMENT,
            response_key="payment_link_sent",
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

        control_intent = resolve_control_intent(
            user_text,
            intent,
            getattr(context.last_nlu, "model_sub_intent", None),
            ConversationState.CONFIRMING_ORDER,
            context,
            nlu_result=context.last_nlu,
            intent_confidence=context.last_intent_confidence,
        )

        if self._is_pending_cancel_scope_confirmation(context):
            if control_intent is not None and control_intent.kind == ControlIntentKind.AFFIRM:
                context.awaiting_confirmation_for = None
                return HandlerResult(
                    next_state=ConversationState.IDLE,
                    response_key="order_cancelled",
                )

            if intent in self.EDIT_INTENTS or (
                control_intent is not None and control_intent.kind == ControlIntentKind.DENY
            ):
                context.awaiting_confirmation_for = None
                return HandlerResult(
                    next_state=ConversationState.IDLE,
                    response_key="order_confirmation_declined",
                )

            return HandlerResult(
                next_state=ConversationState.CONFIRMING_ORDER,
                response_key="confirm_cancel_order_or_edit",
            )

        if control_intent is not None and control_intent.kind == ControlIntentKind.META_CLARIFY:
            log_control_intent_event(
                "meta_clarify_repeated",
                state=ConversationState.CONFIRMING_ORDER.value,
                field_name="order_confirmation",
            )
            return self._summary_result(session)

        if control_intent is not None and control_intent.kind == ControlIntentKind.OPTIONS_REQUEST:
            log_control_intent_event(
                "options_requested",
                state=ConversationState.CONFIRMING_ORDER.value,
                field_name="order_confirmation",
            )
            return self._summary_result(session)

        if intent in self.EDIT_INTENTS:
            log_control_intent_event(
                "control_intent_action",
                state=ConversationState.CONFIRMING_ORDER.value,
                action="route_to_order_correction",
                intent=intent.value,
            )
            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="order_confirmation_declined",
            )

        if control_intent is not None and control_intent.kind == ControlIntentKind.CANCEL:
            context.awaiting_confirmation_for = {"type": "confirm_full_order_cancellation"}
            return HandlerResult(
                next_state=ConversationState.CONFIRMING_ORDER,
                response_key="confirm_cancel_order_or_edit",
            )

        # When the NLU detects a checkout-family intent but confidence falls below
        # the global routing threshold, TurnEngine downgrades intent to UNKNOWN before
        # calling the handler. The raw effective intent is preserved in context.last_nlu,
        # so we consult it here as a state-local fallback. This is safe because this
        # handler only runs in CONFIRMING_ORDER, where the bot just asked "Would you
        # like to checkout?" — any checkout-family NLU signal is unambiguously affirmative.
        _nlu_checkout_signal = (
            context.last_nlu is not None
            and context.last_nlu.effective_intent in self.CHECKOUT_CONFIRM_INTENTS
        )

        # Phrase-based checkout detection for utterances like "i said checkout",
        # "oh yeah checkout", "place the order" that NLU may miss or score below
        # the global confidence threshold.  ControlPhraseClassifier handles them
        # deterministically using an exact-phrase + prefix-stripped match.
        # "just a cup" and other non-checkout phrases return action="none".
        _phrase_checkout_signal = (
            DEFAULT_CLASSIFIER.classify(
                user_text, ConversationState.CONFIRMING_ORDER.value
            ).action == "checkout"
        )

        if (
            intent in self.CHECKOUT_CONFIRM_INTENTS
            or _nlu_checkout_signal
            or _phrase_checkout_signal
            or (control_intent is not None and control_intent.kind == ControlIntentKind.AFFIRM)
        ):
            delivery = context.delivery_address

            if context.order_type == "delivery":
                if context.delivery_address_confirmed or delivery.collected or delivery.confirmed:
                    delivery.source = "voice"
                    return self._build_payment_link_result(session, context)

                if not delivery.area or not delivery.postal_code:
                    return HandlerResult(
                        next_state=ConversationState.WAITING_FOR_DELIVERY_ELIGIBILITY,
                        response_key="ask_for_delivery_area",
                        prompt_field="delivery_area",
                    )

                can_use_checkout_link = (
                    not FORCE_VOICE_ADDRESS_FALLBACK
                    and delivery.has_phone_number
                    and (
                        self.sms_service.is_configured()
                        or getattr(context, "caller_device_type", None) == "chat"
                    )
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
                    delivery.checkout_status = "checkout_sent"
                    delivery.payment_status = "payment_pending"
                    delivery.payment_wait_mode = "undecided"
                    delivery.payment_session_state = "waiting_payment_stay_on_call"

                    return HandlerResult(
                        next_state=ConversationState.WAITING_FOR_CHECKOUT_COMPLETION,
                        response_key="checkout_link_sent",
                        response_payload={"order_number": delivery.order_number},
                        command={
                            "type": "SEND_SMS",
                            "payload": build_payment_sms_payload(
                                template="checkout_link",
                                phone_number=delivery.customer_phone_number,
                                order_number=str(delivery.order_number),
                                link=delivery.address_form_link,
                                order_summary=self.cart_summary_builder.build(session.cart),
                            ),
                        },
                    )

                delivery.source = "voice"
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_DELIVERY_ADDRESS_COLLECTION,
                    response_key="checkout_link_unavailable_fallback_voice",
                    response_payload={
                        "area": delivery.area,
                        "postal_code": delivery.postal_code,
                    },
                    prompt_field="delivery_house_number",
                )

            delivery.source = delivery.source or "voice"
            return self._build_pickup_checkout_result(session, context)

        if intent == Intent.PAYMENT_STATUS:
            return HandlerResult(
                next_state=ConversationState.CONFIRMING_ORDER,
                response_key="payment_not_started",
            )

        if control_intent is not None and control_intent.kind == ControlIntentKind.DENY:
            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="order_confirmation_declined",
            )

        return HandlerResult(
            next_state=ConversationState.CONFIRMING_ORDER,
            response_key="confirm_order_summary_unclear",
            response_payload=self.cart_summary_builder.build(session.cart),
        )

    def _is_pending_cancel_scope_confirmation(self, context) -> bool:
        confirmation = getattr(context, "awaiting_confirmation_for", None)
        return isinstance(confirmation, dict) and confirmation.get("type") == "confirm_full_order_cancellation"

    def _summary_result(self, session) -> HandlerResult:
        return HandlerResult(
            next_state=ConversationState.CONFIRMING_ORDER,
            response_key="confirm_order_summary",
            response_payload=self.cart_summary_builder.build(session.cart),
        )
