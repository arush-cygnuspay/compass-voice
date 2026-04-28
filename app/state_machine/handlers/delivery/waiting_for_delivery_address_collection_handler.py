# app/state_machine/handlers/delivery/waiting_for_delivery_address_collection_handler.py
from __future__ import annotations

import os
import re

from app.intent.confirmation_utils import resolve_confirmation_decision
from app.nlu.intent_resolution.intent import Intent
from app.session.session import Session
from app.state_machine.control_intent_resolver import (
    ControlIntentKind,
    log_control_intent_event,
    resolve_control_intent,
)
from app.state_machine.handlers.base_handler import BaseHandler
from app.state_machine.handlers.common.preorder_redirect_utils import (
    looks_like_ordering_request,
)
from app.state_machine.handlers.payment.payment_flow_support import (
    ensure_payment_link_for_voice_session,
    log_phone_number_unavailable,
)
from app.state_machine.handler_result import HandlerResult
from app.state_machine.models.conversation_context import ConversationContext
from app.state_machine.models.conversation_state import ConversationState
from app.state_machine.flow_sets import ORDERING_INTENTS as _ADDRESS_ORDERING_INTENTS


ADDRESS_FIELD_MAX_REPROMPTS = int(
    os.getenv("COMPASS_ADDRESS_FIELD_MAX_REPROMPTS", "3")
)


class WaitingForDeliveryAddressCollectionHandler(BaseHandler):
    OPTIONAL_NONE_WORDS = {
        "none",
        "no",
        "skip",
        "nothing",
        "no apartment",
        "no suite",
        "no unit",
        "no building",
        "no block",
        "no apartment or suite",
    }

    def __init__(self, cart_summary_builder, checkout_service):
        self.cart_summary_builder = cart_summary_builder
        self.checkout_service = checkout_service

    @staticmethod
    def _escalate_or_reprompt(
        *,
        context: ConversationContext,
        field_name: str,
        response_key: str,
        response_payload: dict | None = None,
    ) -> HandlerResult:
        """Bump the reprompt counter for `field_name` and either return the
        normal repeat HandlerResult (with `attempt_count` injected) or, once
        we hit the max threshold, escalate to a recovery state."""
        attempts = context.bump_reprompt(field_name)
        if attempts >= ADDRESS_FIELD_MAX_REPROMPTS:
            log_control_intent_event(
                "address_field_max_attempts_exceeded",
                field_name=field_name,
                attempts=attempts,
            )
            context.reset_reprompt(field_name)
            context.current_prompt_field = None
            return HandlerResult(
                next_state=ConversationState.TRANSFERRING_TO_HUMAN_AGENT,
                response_key="address_collection_giving_up",
                response_payload={
                    "field_name": field_name,
                    "attempt_count": attempts,
                },
            )
        payload = dict(response_payload or {})
        payload["attempt_count"] = attempts
        return HandlerResult(
            next_state=ConversationState.WAITING_FOR_DELIVERY_ADDRESS_COLLECTION,
            response_key=response_key,
            response_payload=payload,
        )

    def handle(
        self,
        intent: Intent,
        context: ConversationContext,
        user_text: str,
        session: Session | None = None,
    ) -> HandlerResult:
        delivery = context.delivery_address
        text = self._normalize(user_text)
        step = context.current_prompt_field or "delivery_seed_confirmation"
        decision = resolve_confirmation_decision(
            context.last_nlu,
            text,
            resolved_intent=intent,
            expect_confirmation=True,
        )
        control_intent = resolve_control_intent(
            user_text,
            intent,
            getattr(context.last_nlu, "model_sub_intent", None),
            ConversationState.WAITING_FOR_DELIVERY_ADDRESS_COLLECTION,
            context,
            nlu_result=context.last_nlu,
            intent_confidence=context.last_intent_confidence,
        )

        # ── Ordering intents during address collection → redirect gracefully ──
        if intent in _ADDRESS_ORDERING_INTENTS or looks_like_ordering_request(
            context,
            text,
            include_slots=False,
        ):
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_DELIVERY_ADDRESS_COLLECTION,
                response_key="ordering_blocked_need_delivery_address",
                response_payload={"step": step},
            )

        if intent == Intent.CANCEL_ORDER or decision == "cancel":
            payload = (
                self.cart_summary_builder.build(session.cart)
                if session is not None and not session.cart.is_empty()
                else None
            )
            context.current_prompt_field = None
            context.reprompt_attempts.clear()
            return HandlerResult(
                next_state=ConversationState.CONFIRMING_ORDER,
                response_key="confirm_order_summary",
                response_payload=payload,
            )

        if step == "delivery_seed_confirmation":
            if decision == "affirm":
                context.current_prompt_field = "delivery_house_number"
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_DELIVERY_ADDRESS_COLLECTION,
                    response_key="ask_for_delivery_house_number",
                )

            if decision == "deny":
                delivery.area = None
                delivery.postal_code = None
                context.current_prompt_field = "delivery_area"
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_DELIVERY_ADDRESS_COLLECTION,
                    response_key="ask_for_delivery_area",
                )

            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_DELIVERY_ADDRESS_COLLECTION,
                response_key="repeat_delivery_area_zip_confirmation",
                response_payload={
                    "area": delivery.area,
                    "postal_code": delivery.postal_code,
                },
            )

        if step == "delivery_area":
            cleaned_area = self._clean_area_text(text)
            if len(cleaned_area) < 2:
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_DELIVERY_ADDRESS_COLLECTION,
                    response_key="repeat_delivery_area",
                )
            delivery.area = cleaned_area
            context.current_prompt_field = "delivery_postal_code"
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_DELIVERY_ADDRESS_COLLECTION,
                response_key="ask_for_delivery_zip",
            )

        if step == "delivery_postal_code":
            zip_code = self._extract_zip(text)
            if not zip_code:
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_DELIVERY_ADDRESS_COLLECTION,
                    response_key="repeat_delivery_zip",
                )
            delivery.postal_code = zip_code
            context.current_prompt_field = "delivery_seed_confirmation"
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_DELIVERY_ADDRESS_COLLECTION,
                response_key="confirm_delivery_area_zip",
                response_payload={
                    "area": delivery.area,
                    "postal_code": delivery.postal_code,
                },
            )

        if step == "delivery_house_number":
            house_number = self._extract_house_number(user_text)
            if not house_number:
                return self._escalate_or_reprompt(
                    context=context,
                    field_name="delivery_house_number",
                    response_key="repeat_delivery_house_number",
                )
            delivery.house_number = house_number
            context.reset_reprompt("delivery_house_number")
            context.current_prompt_field = "delivery_house_number_confirmation"
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_DELIVERY_ADDRESS_COLLECTION,
                response_key="confirm_delivery_house_number",
                response_payload={"house_number": delivery.house_number},
            )

        if step == "delivery_house_number_confirmation":
            if decision == "affirm":
                context.current_prompt_field = "delivery_street"
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_DELIVERY_ADDRESS_COLLECTION,
                    response_key="ask_for_delivery_street",
                )
            if decision == "deny":
                delivery.house_number = None
                context.current_prompt_field = "delivery_house_number"
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_DELIVERY_ADDRESS_COLLECTION,
                    response_key="ask_for_delivery_house_number",
                )
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_DELIVERY_ADDRESS_COLLECTION,
                response_key="confirm_delivery_house_number",
                response_payload={"house_number": delivery.house_number},
            )

        if step == "delivery_street":
            street_value = self._clean_street_text(user_text)
            if len(street_value) < 2:
                return self._escalate_or_reprompt(
                    context=context,
                    field_name="delivery_street",
                    response_key="repeat_delivery_street",
                )
            delivery.street = street_value
            context.reset_reprompt("delivery_street")
            context.current_prompt_field = "delivery_street_confirmation"
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_DELIVERY_ADDRESS_COLLECTION,
                response_key="confirm_delivery_street",
                response_payload={"street": delivery.street},
            )

        if step == "delivery_street_confirmation":
            if decision == "affirm":
                context.current_prompt_field = "delivery_secondary_address"
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_DELIVERY_ADDRESS_COLLECTION,
                    response_key="ask_for_delivery_secondary_address",
                )
            if decision == "deny":
                delivery.street = None
                context.current_prompt_field = "delivery_street"
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_DELIVERY_ADDRESS_COLLECTION,
                    response_key="ask_for_delivery_street",
                )
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_DELIVERY_ADDRESS_COLLECTION,
                response_key="confirm_delivery_street",
                response_payload={"street": delivery.street},
            )

        if step == "delivery_secondary_address":
            is_skip = (
                text in self.OPTIONAL_NONE_WORDS
                or intent in {Intent.DENY}
                or (
                    control_intent is not None
                    and control_intent.kind == ControlIntentKind.SKIP
                )
            )
            if is_skip:
                delivery.secondary_address = None
                context.reset_reprompt("delivery_secondary_address")
                return self._finish(session, context)

            delivery.secondary_address = user_text.strip()
            context.reset_reprompt("delivery_secondary_address")
            context.current_prompt_field = "delivery_secondary_address_confirmation"
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_DELIVERY_ADDRESS_COLLECTION,
                response_key="confirm_delivery_secondary_address",
                response_payload={"secondary_address": delivery.secondary_address},
            )

        if step == "delivery_secondary_address_confirmation":
            if decision == "affirm":
                context.reset_reprompt("delivery_secondary_address")
                return self._finish(session, context)
            if decision == "deny":
                delivery.secondary_address = None
                context.reset_reprompt("delivery_secondary_address")
                context.current_prompt_field = "delivery_secondary_address"
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_DELIVERY_ADDRESS_COLLECTION,
                    response_key="ask_for_delivery_secondary_address",
                )
            return self._escalate_or_reprompt(
                context=context,
                field_name="delivery_secondary_address",
                response_key="confirm_delivery_secondary_address",
                response_payload={"secondary_address": delivery.secondary_address},
            )

        context.current_prompt_field = "delivery_seed_confirmation"
        return HandlerResult(
            next_state=ConversationState.WAITING_FOR_DELIVERY_ADDRESS_COLLECTION,
            response_key="confirm_delivery_area_zip",
            response_payload={
                "area": delivery.area,
                "postal_code": delivery.postal_code,
            },
        )

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join((text or "").strip().lower().split())

    @staticmethod
    def _clean_area_text(text: str) -> str:
        value = (text or "").strip().lower()
        prefixes = ("it's ", "it is ", "its ", "my area is ", "area is ", "the area is ")
        for prefix in prefixes:
            if value.startswith(prefix):
                value = value[len(prefix):]
                break
        return value.strip(" ,.")

    @staticmethod
    def _clean_street_text(text: str) -> str:
        value = (text or "").strip()
        prefixes = ("it's ", "it is ", "its ", "street is ", "my street is ")
        for prefix in prefixes:
            if value.lower().startswith(prefix):
                value = value[len(prefix):]
                break
        return value.strip(" ,.")

    @staticmethod
    def _extract_house_number(text: str) -> str | None:
        value = (text or "").strip()

        lowered = value.lower().strip()

        for prefix in ("it's ", "it is ", "its "):
            if lowered.startswith(prefix):
                value = value[len(prefix):].strip()
                lowered = value.lower()

        # Prefer explicit "house number X"
        match = re.search(r"\bhouse number\s+([A-Za-z0-9\-]+)\b", value, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Fallback: just a single token number-like answer
        single = value.strip(" ,.")
        if re.fullmatch(r"[A-Za-z0-9\-]+", single):
            bad = {"house", "number", "street", "road", "lane", "its", "it"}
            if single.lower() not in bad:
                return single

        return None

    def _finish(self, session: Session | None, context: ConversationContext) -> HandlerResult:
        delivery = context.delivery_address
        delivery.source = "voice"
        delivery.collected = True
        delivery.confirmed = True
        delivery.payment_link_send_attempts = 0

        context.delivery_address_confirmed = True
        context.current_prompt_field = None

        if session is None:
            return HandlerResult(
                next_state=ConversationState.ERROR_RECOVERY,
                response_key="confirmation_state_error",
            )

        order_summary = self.cart_summary_builder.build(session.cart)
        try:
            payment_info = ensure_payment_link_for_voice_session(
                checkout_service=self.checkout_service,
                session=session,
                order_summary=order_summary,
                address_source="voice",
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
                consumer="waiting_for_delivery_address_collection_handler.payment_link",
            )
            delivery.payment_link_delivery_channel = "in_session"
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_PAYMENT,
                response_key="delivery_address_captured_resume_checkout",
                response_payload={
                    "order_number": delivery.order_number,
                    "payment_link": delivery.payment_link,
                    "payment_link_delivery_channel": "in_session",
                },
            )

        delivery.payment_link_delivery_channel = "sms"
        return HandlerResult(
            next_state=ConversationState.WAITING_FOR_PAYMENT,
            response_key="delivery_address_captured_resume_checkout",
            response_payload={
                "order_number": delivery.order_number,
                "payment_link_delivery_channel": "sms",
            },
            command={
                "type": "SEND_SMS",
                "payload": {
                    "template": "payment_link",
                    "phone_number": delivery.customer_phone_number,
                    "order_number": str(delivery.order_number),
                    "link": delivery.payment_link,
                },
            },
        )
