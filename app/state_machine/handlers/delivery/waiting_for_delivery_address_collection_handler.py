# app/state_machine/handlers/delivery/waiting_for_delivery_address_collection_handler.py
from __future__ import annotations

import re

from app.nlu.intent_resolution.intent import Intent
from app.session.session import Session
from app.state_machine.handlers.base_handler import BaseHandler
from app.state_machine.handlers.common.preorder_redirect_utils import (
    looks_like_ordering_request,
)
from app.state_machine.handlers.payment.payment_flow_support import (
    ensure_payment_link_for_voice_session,
)
from app.state_machine.handler_result import HandlerResult
from app.state_machine.models.conversation_context import ConversationContext
from app.state_machine.models.conversation_state import ConversationState


from app.state_machine.flow_sets import ORDERING_INTENTS as _ADDRESS_ORDERING_INTENTS


class WaitingForDeliveryAddressCollectionHandler(BaseHandler):
    YES_WORDS = {
        "yes", "yeah", "yep", "correct", "right",
        "that is correct", "thats correct", "yes it is", "yeah its correct"
    }
    NO_WORDS = {"no", "nope", "wrong", "incorrect"}

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

        if intent in {Intent.CANCEL, Intent.CANCEL_ORDER}:
            payload = (
                self.cart_summary_builder.build(session.cart)
                if session is not None and not session.cart.is_empty()
                else None
            )
            context.current_prompt_field = None
            return HandlerResult(
                next_state=ConversationState.CONFIRMING_ORDER,
                response_key="confirm_order_summary",
                response_payload=payload,
            )

        if step == "delivery_seed_confirmation":
            if self._is_affirm(intent, text):
                context.current_prompt_field = "delivery_house_number"
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_DELIVERY_ADDRESS_COLLECTION,
                    response_key="ask_for_delivery_house_number",
                )

            if self._is_deny(intent, text):
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
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_DELIVERY_ADDRESS_COLLECTION,
                    response_key="repeat_delivery_house_number",
                )
            delivery.house_number = house_number
            context.current_prompt_field = "delivery_house_number_confirmation"
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_DELIVERY_ADDRESS_COLLECTION,
                response_key="confirm_delivery_house_number",
                response_payload={"house_number": delivery.house_number},
            )

        if step == "delivery_house_number_confirmation":
            if self._is_affirm(intent, text):
                context.current_prompt_field = "delivery_street"
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_DELIVERY_ADDRESS_COLLECTION,
                    response_key="ask_for_delivery_street",
                )
            if self._is_deny(intent, text):
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
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_DELIVERY_ADDRESS_COLLECTION,
                    response_key="repeat_delivery_street",
                )
            delivery.street = street_value
            context.current_prompt_field = "delivery_street_confirmation"
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_DELIVERY_ADDRESS_COLLECTION,
                response_key="confirm_delivery_street",
                response_payload={"street": delivery.street},
            )

        if step == "delivery_street_confirmation":
            if self._is_affirm(intent, text):
                context.current_prompt_field = "delivery_secondary_address"
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_DELIVERY_ADDRESS_COLLECTION,
                    response_key="ask_for_delivery_secondary_address",
                )
            if self._is_deny(intent, text):
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
            if text in self.OPTIONAL_NONE_WORDS:
                delivery.secondary_address = None
                return self._finish(session, context)

            delivery.secondary_address = user_text.strip()
            context.current_prompt_field = "delivery_secondary_address_confirmation"
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_DELIVERY_ADDRESS_COLLECTION,
                response_key="confirm_delivery_secondary_address",
                response_payload={"secondary_address": delivery.secondary_address},
            )

        if step == "delivery_secondary_address_confirmation":
            if self._is_affirm(intent, text):
                return self._finish(session, context)
            if self._is_deny(intent, text):
                delivery.secondary_address = None
                context.current_prompt_field = "delivery_secondary_address"
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_DELIVERY_ADDRESS_COLLECTION,
                    response_key="ask_for_delivery_secondary_address",
                )
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_DELIVERY_ADDRESS_COLLECTION,
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

    @classmethod
    def _is_affirm(cls, intent: Intent, text: str) -> bool:
        return intent in {Intent.AFFIRM, Intent.CONFIRM} or text in cls.YES_WORDS

    @classmethod
    def _is_deny(cls, intent: Intent, text: str) -> bool:
        return intent in {Intent.DENY, Intent.CANCEL} or text in cls.NO_WORDS

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

        phone_number = delivery.customer_phone_number
        if not phone_number:
            return HandlerResult(
                next_state=ConversationState.CONFIRMING_ORDER,
                response_key="payment_link_send_failed",
                response_payload={
                    "reason": "missing_customer_phone_number",
                },
            )

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

        return HandlerResult(
            next_state=ConversationState.WAITING_FOR_PAYMENT,
            response_key="delivery_address_captured_resume_checkout",
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
