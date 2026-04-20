# app/state_machine/handlers/delivery/waiting_for_delivery_eligibility_handler.py
from __future__ import annotations

import re

from app.nlu.intent_resolution.intent import Intent
from app.session.session import Session
from app.state_machine.handlers.base_handler import BaseHandler
from app.state_machine.handler_result import HandlerResult
from app.state_machine.handlers.common.preorder_redirect_utils import (
    looks_like_ordering_request,
)
from app.state_machine.models.conversation_context import ConversationContext
from app.state_machine.models.conversation_state import ConversationState


from app.state_machine.flow_sets import ORDERING_INTENTS as _ORDERING_INTENTS

# Maps delivery step → the reprompt response key to use after redirecting
_STEP_REPROMPT_KEY = {
    "delivery_area": "repeat_delivery_area",
    "delivery_postal_code": "repeat_delivery_zip",
    "delivery_eligibility_confirmation": "repeat_delivery_area_zip_confirmation",
}


class WaitingForDeliveryEligibilityHandler(BaseHandler):
    YES_WORDS = {"yes", "yeah", "yep", "correct", "right", "that is correct", "thats correct", "yes it is", "yeah its correct"}
    NO_WORDS = {"no", "nope", "wrong", "incorrect"}

    def handle(
        self,
        intent: Intent,
        context: ConversationContext,
        user_text: str,
        session: Session | None = None,
    ) -> HandlerResult:
        text = self._normalize(user_text)
        delivery = context.delivery_address
        step = context.current_prompt_field or "delivery_area"

        # ── Ordering intents during delivery setup → redirect gracefully ──
        if intent in _ORDERING_INTENTS or looks_like_ordering_request(
            context,
            text,
            include_slots=False,
        ):
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_DELIVERY_ELIGIBILITY,
                response_key="ordering_blocked_need_delivery_info",
                response_payload={"step": step},
            )

        if intent in {Intent.CANCEL, Intent.CANCEL_ORDER}:
            context.order_type = None
            context.delivery_address_required = False
            context.delivery_address_confirmed = False
            context.onboarding_complete = False
            context.current_prompt_field = None
            delivery.reset_for_new_delivery()
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_ORDER_TYPE,
                response_key="ask_for_order_type",
            )

        if step == "delivery_area":
            if not self._looks_like_area(text):
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_DELIVERY_ELIGIBILITY,
                    response_key="repeat_delivery_area",
                )

            delivery.area = self._clean_area_text(text)
            context.current_prompt_field = "delivery_postal_code"
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_DELIVERY_ELIGIBILITY,
                response_key="ask_for_delivery_zip",
            )

        if step == "delivery_postal_code":
            zip_code = self._extract_zip(text)
            if not zip_code:
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_DELIVERY_ELIGIBILITY,
                    response_key="repeat_delivery_zip",
                )

            delivery.postal_code = zip_code
            context.current_prompt_field = "delivery_eligibility_confirmation"
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_DELIVERY_ELIGIBILITY,
                response_key="confirm_delivery_area_zip",
                response_payload={
                    "area": delivery.area,
                    "postal_code": delivery.postal_code,
                },
            )

        if step == "delivery_eligibility_confirmation":
            if self._is_affirm(intent, text):
                delivery.area_serviceable = True
                context.onboarding_complete = True
                context.current_prompt_field = None
                return HandlerResult(
                    next_state=ConversationState.IDLE,
                    response_key="delivery_area_confirmed",
                    response_payload={
                        "area": delivery.area,
                        "postal_code": delivery.postal_code,
                    },
                )

            if self._is_deny(intent, text):
                delivery.area = None
                delivery.postal_code = None
                delivery.area_serviceable = None
                context.current_prompt_field = "delivery_area"
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_DELIVERY_ELIGIBILITY,
                    response_key="ask_for_delivery_area",
                )

            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_DELIVERY_ELIGIBILITY,
                response_key="repeat_delivery_area_zip_confirmation",
                response_payload={
                    "area": delivery.area,
                    "postal_code": delivery.postal_code,
                },
            )

        context.current_prompt_field = "delivery_area"
        return HandlerResult(
            next_state=ConversationState.WAITING_FOR_DELIVERY_ELIGIBILITY,
            response_key="ask_for_delivery_area",
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
        prefixes = (
            "it's ",
            "it is ",
            "its ",
            "my area is ",
            "area is ",
            "the area is ",
        )
        for prefix in prefixes:
            if value.startswith(prefix):
                value = value[len(prefix):]
                break
        return value.strip(" ,.")

    @staticmethod
    def _extract_zip(text: str) -> str | None:
        match = re.search(r"\b\d{5}(?:-\d{4})?\b", text)
        if match:
            return match.group(0)

        tokens = (text or "").lower().replace("-", " ").split()
        word_to_digit = {
            "zero": "0",
            "oh": "0",
            "o": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
        }

        digits: list[str] = []
        i = 0
        while i < len(tokens):
            token = tokens[i]

            if token in {"double", "triple"} and i + 1 < len(tokens):
                next_token = tokens[i + 1]
                digit = word_to_digit.get(next_token)
                if digit:
                    repeat = 2 if token == "double" else 3
                    digits.extend([digit] * repeat)
                    i += 2
                    continue

            if token.isdigit():
                digits.extend(list(token))
                i += 1
                continue

            digit = word_to_digit.get(token)
            if digit:
                digits.append(digit)

            i += 1

        joined = "".join(digits)
        if len(joined) >= 5:
            return joined[:5]

        return None

    @staticmethod
    def _looks_like_area(text: str) -> bool:
        if not text:
            return False
        if re.fullmatch(r"\d{5}(?:-\d{4})?", text):
            return False
        return len(text) >= 2
