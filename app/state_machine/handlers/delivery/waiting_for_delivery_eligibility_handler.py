# app/state_machine/handlers/delivery/waiting_for_delivery_eligibility_handler.py
from __future__ import annotations

import logging
import re
import time
from typing import Any

from app.nlu.intent_resolution.confirmation_resolver import resolve_confirmation_decision
from app.nlu.intent_resolution.intent import Intent
from app.session.session import Session
from app.state_machine.control_intent_resolver import (
    control_intent_to_confirmation_decision,
    resolve_control_intent,
)
from app.state_machine.handlers.base_handler import BaseHandler
from app.state_machine.handler_result import HandlerResult
from app.state_machine.handlers.common.preorder_redirect_utils import (
    looks_like_ordering_request,
)
from app.state_machine.models.conversation_context import ConversationContext
from app.state_machine.models.conversation_state import ConversationState

from app.state_machine.flow_sets import ORDERING_INTENTS as _ORDERING_INTENTS

logger = logging.getLogger(__name__)


def _log_delivery_event(event_name: str, **data: Any) -> None:
    """Emit a structured delivery timing event. Mirrors log_control_intent_event."""
    logger.info(event_name, extra={"event_name": event_name, **data})


# Maps delivery step → the reprompt response key to use after redirecting
_STEP_REPROMPT_KEY = {
    "delivery_area": "repeat_delivery_area",
    "delivery_postal_code": "repeat_delivery_zip",
    "delivery_eligibility_confirmation": "repeat_delivery_area_zip_confirmation",
}


class WaitingForDeliveryEligibilityHandler(BaseHandler):
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
        decision = resolve_confirmation_decision(
            context.last_nlu,
            text,
            resolved_intent=intent,
            expect_confirmation=True,
        )
        if step == "delivery_eligibility_confirmation":
            decision = self._resolve_delivery_confirmation_decision(
                context=context,
                text=text,
                intent=intent,
            )

        # Raw ZIP replies are often mislabeled by NLU. Accept deterministic ZIP
        # input before any intent-based redirect logic.
        if step == "delivery_postal_code":
            zip_result = self._handle_delivery_postal_code(context, text)
            if zip_result is not None:
                return zip_result

        # Allow ZIP corrections while confirming delivery eligibility.
        if step == "delivery_eligibility_confirmation":
            zip_correction_result = self._handle_confirmation_zip_correction(
                context,
                text,
            )
            if zip_correction_result is not None:
                return zip_correction_result

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

        if intent == Intent.CANCEL_ORDER or decision == "cancel":
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
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_DELIVERY_ELIGIBILITY,
                response_key="repeat_delivery_zip",
            )

        if step == "delivery_eligibility_confirmation":
            if decision == "affirm":
                # Eligibility is confirmed locally after user verifies area + ZIP.
                # This is the only place area_serviceable is set — never during ZIP
                # capture. Log the check duration (pure in-process flag write).
                t_eligibility = time.perf_counter()
                delivery.area_serviceable = True
                context.onboarding_complete = True
                context.current_prompt_field = None
                eligibility_ms = (time.perf_counter() - t_eligibility) * 1000.0
                _log_delivery_event(
                    "delivery_eligibility_confirmed",
                    delivery_eligibility_check_ms=round(eligibility_ms, 3),
                    area=delivery.area,
                    postal_code=delivery.postal_code,
                )
                return HandlerResult(
                    next_state=ConversationState.IDLE,
                    response_key="delivery_area_confirmed",
                    response_payload={
                        "area": delivery.area,
                        "postal_code": delivery.postal_code,
                    },
                )

            if decision == "deny":
                delivery.area = None
                delivery.postal_code = None
                delivery.area_serviceable = None
                context.onboarding_complete = False
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

    def _resolve_delivery_confirmation_decision(
        self,
        *,
        context: ConversationContext,
        text: str,
        intent: Intent,
    ) -> str:
        control_intent = resolve_control_intent(
            transcript=text,
            detected_intent=intent,
            detected_sub_intent=getattr(context.last_nlu, "model_sub_intent", None),
            current_state=ConversationState.WAITING_FOR_DELIVERY_ELIGIBILITY,
            pending_context=context,
            nlu_result=context.last_nlu,
            intent_confidence=getattr(context.last_nlu, "intent_confidence", None),
        )
        decision = control_intent_to_confirmation_decision(control_intent)
        if decision != "unknown":
            return decision

        return resolve_confirmation_decision(
            context.last_nlu,
            text,
            resolved_intent=intent,
            expect_confirmation=True,
        )

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
        if not text:
            return None

        normalized = re.sub(r"[^a-z0-9\s-]", " ", text.lower())

        match = re.search(r"\b(\d{5})(?:-\d{4})?\b", normalized)
        if match:
            return match.group(1)

        spaced_digits_match = re.search(r"(?<!\d)((?:\d[\s-]*){5,9})(?!\d)", normalized)
        if spaced_digits_match:
            digits_only = re.sub(r"\D", "", spaced_digits_match.group(1))
            if len(digits_only) >= 5:
                return digits_only[:5]

        tokens = normalized.replace("-", " ").split()
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
        number_words = {
            "zero": 0,
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
            "eleven": 11,
            "twelve": 12,
            "thirteen": 13,
            "fourteen": 14,
            "fifteen": 15,
            "sixteen": 16,
            "seventeen": 17,
            "eighteen": 18,
            "nineteen": 19,
            "twenty": 20,
            "thirty": 30,
            "forty": 40,
            "fifty": 50,
            "sixty": 60,
            "seventy": 70,
            "eighty": 80,
            "ninety": 90,
        }

        def parse_number_phrase(candidate_tokens: list[str]) -> int | None:
            total = 0
            current = 0
            used = False

            for candidate in candidate_tokens:
                if candidate in number_words:
                    current += number_words[candidate]
                    used = True
                    continue

                if candidate == "hundred":
                    if current == 0:
                        current = 1
                    current *= 100
                    used = True
                    continue

                if candidate == "thousand":
                    if current == 0:
                        current = 1
                    total += current * 1000
                    current = 0
                    used = True
                    continue

                return None

            if not used:
                return None

            return total + current

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

        phrase_tokens: list[str] = []
        for token in tokens + [""]:
            if token in number_words or token in {"hundred", "thousand"}:
                phrase_tokens.append(token)
                continue

            if phrase_tokens:
                parsed_value = parse_number_phrase(phrase_tokens)
                if parsed_value is not None and 10000 <= parsed_value <= 99999:
                    return str(parsed_value)
                phrase_tokens = []

        return None

    @staticmethod
    def _looks_like_area(text: str) -> bool:
        if not text:
            return False
        if re.fullmatch(r"\d{5}(?:-\d{4})?", text):
            return False
        return len(text) >= 2

    def _handle_delivery_postal_code(
        self,
        context: ConversationContext,
        text: str,
    ) -> HandlerResult | None:
        t_start = time.perf_counter()
        zip_code = self._extract_zip(text)
        if not zip_code:
            return None

        context.delivery_address.postal_code = zip_code
        context.current_prompt_field = "delivery_eligibility_confirmation"
        payload = {
            "area": context.delivery_address.area,
            "postal_code": context.delivery_address.postal_code,
        }
        handler_ms = (time.perf_counter() - t_start) * 1000.0
        _log_delivery_event(
            "delivery_zip_captured",
            delivery_zip_handler_ms=round(handler_ms, 3),
            delivery_confirmation_build_ms=round(handler_ms, 3),
            delivery_eligibility_check_ms=0,
            zip=zip_code,
            area=context.delivery_address.area,
        )
        return HandlerResult(
            next_state=ConversationState.WAITING_FOR_DELIVERY_ELIGIBILITY,
            response_key="confirm_delivery_area_zip",
            response_payload=payload,
        )

    def _handle_confirmation_zip_correction(
        self,
        context: ConversationContext,
        text: str,
    ) -> HandlerResult | None:
        zip_code = self._extract_zip(text)
        if not zip_code:
            return None

        context.delivery_address.postal_code = zip_code
        context.current_prompt_field = "delivery_eligibility_confirmation"
        return HandlerResult(
            next_state=ConversationState.WAITING_FOR_DELIVERY_ELIGIBILITY,
            response_key="confirm_delivery_area_zip",
            response_payload={
                "area": context.delivery_address.area,
                "postal_code": context.delivery_address.postal_code,
            },
        )

