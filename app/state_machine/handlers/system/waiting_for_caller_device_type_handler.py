# app/state_machine/handlers/system/waiting_for_caller_device_type_handler.py
"""
Handler for the very first dialog gate: ask the caller whether they are
calling from a landline or a mobile phone.

Why this gate exists
--------------------
Several downstream features (payment links, checkout links, order
confirmation SMS) require us to send the caller a clickable URL via SMS.
Landline callers are limited to pickup-only service in this voice flow.
We confirm they want to proceed, then transfer the call to a live human
agent to complete the order. Mobile callers continue through the normal
pickup / delivery onboarding flow.
"""
from __future__ import annotations

import os

from app.nlu.intent_resolution.intent import Intent
from app.session.session import Session
from app.state_machine.handler_result import HandlerResult
from app.state_machine.handlers.base_handler import BaseHandler
from app.state_machine.models.conversation_context import ConversationContext
from app.state_machine.models.conversation_state import ConversationState

# Number that landline callers should be transferred to so a human can
# take their order. This is the public hand-off line for now.
DEFAULT_HUMAN_AGENT_TRANSFER_NUMBER = "+17036371136"
HUMAN_AGENT_TRANSFER_NUMBER = (
    os.getenv(
        "COMPASS_HUMAN_AGENT_TRANSFER_NUMBER",
        DEFAULT_HUMAN_AGENT_TRANSFER_NUMBER,
    ).strip()
    or DEFAULT_HUMAN_AGENT_TRANSFER_NUMBER
)


# Words / phrases used by callers to describe a landline. Kept generous
# because callers say all sorts of things ("home phone", "house phone",
# "from my office", etc.). Order matters only for readability.
LANDLINE_WORDS = (
    "landline",
    "land line",
    "land-line",
    "home phone",
    "house phone",
    "office phone",
    "work phone",
    "desk phone",
    "wired phone",
    "house line",
    "home line",
)

# Words / phrases used by callers to describe a mobile / cell phone.
MOBILE_WORDS = (
    "mobile",
    "cell phone",
    "cellphone",
    "cell",
    "smartphone",
    "smart phone",
    "iphone",
    "android",
    "my phone",
    "phone",
)


class WaitingForCallerDeviceTypeHandler(BaseHandler):
    """
    Resolves the very first turn of the call: landline vs mobile.

    On landline:
      - Set ``context.caller_device_type = "landline"``
      - Ask whether they want to proceed with pickup-only ordering
      - If they confirm, move to ``TRANSFERRING_TO_HUMAN_AGENT`` and
        emit a command instructing the transport layer (Twilio bridge)
        to redirect the live call to ``HUMAN_AGENT_TRANSFER_NUMBER``
        after the confirmation line finishes playing.

    On mobile:
      - Set ``context.caller_device_type = "phone"``
      - Move to ``WAITING_FOR_ORDER_TYPE`` and ask the standard
        pickup / delivery question.
    """

    def handle(
        self,
        intent: Intent,
        context: ConversationContext,
        user_text: str,
        session: Session | None = None,
    ) -> HandlerResult:
        normalized = " ".join((user_text or "").strip().lower().split())

        if (
            session is not None
            and session.conversation_state
            == ConversationState.WAITING_FOR_LANDLINE_PICKUP_CONFIRMATION
        ):
            return self._handle_landline_pickup_confirmation(
                intent=intent,
                normalized_text=normalized,
            )

        device_type = self._resolve_device_type(normalized)

        if device_type is None:
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_CALLER_DEVICE_TYPE,
                response_key="repeat_caller_device_type",
                response_payload=None,
            )

        if device_type == "landline":
            context.caller_device_type = "landline"
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_LANDLINE_PICKUP_CONFIRMATION,
                response_key="confirm_landline_pickup_only",
            )

        # Mobile / cell phone path — proceed to the normal onboarding.
        context.caller_device_type = "phone"
        return HandlerResult(
            next_state=ConversationState.WAITING_FOR_ORDER_TYPE,
            response_key="ask_for_order_type",
            response_payload=None,
        )

    @staticmethod
    def _resolve_device_type(text: str) -> str | None:
        if not text:
            return None

        # Landline checks come first because some landline phrases
        # ("home phone", "office phone") contain the word "phone" and
        # would otherwise be mis-classified as mobile.
        for phrase in LANDLINE_WORDS:
            if phrase in text:
                return "landline"

        for phrase in MOBILE_WORDS:
            if phrase in text:
                return "phone"

        return None

    def _handle_landline_pickup_confirmation(
        self,
        *,
        intent: Intent,
        normalized_text: str,
    ) -> HandlerResult:
        if intent in {Intent.AFFIRM, Intent.CONFIRM} or self._is_yes_like(normalized_text):
            return HandlerResult(
                next_state=ConversationState.TRANSFERRING_TO_HUMAN_AGENT,
                response_key="transferring_to_human_agent",
                response_payload={
                    "transfer_number": HUMAN_AGENT_TRANSFER_NUMBER,
                },
                command={
                    "type": "transfer_call",
                    "transfer_number": HUMAN_AGENT_TRANSFER_NUMBER,
                },
            )

        if intent in {Intent.DENY, Intent.CANCEL, Intent.CANCEL_ORDER} or self._is_no_like(normalized_text):
            return HandlerResult(
                next_state=ConversationState.COMPLETED,
                response_key="landline_pickup_declined",
            )

        return HandlerResult(
            next_state=ConversationState.WAITING_FOR_LANDLINE_PICKUP_CONFIRMATION,
            response_key="repeat_landline_pickup_only",
        )

    @staticmethod
    def _is_yes_like(text: str) -> bool:
        if not text:
            return False
        return any(
            phrase in text
            for phrase in (
                "yes",
                "yeah",
                "yep",
                "proceed",
                "continue",
                "go ahead",
                "place the order",
                "place my order",
            )
        )

    @staticmethod
    def _is_no_like(text: str) -> bool:
        if not text:
            return False
        return any(
            phrase in text
            for phrase in (
                "no",
                "nope",
                "cancel",
                "stop",
                "not now",
                "do not",
                "don't",
            )
        )
