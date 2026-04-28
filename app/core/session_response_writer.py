"""Response application + output construction.

Owns the writes to ``Session`` that record the final ``response_key`` /
``response_payload`` for a turn, and constructs the ``TurnOutput``
that the transport layer consumes. Behavior moved verbatim from
``turn_engine.py``.

The private static ``_is_payment_pending_response`` helper is duplicated
here so that ``_apply_session_response`` can call it without a
dependency on ``PaymentFlowOrchestrator`` (which will own the canonical
copy in a later commit). Both copies must remain identical.
"""
from __future__ import annotations

import time
from typing import Any

from typing import TYPE_CHECKING

from app.core.response_builder import ResponseBuilder
from app.menu.repository import MenuRepository
from app.nlu.intent_resolution.intent import Intent
from app.session.session import Session
from app.state_machine.models.conversation_state import ConversationState

if TYPE_CHECKING:
    from app.core.turn_engine import TurnOutput


class SessionResponseWriter:
    """Centralizes session-state writes that finalize a turn's response."""

    def __init__(self, responder: ResponseBuilder, menu_repo: MenuRepository) -> None:
        self.responder = responder
        self.menu_repo = menu_repo

    @staticmethod
    def _is_payment_pending_response(
        *,
        state: ConversationState,
        response_key: str,
    ) -> bool:
        if state == ConversationState.WAITING_FOR_PAYMENT:
            return response_key in {"waiting_for_payment", "payment_not_confirmed_yet"}
        if state == ConversationState.WAITING_FOR_CHECKOUT_COMPLETION:
            return response_key in {
                "waiting_for_checkout_completion",
                "payment_not_confirmed_yet",
            }
        return False

    def _apply_session_response(
        self,
        *,
        session: Session,
        intent: Intent,
        response_key: str,
        response_payload: dict[str, Any] | None,
    ) -> None:
        now = time.time()
        session.last_intent = intent
        session.last_response_key = response_key
        session.last_response_payload = response_payload
        session.last_response_at_epoch = now
        session.turn_count += 1

        delivery = getattr(session.conversation_context, "delivery_address", None)
        if delivery is not None:
            if self._is_payment_pending_response(
                state=session.conversation_state,
                response_key=response_key,
            ):
                delivery.payment_status_last_prompt_at_epoch = now
                delivery.payment_status_last_response_key = response_key
                if session.conversation_state == ConversationState.WAITING_FOR_CHECKOUT_COMPLETION:
                    delivery.last_checkout_wait_prompt_at_epoch = now
                    delivery.last_checkout_wait_response_key = response_key
                elif session.conversation_state == ConversationState.WAITING_FOR_PAYMENT:
                    delivery.last_payment_wait_prompt_at_epoch = now
                    delivery.last_payment_wait_response_key = response_key
            elif getattr(delivery, "payment_status_last_response_key", None):
                delivery.payment_status_last_prompt_at_epoch = None
                delivery.payment_status_last_response_key = None
                if session.conversation_state != ConversationState.WAITING_FOR_CHECKOUT_COMPLETION:
                    delivery.last_checkout_wait_prompt_at_epoch = None
                    delivery.last_checkout_wait_response_key = None
                if session.conversation_state != ConversationState.WAITING_FOR_PAYMENT:
                    delivery.last_payment_wait_prompt_at_epoch = None
                    delivery.last_payment_wait_response_key = None

    @staticmethod
    def _build_silent_output(
        *,
        response_key: str,
        response_payload: dict[str, Any] | None,
        end_call_after_playback: bool = False,
        transfer_call_to_number: str | None = None,
    ) -> "TurnOutput":
        from app.core.turn_engine import TurnOutput
        return TurnOutput(
            response_key=response_key,
            response_payload=response_payload,
            internal_response_text="",
            spoken_response_text="",
            end_call_after_playback=end_call_after_playback,
            transfer_call_to_number=transfer_call_to_number,
        )

    @staticmethod
    def _normalize_response_text(text: str | None) -> str:
        return " ".join((text or "").split()).strip()

    def _hydrate_output(
        self,
        *,
        session: Session,
        output: "TurnOutput",
    ) -> "TurnOutput":
        from app.core.turn_engine import TurnOutput
        internal_text = self._normalize_response_text(
            output.internal_response_text
            or self.responder.build(
                response_key=output.response_key,
                context=session.conversation_context,
                payload=output.response_payload,
            )
        )

        spoken_text = self._normalize_response_text(
            output.spoken_response_text or internal_text
        )

        return TurnOutput(
            response_key=output.response_key,
            response_payload=output.response_payload,
            internal_response_text=internal_text,
            spoken_response_text=spoken_text,
            end_call_after_playback=output.end_call_after_playback,
            transfer_call_to_number=output.transfer_call_to_number,
        )
