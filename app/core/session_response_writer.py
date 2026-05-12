"""Response application + output construction.

Owns the writes to ``Session`` that record the final ``response_key`` /
``response_payload`` for a turn, and constructs the ``TurnOutput``
that the transport layer consumes. Behavior moved verbatim from
``turn_engine.py``.
"""
from __future__ import annotations

import time
from typing import Any

from typing import TYPE_CHECKING

from app.core.payment_response_classifier import PaymentResponseClassifier
from app.core.response_builder import ResponseBuilder
from app.menu.repository import MenuRepository
from app.nlu.intent_resolution.intent import Intent
from app.nlu.prompt_type import classify_response_key
from app.session.session import Session
from app.state_machine.models.conversation_state import ConversationState

if TYPE_CHECKING:
    from app.core.turn_engine import TurnOutput


class SessionResponseWriter:
    """Centralizes session-state writes that finalize a turn's response."""

    def __init__(self, responder: ResponseBuilder, menu_repo: MenuRepository) -> None:
        self.responder = responder
        self.menu_repo = menu_repo

    # Response keys that represent an unresolved global fallback. When one
    # of these is the final emission, we must NOT reset the turn-level
    # UNKNOWN miss counter - that counter drives NoInputEscalationPolicy,
    # which keeps the fallback path from looping forever. Any other
    # response_key indicates the system understood the turn well enough
    # to act on it, so the counter is safely reset.
    _UNRESOLVED_FALLBACK_KEYS: frozenset[str] = frozenset({
        "intent_not_allowed",
    })

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
        session.last_prompt_type = classify_response_key(response_key).value
        session.last_response_payload = response_payload
        session.last_response_at_epoch = now
        session.turn_count += 1

        # Single chokepoint: reset turn-level UNKNOWN counter on any
        # non-fallback emission. The intent_not_allowed path bumps the
        # counter BEFORE calling this method and then emits an
        # intent_not_allowed key, so the bump survives.
        if response_key not in self._UNRESOLVED_FALLBACK_KEYS:
            ctx = getattr(session, "conversation_context", None)
            if ctx is not None and hasattr(ctx, "reset_unknown"):
                ctx.reset_unknown()

        delivery = getattr(session.conversation_context, "delivery_address", None)
        if delivery is not None:
            if PaymentResponseClassifier.is_payment_pending_response(
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
        next_state: Any = None,
    ) -> "TurnOutput":
        from app.core.turn_engine import TurnOutput
        return TurnOutput(
            response_key=response_key,
            response_payload=response_payload,
            internal_response_text="",
            spoken_response_text="",
            end_call_after_playback=end_call_after_playback,
            transfer_call_to_number=transfer_call_to_number,
            next_state=next_state,
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
            next_state=output.next_state,
        )
