"""Payment-flow orchestration extracted from TurnEngine.

Owns:
- Payment-prompt cooldown intervals (env-loaded constants).
- ``_should_suppress_payment_prompt_replay`` cooldown logic.
- Cart-edit confirmation resume.
- Payment event emission (from payload + from command).
- The auto-payment-check dispatch handler.

Behavior moved verbatim from ``turn_engine.py``.
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from app.config.payment import get_payment_config
from app.contracts.command_result import CommandResult
from app.core.command_executor import CommandExecutor
from app.core.payment_response_classifier import PaymentResponseClassifier
from app.core.response_builder import ResponseBuilder
from app.core.session_response_writer import SessionResponseWriter
from app.core.turn_diagnostics import TurnDiagnostics
from app.logging.payment_event_logger import PaymentEventLogger
from app.nlu.intent_resolution.intent import Intent
from app.services.checkout_service import CheckoutService
from app.session.session import Session
from app.state_machine.handler_result import HandlerResult
from app.state_machine.handlers.payment.payment_flow_support import (
    verify_payment_for_order,
)
from app.state_machine.models.conversation_state import ConversationState

if TYPE_CHECKING:
    from app.core.turn_engine import TurnOutput


# Sourced from config — no direct os.getenv at module level.
CHECKOUT_PENDING_REMINDER_INTERVAL_SECONDS: float = (
    get_payment_config().checkout_pending_reminder_interval_seconds
)
PAYMENT_PENDING_REMINDER_INTERVAL_SECONDS: float = (
    get_payment_config().payment_pending_reminder_interval_seconds
)


class PaymentFlowOrchestrator:
    """Composes payment-related orchestration concerns previously
    inlined on TurnEngine. State-free; all references injected."""

    def __init__(
        self,
        *,
        checkout_service: CheckoutService,
        payment_event_logger: PaymentEventLogger,
        response_writer: SessionResponseWriter,
        responder: ResponseBuilder,
        diagnostics: TurnDiagnostics,
        command_executor: CommandExecutor,
        cart_summary_builder: Any,
    ) -> None:
        self.checkout_service = checkout_service
        self.payment_event_logger = payment_event_logger
        self.response_writer = response_writer
        self.responder = responder
        self.diagnostics = diagnostics
        self.command_executor = command_executor
        self.cart_summary_builder = cart_summary_builder

    @staticmethod
    def _pending_prompt_interval_for_state(state: ConversationState) -> float:
        if state == ConversationState.WAITING_FOR_CHECKOUT_COMPLETION:
            return CHECKOUT_PENDING_REMINDER_INTERVAL_SECONDS
        if state == ConversationState.WAITING_FOR_PAYMENT:
            return PAYMENT_PENDING_REMINDER_INTERVAL_SECONDS
        return 0.0

    def _should_suppress_payment_prompt_replay(
        self,
        *,
        session: Session,
        prior_state: ConversationState,
        response_key: str,
    ) -> bool:
        cooldown_seconds = self._pending_prompt_interval_for_state(prior_state)
        if cooldown_seconds <= 0:
            return False

        if not PaymentResponseClassifier.is_payment_pending_response(
            state=prior_state,
            response_key=response_key,
        ):
            return False

        last_key = session.last_response_key
        last_at = session.last_response_at_epoch
        delivery = getattr(session.conversation_context, "delivery_address", None)
        if delivery is not None:
            if prior_state == ConversationState.WAITING_FOR_CHECKOUT_COMPLETION:
                last_key = (
                    getattr(delivery, "last_checkout_wait_response_key", None)
                    or getattr(delivery, "payment_status_last_response_key", None)
                    or last_key
                )
                last_at = (
                    getattr(delivery, "last_checkout_wait_prompt_at_epoch", None)
                    if getattr(delivery, "last_checkout_wait_prompt_at_epoch", None) is not None
                    else getattr(delivery, "payment_status_last_prompt_at_epoch", None)
                )
                if last_at is None:
                    last_at = session.last_response_at_epoch
            elif prior_state == ConversationState.WAITING_FOR_PAYMENT:
                last_key = (
                    getattr(delivery, "last_payment_wait_response_key", None)
                    or getattr(delivery, "payment_status_last_response_key", None)
                    or last_key
                )
                last_at = (
                    getattr(delivery, "last_payment_wait_prompt_at_epoch", None)
                    if getattr(delivery, "last_payment_wait_prompt_at_epoch", None) is not None
                    else getattr(delivery, "payment_status_last_prompt_at_epoch", None)
                )
                if last_at is None:
                    last_at = session.last_response_at_epoch
        if not last_key or last_at is None:
            return False

        if not PaymentResponseClassifier.is_payment_pending_response(
            state=prior_state,
            response_key=last_key,
        ):
            return False

        return (time.time() - float(last_at)) < cooldown_seconds

    def _maybe_resume_confirmation_after_cart_edit(
        self,
        *,
        session: Session,
        result: HandlerResult,
    ) -> HandlerResult:
        ctx = session.conversation_context
        if not ctx.resume_order_confirmation_after_edit:
            return result

        if result.response_key in {
            "action_cancelled",
            "item_removal_cancelled",
            "item_replacement_cancelled",
            "item_modification_cancelled",
        }:
            ctx.resume_order_confirmation_after_edit = False
            return HandlerResult(
                next_state=ConversationState.CONFIRMING_ORDER,
                response_key="confirm_order_summary",
                response_payload=self.cart_summary_builder.build(session.cart),
            )

        if session.conversation_state != ConversationState.IDLE:
            return result

        if result.response_key not in {"item_added_successfully", "item_removed_successfully"}:
            return result

        ctx.resume_order_confirmation_after_edit = False
        if session.cart.is_empty():
            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="cart_empty",
            )

        return HandlerResult(
            next_state=ConversationState.CONFIRMING_ORDER,
            response_key="confirm_order_summary",
            response_payload={
                **self.cart_summary_builder.build(session.cart),
                "updated_order": True,
            },
        )

    def _reset_payment_wait_tracking(self, session: Session) -> None:
        delivery = session.conversation_context.delivery_address
        delivery.payment_wait_mode = None
        delivery.payment_session_state = None
        delivery.payment_status = None
        delivery.checkout_status = None
        delivery.last_checkout_wait_prompt_at_epoch = None
        delivery.last_checkout_wait_response_key = None
        delivery.last_payment_wait_prompt_at_epoch = None
        delivery.last_payment_wait_response_key = None
        delivery.payment_status_last_prompt_at_epoch = None
        delivery.payment_status_last_response_key = None

    def _log_payment_event(
        self,
        *,
        session: Session,
        state: ConversationState,
        event_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        order_id = getattr(session.conversation_context.delivery_address, "order_number", None)
        self.payment_event_logger.log(
            event_name=event_name,
            session_id=self.diagnostics._safe_session_id(session),
            order_id=order_id,
            state=state.value,
            metadata=metadata,
        )

    def _emit_payment_events_from_payload(
        self,
        *,
        session: Session,
        state_before: ConversationState,
        payload: dict[str, Any] | None,
    ) -> None:
        if not payload:
            return

        for event in list(payload.get("_payment_events") or []):
            event_name = str((event or {}).get("event_name") or "").strip()
            if not event_name:
                continue
            metadata = dict((event or {}).get("metadata") or {})
            self._log_payment_event(
                session=session,
                state=state_before,
                event_name=event_name,
                metadata=metadata,
            )

    def _emit_payment_events_from_command(
        self,
        *,
        session: Session,
        state_before: ConversationState,
        response_key: str,
        command: dict[str, Any] | None,
        command_result: CommandResult | None,
    ) -> None:
        if not command or not command_result:
            return

        if command.get("type") != "SEND_SMS":
            return

        payload = command.get("payload") or {}
        template = str(payload.get("template") or "").strip()
        if template not in {"payment_link", "checkout_link"}:
            return

        metadata = {
            "template": template,
            "sid": command_result.sid,
        }
        if response_key in {"payment_link_sent", "checkout_link_sent"}:
            self._log_payment_event(
                session=session,
                state=state_before,
                event_name="checkout_link_sent",
                metadata=metadata,
            )
        elif response_key in {"payment_link_resent", "checkout_link_resent"}:
            self._log_payment_event(
                session=session,
                state=state_before,
                event_name="payment_link_resent",
                metadata=metadata,
            )

        if str(payload.get("summary_text") or "").strip():
            self._log_payment_event(
                session=session,
                state=state_before,
                event_name="order_summary_sms_sent",
                metadata=metadata,
            )

    def _handle_auto_payment_check(self, session: Session) -> "TurnOutput":
        """Verify payment status without going through the NLU pipeline.

        Called when the transport layer fires the ``__auto_payment_check__``
        sentinel.  Only meaningful in WAITING_FOR_PAYMENT and
        WAITING_FOR_CHECKOUT_COMPLETION states; all other states return a
        silent no-op so the call flow is never disrupted.
        """
        from app.core.turn_engine import TurnOutput

        ctx = session.conversation_context
        state = session.conversation_state

        if state == ConversationState.WAITING_FOR_PAYMENT:
            pending_state = ConversationState.WAITING_FOR_PAYMENT
            pending_key = "waiting_for_payment"
        elif state == ConversationState.WAITING_FOR_CHECKOUT_COMPLETION:
            pending_state = ConversationState.WAITING_FOR_CHECKOUT_COMPLETION
            pending_key = "waiting_for_checkout_completion"
        else:
            # State changed between scheduling and firing — do nothing.
            return self.response_writer._build_silent_output(
                response_key=session.last_response_key or "waiting_for_payment",
                response_payload=session.last_response_payload,
            )

        delivery = ctx.delivery_address
        order_number = getattr(delivery, "order_number", None)

        result = verify_payment_for_order(
            checkout_service=self.checkout_service,
            order_number=order_number,
            pending_state=pending_state,
            pending_response_key=pending_key,
            delivery=delivery,
        )

        if result.reset_context:
            ctx.reset_item_scope()
        if result.command:
            self.command_executor.execute(session, result.command)

        # Do NOT set session.conversation_state here — TurnEngine applies it
        # from TurnOutput.next_state after this method returns.
        suppress_payment_replay = self._should_suppress_payment_prompt_replay(
            session=session,
            prior_state=state,
            response_key=result.response_key,
        )

        if not suppress_payment_replay:
            self.response_writer._apply_session_response(
                session=session,
                intent=Intent.PAYMENT_STATUS,
                response_key=result.response_key,
                response_payload=result.response_payload,
            )
            self._emit_payment_events_from_payload(
                session=session,
                state_before=state,
                payload=result.response_payload,
            )

        is_completed = result.next_state == ConversationState.COMPLETED
        if suppress_payment_replay:
            return self.response_writer._build_silent_output(
                response_key=result.response_key,
                response_payload=result.response_payload,
                end_call_after_playback=is_completed,
                next_state=result.next_state,
            )

        return self.response_writer._hydrate_output(
            session=session,
            output=TurnOutput(
                response_key=result.response_key,
                response_payload=result.response_payload,
                end_call_after_playback=is_completed,
                next_state=result.next_state,
            ),
        )
