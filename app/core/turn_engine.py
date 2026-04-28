# app/core/turn_engine.py
"""Top-level orchestration entry point.

Composes six focused modules under ``app/core/``:
  * TurnDiagnostics — trace/log/snapshot helpers
  * SessionResponseWriter — session response writes + output construction
  * ItemQueueService — multi-item queue drain
  * PaymentFlowOrchestrator — payment events, cooldown, auto-check
  * FlowGate — flow shortcuts, rewrites, order-type gate
  * HandlerDispatcher — handler registry, dispatch, reprompt guardrail
  * NluOrchestrator — preprocessing + intent + slot resolution
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

from app.cart.read_models.cart_summary_builder import CartSummaryBuilder
from app.core.command_executor import CommandExecutor
from app.core.flow_control.flow_decision import FlowAction
from app.core.flow_control.flow_control_policy import FlowControlPolicy
from app.core.flow_gate import FlowGate
from app.core.handler_dispatcher import HandlerDispatcher
from app.core.item_queue_service import ItemQueueService
from app.core.nlu_orchestrator import INTENT_MIN_CONF, NluOrchestrator
from app.nlu.nlu_resolver import resolve_nlu
from app.core.payment_flow_orchestrator import PaymentFlowOrchestrator
from app.core.response_builder import ResponseBuilder
from app.core.session_response_writer import SessionResponseWriter
from app.core.turn_diagnostics import TurnDiagnostics
from app.logging.payment_event_logger import PaymentEventLogger
from app.logging.nlu_csv_logger import NluCsvLogger
from app.menu.repository import MenuRepository
from app.ml.intent.inference_intent import IntentBundle
from app.ml.slot.inference_slot import SlotBundle
from app.nlu.intent_resolution.intent import Intent
from app.nlu.nlu_result import NLUResult
from app.nlu.query_normalization.text_preprocessor import preprocess_turn_text
from app.services.checkout_service import CheckoutService
from app.services.sms_service import SmsService
from app.session.session import Session
from app.state_machine.handlers.system.waiting_for_caller_device_type_handler import (
    HUMAN_AGENT_TRANSFER_NUMBER,
    WaitingForCallerDeviceTypeHandler,
)
from app.state_machine.handlers.order.waiting_for_order_type_handler import (
    WaitingForOrderTypeHandler,
)
from app.state_machine.models.conversation_state import ConversationState
from app.state_machine.resume_prompt_builder import ResumePromptBuilder
from app.state_machine.state_router import StateRouter


@dataclass(frozen=True, slots=True)
class TurnOutput:
    response_key: str
    response_payload: dict[str, Any] | None = None
    internal_response_text: str | None = None
    spoken_response_text: str | None = None
    end_call_after_playback: bool = False
    # If set, the voice transport layer should redirect the live call to
    # this PSTN number after the spoken response finishes playing. Used
    # to hand landline callers off to a human agent.
    transfer_call_to_number: str | None = None


ROUTE_DEBUG_ENABLED = os.getenv("COMPASS_ROUTE_DEBUG_ENABLED", "0") == "1"


CONFIRMING_ORDER_EXIT_TO_IDLE_INTENTS: set[Intent] = {
    Intent.ASK_ITEM_INFO,
    Intent.ASK_MENU_INFO,
    Intent.ASK_OPTIONS,
    Intent.AVAILABILITY_QUERY,
    Intent.BROWSE_MENU,
    Intent.BROWSE_CATEGORY,
    Intent.RECOMMENDATION_QUERY,
    Intent.SHOW_MENU,
}

from app.state_machine.flow_sets import (
    DELIVERY_GATING_ALLOWED_CONTROL_INTENTS,
    WAITING_STATE_ALLOWED_CONTROL_INTENTS,
)


class TurnEngine:
    def __init__(
        self,
        router: StateRouter,
        menu_repo: MenuRepository,
        intent_bundle: IntentBundle,
        slot_bundle: SlotBundle,
        responder: ResponseBuilder,
        sms_service: SmsService,
        nlu_logger: NluCsvLogger | None = None,
    ) -> None:
        self.router = router
        self.menu_repo = menu_repo
        self.intent_bundle = intent_bundle
        self.slot_bundle = slot_bundle
        self.cart_summary_builder = CartSummaryBuilder(menu_repo)
        self.flow_policy = FlowControlPolicy()
        self.nlu_logger = nlu_logger or NluCsvLogger()
        self.payment_event_logger = PaymentEventLogger()
        self.resume_prompt_builder = ResumePromptBuilder()
        self.responder = responder
        self.sms_service = sms_service
        self.command_executor = CommandExecutor(sms_service)
        self.checkout_service = CheckoutService()
        self.diagnostics = TurnDiagnostics(
            menu_repo=menu_repo,
            nlu_logger=self.nlu_logger,
            responder=responder,
        )
        self.response_writer = SessionResponseWriter(
            responder=responder,
            menu_repo=menu_repo,
        )
        self.dispatcher = HandlerDispatcher(
            menu_repo=menu_repo,
            cart_summary_builder=self.cart_summary_builder,
            sms_service=sms_service,
            checkout_service=self.checkout_service,
            responder=responder,
            command_executor=self.command_executor,
            diagnostics=self.diagnostics,
        )
        self.item_queue_service = ItemQueueService(
            handlers=self.dispatcher.handlers,
            command_executor=self.command_executor,
        )
        self.payment_flow = PaymentFlowOrchestrator(
            checkout_service=self.checkout_service,
            payment_event_logger=self.payment_event_logger,
            response_writer=self.response_writer,
            responder=responder,
            diagnostics=self.diagnostics,
            command_executor=self.command_executor,
            cart_summary_builder=self.cart_summary_builder,
        )
        self.flow_gate = FlowGate(
            handlers=self.dispatcher.handlers,
            menu_repo=menu_repo,
            cart_summary_builder=self.cart_summary_builder,
            response_writer=self.response_writer,
            diagnostics=self.diagnostics,
            payment_flow=self.payment_flow,
            resume_prompt_builder=self.resume_prompt_builder,
            nlu_logger=self.nlu_logger,
        )
        self.nlu = NluOrchestrator(
            intent_bundle=intent_bundle,
            slot_bundle=slot_bundle,
            diagnostics=self.diagnostics,
        )

    def process_turn(
        self,
        session: Session,
        user_text: str,
        trace: Any | None = None,
    ) -> TurnOutput:
        t_total_start = time.perf_counter()
        ctx = session.conversation_context

        if session.conversation_state == ConversationState.COMPLETED:
            return self.response_writer._hydrate_output(
                session=session,
                output=TurnOutput(
                    response_key="order_completed",
                    response_payload=session.last_response_payload,
                    end_call_after_playback=True,
                ),
            )

        # Once a landline caller has been handed off to a human agent the
        # voice transport layer will tear down the WebSocket / call. If a
        # spurious turn still arrives in this state we just acknowledge
        # and tell the bridge to end the agent's side of the call.
        if session.conversation_state == ConversationState.TRANSFERRING_TO_HUMAN_AGENT:
            return self.response_writer._hydrate_output(
                session=session,
                output=TurnOutput(
                    response_key="transferring_to_human_agent",
                    response_payload={
                        "transfer_number": HUMAN_AGENT_TRANSFER_NUMBER,
                    },
                    transfer_call_to_number=HUMAN_AGENT_TRANSFER_NUMBER,
                    end_call_after_playback=True,
                ),
            )

        # Caller-device-type gate. Must run BEFORE the order-type gate
        # because landline callers are routed to a live human and never
        # reach the pickup / delivery question.
        if session.conversation_state == ConversationState.WAITING_FOR_CALLER_DEVICE_TYPE:
            device_handler: WaitingForCallerDeviceTypeHandler = self.dispatcher.get_handler(
                "waiting_for_caller_device_type_handler"
            )
            device_result = device_handler.handle(
                intent=Intent.UNKNOWN,
                context=ctx,
                user_text=user_text,
                session=session,
            )

            session.conversation_state = device_result.next_state
            self.response_writer._apply_session_response(
                session=session,
                intent=Intent.UNKNOWN,
                response_key=device_result.response_key,
                response_payload=device_result.response_payload,
            )

            transfer_number: str | None = None
            command = device_result.command or {}
            if command.get("type") == "transfer_call":
                transfer_number = command.get("transfer_number")

            output = self.response_writer._hydrate_output(
                session=session,
                output=TurnOutput(
                    response_key=device_result.response_key,
                    response_payload=device_result.response_payload,
                    transfer_call_to_number=transfer_number,
                    # Once we hand the call off there is nothing for the
                    # voice agent to do, so end the agent leg cleanly
                    # after the goodbye plays. The transfer command is
                    # carried out by the transport layer and supersedes
                    # a plain hangup if both are set.
                    end_call_after_playback=transfer_number is not None,
                ),
            )
            self.diagnostics._log_if_enabled(
                session=session,
                state_before=ConversationState.WAITING_FOR_CALLER_DEVICE_TYPE,
                nlu=NLUResult(
                    effective_intent=Intent.UNKNOWN,
                    intent_confidence=1.0,
                    raw_text=user_text,
                    normalized_text=preprocess_turn_text(user_text).normalized_text,
                ),
                result=device_result,
                response_key=device_result.response_key,
                response_payload=device_result.response_payload,
                next_state=device_result.next_state,
                total_ms=(time.perf_counter() - t_total_start) * 1000.0,
            )
            return output

        # Auto payment-check probe injected by the transport layer after
        # PAYMENT_AUTO_CHECK_DELAY_SECONDS of silence.  Bypass NLU entirely
        # and call the payment verifier directly — O(1) path, no model inference.
        if user_text == "__auto_payment_check__":
            return self.payment_flow._handle_auto_payment_check(session)

        self.flow_gate._normalize_order_type_gate_state(session)

        if session.conversation_state == ConversationState.WAITING_FOR_ORDER_TYPE:
            handler: WaitingForOrderTypeHandler = self.dispatcher.get_handler("waiting_for_order_type_handler")
            preprocessed = preprocess_turn_text(user_text)
            if self.intent_bundle is not None and self.slot_bundle is not None:
                gate_nlu = resolve_nlu(
                    raw_text=preprocessed.cleaned_text,
                    normalized_text=preprocessed.normalized_text,
                    state=session.conversation_state,
                    pending_action=ctx.pending_action,
                    intent_bundle=self.intent_bundle,
                    slot_bundle=self.slot_bundle,
                )
            else:
                gate_nlu = NLUResult(
                    effective_intent=Intent.UNKNOWN,
                    intent_confidence=0.0,
                    raw_text=preprocessed.cleaned_text,
                    normalized_text=preprocessed.normalized_text,
                )
            ctx.set_last_nlu(user_text=preprocessed.cleaned_text, nlu=gate_nlu)
            gate_intent = (
                gate_nlu.effective_intent
                if gate_nlu.intent_confidence >= INTENT_MIN_CONF
                else Intent.UNKNOWN
            )
            gate_result = handler.handle(
                intent=gate_intent,
                context=ctx,
                user_text=gate_nlu.normalized_text,
                session=session,
            )

            session.conversation_state = gate_result.next_state
            self.response_writer._apply_session_response(
                session=session,
                intent=gate_intent,
                response_key=gate_result.response_key,
                response_payload=gate_result.response_payload,
            )

            total_ms = (time.perf_counter() - t_total_start) * 1000.0
            output = self.response_writer._hydrate_output(
                session=session,
                output=TurnOutput(
                    response_key=gate_result.response_key,
                    response_payload=gate_result.response_payload,
                ),
            )
            self.diagnostics._log_if_enabled(
                session=session,
                state_before=ConversationState.WAITING_FOR_ORDER_TYPE,
                nlu=gate_nlu,
                result=gate_result,
                response_key=gate_result.response_key,
                response_payload=gate_result.response_payload,
                next_state=gate_result.next_state,
                total_ms=total_ms,
            )
            return output

        state_before = session.conversation_state
        ctx.last_user_text = user_text

        self.diagnostics._trace_set_initial_fields(
            trace=trace,
            session=session,
            user_text=user_text,
            state_before=state_before,
            engine_start_monotonic=t_total_start,
        )

        resolution = self.nlu.resolve(session=session, user_text=user_text)
        cleaned_text = resolution.cleaned_text
        normalized_text = resolution.normalized_text
        nlu = resolution.nlu
        intent_result = resolution.intent_result
        t_preprocess = resolution.preprocess_ms / 1000.0
        t_nlu = resolution.nlu_ms / 1000.0

        self.diagnostics._trace_set_attr(trace, "cleaned_text", cleaned_text)
        self.diagnostics._trace_set_attr(trace, "normalized_text", normalized_text)
        self.diagnostics._trace_set_nlu_fields(trace=trace, nlu=nlu)

        phase3_shortcut = self.flow_gate._handle_phase3_control_shortcuts(
            session=session,
            state_before=state_before,
            intent_result=intent_result,
            nlu=nlu,
        )
        if phase3_shortcut is not None:
            self.response_writer._apply_session_response(
                session=session,
                intent=intent_result.intent,
                response_key=phase3_shortcut.response_key,
                response_payload=phase3_shortcut.response_payload,
            )
            self.payment_flow._emit_payment_events_from_payload(
                session=session,
                state_before=state_before,
                payload=phase3_shortcut.response_payload,
            )

            total_ms = (time.perf_counter() - t_total_start) * 1000.0
            self.diagnostics._log_if_enabled(
                session=session,
                state_before=state_before,
                nlu=nlu,
                result=None,
                response_key=phase3_shortcut.response_key,
                response_payload=phase3_shortcut.response_payload,
                preprocess_ms=t_preprocess * 1000.0,
                nlu_ms=t_nlu * 1000.0,
                flow_ms=0.0,
                route_ms=0.0,
                handler_ms=0.0,
                total_ms=total_ms,
            )
            self.diagnostics._finalize_trace_and_timing(
                trace=trace,
                session=session,
                response_key=phase3_shortcut.response_key,
                total_start_monotonic=t_total_start,
                total_ms=total_ms,
                preprocess_ms=t_preprocess * 1000.0,
                nlu_ms=t_nlu * 1000.0,
                flow_ms=0.0,
                route_ms=0.0,
                handler_ms=0.0,
            )
            return self.response_writer._hydrate_output(
                session=session,
                output=phase3_shortcut,
            )

        session.conversation_state = self.flow_gate._rewrite_confirming_order_to_idle_if_needed(
            session=session,
            intent=intent_result.intent,
        )

        intent_result, shortcut_output = self.flow_gate._apply_idle_shortcuts(session, intent_result)
        if shortcut_output is not None:
            self.response_writer._apply_session_response(
                session=session,
                intent=intent_result.intent,
                response_key=shortcut_output.response_key,
                response_payload=shortcut_output.response_payload,
            )

            total_ms = (time.perf_counter() - t_total_start) * 1000.0
            self.diagnostics._log_if_enabled(
                session=session,
                state_before=state_before,
                nlu=nlu,
                result=None,
                response_key=shortcut_output.response_key,
                response_payload=shortcut_output.response_payload,
                preprocess_ms=t_preprocess * 1000.0,
                nlu_ms=t_nlu * 1000.0,
                flow_ms=0.0,
                route_ms=0.0,
                handler_ms=0.0,
                total_ms=total_ms,
            )
            self.diagnostics._finalize_trace_and_timing(
                trace=trace,
                session=session,
                response_key=shortcut_output.response_key,
                total_start_monotonic=t_total_start,
                total_ms=total_ms,
                preprocess_ms=t_preprocess * 1000.0,
                nlu_ms=t_nlu * 1000.0,
                flow_ms=0.0,
                route_ms=0.0,
                handler_ms=0.0,
            )
            return self.response_writer._hydrate_output(
                session=session,
                output=shortcut_output,
            )

        intent_result = self.flow_gate._rewrite_idle_unknown_menu_followup(
            session=session,
            intent_result=intent_result,
            normalized_text=nlu.normalized_text,
        )

        t0 = time.perf_counter()
        flow = self.flow_policy.evaluate(
            state=session.conversation_state,
            intent=intent_result.intent,
            context=ctx,
        )
        t_flow = time.perf_counter() - t0

        if flow.action == FlowAction.BLOCK:
            payload = dict(flow.response_payload or {})
            response_key = flow.response_key or "flow_blocked"

            self.response_writer._apply_session_response(
                session=session,
                intent=intent_result.intent,
                response_key=response_key,
                response_payload=payload,
            )

            total_ms = (time.perf_counter() - t_total_start) * 1000.0
            self.diagnostics._log_if_enabled(
                session=session,
                state_before=state_before,
                nlu=nlu,
                result=None,
                response_key=response_key,
                response_payload=payload,
                preprocess_ms=t_preprocess * 1000.0,
                nlu_ms=t_nlu * 1000.0,
                flow_ms=t_flow * 1000.0,
                route_ms=0.0,
                handler_ms=0.0,
                total_ms=total_ms,
            )
            self.diagnostics._finalize_trace_and_timing(
                trace=trace,
                session=session,
                response_key=response_key,
                total_start_monotonic=t_total_start,
                total_ms=total_ms,
                preprocess_ms=t_preprocess * 1000.0,
                nlu_ms=t_nlu * 1000.0,
                flow_ms=t_flow * 1000.0,
                route_ms=0.0,
                handler_ms=0.0,
            )
            return self.response_writer._hydrate_output(
                session=session,
                output=TurnOutput(
                    response_key=response_key,
                    response_payload=payload,
                ),
            )

        if flow.action == FlowAction.CANCEL:
            ctx.awaiting_flow_confirmation = True
            ctx.return_state = session.conversation_state
            ctx.interrupt_proposal = None

            session.conversation_state = ConversationState.CANCELLATION_CONFIRMATION
            response_key = flow.response_key or "flow_guard_confirm_cancel"
            response_payload = dict(flow.response_payload or {})

            self.response_writer._apply_session_response(
                session=session,
                intent=intent_result.intent,
                response_key=response_key,
                response_payload=response_payload,
            )

            total_ms = (time.perf_counter() - t_total_start) * 1000.0
            self.diagnostics._log_if_enabled(
                session=session,
                state_before=state_before,
                nlu=nlu,
                result=None,
                response_key=response_key,
                response_payload=response_payload,
                preprocess_ms=t_preprocess * 1000.0,
                nlu_ms=t_nlu * 1000.0,
                flow_ms=t_flow * 1000.0,
                route_ms=0.0,
                handler_ms=0.0,
                total_ms=total_ms,
            )
            self.diagnostics._finalize_trace_and_timing(
                trace=trace,
                session=session,
                response_key=response_key,
                total_start_monotonic=t_total_start,
                total_ms=total_ms,
                preprocess_ms=t_preprocess * 1000.0,
                nlu_ms=t_nlu * 1000.0,
                flow_ms=t_flow * 1000.0,
                route_ms=0.0,
                handler_ms=0.0,
            )
            return self.response_writer._hydrate_output(
                session=session,
                output=TurnOutput(
                    response_key=response_key,
                    response_payload=response_payload,
                ),
            )

        if flow.action == FlowAction.HANDLE_READONLY_INTERRUPT:
            readonly_output = self.flow_gate._handle_readonly_interrupt(
                session=session,
                state_before=state_before,
                intent_result=intent_result,
                nlu=nlu,
                trace=trace,
            )
            if readonly_output is not None:
                total_ms = (time.perf_counter() - t_total_start) * 1000.0
                self.diagnostics._finalize_trace_and_timing(
                    trace=trace,
                    session=session,
                    response_key=readonly_output.response_key,
                    total_start_monotonic=t_total_start,
                    total_ms=total_ms,
                    preprocess_ms=t_preprocess * 1000.0,
                    nlu_ms=t_nlu * 1000.0,
                    flow_ms=t_flow * 1000.0,
                    route_ms=0.0,
                    handler_ms=0.0,
                )
                return readonly_output

        if flow.action == FlowAction.REWRITE and flow.effective_intent is not None:
            intent_result = IntentResult(
                intent=flow.effective_intent,
                raw_text=intent_result.raw_text,
            )

        t0 = time.perf_counter()
        route = self.router.route(session.conversation_state, intent_result)
        t_route = time.perf_counter() - t0

        if ROUTE_DEBUG_ENABLED:
            print(
                "[ROUTE]",
                {
                    "state": session.conversation_state.value,
                    "intent": intent_result.intent.value,
                    "handler": route.handler_name,
                    "allowed": route.allowed,
                },
            )

        if not route.allowed or not route.handler_name:
            payload = {
                "state": session.conversation_state.value,
                "intent": intent_result.intent.value,
            }

            self.response_writer._apply_session_response(
                session=session,
                intent=intent_result.intent,
                response_key="intent_not_allowed",
                response_payload=payload,
            )

            total_ms = (time.perf_counter() - t_total_start) * 1000.0
            self.diagnostics._log_if_enabled(
                session=session,
                state_before=state_before,
                nlu=nlu,
                result=None,
                response_key="intent_not_allowed",
                response_payload=payload,
                preprocess_ms=t_preprocess * 1000.0,
                nlu_ms=t_nlu * 1000.0,
                flow_ms=t_flow * 1000.0,
                route_ms=t_route * 1000.0,
                handler_ms=0.0,
                total_ms=total_ms,
            )
            self.diagnostics._finalize_trace_and_timing(
                trace=trace,
                session=session,
                response_key="intent_not_allowed",
                total_start_monotonic=t_total_start,
                total_ms=total_ms,
                preprocess_ms=t_preprocess * 1000.0,
                nlu_ms=t_nlu * 1000.0,
                flow_ms=t_flow * 1000.0,
                route_ms=t_route * 1000.0,
                handler_ms=0.0,
            )
            return self.response_writer._hydrate_output(
                session=session,
                output=TurnOutput(
                    response_key="intent_not_allowed",
                    response_payload=payload,
                ),
            )

        handler = self.dispatcher.get_handler(route.handler_name)
        if handler is None:
            raise KeyError(f"Handler not registered: {route.handler_name}")

        t0 = time.perf_counter()
        result: HandlerResult = handler.handle(
            intent=intent_result.intent,
            context=ctx,
            user_text=nlu.normalized_text,
            session=session,
        )
        t_handler = time.perf_counter() - t0

        command_result: dict[str, Any] | None = None
        transfer_number: str | None = None

        if result.command:
            command_type = result.command.get("type")
            if command_type == "transfer_call":
                transfer_number = result.command.get("transfer_number")
                command_result = {
                    "ok": True,
                    "transport_only": True,
                    "transfer_number": transfer_number,
                }
            else:
                command_result = self.dispatcher._apply_command(session, result.command)

            print(
                "[COMMAND RESULT]",
                {
                    "command": result.command,
                    "result": command_result,
                },
            )

            if command_result.get("ok", False):
                self.payment_flow._emit_payment_events_from_command(
                    session=session,
                    state_before=state_before,
                    response_key=result.response_key,
                    command=result.command,
                    command_result=command_result,
                )

            if not command_result.get("ok", False):
                command_type = result.command.get("type")
                command_payload = result.command.get("payload") or {}

                if command_type == "SEND_SMS":
                    template = command_payload.get("template")
                    delivery = session.conversation_context.delivery_address
                    attempts_made = int(command_result.get("attempts_made", 1) or 1)

                    if template == "checkout_link":
                        # _apply_command already retried internally.
                        # If it still failed after those attempts, fall back to voice immediately.
                        if attempts_made >= 2:
                            delivery.source = "voice"
                            session.conversation_context.current_prompt_field = "delivery_seed_confirmation"
                            result = HandlerResult(
                                next_state=ConversationState.WAITING_FOR_DELIVERY_ADDRESS_COLLECTION,
                                response_key="checkout_link_failed_fallback_voice",
                                response_payload={
                                    "area": delivery.area,
                                    "postal_code": delivery.postal_code,
                                    "order_number": delivery.order_number,
                                    "error_code": command_result.get("error_code"),
                                    "error_message": command_result.get("error_message"),
                                },
                            )
                        else:
                            result = HandlerResult(
                                next_state=ConversationState.CONFIRMING_ORDER,
                                response_key="checkout_link_send_failed",
                                response_payload={
                                    "order_number": delivery.order_number,
                                    "error_code": command_result.get("error_code"),
                                    "error_message": command_result.get("error_message"),
                                },
                            )

                    elif template == "payment_link":
                        # Payment-link flow also retried internally.
                        # If still failed, apologize and stop progression.
                        if attempts_made >= 2:
                            result = HandlerResult(
                                next_state=session.conversation_state,
                                response_key="payment_link_unavailable_now",
                                response_payload={
                                    "order_number": delivery.order_number,
                                    "error_code": command_result.get("error_code"),
                                    "error_message": command_result.get("error_message"),
                                },
                            )
                        else:
                            result = HandlerResult(
                                next_state=session.conversation_state,
                                response_key="payment_link_send_failed",
                                response_payload={
                                    "order_number": delivery.order_number,
                                    "error_code": command_result.get("error_code"),
                                    "error_message": command_result.get("error_message"),
                                },
                            )

        if result.reset_context:
            ctx.reset()

        session.conversation_state = result.next_state

        # ── Multi-item queue drain ────────────────────────────
        # When an item was just added to cart and we return to IDLE,
        # check if there are queued items from a multi-item utterance.
        # If so, auto-start the next queued item.
        queue_drain_result = self.item_queue_service.try_drain(
            session=session,
            current_result=result,
        )
        if queue_drain_result is not None:
            result = queue_drain_result

        result = self.payment_flow._maybe_resume_confirmation_after_cart_edit(
            session=session,
            result=result,
        )

        result = self.dispatcher._apply_reprompt_guardrail(
            session=session,
            state_before=state_before,
            result=result,
        )

        suppress_payment_replay = self.payment_flow._should_suppress_payment_prompt_replay(
            session=session,
            prior_state=state_before,
            response_key=result.response_key,
        )

        if not suppress_payment_replay:
            self.response_writer._apply_session_response(
                session=session,
                intent=intent_result.intent,
                response_key=result.response_key,
                response_payload=result.response_payload,
            )
            self.payment_flow._emit_payment_events_from_payload(
                session=session,
                state_before=state_before,
                payload=result.response_payload,
            )

        total_ms = (time.perf_counter() - t_total_start) * 1000.0
        self.diagnostics._log_if_enabled(
            session=session,
            state_before=state_before,
            nlu=nlu,
            result=result,
            response_key=result.response_key,
            response_payload=result.response_payload,
            next_state=result.next_state,
            preprocess_ms=t_preprocess * 1000.0,
            nlu_ms=t_nlu * 1000.0,
            flow_ms=t_flow * 1000.0,
            route_ms=t_route * 1000.0,
            handler_ms=t_handler * 1000.0,
            total_ms=total_ms,
        )
        self.diagnostics._finalize_trace_and_timing(
            trace=trace,
            session=session,
            response_key=result.response_key,
            total_start_monotonic=t_total_start,
            total_ms=total_ms,
            preprocess_ms=t_preprocess * 1000.0,
            nlu_ms=t_nlu * 1000.0,
            flow_ms=t_flow * 1000.0,
            route_ms=t_route * 1000.0,
            handler_ms=t_handler * 1000.0,
            command=result.command,
        )

        output = TurnOutput(
            response_key=result.response_key,
            response_payload=result.response_payload,
            internal_response_text=getattr(result, "internal_response_text", None),
            spoken_response_text=getattr(result, "spoken_response_text", None),
            end_call_after_playback=(
                result.next_state == ConversationState.COMPLETED
                or transfer_number is not None
            ),
            transfer_call_to_number=transfer_number,
        )
        if suppress_payment_replay:
            return self.response_writer._build_silent_output(
                response_key=output.response_key,
                response_payload=output.response_payload,
                end_call_after_playback=output.end_call_after_playback,
                transfer_call_to_number=output.transfer_call_to_number,
            )

        return self.response_writer._hydrate_output(
            session=session,
            output=output,
        )


