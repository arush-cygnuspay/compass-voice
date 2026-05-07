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

import time
import uuid
from dataclasses import dataclass
from typing import Any

from app.cart.read_models.cart_summary_builder import CartSummaryBuilder
from app.config.realtime import get_realtime_config
from app.contracts.command_result import CommandResult
from app.core.add_item_value_mapper import build_missing_required_fields, build_normalized_values
from app.core.command_executor import CommandExecutor
from app.core.sms_command_fallback import resolve_sms_failure
from app.state_machine.policy.flow_decision import FlowAction
from app.state_machine.policy.flow_control_policy import FlowControlPolicy
from app.state_machine.policy.flow_gate import FlowGate, FlowGateDecision
from app.core.handler_dispatcher import HandlerDispatcher
from app.core.item_queue_service import ItemQueueService
from app.core.nlu_orchestrator import INTENT_MIN_CONF, NluOrchestrator
from app.core.session_turn_lock_manager import SessionTurnLockManager
from app.core.turn_snapshot import TurnSnapshot
from app.nlu.intent_resolution.intent_result import IntentResult
from app.nlu.nlu_resolver import resolve_nlu
from app.core.payment_flow_orchestrator import PaymentFlowOrchestrator
from app.core.response_builder import ResponseBuilder
from app.core.session_response_writer import SessionResponseWriter
from app.core.turn_diagnostics import TurnDiagnostics
from app.diagnostics.backends.csv_backend import CsvDiagnosticsBackend
from app.diagnostics.turn_event import TurnEvent
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
from app.state_machine.handler_result import HandlerResult
from app.config.voice_transfer import HUMAN_AGENT_TRANSFER_NUMBER
from app.state_machine.handlers.order.waiting_for_order_type_handler import (
    WaitingForOrderTypeHandler,
)
from app.state_machine.models.conversation_state import ConversationState
from app.state_machine.resume_prompt_builder import ResumePromptBuilder
from app.state_machine.policy.intent_coercion import IntentCoercionPolicy
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
    # Desired next conversation state — applied by TurnEngine after the turn.
    # None means "leave session.conversation_state unchanged".
    next_state: "ConversationState | None" = None


# Sourced from config — no direct os.getenv at module level.
ROUTE_DEBUG_ENABLED: bool = get_realtime_config().route_debug_enabled


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

        backends: list[Any] = [CsvDiagnosticsBackend(self.nlu_logger)]
        json_log_path = get_realtime_config().nlu_json_log_path
        if json_log_path:
            from app.diagnostics.backends.json_backend import JsonDiagnosticsBackend
            backends.append(JsonDiagnosticsBackend(json_log_path))

        self.diagnostics = TurnDiagnostics(backends=backends)
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
        )
        self.nlu = NluOrchestrator(
            intent_bundle=intent_bundle,
            slot_bundle=slot_bundle,
            diagnostics=self.diagnostics,
        )
        self.intent_coercion = IntentCoercionPolicy(menu_repo=menu_repo)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_response_text(text: str | None) -> str:
        return " ".join((text or "").split()).strip()

    def _build_response_text(
        self,
        session: Session,
        response_key: str,
        response_payload: dict[str, Any] | None,
    ) -> str:
        try:
            return self._normalize_response_text(
                self.responder.build(
                    response_key=response_key,
                    context=session.conversation_context,
                    payload=response_payload,
                )
            )
        except Exception:
            return ""

    def _record_turn_event(
        self,
        *,
        session: Session,
        state_before: ConversationState,
        nlu: Any,
        result: HandlerResult | None,
        response_key: str,
        response_payload: dict[str, Any] | None,
        diag: dict[str, Any],
        next_state: ConversationState | None = None,
        preprocess_ms: float | None = None,
        nlu_ms: float | None = None,
        flow_ms: float | None = None,
        route_ms: float | None = None,
        handler_ms: float | None = None,
        total_ms: float | None = None,
        # Extended diagnostics — optional, populated by callers that have the info.
        coercion_reason: str | None = None,
        route_reason: str | None = None,
        resolved_entity_type: str | None = None,
        resolved_entity_id: str | None = None,
    ) -> None:
        if not self.diagnostics.enabled:
            return
        try:
            context_snapshot = self.diagnostics._snapshot_context_for_logging(session)
            response_text = self._build_response_text(session, response_key, response_payload)
            normalized_values = build_normalized_values(session, self.menu_repo, result)
            missing_fields = build_missing_required_fields(session)
            slots = tuple(getattr(nlu, "slots", ()) or ())

            event = TurnEvent(
                session_id=self.diagnostics._safe_session_id(session),
                turn_index=session.turn_count,
                state_before=state_before.value,
                state_after=session.conversation_state.value,
                next_state=(next_state or session.conversation_state).value,
                pending_action=context_snapshot["pending_action"],
                current_prompt_field=context_snapshot["current_prompt_field"],
                current_item_id=context_snapshot["current_item_id"],
                current_item_name=context_snapshot["current_item_name"],
                raw_user_text=(
                    getattr(nlu, "raw_text", "")
                    or session.conversation_context.last_user_text
                    or ""
                ),
                user_text=session.conversation_context.last_user_text or "",
                normalized_text=getattr(nlu, "normalized_text", "") or "",
                pred_main_intent=getattr(nlu, "model_main_intent", "") or "",
                pred_sub_intent=getattr(nlu, "model_sub_intent", "") or "",
                pred_intent=getattr(getattr(nlu, "effective_intent", None), "value", "") or "",
                pred_intent_confidence=getattr(nlu, "intent_confidence", None),
                slot_model_ran=bool(getattr(nlu, "slot_model_ran", False)),
                slots=slots,
                response_key=response_key,
                response_text=response_text,
                command=result.command if result else None,
                normalized_values=normalized_values,
                missing_required_fields=tuple(missing_fields),
                reprompt_field=diag["reprompt_field"],
                reprompt_count=diag["reprompt_count"],
                reprompt_escalated=diag["reprompt_escalated"],
                reprompt_escalation_count=session.reprompt_escalation_count,
                fallback_triggered=diag["fallback_triggered"],
                fallback_reason=diag["fallback_reason"],
                fallback_count=session.fallback_count,
                slot_extraction_failed=diag["slot_extraction_failed"],
                slot_extraction_failure_count=session.slot_extraction_failure_count,
                invalid_modifier=diag["invalid_modifier"],
                invalid_modifier_count=session.invalid_modifier_count,
                user_repeated=diag["user_repeated"],
                repeated_user_turn_count=session.repeated_user_turn_count,
                preprocess_ms=preprocess_ms,
                nlu_ms=nlu_ms,
                flow_ms=flow_ms,
                route_ms=route_ms,
                handler_ms=handler_ms,
                total_ms=total_ms,
                # Extended diagnostics
                raw_slots=slots,
                effective_slots=slots,
                active_resolution_scope=state_before.value,
                resolved_entity_type=resolved_entity_type,
                resolved_entity_id=resolved_entity_id,
                route_reason=route_reason,
                coercion_reason=coercion_reason,
            )
            self.diagnostics.record(event)
        except Exception as exc:
            print(f"[TURN_EVENT_BUILD_ERROR] {type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process_turn(
        self,
        session: Session,
        user_text: str,
        trace: Any | None = None,
    ) -> TurnOutput:
        t_total_start = time.perf_counter()
        ctx = session.conversation_context
        session.current_turn_id = uuid.uuid4().hex
        snapshot = TurnSnapshot.capture(session)

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

        # Auto payment-check probe injected by the transport layer after
        # PAYMENT_AUTO_CHECK_DELAY_SECONDS of silence.  Bypass NLU entirely
        # and call the payment verifier directly — O(1) path, no model inference.
        if user_text == "__auto_payment_check__":
            payment_output = self.payment_flow._handle_auto_payment_check(session)
            if payment_output.next_state is not None:
                session.conversation_state = payment_output.next_state
            return payment_output

        order_type_gate = self.flow_gate._compute_order_type_gate_state(session)
        if order_type_gate is not None:
            session.conversation_state = order_type_gate

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

            if gate_result.awaiting_flow_confirmation is not None:
                ctx.awaiting_flow_confirmation = gate_result.awaiting_flow_confirmation
            if gate_result.interrupt_proposal is not None:
                ctx.interrupt_proposal = gate_result.interrupt_proposal
            if gate_result.prompt_field is not None:
                ctx.current_prompt_field = gate_result.prompt_field

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
            _diag = self.diagnostics._update_session_diagnostics(
                session=session,
                state_before=ConversationState.WAITING_FOR_ORDER_TYPE,
                nlu=gate_nlu,
                response_key=gate_result.response_key,
                response_payload=gate_result.response_payload,
            )
            self._record_turn_event(
                session=session,
                state_before=ConversationState.WAITING_FOR_ORDER_TYPE,
                nlu=gate_nlu,
                result=gate_result,
                response_key=gate_result.response_key,
                response_payload=gate_result.response_payload,
                diag=_diag,
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

        phase3_decision = self.flow_gate._handle_phase3_control_shortcuts(
            session=session,
            state_before=state_before,
            intent_result=intent_result,
            nlu=nlu,
        )
        if phase3_decision is not None:
            # Apply the state override before doing anything else so that
            # diagnostics and session response reflect the post-transition state.
            if phase3_decision.state_override is not None:
                session.conversation_state = phase3_decision.state_override

            if phase3_decision.output is not None:
                phase3_shortcut = phase3_decision.output
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
                _diag = self.diagnostics._update_session_diagnostics(
                    session=session,
                    state_before=state_before,
                    nlu=nlu,
                    response_key=phase3_shortcut.response_key,
                    response_payload=phase3_shortcut.response_payload,
                )
                self._record_turn_event(
                    session=session,
                    state_before=state_before,
                    nlu=nlu,
                    result=None,
                    response_key=phase3_shortcut.response_key,
                    response_payload=phase3_shortcut.response_payload,
                    diag=_diag,
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
            # output is None — continue processing with updated state.

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
            _diag = self.diagnostics._update_session_diagnostics(
                session=session,
                state_before=state_before,
                nlu=nlu,
                response_key=shortcut_output.response_key,
                response_payload=shortcut_output.response_payload,
            )
            self._record_turn_event(
                session=session,
                state_before=state_before,
                nlu=nlu,
                result=None,
                response_key=shortcut_output.response_key,
                response_payload=shortcut_output.response_payload,
                diag=_diag,
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

        # FSM-aware intent coercion (idle add-item rules).
        # Runs after all FlowGate rewrites so that coercion decisions see the
        # final effective intent, not the raw model output.
        _coercion = self.intent_coercion.coerce(
            state=session.conversation_state,
            intent_result=intent_result,
            nlu=nlu,
            cart=session.cart,
        )
        intent_result = _coercion.intent_result
        _coercion_reason: str | None = _coercion.coercion_reason

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
            _diag = self.diagnostics._update_session_diagnostics(
                session=session,
                state_before=state_before,
                nlu=nlu,
                response_key=response_key,
                response_payload=payload,
            )
            self._record_turn_event(
                session=session,
                state_before=state_before,
                nlu=nlu,
                result=None,
                response_key=response_key,
                response_payload=payload,
                diag=_diag,
                preprocess_ms=t_preprocess * 1000.0,
                nlu_ms=t_nlu * 1000.0,
                flow_ms=t_flow * 1000.0,
                route_ms=0.0,
                handler_ms=0.0,
                total_ms=total_ms,
                coercion_reason=_coercion_reason,
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
            _diag = self.diagnostics._update_session_diagnostics(
                session=session,
                state_before=state_before,
                nlu=nlu,
                response_key=response_key,
                response_payload=response_payload,
            )
            self._record_turn_event(
                session=session,
                state_before=state_before,
                nlu=nlu,
                result=None,
                response_key=response_key,
                response_payload=response_payload,
                diag=_diag,
                preprocess_ms=t_preprocess * 1000.0,
                nlu_ms=t_nlu * 1000.0,
                flow_ms=t_flow * 1000.0,
                route_ms=0.0,
                handler_ms=0.0,
                total_ms=total_ms,
                coercion_reason=_coercion_reason,
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
                _diag = self.diagnostics._update_session_diagnostics(
                    session=session,
                    state_before=state_before,
                    nlu=nlu,
                    response_key=readonly_output.response_key,
                    response_payload=readonly_output.response_payload,
                )
                self._record_turn_event(
                    session=session,
                    state_before=state_before,
                    nlu=nlu,
                    result=None,
                    response_key=readonly_output.response_key,
                    response_payload=readonly_output.response_payload,
                    diag=_diag,
                    preprocess_ms=t_preprocess * 1000.0,
                    nlu_ms=t_nlu * 1000.0,
                    flow_ms=t_flow * 1000.0,
                    route_ms=0.0,
                    handler_ms=0.0,
                    total_ms=total_ms,
                    coercion_reason=_coercion_reason,
                )
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
                if readonly_output.next_state is not None:
                    session.conversation_state = readonly_output.next_state
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
            from app.policies.no_input_escalation_policy import (
                NoInputEscalationPolicy,
                NoInputTier,
            )

            miss_count = ctx.bump_unknown()
            tier = NoInputEscalationPolicy.next_tier(miss_count)

            # Terminal tier - hand off to a human via the existing transfer path.
            if tier == NoInputTier.HANDOFF:
                ctx.reset_unknown()
                session.conversation_state = ConversationState.TRANSFERRING_TO_HUMAN_AGENT
                handoff_payload = {
                    "transfer_number": HUMAN_AGENT_TRANSFER_NUMBER,
                    "reason": "no_input_escalation",
                    "miss_count": miss_count,
                }
                self.response_writer._apply_session_response(
                    session=session,
                    intent=intent_result.intent,
                    response_key="transferring_to_human_agent",
                    response_payload=handoff_payload,
                )
                total_ms = (time.perf_counter() - t_total_start) * 1000.0
                _diag = self.diagnostics._update_session_diagnostics(
                    session=session,
                    state_before=state_before,
                    nlu=nlu,
                    response_key="transferring_to_human_agent",
                    response_payload=handoff_payload,
                )
                self._record_turn_event(
                    session=session,
                    state_before=state_before,
                    nlu=nlu,
                    result=None,
                    response_key="transferring_to_human_agent",
                    response_payload=handoff_payload,
                    diag=_diag,
                    preprocess_ms=t_preprocess * 1000.0,
                    nlu_ms=t_nlu * 1000.0,
                    flow_ms=t_flow * 1000.0,
                    route_ms=t_route * 1000.0,
                    handler_ms=0.0,
                    total_ms=total_ms,
                    coercion_reason=_coercion_reason,
                    route_reason=route.reason,
                )
                self.diagnostics._finalize_trace_and_timing(
                    trace=trace,
                    session=session,
                    response_key="transferring_to_human_agent",
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
                        response_key="transferring_to_human_agent",
                        response_payload=handoff_payload,
                        transfer_call_to_number=HUMAN_AGENT_TRANSFER_NUMBER,
                        end_call_after_playback=True,
                    ),
                )

            payload = {
                "state": session.conversation_state.value,
                "intent": intent_result.intent.value,
                "tier": tier.value,
                "miss_count": miss_count,
            }

            self.response_writer._apply_session_response(
                session=session,
                intent=intent_result.intent,
                response_key="intent_not_allowed",
                response_payload=payload,
            )

            total_ms = (time.perf_counter() - t_total_start) * 1000.0
            _diag = self.diagnostics._update_session_diagnostics(
                session=session,
                state_before=state_before,
                nlu=nlu,
                response_key="intent_not_allowed",
                response_payload=payload,
            )
            self._record_turn_event(
                session=session,
                state_before=state_before,
                nlu=nlu,
                result=None,
                response_key="intent_not_allowed",
                response_payload=payload,
                diag=_diag,
                preprocess_ms=t_preprocess * 1000.0,
                nlu_ms=t_nlu * 1000.0,
                flow_ms=t_flow * 1000.0,
                route_ms=t_route * 1000.0,
                handler_ms=0.0,
                total_ms=total_ms,
                coercion_reason=_coercion_reason,
                route_reason=route.reason,
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

        # NB: turn-level UNKNOWN counter is reset centrally in
        # SessionResponseWriter._apply_session_response on any non-fallback
        # emission, so we don't need to reset here.

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
                command_result = CommandResult(
                    ok=True,
                    transport_only=True,
                    transfer_number=transfer_number,
                )
            else:
                command_result = self.dispatcher._apply_command(session, result.command)

            print(
                "[COMMAND RESULT]",
                {
                    "command": result.command,
                    "result": command_result,
                },
            )

            if command_result.ok:
                self.payment_flow._emit_payment_events_from_command(
                    session=session,
                    state_before=state_before,
                    response_key=result.response_key,
                    command=result.command,
                    command_result=command_result,
                )

            if not command_result.ok:
                fallback = resolve_sms_failure(
                    session=session,
                    result=result,
                    command=result.command,
                    command_result=command_result,
                )
                if fallback is not None:
                    result = fallback

        if result.reset_context:
            ctx.reset_item_scope()

        # Apply engine-owned context mutations from HandlerResult.
        # Handlers express intent through these fields; TurnEngine is the sole writer.
        if result.awaiting_flow_confirmation is not None:
            ctx.awaiting_flow_confirmation = result.awaiting_flow_confirmation
        if result.interrupt_proposal is not None:
            ctx.interrupt_proposal = result.interrupt_proposal
        if result.prompt_field is not None:
            ctx.current_prompt_field = result.prompt_field

        # ── Multi-item queue drain ────────────────────────────
        # Drain guard uses current_result.next_state (not session.conversation_state)
        # so that state is not applied until after the drain chain resolves.
        queue_drain_result = self.item_queue_service.try_drain(
            session=session,
            current_result=result,
        )
        if queue_drain_result is not None:
            result = queue_drain_result

        # Apply state after drain so queue drain can still inspect next_state.
        session.conversation_state = result.next_state

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
        _diag = self.diagnostics._update_session_diagnostics(
            session=session,
            state_before=state_before,
            nlu=nlu,
            response_key=result.response_key,
            response_payload=result.response_payload,
        )
        _resolved_entity_type: str | None = None
        _resolved_entity_id: str | None = None
        if result.command and result.command.get("type") == "ADD_ITEM_TO_CART":
            _resolved_entity_type = "item"
            _resolved_entity_id = result.command.get("payload", {}).get("item_id")
        self._record_turn_event(
            session=session,
            state_before=state_before,
            nlu=nlu,
            result=result,
            response_key=result.response_key,
            response_payload=result.response_payload,
            diag=_diag,
            next_state=result.next_state,
            preprocess_ms=t_preprocess * 1000.0,
            nlu_ms=t_nlu * 1000.0,
            flow_ms=t_flow * 1000.0,
            route_ms=t_route * 1000.0,
            handler_ms=t_handler * 1000.0,
            total_ms=total_ms,
            coercion_reason=_coercion_reason,
            route_reason=route.reason if route else None,
            resolved_entity_type=_resolved_entity_type,
            resolved_entity_id=_resolved_entity_id,
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
