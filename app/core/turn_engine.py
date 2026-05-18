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
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
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
from app.nlu.intent_resolution.intent_mapping import SUB_INTENT_TO_INTENT
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
from app.state_machine.policy.contextual_control_resolver import (
    ContextualControlKind,
    resolve_contextual_control,
)
from app.state_machine.policy.idle_checkout_coercion import coerce_idle_to_checkout
from app.state_machine.policy.intent_coercion import IntentCoercionPolicy
from app.state_machine.state_router import StateRouter
from app.nlu.semantic_repair.gpt_execution_policy import GptExecutionMode, GptExecutionPolicy
from app.nlu.semantic_repair.repair_service import (
    GptRepairService,
    LocalTurnAnalysis,
    _ALL_SHADOW_SKIP_STATE_VALUES,
)
from app.nlu.semantic_repair.gpt_repair_result import GptRepairResult, GPT_NOT_CALLED
from app.nlu.semantic_repair.add_item_service import AddItemExtractorService
from app.nlu.semantic_repair.add_item_extractor import GptAddItemPlan
from app.nlu.semantic_repair.gpt_log_record_builder import (
    build_gpt_repair_csv_row,
    build_gpt_shadow_jsonl_record,
)
from app.logging.gpt_repair_csv_logger import GptRepairCsvLogger
from app.logging.gpt_repair_jsonl_logger import GptRepairJsonlLogger
from app.config.logging import get_logging_config
from app.config.semantic_repair import get_semantic_repair_config as _get_gpt_cfg


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
        self.gpt_repair = GptRepairService()
        self.add_item_extractor = AddItemExtractorService()

        _log_cfg = get_logging_config()
        self.gpt_csv_logger = GptRepairCsvLogger(
            log_path=_log_cfg.gpt_csv_log_path,
            rotate_on_start=_log_cfg.rotate_gpt_logs_on_start,
        )
        self.gpt_jsonl_logger = GptRepairJsonlLogger(
            log_path=_log_cfg.gpt_jsonl_log_path,
            rotate_on_start=_log_cfg.rotate_gpt_logs_on_start,
        )
        # Background executor for all_shadow GPT calls (fire-and-forget)
        self._shadow_executor: ThreadPoolExecutor | None = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_response_text(text: str | None) -> str:
        return " ".join((text or "").split()).strip()

    @staticmethod
    def _map_gpt_intent_name(intent_name: str | None) -> Intent | None:
        if not intent_name:
            return None
        mapped = SUB_INTENT_TO_INTENT.get(intent_name)
        if mapped is not None:
            return mapped
        control_map = {
            "confirm": Intent.CONFIRM,
            "deny": Intent.DENY,
            "cancel": Intent.CANCEL,
            "unknown": Intent.UNKNOWN,
            "show_menu": Intent.SHOW_MENU,
        }
        if intent_name in control_map:
            return control_map[intent_name]
        try:
            return Intent(intent_name)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # all_shadow background GPT helpers
    # ------------------------------------------------------------------

    def _get_shadow_executor(self) -> ThreadPoolExecutor:
        """Return (lazily creating) the background executor for shadow GPT calls."""
        if self._shadow_executor is None:
            self._shadow_executor = ThreadPoolExecutor(
                max_workers=4,
                thread_name_prefix="gpt-shadow",
            )
        return self._shadow_executor

    def _dispatch_shadow_gpt(
        self,
        *,
        nlu: Any,
        analysis: LocalTurnAnalysis,
        state_before: "ConversationState",
        prompt_field: str,
        item_name: str,
        session_id: str,
        turn_index: int,
        response_key: str,
    ) -> None:
        """Submit an all_shadow GPT task to the background thread pool.

        Returns immediately — never blocks the caller.  The background task
        calls GPT, writes JSONL, and discards the result (gpt_applied=False).
        """
        try:
            self._get_shadow_executor().submit(
                self._run_shadow_gpt,
                nlu=nlu,
                analysis=analysis,
                state_before=state_before,
                prompt_field=prompt_field,
                item_name=item_name,
                session_id=session_id,
                turn_index=turn_index,
                response_key=response_key,
            )
        except Exception as exc:
            print(f"[GPT_SHADOW_DISPATCH_ERROR] {type(exc).__name__}: {exc}")

    def _run_shadow_gpt(
        self,
        *,
        nlu: Any,
        analysis: LocalTurnAnalysis,
        state_before: "ConversationState",
        prompt_field: str,
        item_name: str,
        session_id: str,
        turn_index: int,
        response_key: str,
    ) -> None:
        """Background: call GPT, write JSONL. Never mutates live state."""
        try:
            _cfg = _get_gpt_cfg()
            timeout_sec = _cfg.shadow_timeout_ms / 1000.0

            result = self.gpt_repair.call_gpt_for_shadow(
                nlu=nlu,
                analysis=analysis,
                state=state_before,
                prompt_field=prompt_field,
                item_name=item_name,
                timeout_seconds=timeout_sec,
            )

            record = build_gpt_shadow_jsonl_record(
                analysis=analysis,
                result=result,
                nlu=nlu,
                state_before=state_before.value,
                session_id=session_id,
                turn_index=turn_index,
                response_key=response_key,
                final_response_key=response_key,
                call_mode="all_shadow",
                phase=_cfg.phase,
            )
            self.gpt_jsonl_logger.log_turn(record)
        except Exception as exc:
            print(f"[GPT_SHADOW_RUN_ERROR] {type(exc).__name__}: {exc}")

    def _record_turn_memory(
        self,
        session: Session,
        normalized_text: str,
        response_key: str,
        response_payload: dict[str, Any] | None,
    ) -> None:
        """Append current user utterance + bot response to the session turn memory buffer."""
        ctx = getattr(session, "conversation_context", None)
        if ctx is None:
            return
        append = getattr(ctx, "append_turn_memory", None)
        if not callable(append):
            return
        try:
            ctx.append_turn_memory("user", normalized_text)
            bot_text = self._build_response_text(session, response_key, response_payload)
            if bot_text:
                ctx.append_turn_memory("bot", bot_text)
        except Exception:
            pass

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

    @staticmethod
    def _serialize_add_item_items(items: Any) -> str | None:
        """Serialize GptAddItem tuple to a capped JSON string (max 4000 chars)."""
        if not items:
            return None
        import json as _j
        try:
            out = []
            for it in items:
                out.append({
                    "item": getattr(it, "item", ""),
                    "quantity": getattr(it, "quantity", None),
                    "size": getattr(it, "size", None),
                    "variant": getattr(it, "variant", None),
                    "sides": [
                        {
                            "name": getattr(s, "name", ""),
                            "operation": getattr(s, "operation", "add"),
                            "quantity": getattr(s, "quantity", None),
                            "size": getattr(s, "size", None),
                            "variant": getattr(s, "variant", None),
                            "modifiers": list(getattr(s, "modifiers", ())),
                        }
                        for s in (getattr(it, "sides", ()) or ())
                    ],
                    "modifiers": [
                        {
                            "name": getattr(m, "name", ""),
                            "operation": getattr(m, "operation", "add"),
                            "quantity": getattr(m, "quantity", None),
                            "size": getattr(m, "size", None),
                            "variant": getattr(m, "variant", None),
                        }
                        for m in (getattr(it, "modifiers", ()) or ())
                    ],
                    "missing": list(getattr(it, "missing", ())),
                })
            raw = _j.dumps(out, ensure_ascii=False)
            return raw[:4000] if len(raw) > 4000 else raw
        except Exception:
            return None

    @staticmethod
    def _serialize_add_item_global_slots(slots: Any) -> str | None:
        """Serialize global_slots tuple to a JSON string."""
        if not slots:
            return None
        import json as _j
        try:
            out = []
            for s in slots:
                if isinstance(s, dict):
                    out.append(s)
                else:
                    out.append({
                        "name": getattr(s, "name", str(s)),
                        "value": str(getattr(s, "value", s)),
                    })
            return _j.dumps(out, ensure_ascii=False)
        except Exception:
            return None

    @staticmethod
    def _serialize_validated_items(validated_plan: Any) -> str | None:
        """Serialize ValidatedAddItemPlan.items to a capped JSON string (max 4000 chars).

        Shadow-only — never applied to cart or response.
        """
        if validated_plan is None:
            return None
        items = getattr(validated_plan, "items", None)
        if not items:
            return None
        import json as _j
        try:
            out = []
            for vi in items:
                out.append({
                    "item_id": getattr(vi, "item_id", ""),
                    "item_name": getattr(vi, "item_name", ""),
                    "quantity": getattr(vi, "quantity", 1),
                    "variant_id": getattr(vi, "variant_id", None),
                    "variant_label": getattr(vi, "variant_label", None),
                    "sides": [
                        {
                            "group_id": getattr(s, "group_id", ""),
                            "side_item_id": getattr(s, "side_item_id", ""),
                            "name": getattr(s, "name", ""),
                            "quantity": getattr(s, "quantity", 1),
                            "variant_id": getattr(s, "variant_id", None),
                            "variant_label": getattr(s, "variant_label", None),
                        }
                        for s in (getattr(vi, "sides", ()) or ())
                    ],
                    "modifiers": [
                        {
                            "group_id": getattr(m, "group_id", ""),
                            "modifier_id": getattr(m, "modifier_id", ""),
                            "name": getattr(m, "name", ""),
                            "operation": getattr(m, "operation", "add"),
                            "quantity": getattr(m, "quantity", 1),
                        }
                        for m in (getattr(vi, "modifiers", ()) or ())
                    ],
                    "missing_required_groups": list(getattr(vi, "missing_required_groups", ())),
                    "warnings": [
                        {
                            "code": getattr(w, "code", ""),
                            "entity_kind": getattr(w, "entity_kind", ""),
                            "entity_name": getattr(w, "entity_name", ""),
                            "detail": getattr(w, "detail", ""),
                        }
                        for w in (getattr(vi, "warnings", ()) or ())
                    ],
                })
            raw = _j.dumps(out, ensure_ascii=False)
            return raw[:4000] if len(raw) > 4000 else raw
        except Exception:
            return None

    @staticmethod
    def _validated_items_count(validated_plan: Any) -> int | None:
        if validated_plan is None:
            return None
        items = getattr(validated_plan, "items", None)
        return len(items) if items else None

    @staticmethod
    def _serialize_rejected_items(validated_plan: Any) -> str | None:
        if validated_plan is None:
            return None
        rejected = getattr(validated_plan, "rejected_items", None)
        if not rejected:
            return None
        import json as _j
        try:
            return _j.dumps(list(rejected), ensure_ascii=False)
        except Exception:
            return None

    @staticmethod
    def _serialize_validation_warnings(validated_plan: Any) -> str | None:
        """Serialize plan-level + all item-level warnings to a capped JSON string (max 4000 chars)."""
        if validated_plan is None:
            return None
        import json as _j
        try:
            all_warnings = []
            plan_warnings = getattr(validated_plan, "warnings", ()) or ()
            for w in plan_warnings:
                all_warnings.append({
                    "code": getattr(w, "code", ""),
                    "entity_kind": getattr(w, "entity_kind", ""),
                    "entity_name": getattr(w, "entity_name", ""),
                    "detail": getattr(w, "detail", ""),
                })
            items = getattr(validated_plan, "items", ()) or ()
            for vi in items:
                for w in (getattr(vi, "warnings", ()) or ()):
                    all_warnings.append({
                        "code": getattr(w, "code", ""),
                        "entity_kind": getattr(w, "entity_kind", ""),
                        "entity_name": getattr(w, "entity_name", ""),
                        "detail": getattr(w, "detail", ""),
                        "item_name": getattr(vi, "item_name", ""),
                    })
            if not all_warnings:
                return None
            raw = _j.dumps(all_warnings, ensure_ascii=False)
            return raw[:4000] if len(raw) > 4000 else raw
        except Exception:
            return None

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
        # GPT shadow-mode repair fields (phase 2).
        gpt_shadow: tuple[LocalTurnAnalysis, GptRepairResult] | None = None,
        # True when GPT fallback was actually applied (response overridden).
        gpt_fallback_applied: bool = False,
        # ADD_ITEM extractor result (shadow-only, never applied).
        add_item_plan: GptAddItemPlan | None = None,
    ) -> None:
        if not self.diagnostics.enabled:
            return
        try:
            context_snapshot = self.diagnostics._snapshot_context_for_logging(session)
            response_text = self._build_response_text(session, response_key, response_payload)
            normalized_values = build_normalized_values(session, self.menu_repo, result)
            missing_fields = build_missing_required_fields(session)
            slots = tuple(getattr(nlu, "slots", ()) or ())

            # GPT shadow-mode fields
            import json as _json
            _analysis: LocalTurnAnalysis | None = gpt_shadow[0] if gpt_shadow else None
            _gpt: GptRepairResult | None = gpt_shadow[1] if gpt_shadow else None
            _decision = getattr(_analysis, "execution_decision", None) if _analysis else None
            if _decision is None:
                _ctx = getattr(session, "conversation_context", None)
                _current_field = getattr(_ctx, "current_prompt_field", "") or "" if _ctx is not None else ""
                _history = ()
                if _ctx is not None:
                    _history_getter = getattr(_ctx, "get_turn_memory", None)
                    if callable(_history_getter):
                        _history = _history_getter(3)
                _session_reprompts = getattr(session, "reprompt_count_by_field", {}) or {}
                _ctx_reprompt_count = 0
                if _ctx is not None:
                    _reprompt_count_fn = getattr(_ctx, "reprompt_count", None)
                    if callable(_reprompt_count_fn):
                        _ctx_reprompt_count = int(_reprompt_count_fn(_current_field))
                _effective_intent = getattr(nlu, "effective_intent", Intent.UNKNOWN)
                if not isinstance(_effective_intent, Intent):
                    try:
                        _effective_intent = Intent(getattr(_effective_intent, "value", Intent.UNKNOWN.value))
                    except Exception:
                        _effective_intent = Intent.UNKNOWN
                _decision = GptExecutionPolicy().decide(
                    state=state_before,
                    normalized_user_text=getattr(nlu, "normalized_text", "") or "",
                    raw_stt_final_text=getattr(nlu, "raw_text", "") or "",
                    local_intent_top_n=tuple(getattr(nlu, "intent_candidates", ()) or ()),
                    selected_local_intent=_effective_intent,
                    local_intent_confidence=getattr(nlu, "intent_confidence", 0.0) or 0.0,
                    local_slots=tuple(getattr(nlu, "slots", ()) or ()),
                    active_pending_item_context=getattr(_ctx, "pending_add_item", None),
                    available_options_context=tuple(getattr(_ctx, "available_choices_values", ()) or ()) if _ctx is not None else (),
                    fallback_count=getattr(session, "fallback_count", 0),
                    repeated_prompt_count=max(int(_session_reprompts.get(_current_field, 0) or 0), _ctx_reprompt_count),
                    previous_turns_summary=_history,
                    last_response_key=getattr(session, "last_response_key", None),
                    duplicate_transcript=(
                        bool(getattr(session, "last_normalized_user_text", None))
                        and (getattr(session, "last_normalized_user_text", "") or "") == (getattr(nlu, "normalized_text", "") or "")
                    ),
                )
            # _gpt_called: True only when the GPT API was actually invoked.
            # GPT_NOT_CALLED sentinel (model=None) is returned when skipped.
            _gpt_called = _gpt is not None and _gpt is not GPT_NOT_CALLED
            _gpt_applied = bool((getattr(_gpt, "applied", False) if _gpt else False) or gpt_fallback_applied)
            _local_intent = getattr(getattr(nlu, "effective_intent", None), "value", None)

            # Serialize top-K candidates to JSON string for logging
            _candidates_json: str | None = None
            if _analysis and _analysis.intent_candidates:
                try:
                    _candidates_json = _json.dumps([
                        {
                            "intent": c.canonical_intent,
                            "confidence": round(c.confidence, 4),
                        }
                        for c in _analysis.intent_candidates
                    ])
                except Exception:
                    pass

            # Serialize slots to JSON string
            _slots_json: str | None = None
            if slots:
                try:
                    _slots_json = _json.dumps([
                        {"name": s.name, "value": str(s.value)}
                        for s in slots
                        if hasattr(s, "name")
                    ])
                except Exception:
                    pass


            from app.config.semantic_repair import get_semantic_repair_config as _get_src_cfg
            _src_phase: int | None = None
            try:
                _src_phase = _get_src_cfg().phase
            except Exception:
                pass

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
                # GPT local model snapshot
                local_intent_before_gpt=_local_intent if _analysis else None,
                local_sub_intent_before_gpt=(
                    getattr(nlu, "model_sub_intent", None) if _analysis else None
                ),
                local_intent_confidence_before_gpt=(
                    _analysis.intent_confidence if _analysis else None
                ),
                local_intent_candidates_json=_candidates_json,
                local_slots_before_gpt=_slots_json,
                # GPT eligibility block
                gpt_repair_eligible=_analysis.gpt_repair_eligible if _analysis else False,
                gpt_repair_eligible_reason=_analysis.reason if _analysis else None,
                gpt_repair_reason=_analysis.reason if _analysis else None,
                gpt_candidate_count=_analysis.candidate_count if _analysis else None,
                gpt_skipped_reason=_analysis.skipped_reason if _analysis else None,
                gpt_phase=_src_phase if _src_phase is not None else 0,
                # GPT call block
                gpt_called=bool(_gpt_called),
                gpt_payload_build_ms=_gpt.payload_build_ms if _gpt_called and _gpt else None,
                gpt_request_ms=_gpt.request_ms if _gpt_called and _gpt else None,
                gpt_parse_ms=_gpt.parse_ms if _gpt_called and _gpt else None,
                gpt_total_ms=_gpt.total_ms if _gpt_called and _gpt else None,
                gpt_prompt_chars=_gpt.prompt_chars if _gpt_called and _gpt else None,
                gpt_completion_chars=_gpt.completion_chars if _gpt_called and _gpt else None,
                gpt_model=_gpt.model if _gpt else None,
                # GPT suggestion block
                gpt_decision=_gpt.decision if _gpt else None,
                gpt_selected_intent=_gpt.repaired_intent if _gpt else None,
                gpt_selected_control_intent=_gpt.repaired_control_intent if _gpt else None,
                gpt_slot_corrections_json=(
                    _json.dumps(_gpt.slot_corrections) if _gpt and _gpt.slot_corrections else None
                ),
                gpt_confidence=_gpt.confidence if _gpt else None,
                gpt_reason=_gpt.reason if _gpt else None,
                gpt_latency_ms=_gpt.latency_ms if _gpt else None,
                gpt_timeout=_gpt.timeout if _gpt else False,
                gpt_parse_error=_gpt.parse_error if _gpt else None,
                # Final block
                gpt_applied=_gpt_applied,
                gpt_apply_reason=(
                    "fallback_applied" if gpt_fallback_applied
                    else ("intent_repair_applied" if _gpt_applied else ("shadow_mode" if _gpt_called else None))
                ),
                final_intent_after_gpt=(
                    _gpt.repaired_intent if _gpt_applied and _gpt and _gpt.repaired_intent
                    else (_local_intent if _analysis else None)
                ),
                final_slots_after_gpt=_slots_json,
                final_response_key=response_key,
                training_candidate=False,
                # Fallback classification (phase 2: logged only, never applied)
                gpt_fallback_type=_gpt.fallback_type if _gpt else "none",
                fallback_response_key=(
                    f"fallback_{_gpt.fallback_type}"
                    if _gpt and _gpt.fallback_type != "none"
                    else None
                ),
                # ADD_ITEM extractor (shadow-only — never applied to cart/state/response)
                add_item_extractor_called=(
                    add_item_plan.total_ms is not None if add_item_plan else False
                ),
                add_item_eligible=(
                    add_item_plan.eligible if add_item_plan else False
                ),
                add_item_skipped_reason=(
                    add_item_plan.skipped_reason if add_item_plan else None
                ),
                add_item_decision=(
                    add_item_plan.decision if add_item_plan else None
                ),
                add_item_confidence=(
                    add_item_plan.confidence if add_item_plan else None
                ),
                add_item_items_json=self._serialize_add_item_items(
                    add_item_plan.items if add_item_plan else ()
                ),
                add_item_items_count=(
                    len(add_item_plan.items) if add_item_plan and add_item_plan.items else None
                ),
                add_item_global_slots_json=self._serialize_add_item_global_slots(
                    add_item_plan.global_slots if add_item_plan else ()
                ),
                add_item_latency_ms=(
                    add_item_plan.latency_ms if add_item_plan else None
                ),
                add_item_total_ms=(
                    add_item_plan.total_ms if add_item_plan else None
                ),
                add_item_prompt_chars=(
                    add_item_plan.prompt_chars if add_item_plan else None
                ),
                add_item_completion_chars=(
                    add_item_plan.completion_chars if add_item_plan else None
                ),
                add_item_timeout=(
                    add_item_plan.timeout if add_item_plan else False
                ),
                add_item_parse_error=(
                    add_item_plan.parse_error if add_item_plan else None
                ),
                add_item_parse_notes_json=(
                    _json.dumps(list(add_item_plan.parse_notes))
                    if add_item_plan and add_item_plan.parse_notes
                    else None
                ),
                add_item_reason=(
                    add_item_plan.reason if add_item_plan else None
                ),
                add_item_model=(
                    add_item_plan.model if add_item_plan else None
                ),
                # Phase 2: local menu validator results (shadow-only, never applied)
                add_item_validated_items_json=self._serialize_validated_items(
                    add_item_plan.validated_plan if add_item_plan else None
                ),
                add_item_validated_items_count=self._validated_items_count(
                    add_item_plan.validated_plan if add_item_plan else None
                ),
                add_item_rejected_items_json=self._serialize_rejected_items(
                    add_item_plan.validated_plan if add_item_plan else None
                ),
                add_item_validation_warnings_json=self._serialize_validation_warnings(
                    add_item_plan.validated_plan if add_item_plan else None
                ),
                add_item_validator_ms=(
                    add_item_plan.validator_ms if add_item_plan else None
                ),
                add_item_has_blocking_warnings=(
                    add_item_plan.has_blocking_warnings if add_item_plan else False
                ),
            )
            self.diagnostics.record(event)

            # Stamp realtime trace with GPT summary (eligible_only inline path)
            # For all_shadow the trace is stamped earlier with pending_async values.
            _trace_ref = None
            # trace is not directly accessible here; it's stamped in process_turn.

            # Write GPT repair CSV + JSONL for eligible/called/training_candidate/add_item turns.
            if (event.gpt_repair_eligible or event.gpt_called or event.training_candidate
                    or event.add_item_extractor_called):
                try:
                    self.gpt_csv_logger.log_turn(build_gpt_repair_csv_row(event))
                except Exception as _csv_exc:
                    print(f"[GPT_CSV_LOG_ERROR] {type(_csv_exc).__name__}: {_csv_exc}")
                # JSONL is source of truth for nested GPT training data
                if _analysis is not None and _gpt is not None:
                    try:
                        _gpt_cfg = _get_gpt_cfg()
                        _jsonl_record = build_gpt_shadow_jsonl_record(
                            analysis=_analysis,
                            result=_gpt,
                            nlu=nlu,
                            state_before=state_before.value,
                            session_id=event.session_id,
                            turn_index=event.turn_index,
                            response_key=event.response_key,
                            final_response_key=event.final_response_key,
                            call_mode=_gpt_cfg.call_mode or "",
                            phase=_gpt_cfg.phase,
                        )
                        self.gpt_jsonl_logger.log_turn(_jsonl_record)
                    except Exception as _jsonl_exc:
                        print(f"[GPT_JSONL_LOG_ERROR] {type(_jsonl_exc).__name__}: {_jsonl_exc}")
                # Write JSONL for add_item extractor turns (shadow-only logging)
                if event.add_item_extractor_called:
                    try:
                        from app.nlu.semantic_repair.gpt_log_record_builder import (
                            build_gpt_repair_log_record,
                        )
                        self.gpt_jsonl_logger.log_turn(build_gpt_repair_log_record(event))
                    except Exception as _ai_jsonl_exc:
                        print(f"[ADD_ITEM_JSONL_LOG_ERROR] {type(_ai_jsonl_exc).__name__}: {_ai_jsonl_exc}")
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

        # Initialised to None; populated after all coercions complete below.
        _gpt_shadow: tuple[LocalTurnAnalysis, GptRepairResult] | None = None

        # Populated by ADD_ITEM extractor (shadow-only); never applied.
        _add_item_plan: GptAddItemPlan | None = None

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
                    gpt_shadow=_gpt_shadow,

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
                gpt_shadow=_gpt_shadow,

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

        # Idle-checkout coercion: UNKNOWN + checkout/done/payment phrase in
        # IDLE with a non-empty cart → Intent.CHECKOUT.  This fires BEFORE the
        # add-item coercion so that checkout-family phrases (e.g. "payment")
        # are not accidentally coerced to ADD_ITEM.  The resulting CHECKOUT
        # intent routes to StartOrderHandler → confirm_order_summary first;
        # payment never starts without explicit order confirmation.
        _idle_checkout = coerce_idle_to_checkout(
            state=session.conversation_state,
            intent_result=intent_result,
            nlu=nlu,
            cart=session.cart,
        )
        intent_result = _idle_checkout.intent_result
        _idle_checkout_reason: str | None = _idle_checkout.coercion_reason

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
        _coercion_reason: str | None = _coercion.coercion_reason or _idle_checkout_reason

        # Contextual control resolver: state-aware intent override that uses
        # last_prompt_type to correctly interpret turns like "no that's it for
        # now" (IDLE → CHECKOUT) and "you got your card" (CONFIRMING_ORDER →
        # PAYMENT_STATUS).  Runs after all upstream coercions so it sees the
        # final effective intent.
        if get_realtime_config().contextual_control_v2_enabled:
            _cc = resolve_contextual_control(
                state=session.conversation_state,
                last_prompt_type=session.last_prompt_type,
                cart_has_items=not session.cart.is_empty(),
                normalized_text=nlu.normalized_text or "",
                intent=intent_result.intent,
            )
            if _cc.kind == ContextualControlKind.FINISH_ADDING:
                intent_result = IntentResult(
                    intent=Intent.CHECKOUT,
                    raw_text=intent_result.raw_text,
                )
                _coercion_reason = f"contextual_control_v2:{_cc.reason}"
            elif _cc.kind == ContextualControlKind.PAYMENT_STATUS_QUERY:
                intent_result = IntentResult(
                    intent=Intent.PAYMENT_STATUS,
                    raw_text=intent_result.raw_text,
                )
                _coercion_reason = f"contextual_control_v2:{_cc.reason}"

        # GPT shadow-mode repair — runs AFTER all coercions (idle-checkout,
        # add-item, contextual control v2), BEFORE StateRouter.route().
        _engine_elapsed_ms = (time.perf_counter() - t_total_start) * 1000.0
        _gpt_cfg_now = _get_gpt_cfg()
        _actual_shadow_mode = False
        try:
            _gpt_shadow = self.gpt_repair.run(
                nlu=nlu,
                intent_result=intent_result,
                state=session.conversation_state,
                session=session,
                engine_elapsed_ms=_engine_elapsed_ms,
            )
        except Exception as _gpt_err:
            print(f"[GPT_SHADOW_ERROR] {type(_gpt_err).__name__}: {_gpt_err}")

        _gpt_analysis = _gpt_shadow[0] if _gpt_shadow is not None else None
        _actual_shadow_mode = (_gpt_cfg_now.effective_call_mode == "all_shadow")

        # Shadow mode: dispatch background GPT now (non-blocking).
        if _actual_shadow_mode and _gpt_shadow is not None:
            _shadow_analysis = _gpt_shadow[0]
            _shadow_state = session.conversation_state
            _shadow_text = (getattr(nlu, "normalized_text", "") or "").strip()
            if (
                len(_shadow_text) >= 2
                and _shadow_state.value not in _ALL_SHADOW_SKIP_STATE_VALUES
                and os.getenv("OPENAI_API_KEY")
            ):
                _shadow_ctx = getattr(session, "conversation_context", None)
                _shadow_pf = getattr(_shadow_ctx, "current_prompt_field", "") or "" if _shadow_ctx else ""
                _shadow_in = getattr(_shadow_ctx, "current_item_name", "") or "" if _shadow_ctx else ""
                self._dispatch_shadow_gpt(
                    nlu=nlu,
                    analysis=_shadow_analysis,
                    state_before=_shadow_state,
                    prompt_field=_shadow_pf,
                    item_name=_shadow_in,
                    session_id=self.diagnostics._safe_session_id(session),
                    turn_index=session.turn_count,
                    response_key=session.last_response_key or "",
                )
                if trace is not None:
                    for _attr, _val in (
                        ("gpt_called", True),
                        ("gpt_decision", "pending_async"),
                        ("gpt_applied", False),
                    ):
                        if hasattr(trace, _attr):
                            setattr(trace, _attr, _val)

        # Inline / repair-only GPT: stamp realtime trace with actual GPT values.
        if (
            not _actual_shadow_mode
            and trace is not None
            and _gpt_shadow is not None
        ):
            _, _inline_gpt = _gpt_shadow
            if _inline_gpt is not None and _inline_gpt is not GPT_NOT_CALLED:
                for _attr, _val in (
                    ("gpt_called", True),
                    ("gpt_decision", _inline_gpt.decision or ""),
                    ("gpt_selected_intent", _inline_gpt.repaired_intent or ""),
                    ("gpt_confidence", _inline_gpt.confidence),
                    ("gpt_total_ms", _inline_gpt.total_ms),
                    ("gpt_timeout", _inline_gpt.timeout),
                    ("gpt_applied", _inline_gpt.applied),
                    ("gpt_fallback_type", _inline_gpt.fallback_type),
                ):
                    if hasattr(trace, _attr):
                        setattr(trace, _attr, _val)

        # GPT fallback application gate.

        # Fallback may ONLY be applied when:
        #   config.apply_fallbacks == True
        #   AND call_mode is not all_shadow (shadow mode never applies)
        # With apply_fallbacks defaulting to False, fallbacks are logged-only unless
        # explicitly enabled.  In all_shadow mode the gate is always closed.
        _gpt_fallback_allowed = (
            _gpt_cfg_now.apply_fallbacks
            and not _actual_shadow_mode
        )
        if _gpt_shadow is not None:
            _fb_analysis, _fb_gpt = _gpt_shadow
            if (
                _fb_gpt.decision == "fallback"
                and _fb_gpt.fallback_type != "none"
                and _gpt_fallback_allowed
            ):
                _fb_key = f"fallback_{_fb_gpt.fallback_type}"
                self.response_writer._apply_session_response(
                    session=session,
                    intent=intent_result.intent,
                    response_key=_fb_key,
                    response_payload={},
                )
                total_ms = (time.perf_counter() - t_total_start) * 1000.0
                _diag = self.diagnostics._update_session_diagnostics(
                    session=session,
                    state_before=state_before,
                    nlu=nlu,
                    response_key=_fb_key,
                    response_payload={},
                )
                self._record_turn_event(
                    session=session,
                    state_before=state_before,
                    nlu=nlu,
                    result=None,
                    response_key=_fb_key,
                    response_payload={},
                    diag=_diag,
                    preprocess_ms=t_preprocess * 1000.0,
                    nlu_ms=t_nlu * 1000.0,
                    flow_ms=0.0,
                    route_ms=0.0,
                    handler_ms=0.0,
                    total_ms=total_ms,
                    coercion_reason=_coercion_reason,
                    gpt_shadow=_gpt_shadow,
                    gpt_fallback_applied=True,
                )
                self.diagnostics._finalize_trace_and_timing(
                    trace=trace,
                    session=session,
                    response_key=_fb_key,
                    total_start_monotonic=t_total_start,
                    total_ms=total_ms,
                    preprocess_ms=t_preprocess * 1000.0,
                    nlu_ms=t_nlu * 1000.0,
                    flow_ms=0.0,
                    route_ms=0.0,
                    handler_ms=0.0,
                )
                self._record_turn_memory(
                    session=session,
                    normalized_text=nlu.normalized_text or "",
                    response_key=_fb_key,
                    response_payload={},
                )
                return self.response_writer._hydrate_output(
                    session=session,
                    output=TurnOutput(
                        response_key=_fb_key,
                        response_payload={},
                    ),
                )

        # Runs AFTER GPT shadow repair/fallback gate, BEFORE FlowControlPolicy.
        # Never mutates cart, state, intent_result, slots, or response.
        if _get_gpt_cfg().add_item_mode == "shadow":
            _ai_shadow_decision: str | None = None
            _ai_shadow_intent: str | None = None
            if _gpt_shadow is not None:
                _, _ai_gpt_ref = _gpt_shadow
                if _ai_gpt_ref is not None and _ai_gpt_ref is not GPT_NOT_CALLED:
                    _ai_shadow_decision = _ai_gpt_ref.decision
                    _ai_shadow_intent = _ai_gpt_ref.repaired_intent
            try:
                _add_item_plan = self.add_item_extractor.run(
                    session=session,
                    nlu=nlu,
                    intent_result=intent_result,
                    state=session.conversation_state,
                    intent_candidates=getattr(nlu, "intent_candidates", None),
                    gpt_shadow_decision=_ai_shadow_decision,
                    gpt_shadow_repaired_intent=_ai_shadow_intent,
                    menu_repo=self.menu_repo,
                )
            except Exception as _ai_err:
                print(f"[ADD_ITEM_EXTRACTOR_ERROR] {type(_ai_err).__name__}: {_ai_err}")

        # Stamp realtime trace notes with ADD_ITEM summary (for realtime_turn_latency.csv)
        if trace is not None and _add_item_plan is not None:
            _ai_trace_notes = getattr(trace, "notes", None)
            if isinstance(_ai_trace_notes, dict):
                _vp = _add_item_plan.validated_plan
                _ai_trace_notes["add_item"] = {
                    "add_item_extractor_called": _add_item_plan.total_ms is not None,
                    "add_item_decision": _add_item_plan.decision or "",
                    "add_item_items_count": len(_add_item_plan.items) if _add_item_plan.items else None,
                    "add_item_confidence": _add_item_plan.confidence,
                    "add_item_total_ms": _add_item_plan.total_ms,
                    # Phase 2 validator summary
                    "add_item_validated_items_count": (
                        len(getattr(_vp, "items", ()) or ()) if _vp is not None else None
                    ),
                    "add_item_has_blocking_warnings": _add_item_plan.has_blocking_warnings,
                    "add_item_validator_ms": _add_item_plan.validator_ms,
                }

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
                gpt_shadow=_gpt_shadow,

                add_item_plan=_add_item_plan,
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
                gpt_shadow=_gpt_shadow,

                add_item_plan=_add_item_plan,
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
                    gpt_shadow=_gpt_shadow,

                    add_item_plan=_add_item_plan,
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
                    gpt_shadow=_gpt_shadow,

                    add_item_plan=_add_item_plan,
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
                gpt_shadow=_gpt_shadow,

                add_item_plan=_add_item_plan,
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
            gpt_shadow=_gpt_shadow,
            add_item_plan=_add_item_plan,
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

        self._record_turn_memory(
            session=session,
            normalized_text=nlu.normalized_text or "",
            response_key=result.response_key,
            response_payload=result.response_payload,
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
