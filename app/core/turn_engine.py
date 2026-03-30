# app/core/turn_engine.py
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

from app.cart.read_models.cart_summary_builder import CartSummaryBuilder
from app.core.flow_control.flow_decision import FlowAction
from app.core.flow_control.flow_control_policy import FlowControlPolicy
from app.logging.nlu_csv_logger import NluCsvLogger
from app.menu.query_result import MenuQueryType
from app.menu.repository import MenuRepository
from app.ml.intent.inference_intent import IntentBundle
from app.ml.slot.inference_slot import SlotBundle
from app.nlu.intent_resolution.intent import Intent
from app.nlu.intent_resolution.intent_result import IntentResult
from app.nlu.nlu_resolver import resolve_nlu
from app.nlu.query_normalization.text_preprocessor import preprocess_turn_text
from app.session.session import Session
from app.state_machine.conversation_state import ConversationState
from app.state_machine.handler_result import HandlerResult
from app.state_machine.handlers.cart.cart_handlers import CartHandler
from app.state_machine.handlers.common.cancellation_confirmation_handler import (
    CancellationConfirmationHandler,
)
from app.state_machine.handlers.common.waiting_for_quantity_handler import (
    WaitingForQuantityHandler,
)
from app.state_machine.handlers.info.ask_menu_info_handler import AskMenuInfoHandler
from app.state_machine.handlers.info.ask_price_handler import AskPriceHandler
from app.state_machine.handlers.item.add_item.add_item_handler import AddItemHandler
from app.state_machine.handlers.item.add_item.waiting_for_modifier_handler import (
    WaitingForModifierHandler,
)
from app.state_machine.handlers.item.add_item.waiting_for_side_handler import (
    WaitingForSideHandler,
)
from app.state_machine.handlers.item.add_item.waiting_for_side_size_handler import (
    WaitingForSideSizeHandler,
)
from app.state_machine.handlers.item.add_item.waiting_for_size_handler import (
    WaitingForSizeHandler,
)
from app.state_machine.handlers.item.confirming_handler import ConfirmingHandler
from app.state_machine.handlers.item.remove_item_handler import RemoveItemHandler
from app.state_machine.handlers.item.removing_item_handler import RemovingItemHandler
from app.state_machine.handlers.order.confirm_order_handler import ConfirmOrderHandler
from app.state_machine.handlers.order.start_order_handler import StartOrderHandler
from app.state_machine.handlers.payment.waiting_for_payment_handler import (
    WaitingForPaymentHandler,
)
from app.state_machine.resume_prompt_builder import ResumePromptBuilder
from app.state_machine.state_router import StateRouter


@dataclass(frozen=True, slots=True)
class TurnOutput:
    response_key: str
    response_payload: dict[str, Any] | None = None
    internal_response_text: str | None = None
    spoken_response_text: str | None = None


INTENT_MIN_CONF = float(os.getenv("COMPASS_INTENT_CONF_THRESHOLD", "0.55"))
TURN_TIMING_ENABLED = os.getenv("COMPASS_TURN_TIMING_ENABLED", "0") == "1"
ROUTE_DEBUG_ENABLED = os.getenv("COMPASS_ROUTE_DEBUG_ENABLED", "0") == "1"

CONFIRMING_ORDER_EXIT_TO_IDLE_INTENTS: set[Intent] = {
    Intent.ADD_ITEM,
    Intent.REMOVE_ITEM,
    Intent.ASK_ITEM_INFO,
    Intent.ASK_MENU_INFO,
    Intent.ASK_OPTIONS,
    Intent.AVAILABILITY_QUERY,
    Intent.BROWSE_MENU,
    Intent.BROWSE_CATEGORY,
    Intent.RECOMMENDATION_QUERY,
    Intent.SHOW_MENU,
}

WAITING_STATE_ALLOWED_CONTROL_INTENTS = {
    Intent.DENY,
    Intent.CANCEL,
    Intent.CANCEL_ORDER,
    Intent.ASK_OPTIONS,
    Intent.SHOW_CART,
    Intent.SHOW_TOTAL,
    Intent.ASK_PRICE,
    Intent.ASK_ITEM_INFO,
    Intent.ASK_MENU_INFO,
    Intent.AVAILABILITY_QUERY,
    Intent.BROWSE_MENU,
    Intent.BROWSE_CATEGORY,
    Intent.RECOMMENDATION_QUERY,
    Intent.SHOW_MENU,
}

class TurnEngine:
    """
    Stateless turn processor.

    Orchestration only:
      input -> preprocess -> NLU -> flow guard -> route -> handler -> commands -> response

    The optional ``trace`` argument is intentionally duck-typed so the engine can enrich
    realtime latency traces without taking a hard dependency on a specific logger model.
    """

    def __init__(
        self,
        router: StateRouter,
        menu_repo: MenuRepository,
        intent_bundle: IntentBundle,
        slot_bundle: SlotBundle,
        nlu_logger: NluCsvLogger | None = None,
    ) -> None:
        self.router = router
        self.menu_repo = menu_repo
        self.intent_bundle = intent_bundle
        self.slot_bundle = slot_bundle
        self.cart_summary_builder = CartSummaryBuilder(menu_repo)
        self.flow_policy = FlowControlPolicy()
        self.nlu_logger = nlu_logger or NluCsvLogger()
        self.resume_prompt_builder = ResumePromptBuilder()

        self.handlers: dict[str, Any] = {
            "add_item_handler": AddItemHandler(menu_repo=menu_repo),
            "waiting_for_side_handler": WaitingForSideHandler(menu_repo),
            "waiting_for_modifier_handler": WaitingForModifierHandler(menu_repo),
            "waiting_for_size_handler": WaitingForSizeHandler(menu_repo),
            "waiting_for_side_size_handler": WaitingForSideSizeHandler(menu_repo),
            "waiting_for_quantity_handler": WaitingForQuantityHandler(),
            "confirming_handler": ConfirmingHandler(menu_repo),
            "remove_item_handler": RemoveItemHandler(menu_repo),
            "removing_item_handler": RemovingItemHandler(),
            "start_order_handler": StartOrderHandler(self.cart_summary_builder),
            "confirming_order_handler": ConfirmOrderHandler(self.cart_summary_builder),
            "waiting_for_payment_handler": WaitingForPaymentHandler(),
            "cart_handler": CartHandler(self.cart_summary_builder),
            "cancellation_confirmation_handler": CancellationConfirmationHandler(),
            "ask_menu_info_handler": AskMenuInfoHandler(menu_repo),
            "ask_price_handler": AskPriceHandler(menu_repo),
        }

    def process_turn(
        self,
        session: Session,
        user_text: str,
        trace: Any | None = None,
    ) -> TurnOutput:
        t_total_start = time.perf_counter()

        ctx = session.conversation_context
        state_before = session.conversation_state
        ctx.last_user_text = user_text

        self._trace_set_initial_fields(
            trace=trace,
            session=session,
            user_text=user_text,
            state_before=state_before,
            engine_start_monotonic=t_total_start,
        )

        t0 = time.perf_counter()
        preprocessed = preprocess_turn_text(user_text)
        cleaned_text = preprocessed.cleaned_text
        normalized_text = preprocessed.normalized_text
        t_preprocess = time.perf_counter() - t0

        self._trace_set_attr(trace, "cleaned_text", cleaned_text)
        self._trace_set_attr(trace, "normalized_text", normalized_text)

        t0 = time.perf_counter()
        nlu = resolve_nlu(
            raw_text=cleaned_text,
            normalized_text=normalized_text,
            state=session.conversation_state,
            pending_action=ctx.pending_action,
            intent_bundle=self.intent_bundle,
            slot_bundle=self.slot_bundle,
        )
        ctx.set_last_nlu(user_text=cleaned_text, nlu=nlu)
        t_nlu = time.perf_counter() - t0

        self._trace_set_nlu_fields(trace=trace, nlu=nlu)

        detected_intent = (
            nlu.effective_intent
            if nlu.intent_confidence >= INTENT_MIN_CONF
            else Intent.UNKNOWN
        )
        intent_result = IntentResult(
            intent=detected_intent,
            raw_text=nlu.normalized_text,
        )

        if session.conversation_state in {
            ConversationState.WAITING_FOR_SIDE,
            ConversationState.WAITING_FOR_SIDE_SIZE,
            ConversationState.WAITING_FOR_MODIFIER,
            ConversationState.WAITING_FOR_SIZE,
            ConversationState.WAITING_FOR_QUANTITY,
        } and intent_result.intent not in WAITING_STATE_ALLOWED_CONTROL_INTENTS:
            intent_result = IntentResult(
                intent=Intent.UNKNOWN,
                raw_text=nlu.normalized_text,
            )

        session.conversation_state = self._rewrite_confirming_order_to_idle_if_needed(
            session=session,
            intent=intent_result.intent,
        )

        intent_result, shortcut_output = self._apply_idle_shortcuts(session, intent_result)
        if shortcut_output is not None:
            self._apply_session_response(
                session=session,
                intent=intent_result.intent,
                response_key=shortcut_output.response_key,
                response_payload=shortcut_output.response_payload,
            )

            self._log_if_enabled(
                session=session,
                state_before=state_before,
                nlu=nlu,
                result=None,
                response_key=shortcut_output.response_key,
            )

            total_ms = (time.perf_counter() - t_total_start) * 1000.0
            self._finalize_trace_and_timing(
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
            return shortcut_output

        intent_result = self._rewrite_idle_unknown_menu_followup(
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

            self._apply_session_response(
                session=session,
                intent=intent_result.intent,
                response_key=response_key,
                response_payload=payload,
            )

            self._log_if_enabled(
                session=session,
                state_before=state_before,
                nlu=nlu,
                result=None,
                response_key=response_key,
            )

            total_ms = (time.perf_counter() - t_total_start) * 1000.0
            self._finalize_trace_and_timing(
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
            return TurnOutput(
                response_key=response_key,
                response_payload=payload,
            )

        if flow.action == FlowAction.CANCEL:
            ctx.awaiting_flow_confirmation = True
            ctx.return_state = session.conversation_state
            ctx.interrupt_proposal = None

            session.conversation_state = ConversationState.CANCELLATION_CONFIRMATION
            response_key = flow.response_key or "flow_guard_confirm_cancel"
            response_payload = dict(flow.response_payload or {})

            self._apply_session_response(
                session=session,
                intent=intent_result.intent,
                response_key=response_key,
                response_payload=response_payload,
            )

            self._log_if_enabled(
                session=session,
                state_before=state_before,
                nlu=nlu,
                result=None,
                response_key=response_key,
            )

            total_ms = (time.perf_counter() - t_total_start) * 1000.0
            self._finalize_trace_and_timing(
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
            return TurnOutput(
                response_key=response_key,
                response_payload=response_payload,
            )

        if flow.action == FlowAction.HANDLE_READONLY_INTERRUPT:
            readonly_output = self._handle_readonly_interrupt(
                session=session,
                state_before=state_before,
                intent_result=intent_result,
                nlu=nlu,
                trace=trace,
            )
            if readonly_output is not None:
                total_ms = (time.perf_counter() - t_total_start) * 1000.0
                self._finalize_trace_and_timing(
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

            self._apply_session_response(
                session=session,
                intent=intent_result.intent,
                response_key="intent_not_allowed",
                response_payload=payload,
            )

            self._log_if_enabled(
                session=session,
                state_before=state_before,
                nlu=nlu,
                result=None,
                response_key="intent_not_allowed",
            )

            total_ms = (time.perf_counter() - t_total_start) * 1000.0
            self._finalize_trace_and_timing(
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
            return TurnOutput(
                response_key="intent_not_allowed",
                response_payload=payload,
            )

        handler = self.handlers.get(route.handler_name)
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

        if result.command:
            self._apply_command(session, result.command)

        if result.reset_context:
            ctx.reset()

        session.conversation_state = result.next_state
        self._apply_session_response(
            session=session,
            intent=intent_result.intent,
            response_key=result.response_key,
            response_payload=result.response_payload,
        )

        self._log_if_enabled(
            session=session,
            state_before=state_before,
            nlu=nlu,
            result=result,
            response_key=result.response_key,
        )

        total_ms = (time.perf_counter() - t_total_start) * 1000.0
        self._finalize_trace_and_timing(
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

        return TurnOutput(
            response_key=result.response_key,
            response_payload=result.response_payload,
            internal_response_text=getattr(result, "internal_response_text", None),
            spoken_response_text=getattr(result, "spoken_response_text", None),
        )

    def _apply_session_response(
        self,
        *,
        session: Session,
        intent: Intent,
        response_key: str,
        response_payload: dict[str, Any] | None,
    ) -> None:
        session.last_intent = intent
        session.last_response_key = response_key
        session.last_response_payload = response_payload
        session.turn_count += 1

    def _log_if_enabled(
        self,
        *,
        session: Session,
        state_before: ConversationState,
        nlu: Any,
        result: HandlerResult | None,
        response_key: str,
    ) -> None:
        if not self.nlu_logger.enabled:
            return

        context_snapshot = self._snapshot_context_for_logging(session)
        self._log_nlu_and_turn(
            session=session,
            state_before=state_before,
            context_snapshot=context_snapshot,
            nlu=nlu,
            result=result,
            response_key=response_key,
        )

    def _finalize_trace_and_timing(
        self,
        *,
        trace: Any | None,
        session: Session,
        response_key: str,
        total_start_monotonic: float,
        total_ms: float,
        preprocess_ms: float,
        nlu_ms: float,
        flow_ms: float,
        route_ms: float,
        handler_ms: float,
        command: dict[str, Any] | None = None,
    ) -> None:
        self._trace_finalize(
            trace=trace,
            session=session,
            response_key=response_key,
            total_start_monotonic=total_start_monotonic,
            total_ms=total_ms,
            preprocess_ms=preprocess_ms,
            nlu_ms=nlu_ms,
            flow_ms=flow_ms,
            route_ms=route_ms,
            handler_ms=handler_ms,
            command=command,
        )

        self._maybe_print_timing(
            total_ms=total_ms,
            preprocess_ms=preprocess_ms,
            nlu_ms=nlu_ms,
            flow_ms=flow_ms,
            route_ms=route_ms,
            handler_ms=handler_ms,
        )

    def _maybe_print_timing(
        self,
        *,
        total_ms: float,
        preprocess_ms: float,
        nlu_ms: float,
        flow_ms: float,
        route_ms: float,
        handler_ms: float,
    ) -> None:
        if not TURN_TIMING_ENABLED:
            return

        print(
            "[TURN_TIMING]",
            {
                "total_ms": round(total_ms, 3),
                "preprocess_ms": round(preprocess_ms, 3),
                "nlu_ms": round(nlu_ms, 3),
                "flow_ms": round(flow_ms, 3),
                "route_ms": round(route_ms, 3),
                "handler_ms": round(handler_ms, 3),
            },
        )

    def _handle_readonly_interrupt(
        self,
        *,
        session: Session,
        state_before: ConversationState,
        intent_result: IntentResult,
        nlu: Any,
        trace: Any | None = None,
    ) -> TurnOutput | None:
        handler_name = self._readonly_interrupt_handler_name(intent_result.intent)
        if handler_name is None:
            return None

        handler = self.handlers.get(handler_name)
        if handler is None:
            raise KeyError(f"Handler not registered: {handler_name}")

        preserved_state = session.conversation_state
        context_snapshot = (
            self._snapshot_context_for_logging(session)
            if self.nlu_logger.enabled
            else {}
        )

        result: HandlerResult = handler.handle(
            intent=intent_result.intent,
            context=session.conversation_context,
            user_text=nlu.normalized_text,
            session=session,
        )

        if result.command is not None:
            raise ValueError(
                f"Read-only interrupt handler {handler_name} returned command={result.command}. "
                "Read-only interrupt handlers must not mutate session state."
            )

        if result.reset_context:
            raise ValueError(
                f"Read-only interrupt handler {handler_name} attempted reset_context=True."
            )

        resume = self.resume_prompt_builder.build(session)

        session.conversation_state = preserved_state
        session.last_intent = intent_result.intent

        if resume is None:
            session.last_response_key = result.response_key
            session.last_response_payload = result.response_payload
            session.turn_count += 1

            if self.nlu_logger.enabled:
                self._log_nlu_and_turn(
                    session=session,
                    state_before=state_before,
                    context_snapshot=context_snapshot,
                    nlu=nlu,
                    result=result,
                    response_key=result.response_key,
                )

            self._trace_set_attr(trace, "response_key", result.response_key)

            return TurnOutput(
                response_key=result.response_key,
                response_payload=result.response_payload,
            )

        resume_key, resume_payload = resume
        combined_payload = {
            "interrupt_response_key": result.response_key,
            "interrupt_response_payload": result.response_payload,
            "resume_response_key": resume_key,
            "resume_response_payload": resume_payload,
        }

        session.last_response_key = "readonly_interrupt_with_resume"
        session.last_response_payload = combined_payload
        session.turn_count += 1

        if self.nlu_logger.enabled:
            self._log_nlu_and_turn(
                session=session,
                state_before=state_before,
                context_snapshot=context_snapshot,
                nlu=nlu,
                result=result,
                response_key="readonly_interrupt_with_resume",
            )

        self._trace_set_attr(trace, "response_key", "readonly_interrupt_with_resume")

        return TurnOutput(
            response_key="readonly_interrupt_with_resume",
            response_payload=combined_payload,
        )

    def _readonly_interrupt_handler_name(self, intent: Intent) -> str | None:
        if intent == Intent.ASK_PRICE:
            return "ask_price_handler"

        if intent in {
            Intent.ASK_ITEM_INFO,
            Intent.ASK_MENU_INFO,
            Intent.ASK_OPTIONS,
            Intent.AVAILABILITY_QUERY,
            Intent.BROWSE_MENU,
            Intent.BROWSE_CATEGORY,
            Intent.RECOMMENDATION_QUERY,
            Intent.SHOW_MENU,
        }:
            return "ask_menu_info_handler"

        if intent in {Intent.SHOW_CART, Intent.SHOW_TOTAL}:
            return "cart_handler"

        return None

    def _apply_command(self, session: Session, command: dict[str, Any]) -> None:
        cmd_type = command.get("type")
        payload = command.get("payload") or {}

        if cmd_type == "ADD_ITEM_TO_CART":
            from app.cart.cart_item import CartItem

            cart_item = CartItem.create(
                item_id=payload["item_id"],
                quantity=payload["quantity"],
                variant_id=payload.get("variant_id"),
                sides=payload.get("sides", {}),
                side_variants=payload.get("side_variants", {}),
                modifiers=payload.get("modifiers", {}),
            )
            session.cart.add_item(cart_item)
            return

        if cmd_type == "CLEAR_CART":
            session.cart.clear()
            return

        if cmd_type == "REMOVE_ITEM_FROM_CART":
            session.cart.remove_item(payload["cart_item_id"])
            return

        raise ValueError(f"Unknown command type: {cmd_type}")

    def _safe_session_id(self, session: Session) -> str:
        value = getattr(session, "session_id", None)
        if value is not None:
            return str(value)

        value = getattr(session, "id", None)
        if value is not None:
            return str(value)

        return "unknown_session"

    def _safe_pending_action_from_context(self, context: Any) -> str:
        action = getattr(context, "pending_action", None)
        return action.value if action is not None else ""

    def _safe_current_prompt_field_from_context(self, context: Any) -> str:
        return getattr(context, "current_prompt_field", None) or ""

    def _safe_current_item_id_from_context(self, context: Any) -> str:
        return getattr(context, "current_item_id", None) or ""

    def _safe_current_item_name_from_context(self, context: Any) -> str:
        return getattr(context, "current_item_name", None) or ""

    def _snapshot_context_for_logging(self, session: Session) -> dict[str, str]:
        ctx = session.conversation_context
        return {
            "pending_action": self._safe_pending_action_from_context(ctx),
            "current_prompt_field": self._safe_current_prompt_field_from_context(ctx),
            "current_item_id": self._safe_current_item_id_from_context(ctx),
            "current_item_name": self._safe_current_item_name_from_context(ctx),
        }

    def _log_nlu_and_turn(
        self,
        *,
        session: Session,
        state_before: ConversationState,
        context_snapshot: dict[str, str],
        nlu: Any,
        result: HandlerResult | None,
        response_key: str,
    ) -> None:
        try:
            slot_values = getattr(nlu, "slots", ()) or ()

            self.nlu_logger.log_turn(
                session_id=self._safe_session_id(session),
                turn_index=session.turn_count,
                state_before=state_before.value,
                state_after=session.conversation_state.value,
                pending_action=context_snapshot["pending_action"],
                current_prompt_field=context_snapshot["current_prompt_field"],
                current_item_id=context_snapshot["current_item_id"],
                current_item_name=context_snapshot["current_item_name"],
                user_text=session.conversation_context.last_user_text or "",
                normalized_text=getattr(nlu, "normalized_text", "") or "",
                pred_main_intent=getattr(nlu, "model_main_intent", "") or "",
                pred_sub_intent=getattr(nlu, "model_sub_intent", "") or "",
                pred_intent=getattr(
                    getattr(nlu, "effective_intent", None),
                    "value",
                    "",
                )
                or "",
                pred_intent_confidence=getattr(nlu, "intent_confidence", None),
                slot_model_ran=bool(getattr(nlu, "slot_model_ran", False)),
                response_key=response_key,
                command=result.command if result else None,
                slots=slot_values,
            )
        except Exception as exc:
            print(f"[NLU_CSV_LOGGER_ERROR] {type(exc).__name__}: {exc}")

    def _apply_idle_shortcuts(
        self,
        session: Session,
        intent_result: IntentResult,
    ) -> tuple[IntentResult, TurnOutput | None]:
        if session.conversation_state != ConversationState.IDLE:
            return intent_result, None

        checkout_like_intents = {
            Intent.DENY,
            Intent.END_ADDING,
            Intent.START_ORDER,
            Intent.CHECKOUT,
            Intent.CONFIRM_ORDER,
            Intent.FINISH_ORDER,
            Intent.PAYMENT_REQUEST,
            Intent.REVIEW_ORDER,
        }

        if intent_result.intent not in checkout_like_intents:
            return intent_result, None

        if not session.cart.is_empty():
            return (
                IntentResult(
                    intent=Intent.START_ORDER,
                    raw_text=intent_result.raw_text,
                ),
                None,
            )

        return (
            intent_result,
            TurnOutput(
                response_key="idle_nothing_to_checkout",
                response_payload=None,
            ),
        )

    def _rewrite_confirming_order_to_idle_if_needed(
        self,
        *,
        session: Session,
        intent: Intent,
    ) -> ConversationState:
        if session.conversation_state != ConversationState.CONFIRMING_ORDER:
            return session.conversation_state

        if intent in CONFIRMING_ORDER_EXIT_TO_IDLE_INTENTS:
            return ConversationState.IDLE

        return session.conversation_state

    def _rewrite_idle_unknown_menu_followup(
        self,
        *,
        session: Session,
        intent_result: IntentResult,
        normalized_text: str,
    ) -> IntentResult:
        if session.conversation_state != ConversationState.IDLE:
            return intent_result

        if intent_result.intent != Intent.UNKNOWN:
            return intent_result

        if not normalized_text:
            return intent_result

        last_key = getattr(session, "last_response_key", None) or ""
        if last_key not in {
            "show_category",
            "menu_ambiguity",
            "show_menu_categories",
            "show_item_info",
            "show_item_availability",
        }:
            return intent_result

        result = self.menu_repo.resolve_menu_query_normalized(normalized_text, limit=5)
        if result.type in {
            MenuQueryType.ITEM,
            MenuQueryType.ITEM_AMBIGUOUS,
            MenuQueryType.CATEGORY,
            MenuQueryType.CATEGORY_SINGLE_ITEM,
            MenuQueryType.CATEGORY_AMBIGUOUS,
        }:
            return IntentResult(
                intent=Intent.ASK_ITEM_INFO,
                raw_text=intent_result.raw_text,
            )

        if last_key in {"show_category", "menu_ambiguity", "show_menu_categories"}:
            return IntentResult(
                intent=Intent.ASK_ITEM_INFO,
                raw_text=intent_result.raw_text,
            )

        return intent_result

    def _trace_set_initial_fields(
        self,
        *,
        trace: Any | None,
        session: Session,
        user_text: str,
        state_before: ConversationState,
        engine_start_monotonic: float,
    ) -> None:
        self._trace_set_attr(trace, "session_id", self._safe_session_id(session))
        self._trace_set_attr(trace, "user_text", user_text)
        self._trace_set_attr(trace, "state_before", state_before.value)
        self._trace_set_attr(trace, "engine_start_monotonic", engine_start_monotonic)

    def _trace_set_nlu_fields(self, *, trace: Any | None, nlu: Any) -> None:
        normalized_text = getattr(nlu, "normalized_text", "") or ""
        effective_intent = getattr(getattr(nlu, "effective_intent", None), "value", "") or ""

        self._trace_set_attr(trace, "normalized_text", normalized_text)
        self._trace_set_attr(trace, "pred_main_intent", getattr(nlu, "model_main_intent", "") or "")
        self._trace_set_attr(trace, "pred_sub_intent", getattr(nlu, "model_sub_intent", "") or "")
        self._trace_set_attr(trace, "pred_intent", effective_intent)
        self._trace_set_attr(trace, "pred_intent_confidence", getattr(nlu, "intent_confidence", None))
        self._trace_set_attr(trace, "stt_final_text_chars", len(normalized_text))

        slots = getattr(nlu, "slots", ()) or ()
        slot_names: list[str] = []
        slot_values: list[str] = []

        for slot in slots:
            label = getattr(slot, "label", None) or getattr(slot, "name", None)
            value = getattr(slot, "value", None)

            if isinstance(slot, dict):
                label = label or slot.get("label") or slot.get("name")
                value = value if value is not None else slot.get("value")

            if label:
                slot_names.append(str(label))
            if value is not None:
                slot_values.append(str(value))

        self._trace_set_attr(trace, "slot_names", slot_names)
        self._trace_set_attr(trace, "slot_values", slot_values)
        self._trace_set_attr(trace, "intent_model_ms", getattr(nlu, "intent_model_ms", None))
        self._trace_set_attr(trace, "slot_model_ms", getattr(nlu, "slot_model_ms", None))

    def _trace_finalize(
        self,
        *,
        trace: Any | None,
        session: Session,
        response_key: str,
        total_start_monotonic: float,
        total_ms: float,
        preprocess_ms: float,
        nlu_ms: float,
        flow_ms: float,
        route_ms: float,
        handler_ms: float,
        command: dict[str, Any] | None = None,
    ) -> None:
        engine_end = time.perf_counter()

        self._trace_set_attr(trace, "response_key", response_key)
        self._trace_set_attr(trace, "state_after", session.conversation_state.value)
        self._trace_set_attr(trace, "engine_end_monotonic", engine_end)
        self._trace_set_attr(trace, "turn_total_ms", round(total_ms, 3))
        self._trace_set_attr(trace, "preprocess_ms", round(preprocess_ms, 3))
        self._trace_set_attr(trace, "nlu_ms", round(nlu_ms, 3))
        self._trace_set_attr(trace, "flow_ms", round(flow_ms, 3))
        self._trace_set_attr(trace, "route_ms", round(route_ms, 3))
        self._trace_set_attr(trace, "handler_ms", round(handler_ms, 3))
        self._trace_set_attr(
            trace,
            "engine_total_ms",
            round((engine_end - total_start_monotonic) * 1000.0, 3),
        )
        if command is not None:
            self._trace_set_attr(trace, "command", command)

    def _trace_set_attr(self, trace: Any | None, attr_name: str, value: Any) -> None:
        if trace is None:
            return
        try:
            setattr(trace, attr_name, value)
        except Exception:
            return