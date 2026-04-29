"""Diagnostic helpers extracted from TurnEngine.

Pure plumbing for trace setup/finalize, context snapshotting, response-key
classification, session counter mutation, and backend fan-out.

No FSM transitions, no business decisions, no menu_repo, no responder.
All heavy field-building (response_text, normalized_values, missing_fields)
lives in TurnEngine, which assembles a TurnEvent and calls record().

Module-level constants (``REPROMPT_RESPONSE_KEYS``,
``FALLBACK_RESPONSE_KEYS``, ``TURN_TIMING_ENABLED``) are owned here and
re-exported from ``turn_engine`` for backwards compatibility.
"""
from __future__ import annotations

import os
import time
from typing import Any

from app.diagnostics.turn_event import TurnEvent
from app.nlu.intent_resolution.intent import Intent
from app.session.session import Session
from app.state_machine.handler_result import HandlerResult
from app.state_machine.models.conversation_state import ConversationState


TURN_TIMING_ENABLED = os.getenv("COMPASS_TURN_TIMING_ENABLED", "0") == "1"


FALLBACK_RESPONSE_KEYS: set[str] = {
    "repeat_side_options",
    "repeat_modifier_options",
    "repeat_size_options",
    "repeat_side_size_options",
    "invalid_size_option",
    "invalid_side_size_option",
    "invalid_quantity_option",
    "item_not_found",
    "intent_not_allowed",
    "item_context_missing",
    "confirmation_state_error",
    "payment_not_confirmed_yet",
    "payment_verification_error",
    "checkout_link_send_failed",
    "payment_link_send_failed",
    "payment_link_unavailable_now",
}
REPROMPT_RESPONSE_KEYS: set[str] = {
    "repeat_side_options",
    "repeat_modifier_options",
    "repeat_size_options",
    "repeat_side_size_options",
    "invalid_size_option",
    "invalid_side_size_option",
    "invalid_quantity_option",
    "required_side_cannot_skip",
    "required_modifier_cannot_skip",
    "required_size_cannot_skip",
    "required_side_size_cannot_skip",
}


class TurnDiagnostics:
    """Aggregates session counter mutation, trace helpers, and backend fan-out.

    Construction takes a list of ``DiagnosticsBackend`` instances.  Each
    backend receives a ``TurnEvent`` on every logged turn.  TurnEngine is
    responsible for building the event; this class only fans it out.
    """

    def __init__(self, backends: list[Any]) -> None:
        self._backends = backends

    # ------------------------------------------------------------------
    # Backend fan-out
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """True when at least one backend will accept records."""
        return any(getattr(b, "enabled", True) for b in self._backends)

    def record(self, event: TurnEvent) -> None:
        """Fan event out to all registered backends."""
        for backend in self._backends:
            try:
                backend.record(event)
            except Exception as exc:
                print(f"[DIAGNOSTICS_BACKEND_ERROR] {type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------
    # Session counter mutation (must run on every turn regardless of
    # whether any backend is enabled)
    # ------------------------------------------------------------------

    def _update_session_diagnostics(
        self,
        *,
        session: Session,
        state_before: ConversationState,
        nlu: Any,
        response_key: str,
        response_payload: dict[str, Any] | None,
    ) -> dict[str, Any]:
        normalized_text = getattr(nlu, "normalized_text", "") or ""
        user_repeated = bool(
            normalized_text
            and normalized_text == (session.last_normalized_user_text or "")
        )
        if user_repeated:
            session.repeated_user_turn_count += 1
        session.last_normalized_user_text = normalized_text or None

        field = self._infer_prompt_field_for_response(
            response_key=response_key,
            session=session,
        )
        reprompt_escalated = bool((response_payload or {}).get("reprompt_escalation"))
        is_quantity_reprompt = (
            state_before == ConversationState.WAITING_FOR_QUANTITY
            and response_key == "ask_for_quantity"
        )
        reprompt_count = int(session.reprompt_count_by_field.get(field, 0) or 0)
        if field:
            if (
                response_key in REPROMPT_RESPONSE_KEYS
                or response_key in {"list_side_options", "list_modifier_options"}
                or is_quantity_reprompt
            ):
                reprompt_count += 1
                session.reprompt_count_by_field[field] = reprompt_count
            else:
                reprompt_count = 0
                session.reprompt_count_by_field[field] = 0

        fallback_triggered = response_key in FALLBACK_RESPONSE_KEYS or response_key in REPROMPT_RESPONSE_KEYS
        if response_key in {"list_side_options", "list_modifier_options"}:
            fallback_triggered = True
        if fallback_triggered:
            session.fallback_count += 1
        if reprompt_escalated:
            session.reprompt_escalation_count += 1

        slot_extraction_failed = (
            state_before in {
                ConversationState.WAITING_FOR_SIDE,
                ConversationState.WAITING_FOR_MODIFIER,
                ConversationState.WAITING_FOR_SIZE,
                ConversationState.WAITING_FOR_SIDE_SIZE,
            }
            and response_key
            in {
                "repeat_side_options",
                "repeat_modifier_options",
                "invalid_size_option",
                "invalid_side_size_option",
                "list_side_options",
                "list_modifier_options",
            }
            and not (getattr(nlu, "slots", ()) or ())
        )
        if slot_extraction_failed:
            session.slot_extraction_failure_count += 1

        invalid_modifier = (
            state_before == ConversationState.WAITING_FOR_MODIFIER
            and response_key in {"repeat_modifier_options", "too_many_modifier_choices", "list_modifier_options"}
        )
        if invalid_modifier:
            session.invalid_modifier_count += 1

        fallback_reason = ""
        if reprompt_escalated:
            fallback_reason = "reprompt_escalation"
        elif slot_extraction_failed:
            fallback_reason = "slot_extraction_failed"
        elif invalid_modifier:
            fallback_reason = "invalid_modifier"
        elif fallback_triggered:
            fallback_reason = response_key

        return {
            "reprompt_field": field,
            "reprompt_count": reprompt_count,
            "reprompt_escalated": reprompt_escalated,
            "fallback_triggered": fallback_triggered,
            "fallback_reason": fallback_reason,
            "slot_extraction_failed": slot_extraction_failed,
            "invalid_modifier": invalid_modifier,
            "user_repeated": user_repeated,
        }

    # ------------------------------------------------------------------
    # Business-logic helper (used by HandlerDispatcher._apply_reprompt_guardrail)
    # ------------------------------------------------------------------

    def _infer_prompt_field_for_response(
        self,
        *,
        response_key: str,
        session: Session,
    ) -> str:
        key = response_key or ""
        if "modifier" in key:
            return "modifier"
        if "side_size" in key:
            return "side_size"
        if "size" in key:
            return "size"
        if "side" in key:
            return "side"
        if "quantity" in key:
            return "quantity"
        return session.conversation_context.current_prompt_field or ""

    # ------------------------------------------------------------------
    # Context snapshot helpers
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Trace helpers
    # ------------------------------------------------------------------

    def _trace_set_initial_fields(
        self,
        *,
        trace: Any | None,
        session: Session,
        user_text: str,
        state_before: ConversationState,
        engine_start_monotonic: float,
    ) -> None:
        ctx = session.conversation_context
        self._trace_set_attr(trace, "session_id", self._safe_session_id(session))
        self._trace_set_attr(trace, "user_text", user_text)
        self._trace_set_attr(trace, "state_before", state_before.value)
        self._trace_set_attr(trace, "pending_action", self._safe_pending_action_from_context(ctx))
        self._trace_set_attr(trace, "current_prompt_field", self._safe_current_prompt_field_from_context(ctx))
        self._trace_set_attr(trace, "current_item_id", self._safe_current_item_id_from_context(ctx))
        self._trace_set_attr(trace, "current_item_name", self._safe_current_item_name_from_context(ctx))
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
        ctx = session.conversation_context

        self._trace_set_attr(trace, "response_key", response_key)
        self._trace_set_attr(trace, "state_after", session.conversation_state.value)
        self._trace_set_attr(trace, "pending_action", self._safe_pending_action_from_context(ctx))
        self._trace_set_attr(trace, "current_prompt_field", self._safe_current_prompt_field_from_context(ctx))
        self._trace_set_attr(trace, "current_item_id", self._safe_current_item_id_from_context(ctx))
        self._trace_set_attr(trace, "current_item_name", self._safe_current_item_name_from_context(ctx))
        self._trace_set_attr(trace, "engine_end_monotonic", engine_end)
        self._trace_set_attr(trace, "turn_total_ms", round(total_ms, 3))
        self._trace_set_attr(trace, "preprocess_ms", round(preprocess_ms, 3))
        self._trace_set_attr(trace, "nlu_ms", round(nlu_ms, 3))
        self._trace_set_attr(trace, "flow_ms", round(flow_ms, 3))
        self._trace_set_attr(trace, "route_ms", round(route_ms, 3))
        self._trace_set_attr(trace, "handler_ms", round(handler_ms, 3))
        self._trace_set_attr(trace, "engine_total_ms", round((engine_end - total_start_monotonic) * 1000.0, 3))
        if command is not None:
            self._trace_set_attr(trace, "command", command)

    def _trace_set_attr(self, trace: Any | None, attr_name: str, value: Any) -> None:
        if trace is None:
            return
        try:
            setattr(trace, attr_name, value)
        except Exception:
            return
