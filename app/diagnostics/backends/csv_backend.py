# app/diagnostics/backends/csv_backend.py
"""CsvDiagnosticsBackend — adapts TurnEvent to NluCsvLogger.log_turn()."""
from __future__ import annotations

from app.diagnostics.turn_event import TurnEvent
from app.logging.nlu_csv_logger import NluCsvLogger


class CsvDiagnosticsBackend:
    """Maps TurnEvent fields to NluCsvLogger.log_turn(), passing the full event
    as extra kwargs so test CapturingLoggers can assert any field."""

    def __init__(self, logger: NluCsvLogger) -> None:
        self._logger = logger

    @property
    def enabled(self) -> bool:
        return self._logger.enabled

    def record(self, event: TurnEvent) -> None:
        if not self._logger.enabled:
            return
        self._logger.log_turn(
            session_id=event.session_id,
            turn_index=event.turn_index,
            state_before=event.state_before,
            state_after=event.state_after,
            next_state=event.next_state,
            pending_action=event.pending_action,
            current_prompt_field=event.current_prompt_field,
            current_item_id=event.current_item_id,
            current_item_name=event.current_item_name,
            raw_user_text=event.raw_user_text,
            user_text=event.user_text,
            normalized_text=event.normalized_text,
            pred_main_intent=event.pred_main_intent,
            pred_sub_intent=event.pred_sub_intent,
            pred_intent=event.pred_intent,
            pred_intent_confidence=event.pred_intent_confidence,
            slot_model_ran=event.slot_model_ran,
            slots=event.slots,
            response_key=event.response_key,
            response_text=event.response_text,
            command=event.command,
            normalized_values=event.normalized_values,
            missing_required_fields=event.missing_required_fields,
            reprompt_field=event.reprompt_field,
            reprompt_count=event.reprompt_count,
            reprompt_escalated=event.reprompt_escalated,
            reprompt_escalation_count=event.reprompt_escalation_count,
            fallback_triggered=event.fallback_triggered,
            fallback_reason=event.fallback_reason,
            fallback_count=event.fallback_count,
            slot_extraction_failed=event.slot_extraction_failed,
            slot_extraction_failure_count=event.slot_extraction_failure_count,
            invalid_modifier=event.invalid_modifier,
            invalid_modifier_count=event.invalid_modifier_count,
            user_repeated=event.user_repeated,
            repeated_user_turn_count=event.repeated_user_turn_count,
            preprocess_ms=event.preprocess_ms,
            nlu_ms=event.nlu_ms,
            flow_ms=event.flow_ms,
            route_ms=event.route_ms,
            handler_ms=event.handler_ms,
            total_ms=event.total_ms,
        )
