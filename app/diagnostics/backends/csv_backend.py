# app/diagnostics/backends/csv_backend.py
"""CsvDiagnosticsBackend — adapts TurnEvent to NluCsvLogger.log_turn()."""
from __future__ import annotations

from app.diagnostics.turn_event import TurnEvent
from app.logging.nlu_csv_logger import NluCsvLogger


class CsvDiagnosticsBackend:
    """Maps the TurnEvent fields that NluCsvLogger accepts and enqueues a row."""

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
            pending_action=event.pending_action,
            current_prompt_field=event.current_prompt_field,
            current_item_id=event.current_item_id,
            current_item_name=event.current_item_name,
            user_text=event.user_text,
            normalized_text=event.normalized_text,
            pred_main_intent=event.pred_main_intent,
            pred_sub_intent=event.pred_sub_intent,
            pred_intent=event.pred_intent,
            pred_intent_confidence=event.pred_intent_confidence,
            slot_model_ran=event.slot_model_ran,
            response_key=event.response_key,
            command=event.command,
            slots=event.slots,
        )
