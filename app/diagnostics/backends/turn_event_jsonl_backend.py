# app/diagnostics/backends/turn_event_jsonl_backend.py
"""DiagnosticsBackend that writes canonical TurnEvent JSONL records.

Implements the DiagnosticsBackend protocol so it can be registered in
TurnDiagnostics without modifying TurnEngine logic — just add an instance
to the backends list.

The canonical log is the single source of truth for debugging, replay, and
NLU model training.  See app/logging/turn_event_schema.py for the full
record structure.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from app.logging.turn_event_logger import TurnEventLogger

if TYPE_CHECKING:
    from app.diagnostics.turn_event import TurnEvent


class TurnEventJsonlBackend:
    """Write each TurnEvent to the canonical turn_events.jsonl file.

    Parameters
    ----------
    logger:
        Pre-constructed TurnEventLogger instance.  TurnEngine should own
        the lifecycle (shutdown / flush) of this logger.
    call_sid_getter:
        Optional callable that returns the current Twilio call SID.
        If not provided, call_sid is left blank.
    stream_sid_getter:
        Same pattern for Twilio stream SID.
    store_id:
        Fixed restaurant / store identifier for all turns in this process.
    company_id:
        Fixed company identifier.
    """

    enabled: bool = True

    def __init__(
        self,
        logger: TurnEventLogger,
        *,
        store_id: str = "",
        company_id: str = "",
    ) -> None:
        self._logger = logger
        self._store_id = store_id
        self._company_id = company_id

    def record(self, event: "TurnEvent") -> None:
        """Write *event* to the canonical JSONL log.  Never raises."""
        self._logger.log_turn(
            event,
            store_id=self._store_id,
            company_id=self._company_id,
        )
