# app/logging/turn_event_logger.py
"""Thread-safe, append-only JSONL writer for canonical TurnEvent records.

Each line of ``logs/current/turn_events.jsonl`` is one complete JSON object
representing everything that happened during a single conversation turn.

Design principles
-----------------
* Never crashes the call — all exceptions are swallowed after logging.
* Background write queue so the turn response path is never blocked.
* PII redaction via the existing ``sanitize_record`` helper.
* Custom JSON encoder handles enums, datetimes, dataclasses, frozensets.
* ISO-8601 timestamps are never redacted.
* API keys, phone numbers, e-mails, and payment links are redacted in text fields.

Log path
--------
Controlled by ``COMPASS_TURN_EVENTS_JSONL_PATH`` env var.
Default: ``logs/current/turn_events.jsonl``.

Rotation
--------
On process start, if ``COMPASS_ROTATE_TURN_EVENTS_ON_START=true``, the
existing file is moved to ``logs/older/`` with a timestamp suffix (mirrors
the NluCsvLogger rotation pattern).
"""
from __future__ import annotations

import json
import logging
import os
import queue
import threading
from dataclasses import asdict, fields, is_dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID

from app.logging.gpt_repair_csv_logger import sanitize_record
from app.logging.turn_event_schema import build_canonical_record
from app.logging.final_decision_resolver import resolve_final_decision
from app.logging.training_candidate_classifier import classify_training_candidate

if TYPE_CHECKING:
    from app.diagnostics.turn_event import TurnEvent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# JSON encoder
# ---------------------------------------------------------------------------

def _json_default(obj: Any) -> Any:
    """Custom JSON serialisation for types not handled by default encoder."""
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, UUID):
        return str(obj)
    if isinstance(obj, frozenset):
        return list(obj)
    if isinstance(obj, set):
        return list(obj)
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)  # type: ignore[call-overload]
    if hasattr(obj, "__iter__"):
        try:
            return list(obj)
        except Exception:
            pass
    return str(obj)


def _safe_dumps(record: dict[str, Any]) -> str:
    """Serialise *record* to a single JSON line.  Never raises."""
    try:
        return json.dumps(record, ensure_ascii=False, default=_json_default)
    except Exception as exc:
        # Last resort: dump a minimal error envelope
        try:
            return json.dumps(
                {
                    "schema_version": "1",
                    "error": f"serialisation_error: {type(exc).__name__}: {exc}",
                    "session_id": str(record.get("ids", {}).get("session_id", "")),
                },
                ensure_ascii=False,
            )
        except Exception:
            return '{"schema_version":"1","error":"fatal_serialisation_error"}'


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

class TurnEventLogger:
    """Thread-safe, append-only JSONL writer for canonical per-turn records.

    Parameters
    ----------
    log_path:
        Path to the ``.jsonl`` file.  Created (including parent dirs) on first
        write if absent.
    rotate_on_start:
        When True, the existing log file is moved to a sibling ``older/``
        directory on instantiation (same pattern as NluCsvLogger).
    queue_maxsize:
        Maximum number of pending records before new ones are dropped.
    sync_write_immediately:
        When True, bypass the background queue and write synchronously.
        Useful in tests.
    """

    def __init__(
        self,
        log_path: str | Path | None = None,
        *,
        rotate_on_start: bool = False,
        queue_maxsize: int = 5000,
        sync_write_immediately: bool | None = None,
    ) -> None:
        resolved_path = (
            log_path
            or os.getenv("COMPASS_TURN_EVENTS_JSONL_PATH", "logs/current/turn_events.jsonl")
        )
        self.log_path = Path(resolved_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        if rotate_on_start:
            self._rotate()

        self._sync = (
            sync_write_immediately
            if sync_write_immediately is not None
            else os.getenv("COMPASS_TURN_EVENTS_SYNC_WRITE", "0") == "1"
        )

        self._queue: queue.Queue[str] = queue.Queue(maxsize=queue_maxsize)
        self._stop_event = threading.Event()
        self._writer_lock = threading.Lock()
        self._dropped = 0

        if not self._sync:
            self._writer_thread = threading.Thread(
                target=self._writer_loop,
                name="turn-event-jsonl-writer",
                daemon=True,
            )
            self._writer_thread.start()
        else:
            self._writer_thread = None  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_turn(
        self,
        event: "TurnEvent",
        *,
        call_sid: str = "",
        stream_sid: str = "",
        store_id: str = "",
        company_id: str = "",
        previous_assistant_text: str = "",
        spoken_text: str = "",
        tts_chunks: int = 0,
        cart_before_hash: str | None = None,
        cart_after_hash: str | None = None,
        cart_diff: list[dict[str, Any]] | None = None,
        extra_errors: list[str] | None = None,
    ) -> None:
        """Build a canonical record from *event* and enqueue it for writing.

        Never raises — any exception is caught and logged to the standard
        Python logger (``app.logging.turn_event_logger``).
        """
        try:
            self._enqueue(
                event=event,
                call_sid=call_sid,
                stream_sid=stream_sid,
                store_id=store_id,
                company_id=company_id,
                previous_assistant_text=previous_assistant_text,
                spoken_text=spoken_text,
                tts_chunks=tts_chunks,
                cart_before_hash=cart_before_hash,
                cart_after_hash=cart_after_hash,
                cart_diff=cart_diff,
                extra_errors=extra_errors or [],
            )
        except Exception as exc:
            logger.error(
                "turn_event_logger.log_turn_error",
                exc_info=False,
                extra={"error": f"{type(exc).__name__}: {exc}"},
            )

    def flush(self) -> None:
        """Block until all queued records have been written to disk."""
        if not self._sync:
            self._queue.join()

    def shutdown(self) -> None:
        """Flush all pending records then stop the background writer."""
        self.flush()
        self._stop_event.set()
        if self._writer_thread is not None:
            self._writer_thread.join(timeout=2.0)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _enqueue(
        self,
        *,
        event: "TurnEvent",
        call_sid: str,
        stream_sid: str,
        store_id: str,
        company_id: str,
        previous_assistant_text: str,
        spoken_text: str,
        tts_chunks: int,
        cart_before_hash: str | None,
        cart_after_hash: str | None,
        cart_diff: list[dict[str, Any]] | None,
        extra_errors: list[str],
    ) -> None:
        # Derive cross-cutting fields
        final_decision = resolve_final_decision(event)
        training = classify_training_candidate(event, final_decision)

        # Build canonical dict
        record = build_canonical_record(
            event,
            final_decision=final_decision,
            training=training,
            call_sid=call_sid,
            stream_sid=stream_sid,
            store_id=store_id,
            company_id=company_id,
            previous_assistant_text=previous_assistant_text,
            spoken_text=spoken_text,
            tts_chunks=tts_chunks,
            cart_before_hash=cart_before_hash,
            cart_after_hash=cart_after_hash,
            cart_diff=cart_diff,
            errors=extra_errors,
        )

        # PII sanitization (reuses existing sanitize_record from gpt_repair_csv_logger)
        clean = sanitize_record(record)

        # Serialise to a single JSON line
        line = _safe_dumps(clean)

        if self._sync:
            self._write_line(line)
        else:
            try:
                self._queue.put_nowait(line)
            except queue.Full:
                self._dropped += 1
                if self._dropped % 100 == 1:
                    logger.warning(
                        "turn_event_logger.queue_full",
                        extra={"dropped_total": self._dropped},
                    )

    def _writer_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                line = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._write_line(line)
            except Exception as exc:
                logger.error(
                    "turn_event_logger.write_error",
                    extra={"error": f"{type(exc).__name__}: {exc}"},
                )
            finally:
                self._queue.task_done()

    def _write_line(self, line: str) -> None:
        with self._writer_lock:
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(line)
                fh.write("\n")

    def _rotate(self) -> None:
        """Move the existing log file to an ``older/`` sibling directory."""
        if not self.log_path.exists() or self.log_path.stat().st_size == 0:
            return
        try:
            older_dir = self.log_path.parent / "older"
            older_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            dest = older_dir / f"{self.log_path.stem}_{ts}{self.log_path.suffix}"
            self.log_path.rename(dest)
        except Exception as exc:
            logger.warning(
                "turn_event_logger.rotate_error",
                extra={"error": f"{type(exc).__name__}: {exc}"},
            )
