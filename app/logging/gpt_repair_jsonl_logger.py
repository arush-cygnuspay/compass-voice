# app/logging/gpt_repair_jsonl_logger.py
"""Append-only JSONL logger for GPT repair turn records.

Each line is one JSON object produced by gpt_log_record_builder.  The logger
is thread-safe and uses a background writer queue.  Logging failures never
propagate to callers — they are swallowed and printed to stderr.

PII sanitization is applied to all string fields before writing:
  - OPENAI_API_KEY value (if present in the process environment)
  - Payment / checkout links (https://… containing "pay" or "checkout")
  - E-mail addresses
  - Phone numbers (digit sequences of 7+ digits, optionally with dashes/spaces)
"""
from __future__ import annotations

import json
import queue
import threading
from pathlib import Path
from typing import Any

from app.logging.nlu_csv_logger import rotate_log_file
# PII sanitization is centralised in the CSV logger so both loggers share one
# implementation.  Re-exported here so existing imports continue to work.
from app.logging.gpt_repair_csv_logger import sanitize_string, sanitize_record  # noqa: F401


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------


class GptRepairJsonlLogger:
    """Thread-safe, append-only JSONL writer for GPT repair records.

    Parameters
    ----------
    log_path:
        Path to the ``.jsonl`` file (will be created if absent).
    rotate_on_start:
        When True, the existing log file is moved to a sibling ``older/``
        directory before the first write (identical to NluCsvLogger rotation).
    queue_maxsize:
        Maximum number of pending records before new ones are dropped.
    """

    def __init__(
        self,
        log_path: str | Path = "app/logs/gpt_repair_turns.jsonl",
        *,
        rotate_on_start: bool = False,
        queue_maxsize: int = 5000,
    ) -> None:
        self.log_path = Path(log_path)
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=queue_maxsize)
        self._stop_event = threading.Event()
        self._writer_lock = threading.Lock()
        self._dropped = 0

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if rotate_on_start:
            rotate_log_file(self.log_path)

        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name="gpt-repair-jsonl-writer",
            daemon=True,
        )
        self._writer_thread.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_turn(self, record: dict[str, Any]) -> None:
        """Enqueue *record* for writing.  Never raises."""
        try:
            clean = sanitize_record(record)
            self._queue.put_nowait(clean)
        except queue.Full:
            self._dropped += 1
            if self._dropped % 100 == 1:
                print(f"[GPT_JSONL_LOGGER_QUEUE_FULL] dropped={self._dropped}")
        except Exception as exc:
            print(f"[GPT_JSONL_LOGGER_ENQUEUE_ERROR] {type(exc).__name__}: {exc}")

    def flush(self) -> None:
        """Block until all queued records are written."""
        self._queue.join()

    def shutdown(self) -> None:
        """Flush, then stop the background writer thread."""
        self.flush()
        self._stop_event.set()
        self._writer_thread.join(timeout=2.0)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _writer_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                record = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._write(record)
            except Exception as exc:
                print(f"[GPT_JSONL_LOGGER_WRITE_ERROR] {type(exc).__name__}: {exc}")
            finally:
                self._queue.task_done()

    def _write(self, record: dict[str, Any]) -> None:
        with self._writer_lock:
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False, default=str))
                fh.write("\n")
