# app/diagnostics/backends/json_backend.py
"""JsonDiagnosticsBackend — appends full TurnEvent records as JSONL."""
from __future__ import annotations

import dataclasses
import json
import queue
import threading
from pathlib import Path
from typing import Any

from app.diagnostics.turn_event import TurnEvent


def _default_serializer(obj: Any) -> Any:
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)


def _event_to_dict(event: TurnEvent) -> dict[str, Any]:
    d: dict[str, Any] = {}
    for f in dataclasses.fields(event):
        val = getattr(event, f.name)
        if isinstance(val, tuple):
            val = list(val)
        d[f.name] = val
    return d


class JsonDiagnosticsBackend:
    """Thread-safe JSONL writer for full TurnEvent records.

    Each record is a single JSON object followed by a newline.  The writer
    thread is started lazily on the first ``record()`` call so construction
    is cheap.
    """

    def __init__(self, log_path: str | Path, *, queue_maxsize: int = 5000) -> None:
        self._log_path = Path(log_path)
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=queue_maxsize)
        self._stop_event = threading.Event()
        self._writer_thread: threading.Thread | None = None
        self._dropped = 0
        self._started = False
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return True

    def record(self, event: TurnEvent) -> None:
        self._ensure_started()
        try:
            self._queue.put_nowait(_event_to_dict(event))
        except queue.Full:
            self._dropped += 1
            if self._dropped % 100 == 1:
                print("[JSON_DIAGNOSTICS_QUEUE_FULL]", {"dropped": self._dropped})

    def _ensure_started(self) -> None:
        if self._started:
            return
        with self._lock:
            if self._started:
                return
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            self._writer_thread = threading.Thread(
                target=self._writer_loop,
                name="json-diagnostics-writer",
                daemon=True,
            )
            self._writer_thread.start()
            self._started = True

    def _writer_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                row = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                line = json.dumps(row, default=_default_serializer, ensure_ascii=False)
                with self._log_path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            except Exception as exc:
                print(f"[JSON_DIAGNOSTICS_WRITE_ERROR] {type(exc).__name__}: {exc}")
            finally:
                self._queue.task_done()

    def flush(self, timeout: float | None = None) -> None:
        if self._started:
            self._queue.join()

    def shutdown(self) -> None:
        self.flush()
        self._stop_event.set()
        if self._writer_thread is not None:
            self._writer_thread.join(timeout=1.0)
