# app/logging/gpt_repair_csv_logger.py
"""Append-only CSV logger for GPT repair turn records.

Each row corresponds to one TurnEvent where GPT repair was eligible, called,
or a training candidate.  The logger is thread-safe and uses a background
writer queue.  Logging failures never propagate to callers.

PII sanitization is applied to all string fields before writing:
  - OPENAI_API_KEY value (if present in the process environment)
  - Payment / checkout links  (https://… containing "pay" or "checkout")
  - E-mail addresses
  - Phone numbers (digit sequences of 7+ digits, optionally separated)

Safety invariants:
  - ISO 8601 timestamps (YYYY-MM-DDT…) are never redacted.
  - Enum / log fields (gpt_decision, state_before, etc.) are skipped by the
    sanitizer so their values are always preserved as-is.
  - Fields ending in _json are sanitized with a JSON-aware path: the JSON is
    parsed, text-only leaf values are sanitized, then re-serialised.  If the
    JSON parse fails the raw string is sanitized safely without crashing.
"""
from __future__ import annotations

import csv
import json
import os
import queue
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.logging.nlu_csv_logger import _safe_float_str, _safe_str, rotate_log_file

# ---------------------------------------------------------------------------
# PII sanitization
# ---------------------------------------------------------------------------

_PAYMENT_LINK_RE = re.compile(
    r"https?://\S*(?:pay|checkout)\S*",
    re.IGNORECASE,
)
_EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
# Phone patterns:
#   - 7+ consecutive digits (no separators): catches 10-digit and international numbers
#   - NXX-NXX-XXXX and NXX NXX XXXX (3-3-4 North American format)
# ISO dates (YYYY-MM-DD, 4-2-2) are not matched by either pattern.
_PHONE_RE = re.compile(
    r"\b\d{7,}\b"                          # 7+ raw digits
    r"|\b\d{3}[\s\-]\d{3}[\s\-]\d{4}\b",  # NXX-NXX-XXXX or NXX NXX XXXX
    re.ASCII,
)

# ISO 8601 timestamp prefix — strings starting with this pattern must not be
# passed through the phone-number regex (e.g. "2024-01-15T12:34:56.789Z").
_ISO_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T")

# Fields that must never be passed through the PII sanitizer.
# Timestamps (ISO 8601) and IDs look like phone numbers to naive regexes.
# Enum / log fields must be preserved exactly so downstream analysis works.
_SANITIZE_SKIP_FIELDS: frozenset[str] = frozenset({
    # Identifiers and numeric counters
    "timestamp_utc",
    "session_id",
    "turn_index",
    "gpt_phase",
    "gpt_called",
    "gpt_repair_eligible",
    "gpt_applied",
    "training_candidate",
    "gpt_timeout",
    "local_route_allowed",
    "gpt_candidate_count",
    "gpt_prompt_chars",
    "gpt_completion_chars",
    "gpt_latency_ms",
    "gpt_total_ms",
    "local_confidence",
    "gpt_confidence",
    # Enum / log string fields — must be preserved exactly
    "gpt_decision",
    "gpt_apply_reason",
    "gpt_model",
    "local_intent",
    "local_sub_intent",
    "final_intent_after_gpt",
    "final_response_key",
    "response_key",
    "state_before",
    "gpt_eligible_reason",
    "gpt_skipped_reason",
    "gpt_fallback_type",
    "fallback_response_key",
    # ADD_ITEM extractor enum / log fields
    "add_item_decision",
    "add_item_reason",
    "add_item_model",
    "add_item_skipped_reason",
    # ADD_ITEM validator numeric / bool fields
    "add_item_validated_items_count",
    "add_item_validator_ms",
    "add_item_has_blocking_warnings",
})


def _get_api_key_pattern() -> re.Pattern[str] | None:
    key = os.getenv("OPENAI_API_KEY", "")
    if not key or len(key) < 8:
        return None
    return re.compile(re.escape(key))


def sanitize_string(text: str) -> str:
    """Apply PII redaction to a single string value.

    ISO 8601 timestamps are preserved — strings starting with
    YYYY-MM-DDT are returned as-is after API-key / URL / email
    redaction (phone regex is skipped for them).
    """
    key_pat = _get_api_key_pattern()
    if key_pat:
        text = key_pat.sub("[REDACTED_KEY]", text)
    text = _PAYMENT_LINK_RE.sub("[REDACTED_URL]", text)
    text = _EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    # Skip phone regex for ISO timestamps to avoid corrupting datetime values
    if not _ISO_TIMESTAMP_RE.match(text):
        text = _PHONE_RE.sub("[REDACTED_PHONE]", text)
    return text


def sanitize_record(record: Any, *, _field_name: str = "") -> Any:
    """Recursively sanitize all string values in a dict/list structure.

    For dict values, the field name is threaded through so that:
      - Fields in _SANITIZE_SKIP_FIELDS are passed through unchanged.
      - Fields ending in ``_json`` are sanitized with JSON-aware logic.
    """
    if isinstance(record, str):
        return sanitize_string(record)
    if isinstance(record, dict):
        out: dict[str, Any] = {}
        for k, v in record.items():
            if k in _SANITIZE_SKIP_FIELDS:
                out[k] = v
            elif k.endswith("_json") and isinstance(v, str):
                out[k] = _sanitize_json_field(v)
            else:
                out[k] = sanitize_record(v, _field_name=k)
        return out
    if isinstance(record, list):
        return [sanitize_record(item, _field_name=_field_name) for item in record]
    return record


def _sanitize_json_field(raw_json: str) -> str:
    """Sanitize a field that contains a JSON string.

    Parses the JSON, recursively sanitizes only text leaf values (not
    numeric/bool/null), and re-serialises.  Falls back to raw sanitization
    of the string if JSON parsing fails (never crashes).
    """
    try:
        parsed = json.loads(raw_json)
    except (json.JSONDecodeError, ValueError):
        # If it's not valid JSON, sanitize the raw string safely
        return sanitize_string(raw_json)
    sanitized = _sanitize_json_value(parsed)
    try:
        return json.dumps(sanitized, ensure_ascii=False)
    except Exception:
        return raw_json  # last resort: return original


def _sanitize_json_value(value: Any) -> Any:
    """Recursively sanitize text leaf values in a parsed JSON structure."""
    if isinstance(value, str):
        return sanitize_string(value)
    if isinstance(value, dict):
        return {k: _sanitize_json_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_json_value(item) for item in value]
    # int, float, bool, None — pass through unchanged
    return value


# ---------------------------------------------------------------------------
# Field-selective sanitization (apply PII only to text-like fields)
# ---------------------------------------------------------------------------


def _sanitize_row(row: dict[str, Any]) -> dict[str, Any]:
    """Sanitize only text-like fields; leave IDs, timestamps, and numbers untouched."""
    out: dict[str, Any] = {}
    for k, v in row.items():
        if k in _SANITIZE_SKIP_FIELDS:
            out[k] = v
        elif k.endswith("_json") and isinstance(v, str):
            out[k] = _sanitize_json_field(v)
        elif not isinstance(v, str):
            out[k] = v
        else:
            out[k] = sanitize_string(v)
    return out


# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------

HEADERS: list[str] = [
    "timestamp_utc",
    "session_id",
    "turn_index",
    "state_before",
    "response_key",
    # Customer utterance
    "user_text",
    "normalized_text",
    # Local model snapshot (before GPT / coercions)
    "local_intent",
    "local_sub_intent",
    "local_confidence",
    "local_candidates_json",
    "local_slots_json",
    "local_route_allowed",
    "local_route_reject_reason",
    # GPT eligibility
    "gpt_repair_eligible",
    "gpt_eligible_reason",
    "gpt_candidate_count",
    "gpt_skipped_reason",
    "gpt_phase",
    # GPT call
    "gpt_called",
    "gpt_decision",
    "gpt_selected_intent",
    "gpt_selected_control_intent",
    "gpt_slot_corrections_json",
    "gpt_confidence",
    "gpt_reason",
    "gpt_latency_ms",
    "gpt_total_ms",
    "gpt_timeout",
    "gpt_parse_error",
    "gpt_model",
    "gpt_prompt_chars",
    "gpt_completion_chars",
    # Final block
    "gpt_applied",
    "gpt_apply_reason",
    "final_intent_after_gpt",
    "final_response_key",
    "training_candidate",
    # Fallback classification
    "gpt_fallback_type",
    "fallback_response_key",
    "add_item_extractor_called",
    "add_item_eligible",
    "add_item_skipped_reason",
    "add_item_decision",
    "add_item_confidence",
    "add_item_items_json",
    "add_item_items_count",
    "add_item_global_slots_json",
    "add_item_latency_ms",
    "add_item_total_ms",
    "add_item_prompt_chars",
    "add_item_completion_chars",
    "add_item_timeout",
    "add_item_parse_error",
    "add_item_parse_notes_json",
    "add_item_reason",
    "add_item_model",
    # ── ADD_ITEM validator (Phase 2 shadow) ───────────────────────────────
    "add_item_validated_items_json",
    "add_item_validated_items_count",
    "add_item_rejected_items_json",
    "add_item_validation_warnings_json",
    "add_item_validator_ms",
    "add_item_has_blocking_warnings",
]


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------


class GptRepairCsvLogger:
    """Thread-safe, append-only CSV writer for GPT repair records.

    Parameters
    ----------
    log_path:
        Path to the ``.csv`` file (created on first write if absent).
    rotate_on_start:
        When True, the existing log file is archived to a sibling ``older/``
        directory before the first write.
    queue_maxsize:
        Capacity of the background write queue before records are dropped.
    """

    def __init__(
        self,
        log_path: str | Path = "app/logs/gpt_repair_turns.csv",
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

        self._ensure_header()

        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name="gpt-repair-csv-writer",
            daemon=True,
        )
        self._writer_thread.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_turn(self, row: dict[str, Any]) -> None:
        """Enqueue *row* for writing after field-selective PII sanitization.  Never raises."""
        try:
            clean = self._serialize(_sanitize_row(row))
            self._queue.put_nowait(clean)
        except queue.Full:
            self._dropped += 1
            if self._dropped % 100 == 1:
                print(f"[GPT_CSV_LOGGER_QUEUE_FULL] dropped={self._dropped}")
        except Exception as exc:
            print(f"[GPT_CSV_LOGGER_ENQUEUE_ERROR] {type(exc).__name__}: {exc}")

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

    def _ensure_header(self) -> None:
        if self.log_path.exists() and self.log_path.stat().st_size > 0:
            return
        with self._writer_lock:
            if self.log_path.exists() and self.log_path.stat().st_size > 0:
                return
            with self.log_path.open("w", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, fieldnames=HEADERS).writeheader()

    def _serialize(self, row: dict[str, Any]) -> dict[str, str]:
        """Convert all values to CSV-safe strings."""
        out: dict[str, str] = {}
        for col in HEADERS:
            val = row.get(col)
            if isinstance(val, bool):
                out[col] = "1" if val else "0"
            elif isinstance(val, float):
                out[col] = _safe_float_str(val)
            else:
                out[col] = _safe_str(val)
        return out

    def _writer_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                row = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._write(row)
            except Exception as exc:
                print(f"[GPT_CSV_LOGGER_WRITE_ERROR] {type(exc).__name__}: {exc}")
            finally:
                self._queue.task_done()

    def _write(self, row: dict[str, str]) -> None:
        with self._writer_lock:
            self._ensure_header()
            with self.log_path.open("a", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, fieldnames=HEADERS).writerow(row)
