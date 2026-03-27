# app/logging/nlu_csv_logger.py
from __future__ import annotations

import csv
import json
import os
import queue
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def _utc_now() -> tuple[str, str]:
    now = datetime.now(timezone.utc)
    return now.isoformat(), now.date().isoformat()


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _safe_json(value: Any) -> str:
    if value is None:
        return ""
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(value)


def _safe_float_str(value: float | int | None, precision: int = 4) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.{precision}f}"
    except (TypeError, ValueError):
        return ""


def _extract_slot_name(slot: Any) -> str:
    if slot is None:
        return ""

    name = getattr(slot, "name", None)
    if name:
        return str(name)

    label = getattr(slot, "label", None)
    if label:
        return str(label)

    if isinstance(slot, dict):
        return _safe_str(slot.get("name") or slot.get("label"))

    return ""


def _extract_slot_value(slot: Any) -> str:
    if slot is None:
        return ""

    value = getattr(slot, "value", None)
    if value is not None:
        return str(value)

    if isinstance(slot, dict):
        raw_value = slot.get("value")
        if raw_value is not None:
            return str(raw_value)

    return ""


def _join_non_empty(values: Iterable[str]) -> str:
    return ", ".join(value for value in values if value)


@dataclass(frozen=True, slots=True)
class NluLogRow:
    timestamp_utc: str
    date_utc: str
    session_id: str
    turn_index: int
    state_before: str
    state_after: str
    pending_action: str
    current_prompt_field: str
    current_item_id: str
    current_item_name: str
    user_text: str
    normalized_text: str
    pred_main_intent: str
    pred_sub_intent: str
    pred_intent: str
    pred_intent_confidence: str
    slot_model_ran: str
    slot_count: int
    slot_names: str
    slot_values: str
    response_key: str
    command_type: str
    command_summary: str
    outcome: str
    notes: str


class NluCsvLogger:
    HEADERS = [
        "timestamp_utc",
        "date_utc",
        "session_id",
        "turn_index",
        "state_before",
        "state_after",
        "pending_action",
        "current_prompt_field",
        "current_item_id",
        "current_item_name",
        "user_text",
        "normalized_text",
        "pred_main_intent",
        "pred_sub_intent",
        "pred_intent",
        "pred_intent_confidence",
        "slot_model_ran",
        "slot_count",
        "slot_names",
        "slot_values",
        "response_key",
        "command_type",
        "command_summary",
        "outcome",
        "notes",
    ]

    def __init__(
        self,
        *,
        enabled: bool | None = None,
        log_dir: str | None = None,
        filename: str = "nlu_log.csv",
        queue_maxsize: int | None = None,
    ) -> None:
        env_enabled = os.getenv("COMPASS_NLU_CSV_LOGGER_ENABLED", "1") != "0"
        self.enabled = env_enabled if enabled is None else enabled

        base_dir = log_dir or os.getenv("COMPASS_NLU_CSV_LOG_DIR") or "app/logs/nlu"
        self.log_dir = Path(base_dir)
        self.log_path = self.log_dir / filename

        self.queue_maxsize = queue_maxsize or int(
            os.getenv("COMPASS_NLU_CSV_LOGGER_QUEUE_MAXSIZE", "5000")
        )
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=self.queue_maxsize)
        self._stop_event = threading.Event()
        self._writer_lock = threading.Lock()
        self._writer_thread: threading.Thread | None = None
        self._dropped_logs = 0

        if self.enabled:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._ensure_file()
            self._start_writer()

    def _start_writer(self) -> None:
        if self._writer_thread is not None:
            return

        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name="nlu-csv-writer",
            daemon=True,
        )
        self._writer_thread.start()

    def _ensure_file(self) -> None:
        if self.log_path.exists() and self.log_path.stat().st_size > 0:
            return

        with self.log_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.HEADERS)
            writer.writeheader()

    def _writer_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                row = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                with self._writer_lock:
                    self._ensure_file()
                    with self.log_path.open("a", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=self.HEADERS)
                        writer.writerow(row)
            except Exception as exc:
                print(f"[NLU_CSV_LOGGER_WRITE_ERROR] {type(exc).__name__}: {exc}")
            finally:
                self._queue.task_done()

    def flush(self, timeout: float | None = None) -> None:
        if not self.enabled:
            return
        self._queue.join()

    def shutdown(self) -> None:
        if not self.enabled:
            return
        self.flush()
        self._stop_event.set()
        if self._writer_thread is not None:
            self._writer_thread.join(timeout=1.0)

    def _enqueue_row(self, row: dict[str, Any]) -> None:
        if not self.enabled:
            return

        try:
            self._queue.put_nowait(row)
        except queue.Full:
            self._dropped_logs += 1
            if self._dropped_logs % 100 == 1:
                print(
                    "[NLU_CSV_LOGGER_QUEUE_FULL]",
                    {"dropped_logs": self._dropped_logs},
                )

    def _summarize_command(self, command: dict[str, Any] | None) -> tuple[str, str]:
        if not command:
            return "", ""

        cmd_type = _safe_str(command.get("type"))
        payload = command.get("payload") or {}

        item_id = _safe_str(payload.get("item_id"))
        quantity = _safe_str(payload.get("quantity"))
        variant_id = _safe_str(payload.get("variant_id"))
        cart_item_id = _safe_str(payload.get("cart_item_id"))
        sides = payload.get("sides") or {}
        side_variants = payload.get("side_variants") or {}
        modifiers = payload.get("modifiers") or {}

        summary = (
            f"item_id={item_id};"
            f"cart_item_id={cart_item_id};"
            f"quantity={quantity};"
            f"variant_id={variant_id};"
            f"sides={_safe_json(sides)};"
            f"side_variants={_safe_json(side_variants)};"
            f"modifiers={_safe_json(modifiers)}"
        )
        return cmd_type, summary

    def _derive_outcome(
        self,
        *,
        response_key: str | None,
        command: dict[str, Any] | None,
        slot_count: int,
    ) -> str:
        if command:
            return "success_command_emitted"

        if response_key in {
            "item_added_successfully",
            "order_completed",
            "payment_link_sent",
            "confirm_order_summary",
        }:
            return "success"

        if response_key in {
            "repeat_side_options",
            "repeat_modifier_options",
            "repeat_size_options",
            "repeat_side_size_options",
            "invalid_size_option",
            "invalid_side_size_option",
            "invalid_quantity_option",
            "ask_for_side",
            "ask_for_modifier",
            "ask_for_size",
            "ask_for_quantity",
        }:
            return "reprompt"

        if response_key in {
            "confirm_item_ambiguous",
            "confirm_item_from_category",
            "confirm_cancel_current_item",
            "confirm_cancel_current_item_for_new_request",
            "flow_guard_confirm_cancel",
            "readonly_interrupt_with_resume",
        }:
            return "clarification"

        if response_key in {
            "item_not_found",
            "intent_not_allowed",
            "item_context_missing",
            "confirmation_state_error",
            "flow_blocked",
        }:
            return "failure"

        if slot_count > 0:
            return "slots_detected_no_command"

        return "observed"

    def log_turn(
        self,
        *,
        session_id: str,
        turn_index: int,
        state_before: str,
        state_after: str,
        pending_action: str,
        current_prompt_field: str,
        current_item_id: str,
        current_item_name: str,
        user_text: str,
        normalized_text: str,
        pred_main_intent: str,
        pred_sub_intent: str,
        pred_intent: str,
        pred_intent_confidence: float | int | None,
        slot_model_ran: bool,
        response_key: str,
        command: dict[str, Any] | None,
        slots: Iterable[Any] | None = None,
        notes: str = "",
    ) -> None:
        if not self.enabled:
            return

        timestamp_utc, date_utc = _utc_now()
        command_type, command_summary = self._summarize_command(command)

        slot_list = list(slots or [])
        slot_names = [_extract_slot_name(slot) for slot in slot_list]
        slot_values = [_extract_slot_value(slot) for slot in slot_list]
        slot_count = len([name for name in slot_names if name]) or len(slot_list)

        row = NluLogRow(
            timestamp_utc=timestamp_utc,
            date_utc=date_utc,
            session_id=session_id,
            turn_index=turn_index,
            state_before=state_before,
            state_after=state_after,
            pending_action=pending_action,
            current_prompt_field=current_prompt_field,
            current_item_id=current_item_id,
            current_item_name=current_item_name,
            user_text=user_text,
            normalized_text=normalized_text,
            pred_main_intent=pred_main_intent,
            pred_sub_intent=pred_sub_intent,
            pred_intent=pred_intent,
            pred_intent_confidence=_safe_float_str(pred_intent_confidence),
            slot_model_ran="1" if slot_model_ran else "0",
            slot_count=slot_count,
            slot_names=_join_non_empty(slot_names),
            slot_values=_join_non_empty(slot_values),
            response_key=response_key,
            command_type=command_type,
            command_summary=command_summary,
            outcome=self._derive_outcome(
                response_key=response_key,
                command=command,
                slot_count=slot_count,
            ),
            notes=notes,
        )

        self._enqueue_row(asdict(row))