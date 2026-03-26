# app/logging/nlu_csv_logger.py

from __future__ import annotations

import csv
import os
import threading
from dataclasses import dataclass, asdict
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


@dataclass(frozen=True, slots=True)
class TurnLogRow:
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
    response_key: str
    command_type: str
    command_summary: str
    outcome: str
    notes: str


@dataclass(frozen=True, slots=True)
class SlotLogRow:
    timestamp_utc: str
    date_utc: str
    session_id: str
    turn_index: int
    state_before: str
    pending_action: str
    user_text: str
    normalized_text: str
    pred_main_intent: str
    pred_sub_intent: str
    pred_intent: str
    slot_name: str
    slot_value: str
    slot_raw: str
    start: str
    end: str
    slot_confidence: str
    source: str
    resolution_status: str
    resolved_entity_type: str
    resolved_entity_id: str
    resolved_entity_name: str


class NluCsvLogger:
    TURN_HEADERS = [
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
        "response_key",
        "command_type",
        "command_summary",
        "outcome",
        "notes",
    ]

    SLOT_HEADERS = [
        "timestamp_utc",
        "date_utc",
        "session_id",
        "turn_index",
        "state_before",
        "pending_action",
        "user_text",
        "normalized_text",
        "pred_main_intent",
        "pred_sub_intent",
        "pred_intent",
        "slot_name",
        "slot_value",
        "slot_raw",
        "start",
        "end",
        "slot_confidence",
        "source",
        "resolution_status",
        "resolved_entity_type",
        "resolved_entity_id",
        "resolved_entity_name",
    ]

    def __init__(
        self,
        *,
        enabled: bool | None = None,
        log_dir: str | None = None,
        turn_filename: str = "nlu_turn_log.csv",
        slot_filename: str = "nlu_slot_log.csv",
    ) -> None:
        env_enabled = os.getenv("COMPASS_NLU_CSV_LOGGER_ENABLED", "1") != "0"
        self.enabled = env_enabled if enabled is None else enabled

        base_dir = log_dir or os.getenv("COMPASS_NLU_CSV_LOG_DIR") or "app/logs/nlu"
        self.log_dir = Path(base_dir)
        self.turn_path = self.log_dir / turn_filename
        self.slot_path = self.log_dir / slot_filename
        self._lock = threading.Lock()

        if self.enabled:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._ensure_file(self.turn_path, self.TURN_HEADERS)
            self._ensure_file(self.slot_path, self.SLOT_HEADERS)

    def _ensure_file(self, path: Path, headers: list[str]) -> None:
        if path.exists() and path.stat().st_size > 0:
            return

        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()

    def _append_rows(
        self,
        path: Path,
        headers: list[str],
        rows: Iterable[dict[str, Any]],
    ) -> None:
        if not self.enabled:
            return

        with self._lock:
            self._ensure_file(path, headers)
            with path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                for row in rows:
                    writer.writerow(row)

    def _summarize_command(self, command: dict[str, Any] | None) -> tuple[str, str]:
        if not command:
            return "", ""

        cmd_type = _safe_str(command.get("type"))
        payload = command.get("payload") or {}

        item_id = _safe_str(payload.get("item_id"))
        quantity = _safe_str(payload.get("quantity"))
        variant_id = _safe_str(payload.get("variant_id"))
        sides = payload.get("sides") or {}
        side_variants = payload.get("side_variants") or {}
        modifiers = payload.get("modifiers") or {}

        summary = (
            f"item_id={item_id};"
            f"quantity={quantity};"
            f"variant_id={variant_id};"
            f"sides={sides};"
            f"side_variants={side_variants};"
            f"modifiers={modifiers}"
        )
        return cmd_type, summary

    def _derive_outcome(
        self,
        *,
        response_key: str | None,
        command: dict[str, Any] | None,
        slots_count: int,
    ) -> str:
        if command:
            return "success_command_emitted"

        if response_key in {
            "item_added_successfully",
            "order_completed",
            "payment_link_sent",
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
        }:
            return "reprompt"

        if response_key in {
            "confirm_item_ambiguous",
            "confirm_item_from_category",
            "confirm_cancel_current_item",
            "confirm_cancel_current_item_for_new_request",
            "flow_guard_confirm_cancel",
        }:
            return "clarification"

        if response_key in {
            "item_not_found",
            "intent_not_allowed",
            "item_context_missing",
            "confirmation_state_error",
        }:
            return "failure"

        if slots_count > 0:
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
        slots_count: int = 0,
        notes: str = "",
    ) -> None:
        if not self.enabled:
            return

        timestamp_utc, date_utc = _utc_now()
        command_type, command_summary = self._summarize_command(command)

        row = TurnLogRow(
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
            pred_intent_confidence="" if pred_intent_confidence is None else f"{float(pred_intent_confidence):.4f}",
            slot_model_ran="1" if slot_model_ran else "0",
            response_key=response_key,
            command_type=command_type,
            command_summary=command_summary,
            outcome=self._derive_outcome(
                response_key=response_key,
                command=command,
                slots_count=slots_count,
            ),
            notes=notes,
        )

        self._append_rows(self.turn_path, self.TURN_HEADERS, [asdict(row)])

    def log_slots(
        self,
        *,
        session_id: str,
        turn_index: int,
        state_before: str,
        pending_action: str,
        user_text: str,
        normalized_text: str,
        pred_main_intent: str,
        pred_sub_intent: str,
        pred_intent: str,
        slots: Iterable[Any],
        source: str = "unknown",
    ) -> None:
        if not self.enabled:
            return

        timestamp_utc, date_utc = _utc_now()
        rows: list[dict[str, Any]] = []

        for slot in slots:
            row = SlotLogRow(
                timestamp_utc=timestamp_utc,
                date_utc=date_utc,
                session_id=session_id,
                turn_index=turn_index,
                state_before=state_before,
                pending_action=pending_action,
                user_text=user_text,
                normalized_text=normalized_text,
                pred_main_intent=pred_main_intent,
                pred_sub_intent=pred_sub_intent,
                pred_intent=pred_intent,
                slot_name=_safe_str(getattr(slot, "name", "")),
                slot_value=_safe_str(getattr(slot, "value", "")),
                slot_raw=_safe_str(getattr(slot, "raw", "")),
                start=_safe_str(getattr(slot, "start", "")),
                end=_safe_str(getattr(slot, "end", "")),
                slot_confidence=_safe_str(getattr(slot, "confidence", "")),
                source=source,
                resolution_status="predicted",
                resolved_entity_type="",
                resolved_entity_id="",
                resolved_entity_name="",
            )
            rows.append(asdict(row))

        if rows:
            self._append_rows(self.slot_path, self.SLOT_HEADERS, rows)