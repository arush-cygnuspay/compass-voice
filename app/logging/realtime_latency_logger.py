# app/logging/realtime_latency_logger.py
from __future__ import annotations

import csv
import json
import os
import queue
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ms_between(start: float | None, end: float | None) -> float | None:
    if start is None or end is None:
        return None
    return round((end - start) * 1000.0, 3)


def _safe_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def _stringify_list(values: list[Any] | tuple[Any, ...] | None) -> str:
    if not values:
        return ""
    return ", ".join(str(v) for v in values if v is not None and str(v).strip())


@dataclass(slots=True)
class RealtimeTurnTrace:
    call_sid: str = ""
    stream_sid: str = ""
    session_id: str = ""
    turn_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    turn_index: int = 0

    turn_started_at_utc: str = ""
    turn_committed_at_utc: str = ""
    response_first_audio_sent_at_utc: str = ""
    playback_completed_at_utc: str = ""

    first_inbound_media_monotonic: float | None = None
    last_inbound_media_monotonic: float | None = None
    dg_speech_started_monotonic: float | None = None
    dg_final_transcript_monotonic: float | None = None
    turn_commit_monotonic: float | None = None
    engine_start_monotonic: float | None = None
    engine_end_monotonic: float | None = None
    responder_start_monotonic: float | None = None
    responder_end_monotonic: float | None = None
    tts_request_start_monotonic: float | None = None
    tts_first_chunk_monotonic: float | None = None
    tts_last_chunk_monotonic: float | None = None
    twilio_first_outbound_media_monotonic: float | None = None
    twilio_last_outbound_media_monotonic: float | None = None
    twilio_mark_sent_monotonic: float | None = None
    twilio_mark_received_monotonic: float | None = None

    state_before: str = ""
    state_after: str = ""
    pending_action: str = ""
    current_prompt_field: str = ""
    current_item_id: str = ""
    current_item_name: str = ""
    user_text: str = ""
    cleaned_text: str = ""
    normalized_text: str = ""
    response_text: str = ""
    spoken_response_text: str = ""
    response_key: str = ""

    pred_main_intent: str = ""
    pred_sub_intent: str = ""
    pred_intent: str = ""
    pred_intent_confidence: float | None = None

    slot_names: list[str] = field(default_factory=list)
    slot_values: list[str] = field(default_factory=list)

    inbound_audio_bytes: int = 0
    outbound_audio_bytes: int = 0
    outbound_audio_duration_ms: float | None = None
    tts_text_chars: int = 0
    stt_final_text_chars: int = 0

    turn_total_ms: float | None = None
    preprocess_ms: float | None = None
    nlu_ms: float | None = None
    flow_ms: float | None = None
    route_ms: float | None = None
    handler_ms: float | None = None
    engine_total_ms: float | None = None

    intent_model_ms: float | None = None
    slot_model_ms: float | None = None

    command: dict[str, Any] | None = None
    notes: dict[str, Any] = field(default_factory=dict)

    # ── GPT shadow summary (flat; populated after GPT result is known) ────
    # In all_shadow mode these are set to partial values (pending_async) since
    # GPT runs in a background thread after the turn returns.
    # In eligible_only mode these reflect the actual GPT call outcome.
    gpt_called: bool = False
    gpt_decision: str = ""
    gpt_selected_intent: str = ""
    gpt_confidence: float | None = None
    gpt_total_ms: float | None = None
    gpt_timeout: bool = False
    gpt_applied: bool = False
    gpt_fallback_type: str = ""
    gpt_policy_mode: str = ""
    gpt_policy_reason: str = ""
    gpt_prompt_bucket: str = ""
    gpt_used_inline: bool = False
    gpt_used_shadow: bool = False
    gpt_timeout_ms: int | None = None
    gpt_result_applied: bool = False
    gpt_result_rejected: bool = False
    gpt_execution_policy_ms: float | None = None

    # ── ADD_ITEM extractor summary (read from notes["add_item"] after turn) ─
    # Populated only when mode=shadow and the extractor ran.
    add_item_extractor_called: bool = False
    add_item_decision: str = ""
    add_item_items_count: int | None = None
    add_item_confidence: float | None = None
    add_item_total_ms: float | None = None

    # ── ADD_ITEM validator summary (Phase 2 shadow) ───────────────────────
    add_item_validated_items_count: int | None = None
    add_item_has_blocking_warnings: bool = False
    add_item_validator_ms: float | None = None

    def finalize_metrics(self) -> dict[str, Any]:
        # Speech / commit
        latency_user_speech_start_to_stt_final_ms = ms_between(
            self.dg_speech_started_monotonic,
            self.dg_final_transcript_monotonic,
        )
        latency_first_inbound_audio_to_stt_final_ms = ms_between(
            self.first_inbound_media_monotonic,
            self.dg_final_transcript_monotonic,
        )
        latency_stt_final_to_turn_commit_ms = ms_between(
            self.dg_final_transcript_monotonic,
            self.turn_commit_monotonic,
        )
        latency_first_inbound_audio_to_turn_commit_ms = ms_between(
            self.first_inbound_media_monotonic,
            self.turn_commit_monotonic,
        )

        # Engine / response generation
        latency_engine_processing_ms = self.engine_total_ms
        if latency_engine_processing_ms is None:
            latency_engine_processing_ms = ms_between(
                self.engine_start_monotonic,
                self.engine_end_monotonic,
            )

        latency_response_render_ms = ms_between(
            self.responder_start_monotonic,
            self.responder_end_monotonic,
        )

        # TTS
        latency_tts_request_to_first_audio_chunk_ms = ms_between(
            self.tts_request_start_monotonic,
            self.tts_first_chunk_monotonic,
        )
        latency_tts_request_to_last_audio_chunk_ms = ms_between(
            self.tts_request_start_monotonic,
            self.tts_last_chunk_monotonic,
        )

        # First audio sent to Twilio
        latency_turn_commit_to_first_audio_sent_ms = ms_between(
            self.turn_commit_monotonic,
            self.twilio_first_outbound_media_monotonic,
        )
        latency_stt_final_to_first_audio_sent_ms = ms_between(
            self.dg_final_transcript_monotonic,
            self.twilio_first_outbound_media_monotonic,
        )
        latency_engine_end_to_first_audio_sent_ms = ms_between(
            self.engine_end_monotonic,
            self.twilio_first_outbound_media_monotonic,
        )
        latency_tts_first_chunk_to_first_audio_sent_ms = ms_between(
            self.tts_first_chunk_monotonic,
            self.twilio_first_outbound_media_monotonic,
        )

        # Outbound stream / playback
        latency_first_audio_sent_to_last_audio_sent_ms = ms_between(
            self.twilio_first_outbound_media_monotonic,
            self.twilio_last_outbound_media_monotonic,
        )
        latency_mark_sent_to_mark_received_ms = ms_between(
            self.twilio_mark_sent_monotonic,
            self.twilio_mark_received_monotonic,
        )
        latency_first_audio_sent_to_playback_ack_ms = ms_between(
            self.twilio_first_outbound_media_monotonic,
            self.twilio_mark_received_monotonic,
        )
        latency_first_inbound_audio_to_playback_ack_ms = ms_between(
            self.first_inbound_media_monotonic,
            self.twilio_mark_received_monotonic,
        )

        # Executive-facing timings
        # 1) Reply started
        latency_customer_wait_for_first_audio_ms = latency_stt_final_to_first_audio_sent_ms

        # 2) Full turn completed (preferred total turn latency for leadership)
        latency_customer_wait_until_playback_complete_ms = ms_between(
            self.dg_final_transcript_monotonic,
            self.twilio_mark_received_monotonic,
        )

        # 3) Backend+telephony completion after commit
        latency_turn_commit_to_playback_complete_ms = ms_between(
            self.turn_commit_monotonic,
            self.twilio_mark_received_monotonic,
        )

        # Twilio playback / transport estimation
        twilio_estimated_playback_ms = self.outbound_audio_duration_ms
        twilio_estimated_transport_and_ack_overhead_ms = None
        twilio_estimated_one_way_start_ms = None
        if (
            latency_first_audio_sent_to_playback_ack_ms is not None
            and twilio_estimated_playback_ms is not None
        ):
            twilio_estimated_transport_and_ack_overhead_ms = round(
                max(
                    latency_first_audio_sent_to_playback_ack_ms - twilio_estimated_playback_ms,
                    0.0,
                ),
                3,
            )
            twilio_estimated_one_way_start_ms = round(
                twilio_estimated_transport_and_ack_overhead_ms / 2.0,
                3,
            )

        # Residual bucket, not true network-only
        latency_estimated_unattributed_ms = None
        if latency_first_inbound_audio_to_playback_ack_ms is not None:
            subtract = 0.0
            for value in (
                latency_first_inbound_audio_to_turn_commit_ms,
                latency_engine_processing_ms,
                latency_response_render_ms,
                latency_tts_request_to_first_audio_chunk_ms,
                latency_first_audio_sent_to_last_audio_sent_ms,
                latency_mark_sent_to_mark_received_ms,
            ):
                if value is not None:
                    subtract += value

            latency_estimated_unattributed_ms = round(
                latency_first_inbound_audio_to_playback_ack_ms - subtract,
                3,
            )

        payload = asdict(self)
        payload.update(
            {
                "latency_user_speech_start_to_stt_final_ms": latency_user_speech_start_to_stt_final_ms,
                "latency_first_inbound_audio_to_stt_final_ms": latency_first_inbound_audio_to_stt_final_ms,
                "latency_stt_final_to_turn_commit_ms": latency_stt_final_to_turn_commit_ms,
                "latency_first_inbound_audio_to_turn_commit_ms": latency_first_inbound_audio_to_turn_commit_ms,
                "latency_engine_processing_ms": latency_engine_processing_ms,
                "latency_response_render_ms": latency_response_render_ms,
                "latency_tts_request_to_first_audio_chunk_ms": latency_tts_request_to_first_audio_chunk_ms,
                "latency_tts_request_to_last_audio_chunk_ms": latency_tts_request_to_last_audio_chunk_ms,
                "latency_turn_commit_to_first_audio_sent_ms": latency_turn_commit_to_first_audio_sent_ms,
                "latency_stt_final_to_first_audio_sent_ms": latency_stt_final_to_first_audio_sent_ms,
                "latency_engine_end_to_first_audio_sent_ms": latency_engine_end_to_first_audio_sent_ms,
                "latency_tts_first_chunk_to_first_audio_sent_ms": latency_tts_first_chunk_to_first_audio_sent_ms,
                "latency_first_audio_sent_to_last_audio_sent_ms": latency_first_audio_sent_to_last_audio_sent_ms,
                "latency_mark_sent_to_mark_received_ms": latency_mark_sent_to_mark_received_ms,
                "latency_first_audio_sent_to_playback_ack_ms": latency_first_audio_sent_to_playback_ack_ms,
                "latency_first_inbound_audio_to_playback_ack_ms": latency_first_inbound_audio_to_playback_ack_ms,
                "latency_customer_wait_for_first_audio_ms": latency_customer_wait_for_first_audio_ms,
                "latency_customer_wait_until_playback_complete_ms": latency_customer_wait_until_playback_complete_ms,
                "latency_turn_commit_to_playback_complete_ms": latency_turn_commit_to_playback_complete_ms,
                "twilio_estimated_playback_ms": twilio_estimated_playback_ms,
                "twilio_estimated_transport_and_ack_overhead_ms": twilio_estimated_transport_and_ack_overhead_ms,
                "twilio_estimated_one_way_start_ms": twilio_estimated_one_way_start_ms,
                "latency_estimated_unattributed_ms": latency_estimated_unattributed_ms,
                "slot_names_csv": _stringify_list(self.slot_names),
                "slot_values_csv": _stringify_list(self.slot_values),
            }
        )
        return payload


class RealtimeLatencyLogger:
    CSV_COLUMNS = [
        # Identifiers / timestamps
        "turn_index",
        "turn_id",
        "call_sid",
        "stream_sid",
        "session_id",
        "turn_started_at_utc",
        "turn_committed_at_utc",
        "response_first_audio_sent_at_utc",
        "playback_completed_at_utc",

        # User / bot / flow context
        "customer_said",
        "cleaned_text",
        "normalized_text",
        "bot_internal_response_text",
        "bot_spoken_response_text",
        "response_key",
        "state_before",
        "state_after",
        "pending_action",
        "current_prompt_field",
        "current_item_id",
        "current_item_name",
        "intent_main",
        "intent_sub_intent",
        "intent_effective",
        "intent_confidence",
        "slot_names",
        "slot_values",

        # Executive-facing
        "customer_wait_for_first_audio_ms",
        "customer_wait_until_playback_complete_ms",
        "turn_commit_to_playback_complete_ms",

        # Main breakdown
        "stt_final_to_turn_commit_ms",
        "engine_processing_ms",
        "response_render_ms",
        "tts_request_to_first_audio_chunk_ms",
        "turn_commit_to_first_audio_sent_ms",
        "stt_final_to_first_audio_sent_ms",
        "engine_end_to_first_audio_sent_ms",
        "tts_first_chunk_to_first_audio_sent_ms",
        "tts_request_to_last_audio_chunk_ms",
        "first_audio_sent_to_last_audio_sent_ms",
        "first_audio_sent_to_playback_ack_ms",
        "mark_sent_to_mark_received_ms",

        # Twilio estimate columns
        "twilio_estimated_playback_ms",
        "twilio_estimated_transport_and_ack_overhead_ms",
        "twilio_estimated_one_way_start_ms",

        # Supporting latency details
        "first_inbound_audio_to_playback_ack_ms",
        "first_inbound_audio_to_turn_commit_ms",
        "user_speech_start_to_stt_final_ms",
        "first_inbound_audio_to_stt_final_ms",
        "estimated_unattributed_ms",

        # Existing model timings
        "time_preprocess_ms",
        "time_nlu_ms",
        "time_flow_ms",
        "time_route_ms",
        "time_handler_ms",
        "time_intent_det_model_ms",
        "time_slot_model_ms",

        # Audio / payload details
        "inbound_audio_bytes",
        "outbound_audio_bytes",
        "outbound_audio_duration_ms",
        "tts_text_chars",
        "stt_final_text_chars",
        "command",
        "notes",
        # GPT shadow summary columns (appended; existing columns order unchanged)
        "gpt_called",
        "gpt_decision",
        "gpt_selected_intent",
        "gpt_confidence",
        "gpt_total_ms",
        "gpt_timeout",
        "gpt_applied",
        "gpt_fallback_type",
        "gpt_policy_mode",
        "gpt_policy_reason",
        "gpt_prompt_bucket",
        "gpt_used_inline",
        "gpt_used_shadow",
        "gpt_timeout_ms",
        "gpt_result_applied",
        "gpt_result_rejected",
        "gpt_execution_policy_ms",
        # ADD_ITEM extractor summary columns (appended after GPT columns)
        "add_item_extractor_called",
        "add_item_decision",
        "add_item_items_count",
        "add_item_confidence",
        "add_item_total_ms",
        # ADD_ITEM validator summary columns (Phase 2 shadow)
        "add_item_validated_items_count",
        "add_item_has_blocking_warnings",
        "add_item_validator_ms",
    ]

    def __init__(
        self,
        file_path: str | None = None,
        enabled: bool = True,
        csv_file_path: str | None = None,
        write_csv: bool = True,
        queue_maxsize: int | None = None,
        sync_write_immediately: bool | None = None,
        fsync_on_write: bool | None = None,
    ) -> None:
        self.enabled = enabled
        self.write_csv = write_csv

        self.file_path = Path(file_path or "app/logs/realtime_turn_latency.jsonl")
        self.csv_file_path = Path(csv_file_path or "app/logs/realtime_turn_latency.csv")

        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_file_path.parent.mkdir(parents=True, exist_ok=True)

        self.sync_write_immediately = (
            sync_write_immediately
            if sync_write_immediately is not None
            else os.getenv("COMPASS_REALTIME_LATENCY_SYNC_WRITE", "1") == "1"
        )
        self.fsync_on_write = (
            fsync_on_write
            if fsync_on_write is not None
            else os.getenv("COMPASS_REALTIME_LATENCY_FSYNC", "1") == "1"
        )

        self.queue_maxsize = queue_maxsize or int(
            os.getenv("COMPASS_REALTIME_LATENCY_QUEUE_MAXSIZE", "5000")
        )
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=self.queue_maxsize)
        self._stop_event = threading.Event()
        self._writer_thread: threading.Thread | None = None
        self._writer_lock = threading.Lock()
        self._dropped_logs = 0

        if self.enabled and not self.sync_write_immediately:
            self._start_writer()

    def _start_writer(self) -> None:
        if self._writer_thread is not None:
            return

        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name="realtime-latency-writer",
            daemon=True,
        )
        self._writer_thread.start()

    def _writer_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                payload = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                self._write_payload(payload)
            except Exception as exc:
                print(f"[REALTIME_LATENCY_WRITE_ERROR] {type(exc).__name__}: {exc}")
            finally:
                self._queue.task_done()

    def flush(self) -> None:
        if not self.enabled:
            return
        if self.sync_write_immediately:
            return
        self._queue.join()

    def shutdown(self) -> None:
        if not self.enabled:
            return
        self.flush()
        self._stop_event.set()
        if self._writer_thread is not None:
            self._writer_thread.join(timeout=1.0)

    def write(self, trace: RealtimeTurnTrace) -> None:
        if not self.enabled:
            return

        payload = trace.finalize_metrics()

        if self.sync_write_immediately:
            try:
                self._write_payload(payload)
            except Exception as exc:
                print(f"[REALTIME_LATENCY_WRITE_ERROR] {type(exc).__name__}: {exc}")
            return

        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            self._dropped_logs += 1
            if self._dropped_logs % 100 == 1:
                print(
                    "[REALTIME_LATENCY_QUEUE_FULL]",
                    {"dropped_logs": self._dropped_logs},
                )

    def _write_payload(self, payload: dict[str, Any]) -> None:
        with self._writer_lock:
            self._write_jsonl(payload)
            if self.write_csv:
                self._write_csv_row(payload)

    def _write_jsonl(self, payload: dict[str, Any]) -> None:
        line = json.dumps(payload, ensure_ascii=False)
        with self.file_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            if self.fsync_on_write:
                os.fsync(f.fileno())

    def _header_matches(self) -> bool:
        if not self.csv_file_path.exists() or self.csv_file_path.stat().st_size == 0:
            return False

        try:
            with self.csv_file_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                first_row = next(reader, None)
                return first_row == self.CSV_COLUMNS
        except Exception:
            return False

    def _write_csv_header_only(self) -> None:
        with self.csv_file_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=self.CSV_COLUMNS,
                extrasaction="ignore",
            )
            writer.writeheader()
            f.flush()
            if self.fsync_on_write:
                os.fsync(f.fileno())

    def _build_csv_row(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "turn_index": payload.get("turn_index", ""),
            "turn_id": payload.get("turn_id", ""),
            "call_sid": payload.get("call_sid", ""),
            "stream_sid": payload.get("stream_sid", ""),
            "session_id": payload.get("session_id", ""),
            "turn_started_at_utc": payload.get("turn_started_at_utc", ""),
            "turn_committed_at_utc": payload.get("turn_committed_at_utc", ""),
            "response_first_audio_sent_at_utc": payload.get("response_first_audio_sent_at_utc", ""),
            "playback_completed_at_utc": payload.get("playback_completed_at_utc", ""),

            "customer_said": payload.get("user_text", ""),
            "cleaned_text": payload.get("cleaned_text", ""),
            "normalized_text": payload.get("normalized_text", ""),
            "bot_internal_response_text": payload.get("response_text", ""),
            "bot_spoken_response_text": payload.get("spoken_response_text", ""),
            "response_key": payload.get("response_key", ""),
            "state_before": payload.get("state_before", ""),
            "state_after": payload.get("state_after", ""),
            "pending_action": payload.get("pending_action", ""),
            "current_prompt_field": payload.get("current_prompt_field", ""),
            "current_item_id": payload.get("current_item_id", ""),
            "current_item_name": payload.get("current_item_name", ""),
            "intent_main": payload.get("pred_main_intent", ""),
            "intent_sub_intent": payload.get("pred_sub_intent", ""),
            "intent_effective": payload.get("pred_intent", ""),
            "intent_confidence": payload.get("pred_intent_confidence", ""),
            "slot_names": payload.get("slot_names_csv", ""),
            "slot_values": payload.get("slot_values_csv", ""),

            "customer_wait_for_first_audio_ms": payload.get("latency_customer_wait_for_first_audio_ms", ""),
            "customer_wait_until_playback_complete_ms": payload.get("latency_customer_wait_until_playback_complete_ms", ""),
            "turn_commit_to_playback_complete_ms": payload.get("latency_turn_commit_to_playback_complete_ms", ""),

            "stt_final_to_turn_commit_ms": payload.get("latency_stt_final_to_turn_commit_ms", ""),
            "engine_processing_ms": payload.get("latency_engine_processing_ms", ""),
            "response_render_ms": payload.get("latency_response_render_ms", ""),
            "tts_request_to_first_audio_chunk_ms": payload.get("latency_tts_request_to_first_audio_chunk_ms", ""),
            "turn_commit_to_first_audio_sent_ms": payload.get("latency_turn_commit_to_first_audio_sent_ms", ""),
            "stt_final_to_first_audio_sent_ms": payload.get("latency_stt_final_to_first_audio_sent_ms", ""),
            "engine_end_to_first_audio_sent_ms": payload.get("latency_engine_end_to_first_audio_sent_ms", ""),
            "tts_first_chunk_to_first_audio_sent_ms": payload.get("latency_tts_first_chunk_to_first_audio_sent_ms", ""),
            "tts_request_to_last_audio_chunk_ms": payload.get("latency_tts_request_to_last_audio_chunk_ms", ""),
            "first_audio_sent_to_last_audio_sent_ms": payload.get("latency_first_audio_sent_to_last_audio_sent_ms", ""),
            "first_audio_sent_to_playback_ack_ms": payload.get("latency_first_audio_sent_to_playback_ack_ms", ""),
            "mark_sent_to_mark_received_ms": payload.get("latency_mark_sent_to_mark_received_ms", ""),

            "twilio_estimated_playback_ms": payload.get("twilio_estimated_playback_ms", ""),
            "twilio_estimated_transport_and_ack_overhead_ms": payload.get("twilio_estimated_transport_and_ack_overhead_ms", ""),
            "twilio_estimated_one_way_start_ms": payload.get("twilio_estimated_one_way_start_ms", ""),

            "first_inbound_audio_to_playback_ack_ms": payload.get("latency_first_inbound_audio_to_playback_ack_ms", ""),
            "first_inbound_audio_to_turn_commit_ms": payload.get("latency_first_inbound_audio_to_turn_commit_ms", ""),
            "user_speech_start_to_stt_final_ms": payload.get("latency_user_speech_start_to_stt_final_ms", ""),
            "first_inbound_audio_to_stt_final_ms": payload.get("latency_first_inbound_audio_to_stt_final_ms", ""),
            "estimated_unattributed_ms": payload.get("latency_estimated_unattributed_ms", ""),

            "time_preprocess_ms": payload.get("preprocess_ms", ""),
            "time_nlu_ms": payload.get("nlu_ms", ""),
            "time_flow_ms": payload.get("flow_ms", ""),
            "time_route_ms": payload.get("route_ms", ""),
            "time_handler_ms": payload.get("handler_ms", ""),
            "time_intent_det_model_ms": payload.get("intent_model_ms", ""),
            "time_slot_model_ms": payload.get("slot_model_ms", ""),

            "inbound_audio_bytes": payload.get("inbound_audio_bytes", ""),
            "outbound_audio_bytes": payload.get("outbound_audio_bytes", ""),
            "outbound_audio_duration_ms": payload.get("outbound_audio_duration_ms", ""),
            "tts_text_chars": payload.get("tts_text_chars", ""),
            "stt_final_text_chars": payload.get("stt_final_text_chars", ""),
            "command": _safe_json(payload.get("command")),
            "notes": _safe_json(payload.get("notes")),
            # GPT shadow summary
            "gpt_called": payload.get("gpt_called", ""),
            "gpt_decision": payload.get("gpt_decision", ""),
            "gpt_selected_intent": payload.get("gpt_selected_intent", ""),
            "gpt_confidence": payload.get("gpt_confidence", ""),
            "gpt_total_ms": payload.get("gpt_total_ms", ""),
            "gpt_timeout": payload.get("gpt_timeout", ""),
            "gpt_applied": payload.get("gpt_applied", ""),
            "gpt_fallback_type": payload.get("gpt_fallback_type", ""),
            "gpt_policy_mode": payload.get("gpt_policy_mode", ""),
            "gpt_policy_reason": payload.get("gpt_policy_reason", ""),
            "gpt_prompt_bucket": payload.get("gpt_prompt_bucket", ""),
            "gpt_used_inline": payload.get("gpt_used_inline", ""),
            "gpt_used_shadow": payload.get("gpt_used_shadow", ""),
            "gpt_timeout_ms": payload.get("gpt_timeout_ms", ""),
            "gpt_result_applied": payload.get("gpt_result_applied", ""),
            "gpt_result_rejected": payload.get("gpt_result_rejected", ""),
            "gpt_execution_policy_ms": payload.get("gpt_execution_policy_ms", ""),
            # ADD_ITEM extractor summary (read from notes["add_item"])
            "add_item_extractor_called": (
                payload.get("notes", {}).get("add_item", {}).get("add_item_extractor_called", "")
            ),
            "add_item_decision": (
                payload.get("notes", {}).get("add_item", {}).get("add_item_decision", "")
            ),
            "add_item_items_count": (
                payload.get("notes", {}).get("add_item", {}).get("add_item_items_count", "")
            ),
            "add_item_confidence": (
                payload.get("notes", {}).get("add_item", {}).get("add_item_confidence", "")
            ),
            "add_item_total_ms": (
                payload.get("notes", {}).get("add_item", {}).get("add_item_total_ms", "")
            ),
            # ADD_ITEM validator summary (Phase 2 shadow)
            "add_item_validated_items_count": (
                payload.get("notes", {}).get("add_item", {}).get("add_item_validated_items_count", "")
            ),
            "add_item_has_blocking_warnings": (
                payload.get("notes", {}).get("add_item", {}).get("add_item_has_blocking_warnings", "")
            ),
            "add_item_validator_ms": (
                payload.get("notes", {}).get("add_item", {}).get("add_item_validator_ms", "")
            ),
        }

    def _ensure_csv_header(self) -> None:
        if self._header_matches():
            return

        backup_path = self.csv_file_path.with_suffix(".csv.bak")
        if self.csv_file_path.exists() and self.csv_file_path.stat().st_size > 0:
            try:
                if backup_path.exists():
                    backup_path.unlink()
            except Exception:
                pass

            try:
                self.csv_file_path.replace(backup_path)
            except Exception:
                pass

        self._write_csv_header_only()

    def _write_csv_row(self, payload: dict[str, Any]) -> None:
        row = self._build_csv_row(payload)
        self._ensure_csv_header()

        with self.csv_file_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=self.CSV_COLUMNS,
                extrasaction="ignore",
            )
            writer.writerow(row)
            f.flush()
            if self.fsync_on_write:
                os.fsync(f.fileno())