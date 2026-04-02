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
    user_text: str = ""
    cleaned_text: str = ""
    normalized_text: str = ""
    response_text: str = ""
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

    def finalize_metrics(self) -> dict[str, Any]:
        utterance_capture_ms = ms_between(
            self.first_inbound_media_monotonic,
            self.turn_commit_monotonic,
        )
        stt_to_commit_ms = ms_between(
            self.dg_final_transcript_monotonic,
            self.turn_commit_monotonic,
        )
        responder_ms = ms_between(
            self.responder_start_monotonic,
            self.responder_end_monotonic,
        )
        tts_first_byte_ms = ms_between(
            self.tts_request_start_monotonic,
            self.tts_first_chunk_monotonic,
        )
        tts_full_stream_ms = ms_between(
            self.tts_request_start_monotonic,
            self.tts_last_chunk_monotonic,
        )
        twilio_outbound_stream_ms = ms_between(
            self.twilio_first_outbound_media_monotonic,
            self.twilio_last_outbound_media_monotonic,
        )
        playback_ack_ms = ms_between(
            self.twilio_mark_sent_monotonic,
            self.twilio_mark_received_monotonic,
        )
        backend_observed_e2e_ms = ms_between(
            self.first_inbound_media_monotonic,
            self.twilio_mark_received_monotonic,
        )

        engine_ms = self.engine_total_ms
        if engine_ms is None:
            engine_ms = ms_between(
                self.engine_start_monotonic,
                self.engine_end_monotonic,
            )

        stt_ms = None
        if (
            self.dg_speech_started_monotonic is not None
            and self.dg_final_transcript_monotonic is not None
        ):
            stt_ms = ms_between(
                self.dg_speech_started_monotonic,
                self.dg_final_transcript_monotonic,
            )
        elif (
            self.first_inbound_media_monotonic is not None
            and self.dg_final_transcript_monotonic is not None
        ):
            stt_ms = ms_between(
                self.first_inbound_media_monotonic,
                self.dg_final_transcript_monotonic,
            )

        estimated_non_audio_overhead_ms = None
        if backend_observed_e2e_ms is not None:
            subtract = 0.0
            for value in (
                utterance_capture_ms,
                engine_ms,
                responder_ms,
                tts_first_byte_ms,
                twilio_outbound_stream_ms,
                playback_ack_ms,
            ):
                if value is not None:
                    subtract += value
            estimated_non_audio_overhead_ms = round(
                backend_observed_e2e_ms - subtract,
                3,
            )

        payload = asdict(self)
        payload.update(
            {
                "utterance_capture_ms": utterance_capture_ms,
                "stt_to_commit_ms": stt_to_commit_ms,
                "engine_ms": engine_ms,
                "responder_ms": responder_ms,
                "tts_first_byte_ms": tts_first_byte_ms,
                "tts_full_stream_ms": tts_full_stream_ms,
                "twilio_outbound_stream_ms": twilio_outbound_stream_ms,
                "playback_ack_ms": playback_ack_ms,
                "backend_observed_e2e_ms": backend_observed_e2e_ms,
                "estimated_non_audio_overhead_ms": estimated_non_audio_overhead_ms,
                "stt_ms": stt_ms,
                "time_network_ms": estimated_non_audio_overhead_ms,
                "time_tts_ms": tts_first_byte_ms,
                "time_tts_stream_ms": tts_full_stream_ms,
                "time_total_ms": backend_observed_e2e_ms,
                "intent_sub_intent": self.pred_sub_intent,
                "slot_names_csv": _stringify_list(self.slot_names),
                "slot_values_csv": _stringify_list(self.slot_values),
            }
        )
        return payload


class RealtimeLatencyLogger:
    CSV_COLUMNS = [
        "turn_index",
        "turn_started_at_utc",
        "turn_committed_at_utc",
        "response_first_audio_sent_at_utc",
        "playback_completed_at_utc",
        "text",
        "normalized_text",
        "response_text",
        "response_key",
        "state_before",
        "state_after",
        "intent_main",
        "intent_sub_intent",
        "intent_effective",
        "intent_confidence",
        "time_total_ms",
        "time_stt_ms",
        "time_stt_to_commit_ms",
        "time_engine_ms",
        "time_responder_ms",
        "time_tts_ms",
        "time_tts_stream_ms",
        "time_twilio_outbound_stream_ms",
        "time_playback_ack_ms",
        "time_network_ms",
        "time_preprocess_ms",
        "time_nlu_ms",
        "time_flow_ms",
        "time_route_ms",
        "time_handler_ms",
        "time_intent_det_model_ms",
        "time_slot_model_ms",
        "inbound_audio_bytes",
        "outbound_audio_bytes",
        "outbound_audio_duration_ms",
        "tts_text_chars",
        "stt_final_text_chars",
        "slot_names",
        "slot_values",
        "command",
        "notes",
        "turn_id",
        "call_sid",
        "stream_sid",
        "session_id",
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
            "turn_started_at_utc": payload.get("turn_started_at_utc", ""),
            "turn_committed_at_utc": payload.get("turn_committed_at_utc", ""),
            "response_first_audio_sent_at_utc": payload.get("response_first_audio_sent_at_utc", ""),
            "playback_completed_at_utc": payload.get("playback_completed_at_utc", ""),
            "text": payload.get("user_text", ""),
            "normalized_text": payload.get("normalized_text", ""),
            "response_text": payload.get("response_text", ""),
            "response_key": payload.get("response_key", ""),
            "state_before": payload.get("state_before", ""),
            "state_after": payload.get("state_after", ""),
            "intent_main": payload.get("pred_main_intent", ""),
            "intent_sub_intent": payload.get("intent_sub_intent", ""),
            "intent_effective": payload.get("pred_intent", ""),
            "intent_confidence": payload.get("pred_intent_confidence", ""),
            "time_total_ms": payload.get("time_total_ms", ""),
            "time_stt_ms": payload.get("stt_ms", ""),
            "time_stt_to_commit_ms": payload.get("stt_to_commit_ms", ""),
            "time_engine_ms": payload.get("engine_ms", ""),
            "time_responder_ms": payload.get("responder_ms", ""),
            "time_tts_ms": payload.get("time_tts_ms", ""),
            "time_tts_stream_ms": payload.get("time_tts_stream_ms", ""),
            "time_twilio_outbound_stream_ms": payload.get("twilio_outbound_stream_ms", ""),
            "time_playback_ack_ms": payload.get("playback_ack_ms", ""),
            "time_network_ms": payload.get("time_network_ms", ""),
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
            "slot_names": payload.get("slot_names_csv", ""),
            "slot_values": payload.get("slot_values_csv", ""),
            "command": _safe_json(payload.get("command")),
            "notes": _safe_json(payload.get("notes")),
            "turn_id": payload.get("turn_id", ""),
            "call_sid": payload.get("call_sid", ""),
            "stream_sid": payload.get("stream_sid", ""),
            "session_id": payload.get("session_id", ""),
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