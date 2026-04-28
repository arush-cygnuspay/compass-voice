# app/logging/session_event_logger.py
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# Canonical event names emitted by the session event logger.  Keep in sync
# with downstream observability consumers — adding a new event name here is
# fine, renaming an existing one is a breaking change.
SESSION_EVENT_NAMES = frozenset({
    # Session lifecycle
    "session_started",
    "session_ended",
    # Twilio / playback layer
    "bot_speaking_started",
    "bot_speaking_finished",
    "playback_clear_sent",
    "twilio_mark_sent",
    "twilio_mark_received",
    "tts_empty_audio_attempt",
    "tts_empty_audio_exhausted",
    "tts_reconnected",
    "barge_in_guarded",
    "barge_in_committed",
    "outbound_audio_skipped_empty_frame",
    # Order / cart layer
    "cart_item_added",
    "cart_item_removed",
    "cart_item_replaced",
    "cart_item_quantity_changed",
    "post_confirmation_edit",
    # Checkout / payment layer
    "checkout_link_sent",
    "payment_link_sent",
    "payment_status_probe",
    "payment_confirmed",
    "payment_reminder_suppressed",
})


class SessionEventLogger:
    """Append-only structured logger for high-signal session events.

    Lifecycle and shape match :class:`PaymentEventLogger` so downstream
    pipelines can ingest both files with a single reader.  Writes are best
    effort (no exceptions propagated to the request path) — the logger is
    not on the latency-critical path of a turn.
    """

    def __init__(
        self,
        *,
        enabled: bool | None = None,
        log_dir: str | None = None,
        filename: str = "session_events.jsonl",
    ) -> None:
        env_enabled = os.getenv("COMPASS_SESSION_EVENT_LOGGER_ENABLED", "1") != "0"
        self.enabled = env_enabled if enabled is None else enabled
        base_dir = log_dir or os.getenv("COMPASS_SESSION_EVENT_LOG_DIR") or "app/logs/session"
        self.log_dir = Path(base_dir)
        self.log_path = self.log_dir / filename
        if self.enabled:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            if not self.log_path.exists():
                self.log_path.touch()

    def log(
        self,
        *,
        event_name: str,
        session_id: str,
        state: str,
        order_id: str | None = None,
        call_sid: str | None = None,
        stream_sid: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return

        payload = {
            "timestamp_utc": _utc_now_iso(),
            "event_name": str(event_name or "").strip(),
            "session_id": str(session_id or "").strip(),
            "call_sid": str(call_sid or "").strip() or None,
            "stream_sid": str(stream_sid or "").strip() or None,
            "order_id": str(order_id or "").strip() or None,
            "current_fsm_state": str(state or "").strip(),
            "metadata": metadata or {},
        }

        try:
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        except Exception:
            return
