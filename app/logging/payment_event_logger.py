# app/logging/payment_event_logger.py
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class PaymentEventLogger:
    def __init__(
        self,
        *,
        enabled: bool | None = None,
        log_dir: str | None = None,
        filename: str = "payment_events.jsonl",
    ) -> None:
        env_enabled = os.getenv("COMPASS_PAYMENT_EVENT_LOGGER_ENABLED", "1") != "0"
        self.enabled = env_enabled if enabled is None else enabled
        base_dir = log_dir or os.getenv("COMPASS_PAYMENT_EVENT_LOG_DIR") or "app/logs/payment"
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
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return

        payload = {
            "timestamp_utc": _utc_now_iso(),
            "event_name": str(event_name or "").strip(),
            "session_id": str(session_id or "").strip(),
            "order_id": str(order_id or "").strip() or None,
            "current_fsm_state": str(state or "").strip(),
            "metadata": metadata or {},
        }

        try:
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        except Exception:
            return
