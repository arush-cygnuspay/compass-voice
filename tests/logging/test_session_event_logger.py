from __future__ import annotations

import json
from pathlib import Path

from app.logging.session_event_logger import SessionEventLogger


def _read_lines(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_logger_writes_event_to_jsonl(tmp_path: Path) -> None:
    logger = SessionEventLogger(enabled=True, log_dir=str(tmp_path))

    logger.log(
        event_name="payment_link_sent",
        session_id="sess-1",
        state="WAITING_FOR_PAYMENT",
        order_id="ORD-42",
        call_sid="CA-call",
        stream_sid="MZ-stream",
        metadata={"link": "https://checkout.example/abc"},
    )

    rows = _read_lines(logger.log_path)
    assert len(rows) == 1

    row = rows[0]
    assert row["event_name"] == "payment_link_sent"
    assert row["session_id"] == "sess-1"
    assert row["current_fsm_state"] == "WAITING_FOR_PAYMENT"
    assert row["order_id"] == "ORD-42"
    assert row["call_sid"] == "CA-call"
    assert row["stream_sid"] == "MZ-stream"
    assert row["metadata"] == {"link": "https://checkout.example/abc"}
    assert row["timestamp_utc"]


def test_logger_appends_multiple_events(tmp_path: Path) -> None:
    logger = SessionEventLogger(enabled=True, log_dir=str(tmp_path))

    logger.log(event_name="checkout_link_sent", session_id="s", state="X")
    logger.log(event_name="payment_confirmed", session_id="s", state="Y", order_id="ORD-1")

    rows = _read_lines(logger.log_path)
    assert [row["event_name"] for row in rows] == ["checkout_link_sent", "payment_confirmed"]
    assert rows[0]["order_id"] is None
    assert rows[1]["order_id"] == "ORD-1"


def test_logger_disabled_skips_write(tmp_path: Path) -> None:
    logger = SessionEventLogger(enabled=False, log_dir=str(tmp_path))

    logger.log(event_name="payment_link_sent", session_id="s", state="X")

    # Disabled logger must not even create the directory or file.
    assert not logger.log_path.exists()


def test_logger_swallows_disk_errors(tmp_path: Path, monkeypatch) -> None:
    logger = SessionEventLogger(enabled=True, log_dir=str(tmp_path))

    def _raise(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(Path, "open", _raise)

    # Must not propagate — logger is on a non-critical path.
    logger.log(event_name="payment_link_sent", session_id="s", state="X")
