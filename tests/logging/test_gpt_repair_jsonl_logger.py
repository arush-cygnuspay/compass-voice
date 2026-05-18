# tests/logging/test_gpt_repair_jsonl_logger.py
"""Tests for GptRepairJsonlLogger — writing, PII sanitization, rotation."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from app.logging.gpt_repair_jsonl_logger import (
    GptRepairJsonlLogger,
    sanitize_string,
    sanitize_record,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_logger(tmp_path: Path, **kwargs) -> GptRepairJsonlLogger:
    return GptRepairJsonlLogger(
        log_path=tmp_path / "gpt_repair_turns.jsonl",
        **kwargs,
    )


def _read_records(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


# ---------------------------------------------------------------------------
# PII sanitization unit tests
# ---------------------------------------------------------------------------


class TestSanitizeString:
    def test_plain_text_unchanged(self):
        assert sanitize_string("hello world") == "hello world"

    def test_payment_link_redacted(self):
        result = sanitize_string("visit https://pay.example.com/link123 to pay")
        assert "pay.example.com" not in result
        assert "[REDACTED_URL]" in result

    def test_checkout_link_redacted(self):
        result = sanitize_string("go to https://checkout.stripe.com/c/pay/abc")
        assert "checkout.stripe.com" not in result
        assert "[REDACTED_URL]" in result

    def test_email_redacted(self):
        result = sanitize_string("contact admin@example.com for help")
        assert "admin@example.com" not in result
        assert "[REDACTED_EMAIL]" in result

    def test_phone_number_redacted(self):
        result = sanitize_string("call 555-123-4567 now")
        assert "555" not in result or "4567" not in result or "[REDACTED_PHONE]" in result

    def test_long_digit_sequence_redacted(self):
        result = sanitize_string("code 12345678 is valid")
        assert "[REDACTED_PHONE]" in result

    def test_api_key_redacted(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secretkey123")
        result = sanitize_string("key is sk-test-secretkey123 done")
        assert "sk-test-secretkey123" not in result
        assert "[REDACTED_KEY]" in result

    def test_short_key_not_redacted(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "short")
        # Short key (<8 chars) should not trigger redaction pattern build
        result = sanitize_string("short text")
        assert result == "short text"


class TestSanitizeRecord:
    def test_dict_sanitized_recursively(self):
        record = {
            "user": "user@example.com",
            "nested": {"link": "https://checkout.example.com/pay"},
        }
        clean = sanitize_record(record)
        assert "user@example.com" not in clean["user"]
        assert "checkout.example.com" not in clean["nested"]["link"]

    def test_list_sanitized(self):
        items = ["hello", "admin@example.com"]
        clean = sanitize_record(items)
        assert clean[0] == "hello"
        assert "admin@example.com" not in clean[1]

    def test_non_string_passthrough(self):
        assert sanitize_record(42) == 42
        assert sanitize_record(3.14) == 3.14
        assert sanitize_record(None) is None
        assert sanitize_record(True) is True


# ---------------------------------------------------------------------------
# Logger write tests
# ---------------------------------------------------------------------------


class TestGptRepairJsonlLoggerWrite:
    def test_single_record_written(self, tmp_path):
        logger = _make_logger(tmp_path)
        record = {"session_id": "s1", "turn_index": 0, "gpt": {"called": True}}
        logger.log_turn(record)
        logger.flush()

        records = _read_records(tmp_path / "gpt_repair_turns.jsonl")
        assert len(records) == 1
        assert records[0]["session_id"] == "s1"

    def test_multiple_records_appended(self, tmp_path):
        logger = _make_logger(tmp_path)
        for i in range(5):
            logger.log_turn({"turn_index": i})
        logger.flush()

        records = _read_records(tmp_path / "gpt_repair_turns.jsonl")
        assert len(records) == 5
        assert [r["turn_index"] for r in records] == list(range(5))

    def test_record_is_valid_json(self, tmp_path):
        logger = _make_logger(tmp_path)
        logger.log_turn({"key": "value", "num": 3.14, "flag": True})
        logger.flush()

        content = (tmp_path / "gpt_repair_turns.jsonl").read_text()
        loaded = json.loads(content.strip())
        assert loaded["key"] == "value"

    def test_pii_sanitized_before_write(self, tmp_path):
        logger = _make_logger(tmp_path)
        logger.log_turn({"user_text": "email is test@example.com"})
        logger.flush()

        records = _read_records(tmp_path / "gpt_repair_turns.jsonl")
        assert "test@example.com" not in records[0]["user_text"]
        assert "[REDACTED_EMAIL]" in records[0]["user_text"]

    def test_write_failure_does_not_raise(self, tmp_path):
        logger = _make_logger(tmp_path)
        # Point to a directory (unwritable as a file)
        bad_path = tmp_path / "subdir"
        bad_path.mkdir()
        logger.log_path = bad_path  # will fail silently on write

        # Should not raise
        logger.log_turn({"session_id": "s1"})
        logger.flush()

    def test_log_path_created_if_missing(self, tmp_path):
        nested = tmp_path / "deep" / "logs"
        logger = GptRepairJsonlLogger(log_path=nested / "gpt.jsonl")
        logger.log_turn({"turn_index": 0})
        logger.flush()
        assert (nested / "gpt.jsonl").exists()
        logger.shutdown()


# ---------------------------------------------------------------------------
# Rotation tests
# ---------------------------------------------------------------------------


class TestGptRepairJsonlLoggerRotation:
    def test_rotation_moves_existing_file(self, tmp_path):
        log_path = tmp_path / "gpt_repair_turns.jsonl"
        log_path.write_text('{"old": true}\n', encoding="utf-8")

        logger = GptRepairJsonlLogger(log_path=log_path, rotate_on_start=True)
        logger.log_turn({"new": True})
        logger.flush()
        logger.shutdown()

        # Original file should now be a fresh file (new record only)
        records = _read_records(log_path)
        assert records == [{"new": True}]

        # Old record should be in older/
        older_dir = tmp_path / "older"
        assert older_dir.exists()
        archived = list(older_dir.glob("*.jsonl"))
        assert len(archived) == 1
        archived_content = archived[0].read_text(encoding="utf-8").strip()
        assert archived_content == '{"old": true}'

    def test_no_rotation_when_flag_false(self, tmp_path):
        log_path = tmp_path / "gpt_repair_turns.jsonl"
        log_path.write_text('{"old": true}\n', encoding="utf-8")

        logger = GptRepairJsonlLogger(log_path=log_path, rotate_on_start=False)
        logger.log_turn({"new": True})
        logger.flush()
        logger.shutdown()

        records = _read_records(log_path)
        assert len(records) == 2

    def test_rotation_when_file_absent_is_safe(self, tmp_path):
        log_path = tmp_path / "gpt_repair_turns.jsonl"
        assert not log_path.exists()
        # Should not raise
        logger = GptRepairJsonlLogger(log_path=log_path, rotate_on_start=True)
        logger.log_turn({"turn_index": 0})
        logger.flush()
        logger.shutdown()
        assert log_path.exists()


# ---------------------------------------------------------------------------
# Shutdown test
# ---------------------------------------------------------------------------


class TestGptRepairJsonlLoggerShutdown:
    def test_shutdown_flushes_pending(self, tmp_path):
        logger = _make_logger(tmp_path)
        for i in range(10):
            logger.log_turn({"turn_index": i})
        logger.shutdown()

        records = _read_records(tmp_path / "gpt_repair_turns.jsonl")
        assert len(records) == 10
