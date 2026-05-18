# tests/logging/test_gpt_repair_csv_logger.py
"""Tests for GptRepairCsvLogger — writing, PII sanitization, rotation, headers."""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from app.logging.gpt_repair_csv_logger import (
    HEADERS,
    GptRepairCsvLogger,
    sanitize_record,
    sanitize_string,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_logger(tmp_path: Path, **kwargs) -> GptRepairCsvLogger:
    return GptRepairCsvLogger(log_path=tmp_path / "gpt_repair_turns.csv", **kwargs)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _minimal_row(**overrides) -> dict:
    row: dict = {h: "" for h in HEADERS}
    row.update(overrides)
    return row


# ---------------------------------------------------------------------------
# PII sanitization (sourced from CSV logger)
# ---------------------------------------------------------------------------


class TestSanitizeString:
    def test_plain_text_unchanged(self):
        assert sanitize_string("hello world") == "hello world"

    def test_payment_link_redacted(self):
        result = sanitize_string("visit https://pay.example.com/link to pay")
        assert "pay.example.com" not in result
        assert "[REDACTED_URL]" in result

    def test_checkout_link_redacted(self):
        result = sanitize_string("go to https://checkout.stripe.com/c/pay/abc")
        assert "checkout.stripe.com" not in result

    def test_email_redacted(self):
        result = sanitize_string("contact admin@example.com")
        assert "admin@example.com" not in result
        assert "[REDACTED_EMAIL]" in result

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
        assert sanitize_string("short text") == "short text"


class TestSanitizeRecord:
    def test_dict_sanitized_recursively(self):
        record = {"user": "user@example.com", "nested": {"link": "https://checkout.x.com/pay"}}
        clean = sanitize_record(record)
        assert "user@example.com" not in clean["user"]
        assert "checkout.x.com" not in clean["nested"]["link"]

    def test_non_string_passthrough(self):
        assert sanitize_record(42) == 42
        assert sanitize_record(None) is None
        assert sanitize_record(True) is True


# ---------------------------------------------------------------------------
# Header tests
# ---------------------------------------------------------------------------


class TestHeaders:
    def test_headers_count(self):
        # Phase 2 schema: 57 base + 6 ADD_ITEM validator columns = 63 total.
        assert len(HEADERS) == 63

    def test_required_columns_present(self):
        for col in (
            "timestamp_utc", "session_id", "turn_index",
            "user_text", "normalized_text",
            "gpt_repair_eligible", "gpt_called", "gpt_decision",
            "gpt_selected_intent", "training_candidate",
            "gpt_fallback_type", "fallback_response_key",
            # ADD_ITEM validator Phase 2 columns
            "add_item_validated_items_json", "add_item_validated_items_count",
            "add_item_has_blocking_warnings",
        ):
            assert col in HEADERS, f"Missing header: {col}"

    def test_no_duplicate_headers(self):
        assert len(HEADERS) == len(set(HEADERS))


# ---------------------------------------------------------------------------
# Write tests
# ---------------------------------------------------------------------------


class TestGptRepairCsvLoggerWrite:
    def test_header_row_written(self, tmp_path):
        logger = _make_logger(tmp_path)
        logger.shutdown()

        with (tmp_path / "gpt_repair_turns.csv").open(encoding="utf-8") as fh:
            reader = csv.reader(fh)
            written_headers = next(reader)
        assert written_headers == HEADERS

    def test_single_row_written(self, tmp_path):
        logger = _make_logger(tmp_path)
        logger.log_turn(_minimal_row(session_id="s1", turn_index=0, gpt_called=True))
        logger.flush()

        rows = _read_rows(tmp_path / "gpt_repair_turns.csv")
        assert len(rows) == 1
        assert rows[0]["session_id"] == "s1"
        assert rows[0]["gpt_called"] == "1"

    def test_multiple_rows_appended(self, tmp_path):
        logger = _make_logger(tmp_path)
        for i in range(5):
            logger.log_turn(_minimal_row(turn_index=i))
        logger.flush()

        rows = _read_rows(tmp_path / "gpt_repair_turns.csv")
        assert len(rows) == 5

    def test_bool_serialized_as_01(self, tmp_path):
        logger = _make_logger(tmp_path)
        logger.log_turn(_minimal_row(
            gpt_repair_eligible=True,
            gpt_called=False,
            training_candidate=True,
            gpt_timeout=False,
            gpt_applied=False,
        ))
        logger.flush()

        row = _read_rows(tmp_path / "gpt_repair_turns.csv")[0]
        assert row["gpt_repair_eligible"] == "1"
        assert row["gpt_called"] == "0"
        assert row["training_candidate"] == "1"
        assert row["gpt_timeout"] == "0"

    def test_float_serialized(self, tmp_path):
        logger = _make_logger(tmp_path)
        logger.log_turn(_minimal_row(gpt_total_ms=123.456789))
        logger.flush()

        row = _read_rows(tmp_path / "gpt_repair_turns.csv")[0]
        assert row["gpt_total_ms"].startswith("123.45")

    def test_none_values_empty_string(self, tmp_path):
        logger = _make_logger(tmp_path)
        logger.log_turn(_minimal_row(gpt_decision=None, gpt_model=None))
        logger.flush()

        row = _read_rows(tmp_path / "gpt_repair_turns.csv")[0]
        assert row["gpt_decision"] == ""
        assert row["gpt_model"] == ""

    def test_pii_sanitized_before_write(self, tmp_path):
        logger = _make_logger(tmp_path)
        logger.log_turn(_minimal_row(gpt_reason="email test@example.com found"))
        logger.flush()

        row = _read_rows(tmp_path / "gpt_repair_turns.csv")[0]
        assert "test@example.com" not in row["gpt_reason"]
        assert "[REDACTED_EMAIL]" in row["gpt_reason"]

    def test_log_path_created_if_missing(self, tmp_path):
        nested = tmp_path / "deep" / "logs"
        logger = GptRepairCsvLogger(log_path=nested / "gpt.csv")
        logger.log_turn(_minimal_row(turn_index=0))
        logger.flush()
        assert (nested / "gpt.csv").exists()
        logger.shutdown()

    def test_enqueue_does_not_raise(self, tmp_path):
        logger = _make_logger(tmp_path)
        # log_turn must not raise even with a malformed row
        logger.log_turn({"unexpected": object()})  # bad value type
        logger.flush()
        logger.shutdown()


# ---------------------------------------------------------------------------
# Rotation tests
# ---------------------------------------------------------------------------


class TestGptRepairCsvLoggerRotation:
    def test_rotation_archives_existing_file(self, tmp_path):
        log_path = tmp_path / "gpt_repair_turns.csv"
        # Write an old file with just a row (no header needed for this test)
        log_path.write_text("old_data\n", encoding="utf-8")

        logger = GptRepairCsvLogger(log_path=log_path, rotate_on_start=True)
        logger.log_turn(_minimal_row(session_id="new"))
        logger.flush()
        logger.shutdown()

        # New file should have header + 1 row
        rows = _read_rows(log_path)
        assert len(rows) == 1
        assert rows[0]["session_id"] == "new"

        # Old content archived
        older_dir = tmp_path / "older"
        assert older_dir.exists()
        archived = list(older_dir.glob("*.csv"))
        assert len(archived) == 1
        assert "old_data" in archived[0].read_text(encoding="utf-8")

    def test_no_rotation_when_flag_false(self, tmp_path):
        log_path = tmp_path / "gpt_repair_turns.csv"
        # Pre-existing file with header + 1 row
        with log_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=HEADERS)
            w.writeheader()
            w.writerow({h: "old" for h in HEADERS})

        logger = GptRepairCsvLogger(log_path=log_path, rotate_on_start=False)
        logger.log_turn(_minimal_row(session_id="new"))
        logger.flush()
        logger.shutdown()

        rows = _read_rows(log_path)
        # 2 data rows (old + new), no archival
        assert len(rows) == 2
        assert not (tmp_path / "older").exists()

    def test_rotation_when_file_absent_is_safe(self, tmp_path):
        log_path = tmp_path / "gpt_repair_turns.csv"
        logger = GptRepairCsvLogger(log_path=log_path, rotate_on_start=True)
        logger.log_turn(_minimal_row(turn_index=0))
        logger.flush()
        logger.shutdown()
        assert log_path.exists()


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------


class TestShutdown:
    def test_shutdown_flushes_all_pending(self, tmp_path):
        logger = _make_logger(tmp_path)
        for i in range(20):
            logger.log_turn(_minimal_row(turn_index=i))
        logger.shutdown()

        rows = _read_rows(tmp_path / "gpt_repair_turns.csv")
        assert len(rows) == 20


# ---------------------------------------------------------------------------
# CSV robustness — special characters in field values
# ---------------------------------------------------------------------------


class TestCsvRobustness:
    """Prove csv.DictReader can round-trip rows that contain commas, quotes, newlines.

    These tests guard against column corruption caused by unescaped special
    characters in user utterances and JSON fields.  csv.writer with quoting=QUOTE_ALL
    (or the equivalent) must handle all these cases without splitting a single
    logical row into multiple rows or shifting column alignment.
    """

    def _write_and_read(self, tmp_path: Path, **field_overrides) -> dict[str, str]:
        """Helper: log one row then read it back via csv.DictReader."""
        logger = _make_logger(tmp_path)
        logger.log_turn(_minimal_row(**field_overrides))
        logger.flush()
        logger.shutdown()
        rows = _read_rows(tmp_path / "gpt_repair_turns.csv")
        assert len(rows) == 1, f"Expected 1 data row, got {len(rows)}"
        return rows[0]

    def test_comma_in_user_text_does_not_shift_columns(self, tmp_path):
        """A comma inside user_text must not be treated as a column separator."""
        row = self._write_and_read(tmp_path, user_text="I want a burger, fries, and coke")
        assert row["user_text"] == "I want a burger, fries, and coke", (
            f"Column corruption — user_text was split: {row['user_text']!r}"
        )
        # Verify adjacent columns are still intact (not consumed by the comma split)
        assert "normalized_text" in row
        assert "local_intent" in row

    def test_double_quote_in_user_text_does_not_corrupt_csv(self, tmp_path):
        """A double-quote inside a field must be properly escaped by the CSV writer."""
        row = self._write_and_read(tmp_path, user_text='He said "extra cheese please"')
        assert row["user_text"] == 'He said "extra cheese please"', (
            f"Quote corruption — got: {row['user_text']!r}"
        )

    def test_newline_in_user_text_does_not_split_row(self, tmp_path):
        """A newline inside a field must not create a phantom second row."""
        row = self._write_and_read(tmp_path, user_text="burger\nand fries")
        assert row["user_text"] == "burger\nand fries", (
            f"Newline caused row split — got user_text: {row['user_text']!r}"
        )

    def test_comma_and_quote_in_gpt_reason(self, tmp_path):
        """gpt_reason with both commas and quotes round-trips correctly."""
        value = 'intent="add_item", confidence=0.95, text="I want, yes"'
        row = self._write_and_read(tmp_path, gpt_reason=value)
        assert row["gpt_reason"] == value, (
            f"gpt_reason corrupt after CSV round-trip: {row['gpt_reason']!r}"
        )

    def test_json_field_with_nested_commas_and_quotes(self, tmp_path):
        """A JSON string in local_candidates_json with inner commas and quotes parses correctly."""
        import json as _json
        candidates = [{"intent": "add_item,remove", "confidence": 0.9}, {"intent": 'ask "menu"', "confidence": 0.1}]
        json_value = _json.dumps(candidates)
        row = self._write_and_read(tmp_path, local_candidates_json=json_value)
        assert row["local_candidates_json"] == json_value, (
            f"JSON field corrupt after CSV round-trip: {row['local_candidates_json']!r}"
        )
        # Confirm it's still valid JSON after round-trip
        parsed = _json.loads(row["local_candidates_json"])
        assert len(parsed) == 2
        assert parsed[0]["intent"] == "add_item,remove"

    def test_all_headers_present_after_special_char_row(self, tmp_path):
        """A row with a comma-heavy user_text must still produce all expected column keys."""
        row = self._write_and_read(tmp_path, user_text="one, two, three, four, five")
        assert set(row.keys()) == set(HEADERS), (
            f"Column set mismatch after special-char row.\n"
            f"Missing: {set(HEADERS) - set(row.keys())}\n"
            f"Extra: {set(row.keys()) - set(HEADERS)}"
        )

    def test_mixed_special_chars_single_row_correct_column_count(self, tmp_path):
        """A row with comma+quote+newline produces exactly len(HEADERS) columns."""
        logger = _make_logger(tmp_path)
        logger.log_turn(_minimal_row(
            user_text='He said "get me,\nsome fries"',
            gpt_reason='score=0.9, label="food,drink"',
            normalized_text="get me fries",
        ))
        logger.flush()
        logger.shutdown()

        path = tmp_path / "gpt_repair_turns.csv"
        rows = _read_rows(path)
        assert len(rows) == 1, (
            f"Expected exactly 1 data row, got {len(rows)} — "
            "special characters likely caused a phantom row split"
        )
        assert len(rows[0]) == len(HEADERS), (
            f"Expected {len(HEADERS)} columns, got {len(rows[0])} — "
            "column corruption due to unescaped special characters"
        )
