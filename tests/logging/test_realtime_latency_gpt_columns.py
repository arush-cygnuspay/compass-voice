# tests/logging/test_realtime_latency_gpt_columns.py
"""Tests for the 8 GPT summary columns added to realtime_turn_latency.csv (Part 5)."""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from app.logging.realtime_latency_logger import (
    RealtimeLatencyLogger,
    RealtimeTurnTrace,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_GPT_COLUMNS = [
    "gpt_called",
    "gpt_decision",
    "gpt_selected_intent",
    "gpt_confidence",
    "gpt_total_ms",
    "gpt_timeout",
    "gpt_applied",
    "gpt_fallback_type",
]

_LEGACY_ANCHOR_COLUMNS = [
    # A sample of columns that must appear before the new GPT columns
    "turn_index",
    "session_id",
    "response_key",
    "engine_processing_ms",
    "notes",
]


def _read_csv_headers(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as fh:
        return next(csv.reader(fh))


def _make_logger(tmp_path: Path) -> RealtimeLatencyLogger:
    return RealtimeLatencyLogger(
        file_path=str(tmp_path / "rt.jsonl"),
        csv_file_path=str(tmp_path / "rt.csv"),
        enabled=True,
        sync_write_immediately=True,
        fsync_on_write=False,
    )


def _minimal_trace(**overrides) -> RealtimeTurnTrace:
    t = RealtimeTurnTrace()
    for k, v in overrides.items():
        setattr(t, k, v)
    return t


# ---------------------------------------------------------------------------
# RealtimeTurnTrace dataclass — new fields exist
# ---------------------------------------------------------------------------


class TestRealtimeTurnTraceNewFields:
    def test_gpt_called_default_false(self):
        t = RealtimeTurnTrace()
        assert t.gpt_called is False

    def test_gpt_decision_default_empty_string(self):
        t = RealtimeTurnTrace()
        assert t.gpt_decision == ""

    def test_gpt_selected_intent_default_empty_string(self):
        t = RealtimeTurnTrace()
        assert t.gpt_selected_intent == ""

    def test_gpt_confidence_default_none(self):
        t = RealtimeTurnTrace()
        assert t.gpt_confidence is None

    def test_gpt_total_ms_default_none(self):
        t = RealtimeTurnTrace()
        assert t.gpt_total_ms is None

    def test_gpt_timeout_default_false(self):
        t = RealtimeTurnTrace()
        assert t.gpt_timeout is False

    def test_gpt_applied_default_false(self):
        t = RealtimeTurnTrace()
        assert t.gpt_applied is False

    def test_gpt_fallback_type_default_empty_string(self):
        t = RealtimeTurnTrace()
        assert t.gpt_fallback_type == ""

    def test_gpt_fields_can_be_set(self):
        t = RealtimeTurnTrace()
        t.gpt_called = True
        t.gpt_decision = "pending_async"
        t.gpt_applied = False
        t.gpt_confidence = 0.87
        t.gpt_total_ms = 450.0
        t.gpt_timeout = True
        t.gpt_selected_intent = "add_item"
        t.gpt_fallback_type = "off_topic"
        assert t.gpt_called is True
        assert t.gpt_decision == "pending_async"


# ---------------------------------------------------------------------------
# CSV_COLUMNS — new columns present and in correct position
# ---------------------------------------------------------------------------


class TestCsvColumnsDefinition:
    def test_gpt_columns_present(self):
        cols = RealtimeLatencyLogger.CSV_COLUMNS
        for col in _GPT_COLUMNS:
            assert col in cols, f"Missing column: {col}"

    def test_exactly_8_gpt_columns(self):
        cols = RealtimeLatencyLogger.CSV_COLUMNS
        found = [c for c in cols if c.startswith("gpt_")]
        assert len(found) == 8

    def test_gpt_columns_at_end(self):
        cols = RealtimeLatencyLogger.CSV_COLUMNS
        gpt_start = cols.index("gpt_called")
        # All 8 gpt_ columns appear at and after the gpt_called position
        for col in _GPT_COLUMNS:
            assert cols.index(col) >= gpt_start

    def test_legacy_columns_come_before_gpt(self):
        cols = RealtimeLatencyLogger.CSV_COLUMNS
        gpt_start = cols.index("gpt_called")
        for legacy_col in _LEGACY_ANCHOR_COLUMNS:
            assert cols.index(legacy_col) < gpt_start, (
                f"Legacy column '{legacy_col}' should appear before GPT columns"
            )

    def test_notes_column_before_gpt_called(self):
        cols = RealtimeLatencyLogger.CSV_COLUMNS
        assert cols.index("notes") < cols.index("gpt_called")

    def test_no_duplicate_columns(self):
        cols = RealtimeLatencyLogger.CSV_COLUMNS
        assert len(cols) == len(set(cols))

    def test_all_gpt_columns_consecutive_and_ordered(self):
        """GPT columns must appear in their defined order relative to each other.

        (add_item columns were added after gpt_fallback_type, so GPT columns
        are no longer the very last columns — but their relative order is fixed.)
        """
        cols = RealtimeLatencyLogger.CSV_COLUMNS
        indices = [cols.index(c) for c in _GPT_COLUMNS]
        # Indices must be strictly ascending (preserves order)
        assert indices == sorted(indices), (
            f"GPT columns are out of order.\nGot indices: {indices}\nFor columns: {_GPT_COLUMNS}"
        )
        # All GPT columns must appear before any add_item column
        add_item_start = next(
            (cols.index(c) for c in cols if c.startswith("add_item_")),
            len(cols),
        )
        for col, idx in zip(_GPT_COLUMNS, indices):
            assert idx < add_item_start, (
                f"GPT column '{col}' (idx={idx}) must appear before add_item columns (start={add_item_start})"
            )


# ---------------------------------------------------------------------------
# CSV write — all_shadow pending_async pattern
# ---------------------------------------------------------------------------


class TestCsvWriteGptColumns:
    def test_all_shadow_pending_async_written(self, tmp_path):
        logger = _make_logger(tmp_path)
        trace = _minimal_trace(
            session_id="s1",
            turn_index=1,
            gpt_called=True,
            gpt_decision="pending_async",
            gpt_applied=False,
        )
        logger.write(trace)

        headers = _read_csv_headers(tmp_path / "rt.csv")
        assert "gpt_called" in headers
        assert "gpt_decision" in headers

        with (tmp_path / "rt.csv").open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))

        assert len(rows) == 1
        assert rows[0]["gpt_called"] in ("True", "1", True, "true")
        assert rows[0]["gpt_decision"] == "pending_async"
        assert rows[0]["gpt_applied"] in ("False", "0", False, "false")

    def test_eligible_only_actual_values_written(self, tmp_path):
        logger = _make_logger(tmp_path)
        trace = _minimal_trace(
            session_id="s2",
            turn_index=2,
            gpt_called=True,
            gpt_decision="no_repair",
            gpt_selected_intent="add_item",
            gpt_confidence=0.92,
            gpt_total_ms=380.0,
            gpt_timeout=False,
            gpt_applied=False,
            gpt_fallback_type="",
        )
        logger.write(trace)

        with (tmp_path / "rt.csv").open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))

        assert rows[0]["gpt_decision"] == "no_repair"
        assert rows[0]["gpt_selected_intent"] == "add_item"

    def test_gpt_not_called_defaults_written(self, tmp_path):
        logger = _make_logger(tmp_path)
        trace = _minimal_trace(session_id="s3", turn_index=3)
        logger.write(trace)

        with (tmp_path / "rt.csv").open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))

        # gpt_called defaults to False, gpt_decision empty string
        assert rows[0]["gpt_called"] in ("False", "0", False, "false", "")
        assert rows[0]["gpt_decision"] == ""

    def test_header_written_with_gpt_columns(self, tmp_path):
        logger = _make_logger(tmp_path)
        logger.write(_minimal_trace())

        headers = _read_csv_headers(tmp_path / "rt.csv")
        for col in _GPT_COLUMNS:
            assert col in headers

    def test_existing_columns_not_reordered(self, tmp_path):
        logger = _make_logger(tmp_path)
        logger.write(_minimal_trace())

        headers = _read_csv_headers(tmp_path / "rt.csv")
        # Check that legacy columns appear in their expected relative order
        pairs = [
            ("turn_index", "session_id"),
            ("session_id", "response_key"),
            ("engine_processing_ms", "gpt_called"),
        ]
        for before, after in pairs:
            assert headers.index(before) < headers.index(after), (
                f"Column '{before}' should appear before '{after}'"
            )
