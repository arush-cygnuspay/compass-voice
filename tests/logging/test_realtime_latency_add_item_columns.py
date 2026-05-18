# tests/logging/test_realtime_latency_add_item_columns.py
"""Tests for ADD_ITEM extractor summary columns in realtime_turn_latency.csv."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from app.logging.realtime_latency_logger import (
    RealtimeLatencyLogger,
    RealtimeTurnTrace,
)

_ADD_ITEM_COLUMNS = [
    "add_item_extractor_called",
    "add_item_decision",
    "add_item_items_count",
    "add_item_confidence",
    "add_item_total_ms",
]

# Columns that must appear BEFORE the ADD_ITEM columns (order preserved)
_GPT_ANCHOR = "gpt_fallback_type"


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


# ---------------------------------------------------------------------------
# RealtimeTurnTrace fields
# ---------------------------------------------------------------------------

class TestAddItemTraceFields:
    def test_add_item_extractor_called_default_false(self):
        t = RealtimeTurnTrace()
        assert t.add_item_extractor_called is False

    def test_add_item_decision_default_empty(self):
        t = RealtimeTurnTrace()
        assert t.add_item_decision == ""

    def test_add_item_items_count_default_none(self):
        t = RealtimeTurnTrace()
        assert t.add_item_items_count is None

    def test_add_item_confidence_default_none(self):
        t = RealtimeTurnTrace()
        assert t.add_item_confidence is None

    def test_add_item_total_ms_default_none(self):
        t = RealtimeTurnTrace()
        assert t.add_item_total_ms is None


# ---------------------------------------------------------------------------
# CSV_COLUMNS — add_item columns present and ordered correctly
# ---------------------------------------------------------------------------

class TestCsvColumnsAddItem:
    def test_all_add_item_columns_in_csv_columns(self):
        for col in _ADD_ITEM_COLUMNS:
            assert col in RealtimeLatencyLogger.CSV_COLUMNS, f"Missing column: {col}"

    def test_add_item_columns_appear_after_gpt_columns(self):
        cols = RealtimeLatencyLogger.CSV_COLUMNS
        gpt_idx = cols.index(_GPT_ANCHOR)
        for col in _ADD_ITEM_COLUMNS:
            ai_idx = cols.index(col)
            assert ai_idx > gpt_idx, (
                f"{col} (idx={ai_idx}) should come after {_GPT_ANCHOR} (idx={gpt_idx})"
            )

    def test_existing_gpt_columns_not_reordered(self):
        cols = RealtimeLatencyLogger.CSV_COLUMNS
        gpt_cols = [
            "gpt_called",
            "gpt_decision",
            "gpt_selected_intent",
            "gpt_confidence",
            "gpt_total_ms",
            "gpt_timeout",
            "gpt_applied",
            "gpt_fallback_type",
        ]
        indices = [cols.index(c) for c in gpt_cols]
        assert indices == sorted(indices), "GPT columns are out of order"

    def test_legacy_columns_not_reordered(self):
        """turn_index must still be before session_id etc."""
        cols = RealtimeLatencyLogger.CSV_COLUMNS
        assert cols.index("turn_index") < cols.index("response_key")


# ---------------------------------------------------------------------------
# CSV header written correctly
# ---------------------------------------------------------------------------

class TestCsvHeaderWritten:
    def test_csv_header_contains_add_item_columns(self, tmp_path):
        logger = _make_logger(tmp_path)
        try:
            # Trigger header creation by writing a minimal row
            logger.write(RealtimeTurnTrace())
            logger.flush()
            headers = _read_csv_headers(tmp_path / "rt.csv")
            for col in _ADD_ITEM_COLUMNS:
                assert col in headers, f"Missing column in CSV header: {col}"
        finally:
            logger.shutdown()

    def test_csv_header_add_item_after_gpt(self, tmp_path):
        logger = _make_logger(tmp_path)
        try:
            logger.write(RealtimeTurnTrace())
            logger.flush()
            headers = _read_csv_headers(tmp_path / "rt.csv")
            gpt_idx = headers.index(_GPT_ANCHOR)
            for col in _ADD_ITEM_COLUMNS:
                assert headers.index(col) > gpt_idx
        finally:
            logger.shutdown()


# ---------------------------------------------------------------------------
# CSV row content — add_item notes populated from payload["notes"]["add_item"]
# ---------------------------------------------------------------------------

class TestCsvRowAddItemContent:
    def _make_payload_with_add_item(self, *, called=True, decision="ok", items_count=2) -> dict:
        return {
            "turn_index": 1,
            "turn_id": "abc123",
            "session_id": "sess1",
            "notes": {
                "add_item": {
                    "add_item_extractor_called": called,
                    "add_item_decision": decision,
                    "add_item_items_count": items_count,
                    "add_item_confidence": 0.88,
                    "add_item_total_ms": 120.5,
                }
            },
        }

    def test_add_item_values_written_to_csv(self, tmp_path):
        logger = _make_logger(tmp_path)
        try:
            payload = self._make_payload_with_add_item(called=True, decision="ok", items_count=2)
            logger._write_csv_row(payload)
        finally:
            logger.shutdown()

        with (tmp_path / "rt.csv").open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        row = rows[0]
        assert row["add_item_extractor_called"] == "True"
        assert row["add_item_decision"] == "ok"
        assert row["add_item_items_count"] == "2"

    def test_add_item_absent_from_notes_writes_empty_string(self, tmp_path):
        logger = _make_logger(tmp_path)
        try:
            payload = {"turn_index": 1, "notes": {}}
            logger._write_csv_row(payload)
        finally:
            logger.shutdown()

        with (tmp_path / "rt.csv").open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        row = rows[0]
        assert row["add_item_extractor_called"] == ""
        assert row["add_item_decision"] == ""

    def test_add_item_decision_no_repair_written(self, tmp_path):
        logger = _make_logger(tmp_path)
        try:
            payload = self._make_payload_with_add_item(called=False, decision="no_repair", items_count=0)
            logger._write_csv_row(payload)
        finally:
            logger.shutdown()

        with (tmp_path / "rt.csv").open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        row = rows[0]
        assert row["add_item_decision"] == "no_repair"
