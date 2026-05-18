# tests/logging/test_gpt_log_record_builder.py
"""Tests for build_gpt_repair_log_record() and build_gpt_repair_csv_row()."""
from __future__ import annotations

import json

import pytest

from app.diagnostics.turn_event import TurnEvent
from app.logging.gpt_repair_csv_logger import HEADERS
from app.nlu.semantic_repair.gpt_log_record_builder import (
    build_gpt_repair_csv_row,
    build_gpt_repair_log_record,
)


def _make_event(**overrides) -> TurnEvent:
    defaults = dict(
        session_id="sess-abc",
        turn_index=3,
        state_before="idle",
        state_after="idle",
        next_state="idle",
        pending_action="",
        current_prompt_field="",
        current_item_id="",
        current_item_name="",
        raw_user_text="",
        user_text="I want a burger",
        normalized_text="i want a burger",
        pred_main_intent="add_item",
        pred_sub_intent="",
        pred_intent="add_item",
        pred_intent_confidence=0.62,
        slot_model_ran=True,
        slots=(),
        response_key="ask_for_size",
        response_text="What size?",
        command=None,
        normalized_values={},
        missing_required_fields=(),
        reprompt_field="",
        reprompt_count=0,
        reprompt_escalated=False,
        reprompt_escalation_count=0,
        fallback_triggered=False,
        fallback_reason="",
        fallback_count=0,
        slot_extraction_failed=False,
        slot_extraction_failure_count=0,
        invalid_modifier=False,
        invalid_modifier_count=0,
        user_repeated=False,
        repeated_user_turn_count=0,
    )
    defaults.update(overrides)
    return TurnEvent(**defaults)


class TestBuildGptRepairLogRecord:
    def test_top_level_keys_present(self):
        event = _make_event()
        record = build_gpt_repair_log_record(event)
        assert set(record.keys()) >= {
            "timestamp_utc", "session_id", "turn_index",
            "local", "allowed", "context", "gpt", "final",
        }

    def test_session_id_and_turn_index_correct(self):
        event = _make_event(session_id="test-session", turn_index=7)
        record = build_gpt_repair_log_record(event)
        assert record["session_id"] == "test-session"
        assert record["turn_index"] == 7

    def test_local_block_fields(self):
        event = _make_event(
            local_intent_before_gpt="add_item",
            local_sub_intent_before_gpt="burger_type",
            local_intent_confidence_before_gpt=0.62,
            local_intent_candidates_json='[{"intent":"add_item","confidence":0.62}]',
            local_slots_before_gpt='[{"name":"item_name","value":"burger"}]',
            local_route_allowed=True,
            local_route_reject_reason=None,
        )
        local = build_gpt_repair_log_record(event)["local"]
        assert local["intent"] == "add_item"
        assert local["sub_intent"] == "burger_type"
        assert local["confidence"] == 0.62
        assert "add_item" in local["candidates_json"]
        assert local["route_allowed"] is True
        assert local["route_reject_reason"] is None

    def test_allowed_block_fields(self):
        event = _make_event(
            gpt_repair_eligible=True,
            gpt_repair_eligible_reason="low_confidence_gap",
            gpt_candidate_count=5,
            gpt_skipped_reason=None,
            gpt_phase=2,
        )
        allowed = build_gpt_repair_log_record(event)["allowed"]
        assert allowed["repair_eligible"] is True
        assert allowed["eligible_reason"] == "low_confidence_gap"
        assert allowed["candidate_count"] == 5
        assert allowed["phase"] == 2

    def test_context_block_fields(self):
        event = _make_event(state_before="waiting_for_size", response_key="ask_for_size")
        ctx = build_gpt_repair_log_record(event)["context"]
        assert ctx["state_before"] == "waiting_for_size"
        assert ctx["response_key"] == "ask_for_size"

    def test_gpt_block_fields(self):
        event = _make_event(
            gpt_called=True,
            gpt_decision="repair_intent",
            gpt_selected_intent="checkout",
            gpt_selected_control_intent=None,
            gpt_slot_corrections_json=None,
            gpt_confidence=0.92,
            gpt_reason="user wants to pay",
            gpt_latency_ms=220.0,
            gpt_total_ms=225.0,
            gpt_timeout=False,
            gpt_parse_error=None,
            gpt_model="gpt-4o-mini",
            gpt_prompt_chars=300,
            gpt_completion_chars=40,
        )
        gpt = build_gpt_repair_log_record(event)["gpt"]
        assert gpt["called"] is True
        assert gpt["decision"] == "repair_intent"
        assert gpt["selected_intent"] == "checkout"
        assert gpt["confidence"] == 0.92
        assert gpt["model"] == "gpt-4o-mini"
        assert gpt["timeout"] is False

    def test_final_block_fields(self):
        event = _make_event(
            gpt_applied=False,
            gpt_apply_reason="shadow_mode",
            final_intent_after_gpt="add_item",
            final_response_key="ask_for_size",
            training_candidate=True,
        )
        final = build_gpt_repair_log_record(event)["final"]
        assert final["applied"] is False
        assert final["apply_reason"] == "shadow_mode"
        assert final["intent_after_gpt"] == "add_item"
        assert final["response_key"] == "ask_for_size"
        assert final["training_candidate"] is True

    def test_record_json_serializable(self):
        event = _make_event(
            gpt_repair_eligible=True,
            gpt_called=True,
            gpt_decision="repair_intent",
            gpt_selected_intent="checkout",
            gpt_confidence=0.91,
            gpt_total_ms=210.5,
        )
        record = build_gpt_repair_log_record(event)
        # Must not raise
        json.dumps(record)

    def test_default_event_produces_valid_record(self):
        event = _make_event()
        record = build_gpt_repair_log_record(event)
        assert record["local"]["intent"] is None
        assert record["allowed"]["repair_eligible"] is False
        assert record["gpt"]["called"] is False
        assert record["final"]["applied"] is False
        assert record["final"]["training_candidate"] is False

    def test_no_raw_prompt_or_full_menu_in_record(self):
        event = _make_event()
        record = build_gpt_repair_log_record(event)
        # No field should contain "system" or raw_prompt key anywhere
        record_str = json.dumps(record)
        assert "raw_prompt" not in record_str
        assert '"system"' not in record_str

    def test_timestamp_utc_is_iso_string(self):
        event = _make_event()
        record = build_gpt_repair_log_record(event)
        ts = record["timestamp_utc"]
        assert isinstance(ts, str)
        assert "T" in ts  # ISO 8601 format


class TestBuildGptRepairCsvRow:
    """build_gpt_repair_csv_row() must produce a flat dict matching HEADERS."""

    def test_keys_match_headers(self):
        event = _make_event()
        row = build_gpt_repair_csv_row(event)
        assert set(row.keys()) == set(HEADERS)

    def test_session_id_and_turn_index(self):
        event = _make_event(session_id="flat-test", turn_index=5)
        row = build_gpt_repair_csv_row(event)
        assert row["session_id"] == "flat-test"
        assert row["turn_index"] == 5

    def test_local_fields_flat(self):
        event = _make_event(
            local_intent_before_gpt="add_item",
            local_sub_intent_before_gpt="burger_type",
            local_intent_confidence_before_gpt=0.72,
        )
        row = build_gpt_repair_csv_row(event)
        assert row["local_intent"] == "add_item"
        assert row["local_sub_intent"] == "burger_type"
        assert row["local_confidence"] == 0.72

    def test_gpt_fields_flat(self):
        event = _make_event(
            gpt_repair_eligible=True,
            gpt_called=True,
            gpt_decision="repair_intent",
            gpt_selected_intent="checkout",
            gpt_confidence=0.91,
            gpt_phase=2,
        )
        row = build_gpt_repair_csv_row(event)
        assert row["gpt_repair_eligible"] is True
        assert row["gpt_called"] is True
        assert row["gpt_decision"] == "repair_intent"
        assert row["gpt_selected_intent"] == "checkout"
        assert row["gpt_phase"] == 2

    def test_final_fields_flat(self):
        event = _make_event(
            gpt_applied=False,
            gpt_apply_reason="shadow_mode",
            final_intent_after_gpt="add_item",
            final_response_key="ask_for_size",
            training_candidate=True,
        )
        row = build_gpt_repair_csv_row(event)
        assert row["gpt_applied"] is False
        assert row["gpt_apply_reason"] == "shadow_mode"
        assert row["final_intent_after_gpt"] == "add_item"
        assert row["training_candidate"] is True

    def test_no_nested_dicts_in_row(self):
        event = _make_event()
        row = build_gpt_repair_csv_row(event)
        for val in row.values():
            assert not isinstance(val, dict), f"Unexpected nested dict in CSV row"

    def test_timestamp_utc_present(self):
        event = _make_event()
        row = build_gpt_repair_csv_row(event)
        assert "T" in row["timestamp_utc"]
