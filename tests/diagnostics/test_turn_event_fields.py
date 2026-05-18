# tests/diagnostics/test_turn_event_fields.py
"""Tests for the GPT shadow-mode fields added/renamed in TurnEvent."""
import dataclasses
import json

import pytest

from app.diagnostics.turn_event import TurnEvent


def _make_event(**overrides) -> TurnEvent:
    defaults = dict(
        session_id="s1",
        turn_index=0,
        state_before="idle",
        state_after="idle",
        next_state="idle",
        pending_action="",
        current_prompt_field="",
        current_item_id="",
        current_item_name="",
        raw_user_text="hi",
        user_text="hi",
        normalized_text="hi",
        pred_main_intent="",
        pred_sub_intent="",
        pred_intent="",
        pred_intent_confidence=None,
        slot_model_ran=False,
        slots=(),
        response_key="greeting",
        response_text="Hello!",
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


class TestGptFieldDefaults:
    """GPT bool/int fields must have concrete defaults, not None."""

    def test_gpt_repair_eligible_defaults_false(self):
        assert _make_event().gpt_repair_eligible is False

    def test_gpt_called_defaults_false(self):
        assert _make_event().gpt_called is False

    def test_gpt_timeout_defaults_false(self):
        assert _make_event().gpt_timeout is False

    def test_gpt_applied_defaults_false(self):
        assert _make_event().gpt_applied is False

    def test_training_candidate_defaults_false(self):
        assert _make_event().training_candidate is False

    def test_gpt_phase_defaults_zero(self):
        assert _make_event().gpt_phase == 0


class TestGptFieldRenames:
    """Fields renamed from gpt_repaired_* / gpt_slot_corrections."""

    def test_gpt_selected_intent_defaults_none(self):
        assert _make_event().gpt_selected_intent is None

    def test_gpt_selected_control_intent_defaults_none(self):
        assert _make_event().gpt_selected_control_intent is None

    def test_gpt_slot_corrections_json_defaults_none(self):
        assert _make_event().gpt_slot_corrections_json is None

    def test_gpt_selected_intent_accepts_string(self):
        event = _make_event(gpt_selected_intent="checkout")
        assert event.gpt_selected_intent == "checkout"

    def test_gpt_selected_control_intent_accepts_string(self):
        event = _make_event(gpt_selected_control_intent="cancel")
        assert event.gpt_selected_control_intent == "cancel"

    def test_gpt_slot_corrections_json_accepts_string(self):
        payload = json.dumps([{"slot_name": "size", "old_value": None, "new_value": "large", "operation": "add"}])
        event = _make_event(gpt_slot_corrections_json=payload)
        assert event.gpt_slot_corrections_json == payload

    def test_old_field_names_do_not_exist(self):
        event = _make_event()
        assert not hasattr(event, "gpt_repaired_intent")
        assert not hasattr(event, "gpt_repaired_control_intent")
        assert not hasattr(event, "gpt_slot_corrections")


class TestNewLocalSubIntentField:
    """local_sub_intent_before_gpt must be present with None default."""

    def test_defaults_none(self):
        assert _make_event().local_sub_intent_before_gpt is None

    def test_accepts_string(self):
        event = _make_event(local_sub_intent_before_gpt="burger_type")
        assert event.local_sub_intent_before_gpt == "burger_type"

    def test_field_order_after_local_intent(self):
        field_names = [f.name for f in dataclasses.fields(TurnEvent)]
        idx_intent = field_names.index("local_intent_before_gpt")
        idx_sub = field_names.index("local_sub_intent_before_gpt")
        assert idx_sub == idx_intent + 1, "local_sub_intent_before_gpt should immediately follow local_intent_before_gpt"


class TestGptBoolValues:
    """GPT bool fields accept True/False without coercion issues."""

    def test_gpt_repair_eligible_true(self):
        event = _make_event(gpt_repair_eligible=True)
        assert event.gpt_repair_eligible is True

    def test_gpt_called_true(self):
        event = _make_event(gpt_called=True)
        assert event.gpt_called is True

    def test_gpt_timeout_true(self):
        event = _make_event(gpt_timeout=True)
        assert event.gpt_timeout is True

    def test_gpt_applied_true(self):
        event = _make_event(gpt_applied=True)
        assert event.gpt_applied is True

    def test_training_candidate_true(self):
        event = _make_event(training_candidate=True)
        assert event.training_candidate is True


class TestGptFullRecordSerialization:
    """A fully-populated GPT TurnEvent must serialize to JSON without error."""

    def test_full_gpt_event_serializes(self):
        event = _make_event(
            local_intent_before_gpt="add_item",
            local_sub_intent_before_gpt="burger_type",
            local_intent_confidence_before_gpt=0.72,
            local_intent_candidates_json='[{"intent":"add_item","confidence":0.72}]',
            local_slots_before_gpt='[{"name":"item_name","value":"burger"}]',
            local_route_allowed=True,
            local_route_reject_reason=None,
            gpt_repair_eligible=True,
            gpt_repair_eligible_reason="low_confidence_gap",
            gpt_candidate_count=5,
            gpt_skipped_reason=None,
            gpt_phase=2,
            gpt_called=True,
            gpt_payload_build_ms=1.2,
            gpt_request_ms=210.5,
            gpt_parse_ms=0.8,
            gpt_total_ms=212.5,
            gpt_prompt_chars=350,
            gpt_completion_chars=42,
            gpt_model="gpt-4o-mini",
            gpt_decision="repair_intent",
            gpt_selected_intent="checkout",
            gpt_selected_control_intent=None,
            gpt_slot_corrections_json=None,
            gpt_confidence=0.91,
            gpt_reason="user said checkout",
            gpt_latency_ms=210.5,
            gpt_timeout=False,
            gpt_parse_error=None,
            gpt_applied=False,
            gpt_apply_reason="shadow_mode",
            final_intent_after_gpt="add_item",
            final_slots_after_gpt="[]",
            final_response_key="greeting",
            training_candidate=True,
        )
        d = {f.name: getattr(event, f.name) for f in dataclasses.fields(event)}
        # Must not raise
        json.dumps(d, default=str)
