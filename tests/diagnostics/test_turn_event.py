# tests/diagnostics/test_turn_event.py
"""Tests for TurnEvent — construction, immutability, field types."""
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


class TestTurnEventExtendedFields:
    """New optional fields added in Phase A must be None by default and accept values."""

    def test_extended_fields_default_to_none(self):
        event = _make_event()
        assert event.raw_slots is None
        assert event.effective_slots is None
        assert event.active_resolution_scope is None
        assert event.resolved_entity_type is None
        assert event.resolved_entity_id is None
        assert event.route_reason is None
        assert event.coercion_reason is None

    def test_extended_fields_accept_values(self):
        slots = (object(),)
        event = _make_event(
            raw_slots=slots,
            effective_slots=slots,
            active_resolution_scope="idle",
            resolved_entity_type="item",
            resolved_entity_id="item_123",
            route_reason="idle_item_slot_with_menu_evidence",
            coercion_reason="idle_modify_no_target_with_item_slot",
        )
        assert event.raw_slots is slots
        assert event.effective_slots is slots
        assert event.active_resolution_scope == "idle"
        assert event.resolved_entity_type == "item"
        assert event.resolved_entity_id == "item_123"
        assert event.route_reason == "idle_item_slot_with_menu_evidence"
        assert event.coercion_reason == "idle_modify_no_target_with_item_slot"

    def test_serialization_with_extended_fields(self):
        """json_backend must not choke on the new fields."""
        import dataclasses, json
        event = _make_event(
            raw_slots=(),
            active_resolution_scope="waiting_for_side",
            coercion_reason="idle_item_variant_no_cart_target",
        )
        d = {f.name: getattr(event, f.name) for f in dataclasses.fields(event)}
        # Must serialize without raising.
        json.dumps(d, default=str)


class TestTurnEventConstruction:
    def test_basic_construction_succeeds(self):
        event = _make_event()
        assert event.session_id == "s1"
        assert event.response_key == "greeting"

    def test_timing_fields_default_to_none(self):
        event = _make_event()
        assert event.preprocess_ms is None
        assert event.nlu_ms is None
        assert event.flow_ms is None
        assert event.route_ms is None
        assert event.handler_ms is None
        assert event.total_ms is None

    def test_timing_fields_accept_values(self):
        event = _make_event(preprocess_ms=1.5, nlu_ms=12.0, total_ms=50.0)
        assert event.preprocess_ms == 1.5
        assert event.nlu_ms == 12.0
        assert event.total_ms == 50.0

    def test_frozen_raises_on_mutation(self):
        event = _make_event()
        with pytest.raises((AttributeError, TypeError)):
            event.session_id = "other"  # type: ignore[misc]

    def test_slots_is_tuple(self):
        event = _make_event(slots=("a", "b"))
        assert isinstance(event.slots, tuple)
        assert event.slots == ("a", "b")

    def test_missing_required_fields_is_tuple(self):
        event = _make_event(missing_required_fields=("size", "quantity"))
        assert event.missing_required_fields == ("size", "quantity")

    def test_command_none_by_default(self):
        event = _make_event()
        assert event.command is None

    def test_command_dict_accepted(self):
        cmd = {"type": "ADD_ITEM_TO_CART", "payload": {"item_id": "b1"}}
        event = _make_event(command=cmd)
        assert event.command["type"] == "ADD_ITEM_TO_CART"

    def test_pred_intent_confidence_none(self):
        event = _make_event(pred_intent_confidence=None)
        assert event.pred_intent_confidence is None

    def test_pred_intent_confidence_float(self):
        event = _make_event(pred_intent_confidence=0.97)
        assert abs(event.pred_intent_confidence - 0.97) < 1e-9
