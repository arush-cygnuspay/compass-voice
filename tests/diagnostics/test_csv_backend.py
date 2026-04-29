# tests/diagnostics/test_csv_backend.py
"""Tests for CsvDiagnosticsBackend — adapts TurnEvent to NluCsvLogger."""
import unittest
from typing import Any

from app.diagnostics.backends.csv_backend import CsvDiagnosticsBackend
from app.diagnostics.turn_event import TurnEvent


def _make_event(**overrides) -> TurnEvent:
    defaults = dict(
        session_id="s1",
        turn_index=3,
        state_before="idle",
        state_after="waiting_for_modifier",
        next_state="waiting_for_modifier",
        pending_action="",
        current_prompt_field="modifier",
        current_item_id="burger_1",
        current_item_name="Chicken Burger",
        raw_user_text="add cheese",
        user_text="add cheese",
        normalized_text="add cheese",
        pred_main_intent="ADD_ITEM",
        pred_sub_intent="",
        pred_intent="ADD_ITEM",
        pred_intent_confidence=0.95,
        slot_model_ran=True,
        slots=(),
        response_key="ask_for_modifier",
        response_text="Which modifier would you like?",
        command=None,
        normalized_values={},
        missing_required_fields=(),
        reprompt_field="modifier",
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
        total_ms=45.2,
    )
    defaults.update(overrides)
    return TurnEvent(**defaults)


class _CapturingLogger:
    """Stub NluCsvLogger that captures log_turn kwargs instead of writing."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.calls: list[dict[str, Any]] = []

    def log_turn(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


class TestCsvDiagnosticsBackend(unittest.TestCase):
    def test_record_calls_log_turn_when_enabled(self):
        logger = _CapturingLogger(enabled=True)
        backend = CsvDiagnosticsBackend(logger)
        event = _make_event()

        backend.record(event)

        self.assertEqual(len(logger.calls), 1)

    def test_record_does_not_call_log_turn_when_disabled(self):
        logger = _CapturingLogger(enabled=False)
        backend = CsvDiagnosticsBackend(logger)

        backend.record(_make_event())

        self.assertEqual(len(logger.calls), 0)

    def test_enabled_property_reflects_logger(self):
        self.assertTrue(CsvDiagnosticsBackend(_CapturingLogger(enabled=True)).enabled)
        self.assertFalse(CsvDiagnosticsBackend(_CapturingLogger(enabled=False)).enabled)

    def test_record_maps_session_id(self):
        logger = _CapturingLogger()
        backend = CsvDiagnosticsBackend(logger)
        backend.record(_make_event(session_id="abc-123"))
        self.assertEqual(logger.calls[0]["session_id"], "abc-123")

    def test_record_maps_turn_index(self):
        logger = _CapturingLogger()
        backend = CsvDiagnosticsBackend(logger)
        backend.record(_make_event(turn_index=7))
        self.assertEqual(logger.calls[0]["turn_index"], 7)

    def test_record_maps_state_fields(self):
        logger = _CapturingLogger()
        backend = CsvDiagnosticsBackend(logger)
        backend.record(_make_event(state_before="idle", state_after="waiting_for_modifier"))
        call = logger.calls[0]
        self.assertEqual(call["state_before"], "idle")
        self.assertEqual(call["state_after"], "waiting_for_modifier")

    def test_record_maps_response_key(self):
        logger = _CapturingLogger()
        backend = CsvDiagnosticsBackend(logger)
        backend.record(_make_event(response_key="ask_for_modifier"))
        self.assertEqual(logger.calls[0]["response_key"], "ask_for_modifier")

    def test_record_maps_slots(self):
        logger = _CapturingLogger()
        backend = CsvDiagnosticsBackend(logger)
        fake_slot = object()
        backend.record(_make_event(slots=(fake_slot,)))
        self.assertIn(fake_slot, logger.calls[0]["slots"])

    def test_record_maps_command(self):
        logger = _CapturingLogger()
        backend = CsvDiagnosticsBackend(logger)
        cmd = {"type": "ADD_ITEM_TO_CART"}
        backend.record(_make_event(command=cmd))
        self.assertEqual(logger.calls[0]["command"], cmd)

    def test_record_maps_nlu_fields(self):
        logger = _CapturingLogger()
        backend = CsvDiagnosticsBackend(logger)
        backend.record(_make_event(
            pred_main_intent="ADD_ITEM",
            pred_sub_intent="SUB",
            pred_intent="ADD_ITEM",
            pred_intent_confidence=0.88,
            slot_model_ran=True,
        ))
        call = logger.calls[0]
        self.assertEqual(call["pred_main_intent"], "ADD_ITEM")
        self.assertEqual(call["pred_sub_intent"], "SUB")
        self.assertEqual(call["pred_intent"], "ADD_ITEM")
        self.assertAlmostEqual(call["pred_intent_confidence"], 0.88)
        self.assertTrue(call["slot_model_ran"])


if __name__ == "__main__":
    unittest.main()
