"""Smoke tests for TurnDiagnostics after the diagnostics-refactor.

Verifies that the class can be instantiated with an empty backends list and
that the helper methods it still owns work correctly in isolation.
"""
import unittest
from types import SimpleNamespace

from app.core.turn_diagnostics import TurnDiagnostics
from app.session.session import Session
from app.state_machine.models.conversation_context import ConversationContext


class TurnDiagnosticsSmokeTests(unittest.TestCase):
    def _make_diag(self) -> TurnDiagnostics:
        return TurnDiagnostics(backends=[])

    def test_instantiation_with_empty_backends(self):
        diag = self._make_diag()
        self.assertIsInstance(diag, TurnDiagnostics)

    def test_enabled_is_false_with_no_backends(self):
        diag = self._make_diag()
        self.assertFalse(diag.enabled)

    def test_enabled_is_true_when_a_backend_has_enabled_true(self):
        class _AlwaysOn:
            enabled = True
            def record(self, event): pass

        diag = TurnDiagnostics(backends=[_AlwaysOn()])
        self.assertTrue(diag.enabled)

    def test_record_calls_each_backend(self):
        recorded = []

        class _Sink:
            enabled = True
            def record(self, event):
                recorded.append(event)

        diag = TurnDiagnostics(backends=[_Sink(), _Sink()])

        # Build a minimal TurnEvent by importing and constructing it directly
        from app.diagnostics.turn_event import TurnEvent
        event = TurnEvent(
            session_id="s1", turn_index=0,
            state_before="idle", state_after="idle", next_state="idle",
            pending_action="", current_prompt_field="", current_item_id="", current_item_name="",
            raw_user_text="hi", user_text="hi", normalized_text="hi",
            pred_main_intent="", pred_sub_intent="", pred_intent="",
            pred_intent_confidence=None, slot_model_ran=False, slots=(),
            response_key="greeting", response_text="Hello!", command=None,
            normalized_values={}, missing_required_fields=(),
            reprompt_field="", reprompt_count=0,
            reprompt_escalated=False, reprompt_escalation_count=0,
            fallback_triggered=False, fallback_reason="", fallback_count=0,
            slot_extraction_failed=False, slot_extraction_failure_count=0,
            invalid_modifier=False, invalid_modifier_count=0,
            user_repeated=False, repeated_user_turn_count=0,
        )
        diag.record(event)
        self.assertEqual(len(recorded), 2)

    def test_snapshot_context_for_logging_round_trip(self):
        session = Session(session_id="diag-1", restaurant_id="steves_grill")
        ctx = session.conversation_context
        ctx.current_item_id = "burger_1"
        ctx.current_item_name = "Chicken Burger"
        ctx.current_prompt_field = "side"

        diag = self._make_diag()
        snapshot = diag._snapshot_context_for_logging(session)

        self.assertEqual(snapshot["current_item_id"], "burger_1")
        self.assertEqual(snapshot["current_item_name"], "Chicken Burger")
        self.assertEqual(snapshot["current_prompt_field"], "side")
        self.assertEqual(snapshot["pending_action"], "")

    def test_safe_session_id_falls_back_to_unknown(self):
        diag = self._make_diag()
        self.assertEqual(diag._safe_session_id(SimpleNamespace()), "unknown_session")

    def test_infer_prompt_field_for_response_classifies_keys(self):
        diag = self._make_diag()
        session = Session(session_id="s1", restaurant_id="steves_grill")
        for key, expected in (
            ("repeat_modifier_options", "modifier"),
            ("repeat_side_size_options", "side_size"),
            ("repeat_size_options", "size"),
            ("repeat_side_options", "side"),
            ("invalid_quantity_option", "quantity"),
        ):
            with self.subTest(key=key):
                self.assertEqual(
                    diag._infer_prompt_field_for_response(
                        response_key=key, session=session
                    ),
                    expected,
                )


if __name__ == "__main__":
    unittest.main()
