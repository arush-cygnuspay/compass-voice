"""Smoke tests for the TurnDiagnostics module after the Commit 1 extraction.

Verifies that the moved helper class can be instantiated in isolation
and that ``_snapshot_context_for_logging`` round-trips a populated
ConversationContext correctly.
"""
import unittest
from types import SimpleNamespace

from app.core.turn_diagnostics import TurnDiagnostics
from app.session.session import Session
from app.state_machine.models.conversation_context import ConversationContext


class _StubLogger:
    enabled = False


class _StubResponder:
    def build(self, *, response_key, context, payload=None):
        return ""


class _StubMenuRepo:
    pass


class TurnDiagnosticsSmokeTests(unittest.TestCase):
    def test_instantiation_with_constructor_dependencies(self):
        diag = TurnDiagnostics(
            menu_repo=_StubMenuRepo(),
            nlu_logger=_StubLogger(),
            responder=_StubResponder(),
        )
        self.assertIsInstance(diag, TurnDiagnostics)

    def test_snapshot_context_for_logging_round_trip(self):
        session = Session(session_id="diag-1", restaurant_id="demo")
        ctx = session.conversation_context
        ctx.current_item_id = "burger_1"
        ctx.current_item_name = "Chicken Burger"
        ctx.current_prompt_field = "side"

        diag = TurnDiagnostics(
            menu_repo=_StubMenuRepo(),
            nlu_logger=_StubLogger(),
            responder=_StubResponder(),
        )
        snapshot = diag._snapshot_context_for_logging(session)

        self.assertEqual(snapshot["current_item_id"], "burger_1")
        self.assertEqual(snapshot["current_item_name"], "Chicken Burger")
        self.assertEqual(snapshot["current_prompt_field"], "side")
        self.assertEqual(snapshot["pending_action"], "")

    def test_safe_session_id_falls_back_to_unknown(self):
        diag = TurnDiagnostics(
            menu_repo=_StubMenuRepo(),
            nlu_logger=_StubLogger(),
            responder=_StubResponder(),
        )
        # SimpleNamespace with no session_id / id
        self.assertEqual(diag._safe_session_id(SimpleNamespace()), "unknown_session")

    def test_infer_prompt_field_for_response_classifies_keys(self):
        diag = TurnDiagnostics(
            menu_repo=_StubMenuRepo(),
            nlu_logger=_StubLogger(),
            responder=_StubResponder(),
        )
        session = Session(session_id="s1", restaurant_id="demo")
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
