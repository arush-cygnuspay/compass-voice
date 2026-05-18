"""Smoke tests for NluOrchestrator (Commit 7 extraction)."""
import sys
import types
import unittest


_intent_module = types.ModuleType("app.ml.intent.inference_intent")
_slot_module = types.ModuleType("app.ml.slot.inference_slot")


class _IntentBundle:
    pass


class _SlotBundle:
    pass


_intent_module.IntentBundle = _IntentBundle
_intent_module.predict_intent = lambda *a, **k: []
_slot_module.SlotBundle = _SlotBundle
_slot_module.predict_slots = lambda *a, **k: []
sys.modules.setdefault("app.ml.intent.inference_intent", _intent_module)
sys.modules.setdefault("app.ml.slot.inference_slot", _slot_module)


from app.core.nlu_orchestrator import NluOrchestrator, NluResolution
from app.session.session import Session
from app.state_machine.models.conversation_state import ConversationState


class NluOrchestratorSmokeTests(unittest.TestCase):
    def test_resolve_returns_nlu_resolution_with_all_fields(self):
        orchestrator = NluOrchestrator(
            intent_bundle=None,
            slot_bundle=None,
            diagnostics=None,
        )
        session = Session(session_id="nlu-1", restaurant_id="steves_grill")
        session.conversation_state = ConversationState.IDLE

        resolution = orchestrator.resolve(session=session, user_text="hello world")

        self.assertIsInstance(resolution, NluResolution)
        self.assertIsNotNone(resolution.cleaned_text)
        self.assertIsNotNone(resolution.normalized_text)
        self.assertIsNotNone(resolution.nlu)
        self.assertIsNotNone(resolution.intent_result)
        self.assertGreaterEqual(resolution.preprocess_ms, 0.0)
        self.assertGreaterEqual(resolution.nlu_ms, 0.0)

    def test_resolve_sets_last_nlu_on_session_context(self):
        orchestrator = NluOrchestrator(
            intent_bundle=None,
            slot_bundle=None,
            diagnostics=None,
        )
        session = Session(session_id="nlu-2", restaurant_id="steves_grill")
        session.conversation_state = ConversationState.IDLE

        orchestrator.resolve(session=session, user_text="anything")

        self.assertIsNotNone(session.conversation_context.last_nlu)


if __name__ == "__main__":
    unittest.main()
