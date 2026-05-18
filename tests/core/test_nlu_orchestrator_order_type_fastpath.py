"""Tests proving that WAITING_FOR_ORDER_TYPE lexical hits skip full NLU
inference (intent classifier + slot extractor are never called).

Uses module-level stubs for the ML bundles so the test suite can import
the app code without real model artefacts.
"""
from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Stub ML bundles before importing any app code
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# App imports
# ---------------------------------------------------------------------------

from app.core.nlu_orchestrator import NluOrchestrator, NluResolution
from app.nlu.intent_resolution.intent import Intent
from app.session.session import Session
from app.state_machine.models.conversation_state import ConversationState


def _make_orchestrator() -> NluOrchestrator:
    return NluOrchestrator(intent_bundle=None, slot_bundle=None, diagnostics=None)


def _make_session(state: ConversationState) -> Session:
    session = Session(session_id="test-fastpath", restaurant_id="steves_grill")
    session.conversation_state = state
    return session


class TestOrderTypeFastPath(unittest.TestCase):
    """Full NLU model must NOT be called for lexical order-type matches."""

    def _run_with_spies(self, user_text: str, state: ConversationState):
        """Return (resolution, resolve_nlu_called)."""
        orchestrator = _make_orchestrator()
        session = _make_session(state)
        called = []

        with patch("app.core.nlu_orchestrator.resolve_nlu", side_effect=lambda **kw: (
            called.append(True) or __import__("app.nlu.nlu_resolver", fromlist=["resolve_nlu"]).resolve_nlu(**kw)
        )):
            resolution = orchestrator.resolve(session=session, user_text=user_text)

        return resolution, bool(called)

    # -----------------------------------------------------------------------
    # Lexical match → NLU skipped
    # -----------------------------------------------------------------------

    def test_pickup_skips_resolve_nlu(self):
        with patch("app.core.nlu_orchestrator.resolve_nlu") as mock_nlu:
            orchestrator = _make_orchestrator()
            session = _make_session(ConversationState.WAITING_FOR_ORDER_TYPE)
            orchestrator.resolve(session=session, user_text="pickup")
            mock_nlu.assert_not_called()

    def test_delivery_skips_resolve_nlu(self):
        with patch("app.core.nlu_orchestrator.resolve_nlu") as mock_nlu:
            orchestrator = _make_orchestrator()
            session = _make_session(ConversationState.WAITING_FOR_ORDER_TYPE)
            orchestrator.resolve(session=session, user_text="delivery")
            mock_nlu.assert_not_called()

    def test_pick_up_variant_skips_resolve_nlu(self):
        with patch("app.core.nlu_orchestrator.resolve_nlu") as mock_nlu:
            orchestrator = _make_orchestrator()
            session = _make_session(ConversationState.WAITING_FOR_ORDER_TYPE)
            orchestrator.resolve(session=session, user_text="pick up")
            mock_nlu.assert_not_called()

    # -----------------------------------------------------------------------
    # Fast-path result is compatible with downstream router/handler
    # -----------------------------------------------------------------------

    def test_fast_path_returns_nlu_resolution(self):
        with patch("app.core.nlu_orchestrator.resolve_nlu"):
            orchestrator = _make_orchestrator()
            session = _make_session(ConversationState.WAITING_FOR_ORDER_TYPE)
            result = orchestrator.resolve(session=session, user_text="pickup")
        self.assertIsInstance(result, NluResolution)

    def test_fast_path_nlu_ms_is_zero(self):
        with patch("app.core.nlu_orchestrator.resolve_nlu"):
            orchestrator = _make_orchestrator()
            session = _make_session(ConversationState.WAITING_FOR_ORDER_TYPE)
            result = orchestrator.resolve(session=session, user_text="delivery")
        self.assertEqual(result.nlu_ms, 0.0)

    def test_fast_path_nlu_skipped_flag_set(self):
        with patch("app.core.nlu_orchestrator.resolve_nlu"):
            orchestrator = _make_orchestrator()
            session = _make_session(ConversationState.WAITING_FOR_ORDER_TYPE)
            result = orchestrator.resolve(session=session, user_text="takeout")
        self.assertTrue(result.nlu.nlu_skipped)
        self.assertEqual(result.nlu.nlu_skip_reason, "order_type_lexical_match")

    def test_fast_path_sets_last_nlu_on_context(self):
        with patch("app.core.nlu_orchestrator.resolve_nlu"):
            orchestrator = _make_orchestrator()
            session = _make_session(ConversationState.WAITING_FOR_ORDER_TYPE)
            orchestrator.resolve(session=session, user_text="pickup")
        self.assertIsNotNone(session.conversation_context.last_nlu)

    def test_fast_path_intent_result_is_unknown(self):
        with patch("app.core.nlu_orchestrator.resolve_nlu"):
            orchestrator = _make_orchestrator()
            session = _make_session(ConversationState.WAITING_FOR_ORDER_TYPE)
            result = orchestrator.resolve(session=session, user_text="delivery")
        self.assertEqual(result.intent_result.intent, Intent.UNKNOWN)

    # -----------------------------------------------------------------------
    # Unknown text in WAITING_FOR_ORDER_TYPE → falls through to full NLU
    # -----------------------------------------------------------------------

    def test_unmatched_text_calls_resolve_nlu(self):
        with patch("app.core.nlu_orchestrator.resolve_nlu") as mock_nlu:
            mock_nlu.return_value = MagicMock(
                effective_intent=Intent.UNKNOWN,
                intent_confidence=0.0,
                normalized_text="asdfgh",
                nlu_skipped=False,
                nlu_skip_reason=None,
            )
            orchestrator = _make_orchestrator()
            session = _make_session(ConversationState.WAITING_FOR_ORDER_TYPE)
            orchestrator.resolve(session=session, user_text="asdfgh qwerty")
            mock_nlu.assert_called_once()

    # -----------------------------------------------------------------------
    # Other states → normal NLU path always runs
    # -----------------------------------------------------------------------

    def test_idle_state_always_calls_resolve_nlu(self):
        with patch("app.core.nlu_orchestrator.resolve_nlu") as mock_nlu:
            mock_nlu.return_value = MagicMock(
                effective_intent=Intent.UNKNOWN,
                intent_confidence=0.0,
                normalized_text="pickup",
                nlu_skipped=False,
                nlu_skip_reason=None,
            )
            orchestrator = _make_orchestrator()
            session = _make_session(ConversationState.IDLE)
            orchestrator.resolve(session=session, user_text="pickup")
            mock_nlu.assert_called_once()

    def test_confirming_order_state_calls_resolve_nlu(self):
        with patch("app.core.nlu_orchestrator.resolve_nlu") as mock_nlu:
            mock_nlu.return_value = MagicMock(
                effective_intent=Intent.CONFIRM,
                intent_confidence=1.0,
                normalized_text="yes",
                nlu_skipped=False,
                nlu_skip_reason=None,
            )
            orchestrator = _make_orchestrator()
            session = _make_session(ConversationState.CONFIRMING_ORDER)
            orchestrator.resolve(session=session, user_text="yes")
            mock_nlu.assert_called_once()

    # -----------------------------------------------------------------------
    # Performance: fast-path preprocessing is sub-millisecond
    # -----------------------------------------------------------------------

    def test_fast_path_preprocess_ms_is_non_negative(self):
        with patch("app.core.nlu_orchestrator.resolve_nlu"):
            orchestrator = _make_orchestrator()
            session = _make_session(ConversationState.WAITING_FOR_ORDER_TYPE)
            result = orchestrator.resolve(session=session, user_text="pickup")
        self.assertGreaterEqual(result.preprocess_ms, 0.0)


if __name__ == "__main__":
    unittest.main()
