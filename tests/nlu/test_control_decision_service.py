# tests/nlu/test_control_decision_service.py
"""Unit tests for ControlDecisionService and ControlDecision contract."""
import logging
import unittest
from unittest.mock import patch

from app.nlu.control_decision_service import ControlDecision, ControlDecisionService
from app.nlu.fallback_phrase_matcher import FallbackPhraseMatcher
from app.nlu.intent_resolution.intent import Intent
from app.nlu.nlu_result import NLUResult


def _make_nlu(intent: Intent, confidence: float = 0.9) -> NLUResult:
    return NLUResult(
        effective_intent=intent,
        intent_confidence=confidence,
        raw_text="",
        normalized_text="",
    )


class TestControlDecisionContract(unittest.TestCase):
    """Verify ControlDecision dataclass fields are present and immutable."""

    def test_has_required_fields(self):
        d = ControlDecision(intent=None, confidence=0.0, used_fallback=False)
        self.assertIsNone(d.intent)
        self.assertEqual(d.confidence, 0.0)
        self.assertFalse(d.used_fallback)
        self.assertIsNone(d.fallback_source)

    def test_is_frozen(self):
        d = ControlDecision(
            intent=Intent.REQUEST_AGENT,
            confidence=0.9,
            used_fallback=False,
        )
        with self.assertRaises((AttributeError, TypeError)):
            d.intent = None  # type: ignore[misc]


class TestControlDecisionServiceAgentRequest(unittest.TestCase):
    def setUp(self):
        self.svc = ControlDecisionService(confidence_threshold=0.55)

    # --- NLU path -------------------------------------------------------------

    def test_nlu_request_agent_above_threshold(self):
        nlu = _make_nlu(Intent.REQUEST_AGENT, confidence=0.9)
        d = self.svc.resolve_agent_request("agent please", nlu)
        self.assertEqual(d.intent, Intent.REQUEST_AGENT)
        self.assertEqual(d.confidence, 0.9)
        self.assertFalse(d.used_fallback)
        self.assertIsNone(d.fallback_source)

    def test_nlu_request_agent_at_threshold(self):
        nlu = _make_nlu(Intent.REQUEST_AGENT, confidence=0.55)
        d = self.svc.resolve_agent_request("agent", nlu)
        self.assertEqual(d.intent, Intent.REQUEST_AGENT)
        self.assertFalse(d.used_fallback)

    def test_nlu_request_agent_below_threshold_falls_back_to_phrase(self):
        # Low confidence NLU + phrase match → fallback fires
        nlu = _make_nlu(Intent.REQUEST_AGENT, confidence=0.3)
        d = self.svc.resolve_agent_request("agent", nlu)
        self.assertEqual(d.intent, Intent.REQUEST_AGENT)
        self.assertTrue(d.used_fallback)
        self.assertEqual(d.fallback_source, "agent_request_phrase")

    def test_nlu_unrelated_intent_phrase_match_fires_fallback(self):
        nlu = _make_nlu(Intent.UNKNOWN, confidence=0.9)
        d = self.svc.resolve_agent_request("I need a human", nlu)
        self.assertEqual(d.intent, Intent.REQUEST_AGENT)
        self.assertTrue(d.used_fallback)

    def test_nlu_unrelated_intent_no_phrase_match_returns_none(self):
        nlu = _make_nlu(Intent.ADD_ITEM, confidence=0.95)
        d = self.svc.resolve_agent_request("I want a burger", nlu)
        self.assertIsNone(d.intent)
        self.assertFalse(d.used_fallback)

    # --- No NLU (phrase-only callers) -----------------------------------------

    def test_no_nlu_phrase_match(self):
        d = self.svc.resolve_agent_request("agent")
        self.assertEqual(d.intent, Intent.REQUEST_AGENT)
        self.assertTrue(d.used_fallback)
        self.assertEqual(d.confidence, 0.0)

    def test_no_nlu_no_phrase_match(self):
        d = self.svc.resolve_agent_request("I want a coke")
        self.assertIsNone(d.intent)
        self.assertFalse(d.used_fallback)

    def test_empty_text_no_nlu(self):
        d = self.svc.resolve_agent_request("")
        self.assertIsNone(d.intent)

    # --- Fallback logging -----------------------------------------------------

    def test_fallback_emits_log_event(self):
        with patch.object(
            ControlDecisionService, "_emit_fallback"
        ) as mock_log:
            nlu = _make_nlu(Intent.UNKNOWN, confidence=0.1)
            self.svc.resolve_agent_request("human please", nlu)
            mock_log.assert_called_once_with("agent_request_phrase", "human please")

    def test_no_fallback_log_when_nlu_wins(self):
        with patch.object(
            ControlDecisionService, "_emit_fallback"
        ) as mock_log:
            nlu = _make_nlu(Intent.REQUEST_AGENT, confidence=0.8)
            self.svc.resolve_agent_request("agent", nlu)
            mock_log.assert_not_called()


class TestControlDecisionServiceQuantityCorrection(unittest.TestCase):
    def setUp(self):
        self.svc = ControlDecisionService(confidence_threshold=0.55)

    # --- NLU path -------------------------------------------------------------

    def test_nlu_change_quantity_above_threshold(self):
        nlu = _make_nlu(Intent.CHANGE_QUANTITY, confidence=0.8)
        d = self.svc.resolve_quantity_correction("make it 2", nlu)
        self.assertEqual(d.intent, Intent.CHANGE_QUANTITY)
        self.assertFalse(d.used_fallback)

    def test_nlu_change_quantity_below_threshold_phrase_fires(self):
        nlu = _make_nlu(Intent.CHANGE_QUANTITY, confidence=0.3)
        d = self.svc.resolve_quantity_correction("make it 2", nlu)
        self.assertEqual(d.intent, Intent.CHANGE_QUANTITY)
        self.assertTrue(d.used_fallback)
        self.assertEqual(d.fallback_source, "quantity_correction_phrase")

    def test_nlu_unrelated_intent_phrase_fires(self):
        nlu = _make_nlu(Intent.UNKNOWN, confidence=0.9)
        d = self.svc.resolve_quantity_correction("2 instead of 1", nlu)
        self.assertEqual(d.intent, Intent.CHANGE_QUANTITY)
        self.assertTrue(d.used_fallback)

    def test_nlu_unrelated_intent_no_phrase_returns_none(self):
        nlu = _make_nlu(Intent.ADD_ITEM, confidence=0.9)
        d = self.svc.resolve_quantity_correction("give me a burger", nlu)
        self.assertIsNone(d.intent)

    # --- No NLU ---------------------------------------------------------------

    def test_no_nlu_phrase_match_instead_of(self):
        d = self.svc.resolve_quantity_correction("3 instead of 2")
        self.assertEqual(d.intent, Intent.CHANGE_QUANTITY)
        self.assertTrue(d.used_fallback)

    def test_no_nlu_phrase_match_make_it(self):
        d = self.svc.resolve_quantity_correction("make it 4")
        self.assertEqual(d.intent, Intent.CHANGE_QUANTITY)
        self.assertTrue(d.used_fallback)

    def test_no_nlu_no_phrase_returns_none(self):
        d = self.svc.resolve_quantity_correction("I want a shake")
        self.assertIsNone(d.intent)

    def test_empty_text_returns_none(self):
        d = self.svc.resolve_quantity_correction("")
        self.assertIsNone(d.intent)

    # --- False-positive guard --------------------------------------------------

    def test_instead_of_waiting_triggers_fallback(self):
        # "instead of waiting" hits the phrase fallback (low NLU confidence
        # scenario). The flow layer (ControlDecisionService consumer) is
        # responsible for discarding the result when NLU clearly has a
        # different high-confidence intent.
        nlu = _make_nlu(Intent.UNKNOWN, confidence=0.05)
        d = self.svc.resolve_quantity_correction("instead of waiting", nlu)
        self.assertEqual(d.intent, Intent.CHANGE_QUANTITY)
        self.assertTrue(d.used_fallback)

    def test_instead_of_waiting_nlu_wins_with_high_confidence_different_intent(self):
        # High-confidence non-CHANGE_QUANTITY intent: phrase fallback still
        # fires because NLU didn't emit CHANGE_QUANTITY. Caller must handle.
        nlu = _make_nlu(Intent.ADD_ITEM, confidence=0.95)
        d = self.svc.resolve_quantity_correction("instead of waiting", nlu)
        # Phrase matched regardless — fallback fires
        self.assertEqual(d.intent, Intent.CHANGE_QUANTITY)
        self.assertTrue(d.used_fallback)

    # --- Fallback logging -----------------------------------------------------

    def test_fallback_emits_log_event(self):
        with patch.object(
            ControlDecisionService, "_emit_fallback"
        ) as mock_log:
            self.svc.resolve_quantity_correction("make it 2")
            mock_log.assert_called_once_with("quantity_correction_phrase", "make it 2")

    def test_no_fallback_log_when_nlu_wins(self):
        with patch.object(
            ControlDecisionService, "_emit_fallback"
        ) as mock_log:
            nlu = _make_nlu(Intent.CHANGE_QUANTITY, confidence=0.9)
            self.svc.resolve_quantity_correction("make it 2", nlu)
            mock_log.assert_not_called()


class TestControlDecisionServiceEmitFallbackFormat(unittest.TestCase):
    """Verify the fallback_hit log record has the required keys."""

    def test_emit_fallback_log_record_keys(self):
        records: list[logging.LogRecord] = []

        class Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        import app.nlu.control_decision_service as _mod

        handler = Capture()
        _mod.logger.addHandler(handler)
        _mod.logger.setLevel(logging.INFO)
        try:
            ControlDecisionService._emit_fallback("agent_request_phrase", "human")
        finally:
            _mod.logger.removeHandler(handler)

        self.assertEqual(len(records), 1)
        rec = records[0]
        self.assertEqual(rec.event_name, "fallback_hit")  # type: ignore[attr-defined]
        self.assertEqual(rec.source, "agent_request_phrase")  # type: ignore[attr-defined]
        self.assertEqual(rec.text, "human")  # type: ignore[attr-defined]


if __name__ == "__main__":
    unittest.main()
