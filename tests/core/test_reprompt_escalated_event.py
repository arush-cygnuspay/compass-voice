"""Smoke test for the reprompt_escalated log event emitted by
TurnEngine._apply_reprompt_guardrail.
"""
import logging
import sys
import types
import unittest


# ── Stub heavy ML imports before turn_engine pulls them in ─────────
_intent_module = types.ModuleType("app.ml.intent.inference_intent")
_slot_module = types.ModuleType("app.ml.slot.inference_slot")


class _IntentBundle:
    pass


class _SlotBundle:
    pass


_intent_module.IntentBundle = _IntentBundle
_intent_module.predict_intent = lambda *args, **kwargs: []
_slot_module.SlotBundle = _SlotBundle
_slot_module.predict_slots = lambda *args, **kwargs: []
sys.modules.setdefault("app.ml.intent.inference_intent", _intent_module)
sys.modules.setdefault("app.ml.slot.inference_slot", _slot_module)


# Twilio/redis stubs match the convention used by other turn_engine tests.
for _name in ("twilio", "twilio.base", "twilio.base.exceptions", "twilio.rest"):
    sys.modules.setdefault(_name, types.ModuleType(_name))
sys.modules["twilio.base.exceptions"].TwilioRestException = Exception
sys.modules["twilio.rest"].Client = type("_Client", (), {"__init__": lambda *a, **k: None})

_redis_module = types.ModuleType("redis")
_redis_module.Redis = type("_Redis", (), {"__init__": lambda *a, **k: None})
sys.modules.setdefault("redis", _redis_module)


from app.core.handler_dispatcher import HandlerDispatcher
from app.core.turn_diagnostics import TurnDiagnostics
from app.core.turn_engine import TurnEngine
from app.session.session import Session
from app.state_machine.handler_result import HandlerResult
from app.state_machine.models.conversation_state import ConversationState


class _LogCapture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.events: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        if record.getMessage() == "reprompt_escalated":
            self.events.append(record)


class _StubDiagnostics:
    """Stand-in TurnDiagnostics carrying only the helper method
    ``_apply_reprompt_guardrail`` reads from ``self.diagnostics``."""

    def _infer_prompt_field_for_response(
        self,
        *,
        response_key: str,
        session: Session,
    ) -> str:
        return TurnDiagnostics._infer_prompt_field_for_response(
            self, response_key=response_key, session=session
        )


class _StubEngine:
    """Minimal stand-in carrying only the helper method
    ``_apply_reprompt_guardrail`` needs from `self`."""

    def __init__(self) -> None:
        self.diagnostics = _StubDiagnostics()


def _make_session(state: ConversationState, field: str, prior_count: int) -> Session:
    session = Session(session_id="test-1", restaurant_id="steves_grill")
    session.conversation_state = state
    session.reprompt_count_by_field = {field: prior_count}
    return session


class RepromptEscalatedEventTests(unittest.TestCase):
    def setUp(self) -> None:
        self.capture = _LogCapture()
        self.logger = logging.getLogger("app.state_machine.control_intent_resolver")
        self.previous_level = self.logger.level
        self.logger.addHandler(self.capture)
        self.logger.setLevel(logging.INFO)

    def tearDown(self) -> None:
        self.logger.removeHandler(self.capture)
        self.logger.setLevel(self.previous_level)

    def test_modifier_escalation_emits_reprompt_escalated(self):
        engine = _StubEngine()
        session = _make_session(
            ConversationState.WAITING_FOR_MODIFIER,
            "modifier",
            prior_count=2,
        )
        result = HandlerResult(
            next_state=ConversationState.WAITING_FOR_MODIFIER,
            response_key="repeat_modifier_options",
            response_payload={"group_name": "Mods"},
        )

        out = HandlerDispatcher._apply_reprompt_guardrail(
            engine,
            session=session,
            state_before=ConversationState.WAITING_FOR_MODIFIER,
            result=result,
        )

        # response_key gets swapped on escalation.
        self.assertEqual(out.response_key, "list_modifier_options")
        self.assertTrue(out.response_payload["reprompt_escalation"])
        self.assertEqual(len(self.capture.events), 1)
        record = self.capture.events[0]
        self.assertEqual(getattr(record, "field", None), "modifier")
        self.assertEqual(getattr(record, "attempts", None), 3)
        self.assertEqual(
            getattr(record, "original_response_key", None),
            "repeat_modifier_options",
        )
        self.assertEqual(
            getattr(record, "escalated_response_key", None),
            "list_modifier_options",
        )
        self.assertEqual(
            getattr(record, "state", None),
            ConversationState.WAITING_FOR_MODIFIER.value,
        )

    def test_below_threshold_does_not_emit_reprompt_escalated(self):
        engine = _StubEngine()
        session = _make_session(
            ConversationState.WAITING_FOR_SIDE,
            "side",
            prior_count=1,  # next_count = 2, still < 3
        )
        result = HandlerResult(
            next_state=ConversationState.WAITING_FOR_SIDE,
            response_key="repeat_side_options",
        )

        out = HandlerDispatcher._apply_reprompt_guardrail(
            engine,
            session=session,
            state_before=ConversationState.WAITING_FOR_SIDE,
            result=result,
        )

        self.assertEqual(len(self.capture.events), 0)
        # reprompt_count must be stamped even when escalation threshold not yet hit.
        self.assertEqual(out.response_payload.get("reprompt_count"), 2)
        self.assertNotIn("reprompt_escalation", out.response_payload)

    def test_first_miss_stamps_reprompt_count_1(self):
        engine = _StubEngine()
        session = _make_session(
            ConversationState.WAITING_FOR_SIZE,
            "size",
            prior_count=0,
        )
        result = HandlerResult(
            next_state=ConversationState.WAITING_FOR_SIZE,
            response_key="repeat_size_options",
        )

        out = HandlerDispatcher._apply_reprompt_guardrail(
            engine,
            session=session,
            state_before=ConversationState.WAITING_FOR_SIZE,
            result=result,
        )

        self.assertEqual(out.response_payload.get("reprompt_count"), 1)
        self.assertNotIn("reprompt_escalation", out.response_payload)
        self.assertEqual(len(self.capture.events), 0)

    def test_size_escalation_stamps_reprompt_count_and_escalation_flag(self):
        engine = _StubEngine()
        session = _make_session(
            ConversationState.WAITING_FOR_SIZE,
            "size",
            prior_count=2,
        )
        result = HandlerResult(
            next_state=ConversationState.WAITING_FOR_SIZE,
            response_key="repeat_size_options",
            response_payload={"item_name": "Burger"},
        )

        out = HandlerDispatcher._apply_reprompt_guardrail(
            engine,
            session=session,
            state_before=ConversationState.WAITING_FOR_SIZE,
            result=result,
        )

        self.assertEqual(out.response_payload.get("reprompt_count"), 3)
        self.assertTrue(out.response_payload.get("reprompt_escalation"))
        # size field keeps its own key on escalation
        self.assertEqual(out.response_key, "repeat_size_options")


if __name__ == "__main__":
    unittest.main()
