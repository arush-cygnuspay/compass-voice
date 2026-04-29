"""Smoke tests for SessionResponseWriter (Commit 2 extraction)."""
import sys
import types
import unittest


# Stub heavy ML imports before turn_engine is loaded.
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

for _name in ("twilio", "twilio.base", "twilio.base.exceptions", "twilio.rest"):
    sys.modules.setdefault(_name, types.ModuleType(_name))
sys.modules["twilio.base.exceptions"].TwilioRestException = Exception
sys.modules["twilio.rest"].Client = type("_Client", (), {"__init__": lambda *a, **k: None})

_redis_module = types.ModuleType("redis")
_redis_module.Redis = type("_Redis", (), {"__init__": lambda *a, **k: None})
sys.modules.setdefault("redis", _redis_module)


from app.core.session_response_writer import SessionResponseWriter
from app.core.turn_engine import TurnOutput
from app.nlu.intent_resolution.intent import Intent
from app.session.session import Session
from app.state_machine.models.conversation_state import ConversationState


class _StubResponder:
    def build(self, *, response_key, context, payload=None):
        return f"text for {response_key}"


class _StubMenuRepo:
    pass


class SessionResponseWriterApplyTests(unittest.TestCase):
    def test_apply_session_response_writes_session_state(self):
        writer = SessionResponseWriter(_StubResponder(), _StubMenuRepo())
        session = Session(session_id="s1", restaurant_id="demo")
        session.turn_count = 4

        writer._apply_session_response(
            session=session,
            intent=Intent.AFFIRM,
            response_key="confirm_order_summary",
            response_payload={"k": "v"},
        )

        self.assertEqual(session.last_intent, Intent.AFFIRM)
        self.assertEqual(session.last_response_key, "confirm_order_summary")
        self.assertEqual(session.last_response_payload, {"k": "v"})
        self.assertEqual(session.turn_count, 5)
        self.assertIsNotNone(session.last_response_at_epoch)

    def test_session_round_trip_preserves_last_response_timestamp(self):
        session = Session(session_id="s3", restaurant_id="demo")
        session.last_response_at_epoch = 1712345678.9

        restored = Session.from_dict(session.to_dict())

        self.assertEqual(restored.last_response_at_epoch, 1712345678.9)

    def test_build_silent_output_returns_empty_text_fields(self):
        out = SessionResponseWriter._build_silent_output(
            response_key="silence",
            response_payload=None,
            end_call_after_playback=False,
            transfer_call_to_number=None,
        )
        self.assertIsInstance(out, TurnOutput)
        self.assertEqual(out.internal_response_text, "")
        self.assertEqual(out.spoken_response_text, "")

    def test_normalize_response_text_collapses_whitespace(self):
        self.assertEqual(
            SessionResponseWriter._normalize_response_text("  hello   world  "),
            "hello world",
        )
        self.assertEqual(SessionResponseWriter._normalize_response_text(None), "")

    def test_hydrate_output_round_trip_fills_text_fields(self):
        writer = SessionResponseWriter(_StubResponder(), _StubMenuRepo())
        session = Session(session_id="s2", restaurant_id="demo")
        session.conversation_state = ConversationState.IDLE

        skeleton = TurnOutput(
            response_key="ask_for_quantity",
            response_payload=None,
            internal_response_text="",
            spoken_response_text="",
        )

        hydrated = writer._hydrate_output(session=session, output=skeleton)

        self.assertEqual(hydrated.response_key, "ask_for_quantity")
        self.assertEqual(hydrated.internal_response_text, "text for ask_for_quantity")
        self.assertEqual(hydrated.spoken_response_text, "text for ask_for_quantity")


if __name__ == "__main__":
    unittest.main()

