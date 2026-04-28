"""Smoke tests for FlowGate (Commit 5 extraction)."""
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

for _name in ("twilio", "twilio.base", "twilio.base.exceptions", "twilio.rest"):
    sys.modules.setdefault(_name, types.ModuleType(_name))
sys.modules["twilio.base.exceptions"].TwilioRestException = Exception
sys.modules["twilio.rest"].Client = type("_Client", (), {"__init__": lambda *a, **k: None})
_redis_module = types.ModuleType("redis")
_redis_module.Redis = type("_Redis", (), {"__init__": lambda *a, **k: None})
sys.modules.setdefault("redis", _redis_module)


from app.core.flow_gate import FlowGate
from app.nlu.intent_resolution.intent import Intent
from app.nlu.intent_resolution.intent_result import IntentResult
from app.session.session import Session
from app.state_machine.models.conversation_state import ConversationState


class _StubLogger:
    enabled = False


def _make_gate() -> FlowGate:
    return FlowGate(
        handlers={},
        menu_repo=None,
        cart_summary_builder=None,
        response_writer=None,
        diagnostics=None,
        payment_flow=None,
        resume_prompt_builder=None,
        nlu_logger=_StubLogger(),
    )


class FlowGateRewriteTests(unittest.TestCase):
    def test_apply_idle_shortcuts_returns_unchanged_when_state_not_idle(self):
        gate = _make_gate()
        session = Session(session_id="s1", restaurant_id="demo")
        session.conversation_state = ConversationState.WAITING_FOR_MODIFIER
        intent_result = IntentResult(intent=Intent.DENY, raw_text="no")

        rewritten, output = gate._apply_idle_shortcuts(session, intent_result)
        self.assertIs(rewritten, intent_result)
        self.assertIsNone(output)

    def test_rewrite_confirming_order_to_idle_for_browse_intent(self):
        gate = _make_gate()
        session = Session(session_id="s1", restaurant_id="demo")
        session.conversation_state = ConversationState.CONFIRMING_ORDER

        new_state = gate._rewrite_confirming_order_to_idle_if_needed(
            session=session,
            intent=Intent.BROWSE_MENU,
        )
        self.assertEqual(new_state, ConversationState.IDLE)

    def test_rewrite_confirming_order_keeps_state_for_unrelated_intent(self):
        gate = _make_gate()
        session = Session(session_id="s1", restaurant_id="demo")
        session.conversation_state = ConversationState.CONFIRMING_ORDER

        new_state = gate._rewrite_confirming_order_to_idle_if_needed(
            session=session,
            intent=Intent.AFFIRM,
        )
        self.assertEqual(new_state, ConversationState.CONFIRMING_ORDER)


class FlowGateOrderTypeGateTests(unittest.TestCase):
    def test_order_type_required_true_for_unset_order_type(self):
        gate = _make_gate()
        session = Session(session_id="s1", restaurant_id="demo")
        # default conversation_context.order_type is None
        self.assertTrue(gate._order_type_required(session))

    def test_order_type_required_false_for_pickup_or_delivery(self):
        gate = _make_gate()
        session = Session(session_id="s1", restaurant_id="demo")
        session.conversation_context.order_type = "pickup"
        self.assertFalse(gate._order_type_required(session))
        session.conversation_context.order_type = "delivery"
        self.assertFalse(gate._order_type_required(session))

    def test_normalize_order_type_gate_state_routes_to_waiting_state(self):
        gate = _make_gate()
        session = Session(session_id="s1", restaurant_id="demo")
        session.conversation_state = ConversationState.IDLE
        # order_type is None → required → state should flip
        gate._normalize_order_type_gate_state(session)
        self.assertEqual(
            session.conversation_state, ConversationState.WAITING_FOR_ORDER_TYPE
        )

    def test_normalize_order_type_gate_state_does_not_clobber_completed(self):
        gate = _make_gate()
        session = Session(session_id="s1", restaurant_id="demo")
        session.conversation_state = ConversationState.COMPLETED
        gate._normalize_order_type_gate_state(session)
        self.assertEqual(session.conversation_state, ConversationState.COMPLETED)


if __name__ == "__main__":
    unittest.main()
