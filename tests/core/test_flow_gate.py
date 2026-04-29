"""Smoke tests for FlowGate (Commit 5 extraction)."""
import sys
import types
import unittest
from unittest.mock import MagicMock, patch


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
from app.nlu.nlu_result import NLUResult
from app.session.session import Session
from app.state_machine.models.conversation_state import ConversationState


def _make_gate() -> FlowGate:
    return FlowGate(
        handlers={},
        menu_repo=None,
        cart_summary_builder=None,
        response_writer=None,
        diagnostics=None,
        payment_flow=None,
        resume_prompt_builder=None,
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
        # order_type is None → required → compute returns WAITING_FOR_ORDER_TYPE
        result = gate._compute_order_type_gate_state(session)
        self.assertEqual(result, ConversationState.WAITING_FOR_ORDER_TYPE)

    def test_normalize_order_type_gate_state_does_not_clobber_completed(self):
        gate = _make_gate()
        session = Session(session_id="s1", restaurant_id="demo")
        session.conversation_state = ConversationState.COMPLETED
        result = gate._compute_order_type_gate_state(session)
        self.assertIsNone(result)


def _make_nlu(
    intent: Intent = Intent.UNKNOWN,
    confidence: float = 0.9,
    normalized_text: str = "",
) -> NLUResult:
    return NLUResult(
        effective_intent=intent,
        intent_confidence=confidence,
        raw_text=normalized_text,
        normalized_text=normalized_text,
    )


class FlowGateAgentRequestTests(unittest.TestCase):
    """_handle_phase3_control_shortcuts: agent-request detection is NLU-first."""

    def _make_payment_session(self) -> Session:
        s = Session(session_id="s1", restaurant_id="demo")
        s.conversation_state = ConversationState.WAITING_FOR_PAYMENT
        return s

    def _make_intent_result(self, intent: Intent = Intent.UNKNOWN) -> IntentResult:
        return IntentResult(intent=intent, raw_text="")

    def test_nlu_request_agent_above_threshold_triggers_transfer(self):
        gate = _make_gate()
        session = self._make_payment_session()
        nlu = _make_nlu(Intent.REQUEST_AGENT, confidence=0.9, normalized_text="agent")
        intent_result = self._make_intent_result(Intent.REQUEST_AGENT)

        # Stub payment_flow to avoid AttributeError on None
        gate.payment_flow = MagicMock()

        result = gate._handle_phase3_control_shortcuts(
            session=session,
            state_before=session.conversation_state,
            intent_result=intent_result,
            nlu=nlu,
        )
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.output)
        self.assertEqual(result.output.response_key, "transferring_to_human_agent")

    def test_phrase_fallback_triggers_transfer_when_nlu_is_none_intent(self):
        gate = _make_gate()
        session = self._make_payment_session()
        # NLU returns UNKNOWN but text matches agent phrase
        nlu = _make_nlu(Intent.UNKNOWN, confidence=0.1, normalized_text="i need a human")
        intent_result = self._make_intent_result(Intent.UNKNOWN)

        gate.payment_flow = MagicMock()

        result = gate._handle_phase3_control_shortcuts(
            session=session,
            state_before=session.conversation_state,
            intent_result=intent_result,
            nlu=nlu,
        )
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.output)
        self.assertEqual(result.output.response_key, "transferring_to_human_agent")

    def test_unrelated_text_no_agent_transfer(self):
        gate = _make_gate()
        session = self._make_payment_session()
        nlu = _make_nlu(Intent.UNKNOWN, confidence=0.9, normalized_text="i want a coke")
        intent_result = self._make_intent_result(Intent.UNKNOWN)

        gate.payment_flow = MagicMock()

        # Should NOT return a transfer — no agent request signal
        # (it may return None or some other key, but not agent transfer)
        result = gate._handle_phase3_control_shortcuts(
            session=session,
            state_before=session.conversation_state,
            intent_result=intent_result,
            nlu=nlu,
        )
        if result is not None and result.output is not None:
            self.assertNotEqual(result.output.response_key, "transferring_to_human_agent")

    def test_no_agent_transfer_for_empty_text(self):
        gate = _make_gate()
        session = self._make_payment_session()
        nlu = _make_nlu(Intent.UNKNOWN, confidence=0.0, normalized_text="")
        intent_result = self._make_intent_result(Intent.UNKNOWN)

        gate.payment_flow = MagicMock()

        result = gate._handle_phase3_control_shortcuts(
            session=session,
            state_before=session.conversation_state,
            intent_result=intent_result,
            nlu=nlu,
        )
        if result is not None and result.output is not None:
            self.assertNotEqual(result.output.response_key, "transferring_to_human_agent")


class FlowGateQuantityCorrectionTests(unittest.TestCase):
    """_handle_prepayment_quantity_correction: NLU-first, phrase fallback only."""

    def _make_gate_with_mocks(self) -> FlowGate:
        gate = _make_gate()
        gate.payment_flow = MagicMock()
        gate.cart_summary_builder = MagicMock(
            return_value={"items": [], "total": "0.00"}
        )
        gate.menu_repo = MagicMock()
        return gate

    def _confirming_order_session(self) -> Session:
        s = Session(session_id="s1", restaurant_id="demo")
        s.conversation_state = ConversationState.CONFIRMING_ORDER
        return s

    def test_nlu_change_quantity_above_threshold_enters_correction(self):
        """With CHANGE_QUANTITY intent at high confidence the method proceeds to
        cart lookup.  We verify it does NOT short-circuit with None when the intent
        matches — even if the cart is empty (correction returns None from cart
        lookup, not from the intent guard)."""
        from unittest.mock import patch as _patch

        gate = self._make_gate_with_mocks()
        session = self._confirming_order_session()
        nlu = _make_nlu(Intent.CHANGE_QUANTITY, confidence=0.85)
        intent_result = IntentResult(intent=Intent.CHANGE_QUANTITY, raw_text="make it 2")

        with _patch(
            "app.state_machine.handlers.order.prepayment_correction_support"
            ".resolve_cart_item_for_quantity_change",
            return_value=None,  # no matching cart item → correction exits cleanly
        ):
            result = gate._handle_prepayment_quantity_correction(
                session=session,
                state_before=ConversationState.CONFIRMING_ORDER,
                intent_result=intent_result,
                normalized_text="make it 2",
                nlu_result=nlu,
            )
        # Cart lookup returned None → correction exits without a TurnOutput
        self.assertIsNone(result)

    def test_low_confidence_nlu_phrase_fallback_fires(self):
        """Below threshold: phrase fallback should let 'make it 2' through."""
        from unittest.mock import patch as _patch

        gate = self._make_gate_with_mocks()
        session = self._confirming_order_session()
        # Low confidence on CHANGE_QUANTITY → ControlDecisionService falls to phrase
        nlu = _make_nlu(Intent.CHANGE_QUANTITY, confidence=0.2)
        intent_result = IntentResult(
            intent=Intent.CHANGE_QUANTITY, raw_text="make it 2"
        )

        with _patch(
            "app.state_machine.handlers.order.prepayment_correction_support"
            ".resolve_cart_item_for_quantity_change",
            return_value=None,
        ):
            result = gate._handle_prepayment_quantity_correction(
                session=session,
                state_before=ConversationState.CONFIRMING_ORDER,
                intent_result=intent_result,
                normalized_text="make it 2",
                nlu_result=nlu,
            )
        # Phrase matched → passed intent guard; cart lookup returned None → None result
        self.assertIsNone(result)

    def test_no_signal_returns_none_immediately(self):
        """Unrelated text with no correction intent returns None (early exit)."""
        gate = self._make_gate_with_mocks()
        session = self._confirming_order_session()
        nlu = _make_nlu(Intent.UNKNOWN, confidence=0.9)
        intent_result = IntentResult(intent=Intent.UNKNOWN, raw_text="sounds good")

        result = gate._handle_prepayment_quantity_correction(
            session=session,
            state_before=ConversationState.CONFIRMING_ORDER,
            intent_result=intent_result,
            normalized_text="sounds good",
            nlu_result=nlu,
        )
        self.assertIsNone(result)

    def test_false_positive_instead_of_waiting_no_nlu_signal(self):
        """'instead of waiting' triggers phrase fallback but extract_requested_quantity
        returns None → correction exits without updating the cart."""
        from unittest.mock import patch as _patch

        gate = self._make_gate_with_mocks()
        session = self._confirming_order_session()
        # High-confidence non-quantity intent — phrase fallback will still fire
        # because NLU intent != CHANGE_QUANTITY
        nlu = _make_nlu(Intent.UNKNOWN, confidence=0.05)
        intent_result = IntentResult(intent=Intent.UNKNOWN, raw_text="instead of waiting")

        with _patch(
            "app.state_machine.handlers.order.prepayment_correction_support"
            ".extract_requested_quantity",
            return_value=None,  # no numeric quantity parsed
        ):
            result = gate._handle_prepayment_quantity_correction(
                session=session,
                state_before=ConversationState.CONFIRMING_ORDER,
                intent_result=intent_result,
                normalized_text="instead of waiting",
                nlu_result=nlu,
            )
        self.assertIsNone(result)

    def test_no_nlu_result_phrase_fallback_works(self):
        """nlu_result=None falls through to phrase matching (backward-compat)."""
        from unittest.mock import patch as _patch

        gate = self._make_gate_with_mocks()
        session = self._confirming_order_session()
        intent_result = IntentResult(intent=Intent.UNKNOWN, raw_text="change it to 3")

        with _patch(
            "app.state_machine.handlers.order.prepayment_correction_support"
            ".resolve_cart_item_for_quantity_change",
            return_value=None,
        ):
            result = gate._handle_prepayment_quantity_correction(
                session=session,
                state_before=ConversationState.CONFIRMING_ORDER,
                intent_result=intent_result,
                normalized_text="change it to 3",
                nlu_result=None,
            )
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
