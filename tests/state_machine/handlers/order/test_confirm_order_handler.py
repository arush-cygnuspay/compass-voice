import unittest
import sys
import types


twilio_module = types.ModuleType("twilio")
twilio_base_module = types.ModuleType("twilio.base")
twilio_base_exceptions_module = types.ModuleType("twilio.base.exceptions")
twilio_rest_module = types.ModuleType("twilio.rest")


class _TwilioRestException(Exception):
    pass


class _TwilioClient:
    def __init__(self, *args, **kwargs):
        pass


twilio_base_exceptions_module.TwilioRestException = _TwilioRestException
twilio_rest_module.Client = _TwilioClient

sys.modules.setdefault("twilio", twilio_module)
sys.modules.setdefault("twilio.base", twilio_base_module)
sys.modules.setdefault("twilio.base.exceptions", twilio_base_exceptions_module)
sys.modules.setdefault("twilio.rest", twilio_rest_module)

from app.cart.cart_item import CartItem
from app.nlu.intent_resolution.intent import Intent
from app.nlu.nlu_result import NLUResult
from app.session.session import Session
from app.state_machine.handlers.order.confirm_order_handler import ConfirmOrderHandler
from app.state_machine.models.conversation_context import ConversationContext
from app.state_machine.models.conversation_state import ConversationState


class StubCartSummaryBuilder:
    def build(self, cart):
        return {"items": [{"name": "Burger", "quantity": 1}], "total": "$10.00"}


class StubSmsService:
    def is_configured(self):
        return False


class StubCheckoutSession:
    def __init__(self, order_number: str, token: str):
        self.order_number = order_number
        self.token = token


class StubCheckoutService:
    def create_session(self, **kwargs):
        return StubCheckoutSession(order_number="ORD-123", token="checkout-token")

    def build_checkout_url(self, token: str) -> str:
        return f"https://checkout.example/{token}"


def _make_session() -> Session:
    session = Session(session_id="ui-10", restaurant_id="demo")
    session.conversation_state = ConversationState.CONFIRMING_ORDER
    session.cart.add_item(
        CartItem.create(
            item_id="burger",
            quantity=1,
            variant_id=None,
            sides={},
            side_variants={},
            modifiers={},
        )
    )
    return session


def _set_last_nlu(session: Session, intent: Intent, confidence: float = 0.2) -> None:
    session.conversation_context.set_last_nlu(
        user_text="",
        nlu=NLUResult(
            effective_intent=intent,
            intent_confidence=confidence,
            raw_text="",
            normalized_text="",
        ),
    )


def _make_context() -> ConversationContext:
    context = ConversationContext()
    context.order_type = "delivery"
    context.caller_device_type = "chat"
    context.delivery_address.area = "Downtown"
    context.delivery_address.postal_code = "12345"
    context.delivery_address.customer_phone_number = "+923204711572"
    return context


class ConfirmOrderHandlerTests(unittest.TestCase):
    def test_chat_delivery_can_create_checkout_link_without_sms_gate(self):
        handler = ConfirmOrderHandler(
            cart_summary_builder=StubCartSummaryBuilder(),
            sms_service=StubSmsService(),
            checkout_service=StubCheckoutService(),
        )

        session = _make_session()
        context = _make_context()

        result = handler.handle(
            intent=Intent.CHECKOUT,
            context=context,
            user_text="checkout",
            session=session,
        )

        self.assertEqual(result.next_state, ConversationState.WAITING_FOR_CHECKOUT_COMPLETION)
        self.assertEqual(result.response_key, "checkout_link_sent")
        self.assertEqual(result.response_payload, {"order_number": "ORD-123"})
        self.assertEqual(context.delivery_address.order_number, "ORD-123")
        self.assertEqual(
            context.delivery_address.address_form_link,
            "https://checkout.example/checkout-token",
        )

    def test_natural_affirmation_can_start_checkout(self):
        handler = ConfirmOrderHandler(
            cart_summary_builder=StubCartSummaryBuilder(),
            sms_service=StubSmsService(),
            checkout_service=StubCheckoutService(),
        )

        session = _make_session()
        _set_last_nlu(session, Intent.UNKNOWN)
        context = _make_context()

        result = handler.handle(
            intent=Intent.UNKNOWN,
            context=context,
            user_text="go ahead",
            session=session,
        )

        self.assertEqual(result.next_state, ConversationState.WAITING_FOR_CHECKOUT_COMPLETION)
        self.assertEqual(result.response_key, "checkout_link_sent")

    def test_deny_routes_to_change_recovery_instead_of_cancelling_order(self):
        handler = ConfirmOrderHandler(
            cart_summary_builder=StubCartSummaryBuilder(),
            sms_service=StubSmsService(),
            checkout_service=StubCheckoutService(),
        )

        session = _make_session()
        _set_last_nlu(session, Intent.UNKNOWN)
        context = _make_context()

        result = handler.handle(
            intent=Intent.UNKNOWN,
            context=context,
            user_text="change it",
            session=session,
        )

        self.assertEqual(result.next_state, ConversationState.IDLE)
        self.assertEqual(result.response_key, "order_confirmation_declined")


    def test_not_correct_routes_to_correction_flow(self):
        handler = ConfirmOrderHandler(
            cart_summary_builder=StubCartSummaryBuilder(),
            sms_service=StubSmsService(),
            checkout_service=StubCheckoutService(),
        )

        session = _make_session()
        context = _make_context()

        result = handler.handle(
            intent=Intent.UNKNOWN,
            context=context,
            user_text="not correct",
            session=session,
        )

        self.assertEqual(result.next_state, ConversationState.IDLE)
        self.assertEqual(result.response_key, "order_confirmation_declined")

    def test_remove_item_intent_routes_to_correction_flow(self):
        handler = ConfirmOrderHandler(
            cart_summary_builder=StubCartSummaryBuilder(),
            sms_service=StubSmsService(),
            checkout_service=StubCheckoutService(),
        )

        session = _make_session()
        context = _make_context()

        result = handler.handle(
            intent=Intent.REMOVE_ITEM,
            context=context,
            user_text="remove the cake",
            session=session,
        )

        self.assertEqual(result.next_state, ConversationState.IDLE)
        self.assertEqual(result.response_key, "order_confirmation_declined")

    def test_cancel_at_order_confirmation_asks_if_whole_order_should_be_cancelled(self):
        handler = ConfirmOrderHandler(
            cart_summary_builder=StubCartSummaryBuilder(),
            sms_service=StubSmsService(),
            checkout_service=StubCheckoutService(),
        )

        session = _make_session()
        context = _make_context()

        result = handler.handle(
            intent=Intent.UNKNOWN,
            context=context,
            user_text="cancel",
            session=session,
        )

        self.assertEqual(result.next_state, ConversationState.CONFIRMING_ORDER)
        self.assertEqual(result.response_key, "confirm_cancel_order_or_edit")
        self.assertEqual(context.awaiting_confirmation_for, {"type": "confirm_full_order_cancellation"})


def _make_context_with_checkout_nlu(user_text: str = "checkout") -> ConversationContext:
    """Return a context with a low-confidence checkout NLU result set.

    This simulates the NluOrchestrator path: model detects checkout but
    intent_confidence=0.2 < INTENT_MIN_CONF (0.55), so intent_result.intent is
    downgraded to UNKNOWN before the handler is called. The raw effective intent
    lives on context.last_nlu.
    """
    context = _make_context()
    context.set_last_nlu(
        user_text=user_text,
        nlu=NLUResult(
            effective_intent=Intent.CHECKOUT,
            intent_confidence=0.2,
            raw_text=user_text,
            normalized_text=user_text,
            model_main_intent="order",
            model_sub_intent="checkout",
        ),
    )
    return context


class CheckoutIntentLowConfidenceTests(unittest.TestCase):
    """Reproduce the bug: NLU detects checkout but confidence < INTENT_MIN_CONF,
    so TurnEngine downgrades intent to UNKNOWN before calling the handler.
    The handler must consult context.last_nlu.effective_intent as a fallback.

    In production context == session.conversation_context. Tests that set last_nlu
    must set it on the same context object passed to handler.handle().
    """

    def _make_handler(self) -> ConfirmOrderHandler:
        return ConfirmOrderHandler(
            cart_summary_builder=StubCartSummaryBuilder(),
            sms_service=StubSmsService(),
            checkout_service=StubCheckoutService(),
        )

    # ── Failing cases (bug reproduction) ─────────────────────────────────────

    def test_checkout_intent_low_confidence_proceeds_to_checkout(self):
        """Bug: intent downgraded to UNKNOWN by confidence gate; handler must
        still treat checkout-family NLU signal as affirmative confirmation."""
        handler = self._make_handler()
        session = _make_session()
        context = _make_context_with_checkout_nlu("checkout")

        result = handler.handle(
            intent=Intent.UNKNOWN,
            context=context,
            user_text="checkout",
            session=session,
        )

        self.assertNotEqual(result.response_key, "confirm_order_summary_unclear",
                            "checkout NLU signal must not return unclear in confirming_order")
        self.assertEqual(result.next_state, ConversationState.WAITING_FOR_CHECKOUT_COMPLETION)
        self.assertEqual(result.response_key, "checkout_link_sent")

    def test_i_said_checkout_proceeds_to_checkout(self):
        """'I said checkout' yields checkout sub-intent with low confidence —
        must be treated as affirmative in confirming_order."""
        handler = self._make_handler()
        session = _make_session()
        context = _make_context_with_checkout_nlu("i said checkout")

        result = handler.handle(
            intent=Intent.UNKNOWN,
            context=context,
            user_text="i said checkout",
            session=session,
        )

        self.assertNotEqual(result.response_key, "confirm_order_summary_unclear")
        self.assertEqual(result.next_state, ConversationState.WAITING_FOR_CHECKOUT_COMPLETION)

    def test_oh_yeah_checkout_proceeds_to_checkout(self):
        """Affirm + checkout signal in the same utterance must proceed."""
        handler = self._make_handler()
        session = _make_session()
        context = _make_context_with_checkout_nlu("oh yeah checkout")

        result = handler.handle(
            intent=Intent.UNKNOWN,
            context=context,
            user_text="oh yeah checkout",
            session=session,
        )

        self.assertNotEqual(result.response_key, "confirm_order_summary_unclear")
        self.assertEqual(result.next_state, ConversationState.WAITING_FOR_CHECKOUT_COMPLETION)

    def test_yes_checkout_proceeds_to_checkout(self):
        """'yes checkout' must be treated as affirmative."""
        handler = self._make_handler()
        session = _make_session()
        context = _make_context_with_checkout_nlu("yes checkout")

        result = handler.handle(
            intent=Intent.UNKNOWN,
            context=context,
            user_text="yes checkout",
            session=session,
        )

        self.assertNotEqual(result.response_key, "confirm_order_summary_unclear")
        self.assertEqual(result.next_state, ConversationState.WAITING_FOR_CHECKOUT_COMPLETION)

    def test_continue_to_checkout_proceeds(self):
        """'continue to checkout' must be treated as affirmative."""
        handler = self._make_handler()
        session = _make_session()
        context = _make_context_with_checkout_nlu("continue to checkout")

        result = handler.handle(
            intent=Intent.UNKNOWN,
            context=context,
            user_text="continue to checkout",
            session=session,
        )

        self.assertNotEqual(result.response_key, "confirm_order_summary_unclear")
        self.assertEqual(result.next_state, ConversationState.WAITING_FOR_CHECKOUT_COMPLETION)

    # ── Deny / change behavior must be preserved ─────────────────────────────

    def test_deny_intent_in_nlu_does_not_checkout(self):
        """DENY NLU signal must not trigger checkout, even in confirming_order."""
        handler = self._make_handler()
        session = _make_session()
        context = _make_context()
        context.set_last_nlu(
            user_text="no",
            nlu=NLUResult(
                effective_intent=Intent.DENY,
                intent_confidence=0.9,
                raw_text="no",
                normalized_text="no",
            ),
        )

        result = handler.handle(
            intent=Intent.UNKNOWN,
            context=context,
            user_text="no",
            session=session,
        )

        self.assertNotEqual(result.next_state, ConversationState.WAITING_FOR_CHECKOUT_COMPLETION)

    def test_modify_order_does_not_checkout(self):
        """'modify order' must not trigger checkout path."""
        handler = self._make_handler()
        session = _make_session()
        context = _make_context()
        context.set_last_nlu(
            user_text="change item",
            nlu=NLUResult(
                effective_intent=Intent.MODIFY_ITEM,
                intent_confidence=0.8,
                raw_text="change item",
                normalized_text="change item",
            ),
        )

        result = handler.handle(
            intent=Intent.MODIFY_ITEM,
            context=context,
            user_text="change item",
            session=session,
        )

        self.assertNotEqual(result.next_state, ConversationState.WAITING_FOR_CHECKOUT_COMPLETION)

    # ── Unknown utterance still returns unclear ───────────────────────────────

    def test_truly_unknown_utterance_returns_unclear(self):
        """A genuinely unknown utterance must still return confirm_order_summary_unclear."""
        handler = self._make_handler()
        session = _make_session()
        context = _make_context()
        context.set_last_nlu(
            user_text="blah blah blah",
            nlu=NLUResult(
                effective_intent=Intent.UNKNOWN,
                intent_confidence=0.1,
                raw_text="blah blah blah",
                normalized_text="blah blah blah",
            ),
        )

        result = handler.handle(
            intent=Intent.UNKNOWN,
            context=context,
            user_text="blah blah blah",
            session=session,
        )

        self.assertEqual(result.next_state, ConversationState.CONFIRMING_ORDER)
        self.assertEqual(result.response_key, "confirm_order_summary_unclear")

    def test_no_last_nlu_unknown_returns_unclear(self):
        """No NLU context + unknown intent must still return unclear (null safety)."""
        handler = self._make_handler()
        session = _make_session()
        context = _make_context()  # last_nlu is None by default

        result = handler.handle(
            intent=Intent.UNKNOWN,
            context=context,
            user_text="something random",
            session=session,
        )

        self.assertEqual(result.next_state, ConversationState.CONFIRMING_ORDER)
        self.assertEqual(result.response_key, "confirm_order_summary_unclear")


if __name__ == "__main__":
    unittest.main()

