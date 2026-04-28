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

from app.nlu.intent_resolution.intent import Intent
from app.session.session import Session
from app.state_machine.handlers.payment.waiting_for_payment_handler import (
    WaitingForPaymentHandler,
)
from app.state_machine.models.conversation_context import ConversationContext
from app.state_machine.models.conversation_state import ConversationState


class StubCartSummaryBuilder:
    def build(self, cart):
        return {"items": [{"name": "Burger", "quantity": 1}], "total": "$10.00"}


class StubCheckoutService:
    def verify_payment_by_order_number(self, order_number: str) -> dict:
        return {
            "ok": True,
            "paid": False,
            "payment_completed": False,
            "status": "pending",
            "reference": None,
            "session": None,
            "error": None,
        }


class WaitingForPaymentHandlerControlIntentTests(unittest.TestCase):
    def _make_session(self) -> Session:
        session = Session(session_id="call-2", restaurant_id="demo")
        session.conversation_state = ConversationState.WAITING_FOR_PAYMENT
        return session

    def _make_context(self) -> ConversationContext:
        context = ConversationContext()
        context.delivery_address.order_number = "ABC-123"
        context.delivery_address.customer_phone_number = "+15555550123"
        context.delivery_address.payment_link = "https://example.com/pay"
        return context

    def test_repeat_that_repeats_payment_instruction(self):
        handler = WaitingForPaymentHandler(StubCartSummaryBuilder(), StubCheckoutService())
        result = handler.handle(
            intent=Intent.UNKNOWN,
            context=self._make_context(),
            user_text="repeat that",
            session=self._make_session(),
        )

        self.assertEqual(result.next_state, ConversationState.WAITING_FOR_PAYMENT)
        self.assertEqual(result.response_key, "waiting_for_payment")

    def test_cancel_does_not_silently_cancel_payment_wait(self):
        handler = WaitingForPaymentHandler(StubCartSummaryBuilder(), StubCheckoutService())
        result = handler.handle(
            intent=Intent.UNKNOWN,
            context=self._make_context(),
            user_text="cancel",
            session=self._make_session(),
        )

        self.assertEqual(result.next_state, ConversationState.WAITING_FOR_PAYMENT)
        self.assertEqual(result.response_key, "cannot_cancel_during_checkout")


if __name__ == "__main__":
    unittest.main()
