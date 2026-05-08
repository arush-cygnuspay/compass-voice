import unittest
import sys
import types


twilio_module = types.ModuleType("twilio")
twilio_base_module = types.ModuleType("twilio.base")
twilio_base_exceptions_module = types.ModuleType("twilio.base.exceptions")
twilio_rest_module = types.ModuleType("twilio.rest")


class _TwilioClient:
    def __init__(self, *args, **kwargs):
        pass


class _TwilioRestException(Exception):
    pass


twilio_base_exceptions_module.TwilioRestException = _TwilioRestException
twilio_rest_module.Client = _TwilioClient
sys.modules.setdefault("twilio", twilio_module)
sys.modules.setdefault("twilio.base", twilio_base_module)
sys.modules.setdefault("twilio.base.exceptions", twilio_base_exceptions_module)
sys.modules.setdefault("twilio.rest", twilio_rest_module)

from app.state_machine.handlers.payment.payment_flow_support import verify_payment_for_order
from app.state_machine.models.conversation_state import ConversationState


class StubCheckoutService:
    def verify_payment_by_order_number(self, order_number: str) -> dict:
        return {
            "ok": True,
            "paid": False,
            "payment_completed": False,
            "status": "failed",
            "reference": None,
            "session": None,
            "error": None,
        }


class PaymentFlowSupportTests(unittest.TestCase):
    def test_failed_payment_returns_draft_retry_response(self):
        result = verify_payment_for_order(
            checkout_service=StubCheckoutService(),
            order_number="1234567",
            pending_state=ConversationState.WAITING_FOR_PAYMENT,
            pending_response_key="waiting_for_payment",
        )

        self.assertEqual(result.next_state, ConversationState.CONFIRMING_ORDER)
        self.assertEqual(result.response_key, "payment_draft_saved_retry_later")
        self.assertEqual(result.response_payload["order_number"], "1234567")
        events = result.response_payload.get("_payment_events", [])
        self.assertTrue(any(e["event_name"] == "payment_failed" for e in events))


if __name__ == "__main__":
    unittest.main()
