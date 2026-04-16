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


if __name__ == "__main__":
    unittest.main()
