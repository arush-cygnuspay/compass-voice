import unittest
import sys
import types
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient


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

from app.api import chat_demo
from app.services.sms_service import DEFAULT_SMS_OVERRIDE_TO
from app.session.session import Session
from app.state_machine.models.conversation_state import ConversationState


class StubEngine:
    def process_turn(self, session, user_text):
        raise AssertionError("process_turn should not be called for bootstrap requests")


class StubResponder:
    def build(self, response_key, context, payload=None):
        return f"rendered:{response_key}"


def _build_client(engine=None, responder=None) -> TestClient:
    app = FastAPI()
    app.state.engine = engine or StubEngine()
    app.state.responder = responder or StubResponder()
    app.include_router(chat_demo.router)
    return TestClient(app)


class ChatDemoTests(unittest.TestCase):
    def test_blank_chat_bootstraps_browser_session_and_sets_default_sms_number(self):
        session = Session(session_id="ui-1", restaurant_id="steves_grill")

        with patch.object(chat_demo, "load_session", return_value=session), patch.object(
            chat_demo,
            "save_session",
        ) as mocked_save:
            client = _build_client()
            response = client.post(
                "/test/chat",
                json={"session_id": "ui-1", "text": ""},
            )

        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["response_key"], "ask_for_order_type")
        self.assertEqual(payload["state"], "WAITING_FOR_ORDER_TYPE")
        self.assertEqual(payload["response"], "rendered:ask_for_order_type")
        self.assertEqual(payload["sms_phone_number"], DEFAULT_SMS_OVERRIDE_TO)
        self.assertEqual(payload["quick_replies"], ["Pickup", "Delivery"])
        self.assertEqual(session.conversation_context.caller_device_type, "chat")
        self.assertEqual(
            session.conversation_context.delivery_address.customer_phone_number,
            DEFAULT_SMS_OVERRIDE_TO,
        )
        mocked_save.assert_called_once_with(session)

    def test_waiting_checkout_response_exposes_direct_link_and_auto_check_signal(self):
        session = Session(session_id="ui-2", restaurant_id="steves_grill")
        session.conversation_state = ConversationState.WAITING_FOR_CHECKOUT_COMPLETION
        session.last_response_key = "waiting_for_checkout_completion"
        session.conversation_context.delivery_address.order_number = "98765"
        session.conversation_context.delivery_address.address_form_link = "https://example.com/checkout"

        with patch.object(chat_demo, "load_session", return_value=session), patch.object(
            chat_demo,
            "save_session",
        ):
            client = _build_client()
            response = client.post(
                "/test/chat",
                json={"session_id": "ui-2", "text": ""},
            )

        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(payload["waiting_external"])
        self.assertTrue(payload["auto_check_recommended"])
        self.assertEqual(payload["order_number"], "98765")
        self.assertEqual(
            payload["links"],
            [
                {
                    "kind": "checkout",
                    "label": "Open secure checkout",
                    "url": "https://example.com/checkout",
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
