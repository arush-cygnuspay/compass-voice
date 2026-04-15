import sys
import types
import unittest
from unittest.mock import patch


twilio_module = types.ModuleType("twilio")
twilio_base_module = types.ModuleType("twilio.base")
twilio_base_exceptions_module = types.ModuleType("twilio.base.exceptions")
twilio_rest_module = types.ModuleType("twilio.rest")


class _TwilioRestException(Exception):
    pass


class _MessageClient:
    def __init__(self) -> None:
        self.created = []

    def create(self, **kwargs):
        self.created.append(kwargs)
        return types.SimpleNamespace(sid="SM123")


class _TwilioClient:
    def __init__(self, *args, **kwargs):
        self.messages = _MessageClient()


twilio_base_exceptions_module.TwilioRestException = _TwilioRestException
twilio_rest_module.Client = _TwilioClient

sys.modules.setdefault("twilio", twilio_module)
sys.modules.setdefault("twilio.base", twilio_base_module)
sys.modules.setdefault("twilio.base.exceptions", twilio_base_exceptions_module)
sys.modules.setdefault("twilio.rest", twilio_rest_module)

from app.services.sms_service import SmsSendRequest, SmsService


class SmsServiceTests(unittest.TestCase):
    def test_send_uses_default_override_number(self):
        with patch.dict(
            "os.environ",
            {
                "TWILIO_ACCOUNT_SID": "AC123",
                "TWILIO_AUTH_TOKEN": "auth-token",
                "TWILIO_SMS_FROM_NUMBER": "+15551230000",
            },
            clear=False,
        ), patch("app.services.sms_service.Client", _TwilioClient):
            service = SmsService()
            result = service.send(
                SmsSendRequest(
                    template="payment_link",
                    phone_number="+15550001111",
                    order_number="1234567",
                    link="https://example.com/pay",
                )
            )

        self.assertTrue(result.ok)
        self.assertEqual(
            service._client.messages.created[0]["to"],
            "+923204711572",
        )


if __name__ == "__main__":
    unittest.main()
