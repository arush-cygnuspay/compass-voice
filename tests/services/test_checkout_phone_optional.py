import logging
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


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

redis_module = types.ModuleType("redis")


class _RedisClient:
    def __init__(self, *args, **kwargs):
        pass


redis_module.Redis = _RedisClient
sys.modules.setdefault("redis", redis_module)

import app.services.checkout_service as checkout_service_module
import app.session.repository as session_repository_module
from app.services.checkout_service import CheckoutService
from app.session.session import Session
from app.state_machine.models.conversation_state import ConversationState


class DummyPaymentProvider:
    PAID_STATUSES = {"paid", "completed"}

    def __init__(self) -> None:
        pass


class DummySmsService:
    def __init__(self) -> None:
        self.requests = []

    def send(self, request):
        self.requests.append(request)
        return SimpleNamespace(
            ok=True,
            sid="SM123",
            error_code=None,
            error_message=None,
        )


class DummyLiveCallService:
    def __init__(self) -> None:
        self.calls = []

    def announce_order_completed(self, *, call_sid, order_number):
        self.calls.append({"call_sid": call_sid, "order_number": order_number})
        return True


def _configure_checkout_dirs(tmp_path: Path) -> dict[str, Path]:
    checkout_dir = tmp_path / "checkout_sessions"
    payment_dir = tmp_path / "payment_link_sessions"
    order_index_dir = checkout_dir / "_indexes" / "by_order_number"
    latest_payment_index_dir = payment_dir / "_indexes" / "latest_by_checkout_token"
    request_payment_index_dir = payment_dir / "_indexes" / "by_request_id"

    for directory in (
        checkout_dir,
        payment_dir,
        order_index_dir,
        latest_payment_index_dir,
        request_payment_index_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    return {
        "checkout_dir": checkout_dir,
        "payment_dir": payment_dir,
        "checkout_index_dir": checkout_dir / "_indexes",
        "order_index_dir": order_index_dir,
        "payment_index_dir": payment_dir / "_indexes",
        "latest_payment_index_dir": latest_payment_index_dir,
        "request_payment_index_dir": request_payment_index_dir,
    }


class _PhoneUnavailableLogCapture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.events: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        if record.getMessage() == "phone_number_unavailable":
            self.events.append(record)


class CheckoutPhoneOptionalTests(unittest.TestCase):
    def test_handle_payment_completed_with_no_phone_does_not_raise_or_send_sms(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = _configure_checkout_dirs(Path(temp_dir))

            voice_session = Session(session_id="chat-1", restaurant_id="steves_grill")
            voice_session.conversation_state = ConversationState.WAITING_FOR_PAYMENT
            saved_voice_session: dict[str, Session] = {}

            def fake_load_existing_session(session_id: str, restaurant_id: str):
                return voice_session

            def fake_save_session(session: Session):
                saved_voice_session["session"] = session

            log_capture = _PhoneUnavailableLogCapture()
            checkout_logger = logging.getLogger(checkout_service_module.__name__)
            previous_propagate = checkout_logger.propagate
            checkout_logger.addHandler(log_capture)
            checkout_logger.propagate = False
            previous_level = checkout_logger.level
            checkout_logger.setLevel(logging.INFO)

            try:
                with (
                    patch.object(
                        checkout_service_module,
                        "CHECKOUT_DATA_DIR",
                        paths["checkout_dir"],
                    ),
                    patch.object(
                        checkout_service_module,
                        "PAYMENT_LINK_SESSION_DATA_DIR",
                        paths["payment_dir"],
                    ),
                    patch.object(
                        checkout_service_module,
                        "CHECKOUT_INDEX_DIR",
                        paths["checkout_index_dir"],
                    ),
                    patch.object(
                        checkout_service_module,
                        "CHECKOUT_ORDER_INDEX_DIR",
                        paths["order_index_dir"],
                    ),
                    patch.object(
                        checkout_service_module,
                        "PAYMENT_LINK_INDEX_DIR",
                        paths["payment_index_dir"],
                    ),
                    patch.object(
                        checkout_service_module,
                        "PAYMENT_LINK_BY_CHECKOUT_INDEX_DIR",
                        paths["latest_payment_index_dir"],
                    ),
                    patch.object(
                        checkout_service_module,
                        "PAYMENT_LINK_BY_REQUEST_INDEX_DIR",
                        paths["request_payment_index_dir"],
                    ),
                    patch.object(
                        checkout_service_module,
                        "DatacapPaymentLinksService",
                        DummyPaymentProvider,
                    ),
                    patch.object(
                        session_repository_module,
                        "load_existing_session",
                        fake_load_existing_session,
                    ),
                    patch.object(
                        session_repository_module,
                        "save_session",
                        fake_save_session,
                    ),
                ):
                    service = CheckoutService()
                    service.sms_service = DummySmsService()
                    service.live_call_service = DummyLiveCallService()

                    checkout_session = service.create_session(
                        restaurant_id="steves_grill",
                        call_sid=None,
                        order_number="7654321",
                        customer_phone_number=None,
                        address_required=True,
                        area="Downtown",
                        postal_code="12345",
                        order_summary={"total": "$18.25"},
                    )
                    checkout_session.house_number = "42"
                    checkout_session.street = "Main Street"
                    checkout_session.mark_address_completed()
                    service.save_session(checkout_session)

                    updated = service.handle_payment_completed(
                        order_number="7654321",
                        payment_reference="ref-789",
                    )

                    self.assertIsNotNone(updated)
                    self.assertTrue(updated.payment_completed)
                    self.assertEqual(len(service.sms_service.requests), 0)
                    self.assertEqual(len(log_capture.events), 1)
                    record = log_capture.events[0]
                    self.assertEqual(
                        getattr(record, "consumer", None),
                        "checkout_service.handle_payment_completed",
                    )
                    self.assertEqual(
                        getattr(record, "surface", None),
                        "chat_ui",
                    )
            finally:
                checkout_logger.removeHandler(log_capture)
                checkout_logger.propagate = previous_propagate
                checkout_logger.setLevel(previous_level)


if __name__ == "__main__":
    unittest.main()
