import tempfile
import unittest
import sys
import types
from pathlib import Path
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
from app.models.payment_link_session import PaymentLinkSession
from app.services.checkout_service import CheckoutService


class DummyPaymentProvider:
    PAID_STATUSES = {"paid", "completed"}

    def __init__(self) -> None:
        pass

    def get_payment_status(self, *, request_id: str) -> dict:
        return {
            "ok": True,
            "paid": False,
            "status": "failed",
            "reference": None,
            "raw": {"request_id": request_id},
        }


class DummyPaidPaymentProvider:
    PAID_STATUSES = {"paid", "completed"}

    def __init__(self) -> None:
        pass

    def get_payment_status(self, *, request_id: str) -> dict:
        return {
            "ok": True,
            "paid": True,
            "status": "paid",
            "reference": f"paid-{request_id}",
            "raw": {"request_id": request_id},
        }


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


class CheckoutServiceRetryStateTests(unittest.TestCase):
    def test_failed_provider_status_marks_session_retryable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = _configure_checkout_dirs(Path(temp_dir))

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
            ):
                service = CheckoutService()

                checkout_session = service.create_session(
                    restaurant_id="steves_grill",
                    call_sid=None,
                    order_number="1234567",
                    customer_phone_number="+15555550123",
                    address_required=False,
                    area="Downtown",
                    postal_code="12345",
                    order_summary={"total": "$18.25"},
                )
                checkout_session.mark_payment_started()
                service.save_session(checkout_session)

                payment_link_session = PaymentLinkSession(
                    checkout_token=checkout_session.token,
                    invoice_no="1234567",
                    amount="18.25",
                    request_id="REQ123",
                    public_link_url="https://example.com/pay",
                    status="created",
                )
                service.save_payment_link_session(payment_link_session)

                result = service.verify_payment_with_provider(checkout_session.token)

                self.assertTrue(result["ok"])
                self.assertFalse(result["payment_completed"])
                self.assertEqual(result["status"], "failed")
                self.assertIsNotNone(result["session"])
                self.assertTrue(result["session"]["can_retry_payment"])
                self.assertTrue(result["session"]["payment_retry_available"])
                self.assertEqual(result["session"]["last_payment_status"], "failed")
                self.assertEqual(result["session"]["status"], "pending_payment_retry")

    def test_retryable_session_still_rechecks_provider_and_can_complete(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = _configure_checkout_dirs(Path(temp_dir))

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
                    DummyPaidPaymentProvider,
                ),
            ):
                service = CheckoutService()

                checkout_session = service.create_session(
                    restaurant_id="steves_grill",
                    call_sid=None,
                    order_number="1234567",
                    customer_phone_number="+15555550123",
                    address_required=False,
                    area="Downtown",
                    postal_code="12345",
                    order_summary={"total": "$18.25"},
                )
                checkout_session.mark_payment_retryable("failed")
                service.save_session(checkout_session)

                payment_link_session = PaymentLinkSession(
                    checkout_token=checkout_session.token,
                    invoice_no="1234567",
                    amount="18.25",
                    request_id="REQ123",
                    public_link_url="https://example.com/pay",
                    status="failed",
                )
                service.save_payment_link_session(payment_link_session)

                result = service.verify_payment_with_provider(checkout_session.token)

                self.assertTrue(result["ok"])
                self.assertTrue(result["payment_completed"])
                self.assertEqual(result["status"], "paid")
                self.assertEqual(result["reference"], "paid-REQ123")


if __name__ == "__main__":
    unittest.main()
