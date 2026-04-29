# tests/core/test_sms_retry_safety.py
"""Tests for SMS retry safety refactor.

Coverage:
- TransientSmsError and PermanentSmsError are distinct Exception subclasses
- Successful SMS sends exactly once with idempotency key in request
- Transient failure retries up to SMS_MAX_RETRIES times
- Permanent failure does not retry (fails on first attempt)
- Duplicate retry uses the same idempotency key
- Different session_id or payload produces a different key
- Missing session_id handled gracefully
- CommandExecutor result dict shape: ok, sid, error_code, idempotency_key, attempts_made
- Non-SMS commands are unaffected by the refactor
- SmsService.send raises PermanentSmsError for permanent Twilio codes
- SmsService.send raises TransientSmsError for 429 / 5xx Twilio status
- SmsService.send raises TransientSmsError for network errors
- SmsService.send raises PermanentSmsError when not configured
"""
from __future__ import annotations

import hashlib
import json
import sys
import types
import unittest
from unittest.mock import MagicMock, call, patch

# ── Stub heavy ML/Twilio imports ─────────────────────────────────────────────
for _name in ("twilio", "twilio.base", "twilio.base.exceptions", "twilio.rest"):
    sys.modules.setdefault(_name, types.ModuleType(_name))
sys.modules["twilio.base.exceptions"].TwilioRestException = Exception


class _FakeTwilioRestException(Exception):
    def __init__(self, msg: str = "", *, code: int = 0, status: int = 0) -> None:
        super().__init__(msg)
        self.code = code
        self.status = status


sys.modules["twilio.base.exceptions"].TwilioRestException = _FakeTwilioRestException
sys.modules["twilio.rest"].Client = type(
    "_Client", (), {"__init__": lambda *a, **k: None}
)
# ─────────────────────────────────────────────────────────────────────────────

from app.services.sms_exceptions import PermanentSmsError, SmsError, TransientSmsError
from app.core.command_executor import (
    CommandExecutor,
    SMS_MAX_RETRIES,
    _derive_command_id,
    _derive_idempotency_key,
)
from app.services.sms_service import SmsSendRequest, SmsSendResult, SmsService
from app.session.session import Session


# ── helpers ───────────────────────────────────────────────────────────────────

def _session(session_id: str = "sess-001") -> Session:
    s = Session(session_id=session_id, restaurant_id="r1")
    return s


def _sms_command(
    template: str = "payment_link",
    phone_number: str = "+15550001111",
    order_number: str = "ORD-42",
    link: str = "https://pay.example.com/abc",
) -> dict:
    return {
        "type": "SEND_SMS",
        "payload": {
            "template": template,
            "phone_number": phone_number,
            "order_number": order_number,
            "link": link,
        },
    }


def _executor(sms_service: SmsService) -> CommandExecutor:
    return CommandExecutor(sms_service=sms_service)


def _mock_sms_service(*, send_side_effect=None, send_return=None) -> SmsService:
    svc = MagicMock(spec=SmsService)
    if send_side_effect is not None:
        svc.send.side_effect = send_side_effect
    elif send_return is not None:
        svc.send.return_value = send_return
    else:
        svc.send.return_value = SmsSendResult(ok=True, sid="SM123")
    return svc


# ── Exception hierarchy ───────────────────────────────────────────────────────

class TestSmsExceptionHierarchy(unittest.TestCase):
    def test_transient_is_sms_error(self):
        assert issubclass(TransientSmsError, SmsError)

    def test_permanent_is_sms_error(self):
        assert issubclass(PermanentSmsError, SmsError)

    def test_transient_and_permanent_are_not_related(self):
        assert not issubclass(TransientSmsError, PermanentSmsError)
        assert not issubclass(PermanentSmsError, TransientSmsError)

    def test_transient_carries_error_code(self):
        exc = TransientSmsError("timeout", error_code="sms_network_error")
        assert exc.error_code == "sms_network_error"
        assert "timeout" in str(exc)

    def test_permanent_carries_error_code(self):
        exc = PermanentSmsError("bad number", error_code="21211")
        assert exc.error_code == "21211"

    def test_error_code_defaults_to_none(self):
        assert TransientSmsError("x").error_code is None
        assert PermanentSmsError("x").error_code is None


# ── Idempotency key derivation ────────────────────────────────────────────────

class TestIdempotencyKeyDerivation(unittest.TestCase):
    def _expected_key(self, session_id: str, command_id: str) -> str:
        return hashlib.sha256(f"{session_id}:{command_id}".encode()).hexdigest()

    def test_key_is_sha256_hex(self):
        key = _derive_idempotency_key("sess-1", "cmd-abc")
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)

    def test_different_session_id_different_key(self):
        k1 = _derive_idempotency_key("sess-1", "cmd-abc")
        k2 = _derive_idempotency_key("sess-2", "cmd-abc")
        assert k1 != k2

    def test_different_command_id_different_key(self):
        k1 = _derive_idempotency_key("sess-1", "cmd-abc")
        k2 = _derive_idempotency_key("sess-1", "cmd-xyz")
        assert k1 != k2

    def test_same_inputs_same_key(self):
        k1 = _derive_idempotency_key("sess-1", "cmd-abc")
        k2 = _derive_idempotency_key("sess-1", "cmd-abc")
        assert k1 == k2

    def test_command_id_derived_from_payload_is_deterministic(self):
        payload = {"template": "payment_link", "phone_number": "+15550001111",
                   "order_number": "ORD-42", "link": "https://pay.example.com"}
        id1 = _derive_command_id(payload)
        id2 = _derive_command_id(payload)
        assert id1 == id2

    def test_different_payload_different_command_id(self):
        p1 = {"template": "payment_link", "phone_number": "+15550001111"}
        p2 = {"template": "payment_link", "phone_number": "+15550009999"}
        assert _derive_command_id(p1) != _derive_command_id(p2)


# ── Successful SMS ────────────────────────────────────────────────────────────

class TestSuccessfulSmsSend(unittest.TestCase):
    def test_sends_exactly_once_on_success(self):
        svc = _mock_sms_service()
        exe = _executor(svc)

        result = exe.execute(_session(), _sms_command())

        assert svc.send.call_count == 1
        assert result.ok is True

    def test_result_includes_sid(self):
        svc = _mock_sms_service(send_return=SmsSendResult(ok=True, sid="SM-XYZ"))
        exe = _executor(svc)

        result = exe.execute(_session(), _sms_command())

        assert result.sid == "SM-XYZ"

    def test_result_includes_idempotency_key(self):
        svc = _mock_sms_service()
        exe = _executor(svc)

        result = exe.execute(_session("s1"), _sms_command())

        assert result.idempotency_key is not None
        assert len(result.idempotency_key) == 64

    def test_idempotency_key_in_request_passed_to_service(self):
        svc = _mock_sms_service()
        exe = _executor(svc)

        exe.execute(_session("s1"), _sms_command())

        request_arg: SmsSendRequest = svc.send.call_args[0][0]
        assert isinstance(request_arg, SmsSendRequest)
        assert len(request_arg.idempotency_key) == 64

    def test_idempotency_key_matches_derived_value(self):
        svc = _mock_sms_service()
        exe = _executor(svc)
        cmd = _sms_command()
        session = _session("sess-abc")

        exe.execute(session, cmd)

        request_arg: SmsSendRequest = svc.send.call_args[0][0]
        expected_cmd_id = _derive_command_id(cmd["payload"])
        expected_key = _derive_idempotency_key("sess-abc", expected_cmd_id)
        assert request_arg.idempotency_key == expected_key

    def test_attempts_made_is_one_on_success(self):
        svc = _mock_sms_service()
        exe = _executor(svc)

        result = exe.execute(_session(), _sms_command())

        assert result.attempts_made == 1

    def test_error_code_is_none_on_success(self):
        svc = _mock_sms_service()
        exe = _executor(svc)

        result = exe.execute(_session(), _sms_command())

        assert result.error_code is None
        assert result.error_message is None


# ── Transient failure retries ─────────────────────────────────────────────────

class TestTransientSmsRetry(unittest.TestCase):
    def test_retries_on_transient_error(self):
        svc = _mock_sms_service(
            send_side_effect=[
                TransientSmsError("timeout", error_code="sms_network_error"),
                SmsSendResult(ok=True, sid="SM-RETRY"),
            ]
        )
        exe = _executor(svc)

        result = exe.execute(_session(), _sms_command())

        assert svc.send.call_count == 2
        assert result.ok is True
        assert result.attempts_made == 2

    def test_exhausts_max_retries_on_repeated_transient_failure(self):
        svc = _mock_sms_service(
            send_side_effect=[
                TransientSmsError("rate limited", error_code="20429"),
            ] * 10  # more than SMS_MAX_RETRIES
        )
        exe = _executor(svc)

        result = exe.execute(_session(), _sms_command())

        assert svc.send.call_count == SMS_MAX_RETRIES
        assert result.ok is False
        assert result.attempts_made == SMS_MAX_RETRIES

    def test_same_idempotency_key_on_every_retry(self):
        calls: list[SmsSendRequest] = []

        def capture_and_fail(req: SmsSendRequest):
            calls.append(req)
            raise TransientSmsError("timeout")

        svc = _mock_sms_service(send_side_effect=capture_and_fail)
        exe = _executor(svc)

        exe.execute(_session("sess-x"), _sms_command())

        assert len(calls) == SMS_MAX_RETRIES
        keys = {req.idempotency_key for req in calls}
        assert len(keys) == 1, "all retries must use the same idempotency key"

    def test_transient_error_code_in_result(self):
        svc = _mock_sms_service(
            send_side_effect=TransientSmsError("timeout", error_code="sms_network_error")
        )
        exe = _executor(svc)

        result = exe.execute(_session(), _sms_command())

        assert result.ok is False
        assert result.error_code == "sms_network_error"


# ── Permanent failure does not retry ─────────────────────────────────────────

class TestPermanentSmsNoRetry(unittest.TestCase):
    def test_does_not_retry_permanent_failure(self):
        svc = _mock_sms_service(
            send_side_effect=PermanentSmsError("invalid number", error_code="21211")
        )
        exe = _executor(svc)

        result = exe.execute(_session(), _sms_command())

        assert svc.send.call_count == 1
        assert result.ok is False

    def test_attempts_made_is_one_on_permanent_failure(self):
        svc = _mock_sms_service(
            send_side_effect=PermanentSmsError("auth error", error_code="20003")
        )
        exe = _executor(svc)

        result = exe.execute(_session(), _sms_command())

        assert result.attempts_made == 1

    def test_permanent_error_code_in_result(self):
        svc = _mock_sms_service(
            send_side_effect=PermanentSmsError("bad number", error_code="21211")
        )
        exe = _executor(svc)

        result = exe.execute(_session(), _sms_command())

        assert result.error_code == "21211"
        assert result.ok is False


# ── Different session/payload → different idempotency key ────────────────────

class TestIdempotencyKeyUniqueness(unittest.TestCase):
    def _key_for(self, session_id: str, payload: dict) -> str:
        svc = _mock_sms_service()
        exe = _executor(svc)
        exe.execute(_session(session_id), {"type": "SEND_SMS", "payload": payload})
        req: SmsSendRequest = svc.send.call_args[0][0]
        return req.idempotency_key

    def test_different_session_id_different_key(self):
        payload = _sms_command()["payload"]
        k1 = self._key_for("sess-A", payload)
        k2 = self._key_for("sess-B", payload)
        assert k1 != k2

    def test_different_phone_number_different_key(self):
        p1 = {**_sms_command()["payload"], "phone_number": "+15550001111"}
        p2 = {**_sms_command()["payload"], "phone_number": "+15550009999"}
        k1 = self._key_for("s1", p1)
        k2 = self._key_for("s1", p2)
        assert k1 != k2

    def test_same_session_and_payload_same_key(self):
        payload = _sms_command()["payload"]
        k1 = self._key_for("sess-A", payload)
        k2 = self._key_for("sess-A", payload)
        assert k1 == k2


# ── Missing session_id handled gracefully ────────────────────────────────────

class TestMissingSessionId(unittest.TestCase):
    def test_empty_session_id_still_produces_key(self):
        svc = _mock_sms_service()
        exe = _executor(svc)
        session = _session("")  # empty session_id

        result = exe.execute(session, _sms_command())

        assert result.ok is True
        assert len(result.idempotency_key) == 64


# ── Non-SMS commands unaffected ───────────────────────────────────────────────

class TestNonSmsCommandsUnaffected(unittest.TestCase):
    def _executor_with_session(self):
        return _executor(_mock_sms_service()), _session()

    def test_clear_cart_returns_ok(self):
        exe, session = self._executor_with_session()
        result = exe.execute(session, {"type": "CLEAR_CART"})
        assert result.ok is True

    def test_remove_item_from_cart(self):
        exe, session = self._executor_with_session()
        session.cart.add_item = MagicMock()
        session.cart.remove_item = MagicMock()
        result = exe.execute(session, {
            "type": "REMOVE_ITEM_FROM_CART",
            "payload": {"cart_item_id": "item-1"},
        })
        assert result.ok is True

    def test_transfer_call_returns_transport_only(self):
        exe, session = self._executor_with_session()
        result = exe.execute(session, {
            "type": "transfer_call",
            "transfer_number": "+1555999",
        })
        assert result.ok is True
        assert result.transport_only is True
        assert result.transfer_number == "+1555999"

    def test_unknown_command_raises_value_error(self):
        exe, session = self._executor_with_session()
        with self.assertRaises(ValueError):
            exe.execute(session, {"type": "DO_MAGIC"})


# ── SmsService typed exception mapping ───────────────────────────────────────

class TestSmsServiceExceptionMapping(unittest.TestCase):
    """Tests for SmsService._raise_typed_twilio_error classification."""

    def _exc(self, *, code: int = 0, status: int = 0) -> _FakeTwilioRestException:
        return _FakeTwilioRestException("err", code=code, status=status)

    def test_429_is_transient(self):
        from app.services.sms_service import SmsService as _SmsService
        with self.assertRaises(TransientSmsError):
            _SmsService._raise_typed_twilio_error(self._exc(code=20429, status=429))

    def test_500_is_transient(self):
        from app.services.sms_service import SmsService as _SmsService
        with self.assertRaises(TransientSmsError):
            _SmsService._raise_typed_twilio_error(self._exc(code=20500, status=500))

    def test_503_is_transient(self):
        from app.services.sms_service import SmsService as _SmsService
        with self.assertRaises(TransientSmsError):
            _SmsService._raise_typed_twilio_error(self._exc(code=20503, status=503))

    def test_invalid_phone_number_is_permanent(self):
        from app.services.sms_service import SmsService as _SmsService
        with self.assertRaises(PermanentSmsError):
            _SmsService._raise_typed_twilio_error(self._exc(code=21211, status=400))

    def test_auth_error_is_permanent(self):
        from app.services.sms_service import SmsService as _SmsService
        with self.assertRaises(PermanentSmsError):
            _SmsService._raise_typed_twilio_error(self._exc(code=20003, status=401))

    def test_4xx_non_429_is_permanent(self):
        from app.services.sms_service import SmsService as _SmsService
        with self.assertRaises(PermanentSmsError):
            _SmsService._raise_typed_twilio_error(self._exc(code=0, status=400))

    def test_unsubscribed_recipient_is_permanent(self):
        from app.services.sms_service import SmsService as _SmsService
        with self.assertRaises(PermanentSmsError):
            _SmsService._raise_typed_twilio_error(self._exc(code=21610, status=400))

    def test_not_configured_raises_permanent(self):
        from app.services.sms_service import SmsService as _SmsService
        svc = _SmsService.__new__(_SmsService)
        svc._client = None
        svc._from_number = ""
        svc._override_to_number = ""
        svc._account_sid = ""
        svc._auth_token = ""

        with self.assertRaises(PermanentSmsError) as ctx:
            svc.send(SmsSendRequest(
                template="payment_link",
                phone_number="+15550001111",
                idempotency_key="key",
            ))
        assert ctx.exception.error_code == "sms_not_configured"


# ── SmsSendRequest includes idempotency_key ───────────────────────────────────

class TestSmsSendRequestIdempotencyField(unittest.TestCase):
    def test_idempotency_key_defaults_to_empty_string(self):
        req = SmsSendRequest(template="menu_link", phone_number="+15550001111")
        assert req.idempotency_key == ""

    def test_idempotency_key_set_explicitly(self):
        req = SmsSendRequest(
            template="menu_link",
            phone_number="+15550001111",
            idempotency_key="abc123",
        )
        assert req.idempotency_key == "abc123"


if __name__ == "__main__":
    unittest.main()
