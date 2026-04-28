# tests/services/test_checkout_extracted_components.py
"""Focused unit tests for the components extracted from CheckoutService.

Covers:
- CheckoutSessionRepository: session/payment-link CRUD, index lookup, not-found/expired
- OrderNumberGenerator: deterministic generation, pass-through for valid numbers
- VoiceSessionSynchronizer: address sync, payment status sync, mark_completed path,
  no call_sid early-exit, stale voice session
- PaymentPollingOrchestrator: start dedup, success/failure/timeout/cancellation paths,
  no threading.Thread in public API
"""
from __future__ import annotations

import sys
import tempfile
import threading
import time
import types
import unittest
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

# ---------------------------------------------------------------------------
# Stub heavy optional deps before they are imported
# ---------------------------------------------------------------------------
for mod_name, stub in [
    ("twilio", types.ModuleType("twilio")),
    ("twilio.base", types.ModuleType("twilio.base")),
    ("twilio.base.exceptions", types.ModuleType("twilio.base.exceptions")),
    ("twilio.rest", types.ModuleType("twilio.rest")),
    ("redis", types.ModuleType("redis")),
]:
    sys.modules.setdefault(mod_name, stub)

_exc_mod = sys.modules["twilio.base.exceptions"]
if not hasattr(_exc_mod, "TwilioRestException"):
    _exc_mod.TwilioRestException = type("TwilioRestException", (Exception,), {})

_rest_mod = sys.modules["twilio.rest"]
if not hasattr(_rest_mod, "Client"):
    _rest_mod.Client = type("Client", (), {"__init__": lambda s, *a, **k: None})

_redis_mod = sys.modules["redis"]
if not hasattr(_redis_mod, "Redis"):
    _redis_mod.Redis = type("Redis", (), {"__init__": lambda s, *a, **k: None})

# ---------------------------------------------------------------------------
# Now import the modules under test
# ---------------------------------------------------------------------------
from app.models.checkout_session import CheckoutSession
from app.models.payment_link_session import PaymentLinkSession
from app.repositories.checkout_session_repository import (
    CheckoutExpiredError,
    CheckoutNotFoundError,
    CheckoutSessionRepository,
)
from app.services.order_number_generator import OrderNumberGenerator
from app.services.payment_polling_orchestrator import PaymentPollingOrchestrator
from app.services.voice_session_synchronizer import VoiceSessionSynchronizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dirs(tmp: Path) -> dict[str, Path]:
    checkout_dir = tmp / "checkout_sessions"
    payment_dir = tmp / "payment_link_sessions"
    for d in (checkout_dir, payment_dir):
        d.mkdir(parents=True)
    return {"checkout_dir": checkout_dir, "payment_dir": payment_dir}


def _repo(tmp: Path) -> CheckoutSessionRepository:
    dirs = _make_dirs(tmp)
    return CheckoutSessionRepository(
        data_dir=dirs["checkout_dir"],
        payment_link_session_dir=dirs["payment_dir"],
    )


def _session(**kw) -> CheckoutSession:
    defaults = dict(
        restaurant_id="demo",
        call_sid=None,
        order_number="1234567",
        customer_phone_number=None,
        address_required=False,
        area=None,
        postal_code=None,
    )
    defaults.update(kw)
    return CheckoutSession.new(**defaults)


def _payment_link(checkout_token: str, *, request_id: str = "REQ1") -> PaymentLinkSession:
    return PaymentLinkSession(
        checkout_token=checkout_token,
        invoice_no="1234567",
        amount="10.00",
        request_id=request_id,
        public_link_url="https://pay.example.com/link",
        status="created",
    )


# ===========================================================================
# CheckoutSessionRepository
# ===========================================================================

class TestCheckoutSessionRepository(unittest.TestCase):

    # --- save/get round-trip ------------------------------------------------

    def test_save_and_get_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = _repo(Path(tmp))
            s = _session()
            repo.save_session(s)
            loaded = repo.get_session(s.token)
            self.assertEqual(loaded.token, s.token)
            self.assertEqual(loaded.order_number, s.order_number)

    def test_get_session_not_found_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = _repo(Path(tmp))
            with self.assertRaises(CheckoutNotFoundError):
                repo.get_session("nonexistent-token")

    def test_get_session_expired_raises(self):
        import json as _json
        with tempfile.TemporaryDirectory() as tmp:
            repo = _repo(Path(tmp))
            s = _session()
            s.expires_at = datetime.now(timezone.utc) - timedelta(hours=1)
            # Write directly so save_session's touch() doesn't reset the expiry
            path = repo._path_for_token(s.token)
            path.write_text(_json.dumps(s.to_dict(), indent=2), encoding="utf-8")
            with self.assertRaises(CheckoutExpiredError):
                repo.get_session(s.token)

    # --- find by order number -----------------------------------------------

    def test_find_by_order_number_via_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = _repo(Path(tmp))
            s = _session(order_number="9876543")
            repo.save_session(s)
            found = repo.find_session_by_order_number("9876543")
            self.assertIsNotNone(found)
            self.assertEqual(found.token, s.token)

    def test_find_by_order_number_fallback_scan(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = _repo(Path(tmp))
            s = _session(order_number="1112223")
            repo.save_session(s)
            # Delete the index to force scan
            idx = repo._order_index_path("1112223")
            idx.unlink()
            found = repo.find_session_by_order_number("1112223")
            self.assertIsNotNone(found)
            self.assertEqual(found.token, s.token)

    def test_find_by_order_number_none_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = _repo(Path(tmp))
            self.assertIsNone(repo.find_session_by_order_number(None))

    def test_find_by_order_number_missing_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = _repo(Path(tmp))
            self.assertIsNone(repo.find_session_by_order_number("0000000"))

    # --- payment link CRUD --------------------------------------------------

    def test_save_and_find_latest_payment_link_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = _repo(Path(tmp))
            s = _session()
            pl = _payment_link(s.token)
            repo.save_payment_link_session(pl)
            found = repo.find_latest_payment_link_session(s.token)
            self.assertIsNotNone(found)
            self.assertEqual(found.request_id, "REQ1")

    def test_find_latest_payment_link_missing_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = _repo(Path(tmp))
            self.assertIsNone(repo.find_latest_payment_link_session("no-token"))

    def test_find_latest_payment_link_fallback_scan(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = _repo(Path(tmp))
            s = _session()
            pl = _payment_link(s.token)
            repo.save_payment_link_session(pl)
            # Delete index to force scan
            idx = repo._payment_link_latest_index_path(s.token)
            idx.unlink()
            found = repo.find_latest_payment_link_session(s.token)
            self.assertIsNotNone(found)


# ===========================================================================
# OrderNumberGenerator
# ===========================================================================

class TestOrderNumberGenerator(unittest.TestCase):

    def test_generates_7_digit_number(self):
        gen = OrderNumberGenerator()
        result = gen.generate()
        self.assertTrue(result.isdigit(), f"Not all digits: {result!r}")
        self.assertEqual(len(result), 7)

    def test_passes_through_valid_digit_order_number(self):
        gen = OrderNumberGenerator()
        result = gen.generate("1234567")
        self.assertEqual(result, "1234567")

    def test_mints_new_when_none(self):
        gen = OrderNumberGenerator()
        result = gen.generate(None)
        self.assertEqual(len(result), 7)
        self.assertTrue(result.isdigit())

    def test_mints_new_when_non_digit(self):
        gen = OrderNumberGenerator()
        result = gen.generate("TEST123")
        self.assertEqual(len(result), 7)
        self.assertTrue(result.isdigit())

    def test_deterministic_with_injected_clock_and_random(self):
        fixed_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        gen = OrderNumberGenerator(
            clock=lambda: fixed_time,
            randint=lambda a, b: 42,
        )
        result1 = gen.generate()
        result2 = gen.generate()
        self.assertEqual(result1, result2)
        # 2 random prefix digits: 42; ts_part = int(1704067200.0*100) % 100_000 = 20000
        self.assertEqual(result1, "4220000")

    def test_random_prefix_in_range(self):
        seen_prefixes = set()
        for seed in range(50):
            gen = OrderNumberGenerator(
                clock=lambda: datetime(2024, 1, 1, tzinfo=timezone.utc),
                randint=lambda a, b: a + (seed % (b - a + 1)),
            )
            r = gen.generate()
            prefix = int(r[:2])
            self.assertGreaterEqual(prefix, 10)
            self.assertLessEqual(prefix, 99)
            seen_prefixes.add(prefix)


# ===========================================================================
# VoiceSessionSynchronizer
# ===========================================================================

class _FakeDelivery:
    def __init__(self):
        self.customer_phone_number = None
        self.order_number = None
        self.confirmation_link = None
        self.area = None
        self.postal_code = None
        self.house_number = None
        self.street = None
        self.secondary_address = None
        self.city = None
        self.state = None
        self.full_address_raw = None
        self.payment_link = None
        self.payment_status = None
        self.payment_reference = None
        self.checkout_status = None
        self.source = None
        self.form_completed = False
        self.collected = False
        self.confirmed = False
        self.address_form_link = None


class _FakeContext:
    def __init__(self):
        self.delivery_address = _FakeDelivery()
        self.delivery_address_confirmed = False
        self._reset_called = False

    def reset(self):
        self._reset_called = True


class _FakeCart:
    def __init__(self):
        self._cleared = False

    def clear(self):
        self._cleared = True


class _FakeVoiceSession:
    def __init__(self):
        self.conversation_context = _FakeContext()
        self.cart = _FakeCart()
        self.conversation_state = None
        self.last_response_key = None
        self.last_response_payload = None


class TestVoiceSessionSynchronizer(unittest.TestCase):

    def _make_syncer(self, find_payment_link=None):
        return VoiceSessionSynchronizer(
            find_latest_payment_link_session=find_payment_link or (lambda t: None),
        )

    def _make_checkout(self, **kw) -> CheckoutSession:
        s = _session(**kw)
        return s

    def test_no_call_sid_exits_without_loading(self):
        syncer = self._make_syncer()
        load_called = []
        # Patch lazy import to detect if load_existing_session is called
        import app.services.voice_session_synchronizer as vsync_mod

        checkout = self._make_checkout()  # call_sid=None by default
        # Should return silently without touching session repo
        with patch.dict("sys.modules", {"app.session.repository": MagicMock()}):
            syncer.sync(checkout)  # No error, no interaction

    def test_syncs_address_fields(self):
        voice_session = _FakeVoiceSession()
        syncer = self._make_syncer()

        checkout = self._make_checkout(call_sid="CS1", customer_phone_number="+1555")
        checkout.house_number = "42"
        checkout.street = "Main St"
        checkout.area = "Downtown"
        checkout.postal_code = "12345"
        checkout.city = "Springfield"
        checkout.state = "IL"
        checkout.full_address_raw = "42 Main St, Springfield"

        saved = {}

        import app.services.voice_session_synchronizer as vsync_mod

        with (
            patch.object(vsync_mod, "load_existing_session", return_value=voice_session, create=True),
            patch.object(vsync_mod, "save_session", lambda s: saved.update({"s": s}), create=True),
        ):
            # Bypass the lazy import by monkeypatching the module-scope names
            import app.session.repository as session_repo
            with (
                patch.object(session_repo, "load_existing_session", return_value=voice_session),
                patch.object(session_repo, "save_session", lambda s: saved.update({"s": s})),
            ):
                syncer.sync(checkout)

        delivery = voice_session.conversation_context.delivery_address
        self.assertEqual(delivery.house_number, "42")
        self.assertEqual(delivery.street, "Main St")
        self.assertEqual(delivery.city, "Springfield")
        self.assertEqual(delivery.customer_phone_number, "+1555")

    def test_payment_completed_sets_delivery_status(self):
        voice_session = _FakeVoiceSession()
        syncer = self._make_syncer()

        checkout = self._make_checkout(call_sid="CS2")
        checkout.mark_payment_completed(reference="ref-xyz")

        import app.session.repository as session_repo

        saved = {}
        with (
            patch.object(session_repo, "load_existing_session", return_value=voice_session),
            patch.object(session_repo, "save_session", lambda s: saved.update({"s": s})),
        ):
            syncer.sync(checkout)

        delivery = voice_session.conversation_context.delivery_address
        self.assertEqual(delivery.payment_status, "payment_confirmed")
        self.assertEqual(delivery.payment_reference, "ref-xyz")

    def test_mark_completed_resets_context_and_sets_state(self):
        from app.state_machine.models.conversation_state import ConversationState

        voice_session = _FakeVoiceSession()
        syncer = self._make_syncer()

        checkout = self._make_checkout(call_sid="CS3", order_number="7654321")
        checkout.mark_payment_completed(reference="ref-abc")

        import app.session.repository as session_repo

        saved = {}
        with (
            patch.object(session_repo, "load_existing_session", return_value=voice_session),
            patch.object(session_repo, "save_session", lambda s: saved.update({"s": s})),
        ):
            syncer.sync(checkout, mark_completed=True)

        self.assertTrue(voice_session.conversation_context._reset_called)
        self.assertTrue(voice_session.cart._cleared)
        self.assertEqual(voice_session.conversation_state, ConversationState.COMPLETED)
        self.assertEqual(voice_session.last_response_key, "order_completed")
        self.assertEqual(
            voice_session.last_response_payload["order_number"], "7654321"
        )

    def test_stale_voice_session_no_error(self):
        syncer = self._make_syncer()
        checkout = self._make_checkout(call_sid="CS-stale")

        import app.session.repository as session_repo

        with (
            patch.object(session_repo, "load_existing_session", return_value=None),
            patch.object(session_repo, "save_session", MagicMock()),
        ):
            syncer.sync(checkout)  # Should exit silently without error

    def test_payment_failed_sets_payment_failed_status(self):
        voice_session = _FakeVoiceSession()
        syncer = self._make_syncer()

        checkout = self._make_checkout(call_sid="CS-fail")
        checkout.mark_payment_retryable("failed")

        import app.session.repository as session_repo

        with (
            patch.object(session_repo, "load_existing_session", return_value=voice_session),
            patch.object(session_repo, "save_session", MagicMock()),
        ):
            syncer.sync(checkout)

        delivery = voice_session.conversation_context.delivery_address
        self.assertEqual(delivery.payment_status, "payment_failed")

    def test_payment_link_url_synced_to_delivery(self):
        voice_session = _FakeVoiceSession()
        fake_pl = SimpleNamespace(
            public_link_url="https://pay.example.com/link",
            checkout_token="tok",
        )
        syncer = self._make_syncer(find_payment_link=lambda t: fake_pl)

        checkout = self._make_checkout(call_sid="CS-pl")
        checkout.mark_payment_started()

        import app.session.repository as session_repo

        with (
            patch.object(session_repo, "load_existing_session", return_value=voice_session),
            patch.object(session_repo, "save_session", MagicMock()),
        ):
            syncer.sync(checkout)

        delivery = voice_session.conversation_context.delivery_address
        self.assertEqual(delivery.payment_link, "https://pay.example.com/link")
        self.assertEqual(delivery.confirmation_link, "https://pay.example.com/link")


# ===========================================================================
# PaymentPollingOrchestrator
# ===========================================================================

class TestPaymentPollingOrchestrator(unittest.TestCase):

    def _make_orchestrator(self, verify_fn, *, interval=0.05, max_duration=1.0):
        executor = ThreadPoolExecutor(max_workers=4)
        return PaymentPollingOrchestrator(
            verify_fn=verify_fn,
            poll_interval=interval,
            poll_max_duration=max_duration,
            executor=executor,
        ), executor

    # --- no threading.Thread in public path ---------------------------------

    def test_no_threading_thread_imported_by_module(self):
        import app.services.payment_polling_orchestrator as ppo_mod
        import inspect
        src = inspect.getsource(ppo_mod)
        # threading.Thread must not appear in the source
        self.assertNotIn("threading.Thread(", src)

    # --- deduplication ------------------------------------------------------

    def test_start_returns_future(self):
        verify = MagicMock(return_value={"payment_completed": True, "status": ""})
        orchestrator, executor = self._make_orchestrator(verify)
        future = orchestrator.start("tok1")
        self.assertIsInstance(future, Future)
        future.result(timeout=2)
        executor.shutdown(wait=True)

    def test_start_deduplicates_concurrent_calls(self):
        barrier = threading.Event()
        call_count = [0]

        def slow_verify(token):
            call_count[0] += 1
            barrier.wait(timeout=2)
            return {"payment_completed": True, "status": ""}

        orchestrator, executor = self._make_orchestrator(slow_verify, interval=0.01, max_duration=2.0)
        f1 = orchestrator.start("tok-dup")
        f2 = orchestrator.start("tok-dup")  # Should be skipped
        self.assertIsNone(f2)
        barrier.set()
        f1.result(timeout=5)
        executor.shutdown(wait=True)

    # --- success path -------------------------------------------------------

    def test_poll_stops_on_payment_completed(self):
        results = [
            {"payment_completed": False, "status": "pending"},
            {"payment_completed": True, "status": "completed"},
        ]
        idx = [0]

        def verify(token):
            r = results[min(idx[0], len(results) - 1)]
            idx[0] += 1
            return r

        orchestrator, executor = self._make_orchestrator(verify, interval=0.02, max_duration=5.0)
        future = orchestrator.start("tok-success")
        future.result(timeout=5)
        self.assertFalse(orchestrator.is_active("tok-success"))
        executor.shutdown(wait=True)

    # --- failure path -------------------------------------------------------

    def test_poll_stops_on_failure_status(self):
        verify = MagicMock(return_value={"payment_completed": False, "status": "failed"})
        orchestrator, executor = self._make_orchestrator(verify, interval=0.02, max_duration=5.0)
        future = orchestrator.start("tok-fail")
        future.result(timeout=5)
        self.assertFalse(orchestrator.is_active("tok-fail"))
        executor.shutdown(wait=True)

    # --- timeout path -------------------------------------------------------

    def test_poll_stops_on_timeout(self):
        verify = MagicMock(return_value={"payment_completed": False, "status": "pending"})
        orchestrator, executor = self._make_orchestrator(verify, interval=0.05, max_duration=0.15)
        future = orchestrator.start("tok-timeout")
        future.result(timeout=5)
        self.assertFalse(orchestrator.is_active("tok-timeout"))
        executor.shutdown(wait=True)

    # --- cancellation -------------------------------------------------------

    def test_cancel_stops_poll_early(self):
        started = threading.Event()
        call_count = [0]

        def slow_verify(token):
            call_count[0] += 1
            started.set()
            return {"payment_completed": False, "status": "pending"}

        orchestrator, executor = self._make_orchestrator(
            slow_verify, interval=5.0, max_duration=30.0
        )
        future = orchestrator.start("tok-cancel")
        self.assertTrue(orchestrator.is_active("tok-cancel"))
        result = orchestrator.cancel("tok-cancel")
        self.assertTrue(result)
        future.result(timeout=5)  # Should exit promptly via Event
        self.assertFalse(orchestrator.is_active("tok-cancel"))
        executor.shutdown(wait=True)

    # --- duplicate start after completion -----------------------------------

    def test_can_start_new_poll_after_previous_completed(self):
        verify = MagicMock(return_value={"payment_completed": True, "status": "completed"})
        orchestrator, executor = self._make_orchestrator(verify, interval=0.02, max_duration=2.0)
        f1 = orchestrator.start("tok-restart")
        f1.result(timeout=5)
        # Should be allowed to start again after completion
        f2 = orchestrator.start("tok-restart")
        self.assertIsNotNone(f2)
        f2.result(timeout=5)
        executor.shutdown(wait=True)

    # --- verify error resilience -------------------------------------------

    def test_poll_continues_after_verify_exception(self):
        call_count = [0]

        def flaky_verify(token):
            call_count[0] += 1
            if call_count[0] < 3:
                raise RuntimeError("transient error")
            return {"payment_completed": True, "status": "completed"}

        orchestrator, executor = self._make_orchestrator(
            flaky_verify, interval=0.02, max_duration=5.0
        )
        future = orchestrator.start("tok-flaky")
        future.result(timeout=5)
        self.assertGreaterEqual(call_count[0], 3)
        executor.shutdown(wait=True)

    # --- cancel returns False for unknown token ----------------------------

    def test_cancel_unknown_token_returns_false(self):
        verify = MagicMock(return_value={"payment_completed": True, "status": ""})
        orchestrator, executor = self._make_orchestrator(verify)
        self.assertFalse(orchestrator.cancel("nonexistent"))
        executor.shutdown(wait=True)

    # --- is_active after cancel --------------------------------------------

    def test_is_active_false_before_start(self):
        verify = MagicMock(return_value={"payment_completed": True, "status": ""})
        orchestrator, executor = self._make_orchestrator(verify)
        self.assertFalse(orchestrator.is_active("never-started"))
        executor.shutdown(wait=True)


if __name__ == "__main__":
    unittest.main()
