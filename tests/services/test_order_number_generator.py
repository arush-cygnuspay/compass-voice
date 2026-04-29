# tests/services/test_order_number_generator.py
"""Comprehensive tests for OrderNumberGenerator and RedisOrderNumberGenerator.

Validates:
- Generated numbers are always 7-digit all-numeric strings.
- Determinism under injected time/sequence.
- Uniqueness within the same centisecond burst.
- Sequence wraps mod 100 at boundary.
- RedisOrderNumberGenerator uses atomic INCR.
- RedisOrderNumberGenerator falls back to in-process generator on Redis error.
- No random.randint anywhere in the generated values.
- Process-restart safety (fresh generator, same centisecond → seq restarts at 0).
- CheckoutService passes through a valid caller-supplied order number.
- CheckoutService generates when caller supplies None or non-digits.
"""
from __future__ import annotations

import sys
import threading
import types
import unittest

# ---------------------------------------------------------------------------
# Minimal stubs so service/model imports succeed without external packages
# ---------------------------------------------------------------------------
for _m in ("redis",):
    if _m not in sys.modules:
        _stub = types.ModuleType(_m)
        _stub.Redis = object
        sys.modules[_m] = _stub

from app.services.order_number_generator import (
    OrderNumberGenerator,
    RedisOrderNumberGenerator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EPOCH_2024_MS = 1704067200_000  # 2024-01-01 00:00:00 UTC in milliseconds


def _fresh(*, now_ms: int = _EPOCH_2024_MS) -> str:
    return OrderNumberGenerator().generate(now_ms=now_ms)


# ---------------------------------------------------------------------------
# OrderNumberGenerator — format contract
# ---------------------------------------------------------------------------

class OrderNumberFormatTests(unittest.TestCase):

    def test_7_digits_only(self):
        result = _fresh()
        self.assertEqual(len(result), 7)
        self.assertTrue(result.isdigit(), f"Not all digits: {result!r}")

    def test_always_7_digits_at_various_timestamps(self):
        ms_values = [
            0,
            1,
            999,
            1_000_000,
            1704067200_000,   # 2024-01-01
            9999999999_999,   # far future
        ]
        gen = OrderNumberGenerator()
        for ms in ms_values:
            with self.subTest(ms=ms):
                r = gen.generate(now_ms=ms)
                self.assertEqual(len(r), 7, f"Bad length for ms={ms}: {r!r}")
                self.assertTrue(r.isdigit(), f"Not digits for ms={ms}: {r!r}")

    def test_no_random_import_in_generator(self):
        import app.services.order_number_generator as mod
        import inspect
        src = inspect.getsource(mod)
        # The module must not import or use the random module for generation.
        # (Mentions in docstrings/comments are acceptable historical notes.)
        self.assertNotIn("import random", src)


# ---------------------------------------------------------------------------
# OrderNumberGenerator — determinism and sequence
# ---------------------------------------------------------------------------

class OrderNumberDeterminismTests(unittest.TestCase):

    def test_same_ms_same_instance_increments_seq(self):
        gen = OrderNumberGenerator()
        r0 = gen.generate(now_ms=_EPOCH_2024_MS)
        r1 = gen.generate(now_ms=_EPOCH_2024_MS)
        self.assertNotEqual(r0, r1, "Same ms must yield different seq prefixes")
        # Both share the same ts_part
        self.assertEqual(r0[2:], r1[2:], "ts_part must be identical for same ms")
        self.assertEqual(int(r1[:2]), int(r0[:2]) + 1)

    def test_known_output_for_injected_time(self):
        # cs = 1704067200000 // 10 = 170406720000
        # ts_part = 170406720000 % 100_000 = 20000
        # first call: seq=0 → "0020000"
        gen = OrderNumberGenerator()
        result = gen.generate(now_ms=_EPOCH_2024_MS)
        self.assertEqual(result, "0020000")

    def test_new_centisecond_resets_seq_to_zero(self):
        gen = OrderNumberGenerator()
        ms_a = _EPOCH_2024_MS
        ms_b = _EPOCH_2024_MS + 10  # 1 centisecond later
        gen.generate(now_ms=ms_a)   # seq → 0 in bucket A
        gen.generate(now_ms=ms_a)   # seq → 1 in bucket A
        r_b = gen.generate(now_ms=ms_b)  # new bucket → seq resets to 0
        self.assertEqual(int(r_b[:2]), 0)

    def test_seq_wraps_at_100(self):
        gen = OrderNumberGenerator()
        for _ in range(100):
            gen.generate(now_ms=_EPOCH_2024_MS)
        # 101st call: seq wraps back to 0
        r = gen.generate(now_ms=_EPOCH_2024_MS)
        self.assertEqual(int(r[:2]), 0)

    def test_burst_100_calls_all_unique(self):
        gen = OrderNumberGenerator()
        results = [gen.generate(now_ms=_EPOCH_2024_MS) for _ in range(100)]
        self.assertEqual(len(set(results)), 100)

    def test_process_restart_restarts_seq(self):
        # A new generator instance always starts seq at 0 for the first bucket.
        gen1 = OrderNumberGenerator()
        gen2 = OrderNumberGenerator()
        r1 = gen1.generate(now_ms=_EPOCH_2024_MS)
        r2 = gen2.generate(now_ms=_EPOCH_2024_MS)
        self.assertEqual(r1, r2, "Fresh generators produce the same first value")

    def test_clock_ms_callable_is_used_when_now_ms_not_given(self):
        fixed_ms = _EPOCH_2024_MS
        gen = OrderNumberGenerator(clock_ms=lambda: fixed_ms)
        r = gen.generate()
        # Same as injecting now_ms directly
        expected = OrderNumberGenerator().generate(now_ms=fixed_ms)
        self.assertEqual(r, expected)


# ---------------------------------------------------------------------------
# OrderNumberGenerator — thread safety
# ---------------------------------------------------------------------------

class OrderNumberThreadSafetyTests(unittest.TestCase):

    def test_concurrent_burst_all_unique(self):
        gen = OrderNumberGenerator()
        results: list[str] = []
        lock = threading.Lock()

        def worker():
            r = gen.generate(now_ms=_EPOCH_2024_MS)
            with lock:
                results.append(r)

        threads = [threading.Thread(target=worker) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(set(results)), 50, "Concurrent calls must all be unique")


# ---------------------------------------------------------------------------
# RedisOrderNumberGenerator
# ---------------------------------------------------------------------------

class _FakeRedis:
    """Fake Redis client with atomic INCR simulation."""

    def __init__(self) -> None:
        self._counters: dict[str, int] = {}
        self._lock = threading.Lock()

    def incr(self, key: str) -> int:
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + 1
            return self._counters[key]

    def expire(self, key: str, ttl: int) -> None:
        pass


class _BrokenRedis:
    def incr(self, key: str) -> int:
        raise ConnectionError("Redis is down")

    def expire(self, key: str, ttl: int) -> None:
        pass


class RedisOrderNumberGeneratorTests(unittest.TestCase):

    def test_7_digits_all_numeric(self):
        gen = RedisOrderNumberGenerator(_FakeRedis())
        r = gen.generate(now_ms=_EPOCH_2024_MS)
        self.assertEqual(len(r), 7)
        self.assertTrue(r.isdigit())

    def test_atomic_incr_gives_unique_values(self):
        redis = _FakeRedis()
        gen = RedisOrderNumberGenerator(redis)
        results = [gen.generate(now_ms=_EPOCH_2024_MS) for _ in range(10)]
        self.assertEqual(len(set(results)), 10)

    def test_seq_based_on_incr_value(self):
        redis = _FakeRedis()
        gen = RedisOrderNumberGenerator(redis)
        # first INCR returns 1 → seq = (1-1) % 100 = 0
        r = gen.generate(now_ms=_EPOCH_2024_MS)
        self.assertEqual(int(r[:2]), 0)
        # second INCR returns 2 → seq = (2-1) % 100 = 1
        r2 = gen.generate(now_ms=_EPOCH_2024_MS)
        self.assertEqual(int(r2[:2]), 1)

    def test_fallback_when_redis_raises(self):
        gen = RedisOrderNumberGenerator(_BrokenRedis())
        r = gen.generate(now_ms=_EPOCH_2024_MS)
        self.assertEqual(len(r), 7)
        self.assertTrue(r.isdigit())

    def test_concurrent_burst_all_unique(self):
        redis = _FakeRedis()
        gen = RedisOrderNumberGenerator(redis)
        results: list[str] = []
        lock = threading.Lock()

        def worker():
            r = gen.generate(now_ms=_EPOCH_2024_MS)
            with lock:
                results.append(r)

        threads = [threading.Thread(target=worker) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(set(results)), 50)

    def test_different_restaurants_use_separate_keys(self):
        redis = _FakeRedis()
        gen = RedisOrderNumberGenerator(redis)
        r_a = gen.generate("restaurant_a", now_ms=_EPOCH_2024_MS)
        r_b = gen.generate("restaurant_b", now_ms=_EPOCH_2024_MS)
        # Both start seq at 0 because separate Redis keys
        self.assertEqual(int(r_a[:2]), 0)
        self.assertEqual(int(r_b[:2]), 0)
        self.assertTrue(any("restaurant_a" in k for k in redis._counters))
        self.assertTrue(any("restaurant_b" in k for k in redis._counters))


# ---------------------------------------------------------------------------
# Service-layer pass-through / generation decision
# ---------------------------------------------------------------------------

class ServiceLayerOrderNumberTests(unittest.TestCase):
    """Verify the validation logic that now lives in CheckoutService.create_session."""

    def _make_service(self, *, gen: OrderNumberGenerator | None = None):
        import tempfile, pathlib
        from app.services.checkout_service import CheckoutService
        from app.repositories.checkout_session_repository import CheckoutSessionRepository

        tmp = tempfile.mkdtemp()
        repo = CheckoutSessionRepository(
            data_dir=pathlib.Path(tmp) / "checkout",
            payment_link_session_dir=pathlib.Path(tmp) / "payment",
        )
        svc = CheckoutService(
            repository=repo,
            order_number_generator=gen or OrderNumberGenerator(),
        )
        return svc

    def _create(self, svc, *, order_number=None):
        return svc.create_session(
            restaurant_id="rest1",
            call_sid=None,
            order_number=order_number,
            customer_phone_number=None,
            address_required=False,
            area=None,
            postal_code=None,
        )

    def test_valid_caller_order_number_is_preserved(self):
        svc = self._make_service()
        session = self._create(svc, order_number="9876543")
        self.assertEqual(session.order_number, "9876543")

    def test_none_order_number_triggers_generation(self):
        gen = OrderNumberGenerator(clock_ms=lambda: _EPOCH_2024_MS)
        svc = self._make_service(gen=gen)
        session = self._create(svc, order_number=None)
        self.assertEqual(len(session.order_number), 7)
        self.assertTrue(session.order_number.isdigit())

    def test_non_digit_order_number_triggers_generation(self):
        svc = self._make_service()
        session = self._create(svc, order_number="TEST123")
        self.assertTrue(session.order_number.isdigit())
        self.assertEqual(len(session.order_number), 7)

    def test_empty_order_number_triggers_generation(self):
        svc = self._make_service()
        session = self._create(svc, order_number="")
        self.assertTrue(session.order_number.isdigit())

    def test_generated_scoped_to_restaurant_id(self):
        # RedisOrderNumberGenerator uses restaurant_id as part of key.
        redis = _FakeRedis()
        gen = RedisOrderNumberGenerator(redis)
        svc = self._make_service(gen=gen)
        self._create(svc, order_number=None)
        cs = _EPOCH_2024_MS // 10  # this won't match exactly but verify key exists
        # At least one key must be scoped to "rest1"
        self.assertTrue(
            any("rest1" in k for k in redis._counters),
            f"No rest1 key found; keys={list(redis._counters)}",
        )


if __name__ == "__main__":
    unittest.main()
