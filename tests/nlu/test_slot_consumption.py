import logging
import unittest

from app.nlu.nlu_result import SlotValue
from app.nlu.slot_consumption import (
    SlotResolution,
    consume_slot_or_fallback,
)


class _EventCapture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.events: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.events.append(record)


class _NoopFallback:
    def __init__(self, value=None):
        self.value = value
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.value


class SlotConsumptionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.capture = _EventCapture()
        self.logger = logging.getLogger("app.nlu.slot_consumption")
        self.previous_level = self.logger.level
        self.logger.addHandler(self.capture)
        self.logger.setLevel(logging.INFO)

    def tearDown(self) -> None:
        self.logger.removeHandler(self.capture)
        self.logger.setLevel(self.previous_level)

    def _events(self, name: str) -> list[logging.LogRecord]:
        return [r for r in self.capture.events if r.getMessage() == name]

    def test_slot_present_and_parsable_returns_slot_source(self):
        slots = (SlotValue(name="QUANTITY", value="3"),)
        fallback = _NoopFallback()
        result = consume_slot_or_fallback(
            slots=slots,
            slot_labels=("QUANTITY",),
            fallback=fallback,
            parse=lambda s: int(s) if s.isdigit() else None,
            consumer_site="test.quantity",
        )
        self.assertIsInstance(result, SlotResolution)
        self.assertEqual(result.value, 3)
        self.assertEqual(result.source, "slot")
        self.assertEqual(fallback.calls, 0)
        consumed = self._events("nlu_slot_consumed")
        self.assertEqual(len(consumed), 1)
        self.assertEqual(getattr(consumed[0], "consumer_site", None), "test.quantity")
        self.assertEqual(getattr(consumed[0], "slot", None), "QUANTITY")

    def test_slot_missing_regex_returns_value_emits_fallback_event(self):
        fallback = _NoopFallback(value=5)
        result = consume_slot_or_fallback(
            slots=(),
            slot_labels=("QUANTITY",),
            fallback=fallback,
            consumer_site="test.quantity",
        )
        self.assertEqual(result.value, 5)
        self.assertEqual(result.source, "regex_fallback")
        self.assertEqual(fallback.calls, 1)
        events = self._events("nlu_slot_fallback")
        self.assertEqual(len(events), 1)
        self.assertEqual(
            getattr(events[0], "attempted_slots", None), ["QUANTITY"]
        )

    def test_slot_missing_regex_missing_emits_missing_event(self):
        fallback = _NoopFallback(value=None)
        result = consume_slot_or_fallback(
            slots=(),
            slot_labels=("SIZE",),
            fallback=fallback,
            consumer_site="test.size",
        )
        self.assertIsNone(result.value)
        self.assertEqual(result.source, "missing")
        self.assertEqual(fallback.calls, 1)
        events = self._events("nlu_slot_missing")
        self.assertEqual(len(events), 1)
        self.assertEqual(getattr(events[0], "attempted_slots", None), ["SIZE"])

    def test_first_slot_label_wins_second_not_consulted(self):
        slots = (
            SlotValue(name="SIZE", value="large"),
            SlotValue(name="VARIANT", value="medium"),
        )
        fallback = _NoopFallback()
        result = consume_slot_or_fallback(
            slots=slots,
            slot_labels=("SIZE", "VARIANT"),
            fallback=fallback,
            consumer_site="test.size",
        )
        self.assertEqual(result.value, "large")
        self.assertEqual(result.source, "slot")
        consumed = self._events("nlu_slot_consumed")
        self.assertEqual(len(consumed), 1)
        self.assertEqual(getattr(consumed[0], "slot", None), "SIZE")

    def test_slot_present_but_parse_returns_none_falls_through_to_regex(self):
        # slot value cannot be parsed as a positive int → we fall through.
        slots = (SlotValue(name="QUANTITY", value="not-a-number"),)
        fallback = _NoopFallback(value=7)
        result = consume_slot_or_fallback(
            slots=slots,
            slot_labels=("QUANTITY",),
            fallback=fallback,
            parse=lambda s: int(s) if s.isdigit() else None,
            consumer_site="test.quantity",
        )
        self.assertEqual(result.value, 7)
        self.assertEqual(result.source, "regex_fallback")
        self.assertEqual(fallback.calls, 1)
        # Only the fallback event fires, not nlu_slot_consumed.
        self.assertEqual(len(self._events("nlu_slot_consumed")), 0)
        self.assertEqual(len(self._events("nlu_slot_fallback")), 1)


if __name__ == "__main__":
    unittest.main()
