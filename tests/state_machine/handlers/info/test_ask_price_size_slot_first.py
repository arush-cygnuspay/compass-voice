import logging
import unittest

from app.nlu.nlu_result import SlotValue
from app.state_machine.handlers.info.ask_price_handler import AskPriceHandler


class _SlotEventCapture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.events: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        if record.getMessage() in {
            "nlu_slot_consumed",
            "nlu_slot_fallback",
            "nlu_slot_missing",
        }:
            self.events.append(record)


class _StubMenuRepo:
    pass


class AskPriceSizeSlotFirstTests(unittest.TestCase):
    def setUp(self) -> None:
        self.capture = _SlotEventCapture()
        self.logger = logging.getLogger("app.nlu.slot_consumption")
        self.previous_level = self.logger.level
        self.logger.addHandler(self.capture)
        self.logger.setLevel(logging.INFO)

    def tearDown(self) -> None:
        self.logger.removeHandler(self.capture)
        self.logger.setLevel(self.previous_level)

    def _events(self, name: str) -> list[logging.LogRecord]:
        return [r for r in self.capture.events if r.getMessage() == name]

    def test_size_slot_consumed_without_regex(self):
        handler = AskPriceHandler(_StubMenuRepo())
        # The normalized text contains no recognizable size word; only
        # the slot supplies it.
        result = handler._extract_requested_size(
            "how much is the chicken sandwich",
            (SlotValue(name="SIZE", value="medium"),),
        )

        self.assertEqual(result, "medium")
        consumed = self._events("nlu_slot_consumed")
        fallback = self._events("nlu_slot_fallback")
        self.assertEqual(len(consumed), 1)
        self.assertEqual(getattr(consumed[0], "slot", None), "SIZE")
        self.assertEqual(
            getattr(consumed[0], "consumer_site", None),
            "ask_price_handler.size",
        )
        self.assertEqual(len(fallback), 0)

    def test_size_falls_back_to_regex_when_slot_absent(self):
        handler = AskPriceHandler(_StubMenuRepo())
        result = handler._extract_requested_size(
            "how much is the large pizza",
            (),
        )
        self.assertEqual(result, "large")
        self.assertEqual(len(self._events("nlu_slot_consumed")), 0)
        self.assertEqual(len(self._events("nlu_slot_fallback")), 1)


if __name__ == "__main__":
    unittest.main()
