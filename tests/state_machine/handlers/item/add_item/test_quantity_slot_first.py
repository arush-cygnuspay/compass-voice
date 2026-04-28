import logging
import unittest

from app.nlu.nlu_result import NLUResult, SlotValue
from app.nlu.intent_resolution.intent import Intent
from app.state_machine.handlers.item.add_item.add_item_handler import (
    AddItemHandler,
    PendingItemCaptureHelper,
)
from app.state_machine.models.conversation_context import ConversationContext


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


class QuantitySlotFirstTests(unittest.TestCase):
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

    def test_quantity_slot_consumed_without_regex_fallback(self):
        handler = AddItemHandler(_StubMenuRepo())
        context = ConversationContext()
        # An ambiguous user_text that the regex path could not parse to
        # a positive int, ensuring success comes from the slot path only.
        context.last_user_text = "i wanted some"
        context.last_slots = (SlotValue(name="QUANTITY", value="3"),)

        result = handler.capture_helper._infer_quantity_from_text(
            context=context,
            user_text="i wanted some",
        )

        self.assertEqual(result, 3)
        consumed = self._events("nlu_slot_consumed")
        fallback = self._events("nlu_slot_fallback")
        missing = self._events("nlu_slot_missing")
        self.assertEqual(len(consumed), 1)
        self.assertEqual(getattr(consumed[0], "slot", None), "QUANTITY")
        self.assertEqual(
            getattr(consumed[0], "consumer_site", None),
            "add_item_handler.quantity",
        )
        self.assertEqual(len(fallback), 0)
        self.assertEqual(len(missing), 0)

    def test_quantity_falls_back_to_regex_when_slot_absent(self):
        handler = AddItemHandler(_StubMenuRepo())
        context = ConversationContext()
        context.last_slots = ()

        result = handler.capture_helper._infer_quantity_from_text(
            context=context,
            user_text="3",
        )

        self.assertEqual(result, 3)
        consumed = self._events("nlu_slot_consumed")
        fallback = self._events("nlu_slot_fallback")
        self.assertEqual(len(consumed), 0)
        self.assertEqual(len(fallback), 1)
        self.assertEqual(
            getattr(fallback[0], "consumer_site", None),
            "add_item_handler.quantity",
        )


class SizeSlotFirstTests(unittest.TestCase):
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

    def test_add_item_size_slot_consumed_without_regex(self):
        # User text contains no recognizable size word — regex would
        # return None — but the slot supplies "large".
        result = PendingItemCaptureHelper.extract_requested_size(
            user_text="i want one please",
            slots=(SlotValue(name="SIZE", value="large"),),
        )
        self.assertEqual(result, "large")
        consumed = self._events("nlu_slot_consumed")
        fallback = self._events("nlu_slot_fallback")
        self.assertEqual(len(consumed), 1)
        self.assertEqual(getattr(consumed[0], "slot", None), "SIZE")
        self.assertEqual(
            getattr(consumed[0], "consumer_site", None),
            "add_item_handler.size",
        )
        self.assertEqual(len(fallback), 0)

    def test_add_item_size_falls_back_to_regex(self):
        result = PendingItemCaptureHelper.extract_requested_size(
            user_text="i want a small one",
            slots=(),
        )
        self.assertEqual(result, "small")
        self.assertEqual(len(self._events("nlu_slot_consumed")), 0)
        self.assertEqual(len(self._events("nlu_slot_fallback")), 1)


if __name__ == "__main__":
    unittest.main()
