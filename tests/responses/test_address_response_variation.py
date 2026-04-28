import unittest
from types import SimpleNamespace

from app.core.response_builder import ResponseBuilder
from app.state_machine.models.conversation_context import ConversationContext


class _FakeStore:
    def get_item(self, item_id):
        return SimpleNamespace(name="Unused")


class _FakeMenuRepo:
    def __init__(self):
        self.store = _FakeStore()


class AddressResponseVariationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.builder = ResponseBuilder(_FakeMenuRepo())
        self.context = ConversationContext()

    def _build(self, key: str, attempt_count: int) -> str:
        return self.builder.build(key, self.context, {"attempt_count": attempt_count})

    def test_repeat_delivery_house_number_varies_across_three_attempts(self):
        first = self._build("repeat_delivery_house_number", 1)
        second = self._build("repeat_delivery_house_number", 2)
        third = self._build("repeat_delivery_house_number", 3)
        self.assertNotEqual(first, second)
        self.assertNotEqual(second, third)
        self.assertNotEqual(first, third)

    def test_repeat_delivery_street_varies_across_three_attempts(self):
        first = self._build("repeat_delivery_street", 1)
        second = self._build("repeat_delivery_street", 2)
        third = self._build("repeat_delivery_street", 3)
        self.assertNotEqual(first, second)
        self.assertNotEqual(second, third)
        self.assertNotEqual(first, third)

    def test_repeat_delivery_secondary_address_varies_across_three_attempts(self):
        first = self._build("repeat_delivery_secondary_address", 1)
        second = self._build("repeat_delivery_secondary_address", 2)
        third = self._build("repeat_delivery_secondary_address", 3)
        self.assertNotEqual(first, second)
        self.assertNotEqual(second, third)
        self.assertNotEqual(first, third)

    def test_address_collection_giving_up_returns_handoff_text(self):
        text = self.builder.build(
            "address_collection_giving_up",
            self.context,
            {"field_name": "delivery_house_number", "attempt_count": 3},
        )
        self.assertIn("trouble", text.lower())
        self.assertIn("team member", text.lower())


if __name__ == "__main__":
    unittest.main()
