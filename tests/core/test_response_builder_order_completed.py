import unittest
from types import SimpleNamespace

from app.core.response_builder import ResponseBuilder
from app.state_machine.models.conversation_context import ConversationContext


class FakeStore:
    def get_item(self, item_id):
        return SimpleNamespace(name="Unused")


class FakeMenuRepo:
    def __init__(self):
        self.store = FakeStore()


class ResponseBuilderOrderCompletedTests(unittest.TestCase):
    def test_order_completed_includes_order_number_from_payload(self):
        builder = ResponseBuilder(FakeMenuRepo())
        context = ConversationContext()

        text = builder.build(
            "order_completed",
            context,
            {"order_number": "1234567"},
        )

        self.assertIn("Payment confirmed.", text)
        self.assertIn("Your order number is 1 2 3 4 5 6 7.", text)

    def test_order_completed_falls_back_to_context_order_number(self):
        builder = ResponseBuilder(FakeMenuRepo())
        context = ConversationContext()
        context.delivery_address.order_number = "7654321"

        text = builder.build("order_completed", context, {})

        self.assertIn("Your order number is 7 6 5 4 3 2 1.", text)


if __name__ == "__main__":
    unittest.main()
