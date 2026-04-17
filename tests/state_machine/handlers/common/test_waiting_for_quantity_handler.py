import unittest

from app.nlu.intent_resolution.intent import Intent
from app.state_machine.handlers.common.waiting_for_quantity_handler import (
    WaitingForQuantityHandler,
)
from app.state_machine.models.conversation_context import ConversationContext
from app.state_machine.models.conversation_state import ConversationState
from app.state_machine.models.pending_item_models import PendingAddItem


def _build_context() -> ConversationContext:
    context = ConversationContext()
    context.current_item_id = "burger"
    context.current_item_name = "Burger"
    context.pending_add_item = PendingAddItem(
        item_id="burger",
        item_name="Burger",
    )
    context.current_prompt_field = "quantity"
    return context


class WaitingForQuantityHandlerTests(unittest.TestCase):
    def test_non_quantity_phrase_with_article_reasks_instead_of_taking_one(self):
        handler = WaitingForQuantityHandler()
        context = _build_context()

        result = handler.handle(
            intent=Intent.UNKNOWN,
            context=context,
            user_text="a small",
            session=None,
        )

        self.assertEqual(result.next_state, ConversationState.WAITING_FOR_QUANTITY)
        self.assertEqual(result.response_key, "ask_for_quantity")
        self.assertIsNone(context.quantity)

    def test_new_add_request_is_not_mistaken_for_quantity_one(self):
        handler = WaitingForQuantityHandler()
        context = _build_context()

        result = handler.handle(
            intent=Intent.ADD_ITEM,
            context=context,
            user_text="add a coke",
            session=None,
        )

        self.assertEqual(
            result.next_state,
            ConversationState.CANCELLATION_CONFIRMATION,
        )
        self.assertEqual(
            result.response_key,
            "confirm_cancel_current_item_for_new_request",
        )
        self.assertIsNone(context.quantity)


if __name__ == "__main__":
    unittest.main()
