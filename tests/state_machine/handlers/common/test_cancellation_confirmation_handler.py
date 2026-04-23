import unittest

from app.nlu.intent_resolution.intent import Intent
from app.nlu.nlu_result import SlotValue
from app.session.session import Session
from app.state_machine.handlers.common.cancellation_confirmation_handler import (
    CancellationConfirmationHandler,
)
from app.state_machine.models.conversation_context import ConversationContext, InterruptProposal
from app.state_machine.models.conversation_state import ConversationState
from app.state_machine.models.pending_item_models import PendingAddItem


def _build_session_and_context() -> tuple[Session, ConversationContext]:
    session = Session(session_id="test-session", restaurant_id="demo")
    session.conversation_state = ConversationState.CANCELLATION_CONFIRMATION

    context = session.conversation_context
    context.current_item_id = "burger"
    context.current_item_name = "Burger"
    context.current_prompt_field = "quantity"
    context.pending_add_item = PendingAddItem(
        item_id="burger",
        item_name="Burger",
    )
    context.return_state = ConversationState.WAITING_FOR_QUANTITY
    context.awaiting_flow_confirmation = True
    context.interrupt_proposal = InterruptProposal(
        text="add a coke",
        predicted_main_intent="cart",
        predicted_sub_intent="add_item",
    )
    return session, context


class CancellationConfirmationHandlerTests(unittest.TestCase):
    def test_quantity_like_reply_does_not_confirm_cancellation(self):
        handler = CancellationConfirmationHandler()
        session, context = _build_session_and_context()
        context.last_slots = (SlotValue(name="QUANTITY", value="1"),)

        result = handler.handle(
            intent=Intent.CONFIRM,
            context=context,
            user_text="one",
            session=session,
        )

        self.assertEqual(result.next_state, ConversationState.IDLE)
        self.assertEqual(result.response_key, "item_added_successfully")
        self.assertIsNotNone(result.command)
        self.assertEqual(result.command["type"], "ADD_ITEM_TO_CART")
        self.assertEqual(result.command["payload"]["quantity"], 1)
        self.assertFalse(context.awaiting_flow_confirmation)
        self.assertIsNone(context.return_state)
        self.assertIsNone(context.interrupt_proposal)

    def test_deny_clears_confirmation_overlay_state(self):
        handler = CancellationConfirmationHandler()
        session, context = _build_session_and_context()

        result = handler.handle(
            intent=Intent.DENY,
            context=context,
            user_text="no",
            session=session,
        )

        self.assertEqual(result.next_state, ConversationState.WAITING_FOR_QUANTITY)
        self.assertEqual(
            result.response_key,
            "continue_current_item_after_cancel_denied",
        )
        self.assertFalse(context.awaiting_flow_confirmation)
        self.assertIsNone(context.return_state)
        self.assertIsNone(context.interrupt_proposal)


if __name__ == "__main__":
    unittest.main()
