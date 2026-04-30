import unittest

from app.nlu.intent_resolution.intent import Intent
from app.nlu.nlu_result import NLUResult, SlotValue
from app.state_machine.handlers.common.waiting_for_quantity_handler import (
    WaitingForQuantityHandler,
)
from app.state_machine.models.conversation_context import ConversationContext
from app.state_machine.models.conversation_state import ConversationState
from app.state_machine.models.pending_item_models import PendingAddItem


def _build_context(*, quantity_slot: int | None = None) -> ConversationContext:
    context = ConversationContext()
    context.current_item_id = "chicken_taco"
    context.current_item_name = "Chicken Taco"
    context.pending_add_item = PendingAddItem(
        item_id="chicken_taco",
        item_name="Chicken Taco",
    )
    context.current_prompt_field = "quantity"
    if quantity_slot is not None:
        context.last_slots = (
            SlotValue(name="QUANTITY", value=quantity_slot, raw=str(quantity_slot)),
        )
    else:
        context.last_slots = ()
    return context


# ---------------------------------------------------------------------------
# Regression tests: QUANTITY slot must win over AFFIRM / DENY intent labels
# ---------------------------------------------------------------------------

class QuantitySlotPrecedenceTests(unittest.TestCase):
    """
    Covers the bug where affirm/deny intent mis-classification caused the
    handler to re-ask for quantity even though QUANTITY slot was present.

    Log evidence (turns 10-13):
      user: "Two."    intent: conversation/affirm  slots: QUANTITY=2  → looped
      user: "I said two"  intent: cart/deny        slots: QUANTITY=2  → looped
    """

    def test_quantity_accepted_when_intent_is_affirm(self):
        """'Two.' → affirm intent + QUANTITY=2 slot → quantity consumed, flow advances."""
        handler = WaitingForQuantityHandler()
        context = _build_context(quantity_slot=2)

        result = handler.handle(
            intent=Intent.AFFIRM,
            context=context,
            user_text="Two.",
            session=None,
        )

        self.assertNotEqual(result.next_state, ConversationState.WAITING_FOR_QUANTITY,
                            "Handler must not loop back when QUANTITY slot is present")
        self.assertNotIn(result.response_key, {"ask_for_quantity", "invalid_quantity_option"},
                         "Must not re-ask when slot provides a valid quantity")
        self.assertEqual(context.quantity, 2)

    def test_quantity_accepted_when_intent_is_deny(self):
        """'I said two' → deny intent + QUANTITY=2 slot → quantity consumed, flow advances."""
        handler = WaitingForQuantityHandler()
        context = _build_context(quantity_slot=2)

        result = handler.handle(
            intent=Intent.DENY,
            context=context,
            user_text="I said two",
            session=None,
        )

        self.assertNotEqual(result.next_state, ConversationState.WAITING_FOR_QUANTITY)
        self.assertNotIn(result.response_key, {"ask_for_quantity", "invalid_quantity_option"})
        self.assertEqual(context.quantity, 2)

    def test_quantity_accepted_when_intent_is_unknown_and_slot_present(self):
        """QUANTITY=2 slot must be consumed regardless of UNKNOWN intent."""
        handler = WaitingForQuantityHandler()
        context = _build_context(quantity_slot=2)

        result = handler.handle(
            intent=Intent.UNKNOWN,
            context=context,
            user_text="Two.",
            session=None,
        )

        self.assertNotEqual(result.next_state, ConversationState.WAITING_FOR_QUANTITY)
        self.assertEqual(context.quantity, 2)

    def test_quantity_slot_one_accepted(self):
        """Edge: QUANTITY=1 is valid and must advance the flow."""
        handler = WaitingForQuantityHandler()
        context = _build_context(quantity_slot=1)

        result = handler.handle(
            intent=Intent.AFFIRM,
            context=context,
            user_text="one",
            session=None,
        )

        self.assertNotEqual(result.next_state, ConversationState.WAITING_FOR_QUANTITY)
        self.assertEqual(context.quantity, 1)

    def test_word_quantity_text_accepted_without_slot(self):
        """'Three' as plain text (no slot) → normalize_quantity parses it → accepted."""
        handler = WaitingForQuantityHandler()
        context = _build_context(quantity_slot=None)

        result = handler.handle(
            intent=Intent.UNKNOWN,
            context=context,
            user_text="three",
            session=None,
        )

        self.assertNotEqual(result.next_state, ConversationState.WAITING_FOR_QUANTITY)
        self.assertEqual(context.quantity, 3)

    def test_numeric_text_accepted_without_slot(self):
        """'2' as plain text (no slot) → accepted."""
        handler = WaitingForQuantityHandler()
        context = _build_context(quantity_slot=None)

        result = handler.handle(
            intent=Intent.UNKNOWN,
            context=context,
            user_text="2",
            session=None,
        )

        self.assertNotEqual(result.next_state, ConversationState.WAITING_FOR_QUANTITY)
        self.assertEqual(context.quantity, 2)


# ---------------------------------------------------------------------------
# Fallback: no quantity slot, affirm/deny → must still re-ask
# ---------------------------------------------------------------------------

class NoQuantitySlotIntentFallbackTests(unittest.TestCase):

    def test_affirm_without_slot_reasks(self):
        """Pure affirm with no quantity slot → handler must re-ask."""
        handler = WaitingForQuantityHandler()
        context = _build_context(quantity_slot=None)

        result = handler.handle(
            intent=Intent.AFFIRM,
            context=context,
            user_text="yeah",
            session=None,
        )

        self.assertEqual(result.next_state, ConversationState.WAITING_FOR_QUANTITY)
        self.assertEqual(result.response_key, "ask_for_quantity")
        self.assertIsNone(context.quantity)

    def test_deny_without_slot_reasks(self):
        """Pure deny with no quantity slot → handler must re-ask."""
        handler = WaitingForQuantityHandler()
        context = _build_context(quantity_slot=None)

        result = handler.handle(
            intent=Intent.DENY,
            context=context,
            user_text="no",
            session=None,
        )

        self.assertEqual(result.next_state, ConversationState.WAITING_FOR_QUANTITY)
        self.assertEqual(result.response_key, "ask_for_quantity")
        self.assertIsNone(context.quantity)


# ---------------------------------------------------------------------------
# Invalid / zero quantity
# ---------------------------------------------------------------------------

class InvalidQuantityTests(unittest.TestCase):

    def test_zero_quantity_slot_returns_invalid_prompt(self):
        """QUANTITY=0 from slot → invalid_quantity_option, not accepted."""
        handler = WaitingForQuantityHandler()
        context = _build_context(quantity_slot=0)

        result = handler.handle(
            intent=Intent.UNKNOWN,
            context=context,
            user_text="zero",
            session=None,
        )

        self.assertEqual(result.next_state, ConversationState.WAITING_FOR_QUANTITY)
        self.assertEqual(result.response_key, "invalid_quantity_option")
        self.assertIsNone(context.quantity)

    def test_no_quantity_at_all_reasks(self):
        """Completely unrecognised text, no slot → re-ask."""
        handler = WaitingForQuantityHandler()
        context = _build_context(quantity_slot=None)

        result = handler.handle(
            intent=Intent.UNKNOWN,
            context=context,
            user_text="hmm",
            session=None,
        )

        self.assertEqual(result.next_state, ConversationState.WAITING_FOR_QUANTITY)
        self.assertEqual(result.response_key, "ask_for_quantity")
        self.assertIsNone(context.quantity)


# ---------------------------------------------------------------------------
# Existing tests (must remain green)
# ---------------------------------------------------------------------------

class ExistingFlowTests(unittest.TestCase):

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
