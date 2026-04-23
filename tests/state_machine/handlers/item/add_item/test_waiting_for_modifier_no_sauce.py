import unittest

from app.nlu.intent_resolution.intent import Intent
from app.state_machine.handlers.item.add_item.waiting_for_modifier_handler import (
    WaitingForModifierHandler,
)
from app.state_machine.models.conversation_context import ConversationContext
from app.state_machine.models.pending_item_models import (
    PendingAddItem,
    PendingModifierChoice,
    PendingModifierGroup,
)
from app.state_machine.models.conversation_state import ConversationState


def _build_context() -> ConversationContext:
    sauce = PendingModifierChoice(
        modifier_id="sauce",
        name="Sauce",
        group_id="mods",
        normalized_name="sauce",
    )
    onions = PendingModifierChoice(
        modifier_id="onions",
        name="Onions",
        group_id="mods",
        normalized_name="onions",
    )
    group = PendingModifierGroup(
        group_id="mods",
        name="Extras",
        is_required=False,
        min_selector=0,
        max_selector=1,
        choices=[sauce, onions],
        choices_by_modifier_id={"sauce": sauce, "onions": onions},
        choices_by_normalized_name={"sauce": [sauce], "onions": [onions]},
        choice_names=("Sauce", "Onions"),
        normalized_choice_names=("sauce", "onions"),
        top_choice_names=("Sauce", "Onions"),
    )

    context = ConversationContext()
    context.current_item_id = "burger"
    context.current_item_name = "Burger"
    context.pending_add_item = PendingAddItem(
        item_id="burger",
        item_name="Burger",
        modifier_groups=[group],
        modifier_groups_by_id={"mods": group},
        modifier_choice_by_id={"sauce": sauce, "onions": onions},
    )
    return context


class WaitingForModifierNoSauceTests(unittest.TestCase):
    def test_plain_no_skips_optional_modifier_group(self):
        handler = WaitingForModifierHandler()
        context = _build_context()

        result = handler.handle(
            intent=Intent.DENY,
            context=context,
            user_text="no",
            session=None,
        )

        self.assertEqual(result.next_state, ConversationState.WAITING_FOR_QUANTITY)
        self.assertEqual(result.response_key, "ask_for_quantity")
        self.assertIn("mods", context.skipped_modifier_groups)

    def test_no_sauce_is_treated_as_specific_modifier_removal(self):
        handler = WaitingForModifierHandler()
        context = _build_context()

        result = handler.handle(
            intent=Intent.DENY,
            context=context,
            user_text="no sauce",
            session=None,
        )

        self.assertEqual(result.next_state, ConversationState.WAITING_FOR_QUANTITY)
        self.assertEqual(result.response_key, "ask_for_quantity")
        selections = context.selected_modifier_groups["mods"]
        self.assertEqual(len(selections), 1)
        self.assertEqual(selections[0].modifier_id, "sauce")
        self.assertEqual(selections[0].action, "remove")

    def test_done_like_phrase_finishes_optional_modifier_group(self):
        handler = WaitingForModifierHandler()
        context = _build_context()

        result = handler.handle(
            intent=Intent.UNKNOWN,
            context=context,
            user_text="yeah thats good thanks",
            session=None,
        )

        self.assertEqual(result.next_state, ConversationState.WAITING_FOR_QUANTITY)
        self.assertEqual(result.response_key, "ask_for_quantity")


if __name__ == "__main__":
    unittest.main()
