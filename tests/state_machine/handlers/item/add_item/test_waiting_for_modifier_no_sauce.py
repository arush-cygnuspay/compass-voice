import unittest

from app.nlu.intent_resolution.intent import Intent
from app.nlu.nlu_result import SlotValue
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


def _build_context_with_multi_select_modifiers() -> ConversationContext:
    cheese = PendingModifierChoice(
        modifier_id="cheese",
        name="Cheese",
        group_id="mods",
        normalized_name="cheese",
    )
    onions = PendingModifierChoice(
        modifier_id="onions",
        name="Onions",
        group_id="mods",
        normalized_name="onions",
    )
    group = PendingModifierGroup(
        group_id="mods",
        name="Burger Modification",
        is_required=True,
        min_selector=2,
        max_selector=3,
        choices=[cheese, onions],
        choices_by_modifier_id={"cheese": cheese, "onions": onions},
        choices_by_normalized_name={"cheese": [cheese], "onions": [onions]},
        choice_names=("Cheese", "Onions"),
        normalized_choice_names=("cheese", "onions"),
        top_choice_names=("Cheese", "Onions"),
    )

    context = ConversationContext()
    context.current_item_id = "burger"
    context.current_item_name = "Burger"
    context.pending_add_item = PendingAddItem(
        item_id="burger",
        item_name="Burger",
        modifier_groups=[group],
        modifier_groups_by_id={"mods": group},
        modifier_choice_by_id={"cheese": cheese, "onions": onions},
    )
    return context


def _build_context_with_bun_modifiers() -> ConversationContext:
    whole_wheat = PendingModifierChoice(
        modifier_id="whole_wheat_bun",
        name="Whole Wheat Bun",
        group_id="bun_mods",
        normalized_name="whole wheat bun",
        match_texts=("whole wheat bun", "whole wheat", "wheat bun"),
    )
    gluten_free = PendingModifierChoice(
        modifier_id="gluten_free_bun",
        name="Gluten-Free Bun",
        group_id="bun_mods",
        normalized_name="gluten free bun",
        match_texts=("gluten free bun", "gluten free"),
    )
    group = PendingModifierGroup(
        group_id="bun_mods",
        name="Bun Modification",
        is_required=True,
        min_selector=1,
        max_selector=1,
        choices=[whole_wheat, gluten_free],
        choices_by_modifier_id={
            "whole_wheat_bun": whole_wheat,
            "gluten_free_bun": gluten_free,
        },
        choices_by_normalized_name={
            "whole wheat bun": [whole_wheat],
            "gluten free bun": [gluten_free],
        },
        choice_names=("Whole Wheat Bun", "Gluten-Free Bun"),
        normalized_choice_names=("whole wheat bun", "gluten free bun"),
        top_choice_names=("Whole Wheat Bun", "Gluten-Free Bun"),
    )

    context = ConversationContext()
    context.current_item_id = "burger"
    context.current_item_name = "Chicken Burger"
    context.pending_add_item = PendingAddItem(
        item_id="burger",
        item_name="Chicken Burger",
        modifier_groups=[group],
        modifier_groups_by_id={"bun_mods": group},
        modifier_choice_by_id={
            "whole_wheat_bun": whole_wheat,
            "gluten_free_bun": gluten_free,
        },
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

        self.assertEqual(result.next_state, ConversationState.IDLE)
        self.assertEqual(result.response_key, "item_added_successfully")
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

        self.assertEqual(result.next_state, ConversationState.IDLE)
        self.assertEqual(result.response_key, "item_added_successfully")
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

        self.assertEqual(result.next_state, ConversationState.IDLE)
        self.assertEqual(result.response_key, "item_added_successfully")

    def test_i_am_done_finishes_optional_modifier_group(self):
        handler = WaitingForModifierHandler()
        context = _build_context()

        result = handler.handle(
            intent=Intent.UNKNOWN,
            context=context,
            user_text="i am done",
            session=None,
        )

        self.assertEqual(result.next_state, ConversationState.IDLE)
        self.assertEqual(result.response_key, "item_added_successfully")

    def test_over_max_modifier_reply_requires_clarification_without_auto_selecting(self):
        handler = WaitingForModifierHandler()
        context = _build_context()

        result = handler.handle(
            intent=Intent.UNKNOWN,
            context=context,
            user_text="sauce and onions",
            session=None,
        )

        self.assertEqual(result.next_state, ConversationState.WAITING_FOR_MODIFIER)
        self.assertEqual(result.response_key, "too_many_modifier_choices")
        self.assertNotIn("mods", context.selected_modifier_groups)

    def test_invalid_modifier_reprompts_without_losing_active_group_state(self):
        handler = WaitingForModifierHandler()
        context = _build_context()

        result = handler.handle(
            intent=Intent.UNKNOWN,
            context=context,
            user_text="pepperoni",
            session=None,
        )

        self.assertEqual(result.next_state, ConversationState.WAITING_FOR_MODIFIER)
        self.assertEqual(result.response_key, "repeat_modifier_options")
        self.assertEqual(context.current_modifier_group_index, 0)
        self.assertNotIn("mods", context.skipped_modifier_groups)
        self.assertNotIn("mods", context.selected_modifier_groups)

    def test_remove_and_extra_modifiers_are_captured_together(self):
        handler = WaitingForModifierHandler()
        context = _build_context_with_multi_select_modifiers()

        result = handler.handle(
            intent=Intent.UNKNOWN,
            context=context,
            user_text="no onions and extra cheese",
            session=None,
        )

        self.assertEqual(result.next_state, ConversationState.IDLE)
        self.assertEqual(result.response_key, "item_added_successfully")
        selections = context.selected_modifier_groups["mods"]
        self.assertEqual(
            [(selection.modifier_id, selection.action, selection.instruction) for selection in selections],
            [("onions", "remove", None), ("cheese", "add", "extra")],
        )

    def test_invalid_modifier_keeps_valid_selection_when_group_is_still_missing(self):
        handler = WaitingForModifierHandler()
        context = _build_context_with_multi_select_modifiers()

        result = handler.handle(
            intent=Intent.UNKNOWN,
            context=context,
            user_text="extra cheese and pepperoni",
            session=None,
        )

        self.assertEqual(result.next_state, ConversationState.WAITING_FOR_MODIFIER)
        self.assertEqual(result.response_key, "repeat_modifier_options")
        selections = context.selected_modifier_groups["mods"]
        self.assertEqual(len(selections), 1)
        self.assertEqual(selections[0].modifier_id, "cheese")
        self.assertEqual(selections[0].instruction, "extra")


    def test_no_i_do_not_want_any_skips_optional_modifier_group(self):
        handler = WaitingForModifierHandler()
        context = _build_context()

        result = handler.handle(
            intent=Intent.UNKNOWN,
            context=context,
            user_text="no, i don't want any",
            session=None,
        )

        self.assertEqual(result.next_state, ConversationState.IDLE)
        self.assertEqual(result.response_key, "item_added_successfully")
        self.assertIn("mods", context.skipped_modifier_groups)

    def test_skip_it_skips_optional_modifier_group(self):
        handler = WaitingForModifierHandler()
        context = _build_context()

        result = handler.handle(
            intent=Intent.UNKNOWN,
            context=context,
            user_text="skip it",
            session=None,
        )

        self.assertEqual(result.next_state, ConversationState.IDLE)
        self.assertEqual(result.response_key, "item_added_successfully")
        self.assertIn("mods", context.skipped_modifier_groups)

    def test_what_are_the_options_lists_current_modifier_options(self):
        handler = WaitingForModifierHandler()
        context = _build_context()

        result = handler.handle(
            intent=Intent.UNKNOWN,
            context=context,
            user_text="what are the options?",
            session=None,
        )

        self.assertEqual(result.next_state, ConversationState.WAITING_FOR_MODIFIER)
        self.assertEqual(result.response_key, "list_modifier_options")
        self.assertEqual(result.response_payload["group_name"], "Extras")

    def test_whats_available_lists_current_modifier_options(self):
        handler = WaitingForModifierHandler()
        context = _build_context()

        result = handler.handle(
            intent=Intent.UNKNOWN,
            context=context,
            user_text="what's available?",
            session=None,
        )

        self.assertEqual(result.next_state, ConversationState.WAITING_FOR_MODIFIER)
        self.assertEqual(result.response_key, "list_modifier_options")
        self.assertEqual(result.response_payload["group_name"], "Extras")
        self.assertNotIn("mods", context.skipped_modifier_groups)
        self.assertNotIn("mods", context.selected_modifier_groups)

    def test_list_them_lists_current_modifier_options(self):
        handler = WaitingForModifierHandler()
        context = _build_context()

        result = handler.handle(
            intent=Intent.UNKNOWN,
            context=context,
            user_text="list them",
            session=None,
        )

        self.assertEqual(result.next_state, ConversationState.WAITING_FOR_MODIFIER)
        self.assertEqual(result.response_key, "list_modifier_options")
        self.assertEqual(result.response_payload["group_name"], "Extras")

    def test_repeat_that_repeats_modifier_prompt(self):
        handler = WaitingForModifierHandler()
        context = _build_context()

        result = handler.handle(
            intent=Intent.UNKNOWN,
            context=context,
            user_text="repeat that",
            session=None,
        )

        self.assertEqual(result.next_state, ConversationState.WAITING_FOR_MODIFIER)
        self.assertEqual(result.response_key, "repeat_modifier_options")
        self.assertEqual(result.response_payload["repeat_reason"], "meta_clarify")

    def test_slot_first_modifier_matching_resolves_bun_options_from_noisy_utterance(self):
        handler = WaitingForModifierHandler()
        cases = (
            ("oka wheat bun", SlotValue(name="VARIANT", value="wheat bun"), "whole_wheat_bun"),
            ("wheat bun", SlotValue(name="VARIANT", value="wheat bun"), "whole_wheat_bun"),
            ("whole wheat", SlotValue(name="VARIANT", value="whole wheat"), "whole_wheat_bun"),
            ("gluten free bun", SlotValue(name="VARIANT", value="gluten free bun"), "gluten_free_bun"),
            ("oka gluten free bun", SlotValue(name="VARIANT", value="gluten free bun"), "gluten_free_bun"),
        )

        for user_text, slot, expected_modifier_id in cases:
            with self.subTest(user_text=user_text):
                context = _build_context_with_bun_modifiers()
                context.last_slots = (slot,)

                result = handler.handle(
                    intent=Intent.UNKNOWN,
                    context=context,
                    user_text=user_text,
                    session=None,
                )

                self.assertEqual(result.next_state, ConversationState.IDLE)
                self.assertEqual(result.response_key, "item_added_successfully")
                selections = context.selected_modifier_groups["bun_mods"]
                self.assertEqual([selection.modifier_id for selection in selections], [expected_modifier_id])

    def test_invalid_modifier_feedback_prefers_slot_candidate_over_noisy_raw_utterance(self):
        handler = WaitingForModifierHandler()
        context = _build_context_with_bun_modifiers()
        context.last_slots = (SlotValue(name="VARIANT", value="oat bun"),)

        result = handler.handle(
            intent=Intent.UNKNOWN,
            context=context,
            user_text="oka oat bun",
            session=None,
        )

        self.assertEqual(result.next_state, ConversationState.WAITING_FOR_MODIFIER)
        self.assertEqual(result.response_key, "repeat_modifier_options")
        self.assertEqual(result.response_payload["unmatched_names"], ["oat bun"])
        self.assertEqual(result.response_payload["selected_candidate"], "oat bun")
        self.assertEqual(result.response_payload["match_source"], "slot_value")


if __name__ == "__main__":
    unittest.main()
