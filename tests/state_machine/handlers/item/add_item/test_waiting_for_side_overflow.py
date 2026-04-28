import unittest

from app.menu.models import MenuItem, Pricing, SideChoice, SideGroup
from app.nlu.intent_resolution.intent import Intent
from app.nlu.nlu_result import SlotValue
from app.nlu.query_normalization.text_preprocessor import normalize_text
from app.state_machine.handlers.item.add_item.pending_add_item_factory import build_pending_add_item
from app.state_machine.handlers.item.add_item.waiting_for_side_handler import (
    WaitingForSideHandler,
)
from app.state_machine.models.conversation_context import ConversationContext
from app.state_machine.models.pending_item_models import (
    PendingAddItem,
    PendingSideChoice,
    PendingSideGroup,
)
from app.state_machine.models.conversation_state import ConversationState


def _build_context() -> ConversationContext:
    fries = PendingSideChoice(
        item_id="fries",
        name="Fries",
        pricing_mode="fixed",
        normalized_name="fries",
    )
    salad = PendingSideChoice(
        item_id="salad",
        name="Salad",
        pricing_mode="fixed",
        normalized_name="salad",
    )
    group = PendingSideGroup(
        group_id="side",
        name="Side",
        is_required=True,
        min_selector=1,
        max_selector=1,
        choices=[fries, salad],
        choices_by_item_id={"fries": fries, "salad": salad},
        choices_by_normalized_name={"fries": [fries], "salad": [salad]},
        choice_names=("Fries", "Salad"),
        normalized_choice_names=("fries", "salad"),
        top_choice_names=("Fries", "Salad"),
    )

    context = ConversationContext()
    context.current_item_id = "burger"
    context.current_item_name = "Burger"
    context.pending_add_item = PendingAddItem(
        item_id="burger",
        item_name="Burger",
        side_groups=[group],
        side_groups_by_id={"side": group},
        side_choice_by_item_id={"fries": fries, "salad": salad},
    )
    return context


def _build_context_with_single_group(group: SideGroup) -> ConversationContext:
    item = MenuItem(
        item_id="burger",
        name="Burger",
        normalized_name=normalize_text("Burger"),
        aliases=(),
        normalized_aliases=(),
        voice_labels=(),
        pricing=Pricing(mode="fixed", price_cents=1000),
        side_groups=[group],
        modifier_groups=[],
        available=True,
    )
    context = ConversationContext()
    context.current_item_id = "burger"
    context.current_item_name = "Burger"
    context.pending_add_item = build_pending_add_item(item)
    return context


def _build_context_with_required_burger_groups() -> ConversationContext:
    item = MenuItem(
        item_id="burger",
        name="Burger",
        normalized_name=normalize_text("Burger"),
        aliases=(),
        normalized_aliases=(),
        voice_labels=(),
        pricing=Pricing(mode="fixed", price_cents=1000),
        side_groups=[
            SideGroup(
                group_id="cheese",
                name="Choose your cheese",
                normalized_name=normalize_text("Choose your cheese"),
                is_required=True,
                min_selector=1,
                max_selector=1,
                choices=[
                    SideChoice(
                        item_id="american",
                        name="American Cheese",
                        normalized_name=normalize_text("American Cheese"),
                        pricing=Pricing(mode="fixed", price_cents=0),
                    ),
                ],
            ),
            SideGroup(
                group_id="meat",
                name="Choose your meat",
                normalized_name=normalize_text("Choose your meat"),
                is_required=True,
                min_selector=1,
                max_selector=1,
                choices=[
                    SideChoice(
                        item_id="beef",
                        name="Beef Meat",
                        normalized_name=normalize_text("Beef Meat"),
                        pricing=Pricing(mode="fixed", price_cents=0),
                    ),
                ],
            ),
            SideGroup(
                group_id="bun",
                name="Choose your bun",
                normalized_name=normalize_text("Choose your bun"),
                is_required=True,
                min_selector=1,
                max_selector=1,
                choices=[
                    SideChoice(
                        item_id="plain",
                        name="Plain Bun",
                        normalized_name=normalize_text("Plain Bun"),
                        pricing=Pricing(mode="fixed", price_cents=0),
                    ),
                ],
            ),
        ],
        modifier_groups=[],
        available=True,
    )
    context = ConversationContext()
    context.current_item_id = "burger"
    context.current_item_name = "Burger"
    context.pending_add_item = build_pending_add_item(item)
    return context


def _build_context_with_bun_side_group() -> ConversationContext:
    plain = PendingSideChoice(
        item_id="plain",
        name="Plain Bun",
        pricing_mode="fixed",
        normalized_name="plain bun",
        match_texts=("plain bun", "plain"),
    )
    whole_wheat = PendingSideChoice(
        item_id="whole_wheat",
        name="Whole Wheat Bun",
        pricing_mode="fixed",
        normalized_name="whole wheat bun",
        match_texts=("whole wheat bun", "whole wheat", "wheat bun"),
    )
    gluten_free = PendingSideChoice(
        item_id="gluten_free",
        name="Gluten-Free Bun",
        pricing_mode="fixed",
        normalized_name="gluten free bun",
        match_texts=("gluten free bun", "gluten free"),
    )
    group = PendingSideGroup(
        group_id="bun",
        name="Choose Bun",
        is_required=True,
        min_selector=1,
        max_selector=1,
        choices=[plain, whole_wheat, gluten_free],
        choices_by_item_id={
            "plain": plain,
            "whole_wheat": whole_wheat,
            "gluten_free": gluten_free,
        },
        choices_by_normalized_name={
            "plain bun": [plain],
            "whole wheat bun": [whole_wheat],
            "gluten free bun": [gluten_free],
        },
        choice_names=("Plain Bun", "Whole Wheat Bun", "Gluten-Free Bun"),
        normalized_choice_names=("plain bun", "whole wheat bun", "gluten free bun"),
        top_choice_names=("Plain Bun", "Whole Wheat Bun", "Gluten-Free Bun"),
    )
    context = ConversationContext()
    context.current_item_id = "burger"
    context.current_item_name = "Chicken Burger"
    context.pending_add_item = PendingAddItem(
        item_id="burger",
        item_name="Chicken Burger",
        side_groups=[group],
        side_groups_by_id={"bun": group},
        side_choice_by_item_id={
            "plain": plain,
            "whole_wheat": whole_wheat,
            "gluten_free": gluten_free,
        },
    )
    return context


class WaitingForSideOverflowTests(unittest.TestCase):
    def test_over_max_side_reply_requires_clarification_without_auto_selecting(self):
        handler = WaitingForSideHandler()
        context = _build_context()

        result = handler.handle(
            intent=Intent.UNKNOWN,
            context=context,
            user_text="fries and salad",
            session=None,
        )

        self.assertEqual(result.next_state, ConversationState.WAITING_FOR_SIDE)
        self.assertEqual(result.response_key, "too_many_side_choices")
        self.assertNotIn("side", context.selected_side_groups)

    def test_contextual_meat_alias_resolves_beef_when_meat_is_expected(self):
        handler = WaitingForSideHandler()
        context = _build_context_with_single_group(
            SideGroup(
                group_id="meat",
                name="Choose your meat",
                normalized_name=normalize_text("Choose your meat"),
                is_required=True,
                min_selector=1,
                max_selector=1,
                choices=[
                    SideChoice(
                        item_id="beef",
                        name="Beef Meat",
                        normalized_name=normalize_text("Beef Meat"),
                        pricing=Pricing(mode="fixed", price_cents=0),
                    ),
                    SideChoice(
                        item_id="lamb",
                        name="Lamb Meat",
                        normalized_name=normalize_text("Lamb Meat"),
                        pricing=Pricing(mode="fixed", price_cents=0),
                    ),
                ],
            )
        )

        result = handler.handle(
            intent=Intent.UNKNOWN,
            context=context,
            user_text="beef",
            session=None,
        )

        self.assertEqual(result.next_state, ConversationState.WAITING_FOR_QUANTITY)
        self.assertEqual(result.response_key, "ask_for_quantity")
        self.assertEqual(context.selected_side_groups, {"meat": ["beef"]})

    def test_contextual_bun_alias_resolves_plain_when_bun_is_expected(self):
        handler = WaitingForSideHandler()
        context = _build_context_with_single_group(
            SideGroup(
                group_id="bun",
                name="Choose your bun",
                normalized_name=normalize_text("Choose your bun"),
                is_required=True,
                min_selector=1,
                max_selector=1,
                choices=[
                    SideChoice(
                        item_id="plain",
                        name="Plain Bun",
                        normalized_name=normalize_text("Plain Bun"),
                        pricing=Pricing(mode="fixed", price_cents=0),
                    ),
                    SideChoice(
                        item_id="sesame",
                        name="Sesame Bun",
                        normalized_name=normalize_text("Sesame Bun"),
                        pricing=Pricing(mode="fixed", price_cents=0),
                    ),
                ],
            )
        )

        result = handler.handle(
            intent=Intent.UNKNOWN,
            context=context,
            user_text="plain",
            session=None,
        )

        self.assertEqual(result.next_state, ConversationState.WAITING_FOR_QUANTITY)
        self.assertEqual(result.response_key, "ask_for_quantity")
        self.assertEqual(context.selected_side_groups, {"bun": ["plain"]})

    def test_multi_slot_side_capture_fills_remaining_required_groups(self):
        handler = WaitingForSideHandler()
        context = _build_context_with_required_burger_groups()

        result = handler.handle(
            intent=Intent.UNKNOWN,
            context=context,
            user_text="american cheese beef meat plain bun",
            session=None,
        )

        self.assertEqual(result.next_state, ConversationState.WAITING_FOR_QUANTITY)
        self.assertEqual(result.response_key, "ask_for_quantity")
        self.assertEqual(
            context.selected_side_groups,
            {
                "cheese": ["american"],
                "meat": ["beef"],
                "bun": ["plain"],
            },
        )

    def test_partial_side_capture_only_prompts_for_remaining_required_group(self):
        handler = WaitingForSideHandler()
        context = _build_context_with_required_burger_groups()

        result = handler.handle(
            intent=Intent.UNKNOWN,
            context=context,
            user_text="american cheese",
            session=None,
        )

        self.assertEqual(result.next_state, ConversationState.WAITING_FOR_SIDE)
        self.assertEqual(result.response_key, "ask_for_side")
        self.assertEqual(context.selected_side_groups, {"cheese": ["american"]})
        self.assertEqual(context.current_side_group_index, 1)

    def test_option_request_lists_choices_only_when_customer_asks(self):
        handler = WaitingForSideHandler()
        context = _build_context()

        result = handler.handle(
            intent=Intent.UNKNOWN,
            context=context,
            user_text="what are the options",
            session=None,
        )

        self.assertEqual(result.next_state, ConversationState.WAITING_FOR_SIDE)
        self.assertEqual(result.response_key, "list_side_options")


    def test_no_thanks_skips_optional_side_group(self):
        fries = PendingSideChoice(
            item_id="fries",
            name="Fries",
            pricing_mode="fixed",
            normalized_name="fries",
        )
        group = PendingSideGroup(
            group_id="optional_side",
            name="Optional Side",
            is_required=False,
            min_selector=0,
            max_selector=1,
            choices=[fries],
            choices_by_item_id={"fries": fries},
            choices_by_normalized_name={"fries": [fries]},
            choice_names=("Fries",),
            normalized_choice_names=("fries",),
            top_choice_names=("Fries",),
        )
        context = ConversationContext()
        context.current_item_id = "burger"
        context.current_item_name = "Burger"
        context.pending_add_item = PendingAddItem(
            item_id="burger",
            item_name="Burger",
            side_groups=[group],
            side_groups_by_id={"optional_side": group},
            side_choice_by_item_id={"fries": fries},
        )

        result = WaitingForSideHandler().handle(
            intent=Intent.UNKNOWN,
            context=context,
            user_text="no thanks",
            session=None,
        )

        self.assertEqual(result.next_state, ConversationState.WAITING_FOR_QUANTITY)
        self.assertEqual(result.response_key, "ask_for_quantity")
        self.assertIn("optional_side", context.skipped_side_groups)

    def test_what_sides_do_you_have_lists_current_side_options(self):
        handler = WaitingForSideHandler()
        context = _build_context()

        result = handler.handle(
            intent=Intent.UNKNOWN,
            context=context,
            user_text="what sides do you have?",
            session=None,
        )

        self.assertEqual(result.next_state, ConversationState.WAITING_FOR_SIDE)
        self.assertEqual(result.response_key, "list_side_options")

    def test_required_side_cannot_be_skipped_with_skip_it(self):
        handler = WaitingForSideHandler()
        context = _build_context()

        result = handler.handle(
            intent=Intent.UNKNOWN,
            context=context,
            user_text="skip it",
            session=None,
        )

        self.assertEqual(result.next_state, ConversationState.WAITING_FOR_SIDE)
        self.assertEqual(result.response_key, "required_side_cannot_skip")

    def test_slot_first_side_matching_resolves_bun_choice_from_noisy_utterance(self):
        handler = WaitingForSideHandler()
        context = _build_context_with_bun_side_group()
        context.last_slots = (SlotValue(name="VARIANT", value="wheat bun"),)

        result = handler.handle(
            intent=Intent.UNKNOWN,
            context=context,
            user_text="oka wheat bun",
            session=None,
        )

        self.assertEqual(result.next_state, ConversationState.WAITING_FOR_QUANTITY)
        self.assertEqual(result.response_key, "ask_for_quantity")
        self.assertEqual(context.selected_side_groups, {"bun": ["whole_wheat"]})

    def test_plain_pan_fuzzy_matches_plain_bun_when_bun_group_is_active(self):
        handler = WaitingForSideHandler()
        context = _build_context_with_bun_side_group()
        context.last_slots = (SlotValue(name="ITEM", value="plain pan", raw="plain pan"),)

        result = handler.handle(
            intent=Intent.UNKNOWN,
            context=context,
            user_text="plain pan",
            session=None,
        )

        self.assertEqual(result.next_state, ConversationState.WAITING_FOR_QUANTITY)
        self.assertEqual(result.response_key, "ask_for_quantity")
        self.assertEqual(context.selected_side_groups, {"bun": ["plain"]})

    def test_invalid_side_feedback_prefers_slot_candidate_over_noisy_raw_utterance(self):
        handler = WaitingForSideHandler()
        context = _build_context_with_bun_side_group()
        context.last_slots = (SlotValue(name="VARIANT", value="oat bun"),)

        result = handler.handle(
            intent=Intent.UNKNOWN,
            context=context,
            user_text="oka oat bun",
            session=None,
        )

        self.assertEqual(result.next_state, ConversationState.WAITING_FOR_SIDE)
        self.assertEqual(result.response_key, "repeat_side_options")
        self.assertEqual(result.response_payload["unmatched_names"], ["oat bun"])
        self.assertEqual(result.response_payload["selected_candidate"], "oat bun")
        self.assertEqual(result.response_payload["match_source"], "slot_value")


if __name__ == "__main__":
    unittest.main()

