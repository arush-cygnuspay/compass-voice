from app.menu.models import (
    MenuItem,
    ModifierChoice,
    ModifierGroup,
    Pricing,
    PricingVariant,
    SideChoice,
    SideGroup,
)
from app.state_machine.models.conversation_context import ConversationContext
from app.state_machine.models.conversation_state import ConversationState
from app.state_machine.handlers.item.add_item.add_item_flow import (
    build_add_item_command,
    determine_next_add_item_step,
)


def make_variant_item() -> MenuItem:
    return MenuItem(
        item_id="burger_1",
        name="Zinger Burger",
        aliases=["zinger burger", "burger"],
        pricing=Pricing(
            mode="variant",
            variants=[
                PricingVariant(variant_id="small", label="Small", price_cents=500),
                PricingVariant(variant_id="large", label="Large", price_cents=700),
            ],
        ),
        side_groups=[],
        modifier_groups=[],
        available=True,
    )


def make_full_item() -> MenuItem:
    return MenuItem(
        item_id="meal_1",
        name="Zinger Meal",
        aliases=["zinger meal"],
        pricing=Pricing(
            mode="variant",
            variants=[
                PricingVariant(variant_id="regular", label="Regular", price_cents=900),
                PricingVariant(variant_id="large", label="Large", price_cents=1100),
            ],
        ),
        side_groups=[
            SideGroup(
                group_id="side_1",
                name="Choose your side",
                is_required=True,
                min_selector=1,
                max_selector=1,
                choices=[
                    SideChoice(
                        item_id="fries",
                        name="Fries",
                        pricing=Pricing(mode="fixed", price_cents=0),
                    ),
                    SideChoice(
                        item_id="salad",
                        name="Salad",
                        pricing=Pricing(mode="fixed", price_cents=0),
                    ),
                ],
            )
        ],
        modifier_groups=[
            ModifierGroup(
                group_id="mod_1",
                name="Add-on",
                is_required=False,
                min_selector=0,
                max_selector=1,
                choices=[
                    ModifierChoice(modifier_id="cheese", name="Cheese", price_cents=100),
                    ModifierChoice(modifier_id="jalapeno", name="Jalapeno", price_cents=100),
                ],
            )
        ],
        available=True,
    )


def test_determine_next_step_asks_for_size_first():
    item = make_variant_item()
    context = ConversationContext()

    step = determine_next_add_item_step(context)

    assert step.next_state == ConversationState.WAITING_FOR_SIZE
    assert step.response_key == "ask_for_size"
    assert context.current_prompt_field == "size"
    assert context.available_choices_kind == "size"
    assert context.available_choices_values == ("Small", "Large")


def test_determine_next_step_moves_to_side_after_size_selected():
    item = make_full_item()
    context = ConversationContext()
    context.selected_variant_id = "regular"

    step = determine_next_add_item_step(context)

    assert step.next_state == ConversationState.WAITING_FOR_SIDE
    assert step.response_key == "ask_for_side"
    assert context.current_prompt_field == "side"
    assert context.available_choices_kind == "side"
    assert context.available_choices_values == ("Fries", "Salad")


def test_determine_next_step_moves_to_modifier_after_side_selected():
    item = make_full_item()
    context = ConversationContext()
    context.selected_variant_id = "regular"
    context.selected_side_groups = {"side_1": ["fries"]}

    step = determine_next_add_item_step(context)

    assert step.next_state == ConversationState.WAITING_FOR_MODIFIER
    assert step.response_key == "ask_for_modifier"
    assert context.current_prompt_field == "modifier"
    assert context.available_choices_kind == "modifier"
    assert context.available_choices_values == ("Cheese", "Jalapeno")


def test_determine_next_step_moves_to_quantity_after_modifier_phase_done():
    item = make_full_item()
    context = ConversationContext()
    context.selected_variant_id = "regular"
    context.selected_side_groups = {"side_1": ["fries"]}
    context.skipped_modifier_groups = {"mod_1"}

    step = determine_next_add_item_step(context)

    assert step.next_state == ConversationState.WAITING_FOR_QUANTITY
    assert step.response_key == "ask_for_quantity"
    assert context.current_prompt_field == "quantity"


def test_determine_next_step_finalizes_when_all_required_fields_present():
    item = make_full_item()
    context = ConversationContext()
    context.selected_variant_id = "regular"
    context.selected_side_groups = {"side_1": ["fries"]}
    context.skipped_modifier_groups = {"mod_1"}
    context.quantity = 2

    step = determine_next_add_item_step(context)

    assert step.next_state == ConversationState.FINALIZING_ADD_ITEM
    assert step.response_key == "finalize_add_item"


def test_build_add_item_command_uses_context_values():
    item = make_full_item()
    context = ConversationContext()
    context.quantity = 2
    context.selected_variant_id = "large"
    context.selected_side_groups = {"side_1": ["fries"]}
    context.selected_modifier_groups = {"mod_1": ["cheese"]}

    command = build_add_item_command(item, context)

    assert command["type"] == "ADD_ITEM_TO_CART"
    assert command["payload"]["item_id"] == "meal_1"
    assert command["payload"]["quantity"] == 2
    assert command["payload"]["variant_id"] == "large"
    assert command["payload"]["sides"] == {"side_1": ["fries"]}
    assert command["payload"]["modifiers"] == {"mod_1": ["cheese"]}