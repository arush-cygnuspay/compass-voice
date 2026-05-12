from app.menu.models import (
    MenuItem,
    ModifierChoice,
    ModifierGroup,
    Pricing,
    PricingVariant,
    SideChoice,
    SideGroup,
)
from app.nlu.query_normalization.text_preprocessor import normalize_text
from app.state_machine.handlers.item.add_item.add_item_flow import (
    AddItemCommand,
    ReadyToFinalize,
    determine_next_add_item_step,
)
from app.state_machine.handlers.item.add_item.pending_add_item_factory import build_pending_add_item
from app.state_machine.models.conversation_context import ConversationContext
from app.state_machine.models.conversation_state import ConversationState
from app.state_machine.models.pending_item_models import ModifierSelection


def _variant(variant_id: str, label: str, price_cents: int) -> PricingVariant:
    return PricingVariant(
        variant_id=variant_id,
        label=label,
        normalized_label=normalize_text(label),
        price_cents=price_cents,
    )


def _side_choice(item_id: str, name: str) -> SideChoice:
    return SideChoice(
        item_id=item_id,
        name=name,
        normalized_name=normalize_text(name),
        pricing=Pricing(mode="fixed", price_cents=0),
    )


def _modifier_choice(modifier_id: str, name: str, price_cents: int) -> ModifierChoice:
    return ModifierChoice(
        modifier_id=modifier_id,
        name=name,
        normalized_name=normalize_text(name),
        price_cents=price_cents,
    )


def make_variant_item() -> MenuItem:
    return MenuItem(
        item_id="burger_1",
        name="Zinger Burger",
        normalized_name=normalize_text("Zinger Burger"),
        aliases=("zinger burger", "burger"),
        normalized_aliases=(normalize_text("zinger burger"), normalize_text("burger")),
        voice_labels=("zinger burger", "burger"),
        pricing=Pricing(
            mode="variant",
            variants=[
                _variant("small", "Small", 500),
                _variant("large", "Large", 700),
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
        normalized_name=normalize_text("Zinger Meal"),
        aliases=("zinger meal",),
        normalized_aliases=(normalize_text("zinger meal"),),
        voice_labels=("zinger meal",),
        pricing=Pricing(
            mode="variant",
            variants=[
                _variant("regular", "Regular", 900),
                _variant("large", "Large", 1100),
            ],
        ),
        side_groups=[
            SideGroup(
                group_id="side_1",
                name="Choose your side",
                normalized_name=normalize_text("Choose your side"),
                is_required=True,
                min_selector=1,
                max_selector=1,
                choices=[
                    _side_choice("fries", "Fries"),
                    _side_choice("salad", "Salad"),
                ],
            )
        ],
        modifier_groups=[
            ModifierGroup(
                group_id="mod_1",
                name="Add-on",
                normalized_name=normalize_text("Add-on"),
                is_required=False,
                min_selector=0,
                max_selector=1,
                choices=[
                    _modifier_choice("cheese", "Cheese", 100),
                    _modifier_choice("jalapeno", "Jalapeno", 100),
                ],
            )
        ],
        available=True,
    )


def test_determine_next_step_asks_for_size_first():
    context = ConversationContext()
    context.pending_add_item = build_pending_add_item(make_variant_item())

    step = determine_next_add_item_step(context)

    assert step.next_state == ConversationState.WAITING_FOR_SIZE
    assert step.response_key == "ask_for_size"
    assert context.current_prompt_field == "size"
    assert context.available_choices_kind == "size"
    assert context.available_choices_values == ("Small", "Large")


def test_determine_next_step_moves_to_side_after_size_selected():
    context = ConversationContext()
    context.pending_add_item = build_pending_add_item(make_full_item())
    context.selected_variant_id = "regular"

    step = determine_next_add_item_step(context)

    assert step.next_state == ConversationState.WAITING_FOR_SIDE
    assert step.response_key == "ask_for_side"
    assert context.current_prompt_field == "side"
    assert context.available_choices_kind == "side"
    assert context.available_choices_values == ("Fries", "Salad")


def test_determine_next_step_moves_to_modifier_after_side_selected():
    context = ConversationContext()
    context.pending_add_item = build_pending_add_item(make_full_item())
    context.selected_variant_id = "regular"
    context.selected_side_groups = {"side_1": ["fries"]}

    step = determine_next_add_item_step(context)

    assert step.next_state == ConversationState.WAITING_FOR_MODIFIER
    assert step.response_key == "ask_for_modifier"
    assert context.current_prompt_field == "modifier"
    assert context.available_choices_kind == "modifier"
    assert context.available_choices_values == ("Cheese", "Jalapeno")


def test_determine_next_step_defaults_missing_quantity_and_finalizes():
    """When no quantity is explicitly set, the policy defaults to 1 and finalizes."""
    context = ConversationContext()
    context.pending_add_item = build_pending_add_item(make_full_item())
    context.selected_variant_id = "regular"
    context.selected_side_groups = {"side_1": ["fries"]}
    context.skipped_modifier_groups = {"mod_1"}

    step = determine_next_add_item_step(context)

    assert isinstance(step, ReadyToFinalize)
    assert step.command.quantity == 1
    assert context.quantity == 1


def test_determine_next_step_returns_ready_to_finalize_when_all_required_fields_present():
    context = ConversationContext()
    context.pending_add_item = build_pending_add_item(make_full_item())
    context.selected_variant_id = "regular"
    context.selected_side_groups = {"side_1": ["fries"]}
    context.skipped_modifier_groups = {"mod_1"}
    context.quantity = 2

    step = determine_next_add_item_step(context)

    assert isinstance(step, ReadyToFinalize)
    assert isinstance(step.command, AddItemCommand)
    assert step.command.item_id == "meal_1"
    assert step.command.quantity == 2
    assert step.command.variant_id == "regular"
    assert step.command.sides == {"side_1": ["fries"]}


def test_ready_to_finalize_command_to_dict_produces_cart_command():
    context = ConversationContext()
    context.pending_add_item = build_pending_add_item(make_full_item())
    context.quantity = 2
    context.selected_variant_id = "large"
    context.selected_side_groups = {"side_1": ["fries"]}
    context.selected_modifier_groups = {
        "mod_1": [
            ModifierSelection(
                modifier_id="cheese",
                name="Cheese",
                action="add",
                instruction=None,
            )
        ]
    }
    context.skipped_modifier_groups = set()

    step = determine_next_add_item_step(context)

    assert isinstance(step, ReadyToFinalize)
    command = step.command.to_dict()

    assert command["type"] == "ADD_ITEM_TO_CART"
    assert command["payload"]["item_id"] == "meal_1"
    assert command["payload"]["quantity"] == 2
    assert command["payload"]["variant_id"] == "large"
    assert command["payload"]["sides"] == {"side_1": ["fries"]}
    assert command["payload"]["modifiers"] == {
        "mod_1": [
            {
                "modifier_id": "cheese",
                "name": "Cheese",
                "action": "add",
                "instruction": None,
            }
        ]
    }
