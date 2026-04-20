from pathlib import Path

from app.menu.models import MenuItem, ModifierChoice, ModifierGroup, Pricing, SideChoice, SideGroup
from app.menu.repository import MenuRepository
from app.menu.store import MenuStore
from app.menu.query_result import MenuQueryResult, MenuQueryType
from app.nlu.intent_resolution.intent import Intent
from app.nlu.nlu_result import SlotValue
from app.nlu.query_normalization.text_preprocessor import normalize_text
from app.state_machine.handlers.item.add_item.add_item_handler import AddItemHandler
from app.state_machine.handlers.item.add_item.waiting_for_modifier_handler import (
    WaitingForModifierHandler,
)
from app.state_machine.handlers.item.add_item.waiting_for_side_handler import (
    WaitingForSideHandler,
)
from app.state_machine.models.conversation_context import ConversationContext
from app.state_machine.models.pending_item_models import (
    PendingAddItem,
    PendingModifierChoice,
    PendingModifierGroup,
    PendingSideChoice,
    PendingSideGroup,
)
from app.state_machine.models.conversation_state import ConversationState


class FakeMenuRepo:
    def __init__(self, result: MenuQueryResult):
        self._result = result
        self.store = _NullMenuStore()

    def resolve_menu_query_from_slots(self, **kwargs):
        return self._result

    def resolve_menu_query_from_slots_normalized(self, **kwargs):
        return self._result

    def resolve_menu_query(self, text: str, limit: int = 5):
        return self._result

    def resolve_menu_query_normalized(self, text: str, limit: int = 5):
        return self._result


class _NullMenuStore:
    def find_entity(self, *args, **kwargs):
        return []

    def find_item_exact(self, *args, **kwargs):
        return None

    def find_item_ids_by_alias(self, *args, **kwargs):
        return []

    def find_item_ids_by_voice_label(self, *args, **kwargs):
        return []

    def find_discoverable_item_mentions(self, *args, **kwargs):
        return []


def _build_demo_menu_repo() -> MenuRepository:
    data_root = Path(__file__).resolve().parents[5] / "app" / "data" / "restaurants" / "demo"
    store = MenuStore(
        menu_path=data_root / "menu.json",
        entity_index_path=data_root / "entity_index.json",
    )
    return MenuRepository(store)


def _menu_item_with_side_group() -> MenuItem:
    return MenuItem(
        item_id="burger_1",
        name="Chicken Burger",
        normalized_name=normalize_text("Chicken Burger"),
        aliases=("chicken burger",),
        normalized_aliases=(normalize_text("chicken burger"),),
        voice_labels=("chicken burger",),
        pricing=Pricing(mode="fixed", price_cents=600),
        side_groups=[
            SideGroup(
                group_id="drink",
                name="Choose your drink",
                normalized_name=normalize_text("Choose your drink"),
                is_required=False,
                min_selector=0,
                max_selector=1,
                choices=[
                    SideChoice(
                        item_id="coke",
                        name="Coke",
                        normalized_name=normalize_text("Coke"),
                        pricing=Pricing(mode="fixed", price_cents=0),
                    ),
                    SideChoice(
                        item_id="sprite",
                        name="Sprite",
                        normalized_name=normalize_text("Sprite"),
                        pricing=Pricing(mode="fixed", price_cents=0),
                    ),
                ],
            )
        ],
        modifier_groups=[],
        available=True,
    )


def _menu_item_with_modifier_group() -> MenuItem:
    return MenuItem(
        item_id="burger_1",
        name="Chicken Burger",
        normalized_name=normalize_text("Chicken Burger"),
        aliases=("chicken burger",),
        normalized_aliases=(normalize_text("chicken burger"),),
        voice_labels=("chicken burger",),
        pricing=Pricing(mode="fixed", price_cents=600),
        side_groups=[],
        modifier_groups=[
            ModifierGroup(
                group_id="mods",
                name="Add-ons",
                normalized_name=normalize_text("Add-ons"),
                is_required=False,
                min_selector=0,
                max_selector=1,
                choices=[
                    ModifierChoice(
                        modifier_id="cheese",
                        name="Cheese",
                        normalized_name=normalize_text("Cheese"),
                        price_cents=100,
                    ),
                    ModifierChoice(
                        modifier_id="bacon",
                        name="Bacon",
                        normalized_name=normalize_text("Bacon"),
                        price_cents=150,
                    ),
                ],
            )
        ],
        available=True,
    )


def _menu_item_with_side_and_modifier_group() -> MenuItem:
    return MenuItem(
        item_id="taco_1",
        name="Chicken Taco",
        normalized_name=normalize_text("Chicken Taco"),
        aliases=("chicken taco",),
        normalized_aliases=(normalize_text("chicken taco"),),
        voice_labels=("chicken taco",),
        pricing=Pricing(mode="fixed", price_cents=500),
        side_groups=[
            SideGroup(
                group_id="drink",
                name="Choose your drink",
                normalized_name=normalize_text("Choose your drink"),
                is_required=False,
                min_selector=0,
                max_selector=1,
                choices=[
                    SideChoice(
                        item_id="coke_12",
                        name="Coke (12 oz.)",
                        normalized_name=normalize_text("Coke (12 oz.)"),
                        pricing=Pricing(mode="fixed", price_cents=0),
                    ),
                ],
            )
        ],
        modifier_groups=[
            ModifierGroup(
                group_id="mods",
                name="Add-ons",
                normalized_name=normalize_text("Add-ons"),
                is_required=False,
                min_selector=0,
                max_selector=2,
                choices=[
                    ModifierChoice(
                        modifier_id="chicken",
                        name="Chicken",
                        normalized_name=normalize_text("Chicken"),
                        price_cents=100,
                    ),
                    ModifierChoice(
                        modifier_id="cheese",
                        name="Cheese",
                        normalized_name=normalize_text("Cheese"),
                        price_cents=100,
                    ),
                ],
            )
        ],
        available=True,
    )


def _menu_item_with_required_burger_groups() -> MenuItem:
    return MenuItem(
        item_id="burger_2",
        name="Chicken Burger",
        normalized_name=normalize_text("Chicken Burger"),
        aliases=("chicken burger",),
        normalized_aliases=(normalize_text("chicken burger"),),
        voice_labels=("chicken burger",),
        pricing=Pricing(mode="fixed", price_cents=1152),
        side_groups=[
            SideGroup(
                group_id="cheese",
                name="Choose Cheese",
                normalized_name=normalize_text("Choose Cheese"),
                is_required=True,
                min_selector=1,
                max_selector=1,
                choices=[
                    SideChoice(
                        item_id="american_cheese",
                        name="American Cheese",
                        normalized_name=normalize_text("American Cheese"),
                        pricing=Pricing(mode="fixed", price_cents=50),
                    ),
                    SideChoice(
                        item_id="cheddar_cheese",
                        name="Cheddar Cheese",
                        normalized_name=normalize_text("Cheddar Cheese"),
                        pricing=Pricing(mode="fixed", price_cents=50),
                    ),
                ],
            ),
            SideGroup(
                group_id="meat",
                name="Choose Meat",
                normalized_name=normalize_text("Choose Meat"),
                is_required=True,
                min_selector=1,
                max_selector=1,
                choices=[
                    SideChoice(
                        item_id="beef_meat",
                        name="Beef Meat",
                        normalized_name=normalize_text("Beef Meat"),
                        pricing=Pricing(mode="fixed", price_cents=0),
                    ),
                ],
            ),
            SideGroup(
                group_id="bun",
                name="Choose Bun",
                normalized_name=normalize_text("Choose Bun"),
                is_required=True,
                min_selector=1,
                max_selector=1,
                choices=[
                    SideChoice(
                        item_id="plain_bun",
                        name="Plain Bun",
                        normalized_name=normalize_text("Plain Bun"),
                        pricing=Pricing(mode="fixed", price_cents=0),
                    ),
                ],
            ),
        ],
        modifier_groups=[
            ModifierGroup(
                group_id="mods",
                name="Burger Modification",
                normalized_name=normalize_text("Burger Modification"),
                is_required=False,
                min_selector=0,
                max_selector=1000,
                choices=[
                    ModifierChoice(
                        modifier_id="red_onions",
                        name="Red Onions",
                        normalized_name=normalize_text("Red Onions"),
                        price_cents=0,
                    ),
                    ModifierChoice(
                        modifier_id="american_cheese_mod",
                        name="American Cheese",
                        normalized_name=normalize_text("American Cheese"),
                        price_cents=0,
                    ),
                ],
            )
        ],
        available=True,
    )


def test_add_item_handler_reports_side_prefill_over_max_and_unmatched_feedback():
    item = _menu_item_with_side_group()
    repo = FakeMenuRepo(MenuQueryResult(type=MenuQueryType.ITEM, item=item))
    handler = AddItemHandler(repo)
    context = ConversationContext()
    context.last_slots = (
        SlotValue(name="ITEM", value="Chicken Burger"),
        SlotValue(name="ITEM", value="Coke"),
        SlotValue(name="ITEM", value="Sprite"),
        SlotValue(name="ITEM", value="Rice"),
    )

    result = handler.handle(
        intent=Intent.ADD_ITEM,
        context=context,
        user_text="chicken burger with coke sprite and rice",
        session=None,
    )

    assert result.next_state == ConversationState.WAITING_FOR_QUANTITY
    assert result.response_key == "ask_for_quantity"
    assert context.selected_side_groups == {"drink": ["coke"]}
    assert result.response_payload["prefilled_summary"] == "with Coke"
    feedback = result.response_payload["prefill_feedback"]
    assert "I kept Coke and left off Sprite because you can only pick 1." in feedback
    assert "I couldn't find rice." in feedback


def test_add_item_handler_reports_modifier_prefill_over_max_and_unmatched_feedback():
    item = _menu_item_with_modifier_group()
    repo = FakeMenuRepo(MenuQueryResult(type=MenuQueryType.ITEM, item=item))
    handler = AddItemHandler(repo)
    context = ConversationContext()
    context.last_slots = (
        SlotValue(name="ITEM", value="Chicken Burger"),
        SlotValue(name="MODIFIER", value="Cheese"),
        SlotValue(name="MODIFIER", value="Bacon"),
        SlotValue(name="MODIFIER", value="Avocado"),
    )

    result = handler.handle(
        intent=Intent.ADD_ITEM,
        context=context,
        user_text="chicken burger with cheese bacon and avocado",
        session=None,
    )

    assert result.next_state == ConversationState.WAITING_FOR_QUANTITY
    assert result.response_key == "ask_for_quantity"
    assert [sel.modifier_id for sel in context.selected_modifier_groups["mods"]] == ["cheese"]
    assert result.response_payload["prefilled_summary"] == "with Cheese"
    feedback = result.response_payload["prefill_feedback"]
    assert "I kept Cheese and left off Bacon because you can only pick 1." in feedback
    assert "I couldn't find avocado." in feedback


def test_add_item_handler_ignores_item_name_tokens_in_modifier_prefill_feedback():
    item = _menu_item_with_side_and_modifier_group()
    repo = FakeMenuRepo(MenuQueryResult(type=MenuQueryType.ITEM, item=item))
    handler = AddItemHandler(repo)
    context = ConversationContext()
    context.last_slots = (
        SlotValue(name="ITEM", value="Chicken Taco"),
        SlotValue(name="ITEM", value="Coke (12 oz.)"),
        SlotValue(name="MODIFIER", value="American Cheese"),
        SlotValue(name="ITEM", value="Beef Meat"),
    )

    result = handler.handle(
        intent=Intent.ADD_ITEM,
        context=context,
        user_text="chicken taco with american cheese coke and beef meat",
        session=None,
    )

    assert result.next_state == ConversationState.WAITING_FOR_QUANTITY
    assert result.response_key == "ask_for_quantity"
    assert context.selected_side_groups == {"drink": ["coke_12"]}
    assert "mods" not in context.selected_modifier_groups
    assert result.response_payload["prefilled_summary"] == "with Coke (12 oz.)"

    feedback = result.response_payload["prefill_feedback"]
    assert "beef meat" in feedback
    assert "american cheese" in feedback
    assert "chicken taco with american cheese coke and beef meat" not in feedback


def test_add_item_handler_collapses_redundant_prefill_feedback_across_groups():
    item = _menu_item_with_required_burger_groups()
    repo = FakeMenuRepo(MenuQueryResult(type=MenuQueryType.ITEM, item=item))
    handler = AddItemHandler(repo)
    context = ConversationContext()
    context.last_slots = (
        SlotValue(name="ITEM", value="Chicken Burger"),
        SlotValue(name="ITEM", value="Coke"),
        SlotValue(name="MODIFIER", value="Red Onions"),
        SlotValue(name="ITEM", value="No Sauce"),
    )

    result = handler.handle(
        intent=Intent.ADD_ITEM,
        context=context,
        user_text="chicken burger with coke and extra cheese red onions and no sauce",
        session=None,
    )

    assert result.next_state == ConversationState.WAITING_FOR_SIDE
    assert result.response_key == "ask_for_side"
    assert result.response_payload["group_name"] == "Choose Cheese"
    assert [sel.modifier_id for sel in context.selected_modifier_groups["mods"]] == ["red_onions"]

    feedback = result.response_payload["prefill_feedback"]
    assert feedback.count("I couldn't find") == 1
    assert "coke" in feedback
    assert "no sauce" in feedback
    assert "extra cheese red onions" not in feedback
    assert "with coke" not in feedback


def test_add_item_handler_uses_real_menu_truth_for_multi_item_boundaries():
    repo = _build_demo_menu_repo()
    handler = AddItemHandler(repo)
    context = ConversationContext()
    text = normalize_text(
        "a chicken taco with a coke and american cheese plus jelly and sausage and also a chicken burger with red onions"
    )

    context.last_slots = (
        SlotValue(name="ITEM", value="Chicken Taco", raw="chicken taco", start=text.index("chicken taco"), end=text.index("chicken taco") + len("chicken taco"), confidence=1.0),
        SlotValue(name="ITEM", value="Coke (12 oz.)", raw="coke", confidence=1.0),
        SlotValue(name="ITEM", value="American Cheese", raw="american cheese", start=text.index("american cheese"), end=text.index("american cheese") + len("american cheese"), confidence=1.0),
        SlotValue(name="MODIFIER", value="Jelly", raw="jelly", start=text.index("jelly"), end=text.index("jelly") + len("jelly"), confidence=1.0),
        SlotValue(name="MODIFIER", value="Sausage", raw="sausage", start=text.index("sausage"), end=text.index("sausage") + len("sausage"), confidence=1.0),
        SlotValue(name="ITEM", value="Chicken Burger", raw="chicken burger", start=text.index("chicken burger"), end=text.index("chicken burger") + len("chicken burger"), confidence=1.0),
        SlotValue(name="MODIFIER", value="Red Onions", raw="red onions", start=text.index("red onions"), end=text.index("red onions") + len("red onions"), confidence=1.0),
    )

    result = handler.handle(
        intent=Intent.ADD_ITEM,
        context=context,
        user_text=text,
        session=None,
    )

    assert result.next_state == ConversationState.IDLE
    assert result.response_key == "item_added_successfully"
    assert result.response_payload["multi_item_ack"] is True
    assert result.response_payload["queue_count"] == 1
    assert result.response_payload["queued_item_names"] == ["Chicken Burger"]
    assert len(result.response_payload["heard_items_summary"]) == 2
    assert context.pending_item_queue[0].item_slot_value == "Chicken Burger"
    assert result.response_payload["prefilled_summary"] == "with Coke (12 oz.), Sausage, and Jelly"
    assert result.response_payload["quantity"] == 1

    feedback = result.response_payload["prefill_feedback"]
    assert "american cheese" in feedback
    assert "chicken burger" not in feedback
    assert "red onions" not in feedback
    assert "american cheese plus jelly" not in feedback


def test_waiting_for_side_returns_unmatched_feedback_when_nothing_matches():
    coke = PendingSideChoice(
        item_id="coke",
        name="Coke",
        pricing_mode="fixed",
        normalized_name="coke",
    )
    sprite = PendingSideChoice(
        item_id="sprite",
        name="Sprite",
        pricing_mode="fixed",
        normalized_name="sprite",
    )
    group = PendingSideGroup(
        group_id="drink",
        name="Choose your drink",
        is_required=True,
        min_selector=1,
        max_selector=1,
        choices=[coke, sprite],
        choices_by_item_id={"coke": coke, "sprite": sprite},
        choices_by_normalized_name={"coke": [coke], "sprite": [sprite]},
        choice_names=("Coke", "Sprite"),
        normalized_choice_names=("coke", "sprite"),
        top_choice_names=("Coke", "Sprite"),
    )
    context = ConversationContext()
    context.current_item_id = "burger"
    context.current_item_name = "Burger"
    context.pending_add_item = PendingAddItem(
        item_id="burger",
        item_name="Burger",
        side_groups=[group],
        side_groups_by_id={"drink": group},
        side_choice_by_item_id={"coke": coke, "sprite": sprite},
    )

    result = WaitingForSideHandler().handle(
        intent=Intent.UNKNOWN,
        context=context,
        user_text="rice",
        session=None,
    )

    assert result.next_state == ConversationState.WAITING_FOR_SIDE
    assert result.response_key == "repeat_side_options"
    assert result.response_payload["unmatched_names"] == ["rice"]
    assert result.response_payload["repeat_reason"] == "invalid"


def test_waiting_for_modifier_returns_unmatched_feedback_when_nothing_matches():
    cheese = PendingModifierChoice(
        modifier_id="cheese",
        name="Cheese",
        group_id="mods",
        normalized_name="cheese",
    )
    bacon = PendingModifierChoice(
        modifier_id="bacon",
        name="Bacon",
        group_id="mods",
        normalized_name="bacon",
    )
    group = PendingModifierGroup(
        group_id="mods",
        name="Add-ons",
        is_required=True,
        min_selector=1,
        max_selector=1,
        choices=[cheese, bacon],
        choices_by_modifier_id={"cheese": cheese, "bacon": bacon},
        choices_by_normalized_name={"cheese": [cheese], "bacon": [bacon]},
        choice_names=("Cheese", "Bacon"),
        normalized_choice_names=("cheese", "bacon"),
        top_choice_names=("Cheese", "Bacon"),
    )
    context = ConversationContext()
    context.current_item_id = "burger"
    context.current_item_name = "Burger"
    context.pending_add_item = PendingAddItem(
        item_id="burger",
        item_name="Burger",
        modifier_groups=[group],
        modifier_groups_by_id={"mods": group},
        modifier_choice_by_id={"cheese": cheese, "bacon": bacon},
    )

    result = WaitingForModifierHandler().handle(
        intent=Intent.UNKNOWN,
        context=context,
        user_text="avocado",
        session=None,
    )

    assert result.next_state == ConversationState.WAITING_FOR_MODIFIER
    assert result.response_key == "repeat_modifier_options"
    assert result.response_payload["unmatched_names"] == ["avocado"]
    assert result.response_payload["repeat_reason"] == "invalid"
