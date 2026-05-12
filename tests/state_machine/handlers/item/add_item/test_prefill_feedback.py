from pathlib import Path

from app.menu.models import MenuItem, ModifierChoice, ModifierGroup, Pricing, SideChoice, SideGroup
from app.menu.repository import MenuRepository
from app.menu.store import MenuStore
from app.menu.query_result import MenuQueryResult, MenuQueryType
from app.nlu.intent_resolution.intent import Intent
from app.nlu.nlu_result import SlotValue
from app.nlu.query_normalization.text_preprocessor import normalize_text
from app.state_machine.handlers.item.add_item.add_item_handler import AddItemHandler
from app.state_machine.handlers.item.add_item.pending_add_item_factory import build_pending_add_item
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

    assert result.next_state == ConversationState.WAITING_FOR_SIDE
    assert result.response_key == "ask_for_side"
    assert context.selected_side_groups == {}
    assert "prefilled_summary" not in result.response_payload
    feedback = result.response_payload["prefill_feedback"]
    assert "you can only pick 1" in feedback
    assert "coke" in feedback.lower()
    assert "sprite" in feedback.lower()
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

    assert result.next_state == ConversationState.WAITING_FOR_MODIFIER
    assert result.response_key == "ask_for_modifier"
    assert "mods" not in context.selected_modifier_groups
    assert "prefilled_summary" not in result.response_payload
    feedback = result.response_payload["prefill_feedback"]
    assert "you can only pick 1" in feedback
    assert "cheese" in feedback.lower()
    assert "bacon" in feedback.lower()
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

    assert result.next_state == ConversationState.WAITING_FOR_MODIFIER
    assert result.response_key == "ask_for_modifier"
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


def test_add_item_handler_prefills_side_and_adjacent_modifiers_from_sparse_slots():
    repo = _build_demo_menu_repo()
    handler = AddItemHandler(repo)
    context = ConversationContext()
    text = normalize_text("a chicken taco with small coke and sausage jelly")

    context.last_slots = (
        SlotValue(name="ITEM", value="Chicken Taco"),
    )

    result = handler.handle(
        intent=Intent.ADD_ITEM,
        context=context,
        user_text=text,
        session=None,
    )

    # All side/modifier groups satisfied, missing quantity defaults to 1 → item added.
    assert result.next_state == ConversationState.IDLE
    assert result.response_key == "item_added_successfully"
    # Context not reset here (direct handler call, no TurnEngine) — groups still accessible.
    assert context.selected_side_groups == {
        "d9dcbebe-7d66-4659-940c-3890678834b5": ["5e57e093-8aab-4e93-acec-5a696fadd73e"]
    }
    assert {
        key: [selection.modifier_id for selection in value]
        for key, value in context.selected_modifier_groups.items()
    } == {
        "600e5a6e-96d5-4759-9ca4-d229b0f6b9c2": ["917d4025-7326-4fc7-b801-a6d76f14e0a9"],
        "1e0fad0b-4d65-4e2d-a8bf-ad9d20696705": ["fce50f96-e4ad-48ff-914a-cb944402a4f1"],
    }
    assert "prefill_feedback" not in result.response_payload or "small coke" not in result.response_payload.get("prefill_feedback", "")


def test_add_item_handler_prefills_sparse_multi_item_segment_before_missing_groups():
    repo = _build_demo_menu_repo()
    handler = AddItemHandler(repo)
    context = ConversationContext()
    text = normalize_text(
        "i want a chicken taco with coke steak and chicken and a chicken burger with american cheese red onions and fresh mushrooms"
    )

    context.last_slots = (
        SlotValue(
            name="ITEM",
            value="Chicken Taco",
            raw="chicken taco",
            start=text.index("chicken taco"),
            end=text.index("chicken taco") + len("chicken taco"),
            confidence=1.0,
        ),
        SlotValue(
            name="ITEM",
            value="Chicken Burger",
            raw="chicken burger",
            start=text.index("chicken burger"),
            end=text.index("chicken burger") + len("chicken burger"),
            confidence=1.0,
        ),
    )

    result = handler.handle(
        intent=Intent.ADD_ITEM,
        context=context,
        user_text=text,
        session=None,
    )

    assert result.next_state == ConversationState.WAITING_FOR_MODIFIER
    assert result.response_key == "ask_for_modifier"
    assert result.response_payload["group_name"] == "Additional Extras For Biscuits"
    assert result.response_payload["prefilled_summary"] == "with Coke (12 oz.), Steak, and Chicken"
    assert result.response_payload["heard_items_summary"] == ["Chicken Taco", "Chicken Burger"]

    pending = context.pending_add_item
    assert pending is not None

    drink_group = next(group for group in pending.side_groups if group.name == "Can Drinks")
    selected_drink_names = [
        drink_group.choices_by_item_id[item_id].name
        for item_id in context.selected_side_groups.get(drink_group.group_id, [])
    ]
    assert selected_drink_names == ["Coke (12 oz.)"]

    meat_group = next(group for group in pending.modifier_groups if group.name == "Additional Meat for Plates")
    selected_meat_names = [
        selection.name
        for selection in context.selected_modifier_groups.get(meat_group.group_id, [])
    ]
    assert selected_meat_names == ["Steak", "Chicken"]

    prefill_debug = result.response_payload["prefill_debug"]
    assert "coke" in prefill_debug["candidate_phrases"]
    assert "steak" in prefill_debug["candidate_phrases"]
    assert "chicken" in prefill_debug["candidate_phrases"]
    assert "Can Drinks" in prefill_debug["skipped_groups_because_prefilled"]
    assert "Additional Meat for Plates" in prefill_debug["skipped_groups_because_prefilled"]
    assert "Can Drinks" not in prefill_debug["missing_groups_after_prefill"]
    assert len(context.pending_item_queue) == 1
    assert context.pending_item_queue[0].item_slot_value == "Chicken Burger"


def test_add_item_handler_prefills_dequeued_multi_item_segment_for_next_item():
    repo = _build_demo_menu_repo()
    handler = AddItemHandler(repo)
    context = ConversationContext()
    text = normalize_text(
        "i want a chicken taco with coke steak and chicken and a chicken burger with american cheese red onions and fresh mushrooms"
    )

    context.last_slots = (
        SlotValue(
            name="ITEM",
            value="Chicken Taco",
            raw="chicken taco",
            start=text.index("chicken taco"),
            end=text.index("chicken taco") + len("chicken taco"),
            confidence=1.0,
        ),
        SlotValue(
            name="ITEM",
            value="Chicken Burger",
            raw="chicken burger",
            start=text.index("chicken burger"),
            end=text.index("chicken burger") + len("chicken burger"),
            confidence=1.0,
        ),
    )

    first = handler.handle(
        intent=Intent.ADD_ITEM,
        context=context,
        user_text=text,
        session=None,
    )
    assert first.response_key == "ask_for_modifier"
    next_item = context.pending_item_queue.popleft()

    context.reset_task()
    context.last_slots = next_item.segment_slots

    second = handler.handle(
        intent=Intent.ADD_ITEM,
        context=context,
        user_text=next_item.raw_text,
        session=None,
    )

    assert second.next_state == ConversationState.WAITING_FOR_SIDE
    assert second.response_key == "ask_for_side"
    assert second.response_payload["group_name"] == "Choose Meat"
    assert second.response_payload["prefilled_summary"] == "with American Cheese, Red Onions, and Fresh Mushroom"

    pending = context.pending_add_item
    assert pending is not None

    cheese_group = next(group for group in pending.side_groups if group.name == "Choose Cheese")
    selected_cheese_names = [
        cheese_group.choices_by_item_id[item_id].name
        for item_id in context.selected_side_groups.get(cheese_group.group_id, [])
    ]
    assert selected_cheese_names == ["American Cheese"]

    burger_mod_group = next(group for group in pending.modifier_groups if group.name == "Burger Modification")
    selected_modifier_names = [
        selection.name
        for selection in context.selected_modifier_groups.get(burger_mod_group.group_id, [])
    ]
    assert selected_modifier_names == ["Red Onions", "Fresh Mushroom"]

    prefill_debug = second.response_payload["prefill_debug"]
    assert "american cheese" in prefill_debug["candidate_phrases"]
    assert "red onions" in prefill_debug["candidate_phrases"]
    assert "fresh mushrooms" in prefill_debug["candidate_phrases"]
    assert "Choose Cheese" in prefill_debug["skipped_groups_because_prefilled"]
    assert "Choose Cheese" not in prefill_debug["missing_groups_after_prefill"]


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


def test_waiting_for_side_accepts_size_prefixed_choice_name() -> None:
    repo = _build_demo_menu_repo()
    item = next(
        menu_item
        for menu_item in repo.store.items.values()
        if menu_item.normalized_name == normalize_text("Chicken Taco")
    )
    context = ConversationContext()
    context.current_item_id = item.item_id
    context.current_item_name = item.name
    context.pending_add_item = build_pending_add_item(item)

    result = WaitingForSideHandler(repo).handle(
        intent=Intent.UNKNOWN,
        context=context,
        user_text="small coke",
        session=None,
    )

    assert result.next_state == ConversationState.WAITING_FOR_MODIFIER
    assert result.response_key == "ask_for_modifier"
    assert context.selected_side_groups == {
        "d9dcbebe-7d66-4659-940c-3890678834b5": ["5e57e093-8aab-4e93-acec-5a696fadd73e"]
    }


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


def test_waiting_for_modifier_prefills_later_groups_from_same_utterance():
    cheese = PendingModifierChoice(
        modifier_id="cheese",
        name="Cheese",
        group_id="g1",
        normalized_name="cheese",
        match_texts=("cheese",),
    )
    jelly = PendingModifierChoice(
        modifier_id="jelly",
        name="Jelly",
        group_id="g2",
        normalized_name="jelly",
        match_texts=("jelly",),
    )
    sausage = PendingModifierChoice(
        modifier_id="sausage",
        name="Sausage",
        group_id="g2",
        normalized_name="sausage",
        match_texts=("sausage",),
    )
    cheese_group = PendingModifierGroup(
        group_id="g1",
        name="Choose Cheese",
        is_required=True,
        min_selector=1,
        max_selector=1,
        choices=[cheese],
        choices_by_modifier_id={"cheese": cheese},
        choices_by_normalized_name={"cheese": [cheese]},
        choice_names=("Cheese",),
        normalized_choice_names=("cheese",),
        top_choice_names=("Cheese",),
    )
    breakfast_group = PendingModifierGroup(
        group_id="g2",
        name="Breakfast Extras",
        is_required=False,
        min_selector=0,
        max_selector=2,
        choices=[jelly, sausage],
        choices_by_modifier_id={"jelly": jelly, "sausage": sausage},
        choices_by_normalized_name={"jelly": [jelly], "sausage": [sausage]},
        choice_names=("Jelly", "Sausage"),
        normalized_choice_names=("jelly", "sausage"),
        top_choice_names=("Jelly", "Sausage"),
    )
    context = ConversationContext()
    context.current_item_id = "burger"
    context.current_item_name = "Burger"
    context.pending_add_item = PendingAddItem(
        item_id="burger",
        item_name="Burger",
        modifier_groups=[cheese_group, breakfast_group],
        modifier_groups_by_id={"g1": cheese_group, "g2": breakfast_group},
        modifier_choice_by_id={"cheese": cheese, "jelly": jelly, "sausage": sausage},
    )

    result = WaitingForModifierHandler().handle(
        intent=Intent.UNKNOWN,
        context=context,
        user_text="cheese jelly and sausage",
        session=None,
    )

    # All modifier groups satisfied, missing quantity defaults to 1 → item added.
    assert result.next_state == ConversationState.IDLE
    assert result.response_key == "item_added_successfully"
    assert [selection.modifier_id for selection in context.selected_modifier_groups["g1"]] == ["cheese"]
    assert [selection.modifier_id for selection in context.selected_modifier_groups["g2"]] == ["jelly", "sausage"]


def test_waiting_for_modifier_requires_clarification_when_later_group_overflows():
    cheese = PendingModifierChoice(
        modifier_id="cheese",
        name="Cheese",
        group_id="g1",
        normalized_name="cheese",
        match_texts=("cheese",),
    )
    jelly = PendingModifierChoice(
        modifier_id="jelly",
        name="Jelly",
        group_id="g2",
        normalized_name="jelly",
        match_texts=("jelly",),
    )
    sausage = PendingModifierChoice(
        modifier_id="sausage",
        name="Sausage",
        group_id="g2",
        normalized_name="sausage",
        match_texts=("sausage",),
    )
    cheese_group = PendingModifierGroup(
        group_id="g1",
        name="Choose Cheese",
        is_required=True,
        min_selector=1,
        max_selector=1,
        choices=[cheese],
        choices_by_modifier_id={"cheese": cheese},
        choices_by_normalized_name={"cheese": [cheese]},
        choice_names=("Cheese",),
        normalized_choice_names=("cheese",),
        top_choice_names=("Cheese",),
    )
    breakfast_group = PendingModifierGroup(
        group_id="g2",
        name="Breakfast Extras",
        is_required=False,
        min_selector=0,
        max_selector=1,
        choices=[jelly, sausage],
        choices_by_modifier_id={"jelly": jelly, "sausage": sausage},
        choices_by_normalized_name={"jelly": [jelly], "sausage": [sausage]},
        choice_names=("Jelly", "Sausage"),
        normalized_choice_names=("jelly", "sausage"),
        top_choice_names=("Jelly", "Sausage"),
    )
    context = ConversationContext()
    context.current_item_id = "burger"
    context.current_item_name = "Burger"
    context.pending_add_item = PendingAddItem(
        item_id="burger",
        item_name="Burger",
        modifier_groups=[cheese_group, breakfast_group],
        modifier_groups_by_id={"g1": cheese_group, "g2": breakfast_group},
        modifier_choice_by_id={"cheese": cheese, "jelly": jelly, "sausage": sausage},
    )

    result = WaitingForModifierHandler().handle(
        intent=Intent.UNKNOWN,
        context=context,
        user_text="cheese jelly and sausage",
        session=None,
    )

    assert result.next_state == ConversationState.WAITING_FOR_MODIFIER
    assert result.response_key == "too_many_modifier_choices"
    assert [selection.modifier_id for selection in context.selected_modifier_groups["g1"]] == ["cheese"]
    assert "g2" not in context.selected_modifier_groups
    assert context.current_modifier_group_index == 1


def test_conversation_context_round_trip_preserves_match_texts():
    cheese = PendingModifierChoice(
        modifier_id="cheese",
        name="Cheese",
        group_id="mods",
        normalized_name="cheese",
        match_texts=("cheese", "american cheese"),
    )
    coke = PendingSideChoice(
        item_id="coke",
        name="Coke (12 oz.)",
        pricing_mode="fixed",
        normalized_name="coke 12 oz",
        match_texts=("coke", "small coke"),
    )
    side_group = PendingSideGroup(
        group_id="drink",
        name="Drink",
        is_required=False,
        min_selector=0,
        max_selector=1,
        choices=[coke],
        choices_by_item_id={"coke": coke},
        choices_by_normalized_name={"coke 12 oz": [coke]},
        choice_names=("Coke (12 oz.)",),
        normalized_choice_names=("coke 12 oz",),
        top_choice_names=("Coke (12 oz.)",),
    )
    modifier_group = PendingModifierGroup(
        group_id="mods",
        name="Add-ons",
        is_required=False,
        min_selector=0,
        max_selector=1,
        choices=[cheese],
        choices_by_modifier_id={"cheese": cheese},
        choices_by_normalized_name={"cheese": [cheese]},
        choice_names=("Cheese",),
        normalized_choice_names=("cheese",),
        top_choice_names=("Cheese",),
    )
    context = ConversationContext()
    context.pending_add_item = PendingAddItem(
        item_id="burger",
        item_name="Burger",
        side_groups=[side_group],
        modifier_groups=[modifier_group],
        side_groups_by_id={"drink": side_group},
        side_choice_by_item_id={"coke": coke},
        modifier_groups_by_id={"mods": modifier_group},
        modifier_choice_by_id={"cheese": cheese},
    )

    restored = ConversationContext.from_dict(context.to_dict())

    assert restored.pending_add_item is not None
    assert restored.pending_add_item.side_groups[0].choices[0].match_texts == ("coke", "small coke")
    assert restored.pending_add_item.modifier_groups[0].choices[0].match_texts == (
        "cheese",
        "american cheese",
    )
