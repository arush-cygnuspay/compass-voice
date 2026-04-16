from app.core.pending_action import PendingAction
from app.menu.models import MenuItem, ModifierChoice, ModifierGroup, Pricing, SideChoice, SideGroup
from app.nlu.nlu_result import SlotValue
from app.menu.query_result import MenuQueryResult, MenuQueryType
from app.nlu.intent_resolution.intent import Intent
from app.nlu.query_normalization.text_preprocessor import normalize_text
from app.state_machine.models.conversation_context import ConversationContext
from app.state_machine.models.conversation_state import ConversationState
from app.state_machine.handlers.item.add_item.add_item_handler import AddItemHandler


class FakeMenuRepo:
    def __init__(self, result: MenuQueryResult):
        self._result = result

    def resolve_menu_query_from_slots(self, **kwargs):
        return self._result

    def resolve_menu_query_from_slots_normalized(self, **kwargs):
        return self._result

    def resolve_menu_query(self, text: str, limit: int = 5):
        return self._result

    def resolve_menu_query_normalized(self, text: str, limit: int = 5):
        return self._result


def make_item() -> MenuItem:
    return MenuItem(
        item_id="burger_1",
        name="Zinger Burger",
        normalized_name=normalize_text("Zinger Burger"),
        aliases=("zinger burger",),
        normalized_aliases=(normalize_text("zinger burger"),),
        voice_labels=("zinger burger",),
        pricing=Pricing(mode="fixed", price_cents=500),
        side_groups=[],
        modifier_groups=[],
        available=True,
    )


def make_item_with_groups() -> MenuItem:
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
        modifier_groups=[
            ModifierGroup(
                group_id="mods",
                name="Burger Modification",
                normalized_name=normalize_text("Burger Modification"),
                is_required=False,
                min_selector=0,
                max_selector=3,
                choices=[
                    ModifierChoice(
                        modifier_id="cheese",
                        name="Cheese",
                        normalized_name=normalize_text("Cheese"),
                        price_cents=100,
                    ),
                    ModifierChoice(
                        modifier_id="onion",
                        name="Onion",
                        normalized_name=normalize_text("Onion"),
                        price_cents=0,
                    ),
                ],
            )
        ],
        available=True,
    )


def test_add_item_handler_rejects_non_add_intent():
    repo = FakeMenuRepo(MenuQueryResult(type=MenuQueryType.NOT_FOUND))
    handler = AddItemHandler(repo)
    context = ConversationContext()

    result = handler.handle(
        intent=Intent.ASK_PRICE,
        context=context,
        user_text="price of burger",
        session=None,
    )

    assert result.next_state == ConversationState.IDLE
    assert result.response_key == "unhandled_intent"


def test_add_item_handler_enters_flow_for_resolved_item():
    item = make_item()
    repo = FakeMenuRepo(MenuQueryResult(type=MenuQueryType.ITEM, item=item))
    handler = AddItemHandler(repo)
    context = ConversationContext()

    result = handler.handle(
        intent=Intent.ADD_ITEM,
        context=context,
        user_text="add burger",
        session=None,
    )

    assert context.pending_action == PendingAction.ADD_ITEM
    assert context.current_item_id == item.item_id
    assert context.current_item_name == item.name
    assert result.response_key == "ask_for_quantity"
    assert result.next_state == ConversationState.WAITING_FOR_QUANTITY
    assert result.command is None


def test_add_item_handler_returns_ambiguous_confirmation():
    item1 = make_item()
    item2 = MenuItem(
        item_id="burger_2",
        name="Chicken Burger",
        normalized_name=normalize_text("Chicken Burger"),
        aliases=("chicken burger",),
        normalized_aliases=(normalize_text("chicken burger"),),
        voice_labels=("chicken burger",),
        pricing=Pricing(mode="fixed", price_cents=600),
        side_groups=[],
        modifier_groups=[],
        available=True,
    )

    repo = FakeMenuRepo(
        MenuQueryResult(
            type=MenuQueryType.ITEM_AMBIGUOUS,
            matched_items=[item1, item2],
        )
    )
    handler = AddItemHandler(repo)
    context = ConversationContext()

    result = handler.handle(
        intent=Intent.ADD_ITEM,
        context=context,
        user_text="add burger",
        session=None,
    )

    assert result.next_state == ConversationState.CONFIRMING_ITEM
    assert result.response_key == "confirm_item_ambiguous"
    assert result.response_payload["candidate_item_names"] == ["Zinger Burger", "Chicken Burger"]


def test_add_item_handler_returns_item_not_found():
    repo = FakeMenuRepo(MenuQueryResult(type=MenuQueryType.NOT_FOUND))
    handler = AddItemHandler(repo)
    context = ConversationContext()

    result = handler.handle(
        intent=Intent.ADD_ITEM,
        context=context,
        user_text="add dragon burger",
        session=None,
    )

    assert result.next_state == ConversationState.IDLE
    assert result.response_key == "item_not_found"
    assert result.response_payload == {
        "query": "dragon burger",
        "suggested_item_names": [],
        "suggested_category_names": [],
    }


def test_add_item_handler_prefills_only_choices_that_fit_the_current_item():
    item = make_item_with_groups()
    repo = FakeMenuRepo(MenuQueryResult(type=MenuQueryType.ITEM, item=item))
    handler = AddItemHandler(repo)
    context = ConversationContext()
    context.last_slots = (
        SlotValue(name="ITEM", value="Chicken Burger"),
        SlotValue(name="ITEM", value="Coke"),
        SlotValue(name="MODIFIER", value="Cheese"),
        SlotValue(name="ITEM", value="Rice"),
    )

    result = handler.handle(
        intent=Intent.ADD_ITEM,
        context=context,
        user_text="2 chicken burger with coke and cheese and rice",
        session=None,
    )

    assert result.response_key == "item_added_successfully"
    assert result.command is not None
    assert result.command["payload"]["quantity"] == 2
    assert result.command["payload"]["sides"] == {"drink": ["coke"]}
    assert result.command["payload"]["modifiers"] == {
        "mods": [
            {
                "modifier_id": "cheese",
                "name": "Cheese",
                "action": "add",
                "instruction": None,
            }
        ]
    }
