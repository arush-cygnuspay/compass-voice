from app.core.pending_action import PendingAction
from app.menu.models import MenuItem, Pricing
from app.menu.query_result import MenuQueryResult, MenuQueryType
from app.nlu.intent_resolution.intent import Intent
from app.state_machine.models.conversation_context import ConversationContext
from app.state_machine.models.conversation_state import ConversationState
from app.state_machine.handlers.item.add_item.add_item_handler import AddItemHandler


class FakeMenuRepo:
    def __init__(self, result: MenuQueryResult):
        self._result = result

    def resolve_menu_query_from_slots(self, **kwargs):
        return self._result

    def resolve_menu_query(self, text: str, limit: int = 5):
        return self._result


def make_item() -> MenuItem:
    return MenuItem(
        item_id="burger_1",
        name="Zinger Burger",
        aliases=["zinger burger"],
        pricing=Pricing(mode="fixed", price_cents=500),
        side_groups=[],
        modifier_groups=[],
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
    assert result.response_key == "item_added_successfully"
    assert result.next_state == ConversationState.IDLE
    assert result.command is not None
    assert result.command["type"] == "ADD_ITEM_TO_CART"


def test_add_item_handler_returns_ambiguous_confirmation():
    item1 = make_item()
    item2 = MenuItem(
        item_id="burger_2",
        name="Chicken Burger",
        aliases=["chicken burger"],
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
    assert result.response_payload == {"query": "add dragon burger"}