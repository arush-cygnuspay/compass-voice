from unittest.mock import patch

from app.core.turn_engine import TurnEngine
from app.menu.models import MenuItem, Pricing
from app.menu.repository import MenuRepository
from app.nlu.intent_resolution.intent import Intent
from app.session.session import Session
from app.state_machine.conversation_state import ConversationState
from app.state_machine.state_router import StateRouter


class FakeMenuStore:
    def __init__(self, item):
        self.items = {item.item_id: item}

    def get_item(self, item_id):
        return self.items[item_id]

    def find_category_by_name(self, text):
        return None

    def find_entity(self, text, **kwargs):
        return []

    def find_item_exact(self, name):
        lowered = name.lower()
        for item in self.items.values():
            if item.name.lower() == lowered:
                return item
        return None


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


def test_turn_engine_add_item_happy_path():
    item = make_item()
    menu_repo = MenuRepository(FakeMenuStore(item))
    router = StateRouter()
    engine = TurnEngine(router=router, menu_repo=menu_repo)

    session = Session(session_id="s1", restaurant_id="r1")

    fake_nlu = type(
        "FakeNLU",
        (),
        {
            "intent": Intent.ADD_ITEM,
            "intent_confidence": 0.99,
            "normalized_text": "zinger burger",
            "slots": (),
        },
    )()

    with patch("app.core.turn_engine.resolve_nlu", return_value=fake_nlu):
        out = engine.process_turn(session, "add zinger burger")

    assert out.response_key == "item_added_successfully"
    assert session.conversation_state == ConversationState.IDLE
    assert len(session.cart.items) == 1
    assert session.last_intent == Intent.ADD_ITEM