import unittest
from pathlib import Path

from app.menu.repository import MenuRepository
from app.menu.store import MenuStore
from app.nlu.intent_resolution.intent import Intent
from app.nlu.nlu_result import SlotValue
from app.state_machine.handlers.info.ask_price_handler import AskPriceHandler
from app.state_machine.models.conversation_context import ConversationContext
from app.state_machine.models.conversation_state import ConversationState


def _build_repo() -> MenuRepository:
    data_root = Path(__file__).resolve().parents[4] / "app" / "data" / "restaurants" / "steves_grill"
    store = MenuStore(
        menu_path=data_root / "menu.json",
        entity_index_path=data_root / "entity_index.json",
    )
    return MenuRepository(store)


class AskPriceHandlerTests(unittest.TestCase):
    def test_can_quote_modifier_price_when_item_is_present(self):
        repo = _build_repo()
        handler = AskPriceHandler(repo)
        context = ConversationContext()
        context.last_slots = (
            SlotValue(name="ITEM", value="Bourbon Chicken"),
            SlotValue(name="MODIFIER", value="Fresh Mushroom"),
        )

        result = handler.handle(
            intent=Intent.ASK_PRICE,
            context=context,
            user_text="how much is fresh mushroom on bourbon chicken",
            session=None,
        )

        self.assertEqual(result.next_state, ConversationState.IDLE)
        self.assertEqual(result.response_key, "show_modifier_price")
        self.assertEqual(
            result.response_payload,
            {
                "item_name": "Bourbon Chicken",
                "modifier_name": "Fresh Mushroom",
                "price": "$0.00",
            },
        )

    def test_rejects_modifier_without_item_context(self):
        repo = _build_repo()
        handler = AskPriceHandler(repo)
        context = ConversationContext()
        context.last_slots = (SlotValue(name="MODIFIER", value="Fresh Mushroom"),)

        result = handler.handle(
            intent=Intent.ASK_PRICE,
            context=context,
            user_text="how much is fresh mushroom",
            session=None,
        )

        self.assertEqual(result.next_state, ConversationState.IDLE)
        self.assertEqual(result.response_key, "modifier_requires_item_context")


if __name__ == "__main__":
    unittest.main()
