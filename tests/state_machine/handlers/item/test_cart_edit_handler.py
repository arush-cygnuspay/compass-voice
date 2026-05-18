import unittest
from pathlib import Path

from app.cart.cart_item import CartItem
from app.menu.repository import MenuRepository
from app.menu.store import MenuStore
from app.nlu.intent_resolution.intent import Intent
from app.session.session import Session
from app.state_machine.handlers.item.modifying_item_handler import ModifyingItemHandler
from app.state_machine.handlers.item.remove_item_handler import RemoveItemHandler
from app.state_machine.models.conversation_context import ConversationContext
from app.state_machine.models.conversation_state import ConversationState


def _build_repo() -> MenuRepository:
    data_root = Path(__file__).resolve().parents[4] / "app" / "data" / "restaurants" / "steves_grill"
    store = MenuStore(
        menu_path=data_root / "menu.json",
        entity_index_path=data_root / "entity_index.json",
    )
    return MenuRepository(store)


def _build_session(repo: MenuRepository) -> Session:
    session = Session(session_id="edit-1", restaurant_id="steves_grill")
    session.conversation_state = ConversationState.IDLE
    item = repo.resolve_menu_query("bourbon chicken", limit=5).item
    session.cart.add_item(
        CartItem.create(
            item_id=item.item_id,
            quantity=1,
            variant_id=None,
            sides={},
            side_variants={},
            modifiers={},
        )
    )
    return session


class CartEditHandlerTests(unittest.TestCase):
    def test_replace_item_request_can_extract_old_and_new_items(self):
        repo = _build_repo()
        handler = RemoveItemHandler(repo)
        session = _build_session(repo)
        context = ConversationContext()

        result = handler.handle(
            intent=Intent.REPLACE_ITEM,
            context=context,
            user_text="replace bourbon chicken with chicken taco",
            session=session,
        )

        self.assertEqual(result.next_state, ConversationState.MODIFYING_ITEM)
        self.assertEqual(result.response_key, "confirm_replace_item")
        self.assertEqual(
            result.response_payload,
            {
                "item_name": "Bourbon Chicken",
                "replacement_item_name": "Chicken Taco",
            },
        )

    def test_modify_item_confirmation_restarts_add_flow_and_removes_old_cart_item(self):
        repo = _build_repo()
        remove_handler = RemoveItemHandler(repo)
        modify_handler = ModifyingItemHandler(repo)
        session = _build_session(repo)
        context = ConversationContext()

        first = remove_handler.handle(
            intent=Intent.MODIFY_ITEM,
            context=context,
            user_text="modify bourbon chicken",
            session=session,
        )
        self.assertEqual(first.next_state, ConversationState.MODIFYING_ITEM)
        self.assertEqual(first.response_key, "confirm_modify_item")

        session.conversation_state = ConversationState.MODIFYING_ITEM
        cart_item_id = session.cart.get_items()[0].cart_item_id
        second = modify_handler.handle(
            intent=Intent.CONFIRM,
            context=context,
            user_text="yes",
            session=session,
        )

        self.assertEqual(second.next_state, ConversationState.WAITING_FOR_MODIFIER)
        self.assertEqual(second.response_key, "ask_for_modifier")
        self.assertEqual(
            second.command,
            {
                "type": "REMOVE_ITEM_FROM_CART",
                "payload": {"cart_item_id": cart_item_id},
            },
        )


if __name__ == "__main__":
    unittest.main()
