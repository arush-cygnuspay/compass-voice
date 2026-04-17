import unittest

from app.core.pending_action import PendingAction
from app.menu.models import MenuItem, ModifierChoice, ModifierGroup, Pricing, SideChoice, SideGroup
from app.menu.query_result import MenuQueryResult, MenuQueryType
from app.nlu.intent_resolution.intent import Intent
from app.nlu.nlu_result import SlotValue
from app.nlu.query_normalization.text_preprocessor import normalize_text
from app.state_machine.handlers.item.add_item.add_item_handler import AddItemHandler
from app.state_machine.models.conversation_context import ConversationContext
from app.state_machine.models.conversation_state import ConversationState


class FakeMenuRepo:
    def __init__(self, result: MenuQueryResult):
        self._result = result

    def resolve_menu_query_from_slots_normalized(self, **kwargs):
        return self._result

    def resolve_menu_query_normalized(self, text: str, limit: int = 5):
        return self._result


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
                ],
            )
        ],
        available=True,
    )


class AddItemQuantityPromptTests(unittest.TestCase):
    def test_always_asks_for_quantity_even_when_present_in_first_request(self):
        item = make_item_with_groups()
        repo = FakeMenuRepo(MenuQueryResult(type=MenuQueryType.ITEM, item=item))
        handler = AddItemHandler(repo)
        context = ConversationContext()
        context.last_slots = (
            SlotValue(name="ITEM", value="Chicken Burger"),
            SlotValue(name="QUANTITY", value=2),
        )

        result = handler.handle(
            intent=Intent.ADD_ITEM,
            context=context,
            user_text="2 chicken burger with coke and cheese",
            session=None,
        )

        self.assertEqual(context.pending_action, PendingAction.ADD_ITEM)
        self.assertEqual(result.next_state, ConversationState.WAITING_FOR_QUANTITY)
        self.assertEqual(result.response_key, "ask_for_quantity")
        self.assertIsNone(context.quantity)
        self.assertEqual(context.selected_side_groups, {"drink": ["coke"]})
        self.assertEqual(
            context.selected_modifier_groups["mods"][0].modifier_id,
            "cheese",
        )


if __name__ == "__main__":
    unittest.main()
