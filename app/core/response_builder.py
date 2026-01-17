# app/core/response_builder.py
from typing import Optional

from app.menu.models import MenuItem
from app.responses.item_responses import ask_for_side, ask_for_modifier, item_added_successfully, ask_for_size
from app.state_machine.context import ConversationContext
from app.menu.repository import MenuRepository


class ResponseBuilder:
    """
    Converts response keys + context into user-facing text.
    """

    def __init__(self, menu_repo: MenuRepository):
        self.menu_repo = menu_repo

    def build(
            self,
            response_key: str,
            context: ConversationContext,
            payload: Optional[dict] = None,
    ) -> str:
        if response_key == "confirm_item":
            item = self.menu_repo.store.get_item(context.candidate_item_id)
            return f"You want a {item.name}, right? Please say yes or no."

        if response_key == "ask_for_side":
            return ask_for_side(context, self.menu_repo)

        if response_key == "ask_for_modifier":
            return ask_for_modifier(context, self.menu_repo)

        if response_key == "ask_for_size":
            return ask_for_size(context, self.menu_repo)

        if response_key == "ask_for_quantity":
            return "How many would you like?"

        if response_key == "item_added_successfully":
            return item_added_successfully(payload)

        if response_key == "item_not_found":
            return "Sorry, I couldn't find that item. Please try again."

        if response_key == "finish_current_item_first":
            return "Let’s finish adding the current item first."

        if response_key == "cart_empty":
            return "Your cart is empty."

        if response_key == "cart_show":
            return "Here is what you have in your cart."

        # fallback
        return "Sorry, I didn’t understand that."
