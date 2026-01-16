# app/core/response_builder.py

from app.menu.models import MenuItem
from app.state_machine.context import ConversationContext
from app.menu.repository import MenuRepository


class ResponseBuilder:
    """
    Converts response keys + context into user-facing text.
    """

    def __init__(self, menu_repo: MenuRepository):
        self.menu_repo = menu_repo

    def build(self, response_key: str, context: ConversationContext) -> str:
        if response_key == "confirm_item":
            item = self.menu_repo.store.get_item(context.candidate_item_id)
            return f"You want a {item.name}, right? Please say yes or no."

        if response_key == "ask_for_side":
            item = self.menu_repo.store.get_item(context.current_item_id)
            lines = [f"Which side would you like with your {item.name}?"]
            for group in item.side_groups:
                for i, choice in enumerate(group.choices, start=1):
                    lines.append(f"{i}. {choice.name}")
            return "\n".join(lines)

        if response_key == "ask_for_modifier":
            item = self.menu_repo.store.get_item(context.current_item_id)
            lines = [f"Would you like to add any modifiers to your {item.name}?"]
            for group in item.modifier_groups:
                for i, choice in enumerate(group.choices, start=1):
                    lines.append(f"{i}. {choice.name}")
            lines.append("You can also say no.")
            return "\n".join(lines)

        if response_key == "ask_for_size":
            item = self.menu_repo.store.get_item(context.current_item_id)
            variants = item.pricing.variants or []
            lines = ["Which size would you like?"]
            for v in variants:
                lines.append(f"- {v.label}")
            return "\n".join(lines)

        if response_key == "ask_for_quantity":
            return "How many would you like?"

        if response_key == "item_added_successfully":
            return "Got it! Your item has been added to the cart. Would you like to add anything else?"

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
