# app/core/response_builder.py
from typing import Optional

from app.menu.models import MenuItem
from app.responses.cart_responses import render_cart_summary
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

        if response_key == "confirm_order_summary":
            return render_cart_summary(payload)

        if response_key == "order_completed":
            return (
                "Payment confirmed!\n"
                "Your order will be ready in 25 minutes.\n"
                "Thank you for calling the compass."
            )

        if response_key == "cart_empty":
            return "Your cart is empty. Please add items before placing an order."

        if response_key == "order_cancelled":
            return "No problem. You can continue adding items."

        if response_key == "cart_show":
            return "Here is what you have in your cart."

        if response_key == "payment_link_sent":
            return (
                "I’ve sent you a payment link.\n"
                "Please complete the payment and tell me once you’re done."
            )

        if response_key == "waiting_for_payment":
            return "I’m waiting for your payment confirmation."

        if response_key == "repeat_order_confirmation":
            return "Please say yes to proceed or no to cancel."

        # fallback
        return "Sorry, I didn’t understand that."
