# app/core/response_builder.py
from typing import Optional

from app.responses.cart_responses import render_cart_summary
from app.responses.item_responses import *
from app.responses.menu_responses import *
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

        payload = payload or {}

        # -------------------------
        # Meta / Safety / Errors
        # -------------------------
        if response_key == "intent_not_allowed":
            return "Sorry, I can’t do that right now."

        if response_key == "handler_not_implemented":
            return "That feature isn’t available yet."

        if response_key == "confirmation_state_error":
            return "Something went wrong. Let’s start over."

        # -------------------------
        # Item confirmation & build flow
        # -------------------------
        if response_key == "confirm_item":
            item = self.menu_repo.store.get_item(context.candidate_item_id)
            return f"You want a {item.name}, right? Please say yes or no."

        if response_key == "ask_for_side":
            return ask_for_side(context, self.menu_repo)

        if response_key == "ask_for_modifier":
            return ask_for_modifier(context, self.menu_repo)

        # -------------------------
        # Required group enforcement
        # -------------------------
        if response_key == "required_side_cannot_skip":
            return required_side_cannot_skip(context, self.menu_repo)

        if response_key == "required_modifier_cannot_skip":
            return required_modifier_cannot_skip(context, self.menu_repo)

        # -------------------------
        # Repeat / clarification prompts
        # -------------------------
        if response_key == "repeat_side_options":
            return ask_for_side(context, self.menu_repo)

        if response_key == "repeat_modifier_options":
            return ask_for_modifier(context, self.menu_repo)

        if response_key == "ask_for_size":
            return ask_for_size(context, self.menu_repo)

        if response_key == "required_size_cannot_skip":
            return required_size_cannot_skip(context, self.menu_repo)

        if response_key == "repeat_size_options":
            return required_size_cannot_skip(context, self.menu_repo)

        if response_key == "repeat_size_options":
            return required_size_cannot_skip(context, self.menu_repo)

        if response_key == "ask_for_quantity":
            return "How many would you like?"

        if response_key == "item_added_successfully":
            return item_added_successfully(payload)

        if response_key == "item_not_found":
            return "Sorry, I couldn't find that item. Please try again."

        if response_key == "finish_current_item_first":
            return "Let’s finish adding the current item first."

        # -------------------------
        # Menu Info (NEW)
        # -------------------------
        if response_key == "show_category":
            return show_category_response(payload)

        if response_key == "show_item_info":
            return show_item_info_response(payload)

        if response_key == "menu_ambiguity":
            return menu_ambiguity_response(payload)

        if response_key == "menu_not_found":
            return menu_not_found_response()

        if response_key == "show_item_price":
            return show_item_price_response(payload)

        if response_key == "price_not_found":
            return "Sorry, I couldn’t find pricing for that item."

        # -------------------------
        # Cart display utilities
        # -------------------------
        if response_key == "show_cart":
            return render_cart_summary(payload)

        if response_key == "show_total":
            return f"Your total so far is: {payload['total']}"

        if response_key == "cart_empty":
            return "Your cart is empty. Please add items before placing an order."

        if response_key == "confirm_clear_cart":
            return "Are you sure you want to clear your cart?"

        if response_key == "cart_cleared":
            return "Your cart has been cleared."

        # -------------------------
        # Remove item flow
        # -------------------------
        if response_key == "cart_is_empty":
            return "Your cart is empty. There's nothing to remove."

        if response_key == "item_not_found_in_cart":
            return "I couldn't find that item in your cart."

        if response_key == "confirm_remove_item":
            item_name = payload.get("item_name", "that item")
            quantity = payload.get("quantity", 1)
            if quantity > 1:
                return (
                    f"I found {quantity} {item_name}s in your cart. "
                    "Should I remove them? Please say yes or no."
                )
            return (
                f"You have {item_name} in your cart. "
                "Should I remove it? Please say yes or no."
            )

        if response_key == "repeat_remove_confirmation":
            item_name = context.current_item_name or "that item"
            return (
                f"Should I remove {item_name} from your cart? "
                "Please say yes or no."
            )

        if response_key == "item_removal_cancelled":
            item_name = context.current_item_name or "that item"
            return f"Okay, keeping {item_name} in your cart."

        if response_key == "item_removed_successfully":
            item_name = payload.get("item_name", "the item")
            return (
                f"I've removed {item_name} from your cart. "
                "Would you like to remove anything else?"
            )

        # -------------------------
        # Flow resume / acknowledgements
        # -------------------------
        if response_key == "resume_previous_state":
            return "Okay, continuing."

        if response_key == "awaiting_ack":
            return "Okay."

        if response_key == "resume_order_confirmation":
            return "Okay, let’s continue with your order."

        if response_key == "order_cancelled":
            return "No problem. You can continue adding items."

        # -------------------------
        # Order & payment flow
        # -------------------------
        if response_key == "confirm_order_summary":
            summary = render_cart_summary(payload)
            return f"{summary}\n\nWould you like to proceed?"

        if response_key == "payment_link_sent":
            return (
                "I’ve sent you a payment link.\n"
                "Please complete the payment and tell me once you’re done."
            )

        if response_key == "waiting_for_payment":
            return "I’m waiting for your payment confirmation."

        if response_key == "order_completed":
            return (
                "Payment confirmed!\n"
                "Your order will be ready in 25 minutes.\n"
                "Thank you for calling the compass."
            )

        # -------------------------
        # Fallback (last resort)
        # -------------------------
        return "Sorry, I didn’t understand that."

