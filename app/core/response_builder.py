# app/core/response_builder.py

from typing import Callable, Optional, Dict

from app.menu.repository import MenuRepository
from app.state_machine.context import ConversationContext

# ---- imports of response functions ----
from app.responses.cart_responses import render_cart_summary
from app.responses.flow_control_responses import (
    flow_guard_finish_current_step,
    flow_guard_confirm_cancel,
    flow_guard_cancelled,
)
from app.responses.intent_not_allowed import handle_intent_not_allowed
from app.responses.item_responses import (
    ask_for_side,
    ask_for_modifier,
    ask_for_size,
    ask_item_quantity,
    required_side_cannot_skip,
    required_modifier_cannot_skip,
    required_size_cannot_skip,
    repeat_side_options,
    list_side_options,
    clarify_side_choice,
    repeat_modifier_options,
    list_modifier_options,
    clarify_modifier_choice,
    item_added_successfully,
)
from app.responses.menu_responses import (
    show_category_response,
    show_item_info_response,
    menu_ambiguity_response,
    menu_not_found_response,
    show_item_price_response,
)


ResponseFn = Callable[
    [ConversationContext, MenuRepository, dict],
    str,
]


class ResponseBuilder:
    """
    Maps response keys to rendering functions.

    Responsibilities:
    - Dispatch response_key → response function
    - Provide context + menu_repo + payload
    - Never contain formatting or business logic
    """

    def __init__(self, menu_repo: MenuRepository):
        self.menu_repo = menu_repo
        self._registry: Dict[str, ResponseFn] = self._build_registry()

    def build(
        self,
        response_key: str,
        context: ConversationContext,
        payload: Optional[dict] = None,
    ) -> str:
        payload = payload or {}

        renderer = self._registry.get(response_key)
        if not renderer:
            return "Sorry, I didn’t understand that."

        return renderer(context, self.menu_repo, payload)

    # --------------------------------------------------
    # Registry
    # --------------------------------------------------

    def _build_registry(self) -> Dict[str, ResponseFn]:
        """
        Central response registry.
        One response_key → one function.
        """

        return {
            # -------------------------
            # Flow control / guards
            # -------------------------
            "flow_guard_finish_current_step": self._flow_finish_step,
            "flow_guard_confirm_cancel": self._flow_confirm_cancel,
            "flow_guard_cancelled": self._flow_cancelled,

            # -------------------------
            # Errors / safety
            # -------------------------
            "intent_not_allowed": self._intent_not_allowed,
            "handler_not_implemented": lambda *_: "That feature isn’t available yet.",
            "confirmation_state_error": lambda *_: "Something went wrong. Let’s start over.",

            # -------------------------
            # Item build flow
            # -------------------------
            "confirm_item": self._confirm_item,
            "ask_for_side": lambda c, m, p: ask_for_side(c, m),
            "ask_for_modifier": lambda c, m, p: ask_for_modifier(c, m),
            "ask_for_size": lambda c, m, p: ask_for_size(c, m),
            "ask_for_quantity": lambda c, m, p: ask_item_quantity(p),

            # Required enforcement
            "required_side_cannot_skip": lambda c, m, p: required_side_cannot_skip(c, m),
            "required_modifier_cannot_skip": lambda c, m, p: required_modifier_cannot_skip(c, m),
            "required_size_cannot_skip": lambda c, m, p: required_size_cannot_skip(c, m),

            # Repeats / clarifications
            "repeat_side_options": repeat_side_options,
            "list_side_options": list_side_options,
            "clarify_side_choice": clarify_side_choice,
            "repeat_modifier_options": repeat_modifier_options,
            "list_modifier_options": list_modifier_options,
            "clarify_modifier_choice": clarify_modifier_choice,

            # Completion
            "item_added_successfully": lambda c, m, p: item_added_successfully(p),

            # -------------------------
            # Menu info
            # -------------------------
            "show_category": lambda c, m, p: show_category_response(p),
            "show_item_info": lambda c, m, p: show_item_info_response(p),
            "menu_ambiguity": lambda c, m, p: menu_ambiguity_response(p),
            "menu_not_found": lambda *_: menu_not_found_response(),
            "show_item_price": lambda c, m, p: show_item_price_response(p),

            # -------------------------
            # Cart
            # -------------------------
            "show_cart": lambda c, m, p: render_cart_summary(p),
            "show_total": lambda c, m, p: f"Your total so far is: {p['total']}",
            "cart_empty": lambda *_: "Your cart is empty. Please add items before placing an order.",
            "confirm_clear_cart": lambda *_: "Are you sure you want to clear your cart?",
            "cart_cleared": lambda *_: "Your cart has been cleared.",

            # -------------------------
            # Order / payment
            # -------------------------
            "confirm_order_summary": self._confirm_order_summary,
            "payment_link_sent": lambda *_: (
                "I’ve sent you a payment link.\n"
                "Please complete the payment and tell me once you’re done."
            ),
            "waiting_for_payment": lambda *_: "I’m waiting for your payment confirmation.",
            "order_completed": lambda *_: (
                "Payment confirmed!\n"
                "Your order will be ready in 25 minutes.\n"
                "Thank you for calling Compass."
            ),
        }

    # --------------------------------------------------
    # Private adapters (thin glue only)
    # --------------------------------------------------

    def _intent_not_allowed(self, _: ConversationContext, __: MenuRepository, payload: dict) -> str:
        return handle_intent_not_allowed(payload)

    def _flow_finish_step(self, _: ConversationContext, __: MenuRepository, payload: dict) -> str:
        return flow_guard_finish_current_step(payload)

    def _flow_confirm_cancel(self, _: ConversationContext, __: MenuRepository, payload: dict) -> str:
        return flow_guard_confirm_cancel(payload)

    def _flow_cancelled(self, *_):
        return flow_guard_cancelled()

    def _confirm_item(
        self,
        context: ConversationContext,
        menu_repo: MenuRepository,
        _: dict,
    ) -> str:
        item = menu_repo.store.get_item(context.candidate_item_id)
        return f"You want a {item.name}, right? Please say yes or no."

    def _confirm_order_summary(
        self,
        _: ConversationContext,
        __: MenuRepository,
        payload: dict,
    ) -> str:
        summary = render_cart_summary(payload)
        return f"{summary}\n\nWould you like to proceed?"
