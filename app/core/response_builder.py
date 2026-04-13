# app/core/response_builder.py
from __future__ import annotations

from typing import Callable

from app.responses.cart_responses import (
    cart_cleared_response,
    clear_cart_cancelled_response,
    confirm_clear_cart_response,
    render_cart_summary, render_checkout_review_summary,
)
from app.responses.flow_control_responses import (
    flow_guard_cancelled,
    flow_guard_confirm_cancel,
    flow_guard_finish_current_step,
)
from app.responses.intent_not_allowed import handle_intent_not_allowed
from app.responses.item_responses import *
from app.responses.menu_responses import (
    menu_ambiguity_response,
    menu_not_found_response,
    show_category_response,
    show_item_availability_response,
    show_item_info_response,
    show_item_price_response,
    show_menu_categories_response,
)
from app.state_machine.models.conversation_context import ConversationContext

ResponseFn = Callable[[ConversationContext, MenuRepository, dict], str]


class ResponseBuilder:
    """
    Builds short telephony-friendly responses from response keys.
    """

    def __init__(self, menu_repo: MenuRepository) -> None:
        self.menu_repo = menu_repo
        self._registry: dict[str, ResponseFn] = self._build_registry()

    def build(
        self,
        response_key: str,
        context: ConversationContext,
        payload: dict | None = None,
    ) -> str:
        payload = payload or {}
        renderer = self._registry.get(response_key)
        if renderer is None:
            return "Sorry, I didn’t understand that."
        return renderer(context, self.menu_repo, payload)

    def _build_registry(self) -> dict[str, ResponseFn]:
        return {
            "idle_nothing_to_checkout": lambda *_: "Your cart is empty. Add something first.",
            "resume_shopping": lambda *_: "Okay. Add more, remove items, or check your cart.",
            "flow_guard_finish_current_step": self._flow_finish_step,
            "flow_guard_confirm_cancel": self._flow_confirm_cancel,
            "flow_guard_cancelled": self._flow_cancelled,
            "intent_not_allowed": self._intent_not_allowed,
            "handler_not_implemented": lambda *_: "That feature isn’t ready yet.",
            "confirmation_state_error": lambda *_: "Something went wrong. Start again.",
            "confirm_item": self._confirm_item,
            "ask_for_side": lambda c, m, p: ask_for_side(c, m),
            "ask_for_modifier": lambda c, m, p: ask_for_modifier(c, m),
            "ask_for_size": lambda c, m, p: ask_for_size(c, m),
            "ask_for_side_size": self._ask_for_side_size,
            "ask_for_quantity": lambda c, m, p: ask_item_quantity(p),
            "confirm_item_ambiguous": confirm_item_ambiguous,
            "confirm_item_from_category": confirm_item_from_category,
            "item_not_found": item_not_found,
            "repeat_item_request": repeat_item_request,
            "required_side_cannot_skip": lambda c, m, p: required_side_cannot_skip(c, m),
            "required_modifier_cannot_skip": lambda c, m, p: required_modifier_cannot_skip(c, m),
            "required_size_cannot_skip": lambda c, m, p: required_size_cannot_skip(c, m),
            "repeat_side_options": repeat_side_options,
            "list_side_options": list_side_options,
            "clarify_side_choice": clarify_side_choice,
            "too_many_side_choices": too_many_side_choices,
            "repeat_modifier_options": repeat_modifier_options,
            "list_modifier_options": list_modifier_options,
            "clarify_modifier_choice": clarify_modifier_choice,
            "too_many_modifier_choices": too_many_modifier_choices,
            "repeat_size_options": repeat_size_options,
            "invalid_size_option": invalid_size_option,
            "invalid_side_size_option": self._invalid_side_size_option,
            "repeat_side_size_options": self._repeat_side_size_options,
            "required_side_size_cannot_skip": self._required_side_size_cannot_skip,
            "invalid_quantity_option": invalid_quantity_option,
            "item_context_missing": item_context_missing,
            "size_not_applicable": size_not_applicable,
            "item_added_successfully": lambda c, m, p: item_added_successfully(p),
            "confirm_cancel_current_item": confirm_cancel_current_item,
            "confirm_cancel_current_item_for_new_request": confirm_cancel_current_item_for_new_request,
            "continue_current_item_after_cancel_denied": continue_current_item_after_cancel_denied,
            "item_cancelled_successfully": item_cancelled_successfully,
            "show_menu_categories": lambda c, m, p: show_menu_categories_response(p),
            "show_category": lambda c, m, p: show_category_response(p),
            "show_item_availability": lambda c, m, p: show_item_availability_response(p),
            "show_item_info": lambda c, m, p: show_item_info_response(p),
            "menu_ambiguity": lambda c, m, p: menu_ambiguity_response(p),
            "menu_not_found": lambda *_: menu_not_found_response(),
            "show_item_price": lambda c, m, p: show_item_price_response(p),
            "price_not_found": lambda *_: "I couldn’t find that item. Say the item name again.",
            "readonly_interrupt_with_resume": self._readonly_interrupt_with_resume,
            "show_cart": lambda c, m, p: render_cart_summary(p),
            "show_total": lambda c, m, p: f"Your total is {p.get('total', '$0.00')}.",
            "cart_empty": lambda *_: "Your cart is empty.",
            "confirm_clear_cart": lambda *_: confirm_clear_cart_response(),
            "cart_cleared": lambda *_: cart_cleared_response(),
            "clear_cart_cancelled": lambda *_: clear_cart_cancelled_response(),
            "confirm_order_summary": self._confirm_order_summary,
            "payment_link_sent": lambda *_: "Payment link sent. Tell me when done.",
            "waiting_for_payment": lambda *_: "Waiting for payment. Tell me when it’s done.",
            "order_completed": lambda *_: (
                "Payment confirmed. Order placed. It will be ready in 25 minutes. Thank you."
            ),
            "no_active_order_to_cancel": lambda *_: "There’s no active order to cancel.",
            "no_active_payment": lambda *_: "There’s no payment in progress.",
            "payment_not_started": lambda *_: "Payment has not started. Say checkout when ready.",
            "order_cancelled": lambda *_: "Okay, checkout cancelled. Your cart is still here.",
            "confirm_side_choice_guess": lambda c, m, p: f"Did you mean {p.get('choice_name', 'that side')}? Yes or no.",
            "confirm_modifier_choice_guess": lambda c, m, p: f"Did you mean {p.get('choice_name', 'that modifier')}? Yes or no.",
            "confirm_size_choice_guess": lambda c, m, p: f"Did you mean {p.get('choice_name', 'that size')}? Yes or no.",
            "confirm_side_size_choice_guess": lambda c, m, p: f"Did you mean {p.get('choice_name', 'that size')} for {p.get('side_item_name', 'that side')}? Yes or no.",

            "ask_for_order_type": lambda *_: "Welcome to Compass. Is this for pickup or delivery today?",
            "repeat_order_type": lambda *_: "Is this for pickup or delivery?",
            "order_type_captured_pickup": lambda *_: "Got it. Pickup. What would you like to order?",
            "order_type_captured_delivery": lambda *_: "Got it. Delivery. What would you like to order?",

            "checkout_blocked_finish_current_item": lambda *_: "Please finish this item first, or say cancel.",

            "payment_link_send_failed": lambda *_: "I could not send the payment link right now. Please try again.",
            "checkout_link_send_failed": lambda *_: "I could not send the checkout link right now. Please try again.",

            "ask_for_delivery_area": lambda *_: "Got it. Delivery. Please say your delivery area.",
            "repeat_delivery_area": lambda *_: "Please say your delivery area.",
            "ask_for_delivery_zip": lambda *_: "Now please say your ZIP code.",
            "repeat_delivery_zip": lambda *_: "Please say the ZIP code.",
            "confirm_delivery_area_zip": lambda c, m, p: (
                f"Just to confirm, that is {p.get('area', 'your area')}, ZIP code {p.get('postal_code', '')}. Is that correct?"
            ),
            "repeat_delivery_area_zip_confirmation": lambda c, m, p: (
                f"I have {p.get('area', 'your area')}, ZIP code {p.get('postal_code', '')}. Is that correct?"
            ),
            "delivery_area_confirmed": lambda *_: "Great, we deliver there. What would you like to order?",

            "ask_delivery_address_method": lambda *_: (
                "I’ve sent you one secure checkout link. Please fill in your delivery address and complete payment so I can place your order."
            ),
            "checkout_link_unavailable_fallback_voice": lambda *_: (
                "I can’t send the checkout link right now, so I’ll take your delivery address here. Please say your house number."
            ),
            "checkout_link_sent": lambda *_: (
                "I’ve sent you one secure checkout link. Please confirm your delivery address there, allow location access if you want, and complete payment. Tell me when done."
            ),
            "waiting_for_checkout_completion": lambda *_: (
                "I’m waiting for you to finish the checkout link. Tell me when payment is done."
            ),

            "ask_for_delivery_house_number": lambda *_: "Please say your house number.",
            "repeat_delivery_house_number": lambda *_: "Please say your house number.",
            "ask_for_delivery_street": lambda *_: "Please say your street name or street number.",
            "repeat_delivery_street": lambda *_: "Please say your street name or street number.",

            "confirm_delivery_house_number": lambda c, m, p: (
                f"I heard {p.get('house_number', 'that number')}. Is that correct?"
            ),
            "confirm_delivery_street": lambda c, m, p: (
                f"I heard {p.get('street', 'that street')}. Is that correct?"
            ),
            "ask_for_delivery_secondary_address": lambda *_: (
                "Any apartment, suite, unit, building, or block? If none, say none."
            ),
            "confirm_delivery_secondary_address": lambda c, m, p: (
                f"I heard {p.get('secondary_address', 'that address detail')}. Is that correct?"
            ),
            "delivery_address_captured_resume_checkout": lambda *_: (
                "Got it. I have your delivery address. I’m sending your payment link now."
            ),
            "checkout_link_failed_fallback_voice": lambda *_: (
                "I couldn’t send the checkout link right now, so I’ll take your delivery address here. Please say your house number."
            ),
            "payment_link_unavailable_now": lambda *_: (
                "I’m sorry, I couldn’t send the payment link right now, so I’m unable to complete the order at the moment."
            ),
        }

    def _intent_not_allowed(self, _: ConversationContext, __: MenuRepository, payload: dict) -> str:
        return handle_intent_not_allowed(payload)

    def _flow_finish_step(self, _: ConversationContext, __: MenuRepository, payload: dict) -> str:
        return flow_guard_finish_current_step(payload)

    def _flow_confirm_cancel(self, _: ConversationContext, __: MenuRepository, payload: dict) -> str:
        return flow_guard_confirm_cancel(payload)

    def _flow_cancelled(self, *_: object) -> str:
        return flow_guard_cancelled()

    def _confirm_item(self, context: ConversationContext, menu_repo: MenuRepository, payload: dict) -> str:
        item_name = payload.get("item_name") or context.current_item_name
        if not item_name and context.candidate_item_id:
            item = menu_repo.store.get_item(context.candidate_item_id)
            item_name = item.name
        item_name = item_name or "that item"
        return f"{item_name}, right? Yes or no."

    def _confirm_order_summary(self, context: ConversationContext, _: MenuRepository, payload: dict) -> str:
        return f"{render_checkout_review_summary(payload=payload, order_type=context.order_type)} Would you like to checkout?"

    def _ask_for_side_size(self, _: ConversationContext, __: MenuRepository, payload: dict) -> str:
        side_item_name = payload.get("side_item_name") or "that side"
        available_sizes = [str(x).strip() for x in (payload.get("available_sizes") or []) if str(x).strip()]

        if not available_sizes:
            return f"What size for {side_item_name}?"
        if len(available_sizes) == 1:
            return f"Size for {side_item_name}? {available_sizes[0]}."
        if len(available_sizes) == 2:
            return f"Size for {side_item_name}? {available_sizes[0]} or {available_sizes[1]}."
        return f"Size for {side_item_name}? {available_sizes[0]}, {available_sizes[1]}, or {available_sizes[2]}."

    def _repeat_side_size_options(self, _: ConversationContext, __: MenuRepository, payload: dict) -> str:
        side_item_name = payload.get("side_item_name") or "that side"
        available_sizes = [str(x).strip() for x in (payload.get("available_sizes") or []) if str(x).strip()]

        if not available_sizes:
            return f"What size for {side_item_name}?"
        if len(available_sizes) == 1:
            return f"Choose {available_sizes[0]}."
        if len(available_sizes) == 2:
            return f"Choose {available_sizes[0]} or {available_sizes[1]}."
        return f"Choose {available_sizes[0]}, {available_sizes[1]}, or {available_sizes[2]}."

    def _required_side_size_cannot_skip(self, _: ConversationContext, __: MenuRepository, payload: dict) -> str:
        return self._repeat_side_size_options(_, __, payload)

    def _invalid_side_size_option(self, _: ConversationContext, __: MenuRepository, payload: dict) -> str:
        return self._repeat_side_size_options(_, __, payload)

    def _readonly_interrupt_with_resume(
        self,
        context: ConversationContext,
        menu_repo: MenuRepository,
        payload: dict,
    ) -> str:
        interrupt_key = payload.get("interrupt_response_key")
        interrupt_payload = payload.get("interrupt_response_payload") or {}
        resume_key = payload.get("resume_response_key")
        resume_payload = payload.get("resume_response_payload") or {}

        interrupt_renderer = self._registry.get(interrupt_key)
        resume_renderer = self._registry.get(resume_key)

        if interrupt_renderer is None:
            interrupt_text = "Sorry, I couldn’t process that."
        else:
            interrupt_text = interrupt_renderer(context, menu_repo, interrupt_payload)

        if resume_renderer is None:
            return interrupt_text

        resume_text = resume_renderer(context, menu_repo, resume_payload)
        return f"{interrupt_text} {resume_text}"