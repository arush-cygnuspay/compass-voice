# app/core/response_builder.py
from __future__ import annotations

from typing import Callable

from app.menu.repository import MenuRepository
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
from app.responses.item_responses import (
    ask_for_modifier,
    ask_for_side,
    ask_for_size,
    ask_item_quantity,
    clarify_modifier_choice,
    clarify_side_choice,
    confirm_cancel_current_item,
    confirm_cancel_current_item_for_new_request,
    confirm_item_ambiguous,
    confirm_item_from_category,
    continue_current_item_after_cancel_denied,
    invalid_quantity_option,
    invalid_size_option,
    item_added_successfully,
    item_cancelled_successfully,
    item_context_missing,
    item_not_found,
    list_modifier_options,
    list_side_options,
    repeat_item_request,
    repeat_modifier_options,
    repeat_side_options,
    repeat_size_options,
    required_modifier_cannot_skip,
    required_side_cannot_skip,
    required_size_cannot_skip,
    size_not_applicable,
    too_many_modifier_choices,
    too_many_side_choices,
    _build_entity_feedback,
    _added_text,
    _format_item_summary_list,
)
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

        base_response = renderer(context, self.menu_repo, payload)

        # ── Multi-item acknowledgment prefix ──
        # When the user said multiple items in one go, prefix the first
        # item’s prompt with a summary of what we heard.
        if payload.get("multi_item_ack"):
            prefix = self._build_multi_item_ack_prefix(payload)
            if prefix:
                base_response = f"{prefix} {base_response}"

        # ── Queue transition prefix ──
        # When auto-advancing from one queued item to the next,
        # prefix with "Added X. Now for Y."
        if payload.get("queue_transition"):
            prefix = self._build_queue_transition_prefix(payload)
            if prefix:
                base_response = f"{prefix} {base_response}"

        # ── Prefilled confirmation ──
        # When the handler pre-captured sides/modifiers/size from the
        # user’s utterance, confirm what was heard before asking the
        # next question.  e.g. "Chicken Taco with Coke, extra Cheese — got it."
        prefix_parts: list[str] = []

        prefilled = payload.get("prefilled_summary")
        if prefilled:
            item_name = payload.get("prefilled_item_name") or payload.get("item_name") or ""
            confirm = self._build_prefilled_confirmation(item_name, prefilled)
            if confirm:
                prefix_parts.append(confirm)

        prefill_feedback = str(payload.get("prefill_feedback") or "").strip()
        if prefill_feedback:
            prefix_parts.append(prefill_feedback)

        if prefix_parts:
            base_response = f"{' '.join(prefix_parts)} {base_response}"

        return base_response

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
            "ask_for_side": ask_for_side,
            "ask_for_modifier": ask_for_modifier,
            "ask_for_size": ask_for_size,
            "ask_for_side_size": self._ask_for_side_size,
            "ask_for_quantity": lambda c, m, p: ask_item_quantity(p),
            "confirm_item_ambiguous": confirm_item_ambiguous,
            "confirm_item_from_category": confirm_item_from_category,
            "item_not_found": item_not_found,
            "repeat_item_request": repeat_item_request,
            "required_side_cannot_skip": required_side_cannot_skip,
            "required_modifier_cannot_skip": required_modifier_cannot_skip,
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
            "confirm_remove_item": lambda c, m, p: (
                f"Remove {p.get('item_name', 'that item')}?"
            ),
            "confirm_modify_item": lambda c, m, p: (
                f"Update {p.get('item_name', 'that item')}? I'll swap it with the new version."
            ),
            "confirm_replace_item": lambda c, m, p: (
                f"Replace {p.get('item_name', 'that item')} with {p.get('replacement_item_name', 'the new one')}?"
            ),
            "ask_replacement_item": lambda c, m, p: (
                f"What would you like instead of {p.get('item_name', 'that item')}?"
            ),
            "item_removed_successfully": lambda c, m, p: (
                f"Removed {p.get('item_name', 'that item')}. Anything else?"
            ),
            "item_removal_cancelled": lambda *_: "Okay, keeping it.",
            "item_replacement_cancelled": lambda *_: "Okay, no changes.",
            "item_modification_cancelled": lambda *_: "Okay, leaving it as is.",
            "action_cancelled": lambda *_: "Okay, cancelled.",
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
            "show_modifier_price": lambda c, m, p: (
                f"{p.get('modifier_name', 'That option')} on {p.get('item_name', 'that item')} costs {p.get('price', '$0.00')}."
            ),
            "price_not_found": lambda *_: "I couldn’t find that item. Say the item name again.",
            "modifier_requires_item_context": lambda *_: (
                "That goes with a specific item. Which item would you like it on?"
            ),
            "readonly_interrupt_with_resume": self._readonly_interrupt_with_resume,
            "show_cart": lambda c, m, p: render_cart_summary(p),
            "show_total": lambda c, m, p: f"Your total is {p.get('total', '$0.00')}.",
            "cart_empty": lambda *_: "Your cart is empty.",
            "confirm_clear_cart": lambda *_: confirm_clear_cart_response(),
            "cart_cleared": lambda *_: cart_cleared_response(),
            "clear_cart_cancelled": lambda *_: clear_cart_cancelled_response(),
            "confirm_order_summary": self._confirm_order_summary,
            "waiting_for_payment": lambda *_: (
                "Waiting for payment. I'll confirm it as soon as it goes through."
            ),

            "no_active_order_to_cancel": lambda *_: "There’s no active order to cancel.",
            "no_active_payment": lambda *_: "There’s no payment in progress.",
            "payment_not_started": lambda *_: "Payment has not started. Say checkout when ready.",
            "order_cancelled": lambda *_: "Okay, checkout cancelled. Your cart is still here.",
            "confirm_side_choice_guess": lambda c, m, p: f"Did you mean {p.get('choice_name', 'that side')}? Yes or no.",
            "confirm_modifier_choice_guess": lambda c, m, p: f"Did you mean {p.get('choice_name', 'that modifier')}? Yes or no.",
            "confirm_size_choice_guess": lambda c, m, p: f"Did you mean {p.get('choice_name', 'that size')}? Yes or no.",
            "confirm_side_size_choice_guess": lambda c, m, p: f"Did you mean {p.get('choice_name', 'that size')} for {p.get('side_item_name', 'that side')}? Yes or no.",

            "ask_for_caller_device_type": lambda *_: (
                "Welcome to Compass. Are you calling from a landline or a mobile phone?"
            ),
            "repeat_caller_device_type": lambda *_: (
                "Sorry, are you on a landline or mobile phone?"
            ),
            "confirm_landline_pickup_only": lambda *_: (
                "I'll connect you with a team member to place your order. Would you like to proceed?"
            ),
            "repeat_landline_pickup_only": lambda *_: (
                "Would you like to connect with a team member? Yes or no."
            ),
            "transferring_to_human_agent": lambda c, m, p: (
                "Connecting you now. One moment."
            ),
            "landline_pickup_declined": lambda *_: (
                "No problem. Call us anytime. Goodbye."
            ),

            "ask_for_order_type": lambda *_: "Is this for pickup or delivery?",
            "repeat_order_type": lambda *_: "Is this for pickup or delivery?",
            "order_type_captured_pickup": lambda *_: "Pickup. What would you like to order?",
            "order_type_captured_delivery": lambda *_: "Delivery. What would you like to order?",

            # Ordering intents during pre-order setup
            "ordering_blocked_need_order_type": lambda *_: (
                "I'll get to your order right away. First, is this for pickup or delivery?"
            ),
            "ordering_blocked_need_delivery_info": lambda c, m, p: (
                self._ordering_blocked_delivery_info(p)
            ),
            "ordering_blocked_need_delivery_address": lambda c, m, p: (
                self._ordering_blocked_delivery_address(p)
            ),

            "checkout_blocked_finish_current_item": lambda *_: "Please finish this item first, or say cancel.",

            "payment_link_send_failed": lambda *_: (
                "I couldn't send the payment link. Your order is saved. Please try again shortly."
            ),
            "checkout_link_send_failed": lambda *_: (
                "I couldn't send the checkout link. Your order is saved. Please try again shortly."
            ),

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
                "I've sent you a checkout link. Fill in your address and pay there. I'll confirm once it goes through."
            ),
            "waiting_for_checkout_completion": lambda *_: (
                "Still waiting on checkout. I'll confirm as soon as payment goes through."
            ),

            "confirm_delivery_house_number": lambda c, m, p: (
                f"I heard {p.get('house_number', 'that number')}. Is that correct?"
            ),
            "confirm_delivery_street": lambda c, m, p: (
                f"I heard {p.get('street', 'that street')}. Is that correct?"
            ),

            "confirm_delivery_secondary_address": lambda c, m, p: (
                f"I heard {p.get('secondary_address', 'that address detail')}. Is that correct?"
            ),

            "payment_link_unavailable_now": lambda *_: (
                "I couldn’t send the payment link. Your order is saved. Please try again shortly."
            ),

            "checkout_link_sent": lambda *_: (
                "Checkout link sent. Enter your address and pay there. I’ll confirm once it goes through."
            ),
            "checkout_link_unavailable_fallback_voice": lambda *_: (
                "I’ll take your address here instead. What’s your house number?"
            ),
            "checkout_link_failed_fallback_voice": lambda *_: (
                "I’ll take your address here instead. What’s your house number?"
            ),
            "ask_for_delivery_house_number": lambda *_: "Please say your house number.",
            "repeat_delivery_house_number": lambda *_: "Please say your house number.",
            "ask_for_delivery_street": lambda *_: "Please say your street name or street number.",
            "repeat_delivery_street": lambda *_: "Please say your street name or street number.",
            "ask_for_delivery_secondary_address": lambda *_: (
                "Apartment or suite number? Say none if there isn’t one."
            ),
            "delivery_address_captured_resume_checkout": lambda *_: (
                "Got your address. Sending payment link now."
            ),
            "payment_link_sent": lambda *_: (
                "Payment link sent. I’ll confirm once payment goes through."
            ),
            "payment_draft_saved_retry_later": lambda *_: (
                "Payment didn’t go through. Your order is saved. Try again shortly."
            ),
            "order_completed": self._order_completed,

            # Returned when a customer says "I paid" but Datacap has not yet confirmed
            "payment_not_confirmed_yet": lambda *_: (
                "I haven't received confirmation yet. Please complete payment on the link. I'll confirm as soon as it goes through."
            ),

            # Returned when the Datacap verification call itself throws an error.
            "payment_verification_error": lambda *_: (
                "Having trouble checking payment. Give it a moment, I'm still checking."
            ),
        }

    def _build_multi_item_ack_prefix(self, payload: dict) -> str:
        """Build prefix like: 'Got it, chicken taco and 2 chicken burgers. Starting with the chicken taco.'"""
        summaries = payload.get("heard_items_summary", [])
        current = str(payload.get("current_item_name") or "").strip()

        if not summaries:
            return ""

        items_text = _format_item_summary_list(summaries)
        if current:
            return f"Got it, {items_text}. Starting with the {current}."
        return f"Got it, {items_text}."

    def _build_queue_transition_prefix(self, payload: dict) -> str:
        """Build prefix like: 'Added Chicken Taco. Now for the Chicken Burger.'"""
        prev_name = str(payload.get("prev_item_name") or "").strip()
        prev_qty = int(payload.get("prev_quantity", 1) or 1)
        next_name = str(payload.get("next_item_name") or "").strip()

        if not prev_name:
            return ""

        added = _added_text(prev_name, prev_qty)
        if next_name:
            return f"{added}. Now for the {next_name}."
        return f"{added}. Next item."

    @staticmethod
    def _build_prefilled_confirmation(item_name: str, prefilled_summary: str) -> str:
        """Build confirmation like: 'Chicken Taco with Coke, extra Cheese — got it.'"""
        if not prefilled_summary:
            return ""
        item_name = item_name.strip() if item_name else ""
        if item_name:
            return f"{item_name} {prefilled_summary} — got it."
        return f"Got it, {prefilled_summary}."

    # ── Pre-order redirect helpers ──

    @staticmethod
    def _ordering_blocked_delivery_info(payload: dict) -> str:
        step = payload.get("step", "delivery_area")
        if step == "delivery_postal_code":
            return "I'll get your order started right after. What's your ZIP code?"
        if step == "delivery_eligibility_confirmation":
            return "Almost there. Please confirm your delivery area first."
        return "I'll get your order started right after. What's your delivery area?"

    @staticmethod
    def _ordering_blocked_delivery_address(payload: dict) -> str:
        step = payload.get("step", "delivery_house_number")
        if step == "delivery_street":
            return "I'll get your order started right after. What's your street name?"
        if step == "delivery_secondary_address":
            return "Almost done with your address. Apartment or suite number?"
        if step in {"delivery_house_number_confirmation", "delivery_street_confirmation",
                     "delivery_secondary_address_confirmation", "delivery_seed_confirmation"}:
            return "Let me finish confirming your address first. Is that correct?"
        return "I'll get your order started right after. What's your house number?"

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

    def _confirm_order_summary(
            self,
            context: ConversationContext,
            _: MenuRepository,
            payload: dict,
    ) -> str:
        summary = render_checkout_review_summary(
            payload=payload,
            order_type=context.order_type,
        )
        return f"{summary} Would you like to checkout?"

    def _order_completed(
        self,
        context: ConversationContext,
        _: MenuRepository,
        payload: dict,
    ) -> str:
        order_number = str(
            payload.get("order_number")
            or context.delivery_address.order_number
            or ""
        ).strip()
        order_sentence = ""
        if order_number:
            order_sentence = f" Your order number is {self._spoken_order_number(order_number)}."

        return (
            f"Payment confirmed.{order_sentence} "
            "Your order has been placed successfully. Will be ready in 25 minutes. Thank you!"
        )

    def _spoken_order_number(self, order_number: str) -> str:
        if order_number.isdigit():
            return " ".join(order_number)
        return order_number

    def _ask_for_side_size(self, _: ConversationContext, __: MenuRepository, payload: dict) -> str:
        side_item_name = payload.get("side_item_name") or "that side"
        available_sizes = [str(x).strip() for x in (payload.get("available_sizes") or []) if str(x).strip()]
        feedback = _build_entity_feedback(payload)

        if not available_sizes:
            prompt = f"What size for {side_item_name}?"
        elif len(available_sizes) == 1:
            prompt = f"Size for {side_item_name}? {available_sizes[0]}."
        elif len(available_sizes) == 2:
            prompt = f"Size for {side_item_name}? {available_sizes[0]} or {available_sizes[1]}."
        else:
            prompt = f"Size for {side_item_name}? {available_sizes[0]}, {available_sizes[1]}, or {available_sizes[2]}."

        return f"{feedback}{prompt}" if feedback else prompt

    def _repeat_side_size_options(self, _: ConversationContext, __: MenuRepository, payload: dict) -> str:
        side_item_name = payload.get("side_item_name") or "that side"
        available_sizes = [str(x).strip() for x in (payload.get("available_sizes") or []) if str(x).strip()]
        feedback = _build_entity_feedback(payload)

        if not available_sizes:
            prompt = f"What size for {side_item_name}?"
        elif len(available_sizes) == 1:
            prompt = f"Choose {available_sizes[0]}."
        elif len(available_sizes) == 2:
            prompt = f"Choose {available_sizes[0]} or {available_sizes[1]}."
        else:
            prompt = f"Choose {available_sizes[0]}, {available_sizes[1]}, or {available_sizes[2]}."

        return f"{feedback}{prompt}" if feedback else prompt

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
