# app/responses/item_responses.py
from __future__ import annotations

import re

from app.menu.repository import MenuRepository
from app.state_machine.models.conversation_context import ConversationContext
from app.utils.top_k_choices import get_top_k_choices


def _clean_group_label(name: str | None, fallback: str) -> str:
    if not name:
        return fallback

    label = name.strip()
    label = re.sub(r"^choose\s+(your\s+|a\s+|an\s+)?", "", label, flags=re.IGNORECASE)
    label = re.sub(r"\s+", " ", label).strip()
    label = re.sub(r"(ss)\b", "s", label, flags=re.IGNORECASE)

    return label or fallback


def _format_options(options: list[str], max_items: int = 3) -> str:
    clean = [str(option).strip() for option in options if str(option).strip()]
    limited = clean[:max_items]

    if not limited:
        return ""
    if len(limited) == 1:
        return limited[0]
    if len(limited) == 2:
        return f"{limited[0]} or {limited[1]}"
    return f"{limited[0]}, {limited[1]}, or {limited[2]}"


def _top_side_choices(
    context: ConversationContext,
    menu_repo: MenuRepository,
    k: int = 3,
) -> list[str]:
    item = menu_repo.store.get_item(context.current_item_id)
    group = item.side_groups[context.current_side_group_index]
    return [choice.name for choice in get_top_k_choices(group.choices, k=k)]


def _top_modifier_choices(
    context: ConversationContext,
    menu_repo: MenuRepository,
    k: int = 3,
) -> list[str]:
    item = menu_repo.store.get_item(context.current_item_id)
    group = item.modifier_groups[context.current_modifier_group_index]
    return [choice.name for choice in get_top_k_choices(group.choices, k=k)]


def ask_for_side(context: ConversationContext, menu_repo: MenuRepository) -> str:
    item = menu_repo.store.get_item(context.current_item_id)
    group = item.side_groups[context.current_side_group_index]

    group_label = _clean_group_label(group.name, "side").lower()
    options = _format_options([choice.name for choice in group.choices])

    if group.is_required:
        if options:
            return f"Which {group_label} would you like with your {item.name}? {options}."
        return f"Which {group_label} would you like with your {item.name}?"

    if options:
        return f"Would you like a {group_label} with your {item.name}? {options}, or say no."
    return f"Would you like a {group_label} with your {item.name}? You can also say no."


def ask_for_modifier(context: ConversationContext, menu_repo: MenuRepository) -> str:
    item = menu_repo.store.get_item(context.current_item_id)
    group = item.modifier_groups[context.current_modifier_group_index]

    group_label = _clean_group_label(group.name, "add-on").lower()
    options = _format_options([choice.name for choice in group.choices])

    if group.is_required:
        if options:
            return f"Which {group_label} would you like for your {item.name}? {options}."
        return f"Which {group_label} would you like for your {item.name}?"

    if options:
        return f"Any extras for your {item.name}? {options}, or say no."
    return f"Any extras for your {item.name}? You can also say no."


def ask_for_size(context: ConversationContext, menu_repo: MenuRepository) -> str:
    item = menu_repo.store.get_item(context.current_item_id)
    variants = item.pricing.variants or []

    labels = [variant.label for variant in variants if variant.label]
    options = _format_options(labels)

    if options:
        return f"What size would you like for {item.name}? {options}."
    return f"What size would you like for {item.name}?"


def ask_item_quantity(payload: dict) -> str:
    item_name = payload.get("item_name")
    if item_name:
        return f"How many {item_name} would you like?"
    return "How many would you like?"


def repeat_item_request(context, menu_repo, payload) -> str:
    return "Which item would you like?"


def item_not_found(context, menu_repo, payload) -> str:
    query = str(payload.get("query") or "").strip()
    item_names = payload.get("suggested_item_names") or []
    category_names = payload.get("suggested_category_names") or []

    if item_names:
        options = _format_options(item_names)
        if query:
            return f"I couldn’t find {query}. You can try {options}."
        return f"I couldn’t find that item. You can try {options}."

    if category_names:
        options = _format_options(category_names)
        if query:
            return f"I couldn’t find {query}. You can try {options}."
        return f"I couldn’t find that item. You can try {options}."

    return "I couldn’t find that item. Please try another one."


def confirm_item_ambiguous(context, menu_repo, payload) -> str:
    item_names = payload.get("candidate_item_names") or []
    category_names = payload.get("candidate_category_names") or []

    if item_names:
        return f"Did you mean {_format_options(item_names)}?"
    if category_names:
        return f"Did you mean {_format_options(category_names)}?"
    return "I found a few matches. Which one did you mean?"


def confirm_item_from_category(context, menu_repo, payload) -> str:
    category_name = payload.get("category_name")
    item_names = payload.get("candidate_item_names") or []

    if item_names and category_name:
        return f"In {category_name}, I found {_format_options(item_names)}. Which one would you like?"
    if item_names:
        return f"I found {_format_options(item_names)}. Which one would you like?"
    return "Which item would you like?"


def repeat_side_options(context, menu_repo, payload) -> str:
    options = _format_options(payload.get("top_choices") or _top_side_choices(context, menu_repo))
    if options:
        return f"Please choose one of these: {options}."
    return "Please choose one of the available sides."


def repeat_modifier_options(context, menu_repo, payload) -> str:
    options = _format_options(payload.get("top_choices") or _top_modifier_choices(context, menu_repo))
    if options:
        return f"Please choose one of these: {options}."
    return "Please choose one of the available options."


def too_many_side_choices(context, menu_repo, payload) -> str:
    options = _format_options(payload.get("top_choices") or _top_side_choices(context, menu_repo))
    if options:
        return f"That is too many sides. Please choose from {options}."
    return "That is too many sides. Please choose fewer options."


def too_many_modifier_choices(context, menu_repo, payload) -> str:
    options = _format_options(payload.get("top_choices") or _top_modifier_choices(context, menu_repo))
    if options:
        return f"That is too many extras. Please choose from {options}."
    return "That is too many extras. Please choose fewer options."


def required_side_cannot_skip(context, menu_repo) -> str:
    options = _format_options(_top_side_choices(context, menu_repo))
    if options:
        return f"This item needs a side. Please choose {options}."
    return "This item needs a side. Please choose one."


def required_modifier_cannot_skip(context, menu_repo) -> str:
    options = _format_options(_top_modifier_choices(context, menu_repo))
    if options:
        return f"This item needs an option. Please choose {options}."
    return "This item needs an option. Please choose one."


def required_size_cannot_skip(context, menu_repo) -> str:
    item = menu_repo.store.get_item(context.current_item_id)
    labels = [variant.label for variant in (item.pricing.variants or []) if variant.label]
    options = _format_options(labels)

    if options:
        return f"Please choose a size for {item.name}: {options}."
    return f"Please choose a size for {item.name}."


def invalid_size_option(context, menu_repo, payload) -> str:
    item = menu_repo.store.get_item(context.current_item_id)
    labels = [variant.label for variant in (item.pricing.variants or []) if variant.label]
    options = _format_options(labels)

    if options:
        return f"That size is not available for {item.name}. Please choose {options}."
    return f"That size is not available for {item.name}. Please choose another size."


def invalid_quantity_option(context, menu_repo, payload) -> str:
    item_name = payload.get("item_name") or context.current_item_name
    if item_name:
        return f"Please give a valid quantity for {item_name}."
    return "Please give a valid quantity."


def item_added_successfully(payload: dict) -> str:
    quantity = int(payload["quantity"])
    item_name = str(payload.get("item_name") or "").strip()

    if quantity > 1 and item_name:
        return f"Added {quantity} {item_name}. Would you like anything else?"
    if quantity > 1:
        return f"Added {quantity}. Would you like anything else?"
    if item_name:
        return f"{item_name} added. Would you like anything else?"
    return "Added. Would you like anything else?"


def confirm_cancel_current_item(context, menu_repo, payload) -> str:
    item_name = payload.get("item_name") or context.current_item_name or "this item"
    return f"Do you want to cancel {item_name}?. Please say yes or no."


def confirm_cancel_current_item_for_new_request(context, menu_repo, payload) -> str:
    item_name = payload.get("item_name") or context.current_item_name or "this item"
    return f"You are still adding {item_name}. Do you want to cancel it, and do something else?. Please say yes or no."


def continue_current_item_after_cancel_denied(context, menu_repo, payload) -> str:
    options = payload.get("available_choices") or list(context.available_choices_values)
    formatted = _format_options(options)

    if formatted:
        return f"Okay, let’s continue. Please choose {formatted}."
    return "Okay, let’s continue."


def item_cancelled_successfully(context, menu_repo, payload) -> str:
    return "Okay, cancelled. What would you like next?"


def repeat_size_options(context, menu_repo, payload) -> str:
    item = menu_repo.store.get_item(context.current_item_id)
    labels = [variant.label for variant in (item.pricing.variants or []) if variant.label]
    options = _format_options(labels)

    if options:
        return f"Available sizes for {item.name} are {options}."
    return f"What size would you like for {item.name}?"


def item_context_missing(context, menu_repo, payload) -> str:
    return "Something went wrong with that item. Let’s start again."


def size_not_applicable(context, menu_repo, payload) -> str:
    item_name = payload.get("item_name") or context.current_item_name or "That item"
    return f"{item_name} does not need a size. Let’s continue."


def list_side_options(context, menu_repo, payload) -> str:
    options = _format_options(payload.get("top_choices") or _top_side_choices(context, menu_repo))
    if options:
        return f"Your side options are {options}."
    return "Here are the available side options."


def clarify_side_choice(context, menu_repo, payload) -> str:
    options = _format_options(payload.get("top_choices") or _top_side_choices(context, menu_repo))
    if options:
        return f"Did you mean {options}?"
    return "Which side did you want?"


def list_modifier_options(context, menu_repo, payload) -> str:
    options = _format_options(payload.get("top_choices") or _top_modifier_choices(context, menu_repo))
    if options:
        return f"Your options are {options}."
    return "Here are the available options."


def clarify_modifier_choice(context, menu_repo, payload) -> str:
    options = _format_options(payload.get("top_choices") or _top_modifier_choices(context, menu_repo))
    if options:
        return f"Did you mean {options}?"
    return "Which option did you want?"