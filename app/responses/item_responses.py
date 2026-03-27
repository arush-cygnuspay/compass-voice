# app/responses/item_responses.py
from __future__ import annotations

import re

from app.menu.repository import MenuRepository
from app.state_machine.conversation_context import ConversationContext
from app.utils.top_k_choices import get_top_k_choices


def _clean_group_label(name: str | None, fallback: str) -> str:
    if not name:
        return fallback

    label = name.strip()
    label = re.sub(r"^choose\s+(your\s+|a\s+|an\s+)?", "", label, flags=re.IGNORECASE)
    label = re.sub(r"\s+", " ", label).strip()
    label = re.sub(r"(ss)\b", "s", label, flags=re.IGNORECASE)

    return label or fallback


def _format_options(options: list[str]) -> str:
    clean = [str(option).strip() for option in options if str(option).strip()]
    if not clean:
        return ""
    if len(clean) == 1:
        return clean[0]
    if len(clean) == 2:
        return f"{clean[0]} or {clean[1]}"
    return f"{clean[0]}, {clean[1]}, or {clean[2]}"


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
    top_choices = [choice.name for choice in group.choices[:3]]
    options = _format_options(top_choices)
    min_selector = max(int(group.min_selector or 1), 1)
    max_selector = int(group.max_selector or 1)

    if group.is_required:
        if min_selector == 1 and max_selector == 1:
            return f"Which {group_label} with your {item.name}? {options}."
        return f"Choose up to {max_selector} {group_label}s for {item.name}. {options}."

    return f"Any {group_label} with your {item.name}? {options}, or say no."


def ask_for_modifier(context: ConversationContext, menu_repo: MenuRepository) -> str:
    item = menu_repo.store.get_item(context.current_item_id)
    group = item.modifier_groups[context.current_modifier_group_index]

    group_label = _clean_group_label(group.name, "add-on").lower()
    top_choices = [choice.name for choice in group.choices[:4]]
    options = _format_options(top_choices)
    min_selector = max(int(group.min_selector or 1), 1)
    max_selector = int(group.max_selector or 1)

    if group.is_required:
        if min_selector == 1 and max_selector == 1:
            return f"Which {group_label} for {item.name}? {options}."
        return f"Choose up to {max_selector} {group_label}s for {item.name}. {options}."

    return f"Any {group_label} for {item.name}? {options}, or say no."


def ask_for_size(context: ConversationContext, menu_repo: MenuRepository) -> str:
    item = menu_repo.store.get_item(context.current_item_id)
    variants = item.pricing.variants or []

    if not variants:
        return f"What size {item.name}?"

    labels = [variant.label for variant in variants if variant.label]
    options = _format_options(labels[:3])

    return f"What size {item.name}? {options}."


def ask_item_quantity(payload: dict) -> str:
    item_name = payload.get("item_name")
    if item_name:
        return f"How many {item_name}?"
    return "How many?"


def repeat_item_request(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict,
) -> str:
    return "Which item would you like?"


def item_not_found(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict,
) -> str:
    query = (payload.get("query") or "").strip()
    item_names = [str(x).strip() for x in (payload.get("suggested_item_names") or []) if str(x).strip()]
    category_names = [str(x).strip() for x in (payload.get("suggested_category_names") or []) if str(x).strip()]

    prefix = f"I couldn't find {query}." if query else "I couldn't find that item."

    if item_names and category_names:
        return f"{prefix} We have {_format_options(item_names[:3])}, or {_format_options(category_names[:4])}. Which one?"
    if item_names:
        return f"{prefix} We have {_format_options(item_names[:3])}. Which one?"
    if category_names:
        return f"{prefix} Try {_format_options(category_names[:4])}. Which category?"
    return f"{prefix} Try another item."


def confirm_item_ambiguous(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict,
) -> str:
    item_names = payload.get("candidate_item_names") or []
    category_names = payload.get("candidate_category_names") or []

    if item_names:
        return f"I found {_format_options(item_names[:3])}. Which one?"
    if category_names:
        return f"I found {_format_options(category_names[:3])}. Which category?"
    return "I found a few matches. Which one?"


def confirm_item_from_category(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict,
) -> str:
    category_name = payload.get("category_name") or "that category"
    item_names = payload.get("candidate_item_names") or []

    if not item_names:
        return f"I found {category_name}. Which item?"
    if len(item_names) == 1:
        return f"I found {item_names[0]} in {category_name}. Want that?"
    return f"In {category_name}, I found {_format_options(item_names[:4])}. Which one?"


def repeat_side_options(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict,
) -> str:
    item = menu_repo.store.get_item(context.current_item_id)
    group = item.side_groups[context.current_side_group_index]

    group_label = _clean_group_label(payload.get("group_name") or group.name, "side").lower()
    top_choices = payload.get("top_choices") or _top_side_choices(context, menu_repo, k=3)
    options = _format_options(top_choices)
    reason = payload.get("repeat_reason", "invalid")

    if reason == "options":
        return f"{group_label.title()} options: {options}."
    return f"Not a valid {group_label}. Choose {options}."


def repeat_modifier_options(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict,
) -> str:
    item = menu_repo.store.get_item(context.current_item_id)
    group = item.modifier_groups[context.current_modifier_group_index]

    group_label = _clean_group_label(payload.get("group_name") or group.name, "add-on").lower()
    top_choices = payload.get("top_choices") or _top_modifier_choices(context, menu_repo, k=3)
    options = _format_options(top_choices)
    reason = payload.get("repeat_reason", "invalid")

    if reason == "options":
        return f"{group_label.title()} options: {options}."
    return f"Not a valid {group_label}. Choose {options}."


def too_many_side_choices(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict,
) -> str:
    item = menu_repo.store.get_item(context.current_item_id)
    group = item.side_groups[context.current_side_group_index]
    group_label = _clean_group_label(payload.get("group_name") or group.name, "side").lower()
    options = _format_options(payload.get("top_choices") or _top_side_choices(context, menu_repo, k=3))

    return f"Too many {group_label}s for {item.name}. Choose {options}."


def too_many_modifier_choices(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict,
) -> str:
    item = menu_repo.store.get_item(context.current_item_id)
    group = item.modifier_groups[context.current_modifier_group_index]
    group_label = _clean_group_label(payload.get("group_name") or group.name, "add-on").lower()
    options = _format_options(payload.get("top_choices") or _top_modifier_choices(context, menu_repo, k=3))

    return f"Too many {group_label}s for {item.name}. Choose {options}."


def required_side_cannot_skip(
    context: ConversationContext,
    menu_repo: MenuRepository,
) -> str:
    item = menu_repo.store.get_item(context.current_item_id)
    group = item.side_groups[context.current_side_group_index]
    group_label = _clean_group_label(group.name, "side").lower()
    options = _format_options(_top_side_choices(context, menu_repo, k=3))

    return f"{item.name} needs a {group_label}. Choose {options}."


def required_modifier_cannot_skip(
    context: ConversationContext,
    menu_repo: MenuRepository,
) -> str:
    item = menu_repo.store.get_item(context.current_item_id)
    group = item.modifier_groups[context.current_modifier_group_index]
    group_label = _clean_group_label(group.name, "add-on").lower()
    options = _format_options(_top_modifier_choices(context, menu_repo, k=3))

    return f"{item.name} needs a {group_label}. Choose {options}."


def required_size_cannot_skip(
    context: ConversationContext,
    menu_repo: MenuRepository,
) -> str:
    item = menu_repo.store.get_item(context.current_item_id)
    labels = [variant.label for variant in (item.pricing.variants or []) if variant.label]
    options = _format_options(labels[:3])

    return f"Choose a size for {item.name}. {options}."


def invalid_size_option(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict,
) -> str:
    item = menu_repo.store.get_item(context.current_item_id)
    labels = [variant.label for variant in (item.pricing.variants or []) if variant.label]
    options = _format_options(labels[:3])

    return f"That size isn't available for {item.name}. Choose {options}."


def invalid_quantity_option(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict,
) -> str:
    item_name = payload.get("item_name") or context.current_item_name
    if item_name:
        return f"Give a valid quantity for {item_name}."
    return "Give a valid quantity."


def item_added_successfully(payload: dict) -> str:
    item_name = str(payload["item_name"]).strip()
    quantity = int(payload["quantity"])

    if quantity > 1:
        return f"Added {quantity} {item_name}. Anything else or checkout?"
    return f"Added {item_name}. Anything else or checkout?"


def confirm_cancel_current_item(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict,
) -> str:
    item_name = payload.get("item_name") or context.current_item_name or "this item"
    return f"You're still adding {item_name}. Cancel it? Yes or no."


def confirm_cancel_current_item_for_new_request(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict,
) -> str:
    item_name = payload.get("item_name") or context.current_item_name or "this item"
    return f"You're still adding {item_name}. Cancel and do something else? Yes or no."


def continue_current_item_after_cancel_denied(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict,
) -> str:
    field_name = payload.get("field_name") or context.current_prompt_field or "option"
    choices = payload.get("available_choices") or list(context.available_choices_values)
    options = _format_options(choices[:3]) if choices else ""

    if options:
        return f"Okay. Choose {field_name}: {options}."
    return f"Okay. Choose {field_name}."


def item_cancelled_successfully(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict,
) -> str:
    return "Okay, cancelled. What next?"


def repeat_size_options(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict,
) -> str:
    item = menu_repo.store.get_item(context.current_item_id)
    labels = [variant.label for variant in (item.pricing.variants or []) if variant.label]
    options = _format_options(labels[:3])

    if options:
        return f"Sizes for {item.name}: {options}."
    return f"What size for {item.name}?"


def item_context_missing(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict,
) -> str:
    return "Something went wrong. Let's start again."


def size_not_applicable(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict,
) -> str:
    item_name = payload.get("item_name") or context.current_item_name or "That item"
    return f"{item_name} has no size options. Let's continue."


def list_side_options(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict,
) -> str:
    item = menu_repo.store.get_item(context.current_item_id)
    group = item.side_groups[context.current_side_group_index]

    group_label = _clean_group_label(payload.get("group_name") or group.name, "side").lower()
    top_choices = payload.get("top_choices") or _top_side_choices(context, menu_repo, k=3)
    options = _format_options(top_choices)

    if options:
        return f"{group_label.title()} options for {item.name}: {options}."
    return f"There are {group_label} options for {item.name}."


def clarify_side_choice(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict,
) -> str:
    item = menu_repo.store.get_item(context.current_item_id)
    group = item.side_groups[context.current_side_group_index]

    group_label = _clean_group_label(payload.get("group_name") or group.name, "side").lower()
    top_choices = payload.get("top_choices") or _top_side_choices(context, menu_repo, k=3)
    options = _format_options(top_choices)

    if options:
        return f"Did you mean {options} for your {group_label}?"
    return f"Repeat which {group_label} you want with {item.name}."


def list_modifier_options(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict,
) -> str:
    item = menu_repo.store.get_item(context.current_item_id)
    group = item.modifier_groups[context.current_modifier_group_index]

    group_label = _clean_group_label(payload.get("group_name") or group.name, "add-on").lower()
    top_choices = payload.get("top_choices") or _top_modifier_choices(context, menu_repo, k=3)
    options = _format_options(top_choices)

    if options:
        return f"{group_label.title()} options for {item.name}: {options}."
    return f"There are {group_label} options for {item.name}."


def clarify_modifier_choice(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict,
) -> str:
    item = menu_repo.store.get_item(context.current_item_id)
    group = item.modifier_groups[context.current_modifier_group_index]

    group_label = _clean_group_label(payload.get("group_name") or group.name, "add-on").lower()
    top_choices = payload.get("top_choices") or _top_modifier_choices(context, menu_repo, k=3)
    options = _format_options(top_choices)

    if options:
        return f"Did you mean {options} for your {group_label}?"
    return f"Repeat which {group_label} you want for {item.name}."