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

    if not label:
        return fallback

    return label


def _format_options(options: list[str]) -> str:
    clean = [str(option).strip() for option in options if str(option).strip()]
    if not clean:
        return ""
    if len(clean) == 1:
        return clean[0]
    if len(clean) == 2:
        return f"{clean[0]} or {clean[1]}"
    return f"{clean[0]}, {clean[1]}, or {clean[2]}"


def _top_side_choices(context: ConversationContext, menu_repo: MenuRepository, k: int = 3) -> list[str]:
    item = menu_repo.store.get_item(context.current_item_id)
    group = item.side_groups[context.current_side_group_index]
    return [choice.name for choice in get_top_k_choices(group.choices, k=k)]


def _top_modifier_choices(context: ConversationContext, menu_repo: MenuRepository, k: int = 3) -> list[str]:
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
            return f"Which {group_label} would you like with your {item.name}? You can choose {options}."
        return (
            f"Which {group_label}s would you like with your {item.name}? "
            f"You can choose up to {max_selector}. Popular options include {options}."
        )

    return f"Would you like any {group_label} with your {item.name}? You can choose {options}, or say no."


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
            return f"Which {group_label} would you like for your {item.name}? You can choose {options}."
        return (
            f"Which {group_label}s would you like for your {item.name}? "
            f"You can choose up to {max_selector}. Popular options include {options}."
        )

    return f"Would you like any {group_label} for your {item.name}? You can choose {options}, or say no."


def ask_for_size(context: ConversationContext, menu_repo: MenuRepository) -> str:
    item = menu_repo.store.get_item(context.current_item_id)
    variants = item.pricing.variants or []

    if not variants:
        return f"Which size {item.name} would you like?"

    labels = [variant.label for variant in variants if variant.label]
    options = _format_options(labels[:3])

    return f"Which size {item.name} would you like? You can choose {options}."


def ask_item_quantity(payload: dict) -> str:
    item_name = payload.get("item_name")
    if item_name:
        return f"How many {item_name} would you like?"
    return "How many would you like?"


def repeat_item_request(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict,
) -> str:
    return "I did not catch which item you want. Please say the menu item again."


# app/responses/item_responses.py

def item_not_found(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict,
) -> str:
    query = (payload.get("query") or "").strip()
    item_names = [str(x).strip() for x in (payload.get("suggested_item_names") or []) if str(x).strip()]
    category_names = [str(x).strip() for x in (payload.get("suggested_category_names") or []) if str(x).strip()]

    prefix = (
        f"Sorry, I could not find {query} on the menu."
        if query
        else "Sorry, I could not find that item on the menu."
    )

    if item_names and category_names:
        return (
            f"{prefix} We do have {_format_options(item_names[:3])}. "
            f"You can also choose from {_format_options(category_names[:4])}. "
            f"Which would you like?"
        )

    if item_names:
        return (
            f"{prefix} We do have {_format_options(item_names[:3])}. "
            f"Which one would you like?"
        )

    if category_names:
        return (
            f"{prefix} You can choose from {_format_options(category_names[:4])}. "
            f"Which category would you like?"
        )

    return f"{prefix} Please try another item."


def confirm_item_ambiguous(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict,
) -> str:
    item_names = payload.get("candidate_item_names") or []
    category_names = payload.get("candidate_category_names") or []

    if item_names:
        return f"I found multiple matching items: {_format_options(item_names[:3])}. Which one would you like?"

    if category_names:
        return f"I found multiple matching categories: {_format_options(category_names[:3])}. Which one did you mean?"

    return "I found multiple matching items. Which one would you like?"


def confirm_item_from_category(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict,
) -> str:
    category_name = payload.get("category_name") or "that category"
    item_names = payload.get("candidate_item_names") or []

    if not item_names:
        return f"I found the category {category_name}. Which item would you like?"

    if len(item_names) == 1:
        return f"I found {item_names[0]} in {category_name}. Would you like that?"

    return f"I found several items in {category_name}: {_format_options(item_names[:4])}. Which one would you like?"


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
        return f"You can choose {options}. Which {group_label} would you like with your {item.name}?"

    return (
        f"That is not one of the available {group_label} options. "
        f"You can choose {options}. Which {group_label} would you like with your {item.name}?"
    )


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
        return f"You can choose {options}. Which {group_label} would you like for your {item.name}?"

    return (
        f"That is not one of the available {group_label} options. "
        f"You can choose {options}. Which {group_label} would you like for your {item.name}?"
    )


def too_many_side_choices(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict,
) -> str:
    item = menu_repo.store.get_item(context.current_item_id)
    group = item.side_groups[context.current_side_group_index]
    group_label = _clean_group_label(payload.get("group_name") or group.name, "side").lower()
    options = _format_options(payload.get("top_choices") or _top_side_choices(context, menu_repo, k=3))

    return (
        f"You selected too many {group_label} options for your {item.name}. "
        f"You can choose {options}. Please try again."
    )


def too_many_modifier_choices(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict,
) -> str:
    item = menu_repo.store.get_item(context.current_item_id)
    group = item.modifier_groups[context.current_modifier_group_index]
    group_label = _clean_group_label(payload.get("group_name") or group.name, "add-on").lower()
    options = _format_options(payload.get("top_choices") or _top_modifier_choices(context, menu_repo, k=3))

    return (
        f"You selected too many {group_label} options for your {item.name}. "
        f"You can choose {options}. Please try again."
    )


def required_side_cannot_skip(
    context: ConversationContext,
    menu_repo: MenuRepository,
) -> str:
    item = menu_repo.store.get_item(context.current_item_id)
    group = item.side_groups[context.current_side_group_index]
    group_label = _clean_group_label(group.name, "side").lower()
    options = _format_options(_top_side_choices(context, menu_repo, k=3))

    return f"This item requires a {group_label}. You can choose {options}. Which {group_label} would you like with your {item.name}?"


def required_modifier_cannot_skip(
    context: ConversationContext,
    menu_repo: MenuRepository,
) -> str:
    item = menu_repo.store.get_item(context.current_item_id)
    group = item.modifier_groups[context.current_modifier_group_index]
    group_label = _clean_group_label(group.name, "add-on").lower()
    options = _format_options(_top_modifier_choices(context, menu_repo, k=3))

    return f"This item requires a {group_label}. You can choose {options}. Which {group_label} would you like for your {item.name}?"


def required_size_cannot_skip(
    context: ConversationContext,
    menu_repo: MenuRepository,
) -> str:
    item = menu_repo.store.get_item(context.current_item_id)
    labels = [variant.label for variant in (item.pricing.variants or []) if variant.label]
    options = _format_options(labels[:3])

    return f"You need to choose a valid size for {item.name}. You can choose {options}."


def invalid_size_option(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict,
) -> str:
    item = menu_repo.store.get_item(context.current_item_id)
    labels = [variant.label for variant in (item.pricing.variants or []) if variant.label]
    options = _format_options(labels[:3])

    return f"That is not an available size for {item.name}. You can choose {options}."


def invalid_quantity_option(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict,
) -> str:
    item_name = payload.get("item_name") or context.current_item_name
    if item_name:
        return f"Please tell me a valid quantity for {item_name}."
    return "Please tell me a valid quantity."


def item_added_successfully(payload: dict) -> str:
    item_name = str(payload["item_name"]).strip()
    quantity = int(payload["quantity"])

    if quantity > 1:
        return (
            f"{quantity} x {item_name} have been added to your cart. "
            f"Would you like to add anything else or check out?"
        )

    return (
        f"Your {item_name} has been added to the cart. "
        f"Would you like to add anything else or check out?"
    )


def confirm_cancel_current_item(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict,
) -> str:
    item_name = payload.get("item_name") or context.current_item_name or "this item"
    return f"You are still adding {item_name}. Do you want to cancel it? Please say yes or no."


def confirm_cancel_current_item_for_new_request(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict,
) -> str:
    item_name = payload.get("item_name") or context.current_item_name or "this item"
    return f"I am still adding {item_name}. Do you want to cancel this item and do something else? Please say yes or no."


def continue_current_item_after_cancel_denied(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict,
) -> str:
    field_name = payload.get("field_name") or context.current_prompt_field or "option"
    choices = payload.get("available_choices") or list(context.available_choices_values)
    options = _format_options(choices[:3]) if choices else ""

    if options:
        return f"Okay, let’s continue. Please choose a {field_name}. You can choose {options}."

    return f"Okay, let’s continue. Please choose a {field_name}."


def item_cancelled_successfully(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict,
) -> str:
    return "Okay, I cancelled that item. What would you like to do next?"


def repeat_size_options(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict,
) -> str:
    item = menu_repo.store.get_item(context.current_item_id)
    labels = [variant.label for variant in (item.pricing.variants or []) if variant.label]
    options = _format_options(labels[:3])

    if options:
        return f"You can choose {options} for your {item.name}. Which size would you like?"
    return f"Which size would you like for your {item.name}?"


def item_context_missing(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict,
) -> str:
    return "Something went wrong with the item you were configuring. Let’s start again."


def size_not_applicable(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict,
) -> str:
    item_name = payload.get("item_name") or context.current_item_name or "that item"
    return f"{item_name} does not have size options. Let’s continue."


def list_side_options(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict,
) -> str:
    item = menu_repo.store.get_item(context.current_item_id)
    group = item.side_groups[context.current_side_group_index]

    group_label = _clean_group_label(
        payload.get("group_name") or group.name,
        "side",
    ).lower()

    top_choices = payload.get("top_choices") or _top_side_choices(context, menu_repo, k=3)
    options = _format_options(top_choices)

    if options:
        return f"For your {item.name}, you can choose {options} as your {group_label}."

    return f"You can choose any available {group_label} for your {item.name}."


def clarify_side_choice(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict,
) -> str:
    item = menu_repo.store.get_item(context.current_item_id)
    group = item.side_groups[context.current_side_group_index]

    group_label = _clean_group_label(
        payload.get("group_name") or group.name,
        "side",
    ).lower()

    top_choices = payload.get("top_choices") or _top_side_choices(context, menu_repo, k=3)
    options = _format_options(top_choices)

    if options:
        return f"Did you mean {options} for your {group_label}, or something else with your {item.name}?"

    return f"Can you repeat which {group_label} you want with your {item.name}?"


def list_modifier_options(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict,
) -> str:
    item = menu_repo.store.get_item(context.current_item_id)
    group = item.modifier_groups[context.current_modifier_group_index]

    group_label = _clean_group_label(
        payload.get("group_name") or group.name,
        "add-on",
    ).lower()

    top_choices = payload.get("top_choices") or _top_modifier_choices(context, menu_repo, k=3)
    options = _format_options(top_choices)

    if options:
        return f"For your {item.name}, you can choose {options} as your {group_label}."

    return f"You can choose any available {group_label} for your {item.name}."


def clarify_modifier_choice(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict,
) -> str:
    item = menu_repo.store.get_item(context.current_item_id)
    group = item.modifier_groups[context.current_modifier_group_index]

    group_label = _clean_group_label(
        payload.get("group_name") or group.name,
        "add-on",
    ).lower()

    top_choices = payload.get("top_choices") or _top_modifier_choices(context, menu_repo, k=3)
    options = _format_options(top_choices)

    if options:
        return f"Did you mean {options} for your {group_label}, or something else for your {item.name}?"

    return f"Can you repeat which {group_label} you want for your {item.name}?"
