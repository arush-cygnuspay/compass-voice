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


def _pluralize(word: str, count: int) -> str:
    return word if count == 1 else f"{word}s"


def _format_selected_names(selected_names: list[str]) -> str:
    clean = [str(name).strip() for name in selected_names if str(name).strip()]
    if not clean:
        return ""
    if len(clean) == 1:
        return clean[0]
    if len(clean) == 2:
        return f"{clean[0]} and {clean[1]}"
    return f"{', '.join(clean[:-1])}, and {clean[-1]}"


def _payload_value(payload: dict, key: str, default):
    if key in payload:
        return payload[key]
    return default


def _group_payload(
    *,
    payload: dict,
    group_name: str,
    option_names: list[str],
    selected_names: list[str],
    min_selector: int,
    max_selector: int,
) -> dict:
    selected_count = len(selected_names)
    effective_max = max_selector if max_selector > 0 else len(option_names)
    if option_names and effective_max > len(option_names):
        effective_max = len(option_names)

    remaining_to_min = max(min_selector - selected_count, 0)
    remaining_to_max = max(effective_max - selected_count, 0) if effective_max > 0 else 0

    return {
        "group_name": _payload_value(payload, "group_name", group_name),
        "top_choices": _payload_value(payload, "top_choices", option_names[:4]),
        "all_choices": _payload_value(payload, "all_choices", option_names),
        "selected_names": _payload_value(payload, "selected_names", selected_names),
        "selected_count": int(_payload_value(payload, "selected_count", selected_count) or 0),
        "min_selector": int(_payload_value(payload, "min_selector", min_selector) or 0),
        "max_selector": int(_payload_value(payload, "max_selector", effective_max) or 0),
        "remaining_to_min": int(_payload_value(payload, "remaining_to_min", remaining_to_min) or 0),
        "remaining_to_max": int(_payload_value(payload, "remaining_to_max", remaining_to_max) or 0),
    }


def _current_side_payload(context: ConversationContext, menu_repo: MenuRepository, payload: dict | None = None) -> dict:
    payload = payload or {}
    item = menu_repo.store.get_item(context.current_item_id)
    group = item.side_groups[context.current_side_group_index]
    selected_ids = list(context.selected_side_groups.get(group.group_id, []))
    selected_names = [
        group.choices_by_item_id[item_id].name
        for item_id in selected_ids
        if item_id in group.choices_by_item_id
    ]

    return _group_payload(
        payload=payload,
        group_name=group.name,
        option_names=[choice.name for choice in group.choices],
        selected_names=selected_names,
        min_selector=int(group.min_selector or 0),
        max_selector=int(group.max_selector or 0),
    )


def _current_modifier_payload(context: ConversationContext, menu_repo: MenuRepository, payload: dict | None = None) -> dict:
    payload = payload or {}
    item = menu_repo.store.get_item(context.current_item_id)
    group = item.modifier_groups[context.current_modifier_group_index]
    selected_names = []
    for selection in context.selected_modifier_groups.get(group.group_id, []):
        if selection.action == "remove":
            selected_names.append(f"no {selection.name}")
        elif selection.instruction == "extra":
            selected_names.append(f"extra {selection.name}")
        elif selection.instruction == "less":
            selected_names.append(f"less {selection.name}")
        else:
            selected_names.append(selection.name)

    return _group_payload(
        payload=payload,
        group_name=group.name,
        option_names=[choice.name for choice in group.choices],
        selected_names=selected_names,
        min_selector=int(group.min_selector or 0),
        max_selector=int(group.max_selector or 0),
    )


def _initial_multi_select_prompt(*, item_name: str, group_label: str, options: str, min_selector: int, max_selector: int, optional: bool) -> str:
    if optional:
        if max_selector == 1:
            return f"Would you like a {group_label} with your {item_name}? {options}, or say no."
        return (
            f"Would you like any {group_label} for your {item_name}? "
            f"You can choose up to {max_selector}. You can say them all at once. "
            f"{options}, or say no."
        )

    if min_selector == max_selector:
        if min_selector == 1:
            return f"Which {group_label} would you like with your {item_name}? {options}."
        return (
            f"Please choose {min_selector} options for your {item_name}. "
            f"You can say them all at once. {options}."
        )

    if max_selector > 0:
        if min_selector == 1:
            return (
                f"Please choose 1 option for your {item_name}. "
                f"You can choose up to {max_selector} and say them all at once. {options}."
            )
        return (
            f"Please choose at least {min_selector} options for your {item_name}. "
            f"You can choose up to {max_selector} and say them all at once. {options}."
        )

    return (
        f"Please choose at least {min_selector} options for your {item_name}. "
        f"You can say them all at once. {options}."
    )


def _progress_prompt(payload: dict, *, item_word: str, invalid_lead: str) -> str:
    top_choices = _payload_value(payload, "top_choices", [])
    all_choices = _payload_value(payload, "all_choices", [])
    option_values = top_choices if top_choices else all_choices
    options = _format_options(option_values)
    selected_names = payload.get("selected_names") or []
    selected_part = ""
    if selected_names:
        selected_part = f"You already picked {_format_selected_names(selected_names)}. "

    reason = payload.get("repeat_reason")
    if reason == "need_more":
        remaining = max(int(payload.get("remaining_to_min", 0) or 0), 0)
        need_more = f"Please choose {remaining} more {_pluralize(item_word, remaining)}."
        if remaining == 1:
            need_more = f"Please choose 1 more {item_word}."
        return (
            f"{selected_part}{need_more} Remaining options are {options}."
            if options
            else f"{selected_part}{need_more}"
        )

    if reason == "optional_more":
        remaining = max(int(payload.get("remaining_to_max", 0) or 0), 0)
        if remaining > 0:
            add_more = f"You can add up to {remaining} more, or say done."
            if remaining == 1:
                add_more = "You can add 1 more, or say done."
            return (
                f"{selected_part}{add_more} Remaining options are {options}."
                if options
                else f"{selected_part}{add_more}"
            )
        return f"{selected_part}Say done when you're ready."

    remaining = max(int(payload.get("remaining_to_min", 0) or 0), 0)
    if remaining == 0:
        if options:
            remaining_to_max = max(int(payload.get("remaining_to_max", 0) or 0), 0)
            if remaining_to_max > 0:
                add_more = f"You can add up to {remaining_to_max} more, or say done."
                if remaining_to_max == 1:
                    add_more = "You can add 1 more, or say done."
                return f"{selected_part}{add_more} Remaining options are {options}."
        return f"{selected_part}Say done when you're ready."

    invalid = invalid_lead
    if remaining > 0:
        invalid = f"{invalid_lead} Please choose {remaining} more {_pluralize(item_word, remaining)}."
        if remaining == 1:
            invalid = f"{invalid_lead} Please choose 1 more {item_word}."
    return (
        f"{selected_part}{invalid} Remaining options are {options}."
        if options
        else f"{selected_part}{invalid}"
    )


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


def ask_for_side(context: ConversationContext, menu_repo: MenuRepository, payload: dict | None = None) -> str:
    item = menu_repo.store.get_item(context.current_item_id)
    group = item.side_groups[context.current_side_group_index]
    group_payload = _current_side_payload(context, menu_repo, payload)

    group_label = _clean_group_label(group.name, "side").lower()
    options = _format_options([choice.name for choice in group.choices])
    min_selector = int(group_payload.get("min_selector", 0) or 0)
    max_selector = int(group_payload.get("max_selector", 0) or 0)

    if options and (min_selector > 0 or max_selector > 1):
        return _initial_multi_select_prompt(
            item_name=item.name,
            group_label=group_label,
            options=options,
            min_selector=min_selector,
            max_selector=max_selector,
            optional=min_selector == 0,
        )

    if min_selector > 0:
        if options:
            return f"Which {group_label} would you like with your {item.name}? {options}."
        return f"Which {group_label} would you like with your {item.name}?"

    if options:
        return f"Would you like a {group_label} with your {item.name}? {options}, or say no."
    return f"Would you like a {group_label} with your {item.name}? You can also say no."


def ask_for_modifier(context: ConversationContext, menu_repo: MenuRepository, payload: dict | None = None) -> str:
    item = menu_repo.store.get_item(context.current_item_id)
    group = item.modifier_groups[context.current_modifier_group_index]
    group_payload = _current_modifier_payload(context, menu_repo, payload)

    group_label = _clean_group_label(group.name, "add-on").lower()
    options = _format_options([choice.name for choice in group.choices])
    min_selector = int(group_payload.get("min_selector", 0) or 0)
    max_selector = int(group_payload.get("max_selector", 0) or 0)

    if options and (min_selector > 0 or max_selector > 1):
        return _initial_multi_select_prompt(
            item_name=item.name,
            group_label=group_label,
            options=options,
            min_selector=min_selector,
            max_selector=max_selector,
            optional=min_selector == 0,
        )

    if min_selector > 0:
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
    side_payload = _current_side_payload(context, menu_repo, payload)
    if side_payload.get("top_choices") or side_payload.get("all_choices"):
        return _progress_prompt(
            side_payload,
            item_word="side",
            invalid_lead="I didn't catch a valid side.",
        )
    return "Please choose one of the available sides."


def repeat_modifier_options(context, menu_repo, payload) -> str:
    modifier_payload = _current_modifier_payload(context, menu_repo, payload)
    if modifier_payload.get("top_choices") or modifier_payload.get("all_choices"):
        return _progress_prompt(
            modifier_payload,
            item_word="option",
            invalid_lead="I didn't catch a valid option.",
        )
    return "Please choose one of the available options."


def too_many_side_choices(context, menu_repo, payload) -> str:
    side_payload = _current_side_payload(context, menu_repo, payload)
    options = _format_options(side_payload.get("top_choices") or side_payload.get("all_choices") or _top_side_choices(context, menu_repo))
    max_selector = int(side_payload.get("max_selector", 0) or 0)
    if options:
        if max_selector > 1:
            return f"That is too many sides. You can choose up to {max_selector}. Please pick again from {options}."
        return f"That is too many sides. Please choose from {options}."
    return "That is too many sides. Please choose fewer options."


def too_many_modifier_choices(context, menu_repo, payload) -> str:
    modifier_payload = _current_modifier_payload(context, menu_repo, payload)
    options = _format_options(modifier_payload.get("top_choices") or modifier_payload.get("all_choices") or _top_modifier_choices(context, menu_repo))
    max_selector = int(modifier_payload.get("max_selector", 0) or 0)
    if options:
        if max_selector > 1:
            return f"That is too many extras. You can choose up to {max_selector}. Please pick again from {options}."
        return f"That is too many extras. Please choose from {options}."
    return "That is too many extras. Please choose fewer options."


def required_side_cannot_skip(context, menu_repo, payload: dict | None = None) -> str:
    side_payload = _current_side_payload(context, menu_repo, payload)
    options = _format_options(side_payload.get("top_choices") or _top_side_choices(context, menu_repo))
    remaining = max(int(side_payload.get("remaining_to_min", 0) or 0), 0)
    if options:
        if remaining > 1:
            return f"This item still needs {remaining} more sides. Please choose from {options}."
        return f"This item needs a side. Please choose {options}."
    return "This item needs a side. Please choose one."


def required_modifier_cannot_skip(context, menu_repo, payload: dict | None = None) -> str:
    modifier_payload = _current_modifier_payload(context, menu_repo, payload)
    options = _format_options(modifier_payload.get("top_choices") or _top_modifier_choices(context, menu_repo))
    remaining = max(int(modifier_payload.get("remaining_to_min", 0) or 0), 0)
    if options:
        if remaining > 1:
            return f"This item still needs {remaining} more options. Please choose from {options}."
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
    side_payload = _current_side_payload(context, menu_repo, payload)
    options = _format_options(side_payload.get("top_choices") or _top_side_choices(context, menu_repo))
    if options:
        max_selector = int(side_payload.get("max_selector", 0) or 0)
        if max_selector > 1:
            return f"You can choose up to {max_selector}. Your side options are {options}."
        return f"Your side options are {options}."
    return "Here are the available side options."


def clarify_side_choice(context, menu_repo, payload) -> str:
    options = _format_options(payload.get("top_choices") or _top_side_choices(context, menu_repo))
    if options:
        return f"Did you mean {options}?"
    return "Which side did you want?"


def list_modifier_options(context, menu_repo, payload) -> str:
    modifier_payload = _current_modifier_payload(context, menu_repo, payload)
    options = _format_options(modifier_payload.get("top_choices") or _top_modifier_choices(context, menu_repo))
    if options:
        max_selector = int(modifier_payload.get("max_selector", 0) or 0)
        if max_selector > 1:
            return f"You can choose up to {max_selector}. Your options are {options}."
        return f"Your options are {options}."
    return "Here are the available options."


def clarify_modifier_choice(context, menu_repo, payload) -> str:
    options = _format_options(payload.get("top_choices") or _top_modifier_choices(context, menu_repo))
    if options:
        return f"Did you mean {options}?"
    return "Which option did you want?"
