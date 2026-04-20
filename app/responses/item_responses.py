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
    """Format a list of options for voice output.

    Shows up to ``max_items`` names.  When there are more options beyond what
    is shown, appends "and N more" so the caller knows there are extras
    they can ask about.
    """
    clean = [str(option).strip() for option in options if str(option).strip()]
    total = len(clean)
    limited = clean[:max_items]
    extras = total - len(limited)

    if not limited:
        return ""

    # Build the readable list — "A, B, C, or D"
    if len(limited) == 1:
        base = limited[0]
    elif len(limited) == 2:
        base = f"{limited[0]} or {limited[1]}"
    else:
        lead = ", ".join(limited[:-1])
        base = f"{lead}, or {limited[-1]}"

    if extras > 0:
        return f"{base}, and {extras} more"
    return base


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


def _build_entity_feedback(payload: dict) -> str:
    """Build spoken feedback for matched and unmatched entity names.

    Returns a prefix like:
        "Got Bacon and Mushrooms. I couldn't find tenderloin. "
        "Got Bacon. "
        "I couldn't find tenderloin. "
        ""
    """
    parts: list[str] = []

    matched = payload.get("matched_names") or []
    if matched:
        parts.append(f"Got {_format_selected_names(matched)}.")

    unmatched = payload.get("unmatched_names") or []
    if unmatched:
        parts.append(f"I couldn't find {_format_selected_names(unmatched)}.")

    if not parts:
        return ""
    return " ".join(parts) + " "


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

    # Build lookup from choices list (menu SideGroup has choices list, not dict)
    choices_by_id = getattr(group, "choices_by_item_id", None)
    if choices_by_id is None:
        choices_by_id = {choice.item_id: choice for choice in group.choices}
    selected_names = [
        choices_by_id[item_id].name
        for item_id in selected_ids
        if item_id in choices_by_id
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
        elif selection.instruction == "on_side":
            selected_names.append(f"{selection.name} on the side")
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


def _initial_multi_select_prompt(*, item_name: str, group_label: str, options: str, min_selector: int, max_selector: int, optional: bool, total_choices: int = 0) -> str:
    # For large groups (>6 options), hint that user can ask for the full list.
    ask_hint = " Ask for options to hear more." if total_choices > 6 else ""

    if optional:
        if max_selector == 1:
            return f"Any {group_label} with your {item_name}? {options}, or no.{ask_hint}"
        return (
            f"Any {group_label} for your {item_name}? Up to {max_selector}. "
            f"{options}, or no.{ask_hint}"
        )

    if min_selector == max_selector:
        if min_selector == 1:
            return f"Which {group_label} for your {item_name}? {options}.{ask_hint}"
        return f"Pick {min_selector} {group_label} for your {item_name}. {options}.{ask_hint}"

    if max_selector > 0:
        return f"Pick up to {max_selector} {group_label} for your {item_name}. {options}.{ask_hint}"

    return f"Pick at least {min_selector} {group_label} for your {item_name}. {options}.{ask_hint}"


def _progress_prompt(payload: dict, *, item_word: str, invalid_lead: str) -> str:
    top_choices = _payload_value(payload, "top_choices", [])
    all_choices = _payload_value(payload, "all_choices", [])
    option_values = top_choices if top_choices else all_choices
    options = _format_options(option_values)

    reason = payload.get("repeat_reason")
    if reason == "need_more":
        remaining = max(int(payload.get("remaining_to_min", 0) or 0), 0)
        if remaining == 1:
            prompt = f"Pick 1 more."
        else:
            prompt = f"Pick {remaining} more."
        return f"{prompt} {options}." if options else prompt

    if reason == "optional_more":
        remaining = max(int(payload.get("remaining_to_max", 0) or 0), 0)
        if remaining > 0:
            if remaining == 1:
                return f"Add 1 more, or say done. {options}." if options else "Add 1 more, or say done."
            return f"Up to {remaining} more, or say done. {options}." if options else f"Up to {remaining} more, or say done."
        return "Say done when ready."

    remaining = max(int(payload.get("remaining_to_min", 0) or 0), 0)
    if remaining == 0:
        remaining_to_max = max(int(payload.get("remaining_to_max", 0) or 0), 0)
        if remaining_to_max > 0 and options:
            if remaining_to_max == 1:
                return f"Add 1 more, or say done. {options}."
            return f"Up to {remaining_to_max} more, or say done. {options}."
        return "Say done when ready."

    if remaining == 1:
        prompt = f"{invalid_lead} Pick 1 more."
    else:
        prompt = f"{invalid_lead} Pick {remaining} more."
    return f"{prompt} {options}." if options else prompt


def _top_side_choices(
    context: ConversationContext,
    menu_repo: MenuRepository,
    k: int = 4,
) -> list[str]:
    item = menu_repo.store.get_item(context.current_item_id)
    group = item.side_groups[context.current_side_group_index]
    return [choice.name for choice in get_top_k_choices(group.choices, k=k)]


def _top_modifier_choices(
    context: ConversationContext,
    menu_repo: MenuRepository,
    k: int = 4,
) -> list[str]:
    item = menu_repo.store.get_item(context.current_item_id)
    group = item.modifier_groups[context.current_modifier_group_index]
    return [choice.name for choice in get_top_k_choices(group.choices, k=k)]


def ask_for_side(context: ConversationContext, menu_repo: MenuRepository, payload: dict | None = None) -> str:
    item = menu_repo.store.get_item(context.current_item_id)
    group = item.side_groups[context.current_side_group_index]
    group_payload = _current_side_payload(context, menu_repo, payload)

    group_label = _clean_group_label(group.name, "side").lower()
    choice_names = [choice.name for choice in group.choices]
    options = _format_options(choice_names)
    total = len(choice_names)
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
            total_choices=total,
        )

    ask_hint = " Ask for options to hear more." if total > 6 else ""
    if min_selector > 0:
        if options:
            return f"Which {group_label} for your {item.name}? {options}.{ask_hint}"
        return f"Which {group_label} for your {item.name}?"

    if options:
        return f"Any {group_label} with your {item.name}? {options}, or no.{ask_hint}"
    return f"Any {group_label} with your {item.name}?"


def ask_for_modifier(context: ConversationContext, menu_repo: MenuRepository, payload: dict | None = None) -> str:
    item = menu_repo.store.get_item(context.current_item_id)
    group = item.modifier_groups[context.current_modifier_group_index]
    group_payload = _current_modifier_payload(context, menu_repo, payload)

    group_label = _clean_group_label(group.name, "add-on").lower()
    choice_names = [choice.name for choice in group.choices]
    options = _format_options(choice_names)
    total = len(choice_names)
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
            total_choices=total,
        )

    ask_hint = " Ask for options to hear more." if total > 6 else ""
    if min_selector > 0:
        if options:
            return f"Which {group_label} for your {item.name}? {options}.{ask_hint}"
        return f"Which {group_label} for your {item.name}?"

    if options:
        return f"Any extras for your {item.name}? {options}, or no.{ask_hint}"
    return f"Any extras for your {item.name}?"


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
    feedback = _build_entity_feedback(payload or {})
    side_payload = _current_side_payload(context, menu_repo, payload)
    if side_payload.get("top_choices") or side_payload.get("all_choices"):
        prompt = _progress_prompt(
            side_payload,
            item_word="side",
            invalid_lead="I didn't catch a valid side.",
        )
        return f"{feedback}{prompt}" if feedback else prompt
    return f"{feedback}Please choose one of the available sides." if feedback else "Please choose one of the available sides."


def repeat_modifier_options(context, menu_repo, payload) -> str:
    feedback = _build_entity_feedback(payload or {})
    modifier_payload = _current_modifier_payload(context, menu_repo, payload)
    if modifier_payload.get("top_choices") or modifier_payload.get("all_choices"):
        prompt = _progress_prompt(
            modifier_payload,
            item_word="option",
            invalid_lead="I didn't catch a valid option.",
        )
        return f"{feedback}{prompt}" if feedback else prompt
    return f"{feedback}Please choose one of the available options." if feedback else "Please choose one of the available options."


def too_many_side_choices(context, menu_repo, payload) -> str:
    side_payload = _current_side_payload(context, menu_repo, payload)
    max_selector = int(side_payload.get("max_selector", 0) or 0)

    accepted_names = payload.get("accepted_names") or []
    dropped_names = payload.get("dropped_names") or []
    unmatched = payload.get("unmatched_names") or []

    parts: list[str] = []

    if accepted_names:
        parts.append(f"I added {_format_selected_names(accepted_names)}.")
    if dropped_names:
        limit_note = f"You can only pick {max_selector}" if max_selector > 0 else "That's the limit"
        parts.append(f"{limit_note}, so I couldn't add {_format_selected_names(dropped_names)}.")
    if unmatched:
        parts.append(f"I couldn't find {_format_selected_names(unmatched)}.")

    if parts:
        return " ".join(parts) + " Say done when you're ready."

    # Fallback (shouldn't normally reach here)
    options = _format_options(side_payload.get("top_choices") or side_payload.get("all_choices") or _top_side_choices(context, menu_repo))
    if options:
        if max_selector > 1:
            return f"That is too many sides. You can choose up to {max_selector}. Please pick again from {options}."
        return f"That is too many sides. Please choose from {options}."
    return "That is too many sides. Please choose fewer options."


def too_many_modifier_choices(context, menu_repo, payload) -> str:
    modifier_payload = _current_modifier_payload(context, menu_repo, payload)
    max_selector = int(modifier_payload.get("max_selector", 0) or 0)

    accepted_names = payload.get("accepted_names") or []
    dropped_names = payload.get("dropped_names") or []
    unmatched = payload.get("unmatched_names") or []

    parts: list[str] = []

    if accepted_names:
        parts.append(f"I added {_format_selected_names(accepted_names)}.")
    if dropped_names:
        limit_note = f"You can only pick {max_selector}" if max_selector > 0 else "That's the limit"
        parts.append(f"{limit_note}, so I couldn't add {_format_selected_names(dropped_names)}.")
    if unmatched:
        parts.append(f"I couldn't find {_format_selected_names(unmatched)}.")

    if parts:
        return " ".join(parts) + " Say done when you're ready."

    # Fallback
    options = _format_options(modifier_payload.get("top_choices") or modifier_payload.get("all_choices") or _top_modifier_choices(context, menu_repo))
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
            return f"Need {remaining} more sides. {options}."
        return f"A side is required. {options}."
    return "A side is required."


def required_modifier_cannot_skip(context, menu_repo, payload: dict | None = None) -> str:
    modifier_payload = _current_modifier_payload(context, menu_repo, payload)
    options = _format_options(modifier_payload.get("top_choices") or _top_modifier_choices(context, menu_repo))
    remaining = max(int(modifier_payload.get("remaining_to_min", 0) or 0), 0)
    if options:
        if remaining > 1:
            return f"Need {remaining} more options. {options}."
        return f"An option is required. {options}."
    return "An option is required."


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


def _format_item_summary_list(items: list[str]) -> str:
    """Format a list of item summaries for spoken output."""
    clean = [str(item).strip() for item in items if str(item).strip()]
    if not clean:
        return ""
    if len(clean) == 1:
        return clean[0]
    if len(clean) == 2:
        return f"{clean[0]} and {clean[1]}"
    return f"{', '.join(clean[:-1])}, and {clean[-1]}"


def _added_text(item_name: str, quantity: int) -> str:
    """Build "Added X" or "Added 2 X" text."""
    if quantity > 1 and item_name:
        return f"Added {quantity} {item_name}"
    if item_name:
        return f"{item_name} added"
    return "Added"


def item_added_successfully(payload: dict) -> str:
    quantity = int(payload.get("quantity", 1))
    item_name = str(payload.get("item_name") or "").strip()
    prefilled = str(payload.get("prefilled_summary") or "").strip()

    # Entity feedback (unmatched entities from the final step)
    unmatched = payload.get("unmatched_names") or []
    unmatched_note = ""
    if unmatched:
        unmatched_note = f" I couldn't find {_format_selected_names(unmatched)}."

    # ── Queue transition: previous item was added, now moving to next ──
    if payload.get("queue_transition"):
        prev_name = str(payload.get("prev_item_name") or "").strip()
        prev_qty = int(payload.get("prev_quantity", 1) or 1)
        next_name = str(payload.get("next_item_name") or "").strip()
        remaining = int(payload.get("remaining_queue_count", 0) or 0)

        added = _added_text(prev_name, prev_qty)
        # This item was also instantly added (all slots prefilled)
        this_added = _added_text(item_name, quantity)

        if remaining > 0:
            return f"{added}. {this_added}.{unmatched_note} {remaining} more to go. Would you like anything else?"
        return f"{added}. {this_added}.{unmatched_note} That's everything. Would you like anything else?"

    # ── Standard single-item response ──
    # Include prefilled details when present so user hears what was captured.
    detail = f" {prefilled}" if prefilled else ""
    if quantity > 1 and item_name:
        return f"Added {quantity} {item_name}{detail}.{unmatched_note} Would you like anything else?"
    if quantity > 1:
        return f"Added {quantity}{detail}.{unmatched_note} Would you like anything else?"
    if item_name:
        return f"{item_name}{detail} added.{unmatched_note} Would you like anything else?"
    return f"Added.{unmatched_note} Would you like anything else?"


def confirm_cancel_current_item(context, menu_repo, payload) -> str:
    item_name = payload.get("item_name") or context.current_item_name or "this item"
    return f"Cancel {item_name}?"


def confirm_cancel_current_item_for_new_request(context, menu_repo, payload) -> str:
    item_name = payload.get("item_name") or context.current_item_name or "this item"
    return f"Still adding {item_name}. Cancel it and move on?"


def continue_current_item_after_cancel_denied(context, menu_repo, payload) -> str:
    options = payload.get("available_choices") or list(context.available_choices_values)
    formatted = _format_options(options)

    if formatted:
        return f"Okay, continuing. {formatted}."
    return "Okay, continuing."


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
    # When user asks for options, show more (up to 6) so they get a fuller picture
    all_choices = side_payload.get("all_choices") or side_payload.get("top_choices") or _top_side_choices(context, menu_repo, k=6)
    options = _format_options(all_choices, max_items=6)
    if options:
        max_selector = int(side_payload.get("max_selector", 0) or 0)
        if max_selector > 1:
            return f"Up to {max_selector}. {options}."
        return f"Your options are {options}."
    return "Let me list the side options."


def clarify_side_choice(context, menu_repo, payload) -> str:
    options = _format_options(payload.get("top_choices") or _top_side_choices(context, menu_repo))
    if options:
        return f"Did you mean {options}?"
    return "Which side did you want?"


def list_modifier_options(context, menu_repo, payload) -> str:
    modifier_payload = _current_modifier_payload(context, menu_repo, payload)
    all_choices = modifier_payload.get("all_choices") or modifier_payload.get("top_choices") or _top_modifier_choices(context, menu_repo, k=6)
    options = _format_options(all_choices, max_items=6)
    if options:
        max_selector = int(modifier_payload.get("max_selector", 0) or 0)
        if max_selector > 1:
            return f"Up to {max_selector}. {options}."
        return f"Your options are {options}."
    return "Let me list the options."


def clarify_modifier_choice(context, menu_repo, payload) -> str:
    options = _format_options(payload.get("top_choices") or _top_modifier_choices(context, menu_repo))
    if options:
        return f"Did you mean {options}?"
    return "Which option did you want?"
