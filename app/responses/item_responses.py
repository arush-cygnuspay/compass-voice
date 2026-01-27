# app/responses/item_responses.py
from app.menu.repository import MenuRepository
from app.state_machine.context import ConversationContext
from app.utils.top_k_choices import get_top_k_choices
import re


def _clean_group_label(name: str, fallback: str) -> str:
    if not name:
        return fallback
    label = name.strip()
    label = re.sub(r"^choose\s+(your\s+|a\s+)?", "", label, flags=re.IGNORECASE)
    label = re.sub(r"\s+", " ", label).strip()
    return label if label else fallback


def ask_for_side(context: ConversationContext, menu_repo: MenuRepository) -> str:
    item = menu_repo.store.get_item(context.current_item_id)
    idx = context.current_side_group_index
    group = item.side_groups[idx]
    group_label = _clean_group_label(group.name, "side")
    lead = "Which" if idx == 0 else "Now, which"

    if group.is_required:
        count = max(group.min_selector, 1)
        if count == 1:
            return f"{lead} {group_label} would you like with your {item.name}?"
        return (
            f"{lead} {group_label}s would you like with your {item.name}? "
            f"You can choose {count}."
        )

    return (
        f"{lead} {group_label} would you like with your {item.name}? "
        "If you want one, just tell me which."
    )


def repeat_side_options(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict,
) -> str:
    item = menu_repo.store.get_item(context.current_item_id)
    group = item.side_groups[context.current_side_group_index]
    group_label = (_clean_group_label(payload.get("group_name"), "side") if payload.get("group_name") else _clean_group_label(group.name, "side")).lower()
    top_choices = payload.get("top_choices") or [c.name for c in get_top_k_choices(group.choices, k=3)]

    if not top_choices:
        return (
            f"Sorry, that’s not available. Which {group_label} would you like "
            f"with your {item.name}?"
        )

    if len(top_choices) == 1:
        options = top_choices[0]
    elif len(top_choices) == 2:
        options = f"{top_choices[0]} or {top_choices[1]}"
    else:
        options = f"{top_choices[0]}, {top_choices[1]}, or {top_choices[2]}"

    return (
        f"Sorry, that’s not available. Popular {group_label}s include {options}. "
        f"Which {group_label} would you like with your {item.name}?"
    )


def too_many_side_choices(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict,
) -> str:
    item = menu_repo.store.get_item(context.current_item_id)
    group = item.side_groups[context.current_side_group_index]
    group_label = (_clean_group_label(payload.get("group_name"), "side") if payload.get("group_name") else _clean_group_label(group.name, "side")).lower()
    top_choices = payload.get("top_choices") or [c.name for c in get_top_k_choices(group.choices, k=3)]
    options = _format_options(top_choices)
    return (
        f"You can choose only one {group_label} for your {item.name}. "
        f"Popular choices include {options}. Which one would you like?"
    )


def ask_for_modifier(context: ConversationContext, menu_repo: MenuRepository) -> str:
    item = menu_repo.store.get_item(context.current_item_id)

    idx = context.current_modifier_group_index
    group = item.modifier_groups[idx]
    group_label = _clean_group_label(group.name, "add-on")

    lead = "Any" if idx == 0 else "Now, any"
    return f"{lead} {group_label.lower()}s you’d like with your {item.name}?"


def ask_for_size(context: ConversationContext, menu_repo: MenuRepository) -> str:
    item = menu_repo.store.get_item(context.current_item_id)
    lines = ["Which size would you like?"]

    for v in item.pricing.variants:
        lines.append(f"- {v.label} (${v.price_cents / 100:.2f})")

    return "\n".join(lines)

def required_side_cannot_skip(
    context: ConversationContext,
    menu_repo: MenuRepository,
) -> str:
    item = menu_repo.store.get_item(context.current_item_id)
    group = item.side_groups[context.current_side_group_index]
    group_label = _clean_group_label(group.name, "side")
    count = max(group.min_selector, 1)
    top_choices = getattr(context, "payload", None)
    base = (
        f"This item needs {count} {group_label}s. "
        f"Which {group_label}s would you like with your {item.name}? You can choose {count}."
    ) if count > 1 else (
        f"This item needs a {group_label}. "
        f"Which {group_label} would you like with your {item.name}?"
    )
    return _append_top_choices(base, context, group.choices)


def required_modifier_cannot_skip(
    context: ConversationContext,
    menu_repo: MenuRepository,
) -> str:
    item = menu_repo.store.get_item(context.current_item_id)
    group = item.modifier_groups[context.current_modifier_group_index]
    group_label = _clean_group_label(group.name, "add-on")
    base = (
        f"This item needs a {group_label.lower()}. "
        f"Which {group_label.lower()} would you like with your {item.name}?"
    )
    return _append_top_choices(base, context, group.choices)


def too_many_modifier_choices(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict,
) -> str:
    item = menu_repo.store.get_item(context.current_item_id)
    group = item.modifier_groups[context.current_modifier_group_index]
    group_label = (_clean_group_label(payload.get("group_name"), "add-on") if payload.get("group_name") else _clean_group_label(group.name, "add-on")).lower()
    top_choices = payload.get("top_choices") or [c.name for c in get_top_k_choices(group.choices, k=3)]
    options = _format_options(top_choices)
    return (
        f"You can choose only one {group_label} here. "
        f"Popular choices include {options}. Which one would you like?"
    )


def _append_top_choices(text: str, context: ConversationContext, choices) -> str:
    try:
        top_choices = context.payload.get("top_choices") if hasattr(context, "payload") else None
    except Exception:
        top_choices = None
    if not top_choices and choices:
        top_choices = [c.name for c in get_top_k_choices(choices, k=3)]
    if not top_choices:
        return text
    formatted = _format_options(top_choices)
    if not formatted:
        return text
    return f"{text} Popular choices include {formatted}."


def required_size_cannot_skip(
    context: ConversationContext,
    menu_repo: MenuRepository,
) -> str:
    item = menu_repo.store.get_item(context.current_item_id)

    lines = [
        "You must choose a valid size option to continue.",
        f"Please select one of the following sizes for your {item.name}:",
    ]

    for v in item.pricing.variants:
        lines.append(
            f"- {v.label} (${v.price_cents / 100:.2f})"
        )

    return "\n".join(lines)

def item_added_successfully(
    payload: dict,
) -> str:
    item_name = payload["item_name"]
    quantity = payload["quantity"]

    if quantity > 1:
        return (
            f"{quantity} {item_name}s have been added to your cart.\n"
            "Would you like to add anything else?"
        )

    return (
        f"Your {item_name} has been added to the cart.\n"
        "Would you like to add anything else?"
    )


def normalize_group_name(group_name: str, verb: str) -> str:
    """
    Ensures we don't duplicate verbs like:
    'choose choose cheese'
    """
    normalized = group_name.strip()

    lower = normalized.lower()
    verb_lower = verb.lower()

    if lower.startswith(verb_lower):
        return normalized[len(verb):].lstrip()

    return normalized

def repeat_modifier_options(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict,
) -> str:
    item = menu_repo.store.get_item(context.current_item_id)
    group = item.modifier_groups[context.current_modifier_group_index]
    group_label = (_clean_group_label(payload.get("group_name"), "add-on") if payload.get("group_name") else _clean_group_label(group.name, "add-on")).lower()
    invalid = payload.get("invalid_terms") or []
    top_choices = payload.get("top_choices") or [c.name for c in get_top_k_choices(group.choices, k=3)]

    invalid_text = "" if not invalid else f"I couldn’t find {', '.join(invalid)}. "

    if not top_choices:
        return (
            f"{invalid_text}Which {group_label} would you like "
            f"for your {item.name}?"
        ).strip()

    if len(top_choices) == 1:
        options = top_choices[0]
    elif len(top_choices) == 2:
        options = f"{top_choices[0]} or {top_choices[1]}"
    else:
        options = f"{top_choices[0]}, {top_choices[1]}, or {top_choices[2]}"

    return (
        f"{invalid_text}Popular {group_label}s include {options}. "
        f"Which would you like for your {item.name}?"
    ).strip()


def _format_options(options: list[str]) -> str:
    if not options:
        return ""
    if len(options) == 1:
        return options[0]
    if len(options) == 2:
        return f"{options[0]} or {options[1]}"
    return f"{options[0]}, {options[1]}, or {options[2]}"


def list_side_options(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict,
) -> str:
    item = menu_repo.store.get_item(context.current_item_id)
    group = item.side_groups[context.current_side_group_index]
    group_label = (_clean_group_label(payload.get("group_name"), "side") if payload.get("group_name") else _clean_group_label(group.name, "side")).lower()
    top_choices = payload.get("top_choices") or [c.name for c in get_top_k_choices(group.choices, k=3)]

    if not top_choices:
        return f"You can pick any {group_label} available for your {item.name}."

    options = _format_options(top_choices)
    return f"You can choose  {options}."


def clarify_side_choice(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict,
) -> str:
    item = menu_repo.store.get_item(context.current_item_id)
    group = item.side_groups[context.current_side_group_index]
    group_label = (_clean_group_label(payload.get("group_name"), "side") if payload.get("group_name") else _clean_group_label(group.name, "side")).lower()
    top_choices = payload.get("top_choices") or [c.name for c in get_top_k_choices(group.choices, k=3)]
    options = _format_options(top_choices)
    return f"Did you mean {options}, or something else for your {group_label}?"


def list_modifier_options(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict,
) -> str:
    item = menu_repo.store.get_item(context.current_item_id)
    group = item.modifier_groups[context.current_modifier_group_index]
    group_label = (_clean_group_label(payload.get("group_name"), "add-on") if payload.get("group_name") else _clean_group_label(group.name, "add-on")).lower()
    top_choices = payload.get("top_choices") or [c.name for c in get_top_k_choices(group.choices, k=3)]

    if not top_choices:
        return f"You can add any {group_label} you like to your {item.name}."

    options = _format_options(top_choices)
    return f"You can choose  {options}."


def clarify_modifier_choice(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict,
) -> str:
    item = menu_repo.store.get_item(context.current_item_id)
    group = item.modifier_groups[context.current_modifier_group_index]
    group_label = (_clean_group_label(payload.get("group_name"), "add-on") if payload.get("group_name") else _clean_group_label(group.name, "add-on")).lower()
    top_choices = payload.get("top_choices") or [c.name for c in get_top_k_choices(group.choices, k=3)]
    options = _format_options(top_choices)
    return f"Did you mean {options}, or something else for your {group_label}?"
