# app/responses/item/sizes.py
"""Voice responses for the size-selection step of the add-item flow."""
from __future__ import annotations

from app.menu.repository import MenuRepository
from app.policies.prompt_reprompt_policy import PromptRepromptPolicy, RepromptAction
from app.responses.item.format_utils import _format_options
from app.state_machine.models.conversation_context import ConversationContext


def ask_for_size(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict | None = None,
) -> str:
    payload = payload or {}
    item_name = payload.get("current_item_name")
    if not item_name:
        item = menu_repo.store.get_item(context.current_item_id)
        item_name = item.name
    return f"What size would you like for {item_name}?"


def repeat_size_options(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict | None,
) -> str:
    item = menu_repo.store.get_item(context.current_item_id)
    labels = [variant.label for variant in (item.pricing.variants or []) if variant.label]
    options = _format_options(labels)
    item_name = item.name

    if (payload or {}).get("list_options_requested"):
        if options:
            return f"Available sizes for {item_name} are {options}."
        return f"What size would you like for {item_name}?"

    miss_count = int((payload or {}).get("reprompt_count") or 0)
    action = PromptRepromptPolicy.next_action("size", miss_count)

    if action == RepromptAction.FULL_OPTIONS:
        if options:
            return f"Available sizes for {item_name} are {options}."
        return f"What size would you like for {item_name}?"

    if action == RepromptAction.CONCISE:
        return f"What size for {item_name}?"

    if action == RepromptAction.LIST_OPTIONS_HINT:
        return f"I didn't catch that. Say 'list options' to hear all sizes for {item_name}."

    if options:
        return f"Please say one of these sizes for {item_name}: {options}."
    return f"Please say the size for {item_name}."


def required_size_cannot_skip(
    context: ConversationContext,
    menu_repo: MenuRepository,
) -> str:
    item = menu_repo.store.get_item(context.current_item_id)
    labels = [variant.label for variant in (item.pricing.variants or []) if variant.label]
    options = _format_options(labels)
    if options:
        return f"Please choose a size for {item.name}: {options}."
    return f"Please choose a size for {item.name}."


def invalid_size_option(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict | None,
) -> str:
    item = menu_repo.store.get_item(context.current_item_id)
    labels = [variant.label for variant in (item.pricing.variants or []) if variant.label]
    options = _format_options(labels)

    if (payload or {}).get("reprompt_escalation"):
        if options:
            return f"Let's make this easy. Say {options} for {item.name}."
        return f"Let's make this easy. Please say the size for {item.name}."

    if options:
        return f"That size is not available for {item.name}. Please choose {options}."
    return f"That size is not available for {item.name}. Please choose another size."
