# app/responses/item/modifiers.py
"""Voice responses for the modifier-selection step of the add-item flow."""
from __future__ import annotations

from app.menu.repository import MenuRepository
from app.policies.prompt_reprompt_policy import PromptRepromptPolicy, RepromptAction
from app.responses.item.format_utils import (
    _GENERIC_MODIFIER_NOUN,
    _build_entity_feedback,
    _current_modifier_payload,
    _format_examples,
    _format_options,
    _format_selected_names,
    _progress_prompt,
    _top_modifier_choices,
)
from app.state_machine.models.conversation_context import ConversationContext


def ask_for_modifier(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict | None = None,
) -> str:
    payload = payload or {}

    item_name: str | None = None
    noun = _GENERIC_MODIFIER_NOUN
    verb = "would you like"
    min_selector = 0
    top_choices: list[str] = list(payload.get("top_choices") or [])

    try:
        item = menu_repo.store.get_item(context.current_item_id)
        group = item.modifier_groups[context.current_modifier_group_index]
        item_name = item.name
        noun = (getattr(group, "prompt_noun", None) or _GENERIC_MODIFIER_NOUN).strip() or _GENERIC_MODIFIER_NOUN
        verb = (getattr(group, "prompt_verb", None) or "would you like").strip() or "would you like"
        group_payload = _current_modifier_payload(context, menu_repo, payload)
        min_selector = int(group_payload.get("min_selector", 0) or 0)
        top_choices = group_payload.get("top_choices") or top_choices
    except Exception:
        pass

    item_name = (
        item_name
        or payload.get("current_item_name")
        or getattr(context, "current_item_name", None)
        or "your item"
    )

    examples = _format_examples(top_choices)
    if examples:
        prompt = f"Any {noun} {verb}, like {examples}?"
    else:
        prompt = f"Any {noun} {verb} on your {item_name}?"

    if min_selector == 0:
        return f"{prompt} You can say none."
    return prompt


def repeat_modifier_options(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict | None,
) -> str:
    feedback = _build_entity_feedback(payload or {})
    miss_count = int((payload or {}).get("reprompt_count") or 0)
    action = PromptRepromptPolicy.next_action("modifier", miss_count)

    if action == RepromptAction.CONCISE:
        return f"{feedback}Which option?" if feedback else "Which option?"

    if action == RepromptAction.LIST_OPTIONS_HINT:
        hint = "I didn't catch that. Say 'list options' to hear all choices."
        return f"{feedback}{hint}" if feedback else hint

    modifier_payload = _current_modifier_payload(context, menu_repo, payload)
    if modifier_payload.get("top_choices") or modifier_payload.get("all_choices"):
        prompt = _progress_prompt(
            modifier_payload,
            item_word="option",
            invalid_lead="I didn't catch a valid option.",
        )
        return f"{feedback}{prompt}" if feedback else prompt
    return (
        f"{feedback}Please choose one of the available options."
        if feedback
        else "Please choose one of the available options."
    )


def too_many_modifier_choices(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict | None,
) -> str:
    modifier_payload = _current_modifier_payload(context, menu_repo, payload)
    max_selector = int(modifier_payload.get("max_selector", 0) or 0)

    accepted_names = (payload or {}).get("accepted_names") or []
    requested_names = (payload or {}).get("requested_names") or []
    dropped_names = (payload or {}).get("dropped_names") or []
    unmatched = (payload or {}).get("unmatched_names") or []

    parts: list[str] = []

    if accepted_names:
        parts.append(f"I added {_format_selected_names(accepted_names)}.")
    if dropped_names and not accepted_names:
        limit_note = f"You can only pick {max_selector}" if max_selector > 0 else "That's the limit"
        parts.append(
            f"{limit_note}. Please choose again from {_format_selected_names(requested_names or dropped_names)}."
        )
    elif dropped_names:
        limit_note = f"You can only pick {max_selector}" if max_selector > 0 else "That's the limit"
        parts.append(f"{limit_note}, so I couldn't add {_format_selected_names(dropped_names)}.")
    if unmatched:
        parts.append(f"I couldn't find {_format_selected_names(unmatched)}.")

    if parts:
        if accepted_names:
            return " ".join(parts) + " Say done when you're ready."
        return " ".join(parts)

    options = _format_options(
        modifier_payload.get("top_choices")
        or modifier_payload.get("all_choices")
        or _top_modifier_choices(context, menu_repo)
    )
    if options:
        if max_selector > 1:
            return f"That is too many extras. You can choose up to {max_selector}. Please pick again from {options}."
        return f"That is too many extras. Please choose from {options}."
    return "That is too many extras. Please choose fewer options."


def required_modifier_cannot_skip(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict | None = None,
) -> str:
    modifier_payload = _current_modifier_payload(context, menu_repo, payload)
    options = _format_options(modifier_payload.get("top_choices") or _top_modifier_choices(context, menu_repo))
    remaining = max(int(modifier_payload.get("remaining_to_min", 0) or 0), 0)
    if options:
        if remaining > 1:
            return f"Need {remaining} more options. {options}."
        return f"An option is required. {options}."
    return "An option is required."


def list_modifier_options(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict | None,
) -> str:
    modifier_payload = _current_modifier_payload(context, menu_repo, payload)
    all_choices = (
        modifier_payload.get("all_choices")
        or modifier_payload.get("top_choices")
        or _top_modifier_choices(context, menu_repo, k=6)
    )
    options = _format_options(all_choices, max_items=6)
    if options:
        max_selector = int(modifier_payload.get("max_selector", 0) or 0)
        prefix = "Let me list them. " if (payload or {}).get("reprompt_escalation") else ""
        if max_selector > 1:
            return f"{prefix}Up to {max_selector}. {options}."
        return f"{prefix}Your options are {options}."
    return "Let me list the options."


def clarify_modifier_choice(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict | None,
) -> str:
    options = _format_options((payload or {}).get("top_choices") or _top_modifier_choices(context, menu_repo))
    if options:
        return f"Did you mean {options}?"
    return "Which option did you want?"
