# app/responses/item/sides.py
"""Voice responses for the side-selection step of the add-item flow."""
from __future__ import annotations

from app.menu.repository import MenuRepository
from app.responses.item.format_utils import (
    _GENERIC_SIDE_NOUN,
    _build_entity_feedback,
    _current_side_payload,
    _format_examples,
    _format_options,
    _format_selected_names,
    _progress_prompt,
    _top_side_choices,
)
from app.state_machine.models.conversation_context import ConversationContext


def ask_for_side(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict | None = None,
) -> str:
    payload = payload or {}

    item_name: str | None = None
    noun = _GENERIC_SIDE_NOUN
    verb = "would you like"
    min_selector = 0
    top_choices: list[str] = list(payload.get("top_choices") or [])

    try:
        item = menu_repo.store.get_item(context.current_item_id)
        group = item.side_groups[context.current_side_group_index]
        item_name = item.name
        noun = (getattr(group, "prompt_noun", None) or _GENERIC_SIDE_NOUN).strip() or _GENERIC_SIDE_NOUN
        verb = (getattr(group, "prompt_verb", None) or "would you like").strip() or "would you like"
        group_payload = _current_side_payload(context, menu_repo, payload)
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
        prompt = f"Any {noun} {verb} with your {item_name}?"

    if min_selector == 0:
        return f"{prompt} You can say none."
    return prompt


def repeat_side_options(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict | None,
) -> str:
    feedback = _build_entity_feedback(payload or {})
    side_payload = _current_side_payload(context, menu_repo, payload)
    if side_payload.get("top_choices") or side_payload.get("all_choices"):
        group_name = (side_payload.get("group_name") or "").strip().lower()
        invalid_lead = (
            f"I didn't catch a valid {group_name} option."
            if group_name
            else "I didn't catch a valid side."
        )
        prompt = _progress_prompt(side_payload, item_word="side", invalid_lead=invalid_lead)
        return f"{feedback}{prompt}" if feedback else prompt
    return (
        f"{feedback}Please choose one of the available sides."
        if feedback
        else "Please choose one of the available sides."
    )


def too_many_side_choices(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict | None,
) -> str:
    side_payload = _current_side_payload(context, menu_repo, payload)
    max_selector = int(side_payload.get("max_selector", 0) or 0)

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
        side_payload.get("top_choices")
        or side_payload.get("all_choices")
        or _top_side_choices(context, menu_repo)
    )
    if options:
        if max_selector > 1:
            return f"That is too many sides. You can choose up to {max_selector}. Please pick again from {options}."
        return f"That is too many sides. Please choose from {options}."
    return "That is too many sides. Please choose fewer options."


def required_side_cannot_skip(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict | None = None,
) -> str:
    side_payload = _current_side_payload(context, menu_repo, payload)
    options = _format_options(side_payload.get("top_choices") or _top_side_choices(context, menu_repo))
    remaining = max(int(side_payload.get("remaining_to_min", 0) or 0), 0)
    if options:
        if remaining > 1:
            return f"Need {remaining} more sides. {options}."
        return f"A side is required. {options}."
    return "A side is required."


def list_side_options(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict | None,
) -> str:
    side_payload = _current_side_payload(context, menu_repo, payload)
    all_choices = (
        side_payload.get("all_choices")
        or side_payload.get("top_choices")
        or _top_side_choices(context, menu_repo, k=6)
    )
    options = _format_options(all_choices, max_items=6)
    if options:
        max_selector = int(side_payload.get("max_selector", 0) or 0)
        prefix = "Let me list them. " if (payload or {}).get("reprompt_escalation") else ""
        if max_selector > 1:
            return f"{prefix}Up to {max_selector}. {options}."
        return f"{prefix}Your options are {options}."
    return "Let me list the side options."


def clarify_side_choice(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict | None,
) -> str:
    options = _format_options((payload or {}).get("top_choices") or _top_side_choices(context, menu_repo))
    if options:
        return f"Did you mean {options}?"
    return "Which side did you want?"
