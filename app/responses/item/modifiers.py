# app/responses/item/modifiers.py
"""Voice responses for the modifier-selection step of the add-item flow."""
from __future__ import annotations

from app.menu.repository import MenuRepository
from app.policies.prompt_reprompt_policy import PromptRepromptPolicy, RepromptAction
from app.responses.item.format_utils import (
    _GENERIC_MODIFIER_NOUN,
    _build_entity_feedback,
    _current_modifier_payload,
    _format_options,
    _format_selected_names,
    _progress_prompt,
    _top_modifier_choices,
)
from app.state_machine.handlers.item.add_item.group_classification import (
    ordinal_word,
    speech_noun_for_modifier_group,
)
from app.state_machine.models.conversation_context import ConversationContext

_MODIFIER_OVERFLOW_HINT = "or say 'options' to hear them all"


def build_modifier_prompt_lead(
    *,
    position: int,
    total: int,
    is_last: bool,
    speech_noun: str,
) -> str:
    """Return the opening sentence for a modifier-selection prompt.

    Single group:       "Which {noun} would you like?"
    First of two:       "Choose your {noun}."
    Last of two+:       "Lastly, choose your {noun}."
    Middle/last (3+):   "Now choose your {ordinal} {noun}."
    """
    noun = speech_noun or _GENERIC_MODIFIER_NOUN

    if total <= 1:
        return f"Which {noun} would you like?"

    if is_last:
        return f"Lastly, choose your {noun}."

    if position == 0:
        if total == 2:
            return f"Choose your {noun}."
        return f"Choose your first {noun}."

    return f"Now choose your {ordinal_word(position + 1)} {noun}."


def ask_for_modifier(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict | None = None,
) -> str:
    payload = payload or {}

    min_selector = 0
    top_choices: list[str] = list(payload.get("top_choices") or [])

    try:
        group_payload = _current_modifier_payload(context, menu_repo, payload)
        min_selector = int(group_payload.get("min_selector", 0) or 0)
        top_choices = group_payload.get("top_choices") or top_choices
    except Exception:
        pass

    speech_noun = str(payload.get("speech_noun") or _GENERIC_MODIFIER_NOUN)
    total_choices = int(payload.get("total_choices") or len(top_choices))
    options_str = _format_options(
        top_choices,
        max_items=6,
        overflow_hint=_MODIFIER_OVERFLOW_HINT if total_choices > 6 else None,
        has_more=total_choices > 6 if total_choices else None,
    )

    # ── Multi-select path (required group, min ≥ 2) ──────────────────────────
    if min_selector > 1:
        selected_count = int(payload.get("selected_count") or 0)
        remaining = int(
            payload.get("remaining_to_min") or max(min_selector - selected_count, 0)
        )
        if remaining > 1:
            if options_str:
                return f"Choose {remaining} {speech_noun}s: {options_str}."
            return f"Choose {remaining} {speech_noun}s."
        # remaining == 1
        if options_str:
            return f"Choose 1 more {speech_noun}: {options_str}."
        return f"Choose 1 more {speech_noun}."

    # Progressive lead
    position = int(payload.get("modifier_group_position") or 0)
    total = int(payload.get("total_modifier_groups") or 1)
    is_last = bool(payload.get("is_last_modifier_prompt", total <= 1))

    lead = build_modifier_prompt_lead(
        position=position,
        total=total,
        is_last=is_last,
        speech_noun=speech_noun,
    )

    if options_str:
        prompt = f"{lead} {options_str}."
    else:
        item_name = (
            payload.get("item_name")
            or payload.get("current_item_name")
            or getattr(context, "current_item_name", None)
        )
        if item_name:
            prompt = f"Any {speech_noun} with your {item_name}?"
        else:
            prompt = lead

    if min_selector == 0:
        return f"{prompt} You can say none."
    return prompt


def repeat_modifier_options(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict | None,
) -> str:
    _payload = payload or {}
    feedback = _build_entity_feedback(_payload)
    reason = _payload.get("repeat_reason")

    # Duplicate-modifier shortcut: skip reprompt policy, give targeted feedback.
    if reason == "duplicate":
        duplicate_names = _payload.get("duplicate_names") or []
        noun = str(_payload.get("speech_noun") or _GENERIC_MODIFIER_NOUN)
        if duplicate_names:
            already_text = _format_selected_names(duplicate_names)
            return f"I already have {already_text}. Choose another {noun}."
        return f"You already selected that {noun}. Choose a different one."

    miss_count = int(_payload.get("reprompt_count") or 0)
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
    noun = str((payload or {}).get("speech_noun") or _GENERIC_MODIFIER_NOUN)
    article = "An" if noun[0].lower() in "aeiou" else "A"
    if options:
        if remaining > 1:
            return f"Need {remaining} more {noun}s. {options}."
        return f"{article} {noun} is required. {options}."
    if remaining > 1:
        return f"Need {remaining} more {noun}s."
    return f"{article} {noun} is required."


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
