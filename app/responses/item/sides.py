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
from app.state_machine.handlers.item.add_item.group_classification import ordinal_word
from app.state_machine.models.conversation_context import ConversationContext

_SIDE_OVERFLOW_HINT = "or say 'options' to hear them all"


def build_side_prompt_lead(
    *,
    position: int,
    total: int,
    is_drink_group: bool,
    is_last_side_prompt: bool,
    speech_noun: str,
) -> str:
    """Return the opening sentence for a side-selection prompt.

    Implements progressive/ordinal wording per the UX spec:

    One group total:
        drink  → "Which drink would you like?"
        side   → "Which side would you like?"

    Two groups, first is food then drink:
        pos 0  → "Choose your side."
        pos 1  → "Lastly, choose your drink."

    Three or more groups, last is drink:
        pos 0  → "Choose your first side."
        pos 1  → "Now choose your second side."
        pos N  → "Lastly, choose your drink."  (drink, last)
        pos N  → "Now choose your Nth side."   (non-drink, last)

    Middle positions always use "Now choose your Nth <noun>."
    """
    noun = speech_noun or _GENERIC_SIDE_NOUN

    # ── Single group: no ordinal, no "Lastly" ────────────────────────────────
    if total <= 1:
        return f"Which {noun} would you like?"

    # ── Last group ────────────────────────────────────────────────────────────
    if is_last_side_prompt:
        if is_drink_group:
            return f"Lastly, choose your {noun}."
        # Non-drink final group — use "Now" + ordinal
        return f"Now choose your {ordinal_word(position + 1)} {noun}."

    # ── First group ───────────────────────────────────────────────────────────
    if position == 0:
        if total == 2:
            return f"Choose your {noun}."
        return f"Choose your first {noun}."

    # ── Middle groups ─────────────────────────────────────────────────────────
    return f"Now choose your {ordinal_word(position + 1)} {noun}."


def ask_for_side(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict | None = None,
) -> str:
    payload = payload or {}

    min_selector = 0
    top_choices: list[str] = list(payload.get("top_choices") or [])

    try:
        group_payload = _current_side_payload(context, menu_repo, payload)
        min_selector = int(group_payload.get("min_selector", 0) or 0)
        top_choices = group_payload.get("top_choices") or top_choices
    except Exception:
        pass

    # ── Progressive lead ──────────────────────────────────────────────────────
    position = int(payload.get("side_group_position") or 0)
    total = int(payload.get("total_side_groups") or 1)
    is_drink = bool(payload.get("is_drink_group", False))
    is_last = bool(payload.get("is_last_side_prompt", total <= 1))
    speech_noun = str(payload.get("speech_noun") or _GENERIC_SIDE_NOUN)

    lead = build_side_prompt_lead(
        position=position,
        total=total,
        is_drink_group=is_drink,
        is_last_side_prompt=is_last,
        speech_noun=speech_noun,
    )

    total_choices = int(payload.get("total_choices") or len(top_choices))
    options_str = _format_options(
        top_choices,
        max_items=6,
        overflow_hint=_SIDE_OVERFLOW_HINT if total_choices > 6 else None,
        has_more=total_choices > 6 if total_choices else None,
    )

    if options_str:
        prompt = f"{lead} {options_str}."
    else:
        # No options to list — degrade gracefully with item name if available.
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
    noun = str((payload or {}).get("speech_noun") or _GENERIC_SIDE_NOUN)
    if options:
        if remaining > 1:
            return f"Need {remaining} more {noun}s. {options}."
        return f"A {noun} is required. {options}."
    if remaining > 1:
        return f"Need {remaining} more {noun}s."
    return f"A {noun} is required."


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


def block_new_item_until_required_done(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict | None,
) -> str:
    """Tell the customer they must finish the current required group before
    starting a new item.  Always offers an escape via 'cancel'."""
    p = payload or {}
    item_name = str(p.get("pending_item_name") or "").strip()
    group_noun = str(p.get("group_prompt_noun") or "option").strip() or "option"
    if item_name:
        return (
            f"I'm still finishing your {item_name}. "
            f"Please choose a {group_noun} first, or say cancel to drop it."
        )
    return (
        f"Please choose a {group_noun} first, or say cancel to drop the current item."
    )
