# app/responses/item/format_utils.py
"""Shared formatting helpers for item-centric voice responses.

All symbols here are private to the responses.item package.  External code
should import from the public modules (sides, modifiers, sizes, etc.) or from
the package __init__, never directly from this file.
"""
from __future__ import annotations

import re

from app.menu.repository import MenuRepository
from app.nlu.modifier_instructions import speak as _speak_modifier
from app.nlu.utterance_filter import DEFAULT_FILTER as _FILLER_FILTER
from app.state_machine.models.conversation_context import ConversationContext
from app.utils.top_k_choices import get_top_k_choices

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_GENERIC_SIDE_NOUN = "side"
_GENERIC_MODIFIER_NOUN = "add-on"

# Tokens that carry no menu-item signal and should not be echoed in entity
# feedback.  A phrase whose non-stop tokens number fewer than 2 is treated as
# pure filler/conversational residue.
_FEEDBACK_STOP_TOKENS: frozenset[str] = frozenset({
    "i", "me", "my", "am", "is", "are", "was", "were",
    "do", "did", "does", "be", "been", "have", "has",
    "okay", "ok", "then", "give", "want", "would", "like",
    "can", "get", "please", "just", "to", "order",
    "a", "an", "the", "and", "or", "for", "in", "at", "on", "it",
    "this", "that",
})

# Patterns that indicate a menu item is user-customizable (e.g. "Build Your Own Pizza").
_CUSTOMIZABLE_PATTERN = re.compile(
    r"\b(build\s+your\s+own|create\s+your\s+own|make\s+your\s+own|byo)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Pattern-detection helpers
# ---------------------------------------------------------------------------

def _looks_customizable(name: str) -> bool:
    return bool(_CUSTOMIZABLE_PATTERN.search(name or ""))


def _has_echable_content(text: str) -> bool:
    """Return True when text contains genuine menu-item content.

    Delegates to FillerFilter.is_filler_only so that control phrases
    ("no skip that", "can you repeat", "add done") and structural filler
    ("to order a", "okay then give me a") are never echoed as unavailable
    items, while real candidates ("dragon burger", "bacon cheeseburger")
    are preserved.
    """
    return not _FILLER_FILTER.is_filler_only(text)


def _clean_group_label(name: str | None, fallback: str) -> str:
    if not name:
        return fallback
    label = name.strip()
    label = re.sub(r"^choose\s+(your\s+|a\s+|an\s+)?", "", label, flags=re.IGNORECASE)
    label = re.sub(r"\s+", " ", label).strip()
    label = re.sub(r"(ss)\b", "s", label, flags=re.IGNORECASE)
    return label or fallback


# ---------------------------------------------------------------------------
# List-formatting helpers
# ---------------------------------------------------------------------------

def _format_suggestions(items: list[str], max_items: int = 4) -> str:
    clean = [str(s).strip() for s in items if str(s).strip()][:max_items]
    if not clean:
        return ""
    if len(clean) == 1:
        return clean[0]
    if len(clean) == 2:
        return f"{clean[0]} or {clean[1]}"
    return ", ".join(clean[:-1]) + f", or {clean[-1]}"


def _format_examples(choices: list[str], max_items: int = 3) -> str:
    clean = [str(c).strip() for c in choices if c and str(c).strip()]
    if not clean:
        return ""
    sample = clean[:max_items]
    if len(sample) == 1:
        return sample[0]
    return ", ".join(sample[:-1]) + f", or {sample[-1]}"


def _format_options(
    options: list[str],
    max_items: int = 6,
    overflow_hint: str | None = None,
    has_more: bool | None = None,
) -> str:
    """Format a list of options for voice output.

    Truncates silently at *max_items*.  When *overflow_hint* is provided and
    there are more options than *max_items*, the hint is appended after the
    last item (e.g. "or say 'options' to hear them all").  Never emits "and N
    more".  Pass *has_more=True* to force the hint when the caller knows there
    are more choices but the list is already pre-capped at *max_items*.
    """
    clean = [str(option).strip() for option in options if str(option).strip()]
    has_overflow = (len(clean) > max_items) if has_more is None else (has_more or len(clean) > max_items)
    limited = clean[:max_items]

    if not limited:
        return ""

    if len(limited) == 1:
        base = limited[0]
    elif len(limited) == 2:
        base = f"{limited[0]} or {limited[1]}"
    else:
        lead = ", ".join(limited[:-1])
        base = f"{lead}, or {limited[-1]}"

    if has_overflow and overflow_hint:
        return f"{base}, {overflow_hint}"
    return base


def format_limited_options(
    options: "list[str] | tuple[str, ...]",
    *,
    max_spoken: int = 4,
    overflow_suffix: str = "Say options to hear more",
) -> str:
    """Format options for voice with a concise spoken limit and an overflow hint.

    Speaks at most *max_spoken* options (default 4).  When more options exist,
    appends *overflow_suffix* so the customer knows they can request the full
    list.  The suffix is omitted when all options fit within *max_spoken*.

    Examples
    --------
    Six-item list with max_spoken=4:
        ["Plain", "Sesame", "Potato", "Brioche", "Pretzel", "Wheat"]
        → "Plain, Sesame, Potato, or Brioche. Say options to hear more"

    Three-item list (no overflow):
        ["Small", "Medium", "Large"]
        → "Small, Medium, or Large"
    """
    clean = [str(o).strip() for o in (options or []) if str(o).strip()]
    has_more = len(clean) > max_spoken
    limited = clean[:max_spoken]
    if not limited:
        return ""
    formatted = _format_options(limited, max_items=max_spoken)
    if has_more and overflow_suffix:
        return f"{formatted}. {overflow_suffix}"
    return formatted


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


def _format_names_with_counts(names: list[str]) -> str:
    """Format a list of names (potentially with duplicates) for voice output.

    Repeated names are collapsed into natural spoken phrases:
        ["Coke"] → "Coke"
        ["Coke", "Coke"] → "Coke twice"
        ["Coke", "Coke", "Coke"] → "Coke 3 times"
        ["Coke", "Coke", "Sprite"] → "Coke twice and Sprite"

    Order of first appearance is preserved.
    """
    from collections import Counter, OrderedDict

    clean = [str(n).strip() for n in names if str(n).strip()]
    if not clean:
        return ""

    counts: dict[str, int] = {}
    order: list[str] = []
    for name in clean:
        if name not in counts:
            counts[name] = 0
            order.append(name)
        counts[name] += 1

    parts: list[str] = []
    for name in order:
        count = counts[name]
        if count == 1:
            parts.append(name)
        elif count == 2:
            parts.append(f"{name} twice")
        else:
            parts.append(f"{name} {count} times")

    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return f"{', '.join(parts[:-1])}, and {parts[-1]}"


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


def spoken_quantity_label(quantity: int, name: str, plural: str | None = None) -> str:
    """Voice-safe quantity + name label.  Never emits ``x`` notation.

    When *quantity* is 1, returns *name* alone (no leading "1").
    For *quantity* > 1:
    - If an explicit *plural* form is given, uses that.
    - Otherwise returns ``"{quantity} {name}"`` with **no automatic pluralisation**
      to avoid naïve ``"French Friess"``-style errors on menu item names.

    Examples::

        spoken_quantity_label(1, "Coke")                   → "Coke"
        spoken_quantity_label(2, "Coke")                   → "2 Coke"
        spoken_quantity_label(2, "Coke", plural="Cokes")   → "2 Cokes"
        spoken_quantity_label(3, "French Fries")           → "3 French Fries"
    """
    q = max(1, int(quantity))
    if q == 1:
        return name or ""
    if plural is not None:
        return f"{q} {plural}"
    return f"{q} {name}"


def compact_quantity_label(quantity: int, name: str) -> str:
    """Compact voice-safe label for cart summaries.  Never emits ``x`` notation.

    Like :func:`spoken_quantity_label` without an explicit plural parameter.
    TTS engines read "2 Coke" as "two Coke" which is acceptable in a cart list
    context and avoids naïve pluralisation bugs on complex menu-item names.

    Examples::

        compact_quantity_label(1, "Coke")  → "Coke"
        compact_quantity_label(2, "Coke")  → "2 Coke"
    """
    q = max(1, int(quantity))
    if q == 1:
        return name or ""
    return f"{q} {name}"


def _added_text(item_name: str, quantity: int) -> str:
    """Build 'Added X' or 'Added 2 X' text."""
    if quantity > 1 and item_name:
        return f"Added {quantity} {item_name}"
    if item_name:
        return f"{item_name} added"
    return "Added"


def _payload_value(payload: dict, key: str, default):
    if key in payload:
        return payload[key]
    return default


# ---------------------------------------------------------------------------
# Entity-feedback helper (also imported by response_builder)
# ---------------------------------------------------------------------------

def _build_entity_feedback(payload: dict) -> str:
    """Build spoken feedback for matched and unmatched entity names.

    Returns a prefix like:
        "Got Bacon and Mushrooms. I couldn't find tenderloin. "
        "Got Bacon. "
        ""
    """
    parts: list[str] = []

    matched = payload.get("matched_names") or []
    if matched:
        parts.append(f"Got {_format_names_with_counts(matched)}.")

    unmatched = [u for u in (payload.get("unmatched_names") or []) if _has_echable_content(u)]
    if unmatched:
        parts.append(f"I couldn't find {_format_selected_names(unmatched)}.")

    if not parts:
        return ""
    return " ".join(parts) + " "


# ---------------------------------------------------------------------------
# Group payload builder
# ---------------------------------------------------------------------------

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

    result = {
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
    # Propagate control fields from the outer handler payload so that
    # _progress_prompt can distinguish valid partial-selection reprompts
    # ("need_more") from genuine invalid-input reprompts ("invalid").
    for _ctrl_key in (
        "repeat_reason",
        "matched_names",
        "unmatched_names",
        "requested_names",
        "over_max",
        "dropped_names",
        # Side-group prompt metadata — pass through untouched.
        "side_group_position",
        "total_side_groups",
        "is_drink_group",
        "is_last_side_prompt",
        "speech_noun",
        # Modifier-group prompt metadata — pass through untouched.
        "modifier_group_position",
        "total_modifier_groups",
        "is_last_modifier_prompt",
        # Uncapped option count for overflow-hint decisions.
        "total_choices",
    ):
        if _ctrl_key in payload:
            result[_ctrl_key] = payload[_ctrl_key]
    return result


# ---------------------------------------------------------------------------
# Live-context payload extractors (used by sides and modifiers modules)
# ---------------------------------------------------------------------------

def _current_side_payload(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict | None = None,
) -> dict:
    payload = payload or {}
    item = menu_repo.store.get_item(context.current_item_id)
    group = item.side_groups[context.current_side_group_index]
    selected_ids = list(context.selected_side_groups.get(group.group_id, []))

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


def _current_modifier_payload(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict | None = None,
) -> dict:
    payload = payload or {}
    item = menu_repo.store.get_item(context.current_item_id)
    group = item.modifier_groups[context.current_modifier_group_index]
    selected_names = [
        _speak_modifier(
            selection.name,
            action=selection.action,
            instruction=selection.instruction,
        )
        for selection in context.selected_modifier_groups.get(group.group_id, [])
    ]

    return _group_payload(
        payload=payload,
        group_name=group.name,
        option_names=[choice.name for choice in group.choices],
        selected_names=selected_names,
        min_selector=int(group.min_selector or 0),
        max_selector=int(group.max_selector or 0),
    )


def _top_side_choices(
    context: ConversationContext,
    menu_repo: MenuRepository,
    k: int = 6,
) -> list[str]:
    item = menu_repo.store.get_item(context.current_item_id)
    group = item.side_groups[context.current_side_group_index]
    return [choice.name for choice in get_top_k_choices(group.choices, k=k)]


def _top_modifier_choices(
    context: ConversationContext,
    menu_repo: MenuRepository,
    k: int = 6,
) -> list[str]:
    item = menu_repo.store.get_item(context.current_item_id)
    group = item.modifier_groups[context.current_modifier_group_index]
    return [choice.name for choice in get_top_k_choices(group.choices, k=k)]


# ---------------------------------------------------------------------------
# Multi-select prompt builders
# ---------------------------------------------------------------------------

def _initial_multi_select_prompt(
    *,
    item_name: str,
    group_label: str,
    options: str,
    min_selector: int,
    max_selector: int,
    optional: bool,
    total_choices: int = 0,
) -> str:
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
        prompt = "Pick 1 more." if remaining == 1 else f"Pick {remaining} more."
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

    selected_count = max(int(payload.get("selected_count", 0) or 0), 0)
    if reason == "invalid":
        # Explicit invalid signal — use invalid_lead regardless of how many are selected.
        prompt = f"{invalid_lead} Pick 1 more." if remaining == 1 else f"{invalid_lead} Pick {remaining} more."
    elif selected_count == 0:
        prompt = "Please choose one option." if remaining == 1 else f"Please choose {remaining} options."
    else:
        # Valid partial selection: user already chose something; we just need more.
        # Do NOT use invalid_lead here — the selection was accepted.
        prompt = "Pick 1 more." if remaining == 1 else f"Pick {remaining} more."
    return f"{prompt} {options}." if options else prompt
