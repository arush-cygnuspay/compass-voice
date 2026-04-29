# app/state_machine/flow_sets/intent_policy.py
"""Intent behavior policy and group-selection signal sets.

Import from here — or from the parent package ``app.state_machine.flow_sets`` —
to get intent groupings and phrase-matching helpers.  This module is the single
source of truth for which intents trigger soft-switches, group-done signals,
and delivery gating pass-throughs.
"""
from __future__ import annotations

from app.nlu.intent_resolution.intent import Intent
from app.nlu.query_normalization.text_preprocessor import normalize_text


# ─────────────────────────────────────────────────────────────────────────────
# INTENT GROUPINGS
# ─────────────────────────────────────────────────────────────────────────────

# Intents that signal "I'm done ordering / let me checkout" — in group selection
# states (side/modifier), these mean "done with this group", NOT "interrupt item".
GROUP_DONE_INTENTS: set[Intent] = {
    Intent.END_ADDING,
    Intent.CHECKOUT,
    Intent.FINISH_ORDER,
    Intent.CONFIRM_ORDER,
    Intent.REVIEW_ORDER,
}

# Ordering intents that should be redirected (not consumed) when user is in
# pre-order states like order-type, delivery eligibility, or address collection.
ORDERING_INTENTS: set[Intent] = {
    Intent.ADD_ITEM,
    Intent.REMOVE_ITEM,
    Intent.MODIFY_ITEM,
    Intent.SHOW_MENU,
    Intent.ASK_MENU_INFO,
    Intent.ASK_PRICE,
    Intent.SHOW_CART,
    Intent.SHOW_TOTAL,
}

# Full set of intents that trigger a soft-switch (cancellation confirmation)
# when the user is mid-item-flow.
SOFT_SWITCH_INTENTS: set[Intent] = {
    Intent.ADD_ITEM,
    Intent.REMOVE_ITEM,
    Intent.MODIFY_ITEM,
    Intent.SHOW_MENU,
    Intent.ASK_MENU_INFO,
    Intent.ASK_PRICE,
    Intent.SHOW_CART,
    Intent.SHOW_TOTAL,
    Intent.START_ORDER,
    Intent.END_ADDING,
    Intent.CHECKOUT,
    Intent.CONFIRM_ORDER,
    Intent.FINISH_ORDER,
    Intent.REVIEW_ORDER,
    Intent.PAYMENT_REQUEST,
    Intent.CANCEL_ORDER,
}

# Reduced soft-switch set for side/modifier handlers — excludes GROUP_DONE_INTENTS
# because those are intercepted earlier as "done with this group" signals.
SOFT_SWITCH_INTENTS_REDUCED: set[Intent] = SOFT_SWITCH_INTENTS - GROUP_DONE_INTENTS

# Intents allowed through during generic waiting states (side/modifier/size/qty).
WAITING_STATE_ALLOWED_CONTROL_INTENTS: set[Intent] = {
    Intent.DENY,
    Intent.CANCEL,
    Intent.CANCEL_ORDER,
    Intent.ASK_OPTIONS,
    Intent.SHOW_CART,
    Intent.SHOW_TOTAL,
    Intent.ASK_PRICE,
    Intent.ASK_ITEM_INFO,
    Intent.ASK_MENU_INFO,
    Intent.AVAILABILITY_QUERY,
    Intent.BROWSE_MENU,
    Intent.BROWSE_CATEGORY,
    Intent.RECOMMENDATION_QUERY,
    Intent.SHOW_MENU,
    Intent.ADD_ITEM,
    Intent.REMOVE_ITEM,
    Intent.MODIFY_ITEM,
}

# Intents allowed through during delivery gating states.
DELIVERY_GATING_ALLOWED_CONTROL_INTENTS: set[Intent] = {
    Intent.AFFIRM,
    Intent.CONFIRM,
    Intent.DENY,
    Intent.CANCEL,
    Intent.CANCEL_ORDER,
    Intent.ADD_ITEM,
    Intent.REMOVE_ITEM,
    Intent.MODIFY_ITEM,
    Intent.SHOW_MENU,
    Intent.ASK_MENU_INFO,
    Intent.ASK_PRICE,
    Intent.SHOW_CART,
    Intent.SHOW_TOTAL,
}


# ─────────────────────────────────────────────────────────────────────────────
# GROUP SELECTION WORD SETS (side/modifier handlers)
# ─────────────────────────────────────────────────────────────────────────────

DONE_WORDS: set[str] = {
    "done",
    "i am done",
    "i said done",
    "done and done",
    "thats all",
    "that is all",
    "thats it",
    "that is it",
    "finished",
    "i am finished",
    "im finished",
    "continue",
    "next",
    "no more",
    "nothing else",
    "im good",
    "i dont want anymore",
    "i dont want any more",
    "thats enough",
    "im done",
    "all good",
    "good",
    "nah thats it",
    "thats good",
    "were good",
    "that should do it",
    "thatll do it",
    "that will do it",
    "should be good",
    "should be enough",
    "sounds good",
}

SKIP_WORDS: set[str] = {
    "no",
    "none",
    "nothing",
    "skip",
    "skip it",
    "no thanks",
    "leave it",
    "leave it off",
    "i am good",
}

MORE_OPTIONS_WORDS: set[str] = {
    "other options",
    "more options",
    "what else",
    "what else do you have",
    "what else you got",
    "what are the options",
    "what can i choose",
    "tell me the choices",
    "what do you have",
    "list options",
    "available toppings",
    "next options",
    "show me more",
    "any others",
    "anything else available",
    "what are my options",
    "options",
}

_SIGNAL_LEADING_FILLERS: tuple[str, ...] = (
    "no",
    "nah",
    "nope",
    "yeah",
    "yep",
    "yup",
    "ok",
    "okay",
    "alright",
    "all right",
    "well",
    "so",
    "just",
    "please",
)

_SIGNAL_TRAILING_FILLERS: tuple[str, ...] = (
    "please",
    "thanks",
    "thank you",
    "for now",
)


def _strip_signal_wrappers(text: str) -> str:
    value = normalize_text(text)
    if not value:
        return ""

    changed = True
    while changed and value:
        changed = False

        for phrase in _SIGNAL_LEADING_FILLERS:
            prefix = f"{phrase} "
            if value.startswith(prefix):
                value = value[len(prefix):].strip()
                changed = True
                break

        if changed:
            continue

        for phrase in _SIGNAL_TRAILING_FILLERS:
            suffix = f" {phrase}"
            if value.endswith(suffix):
                value = value[: -len(suffix)].strip()
                changed = True
                break

    return value


def _signal_candidates(text: str) -> set[str]:
    normalized = normalize_text(text)
    if not normalized:
        return set()

    candidates: set[str] = {normalized}
    queue: list[str] = [normalized]
    seen: set[str] = set()

    while queue:
        value = queue.pop()
        if not value or value in seen:
            continue
        seen.add(value)
        candidates.add(value)

        for phrase in _SIGNAL_LEADING_FILLERS:
            prefix = f"{phrase} "
            if value.startswith(prefix):
                queue.append(value[len(prefix):].strip())

        for phrase in _SIGNAL_TRAILING_FILLERS:
            suffix = f" {phrase}"
            if value.endswith(suffix):
                queue.append(value[: -len(suffix)].strip())

        if " and " in value:
            parts = [part.strip() for part in value.split(" and ") if part.strip()]
            queue.extend(parts)
            if len(parts) >= 2 and len(set(parts)) == 1:
                queue.append(parts[0])

    return candidates


def _matches_signal_phrase(text: str, phrases: set[str]) -> bool:
    candidates = _signal_candidates(text)
    return any(candidate in phrases for candidate in candidates if candidate)


def looks_like_done_answer(text: str) -> bool:
    return _matches_signal_phrase(text, DONE_WORDS)


def looks_like_skip_answer(text: str) -> bool:
    return _matches_signal_phrase(text, SKIP_WORDS)


def looks_like_more_options_answer(text: str) -> bool:
    return _matches_signal_phrase(text, MORE_OPTIONS_WORDS)


__all__ = [
    "GROUP_DONE_INTENTS",
    "ORDERING_INTENTS",
    "SOFT_SWITCH_INTENTS",
    "SOFT_SWITCH_INTENTS_REDUCED",
    "WAITING_STATE_ALLOWED_CONTROL_INTENTS",
    "DELIVERY_GATING_ALLOWED_CONTROL_INTENTS",
    "DONE_WORDS",
    "SKIP_WORDS",
    "MORE_OPTIONS_WORDS",
    "looks_like_done_answer",
    "looks_like_skip_answer",
    "looks_like_more_options_answer",
]
