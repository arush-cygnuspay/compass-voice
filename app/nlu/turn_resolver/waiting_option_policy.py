# app/nlu/turn_resolver/waiting_option_policy.py
"""Trigger policy for bucket-2 waiting-state GPT option resolution.

should_call_waiting_option_gpt() decides whether a given turn in a
waiting state (modifier/side/size/side_size) warrants a GPT call.
The existing deterministic exact/fuzzy match is always the fast path;
this policy is consulted only when that path needs help.

Returns (should_call: bool, reason: str).

Safety contract
---------------
* Pure functions only — no GPT calls, no context mutation.
* Never raises.
"""
from __future__ import annotations

import re

# ── States that may trigger bucket-2 resolution ──────────────────────────────

_WAITING_STATES: frozenset[str] = frozenset({
    "waiting_for_modifier",
    "waiting_for_side",
    "waiting_for_size",
    "waiting_for_side_size",
})

# ── Slot names that indicate the user is naming a menu item (not a modifier/side/size) ─

_ITEM_SLOT_NAMES: frozenset[str] = frozenset({"ITEM", "MENU_ITEM"})

# ── Regex patterns ────────────────────────────────────────────────────────────

# Ordinal references: "the second one", "first", "third option", …
_ORDINAL_RE = re.compile(
    r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|last"
    r"|1st|2nd|3rd|4th|5th|6th|7th|8th)\b",
    re.IGNORECASE,
)

# Option-list requests: "what do you have", "list options", "options", …
_OPTION_LIST_RE = re.compile(
    r"\b("
    r"what\s+do\s+you\s+have"
    r"|what\s+are\s+(my|the)\s+options?"
    r"|list\s+(the\s+)?options?"
    r"|options?"
    r"|choices?"
    r"|what\s+can\s+i\s+(get|have|choose)"
    r")\b",
    re.IGNORECASE,
)

# Contextual / anaphoric references: "that one", "do it", …
_CONTEXTUAL_RE = re.compile(
    r"\b("
    r"that\s+one"
    r"|do\s+it"
    r"|go\s+with\s+that"
    r"|i\s+said\s+that"
    r"|yeah\s+that"
    r"|the\s+(first|second|third|fourth|last)\s+one"
    r")\b",
    re.IGNORECASE,
)

# Negation: "no coke", "not X", "without X", "nope", …
_NEGATION_RE = re.compile(
    r"(?:^|\s)(no|nope|nah|not|without|none)\b",
    re.IGNORECASE,
)

# ── Policy thresholds ─────────────────────────────────────────────────────────

_LOW_CONFIDENCE_THRESHOLD: float = 0.70
_REPROMPT_THRESHOLD: int = 1  # trigger GPT after N consecutive failures


# ── Public API ────────────────────────────────────────────────────────────────


def should_call_waiting_option_gpt(
    *,
    state: str,
    user_text: str,
    local_intent: str | None,
    local_confidence: float | None,
    local_slots: list | tuple | None,
    deterministic_match_result: object = None,
    reprompt_count: int = 0,
) -> tuple[bool, str]:
    """Decide whether to invoke bucket-2 GPT for a waiting-state turn.

    Returns
    -------
    (True, reason)  — GPT should be called; ``reason`` is a log-friendly code.
    (False, reason) — GPT should NOT be called; ``reason`` explains why.

    Decision order (first match wins):
    1.  state not in waiting states   → False, "not_waiting_state"
    2.  empty / silence utterance     → False, "empty_text"
    3.  deterministic success         → False, "deterministic_success"
    4.  ordinal phrase                → True,  "ordinal_phrase"
    5.  options-list request          → True,  "options_list_request"
    6.  contextual phrase             → True,  "contextual_phrase"
    7.  negation phrase               → True,  "negation_phrase"
    8.  deterministic no-match        → True,  "deterministic_no_match"
    9.  unknown / missing intent      → True,  "unknown_intent"
    10. low confidence (<0.70)        → True,  "low_confidence"
    11. intent invalid for state      → True,  "invalid_intent_for_state"
    12. item slots in waiting state   → True,  "item_slots_in_waiting_state"
    13. reprompt_count >= threshold   → True,  "reprompt_count_threshold"
    Otherwise                         → False, "no_trigger"
    """
    # 1. Not a waiting state
    normalized_state = (state or "").lower().strip()
    if normalized_state not in _WAITING_STATES:
        return False, "not_waiting_state"

    # 2. Empty / silence
    text = (user_text or "").strip()
    if not text:
        return False, "empty_text"

    # 3. Deterministic success
    if _is_deterministic_success(deterministic_match_result):
        return False, "deterministic_success"

    # 4. Ordinal phrase
    if _ORDINAL_RE.search(text):
        return True, "ordinal_phrase"

    # 5. Options-list request
    if _OPTION_LIST_RE.search(text):
        return True, "options_list_request"

    # 6. Contextual / anaphoric reference
    if _CONTEXTUAL_RE.search(text):
        return True, "contextual_phrase"

    # 7. Negation phrase
    if _NEGATION_RE.search(text):
        return True, "negation_phrase"

    # 8. Deterministic no-match
    if _is_deterministic_no_match(deterministic_match_result):
        return True, "deterministic_no_match"

    # 9. Unknown / missing intent
    intent_upper = (local_intent or "").upper().strip()
    if not intent_upper or intent_upper in {"UNKNOWN", "NONE"}:
        return True, "unknown_intent"

    # 10. Low confidence
    conf = float(local_confidence or 0.0)
    if conf < _LOW_CONFIDENCE_THRESHOLD:
        return True, "low_confidence"

    # 11. Intent invalid for this waiting state
    if _intent_invalid_for_state(intent_upper, normalized_state):
        return True, "invalid_intent_for_state"

    # 12. Item-type slots present in modifier/side/size state
    if _has_item_slots(local_slots):
        return True, "item_slots_in_waiting_state"

    # 13. Reprompt threshold reached
    if reprompt_count >= _REPROMPT_THRESHOLD:
        return True, "reprompt_count_threshold"

    return False, "no_trigger"


# ── Private helpers ───────────────────────────────────────────────────────────


def _is_deterministic_success(result: object) -> bool:
    """Return True if *result* indicates a successful deterministic match."""
    if result is None:
        return False
    # OptionResolverResult / ModifierGroupResolver result pattern
    if hasattr(result, "selections") and result.selections:
        return True
    if hasattr(result, "matched_item_ids") and result.matched_item_ids:
        return True
    if isinstance(result, bool):
        return result
    if isinstance(result, dict):
        return bool(
            result.get("matched")
            or result.get("selections")
            or result.get("matched_item_ids")
        )
    return False


def _is_deterministic_no_match(result: object) -> bool:
    """Return True if *result* indicates the deterministic path found no match."""
    if result is None:
        return True  # no result object at all → treat as no-match
    if hasattr(result, "selections") and not result.selections:
        return True
    if hasattr(result, "matched_item_ids") and not result.matched_item_ids:
        return True
    if isinstance(result, bool):
        return not result
    return False


def _intent_invalid_for_state(intent: str, state: str) -> bool:
    """Return True when the local intent is not appropriate for the waiting state."""
    if state == "waiting_for_modifier":
        valid: frozenset[str] = frozenset({"ADD_MODIFIER", "SELECT_MODIFIER", "ADD_ITEM", "MODIFY_ITEM"})
    elif state == "waiting_for_side":
        valid = frozenset({"SELECT_SIDE", "ADD_SIDE", "ADD_ITEM"})
    elif state in {"waiting_for_size", "waiting_for_side_size"}:
        valid = frozenset({"SELECT_SIZE", "ADD_SIZE", "SELECT_VARIANT"})
    else:
        return False
    return bool(intent) and intent not in valid


def _has_item_slots(local_slots: list | tuple | None) -> bool:
    """Return True if any slot is an item-type slot (ITEM / MENU_ITEM)."""
    if not local_slots:
        return False
    for slot in local_slots:
        name = ""
        if isinstance(slot, dict):
            name = str(slot.get("n") or slot.get("name") or "").upper()
        elif hasattr(slot, "name"):
            name = str(slot.name).upper()
        if name in _ITEM_SLOT_NAMES:
            return True
    return False
