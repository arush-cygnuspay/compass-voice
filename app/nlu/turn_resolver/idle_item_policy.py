# app/nlu/turn_resolver/idle_item_policy.py
"""Trigger policy for bucket-0 idle natural item resolution.

should_call_idle_item_resolver() decides whether a given idle-state turn
warrants a GPT call to interpret a bare menu item phrase.

Returns (should_call: bool, reason: str).

Safety contract
---------------
* Pure functions only — no GPT calls, no context mutation.
* Never raises.
* Conservative: does NOT trigger when there is strong evidence of a
  non-item intent (checkout / cancel / payment).
"""
from __future__ import annotations

import re

# ── States eligible for idle item resolution ──────────────────────────────────

_IDLE_STATES: frozenset[str] = frozenset({"idle"})

# ── Thresholds ────────────────────────────────────────────────────────────────

# add_item confidence below which GPT is considered useful even for ADD_ITEM.
_HIGH_CONF_LOCAL_THRESHOLD: float = 0.85

# ── Regex patterns ────────────────────────────────────────────────────────────

# Explicit "I want / add / order / give me" preambles — when present the local
# NLU already has strong signal.  We still trigger GPT for low-confidence cases.
_EXPLICIT_ADD_RE = re.compile(
    r"\b(i\s+want|i\s+'d?\s+like|add|order|give\s+me|can\s+i\s+(have|get)|"
    r"let\s+me\s+(get|have)|i\s+('?ll\s+)?take|i\s+need)\b",
    re.IGNORECASE,
)

# Terminal/control phrases that are clearly NOT item orders.
_CHECKOUT_RE = re.compile(
    r"\b(checkout|check\s+out|pay|payment|that('?s|\s+is)\s+all|"
    r"i('?m|\s+am)\s+(done|finished|ready)|place\s+(my\s+)?order|"
    r"confirm\s+(my\s+)?order|submit|finalize)\b",
    re.IGNORECASE,
)
_CANCEL_RE = re.compile(
    r"\b(cancel|never\s+mind|nevermind|start\s+over|forget\s+it|"
    r"stop|exit|quit|hang\s+up)\b",
    re.IGNORECASE,
)

# Size/variant signal words that strongly suggest a food item utterance.
_SIZE_VARIANT_RE = re.compile(
    r"\b(small|medium|large|xl|extra\s+large|"
    r"six\s+piece|6\s+piece|8\s+piece|twelve\s+piece|12\s+piece|"
    r"combo|meal|platter)\b",
    re.IGNORECASE,
)

# Compound utterance signals.
_WITH_WORD_RE = re.compile(r"\bwith\b", re.IGNORECASE)

# Previous bot prompts that indicate an ordering context.
_ORDER_PROMPT_RE = re.compile(
    r"\b(what\s+(can|would|will)\s+(you|i)|"
    r"what\s+(would|do)\s+you\s+(like|want)|"
    r"what\s+are\s+you\s+(having|ordering|getting)|"
    r"go\s+ahead|ready\s+to\s+order|what('?s|\s+is)\s+your\s+order)\b",
    re.IGNORECASE,
)

# Slot type names treated as item evidence.
_ITEM_SLOT_NAMES: frozenset[str] = frozenset({"ITEM", "MENU_ITEM"})
_SIZE_SLOT_NAMES: frozenset[str] = frozenset({"SIZE", "VARIANT"})
_SIDE_SLOT_NAMES: frozenset[str] = frozenset({"SIDE"})

# Noise-only tokens that alone are not item evidence.
_NOISE_TOKENS: frozenset[str] = frozenset({
    "um", "uh", "like", "just", "yes", "no", "ok", "okay", "sure",
    "hmm", "hm", "er", "ah", "oh", "well", "hey", "please",
})


# ── Public API ────────────────────────────────────────────────────────────────


def should_call_idle_item_resolver(
    *,
    state: str,
    user_text: str,
    normalized_text: str,
    local_intent: str | None,
    local_confidence: float | None,
    local_slots: list | tuple | None,
    menu_candidates: list | tuple | None,
    previous_assistant_prompt: str | None,
) -> tuple[bool, str]:
    """Decide whether to invoke bucket-0 GPT for an idle-state turn.

    Returns
    -------
    (True, reason)  — GPT should be called; ``reason`` is a log-friendly code.
    (False, reason) — GPT should NOT be called; ``reason`` explains why.

    Decision order (first match wins):
    1.  state is not idle                          → False, "not_idle_state"
    2.  empty / noise-only text                    → False, "empty_text"
    3.  clear checkout phrase                      → False, "checkout_phrase"
    4.  clear cancel / handoff phrase              → False, "cancel_phrase"
    5.  high-confidence add_item intent            → False, "high_confidence_local"
    6.  previous prompt asked what to order        → True,  "previous_prompt_asked_order"
    7.  has menu candidate AND bare item phrase    → True,  "bare_menu_phrase"
    8.  has menu candidate in idle                 → True,  "menu_candidate_in_idle"
    9.  size/variant word + menu candidate         → True,  "size_or_variant_menu_phrase"
    10. "with" conjunction + menu candidate        → True,  "with_related_item_phrase"
    11. low-confidence add_item intent             → True,  "low_confidence_add_item"
    12. item slots with non-add_item intent        → True,  "item_slots_with_non_add_intent"
    Otherwise                                      → False, "no_trigger"
    """
    # 1. State must be idle
    normalized_state = (state or "").lower().strip()
    if normalized_state not in _IDLE_STATES:
        return False, "not_idle_state"

    # 2. Empty / noise-only text
    text = (user_text or "").strip()
    norm = (normalized_text or text).strip()
    if not _is_meaningful(text):
        return False, "empty_text"

    # 3. Clear checkout phrase
    if _CHECKOUT_RE.search(text):
        return False, "checkout_phrase"

    # 4. Clear cancel / handoff phrase
    if _CANCEL_RE.search(text):
        return False, "cancel_phrase"

    # 5. High-confidence local add_item — GPT not needed
    intent_upper = (local_intent or "").upper().strip()
    conf = float(local_confidence or 0.0)
    if intent_upper == "ADD_ITEM" and conf >= _HIGH_CONF_LOCAL_THRESHOLD:
        return False, "high_confidence_local"

    has_candidates = bool(menu_candidates)

    # 6. Previous prompt asked what to order
    prev_prompt = (previous_assistant_prompt or "").strip()
    if prev_prompt and _ORDER_PROMPT_RE.search(prev_prompt):
        return True, "previous_prompt_asked_order"

    # 7. Has menu candidate AND text is a bare item phrase (no explicit preamble)
    if has_candidates and not _EXPLICIT_ADD_RE.search(text):
        return True, "bare_menu_phrase"

    # 8. Has menu candidate in idle (with any preamble)
    if has_candidates:
        return True, "menu_candidate_in_idle"

    # 9. Size/variant word + menu candidate (candidate computed from a token)
    if _SIZE_VARIANT_RE.search(text) and has_candidates:
        return True, "size_or_variant_menu_phrase"

    # 10. "with" conjunction + menu candidate
    if _WITH_WORD_RE.search(text) and has_candidates:
        return True, "with_related_item_phrase"

    # 11. Low-confidence add_item
    if intent_upper == "ADD_ITEM" and conf < _HIGH_CONF_LOCAL_THRESHOLD:
        return True, "low_confidence_add_item"

    # 12. Item-type slots with wrong intent
    if _has_item_slots(local_slots) and intent_upper not in {"ADD_ITEM", ""}:
        return True, "item_slots_with_non_add_intent"

    return False, "no_trigger"


# ── Private helpers ───────────────────────────────────────────────────────────


def _is_meaningful(text: str) -> bool:
    """Return True when text has at least one non-noise token."""
    tokens = re.findall(r"\b\w+\b", (text or "").lower())
    meaningful = [t for t in tokens if t not in _NOISE_TOKENS]
    return len(meaningful) >= 1


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
