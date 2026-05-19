# app/nlu/turn_resolver/bucket_policy.py
"""Stateless bucket classification for GPT turn-resolution routing.

``pick_bucket()`` inspects the current state, local NLU result, and config
to decide which GPT resolution bucket (if any) should handle this turn.

Bucket precedence (highest → lowest):
  1. option_resolution (Bucket 2) — waiting states with failed local match
  2. multi_item_add_planning (Bucket 3) — idle with multi-item evidence
  3. idle_menu_item_resolution (Bucket 0) — idle with low-confidence/UNKNOWN

None is returned when no bucket applies (local deterministic path only).

All logic is purely classificatory — no GPT calls, no state mutation.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

from app.nlu.intent_resolution.intent import Intent
from app.state_machine.models.conversation_state import ConversationState

if TYPE_CHECKING:
    from app.config.semantic_repair import SemanticRepairConfig
    from app.nlu.nlu_result import SlotValue

# ---------------------------------------------------------------------------
# Bucket name constants
# ---------------------------------------------------------------------------

BUCKET_IDLE_ITEM = "idle_menu_item_resolution"   # Bucket 0
BUCKET_OPTION = "option_resolution"               # Bucket 2
BUCKET_MULTI_ITEM = "multi_item_add_planning"     # Bucket 3

# ---------------------------------------------------------------------------
# State sets
# ---------------------------------------------------------------------------

_WAITING_STATES: frozenset[ConversationState] = frozenset({
    ConversationState.WAITING_FOR_SIZE,
    ConversationState.WAITING_FOR_SIDE,
    ConversationState.WAITING_FOR_SIDE_SIZE,
    ConversationState.WAITING_FOR_MODIFIER,
})

_TERMINAL_STATES: frozenset[ConversationState] = frozenset({
    ConversationState.COMPLETED,
    ConversationState.ERROR_RECOVERY,
    ConversationState.TRANSFERRING_TO_HUMAN_AGENT,
})

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

# Confidence below which Bucket 0 considers the local result unreliable.
LOW_CONFIDENCE_THRESHOLD: float = 0.55

# Minimum text token count before Bucket 0 is eligible (avoids "um", "uh").
_MIN_ITEM_QUERY_TOKENS: int = 2

# ---------------------------------------------------------------------------
# Slot name sets
# ---------------------------------------------------------------------------

_ITEM_SLOT_NAMES: frozenset[str] = frozenset({"ITEM", "MENU_ITEM"})
_MODIFIER_SLOT_NAMES: frozenset[str] = frozenset({"MODIFIER"})
_SIDE_SLOT_NAMES: frozenset[str] = frozenset({"SIDE"})

# ---------------------------------------------------------------------------
# Complexity signals (same lexicon as add_item_planner_routing_policy)
# ---------------------------------------------------------------------------

_COMPLEXITY_WORDS: frozenset[str] = frozenset({
    "with", "without", "extra", "light",
})
_NO_WORD_PATTERN = re.compile(r"\bno\s+\w")
_CONJUNCTION = "and"
_NUMBER_WORDS: frozenset[str] = frozenset({
    "one", "two", "three", "four", "five",
    "six", "seven", "eight", "nine", "ten",
})
_DIGIT_PATTERN = re.compile(r"\b[2-9]\b")

# Noise-only tokens that alone do not constitute an item query.
_NOISE_TOKENS: frozenset[str] = frozenset({
    "um", "uh", "like", "just", "yes", "no", "ok", "okay", "sure",
    "hmm", "hmm", "hm", "er", "ah", "oh", "well", "hey",
})

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _slot_names(slots: "tuple[SlotValue, ...]") -> frozenset[str]:
    return frozenset(s.name.upper() for s in slots)


def _has_multi_item_signals(
    text: str,
    slots: "tuple[SlotValue, ...]",
) -> bool:
    """Return True when text / slots indicate multiple items or complex structure."""
    lower = (text or "").lower()
    if not lower.strip():
        return False

    words = set(re.findall(r"\b\w+\b", lower))

    if _COMPLEXITY_WORDS & words:
        return True
    if _NO_WORD_PATTERN.search(lower):
        return True
    if "," in text:
        return True
    if _CONJUNCTION in words:
        return True

    num_count = sum(1 for w in words if w in _NUMBER_WORDS)
    num_count += len(_DIGIT_PATTERN.findall(lower))
    if num_count >= 2:
        return True

    slot_names = _slot_names(slots)
    if _ITEM_SLOT_NAMES & slot_names and (
        _MODIFIER_SLOT_NAMES & slot_names or _SIDE_SLOT_NAMES & slot_names
    ):
        return True

    item_count = sum(1 for s in slots if s.name.upper() in _ITEM_SLOT_NAMES)
    return item_count >= 2


def _looks_like_item_query(
    text: str,
    slots: "tuple[SlotValue, ...]",
) -> bool:
    """Return True when text plausibly refers to a menu item (Bucket 0 guard).

    Filters out noise-only utterances and short control phrases that are
    unlikely to be item names.
    """
    lower = (text or "").strip().lower()
    if not lower:
        return False

    tokens = re.findall(r"\b\w+\b", lower)
    meaningful = [t for t in tokens if t not in _NOISE_TOKENS]

    if len(meaningful) < _MIN_ITEM_QUERY_TOKENS:
        return False

    # Any ITEM slot is direct evidence
    if any(s.name.upper() in _ITEM_SLOT_NAMES for s in slots):
        return True

    # Text looks like it could be a food item if it has >= 2 meaningful words
    return True


def _get_mode(config: "SemanticRepairConfig | None", attr: str, default: str = "disabled") -> str:
    """Safely read a mode attribute from config, falling back to default."""
    if config is None:
        return default
    return str(getattr(config, attr, default) or default)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def pick_bucket(
    state: ConversationState,
    local_intent: Intent,
    local_confidence: float,
    local_slots: "tuple[SlotValue, ...]",
    user_text: str,
    *,
    reprompt_count: int = 0,
    option_match_failed: bool = False,
    config: "SemanticRepairConfig | None" = None,
) -> str | None:
    """Return the bucket name that should handle this turn, or None.

    Parameters
    ----------
    state:
        Current conversation state from ``ConversationContext``.
    local_intent:
        Effective intent from local NLU (``NLUResult.effective_intent``).
    local_confidence:
        Intent confidence from local NLU (``NLUResult.intent_confidence``).
    local_slots:
        Slot values from local NLU (``NLUResult.slots``).
    user_text:
        Normalized user utterance for this turn.
    reprompt_count:
        Number of previous reprompts for the current field (from context).
    option_match_failed:
        True when the calling waiting-state handler's local matcher found
        no selections.  Required for Bucket 2 eligibility.
    config:
        ``SemanticRepairConfig`` snapshot.  If None, all buckets return None
        (safe default — disabled).

    Returns
    -------
    One of ``BUCKET_IDLE_ITEM``, ``BUCKET_OPTION``, ``BUCKET_MULTI_ITEM``,
    or None when no bucket applies.
    """
    # Terminal states are never eligible
    if state in _TERMINAL_STATES:
        return None

    # Empty / noise-only input is not eligible
    text = (user_text or "").strip()
    if not text:
        return None

    b0_mode = _get_mode(config, "bucket_0_mode")
    b2_mode = _get_mode(config, "bucket_2_mode")
    b3_mode = _get_mode(config, "bucket_3_mode")

    # ── Bucket 2: option_resolution ──────────────────────────────────────
    # Waiting state + local matcher found nothing.
    if state in _WAITING_STATES and option_match_failed and b2_mode != "disabled":
        return BUCKET_OPTION

    if state == ConversationState.IDLE:
        # ── Bucket 3: multi_item_add_planning ─────────────────────────────
        if b3_mode != "disabled" and _has_multi_item_signals(text, local_slots):
            return BUCKET_MULTI_ITEM

        # ── Bucket 0: idle_menu_item_resolution ───────────────────────────
        if b0_mode != "disabled":
            is_low_conf = (
                local_intent == Intent.UNKNOWN
                or local_confidence < LOW_CONFIDENCE_THRESHOLD
            )
            if is_low_conf and _looks_like_item_query(text, local_slots):
                return BUCKET_IDLE_ITEM

    return None
