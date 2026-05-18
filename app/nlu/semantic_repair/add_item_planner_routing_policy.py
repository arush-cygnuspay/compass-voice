# app/nlu/semantic_repair/add_item_planner_routing_policy.py
"""Phase 4 GPT Add-Item Planner routing policy.

Decides whether and how to invoke the GPT add-item planner for a given
customer utterance entering the IDLE / add-item flow.

Route modes
-----------
NO_GPT      — do not call GPT (mode disabled, simple utterance, or no evidence)
SHADOW_GPT  — call GPT for logging/training only; result is NEVER applied
INLINE_GPT  — call GPT; apply result when apply gate approves

Complexity signals (any one triggers "complex" classification)
--------------------------------------------------------------
- "with" / "without" present           — modifier attachment signal
- "extra" / "light" present            — quantity modifier signal
- "no" as standalone word before noun  — negated modifier signal
- Comma in utterance                   — list / multi-entity signal
- "and" word present                   — conjunction / multi-item signal
- Multiple number words (two X, one Y) — multi-quantity signal
- ITEM slot + MODIFIER or SIDE slot    — NLU-detected complex grouping
- Multiple ITEM slots detected         — multi-item evidence

Simple high-confidence bypass (prevents unnecessary GPT calls)
--------------------------------------------------------------
- local intent == ADD_ITEM
- local confidence >= _SIMPLE_CONFIDENCE_THRESHOLD (0.85)
- No complexity signals detected
- Only one ITEM slot, no MODIFIER/SIDE slot
  → NO_GPT — local deterministic path is sufficient
"""
from __future__ import annotations

import re
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.config.semantic_repair import SemanticRepairConfig

# ---------------------------------------------------------------------------
# Complexity detection constants
# ---------------------------------------------------------------------------

# Words whose presence in the utterance signals modifier/side attachment or
# negation — always trigger complexity even in short utterances.
_COMPLEXITY_WORDS: frozenset[str] = frozenset({
    "with",
    "without",
    "extra",
    "light",
})

# "no" triggers complexity when followed by a word (e.g. "no onions").
# Bare "no" as the entire utterance is an answer, not a modifier signal.
_NO_WORD_PATTERN = re.compile(r"\bno\s+\w")

# Conjunction that may link multiple item entities.
_CONJUNCTION = "and"

# English number words that indicate quantity.
_NUMBER_WORDS: frozenset[str] = frozenset({
    "one", "two", "three", "four", "five",
    "six", "seven", "eight", "nine", "ten",
})

# Regex for digit quantities (1-9 at word boundaries).
_DIGIT_PATTERN = re.compile(r"\b[2-9]\b")

# Local intent values that indicate add-item intent (case-insensitive).
_ADD_ITEM_INTENTS: frozenset[str] = frozenset({
    "add_item", "ADD_ITEM",
})

# Slot names that indicate item evidence.
_ITEM_SLOT_NAMES: frozenset[str] = frozenset({"ITEM", "MENU_ITEM"})
_MODIFIER_SLOT_NAMES: frozenset[str] = frozenset({"MODIFIER"})
_SIDE_SLOT_NAMES: frozenset[str] = frozenset({"SIDE"})

# Confidence threshold below which we always try the planner in complex cases.
# Above this threshold AND simple utterance → skip GPT.
_SIMPLE_CONFIDENCE_THRESHOLD: float = 0.85


# ---------------------------------------------------------------------------
# Route mode enum
# ---------------------------------------------------------------------------


class AddItemPlannerRouteMode(str, Enum):
    """Route decision for a single add-item planner invocation."""

    NO_GPT = "no_gpt"
    """GPT is not called — simple utterance, mode disabled, or no evidence."""

    SHADOW_GPT = "shadow_gpt"
    """GPT is called but result is always logged-only; never applied."""

    INLINE_GPT = "inline_gpt"
    """GPT is called; result applied when apply gate approves."""


# ---------------------------------------------------------------------------
# Complexity detection
# ---------------------------------------------------------------------------


def is_complex_utterance(
    text: str,
    local_slots: list[dict] | None = None,
) -> bool:
    """Return True when *text* has structure that benefits from GPT planning.

    Evaluates both lexical signals in the text and NLU slot evidence.
    Stateless — safe to call from any context.
    """
    lower = (text or "").lower()
    if not lower.strip():
        return False

    words = set(re.findall(r"\b\w+\b", lower))

    # Modifier/side attachment keywords
    if _COMPLEXITY_WORDS & words:
        return True

    # "no <noun>" pattern (negated modifier), but not bare "no"
    if _NO_WORD_PATTERN.search(lower):
        return True

    # Comma → list structure
    if "," in text:
        return True

    # Conjunction word
    if _CONJUNCTION in words:
        return True

    # Multiple number words → multi-quantity / multi-item
    num_count = sum(1 for w in words if w in _NUMBER_WORDS)
    num_count += len(_DIGIT_PATTERN.findall(lower))
    if num_count >= 2:
        return True

    # NLU slot evidence
    slots = local_slots or []
    slot_names = {s.get("n", "").upper() for s in slots}

    # ITEM + MODIFIER or SIDE → complex grouping
    if _ITEM_SLOT_NAMES & slot_names and (
        _MODIFIER_SLOT_NAMES & slot_names or _SIDE_SLOT_NAMES & slot_names
    ):
        return True

    # Multiple ITEM slots → multi-item
    item_slot_count = sum(
        1 for s in slots if s.get("n", "").upper() in _ITEM_SLOT_NAMES
    )
    if item_slot_count >= 2:
        return True

    return False


def _has_item_evidence(
    local_intent: str | None,
    local_slots: list[dict] | None,
    item_candidates_exist: bool,
) -> bool:
    """Return True when there is evidence that the utterance concerns menu items."""
    if item_candidates_exist:
        return True
    intent = (local_intent or "").upper()
    if intent in {i.upper() for i in _ADD_ITEM_INTENTS} or intent == "UNKNOWN":
        return True
    slots = local_slots or []
    return any(s.get("n", "").upper() in _ITEM_SLOT_NAMES for s in slots)


def _is_simple_high_confidence(
    text: str,
    local_intent: str | None,
    local_confidence: float,
    local_slots: list[dict] | None,
) -> bool:
    """Return True for simple, high-confidence add-item utterances.

    When True, the planner routing returns NO_GPT because the local
    deterministic path handles the utterance correctly on its own.
    """
    if (local_intent or "").upper() not in {i.upper() for i in _ADD_ITEM_INTENTS}:
        return False
    if local_confidence < _SIMPLE_CONFIDENCE_THRESHOLD:
        return False
    if is_complex_utterance(text, local_slots):
        return False
    # Allow only a single ITEM slot and no modifier/side slots
    slots = local_slots or []
    slot_names = [s.get("n", "").upper() for s in slots]
    item_count = sum(1 for n in slot_names if n in _ITEM_SLOT_NAMES)
    has_mod_or_side = any(
        n in _MODIFIER_SLOT_NAMES | _SIDE_SLOT_NAMES for n in slot_names
    )
    return item_count <= 1 and not has_mod_or_side


# ---------------------------------------------------------------------------
# Routing policy
# ---------------------------------------------------------------------------


class GptAddItemPlannerRoutingPolicy:
    """Stateless policy: inspect config + turn context, return route mode.

    Thread-safe: no mutable state.  Instantiate once and share across turns.
    """

    def decide(
        self,
        *,
        config: "SemanticRepairConfig",
        user_text: str,
        local_intent: str | None = None,
        local_confidence: float = 0.0,
        local_slots: list[dict] | None = None,
        item_candidates_exist: bool = False,
    ) -> tuple[AddItemPlannerRouteMode, str]:
        """Return (route_mode, reason_code) for the current turn.

        Parameters
        ----------
        config:
            Current SemanticRepairConfig snapshot.
        user_text:
            Normalized customer utterance.
        local_intent:
            Local NLU intent label (e.g. "add_item", "UNKNOWN").
        local_confidence:
            Local NLU intent confidence (0.0–1.0).
        local_slots:
            Local NLU slot list; each entry is {"n": name, "v": value}.
        item_candidates_exist:
            True when menu candidates were already resolved for this utterance.

        Returns
        -------
        (AddItemPlannerRouteMode, reason_str)
        """
        # ── Global guard 1: empty / silence text ─────────────────────────
        if not (user_text or "").strip():
            return AddItemPlannerRouteMode.NO_GPT, "empty_text"

        # ── Global guard 2: mode disabled ────────────────────────────────
        mode = getattr(config, "add_item_planner_mode", "disabled")
        if mode == "disabled":
            return AddItemPlannerRouteMode.NO_GPT, "mode_disabled"

        # ── Simple high-confidence bypass ────────────────────────────────
        if _is_simple_high_confidence(
            user_text, local_intent, local_confidence, local_slots
        ):
            return AddItemPlannerRouteMode.NO_GPT, "simple_high_confidence_local"

        # ── Complexity check ─────────────────────────────────────────────
        complex_utterance = is_complex_utterance(user_text, local_slots)
        has_evidence = _has_item_evidence(
            local_intent, local_slots, item_candidates_exist
        )

        if not complex_utterance and not has_evidence:
            return AddItemPlannerRouteMode.NO_GPT, "no_complexity_no_evidence"

        # ── Mode-specific routing ─────────────────────────────────────────
        if mode == "shadow":
            # In shadow mode trigger on complexity OR item evidence
            if complex_utterance or has_evidence:
                return AddItemPlannerRouteMode.SHADOW_GPT, "shadow_complex_or_evidence"
            return AddItemPlannerRouteMode.NO_GPT, "shadow_no_trigger"

        if mode == "inline":
            # In inline mode require complexity for GPT to add value
            if complex_utterance:
                return AddItemPlannerRouteMode.INLINE_GPT, "inline_complex"
            if has_evidence and not complex_utterance:
                # Evidence but not complex — shadow observation is safer
                return AddItemPlannerRouteMode.NO_GPT, "inline_not_complex"

        return AddItemPlannerRouteMode.NO_GPT, "no_trigger"
