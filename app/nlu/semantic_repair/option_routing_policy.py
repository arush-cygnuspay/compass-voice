# app/nlu/semantic_repair/option_routing_policy.py
"""Phase 3 GPT option resolver routing policy.

Decides whether and how to invoke the GPT option resolver when local
deterministic modifier matching fails in WAITING_FOR_MODIFIER (and
future option states).

Route modes
-----------
NO_GPT      — do not call GPT (mode disabled, local succeeded, or conditions not met)
SHADOW_GPT  — call GPT for logging/training only; result is NEVER applied
INLINE_GPT  — call GPT; apply result when safe (safe_to_apply=True from validator)

Routing rules (evaluated in order)
-----------------------------------
Global guards (all modes):
  1. Empty / silence text                           → NO_GPT
  2. config.option_resolver_mode == "disabled"      → NO_GPT

Mode-specific rules:
  3. mode == "shadow":
       local_resolved                               → NO_GPT (no value in redundant call)
       options_exist AND not local_resolved         → SHADOW_GPT
  4. mode == "inline":
       repeat_count >= repeat_threshold             → INLINE_GPT  (repeat-loop recovery)
       not local_resolved AND text_len >= 3
         AND options_exist                          → INLINE_GPT
       not local_resolved AND has_correction_signal
         AND options_exist                          → INLINE_GPT  (self-correction signal)
       otherwise                                    → NO_GPT
"""
from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.config.semantic_repair import SemanticRepairConfig

# ---------------------------------------------------------------------------
# Correction-signal detection
# ---------------------------------------------------------------------------

# Words / phrases at the START of an utterance that signal user self-correction.
# Intentionally conservative: "no" and "not" are excluded because they are
# legitimate negated-modifier prefixes ("no onions", "not the american cheese").
# Only phrases that unambiguously mean "I want to correct what I just said"
# are included here.
_CORRECTION_STARTERS: frozenset[str] = frozenset({
    "actually",
    "i mean",
    "i meant",
    "instead",
    "wait,",
    "wait ",
    "no wait",
    "scratch that",
    "never mind i meant",
    "correction,",
    "correction ",
})


def has_correction_signal(text: str) -> bool:
    """Return True when *text* starts with a self-correction phrase.

    Used by the routing policy to escalate to INLINE_GPT even for short
    utterances, because a correction like "actually mozzarella" strongly
    implies the user is trying to select a specific option.

    Intentionally conservative — "no" alone is NOT a correction signal
    because it is a valid negated-modifier prefix in WAITING_FOR_MODIFIER.
    """
    normalized = (text or "").strip().lower()
    if not normalized:
        return False
    return any(normalized.startswith(phrase) for phrase in _CORRECTION_STARTERS)


# ---------------------------------------------------------------------------
# Route modes
# ---------------------------------------------------------------------------


class OptionRouteMode(str, Enum):
    """Route decision for a single option resolution attempt."""

    NO_GPT = "no_gpt"
    """GPT is not called — local resolution succeeded, text empty, or mode disabled."""

    SHADOW_GPT = "shadow_gpt"
    """GPT is called but result is always logged-only; never applied to cart/state."""

    INLINE_GPT = "inline_gpt"
    """GPT is called; result applied when validator confirms safe_to_apply=True."""


# ---------------------------------------------------------------------------
# Routing policy
# ---------------------------------------------------------------------------


class GptRoutingPolicy:
    """Stateless policy: inspect config + turn context, return OptionRouteMode.

    Thread-safe: no mutable state.  Instantiate once and share across turns.
    """

    def decide(
        self,
        *,
        config: "SemanticRepairConfig",
        local_resolved: bool,
        user_text: str,
        options_exist: bool,
        repeat_count: int,
        has_correction: bool = False,
    ) -> OptionRouteMode:
        """Return the appropriate route mode for the current turn.

        Parameters
        ----------
        config:
            Current SemanticRepairConfig snapshot.
        local_resolved:
            True when ModifierGroupResolver.resolve() found at least one selection.
            When True the routing policy returns NO_GPT — GPT adds no value.
        user_text:
            Normalized user text for this turn.
            Used for length check (>= 3) and empty-text guard.
        options_exist:
            True when the current modifier group has at least one choice.
        repeat_count:
            Consecutive failed reprompts on this field in this session.
            Used for repeat-loop escalation.
        has_correction:
            True when user_text contains a self-correction phrase such as
            "actually", "I mean", "instead", etc.  When True, INLINE_GPT
            is triggered even for short text (overrides the len >= 3 guard).
        """
        # ── Global guard 1: silence / empty text ──────────────────────────
        if not (user_text or "").strip():
            return OptionRouteMode.NO_GPT

        # ── Global guard 2: mode disabled ─────────────────────────────────
        mode = getattr(config, "option_resolver_mode", "disabled")
        if mode == "disabled":
            return OptionRouteMode.NO_GPT

        repeat_threshold = int(getattr(config, "option_resolver_repeat_threshold", 2))

        # ── Rule: shadow mode ─────────────────────────────────────────────
        if mode == "shadow":
            # No value calling GPT when local already resolved.
            if local_resolved:
                return OptionRouteMode.NO_GPT
            if options_exist:
                return OptionRouteMode.SHADOW_GPT
            return OptionRouteMode.NO_GPT

        # ── Rule: inline mode ─────────────────────────────────────────────
        if mode == "inline":
            # 3a. Repeat-loop recovery: user is stuck — escalate even for
            #     short or ambiguous text.
            if repeat_count >= repeat_threshold:
                return OptionRouteMode.INLINE_GPT

            # 3b. Self-correction signal: user said "actually X", "I mean X",
            #     etc. — escalate even if text is short.
            if not local_resolved and has_correction and options_exist:
                return OptionRouteMode.INLINE_GPT

            # 3c. Normal trigger: local failed, text is substantial, choices exist.
            text_len = len((user_text or "").strip())
            if not local_resolved and text_len >= 3 and options_exist:
                return OptionRouteMode.INLINE_GPT

        return OptionRouteMode.NO_GPT
