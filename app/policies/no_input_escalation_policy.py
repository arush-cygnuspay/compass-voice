# app/policies/no_input_escalation_policy.py
"""Turn-level no-input / unknown-intent escalation policy.

Stateless complement to ``PromptRepromptPolicy``: the latter handles
field-level slot misses (re-asking for a side, modifier, size, etc.);
this one handles the *global* "I didn't catch that" / intent-not-allowed
path that fires when the router rejects a (state, intent) pair.

The contract mirrors PromptRepromptPolicy on purpose so handlers,
renderers, and tests can reason about both the same way:

    miss_count    → tier
    0..1          → REPROMPT_STATE      (state-anchored short re-ask)
    2             → REPROMPT_WITH_HINT  (concrete options / examples)
    3             → OFFER_HELP          (offer to read options or hand off)
    4+            → HANDOFF             (transfer to human agent)

Thresholds are exposed as class attributes so the realtime config can
override them without touching the policy logic.
"""
from __future__ import annotations

from enum import Enum


class NoInputTier(str, Enum):
    """Ordered tiers returned by ``NoInputEscalationPolicy.next_tier``."""

    REPROMPT_STATE = "reprompt_state"
    REPROMPT_WITH_HINT = "reprompt_with_hint"
    OFFER_HELP = "offer_help"
    HANDOFF = "handoff"


class NoInputEscalationPolicy:
    """Stateless tier-mapping for consecutive global-fallback misses.

    ``miss_count`` is the 1-based index of the current failed attempt
    (i.e. ``conversation_context.consecutive_unknown_count`` AFTER the
    bump for this turn).
    """

    # Default thresholds — read by ``next_tier``. Override via subclass
    # or by mutating these attributes from app.config at startup if you
    # want different escalation aggressiveness.
    HINT_AT: int = 2
    HELP_AT: int = 3
    HANDOFF_AT: int = 4

    @classmethod
    def next_tier(cls, miss_count: int) -> NoInputTier:
        if miss_count <= 0:
            return NoInputTier.REPROMPT_STATE
        if miss_count < cls.HINT_AT:
            return NoInputTier.REPROMPT_STATE
        if miss_count < cls.HELP_AT:
            return NoInputTier.REPROMPT_WITH_HINT
        if miss_count < cls.HANDOFF_AT:
            return NoInputTier.OFFER_HELP
        return NoInputTier.HANDOFF
