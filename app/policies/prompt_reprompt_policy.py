# app/policies/prompt_reprompt_policy.py
"""Field-level reprompt escalation policy.

Owns the miss-count → prompt-action decision so handlers and renderers
stay free of escalation logic.  No mutable state; purely a function.
"""
from __future__ import annotations

from enum import Enum


class RepromptAction(str, Enum):
    """Ordered tiers returned by PromptRepromptPolicy.next_action()."""

    FULL_OPTIONS = "full_options"
    CONCISE = "concise"
    LIST_OPTIONS_HINT = "list_options_hint"
    ESCALATE_OR_SKIP = "escalate_or_skip"


class PromptRepromptPolicy:
    """Stateless policy: maps (field, miss_count) → RepromptAction.

    miss_count is the 1-based index of the current failed attempt for
    the given field within the session (supplied by HandlerDispatcher
    via ``payload["reprompt_count"]``).

    Tier contract
    -------------
    0   → FULL_OPTIONS   (initial prompt / backward-compat default)
    1   → CONCISE        (first miss — short re-ask, no option list)
    2   → LIST_OPTIONS_HINT (second miss — suggest "say list options")
    3+  → ESCALATE_OR_SKIP  (third miss — show full list, recovery path)
    """

    @staticmethod
    def next_action(field: str, miss_count: int) -> RepromptAction:  # noqa: ARG004
        if miss_count <= 0:
            return RepromptAction.FULL_OPTIONS
        if miss_count == 1:
            return RepromptAction.CONCISE
        if miss_count == 2:
            return RepromptAction.LIST_OPTIONS_HINT
        return RepromptAction.ESCALATE_OR_SKIP
