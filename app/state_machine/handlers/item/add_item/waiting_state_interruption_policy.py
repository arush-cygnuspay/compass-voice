# app/state_machine/handlers/item/add_item/waiting_state_interruption_policy.py
"""Waiting-state new-item interruption policy.

When the bot is in WAITING_FOR_SIDE, WAITING_FOR_MODIFIER, or WAITING_FOR_SIZE
and the user's utterance looks like a request for a *new* menu item rather than
an answer to the current required group prompt, this policy decides whether to:

  BLOCK   — Emit ``block_new_item_until_required_done`` and keep the current
             waiting state.  Used when the group is required and the user has
             not yet satisfied the minimum selection count.

  ALLOW   — Let the handler proceed with its normal resolution logic (no-op).

Design constraints
------------------
* Never silently abandon a required side/modifier/size.
* Do not bypass mandatory group completion.
* Do not touch payment or checkout flows.
* Policy is purely advisory — the calling handler retains full control.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.nlu.query_normalization.text_preprocessor import normalize_text
from app.nlu.order_scaffolding import ORDER_FILLER_PREFIXES, ORDER_FILLER_TOKENS
from app.state_machine.handlers.item.add_item.group_collection_utils import (
    effective_group_selector_bounds,
)


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

class InterruptionDecision(Enum):
    BLOCK = "block"
    ALLOW = "allow"


@dataclass(frozen=True, slots=True)
class InterruptionPolicyResult:
    decision: InterruptionDecision
    # Only set when decision == BLOCK:
    pending_item_name: str = ""
    group_prompt_noun: str = ""   # e.g. "side", "modifier", "size"
    remaining_to_min: int = 0


# ---------------------------------------------------------------------------
# Heuristics: does this utterance look like a new-item request?
# ---------------------------------------------------------------------------

def _strip_filler_prefix(text: str) -> str:
    """Remove the longest matching ORDER_FILLER_PREFIX from the start."""
    for prefix in ORDER_FILLER_PREFIXES:
        if text.startswith(prefix):
            return text[len(prefix):]
    return text


def _looks_like_new_item_request(normalized_text: str) -> bool:
    """Return True when *normalized_text* looks like the user is asking for
    a new menu item rather than answering the current group prompt.

    Heuristics (any match → True):
    1. Starts with a known ordering filler prefix ("can i get", "add a", …).
    2. After filler-stripping, is entirely stop tokens (pure filler, no content)
       — this catches bare "add" or "i want" with nothing after.

    Conservative by design: false negatives (not blocking when we should) are
    less harmful than false positives (blocking a legitimate option answer).
    """
    if not normalized_text:
        return False

    # Heuristic 1: starts with a filler ordering prefix (longest match first).
    # NOTE: prefixes intentionally carry a trailing space (e.g. "a ", "an ")
    # that acts as a word-boundary guard — do NOT strip them before matching,
    # or "avocado".startswith("a") would spuriously fire.
    for prefix in ORDER_FILLER_PREFIXES:
        if prefix and normalized_text.startswith(prefix):
            # Stripping a prefix means the user is framing a new order.
            remainder = normalized_text[len(prefix):].strip()
            if remainder:
                # There's content after the prefix — almost certainly a new item.
                return True

    # Heuristic 2: the whole phrase is just order-scaffolding words with no
    # menu-item content (e.g. bare "add" or "i want to order").
    tokens = [t for t in normalized_text.split() if t]
    non_filler = [t for t in tokens if t not in ORDER_FILLER_TOKENS]
    if not non_filler:
        return False  # Nothing but fillers — too ambiguous; let handler decide.

    return False


# ---------------------------------------------------------------------------
# Main policy entry point
# ---------------------------------------------------------------------------

def evaluate_waiting_state_interruption(
    *,
    normalized_user_text: str,
    pending_item_name: str,
    group_is_required: bool,
    group_prompt_noun: str,
    selected_count: int,
    min_selector: int,
) -> InterruptionPolicyResult:
    """Decide whether a new-item utterance should BLOCK the current waiting state.

    Parameters
    ----------
    normalized_user_text:
        The user's input, already normalized (lowercase, punctuation stripped).
    pending_item_name:
        The item currently being configured (e.g. "Cheese Burger").
    group_is_required:
        Whether the current group is required (``is_required=True``).
    group_prompt_noun:
        Human-readable noun for the group (e.g. "side", "protein", "sauce").
    selected_count:
        How many choices the user has already made for this group.
    min_selector:
        Minimum selections required by this group.

    Returns
    -------
    InterruptionPolicyResult with decision=BLOCK or ALLOW.
    """
    # Only block when:
    #   • the utterance actually looks like a new-item request, AND
    #   • the group is required, AND
    #   • the user hasn't yet satisfied the minimum selection count.
    remaining = max(min_selector - selected_count, 0)
    if (
        group_is_required
        and remaining > 0
        and _looks_like_new_item_request(normalized_user_text)
    ):
        return InterruptionPolicyResult(
            decision=InterruptionDecision.BLOCK,
            pending_item_name=pending_item_name,
            group_prompt_noun=group_prompt_noun or "option",
            remaining_to_min=remaining,
        )

    return InterruptionPolicyResult(decision=InterruptionDecision.ALLOW)


def evaluate_waiting_side_interruption(
    *,
    normalized_user_text: str,
    pending_item_name: str,
    group,  # PendingSideGroup
    selected_count: int,
) -> InterruptionPolicyResult:
    """Convenience wrapper for WAITING_FOR_SIDE state."""
    from app.state_machine.handlers.item.add_item.group_classification import (
        speech_noun_for_side_group,
    )
    min_sel, _ = effective_group_selector_bounds(group)
    return evaluate_waiting_state_interruption(
        normalized_user_text=normalized_user_text,
        pending_item_name=pending_item_name,
        group_is_required=bool(group.is_required),
        group_prompt_noun=speech_noun_for_side_group(group.name),
        selected_count=selected_count,
        min_selector=min_sel,
    )


def evaluate_waiting_modifier_interruption(
    *,
    normalized_user_text: str,
    pending_item_name: str,
    group,  # PendingModifierGroup
    selected_count: int,
) -> InterruptionPolicyResult:
    """Convenience wrapper for WAITING_FOR_MODIFIER state."""
    min_sel, _ = effective_group_selector_bounds(group)
    return evaluate_waiting_state_interruption(
        normalized_user_text=normalized_user_text,
        pending_item_name=pending_item_name,
        group_is_required=bool(group.is_required),
        group_prompt_noun=getattr(group, "prompt_noun", None) or "modifier",
        selected_count=selected_count,
        min_selector=min_sel,
    )
