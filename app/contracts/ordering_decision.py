# app/contracts/ordering_decision.py
"""Immutable decision DTOs returned by OrderingDecisionEngine.

These types cross the boundary between the engine (pure logic) and handlers
(side-effectful executors).  All fields are deliberately primitive or frozen
so the engine remains testable without handler infrastructure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class OrderingAction(Enum):
    """The single action the system should take next."""

    ADD_ITEM = "add_item"
    ASK_REQUIRED_GROUP = "ask_required_group"
    ASK_QUANTITY = "ask_quantity"
    ASK_ITEM_CONFIRMATION = "ask_item_confirmation"
    SUGGEST_ALTERNATIVES = "suggest_alternatives"
    ESCALATE_TO_AGENT = "escalate_to_agent"
    CONTINUE_ORDERING = "continue_ordering"
    REVIEW_ORDER = "review_order"
    CHECKOUT = "checkout"
    CANCEL_ORDER = "cancel_order"
    UNCLEAR = "unclear"


@dataclass(frozen=True, slots=True)
class CandidateDecision:
    """One item offered as a disambiguation candidate."""

    item_id: str
    item_name: str
    score: float = 0.0


@dataclass(frozen=True, slots=True)
class OrderingDecision:
    """Immutable snapshot of what the engine decided and why.

    The ``action`` field is the primary discriminant.  Handlers read the
    remaining fields to build HandlerResult / mutate ConversationContext.

    Field conventions
    -----------------
    item_id / item_name
        Set when action is ADD_ITEM or ASK_ITEM_CONFIRMATION with a single
        best candidate (near-miss or candidate_selected flow).
    category_id / category_name
        Set when reason == "category_detected".
    candidates
        Non-empty for CATEGORY and AMBIGUOUS confirmation flows.  Contains
        the shortlist the user must pick from.
    query
        The raw (user-facing) text that triggered a NOT_FOUND path.
    suggestions
        String names to surface in a LOW-confidence fallback response.
        Max 4, already rotated for attempt 2.
    tier
        "HIGH" | "MEDIUM" for near-miss, "LOW" for no-match.  None for
        non-not-found actions.
    attempt
        Predicted post-bump not-found attempt count for this query.  The
        handler must call context.bump_not_found() to materialise it.
    matched_category_names
        Category names for AMBIGUOUS results that span categories.
    requires_confirmation
        True whenever the engine expects the user to confirm before the
        action is executed.
    """

    action: OrderingAction
    reason: str | None = None

    # Item identity
    item_id: str | None = None
    item_name: str | None = None

    # Quantity (not yet populated by this engine pass; reserved for future)
    quantity: int | None = None
    quantity_source: Literal["slot", "text", "default"] | None = None

    # Category context
    category_id: str | None = None
    category_name: str | None = None

    # Disambiguation candidates
    candidates: tuple[CandidateDecision, ...] = ()
    matched_category_names: tuple[str, ...] = ()

    # FSM hints
    next_state: str | None = None
    handler_result_hint: str | None = None
    requires_confirmation: bool = False

    # Not-found specifics
    query: str | None = None
    suggestions: tuple[str, ...] = ()
    tier: Literal["HIGH", "MEDIUM", "LOW"] | None = None
    attempt: int = 0
