# app/state_machine/handlers/item/add_item/item_quantity_policy.py
"""Centralized item quantity policy for the add-item flow.

Single authoritative place for deciding whether an item's quantity is
  - explicit   : already set to a valid positive int
  - implicit_default : not mentioned; default to 1
  - ambiguous  : vague expression ("some", "a few", "several") → ask
  - invalid    : zero / negative / malformed → ask with error prompt
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from app.state_machine.models.conversation_context import ConversationContext

logger = logging.getLogger(__name__)

# Leading vague patterns anchored to start of (item-name-stripped) text.
# "burger with some modifications" must NOT trigger vague detection — the
# anchor ensures only a leading "some"/"a few"/"several" fires.
_LEADING_VAGUE_PATTERNS = (
    re.compile(r"^some\b", re.IGNORECASE),
    re.compile(r"^a few\b", re.IGNORECASE),
    re.compile(r"^several\b", re.IGNORECASE),
)


@dataclass(frozen=True, slots=True)
class NormalizedItemQuantity:
    quantity: int | None
    source: Literal["explicit", "implicit_default", "ambiguous", "invalid"]
    needs_clarification: bool
    reason: str | None = None


def normalize_item_quantity(
    context: "ConversationContext",
    *,
    user_text: str = "",
    explicit_quantity_hint: int | None = None,
) -> NormalizedItemQuantity:
    """Return the effective item quantity per business policy.

    Priority order:
    1. ``explicit_quantity_hint`` when provided (overrides context.quantity).
    2. ``context.quantity`` when it is already a valid positive int.
    3. Leading vague expression in ``user_text`` → ambiguous, needs clarification.
    4. Everything else → implicit default of 1 (never ask unnecessarily).

    This function does NOT mutate *context*.  Callers are responsible for
    writing ``context.quantity = result.quantity`` when they decide to apply
    the result.
    """
    # --- 1. explicit_quantity_hint (slot / caller override) ------------------
    if explicit_quantity_hint is not None:
        if isinstance(explicit_quantity_hint, int) and explicit_quantity_hint > 0:
            result = NormalizedItemQuantity(
                quantity=explicit_quantity_hint,
                source="explicit",
                needs_clarification=False,
            )
        else:
            result = NormalizedItemQuantity(
                quantity=None,
                source="invalid",
                needs_clarification=True,
                reason="non_positive_quantity",
            )
        _emit(result, context)
        return result

    # --- 2. context.quantity already set -------------------------------------
    qty = context.quantity
    if isinstance(qty, int):
        if qty > 0:
            result = NormalizedItemQuantity(
                quantity=qty,
                source="explicit",
                needs_clarification=False,
            )
            _emit(result, context)
            return result
        # Zero or negative — invalid
        result = NormalizedItemQuantity(
            quantity=None,
            source="invalid",
            needs_clarification=True,
            reason="non_positive_quantity",
        )
        _emit(result, context)
        return result

    # --- 3. Leading vague expression in user_text ----------------------------
    if user_text:
        normalized = user_text.lower().strip()
        if any(p.search(normalized) for p in _LEADING_VAGUE_PATTERNS):
            result = NormalizedItemQuantity(
                quantity=None,
                source="ambiguous",
                needs_clarification=True,
                reason="vague_quantity",
            )
            _emit(result, context)
            return result

    # --- 4. Implicit default -------------------------------------------------
    result = NormalizedItemQuantity(
        quantity=1,
        source="implicit_default",
        needs_clarification=False,
    )
    _emit(result, context)
    return result


def _emit(result: NormalizedItemQuantity, context: "ConversationContext") -> None:
    """Emit a structured log event for the quantity policy decision."""
    pending = getattr(context, "pending_add_item", None)
    item_name = getattr(pending, "item_name", None) if pending else None
    item_id = getattr(context, "current_item_id", None)

    logger.debug(
        "quantity_policy_applied source=%s quantity=%s needs_clarification=%s "
        "reason=%s item_id=%s item_name=%s",
        result.source,
        result.quantity,
        result.needs_clarification,
        result.reason,
        item_id,
        item_name,
    )
