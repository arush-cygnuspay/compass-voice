# app/services/order_lifecycle_guard.py
"""OrderLifecycleGuard — stateless lifecycle decision service for Compass Voice.

Answers four safety questions:

  1. check_item_resolution(raw_text, candidates) → LifecycleDecision
     Was the menu query resolved?  Returns NOT_FOUND / AMBIGUOUS / OK.

  2. check_pending_requirements(pending_add_item, context) → LifecycleDecision
     Does the in-progress item still need size / side / modifier choices?

  3. can_checkout(cart, context) → LifecycleDecision
     Is the cart in a state that allows checkout?

  4. can_send_payment_link(order_state) → LifecycleDecision
     Is the order ready for a payment link to be sent?

Design principles
-----------------
* Pure functions — no handler imports, no side effects, no I/O.
* Returns immediately and never raises.
* Voice-friendly response strings are baked into each LifecycleDecision so
  callers can use them directly without a separate lookup.
* Operates on lightweight interfaces (duck-typed).  Does NOT import
  ConversationContext, Cart, or handler classes at the top level to avoid
  circular imports; type annotations use string literals where needed.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    # Import only for static-analysis / type checking — never at runtime.
    from app.state_machine.models.conversation_context import ConversationContext
    from app.state_machine.models.pending_item_models import PendingAddItem
    from app.cart.cart import Cart

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


class LifecycleCode(str, Enum):
    """Machine-readable outcome codes returned by every guard method."""

    OK = "ok"
    ITEM_NOT_FOUND = "item_not_found"
    ITEM_AMBIGUOUS = "item_ambiguous"
    ITEM_UNAVAILABLE = "item_unavailable"
    SIZE_REQUIRED = "size_required"
    SIDE_REQUIRED = "side_required"
    MODIFIER_REQUIRED = "modifier_required"
    CART_EMPTY = "cart_empty"
    CART_INCOMPLETE = "cart_incomplete"
    ORDER_NOT_READY = "order_not_ready"
    PAYMENT_NOT_READY = "payment_not_ready"


@dataclass(frozen=True, slots=True)
class LifecycleDecision:
    """Immutable result of a lifecycle guard check.

    Attributes
    ----------
    code:
        Machine-readable outcome.
    blocking:
        True when the caller must NOT proceed (gate is closed).
    response:
        Voice-ready response string.  Empty only when code == OK.
    details:
        Optional structured metadata (item names, group names, etc.)
        for richer response templating.  Never contains PII or API keys.
    """

    code: LifecycleCode
    blocking: bool
    response: str
    details: dict[str, Any] = field(default_factory=dict)


# Singleton "everything is fine" decision so callers can identity-check.
_OK = LifecycleDecision(code=LifecycleCode.OK, blocking=False, response="")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_item_resolution(
    raw_text: str,
    candidates: Sequence[str],
    *,
    menu_context: Sequence[str] | None = None,
    unavailable: bool = False,
) -> LifecycleDecision:
    """Return a lifecycle decision for a failed menu query.

    Parameters
    ----------
    raw_text:
        The exact item name or phrase the user requested.
    candidates:
        Nearby item names that the menu *does* carry.  Shown to the user
        as alternatives.  Pass an empty sequence when there are none.
    menu_context:
        Optional broader list of available items (not displayed verbatim
        — used for logging/context only).
    unavailable:
        Set True when the item exists in the database but is currently
        marked unavailable (sold out, seasonal, etc.) rather than simply
        absent.

    Returns
    -------
    LifecycleDecision with code ITEM_NOT_FOUND or ITEM_UNAVAILABLE and a
    voice-friendly response string.  Never returns OK.
    """
    item_label = (raw_text or "").strip()
    close = [c for c in (candidates or []) if c]

    if unavailable:
        code = LifecycleCode.ITEM_UNAVAILABLE
        if close:
            alts = _format_candidate_list(close[:3])
            response = (
                f"Sorry, {item_label or 'that item'} isn't available right now. "
                f"We do have {alts}. What would you like?"
            )
        else:
            response = (
                f"Sorry, {item_label or 'that item'} isn't available right now. "
                "What else can I get you?"
            )
    else:
        code = LifecycleCode.ITEM_NOT_FOUND
        if close:
            alts = _format_candidate_list(close[:3])
            response = (
                f"Sorry, we don't have {item_label or 'that'}. "
                f"We do have {alts}. Would you like one of those?"
            )
        else:
            response = (
                "Sorry, we don't have that on the menu. "
                "What else would you like?"
            )

    logger.debug(
        "lifecycle_guard.check_item_resolution",
        extra={
            "code": code.value,
            "raw_text": item_label,
            "candidate_count": len(close),
            "unavailable": unavailable,
        },
    )

    return LifecycleDecision(
        code=code,
        blocking=True,
        response=response,
        details={
            "raw_text": item_label,
            "candidates": list(close[:3]),
            "unavailable": unavailable,
        },
    )


def check_pending_requirements(
    pending: "PendingAddItem",
    context: "ConversationContext",
) -> LifecycleDecision:
    """Return the first unresolved *required* choice for the in-progress item.

    Mirrors the priority order of determine_next_add_item_step():
      1. Required modifier groups
      2. Required side groups
      3. Item size / variant

    Returns OK when all required choices are satisfied.
    Does NOT check optional groups (those don't block checkout).
    """
    if pending is None:
        return _OK

    # 1. Required modifier groups ─────────────────────────────────────────
    for group in getattr(pending, "modifier_groups", ()):
        if not getattr(group, "is_required", False):
            continue
        group_id = getattr(group, "group_id", "")
        selected = context.selected_modifier_groups.get(group_id, ())
        skipped = group_id in context.skipped_modifier_groups
        if not _modifier_group_satisfied(group, selected, skipped):
            group_name = getattr(group, "name", "modifier")
            item_name = getattr(pending, "item_name", "your item")
            top_choices = list(getattr(group, "top_choice_names", ()))[:4]
            choices_text = _format_candidate_list(top_choices) if top_choices else ""
            if choices_text:
                resp = (
                    f"I still need your {group_name} for the {item_name}. "
                    f"Would you like {choices_text}?"
                )
            else:
                resp = f"I still need your {group_name} for the {item_name}. Which one would you like?"
            return LifecycleDecision(
                code=LifecycleCode.MODIFIER_REQUIRED,
                blocking=True,
                response=resp,
                details={
                    "item_name": item_name,
                    "group_name": group_name,
                    "group_id": group_id,
                    "top_choices": top_choices,
                },
            )

    # 2. Required side groups ─────────────────────────────────────────────
    for group in getattr(pending, "side_groups", ()):
        if not getattr(group, "is_required", False):
            continue
        if getattr(group, "is_suggested_addon", False):
            continue
        group_id = getattr(group, "group_id", "")
        selected = context.selected_side_groups.get(group_id, ())
        skipped = group_id in context.skipped_side_groups
        min_sel = getattr(group, "min_selector", 1)
        if not (len(selected) >= min_sel or skipped):
            group_name = getattr(group, "name", "side")
            item_name = getattr(pending, "item_name", "your item")
            top_choices = list(getattr(group, "top_choice_names", ()))[:4]
            choices_text = _format_candidate_list(top_choices) if top_choices else ""
            if choices_text:
                resp = (
                    f"I still need your {group_name} for the {item_name}. "
                    f"Would you like {choices_text}?"
                )
            else:
                resp = f"I still need your {group_name} for the {item_name}. Which would you like?"
            return LifecycleDecision(
                code=LifecycleCode.SIDE_REQUIRED,
                blocking=True,
                response=resp,
                details={
                    "item_name": item_name,
                    "group_name": group_name,
                    "group_id": group_id,
                    "top_choices": top_choices,
                },
            )

    # 3. Item size / variant ──────────────────────────────────────────────
    item_variants = getattr(pending, "item_variants", ())
    if item_variants and not context.selected_variant_id:
        item_name = getattr(pending, "item_name", "that")
        size_names = list(getattr(pending, "item_variant_names", ()))[:4]
        choices_text = _format_candidate_list(size_names) if size_names else ""
        if choices_text:
            resp = f"What size {item_name} would you like — {choices_text}?"
        else:
            resp = f"What size would you like for the {item_name}?"
        return LifecycleDecision(
            code=LifecycleCode.SIZE_REQUIRED,
            blocking=True,
            response=resp,
            details={
                "item_name": item_name,
                "available_sizes": size_names,
            },
        )

    return _OK


def can_checkout(
    cart: "Cart",
    context: "ConversationContext",
) -> LifecycleDecision:
    """Return whether the cart and current item flow allow checkout.

    Gates (in priority order)
    -------------------------
    1. Cart is empty → CART_EMPTY
    2. A pending item exists with unresolved required choices → appropriate
       MODIFIER_REQUIRED / SIDE_REQUIRED / SIZE_REQUIRED decision.
    3. A pending item exists with no unresolved choices (shouldn't normally
       happen — flow should have finalized) → CART_INCOMPLETE.
    4. All clear → OK.

    Parameters
    ----------
    cart:
        Live cart object.  Must expose ``is_empty() → bool``.
    context:
        Current conversation context.  Used to detect pending item state.
    """
    try:
        if cart.is_empty():
            return LifecycleDecision(
                code=LifecycleCode.CART_EMPTY,
                blocking=True,
                response="Your cart is empty. What would you like to order?",
            )
    except Exception:
        # Cart unavailable — let the caller decide; don't block blindly.
        pass

    # Delivery order with address not yet collected → must ask for address first.
    order_type = getattr(context, "order_type", None)
    if order_type == "delivery":
        delivery_required = getattr(context, "delivery_address_required", False)
        if delivery_required:
            addr = getattr(context, "delivery_address", None)
            addr_collected = bool(getattr(addr, "collected", False))
            if not addr_collected:
                return LifecycleDecision(
                    code=LifecycleCode.CART_INCOMPLETE,
                    blocking=True,
                    response=(
                        "To complete your delivery order, I'll need your delivery address first. "
                        "What's the delivery address?"
                    ),
                    details={"reason": "delivery_address_required"},
                )

    pending = getattr(context, "pending_add_item", None)
    if pending is not None:
        req_check = check_pending_requirements(pending, context)
        if req_check.blocking:
            return req_check
        # Pending item exists but all requirements met — item hasn't been
        # finalized yet (unusual state; treat as incomplete).
        item_name = getattr(pending, "item_name", "your item")
        return LifecycleDecision(
            code=LifecycleCode.CART_INCOMPLETE,
            blocking=True,
            response=(
                f"I still need to finish adding the {item_name} before we check out. "
                "What would you like for it?"
            ),
            details={"item_name": item_name},
        )

    # Block checkout if structured staged items are waiting to be processed.
    staged_queue = getattr(context, "staged_item_queue", None)
    if staged_queue and len(staged_queue) > 0:
        return LifecycleDecision(
            code=LifecycleCode.CART_INCOMPLETE,
            blocking=True,
            response="I still need to finish the remaining items first.",
            details={"reason": "staged_items_pending", "staged_count": len(staged_queue)},
        )

    # Block checkout if legacy pending item queue is non-empty.
    pending_queue = getattr(context, "pending_item_queue", None)
    if pending_queue and len(pending_queue) > 0:
        return LifecycleDecision(
            code=LifecycleCode.CART_INCOMPLETE,
            blocking=True,
            response="I still need to finish the remaining items first.",
            details={"reason": "pending_items_queued", "pending_count": len(pending_queue)},
        )

    return _OK


def can_send_payment_link(
    order_state: Any,
    *,
    cart: "Cart | None" = None,
) -> LifecycleDecision:
    """Return whether the order is ready for a payment link to be sent.

    Parameters
    ----------
    order_state:
        Dict-like object that carries order readiness signals.  Expected
        keys (all optional — missing key → not ready):
          - "order_number"    — non-empty string or int
          - "payment_ready"   — truthy bool
          - "submit_ok"       — truthy bool (order was submitted successfully)
    cart:
        If supplied, used to verify the cart is non-empty as a sanity check.

    Returns
    -------
    OK when all present signals are positive.
    PAYMENT_NOT_READY otherwise.
    """
    _PAYMENT_FAIL_RESPONSE = (
        "Your order is ready, but I wasn't able to send the payment link. "
        "Let me connect you with someone who can help."
    )

    if order_state is None:
        return LifecycleDecision(
            code=LifecycleCode.PAYMENT_NOT_READY,
            blocking=True,
            response=_PAYMENT_FAIL_RESPONSE,
            details={"reason": "order_state_missing"},
        )

    # Allow both dict-like and attribute-style access.
    def _get(key: str) -> Any:
        try:
            return order_state[key]
        except (TypeError, KeyError):
            pass
        return getattr(order_state, key, None)

    order_number = _get("order_number")
    if not order_number:
        return LifecycleDecision(
            code=LifecycleCode.PAYMENT_NOT_READY,
            blocking=True,
            response=_PAYMENT_FAIL_RESPONSE,
            details={"reason": "no_order_number"},
        )

    submit_ok = _get("submit_ok")
    payment_ready = _get("payment_ready")
    if submit_ok is False or payment_ready is False:
        return LifecycleDecision(
            code=LifecycleCode.PAYMENT_NOT_READY,
            blocking=True,
            response=_PAYMENT_FAIL_RESPONSE,
            details={"reason": "submit_or_payment_not_ready"},
        )

    if cart is not None:
        try:
            if cart.is_empty():
                return LifecycleDecision(
                    code=LifecycleCode.PAYMENT_NOT_READY,
                    blocking=True,
                    response=_PAYMENT_FAIL_RESPONSE,
                    details={"reason": "cart_empty_at_payment"},
                )
        except Exception:
            pass

    return _OK


def build_blocking_response(decision: LifecycleDecision) -> str:
    """Return the voice-ready response string from a LifecycleDecision.

    For OK decisions this returns an empty string.
    For blocking decisions this returns the pre-built ``response`` field.
    """
    return decision.response


# ---------------------------------------------------------------------------
# Helpers (private)
# ---------------------------------------------------------------------------


def _format_candidate_list(names: Sequence[str]) -> str:
    """Format up to 3 names into a natural-language list.

    Examples
    --------
    ["Coke"]                       → "Coke"
    ["Coke", "Sprite"]             → "Coke or Sprite"
    ["Coke", "Sprite", "Water"]    → "Coke, Sprite, or Water"
    """
    items = [n for n in names if n][:3]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} or {items[1]}"
    return f"{items[0]}, {items[1]}, or {items[2]}"


def _modifier_group_satisfied(group: Any, selected: Any, skipped: bool) -> bool:
    """Return True when a modifier group no longer blocks checkout.

    Mirrors the logic in add_item_flow._modifier_group_satisfied without
    importing from the handler layer.
    """
    selected_count = len(selected or ())
    min_selector = int(getattr(group, "min_selector", 0))
    if getattr(group, "is_required", False):
        return selected_count >= min_selector
    return selected_count > 0 or skipped
