# app/nlu/turn_resolver/control_flow_validator.py
"""Validates ControlFlowResolution before any handler acts on it.

validate_control_flow_resolution() is the final safety gate for all
control-flow GPT results.  Handlers MUST call it before applying any
action from ControlFlowResolver.

Safety contract
---------------
* Never raises into callers.
* Checkout / payment actions are re-checked against OrderLifecycleGuard.
* Order type changes are blocked when payment is already sent or order submitted.
* Confidence gate applies to all applicable actions.
* Control actions (clarify, fallback) always pass structural validation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.nlu.turn_resolver.control_flow_resolver import ControlFlowResolution

# Default minimum confidence for auto-apply.
_DEFAULT_MIN_CONFIDENCE: float = 0.70

# Valid requested_order_type values.
_VALID_ORDER_TYPES: frozenset[str] = frozenset({"pickup", "delivery"})

# Valid payment_preference values.
_VALID_PAYMENT_PREFERENCES: frozenset[str] = frozenset({"send_link", "pay_on_arrival"})

# States where payment permission makes semantic sense.
_PAYMENT_PERMISSION_STATES: frozenset[str] = frozenset({
    "waiting_for_pickup_sms_permission",
    "waiting_for_payment_permission",
})

# States where a mid-order type switch is allowed.
_ORDER_TYPE_CHANGEABLE_STATES: frozenset[str] = frozenset({
    "idle",
    "greeting",
    "confirming_item",
    "waiting_for_size",
    "waiting_for_side",
    "waiting_for_side_size",
    "waiting_for_modifier",
    "waiting_for_quantity",
    "confirming_order",
    "modifying_item",
    "removing_item",
    "waiting_for_order_type",
    "waiting_for_delivery_eligibility",
    "waiting_for_delivery_address_collection",
    "cancellation_confirmation",
})

# States/conditions that block order type changes.
_PAYMENT_BLOCKING_STATES: frozenset[str] = frozenset({
    "waiting_for_payment",
    "waiting_for_checkout_completion",
    "waiting_for_pickup_sms_permission",
    "completed",
})

# Control / informational actions that always pass structural validation.
_CONTROL_ACTIONS: frozenset[str] = frozenset({
    "clarify",
    "fallback",
    "continue_ordering",
    "cancel_pending",
})


# ── Result dataclass ──────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ControlFlowValidationResult:
    """Outcome of validate_control_flow_resolution()."""

    is_valid: bool
    reason: str
    block_reason: str | None = None


VALIDATION_OK = ControlFlowValidationResult(is_valid=True, reason="ok")


# ── Public API ────────────────────────────────────────────────────────────────


def validate_control_flow_resolution(
    resolution: "ControlFlowResolution",
    context: Any,
    state: str,
    *,
    min_confidence: float = _DEFAULT_MIN_CONFIDENCE,
) -> ControlFlowValidationResult:
    """Validate a ControlFlowResolution before a handler applies it.

    Parameters
    ----------
    resolution:
        The resolution returned by ControlFlowResolver.resolve_sync().
    context:
        ConversationContext — used to check payment/submission flags and
        cart/pending-item state via OrderLifecycleGuard.
    state:
        Current FSM state string (lowercase).
    min_confidence:
        Minimum GPT confidence required for applicable actions.

    Returns
    -------
    VALIDATION_OK when the resolution may be applied.
    ControlFlowValidationResult(is_valid=False, …) when it must not.
    """
    try:
        return _validate(resolution, context, (state or "").lower().strip(), min_confidence)
    except Exception as exc:
        return ControlFlowValidationResult(
            is_valid=False,
            reason="validation_exception",
            block_reason=str(exc)[:200],
        )


# ── Private helpers ───────────────────────────────────────────────────────────


def _validate(
    resolution: "ControlFlowResolution",
    context: Any,
    state: str,
    min_confidence: float,
) -> ControlFlowValidationResult:
    action = (resolution.action or "fallback").lower()

    # Confidence gate — apply to all non-control actions.
    if action not in _CONTROL_ACTIONS:
        if resolution.confidence < min_confidence:
            return ControlFlowValidationResult(
                is_valid=False,
                reason="low_confidence",
                block_reason=(
                    f"confidence={resolution.confidence:.3f} < "
                    f"threshold={min_confidence:.3f}"
                ),
            )

    # Control / informational actions always pass.
    if action in _CONTROL_ACTIONS:
        return VALIDATION_OK

    # ── Checkout actions ──────────────────────────────────────────────────
    if action in {"confirm_checkout", "request_checkout"}:
        return _validate_checkout(context)

    if action == "deny_checkout":
        # Deny is always valid — customer is declining/modifying.
        return VALIDATION_OK

    # ── Payment permission actions ────────────────────────────────────────
    if action in {"confirm_payment_link", "deny_payment_link", "pay_on_arrival"}:
        return _validate_payment_permission(state)

    # ── Order type actions ────────────────────────────────────────────────
    if action in {"change_order_type", "select_initial_order_type"}:
        return _validate_order_type_action(resolution, context, state)

    # ── Cancel order ──────────────────────────────────────────────────────
    if action == "cancel_order":
        return VALIDATION_OK  # Cancellation routing is handler's responsibility.

    # Unknown action — reject.
    return ControlFlowValidationResult(
        is_valid=False,
        reason="unknown_action",
        block_reason=f"action={action!r} is not a recognised ControlFlowAction",
    )


def _validate_checkout(context: Any) -> ControlFlowValidationResult:
    """Validate checkout using OrderLifecycleGuard.can_checkout()."""
    try:
        from app.services.order_lifecycle_guard import can_checkout, LifecycleCode

        cart = getattr(context, "cart", None)
        if cart is None:
            # No cart attached — let handler decide; don't block blindly.
            return VALIDATION_OK

        decision = can_checkout(cart, context)
        if decision.blocking:
            return ControlFlowValidationResult(
                is_valid=False,
                reason=f"lifecycle_blocked:{decision.code.value}",
                block_reason=decision.response,
            )
        return VALIDATION_OK
    except Exception as exc:
        # If lifecycle guard is unavailable, pass validation and let the
        # handler's own guard fire.
        return ControlFlowValidationResult(
            is_valid=False,
            reason="lifecycle_guard_error",
            block_reason=str(exc)[:200],
        )


def _validate_payment_permission(state: str) -> ControlFlowValidationResult:
    """Payment permission actions are only valid in payment-permission states."""
    if state not in _PAYMENT_PERMISSION_STATES:
        return ControlFlowValidationResult(
            is_valid=False,
            reason="invalid_state_for_payment_permission",
            block_reason=(
                f"payment permission action requires state in "
                f"{sorted(_PAYMENT_PERMISSION_STATES)}, got {state!r}"
            ),
        )
    return VALIDATION_OK


def _validate_order_type_action(
    resolution: "ControlFlowResolution",
    context: Any,
    state: str,
) -> ControlFlowValidationResult:
    """Validate an order-type change or initial selection action."""
    # requested_order_type must be valid.
    requested = (resolution.requested_order_type or "").lower()
    if requested not in _VALID_ORDER_TYPES:
        return ControlFlowValidationResult(
            is_valid=False,
            reason="invalid_requested_order_type",
            block_reason=f"requested_order_type={requested!r} must be 'pickup' or 'delivery'",
        )

    # Block if payment already sent or order submitted.
    if state in _PAYMENT_BLOCKING_STATES:
        return ControlFlowValidationResult(
            is_valid=False,
            reason="order_type_change_blocked_payment_state",
            block_reason=f"cannot change order type in payment state {state!r}",
        )

    if _payment_already_sent(context):
        return ControlFlowValidationResult(
            is_valid=False,
            reason="order_type_change_blocked_payment_sent",
            block_reason="payment link already sent; cannot change order type",
        )

    if _order_already_submitted(context):
        return ControlFlowValidationResult(
            is_valid=False,
            reason="order_type_change_blocked_submitted",
            block_reason="order already submitted; cannot change order type",
        )

    return VALIDATION_OK


def _payment_already_sent(context: Any) -> bool:
    """Duck-typed check for payment link already sent."""
    addr = getattr(context, "delivery_address", None)
    if addr is not None:
        if getattr(addr, "payment_link", None):
            return True
        if getattr(addr, "payment_link_send_attempts", 0) > 0:
            return True
    return bool(getattr(context, "payment_link_sent", False))


def _order_already_submitted(context: Any) -> bool:
    """Duck-typed check for order already submitted."""
    return bool(getattr(context, "order_submitted", False))
