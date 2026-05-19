# app/nlu/turn_resolver/waiting_option_validator.py
"""Validates WaitingOptionResolution before any handler applies it.

validate_waiting_option_resolution() must be called by every handler before
acting on a GPT result.  It is the final safety gate — a handler that bypasses
validation and applies an invalid result directly violates the safety contract.

Safety contract
---------------
* Never raises into callers.
* Control actions (list_options, skip, cancel, etc.) are always structurally
  valid; semantic validity is the handler's responsibility.
* SELECT is only valid when confidence >= threshold AND names/IDs are in the
  allowed_options for the current group.
* NEGATE is valid when the negated targets are in allowed_options.
* Unknown actions always fail validation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.nlu.turn_resolver.waiting_option_resolver import WaitingOptionResolution

# Default minimum confidence for auto-apply
_DEFAULT_MIN_CONFIDENCE: float = 0.70

# Actions that are always structurally valid — semantic checks are the
# handler's responsibility, not the validator's.
_CONTROL_ACTIONS: frozenset[str] = frozenset({
    "list_options",
    "skip",
    "cancel",
    "checkout_request",
    "change_order_type",
    "clarify",
    "fallback",
})


# ── Result dataclass ──────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Outcome of validate_waiting_option_resolution()."""

    is_valid: bool
    reason: str
    block_reason: str | None = None


VALIDATION_OK = ValidationResult(is_valid=True, reason="ok")


# ── Public API ────────────────────────────────────────────────────────────────


def validate_waiting_option_resolution(
    resolution: "WaitingOptionResolution",
    allowed_options: tuple[dict, ...],
    state: str,
    context: Any,
    *,
    min_confidence: float = _DEFAULT_MIN_CONFIDENCE,
) -> ValidationResult:
    """Validate a WaitingOptionResolution before a handler applies it.

    Parameters
    ----------
    resolution:
        The resolution returned by WaitingOptionResolver.resolve_sync().
    allowed_options:
        Tuple of option dicts from AllowedOptionExtractor (current group only).
    state:
        Current conversation state string (e.g. "waiting_for_modifier").
    context:
        ConversationContext — reserved for future min/max group checks.
    min_confidence:
        Minimum GPT confidence for SELECT to be considered safe to apply.

    Returns
    -------
    VALIDATION_OK if the resolution may be applied.
    ValidationResult(is_valid=False, reason=...) if it must not be applied.
    """
    try:
        return _validate(resolution, allowed_options, state, context, min_confidence)
    except Exception as exc:
        return ValidationResult(
            is_valid=False,
            reason="validation_exception",
            block_reason=str(exc)[:200],
        )


# ── Private helpers ───────────────────────────────────────────────────────────


def _validate(
    resolution: "WaitingOptionResolution",
    allowed_options: tuple[dict, ...],
    state: str,
    context: Any,
    min_confidence: float,
) -> ValidationResult:
    action = (resolution.action or "fallback").lower()

    # Control / informational actions — structurally always OK
    if action in _CONTROL_ACTIONS:
        return VALIDATION_OK

    if action == "select":
        return _validate_select(resolution, allowed_options, min_confidence)

    if action == "negate":
        return _validate_negate(resolution, allowed_options)

    # Unknown action
    return ValidationResult(
        is_valid=False,
        reason="unknown_action",
        block_reason=f"action={action!r} is not a recognised WaitingOptionAction",
    )


def _validate_select(
    resolution: "WaitingOptionResolution",
    allowed_options: tuple[dict, ...],
    min_confidence: float,
) -> ValidationResult:
    # 1. Confidence gate
    if resolution.confidence < min_confidence:
        return ValidationResult(
            is_valid=False,
            reason="low_confidence",
            block_reason=(
                f"confidence={resolution.confidence:.3f} < "
                f"threshold={min_confidence:.3f}"
            ),
        )

    # 2. Must have at least one selection
    has_ids = bool(resolution.selected_option_ids)
    has_names = bool(resolution.selected_option_names)
    if not has_ids and not has_names:
        return ValidationResult(
            is_valid=False,
            reason="no_selections",
            block_reason="action=select but no selected_option_ids or selected_option_names",
        )

    # 3. Build lookup sets from allowed options
    allowed_ids, allowed_names_lower = _build_allowed_sets(allowed_options)

    # 4. Validate IDs (skip empty strings)
    for opt_id in resolution.selected_option_ids:
        if opt_id and opt_id not in allowed_ids:
            return ValidationResult(
                is_valid=False,
                reason="unknown_option_id",
                block_reason=f"id={opt_id!r} not found in allowed_options",
            )

    # 5. Validate names
    for name in resolution.selected_option_names:
        if name and name.lower() not in allowed_names_lower:
            return ValidationResult(
                is_valid=False,
                reason="unknown_option_name",
                block_reason=f"name={name!r} not found in allowed_options",
            )

    return VALIDATION_OK


def _validate_negate(
    resolution: "WaitingOptionResolution",
    allowed_options: tuple[dict, ...],
) -> ValidationResult:
    # Negate with no target → let the handler decide (clarify or skip)
    if not resolution.negated_option_ids and not resolution.selected_option_names:
        return VALIDATION_OK

    allowed_ids, allowed_names_lower = _build_allowed_sets(allowed_options)

    for opt_id in resolution.negated_option_ids:
        if opt_id and opt_id not in allowed_ids:
            return ValidationResult(
                is_valid=False,
                reason="unknown_negate_id",
                block_reason=f"negated id={opt_id!r} not found in allowed_options",
            )

    return VALIDATION_OK


def _build_allowed_sets(
    allowed_options: tuple[dict, ...],
) -> tuple[set[str], set[str]]:
    """Build fast-lookup sets for IDs and lower-cased names."""
    ids: set[str] = set()
    names: set[str] = set()
    for opt in allowed_options:
        oid = str(
            opt.get("modifier_id")
            or opt.get("item_id")
            or opt.get("variant_id")
            or ""
        )
        oname = str(opt.get("name") or "").strip().lower()
        if oid:
            ids.add(oid)
        if oname:
            names.add(oname)
    return ids, names
