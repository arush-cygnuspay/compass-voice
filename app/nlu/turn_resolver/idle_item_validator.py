# app/nlu/turn_resolver/idle_item_validator.py
"""Validates IdleItemResolution before any handler applies it.

validate_idle_item_resolution() is the safety gate for bucket-0 GPT results.
It must be called before routing to AddItemHandler or any cart-touching code.

Safety contract
---------------
* Never raises into callers.
* Never mutates cart, context, or FSM state.
* Returns ValidationResult(is_valid=False) on any failure — callers must
  fall back to local deterministic path.
* "6 piece wings" must NOT become quantity=6 — only item_name + variant.
* item_id validation is advisory only (IDs may not be known at resolver time);
  item_name validation against menu_candidates is the primary gate.
* Confidence below threshold → reject.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

# Reuse the same ValidationResult type as the waiting-state validator
# (is_valid / reason / block_reason fields).
from app.nlu.turn_resolver.waiting_option_validator import ValidationResult, VALIDATION_OK

if TYPE_CHECKING:
    from app.nlu.turn_resolver.idle_item_resolver import IdleItemResolution

# Allowed decision values from the resolver
_APPLY_DECISIONS: frozenset[str] = frozenset({"execute"})
_CONTROL_DECISIONS: frozenset[str] = frozenset({"clarify", "reject", "fallback"})

# Intents allowed in idle state
_IDLE_ALLOWED_INTENTS: frozenset[str] = frozenset({
    "add_item",
    "ask_item_info",
    "ask_menu",
    "checkout",
    "cancel",
    "unknown",
})

# Minimum confidence for auto-apply
_DEFAULT_MIN_CONFIDENCE: float = 0.70

# Words that should NOT inflate the quantity from a variant name
_VARIANT_QUANTITY_WORDS: frozenset[str] = frozenset({
    "piece", "pieces", "pc", "pcs", "wing", "wings", "bone",
    "boneless", "nugget", "nuggets", "strip", "strips",
})


def validate_idle_item_resolution(
    resolution: "IdleItemResolution",
    menu_candidates: tuple[dict, ...],
    context: Any,
    *,
    min_confidence: float = _DEFAULT_MIN_CONFIDENCE,
) -> ValidationResult:
    """Validate an IdleItemResolution before applying it.

    Parameters
    ----------
    resolution:
        The resolution returned by IdleItemResolver.resolve_sync().
    menu_candidates:
        Tuple of candidate dicts from MenuCandidateProvider — used to verify
        that GPT-returned item names are actually in the candidate set.
    context:
        ConversationContext — reserved for future group-min/max checks.
    min_confidence:
        Minimum GPT confidence for the resolution to be safe to apply.

    Returns
    -------
    VALIDATION_OK (is_valid=True) when safe to apply.
    ValidationResult(is_valid=False, reason=...) when the result must NOT be applied.
    """
    try:
        return _validate(resolution, menu_candidates, context, min_confidence)
    except Exception as exc:
        return ValidationResult(
            is_valid=False,
            reason="validation_exception",
            block_reason=str(exc)[:200],
        )


# ── Private helpers ───────────────────────────────────────────────────────────


def _validate(
    resolution: "IdleItemResolution",
    menu_candidates: tuple[dict, ...],
    context: Any,
    min_confidence: float,
) -> ValidationResult:
    decision = (resolution.decision or "fallback").lower()

    # Control / informational decisions — structurally always OK.
    if decision in _CONTROL_DECISIONS:
        return VALIDATION_OK

    if decision != "execute":
        return ValidationResult(
            is_valid=False,
            reason="unknown_decision",
            block_reason=f"decision={decision!r} is not a recognised IdleItemDecision",
        )

    # ── Execute path validation ───────────────────────────────────────────────

    # 1. Intent must be allowed in idle
    intent = (resolution.intent or "unknown").lower()
    if intent not in _IDLE_ALLOWED_INTENTS:
        return ValidationResult(
            is_valid=False,
            reason="invalid_intent_for_idle",
            block_reason=f"intent={intent!r} is not allowed in idle state",
        )

    # 2. Confidence gate
    if resolution.confidence < min_confidence:
        return ValidationResult(
            is_valid=False,
            reason="low_confidence",
            block_reason=(
                f"confidence={resolution.confidence:.3f} < "
                f"threshold={min_confidence:.3f}"
            ),
        )

    # 3. Must have at least one item in plan
    if not resolution.item_plan:
        return ValidationResult(
            is_valid=False,
            reason="empty_item_plan",
            block_reason="decision=execute but item_plan is empty",
        )

    # 4. Build candidate lookup set (names, lower-cased)
    candidate_names_lower: set[str] = {
        str(c.get("name") or "").strip().lower()
        for c in menu_candidates
        if isinstance(c, dict) and c.get("name")
    }

    # 5. Validate each resolved item
    for item in resolution.item_plan:
        result = _validate_item(item, candidate_names_lower)
        if not result.is_valid:
            return result

    return VALIDATION_OK


def _validate_item(item: Any, candidate_names_lower: set[str]) -> ValidationResult:
    """Validate one IdleResolvedItem."""
    item_name = str(getattr(item, "item_name", "") or "").strip()
    if not item_name:
        return ValidationResult(
            is_valid=False,
            reason="missing_item_name",
            block_reason="item_plan entry has no item_name",
        )

    # 6. Item name must be in candidates (when candidates are available)
    if candidate_names_lower and item_name.lower() not in candidate_names_lower:
        return ValidationResult(
            is_valid=False,
            reason="item_not_in_candidates",
            block_reason=(
                f"item_name={item_name!r} not found in menu_candidates"
            ),
        )

    # 7. Quantity sanity — prevent "6 piece wings" from becoming quantity=6
    quantity = int(getattr(item, "quantity", 1) or 1)
    variant_name = str(getattr(item, "variant_name", "") or "").strip().lower()
    if quantity > 1 and _quantity_looks_like_variant(quantity, variant_name):
        return ValidationResult(
            is_valid=False,
            reason="quantity_is_variant_not_count",
            block_reason=(
                f"quantity={quantity} looks like a variant piece count, "
                f"not a customer-requested count. variant={variant_name!r}"
            ),
        )

    # 8. Quantity must be reasonable (1–20)
    if quantity < 1 or quantity > 20:
        return ValidationResult(
            is_valid=False,
            reason="invalid_quantity",
            block_reason=f"quantity={quantity} is out of range [1, 20]",
        )

    return VALIDATION_OK


def _quantity_looks_like_variant(quantity: int, variant_name: str) -> bool:
    """Return True when quantity is suspicious (likely a piece count, not a repeat order).

    Heuristic: if the variant_name contains a piece-count word AND the quantity
    matches a typical piece-count value (4, 6, 8, 10, 12, 24), flag as variant.
    The GPT prompt instructs it to treat "6 piece wings" as variant=6pc, qty=1,
    but a broken GPT might set quantity=6.  Validated here.
    """
    if not variant_name:
        return False
    has_variant_word = any(w in variant_name for w in _VARIANT_QUANTITY_WORDS)
    if not has_variant_word:
        return False
    # Common piece counts for wings/nuggets/etc.
    _PIECE_COUNTS: frozenset[int] = frozenset({4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 50})
    return quantity in _PIECE_COUNTS
