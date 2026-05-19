# app/nlu/turn_resolver/validators.py
"""Validation gates for GPT turn-resolution output.

Each validator takes a ``GptTurnResolution`` and domain context, and returns
a ``ValidationResult`` indicating whether the result is safe to apply.

Safety contract
---------------
* ``safe_to_apply`` is ONLY set True here — never in GPT parsing code.
* Validation is always deterministic and local — no GPT calls.
* Validators never mutate cart, session, or FSM state.
* A failed validation always falls back to local deterministic path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Collection

from app.nlu.turn_resolver.schemas import GptTurnResolution

# Minimum confidence for safe application when not overridden by config.
_DEFAULT_MIN_CONFIDENCE: float = 0.75


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Outcome of a turn-resolution validation check.

    Fields
    ------
    is_safe:
        True when all gates pass and it is safe to apply the GPT result.
    reject_reason:
        Short code explaining why the result was rejected (when is_safe=False).
        None when is_safe=True.
    """

    is_safe: bool
    reject_reason: str | None = None


# Reusable sentinels
_SAFE = ValidationResult(is_safe=True)


def _reject(reason: str) -> ValidationResult:
    return ValidationResult(is_safe=False, reject_reason=reason)


# ---------------------------------------------------------------------------
# Bucket 0: idle_menu_item_resolution
# ---------------------------------------------------------------------------


def validate_bucket0_result(
    gpt_result: GptTurnResolution,
    allowed_intents: Collection[str],
    *,
    min_confidence: float = _DEFAULT_MIN_CONFIDENCE,
) -> ValidationResult:
    """Validate Bucket 0 (idle low-confidence item resolution) GPT output.

    Gates
    -----
    1. decision must be "add_items" or "clarify" (not "error" / "skipped")
    2. If decision == "add_items", items must be non-empty
    3. Each item_name must be non-empty
    4. If intent is provided, it must be in allowed_intents
    5. confidence must be >= min_confidence for safe apply

    Parameters
    ----------
    gpt_result:
        The GptTurnResolution produced by the GPT call.
    allowed_intents:
        Intent strings that are valid for the current state.
        If empty, intent gate is skipped.
    min_confidence:
        Minimum GPT confidence for safe application.
    """
    if gpt_result.decision in {"skipped", "error"}:
        return _reject(f"decision_{gpt_result.decision}")

    if gpt_result.decision == "add_items":
        if not gpt_result.items:
            return _reject("add_items_no_items")
        for item in gpt_result.items:
            if not (item.item_name or "").strip():
                return _reject("item_name_empty")

    if gpt_result.intent and allowed_intents:
        if gpt_result.intent not in allowed_intents:
            return _reject(f"intent_not_allowed:{gpt_result.intent}")

    conf = gpt_result.confidence
    if conf is not None and conf < min_confidence:
        return _reject(f"confidence_below_threshold:{conf:.2f}")

    return _SAFE


# ---------------------------------------------------------------------------
# Bucket 2: option_resolution
# ---------------------------------------------------------------------------


def validate_bucket2_result(
    gpt_result: GptTurnResolution,
    choice_names: Collection[str],
    *,
    min_confidence: float = _DEFAULT_MIN_CONFIDENCE,
) -> ValidationResult:
    """Validate Bucket 2 (waiting-state option resolution) GPT output.

    Gates
    -----
    1. decision must be "select_option" (others are non-actionable)
    2. selected_option_names must be non-empty
    3. All selected names must appear in choice_names (case-insensitive)
    4. confidence must be >= min_confidence

    Parameters
    ----------
    gpt_result:
        The GptTurnResolution produced by the GPT call.
    choice_names:
        The option names available in the current modifier/side group.
    min_confidence:
        Minimum GPT confidence for safe application.
    """
    if gpt_result.decision != "select_option":
        return _reject(f"decision_not_select_option:{gpt_result.decision}")

    if not gpt_result.selected_option_names:
        return _reject("no_selected_option_names")

    # Normalize for case-insensitive matching
    normalized_choices = {n.strip().lower() for n in choice_names if n}
    if normalized_choices:
        unmatched = [
            name for name in gpt_result.selected_option_names
            if name.strip().lower() not in normalized_choices
        ]
        if unmatched:
            return _reject(f"option_names_not_in_choices:{','.join(unmatched[:3])}")

    conf = gpt_result.confidence
    if conf is not None and conf < min_confidence:
        return _reject(f"confidence_below_threshold:{conf:.2f}")

    return _SAFE


# ---------------------------------------------------------------------------
# Bucket 3: multi_item_add_planning
# ---------------------------------------------------------------------------


def validate_bucket3_result(
    gpt_result: GptTurnResolution,
    known_item_names: Collection[str],
    *,
    min_confidence: float = _DEFAULT_MIN_CONFIDENCE,
    require_menu_match: bool = False,
) -> ValidationResult:
    """Validate Bucket 3 (multi-item add planning) GPT output.

    Gates
    -----
    1. decision must be "add_items"
    2. items must be non-empty
    3. Each item_name must be non-empty
    4. If require_menu_match=True, all item_names must be in known_item_names
    5. confidence must be >= min_confidence

    Parameters
    ----------
    gpt_result:
        The GptTurnResolution produced by the GPT call.
    known_item_names:
        Resolved menu item names for candidate matching.
        If empty (when require_menu_match=False), item name gate is skipped.
    min_confidence:
        Minimum GPT confidence for safe application.
    require_menu_match:
        When True, all item names must resolve to a known menu item.
        Default False (log-only mode does not need menu validation).
    """
    if gpt_result.decision != "add_items":
        return _reject(f"decision_not_add_items:{gpt_result.decision}")

    if not gpt_result.items:
        return _reject("no_items")

    for item in gpt_result.items:
        if not (item.item_name or "").strip():
            return _reject("item_name_empty")

    if require_menu_match and known_item_names:
        normalized_known = {n.strip().lower() for n in known_item_names if n}
        unresolved = [
            it.item_name for it in gpt_result.items
            if it.item_name.strip().lower() not in normalized_known
        ]
        if unresolved:
            return _reject(f"items_not_on_menu:{','.join(unresolved[:3])}")

    conf = gpt_result.confidence
    if conf is not None and conf < min_confidence:
        return _reject(f"confidence_below_threshold:{conf:.2f}")

    return _SAFE


# ---------------------------------------------------------------------------
# Dispatch helper
# ---------------------------------------------------------------------------


def validate_gpt_result(
    bucket: str,
    gpt_result: GptTurnResolution,
    *,
    allowed_intents: Collection[str] = (),
    choice_names: Collection[str] = (),
    known_item_names: Collection[str] = (),
    min_confidence: float = _DEFAULT_MIN_CONFIDENCE,
) -> ValidationResult:
    """Dispatch to the correct per-bucket validator.

    Parameters
    ----------
    bucket:
        The bucket name from ``pick_bucket()`` (BUCKET_* constants).
    gpt_result:
        The GptTurnResolution to validate.
    allowed_intents:
        For Bucket 0 intent gate (may be empty to skip).
    choice_names:
        For Bucket 2 option-name gate.
    known_item_names:
        For Bucket 3 menu-match gate (require_menu_match=False by default).
    min_confidence:
        Minimum confidence for all buckets.
    """
    from app.nlu.turn_resolver.bucket_policy import (
        BUCKET_IDLE_ITEM,
        BUCKET_MULTI_ITEM,
        BUCKET_OPTION,
    )

    if bucket == BUCKET_IDLE_ITEM:
        return validate_bucket0_result(
            gpt_result,
            allowed_intents,
            min_confidence=min_confidence,
        )
    if bucket == BUCKET_OPTION:
        return validate_bucket2_result(
            gpt_result,
            choice_names,
            min_confidence=min_confidence,
        )
    if bucket == BUCKET_MULTI_ITEM:
        return validate_bucket3_result(
            gpt_result,
            known_item_names,
            min_confidence=min_confidence,
        )

    return _reject(f"unknown_bucket:{bucket}")
