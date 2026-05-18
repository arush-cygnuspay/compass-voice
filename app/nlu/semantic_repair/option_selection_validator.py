# app/nlu/semantic_repair/option_selection_validator.py
"""Phase 3 GPT option selection validator.

Validates GPT-returned option names against the current modifier group.
Never raises — always returns a (possibly modified) OptionResolverResult.

Validation rules (all must pass for safe_to_apply=True)
--------------------------------------------------------
1. decision must be "select_option" with at least one name.
2. route_mode must be "inline_gpt" (shadow is never safe to apply).
3. confidence >= min_confidence threshold.
4. Count of selected_names must not exceed group.max_selector
   (when max_selector > 0).
5. Every name in selected_names must case-insensitively match a
   choice.name in the modifier group (all-or-nothing — any
   unresolvable name marks the whole result unsafe).
6. Duplicate modifier IDs (same name selected twice, or name that
   maps to an already-existing selection ID) are silently handled
   downstream — the validator allows single-group duplicates
   through but build_modifier_selections_from_names() deduplicates.

Notes on name-based validation
-------------------------------
Validation is name-based because GPT returns the canonical choice name
and modifier_ids are internal opaque keys the LLM does not see.
Name lookup is group-scoped: the validator only checks names against
the CURRENT modifier group's choices, so a valid name in a different
group is rejected.  This property is intentional and documented.
"""
from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.nlu.semantic_repair.option_resolver_result import OptionResolverResult
    from app.state_machine.models.pending_item_models import (
        ModifierSelection,
        PendingModifierGroup,
    )


class GptOptionSelectionValidator:
    """Validate GPT option names against the current modifier group choices.

    Instantiate once and call validate() per resolver result.  Thread-safe.
    """

    def validate(
        self,
        *,
        result: "OptionResolverResult",
        group: "PendingModifierGroup",
        min_confidence: float = 0.75,
    ) -> "OptionResolverResult":
        """Return a new OptionResolverResult with safe_to_apply set correctly.

        Parameters
        ----------
        result:
            Raw OptionResolverResult from GptOptionResolverService.
        group:
            The PendingModifierGroup whose choices are the allowed options.
        min_confidence:
            Minimum confidence threshold (from config.option_resolver_min_confidence).
        """
        # Rule 1: only "select_option" with names can ever be safe.
        if result.decision != "select_option" or not result.selected_names:
            return replace(result, safe_to_apply=False)

        # Rule 2: shadow mode is never safe to apply.
        if result.route_mode != "inline_gpt":
            return replace(result, safe_to_apply=False)

        # Rule 3: confidence threshold.
        if result.confidence < min_confidence:
            return replace(result, safe_to_apply=False)

        # Rule 4: max_selector guard.
        # group.max_selector == 0 means unlimited; otherwise cap is enforced.
        max_sel = getattr(group, "max_selector", 0)
        if max_sel > 0 and len(result.selected_names) > max_sel:
            return replace(result, safe_to_apply=False)

        # Rule 5: every selected name must exist in the current group (case-insensitive).
        name_to_id: dict[str, str] = {
            (choice.name or "").strip().lower(): choice.modifier_id
            for choice in group.choices
            if choice.name
        }
        seen_ids: set[str] = set()
        all_valid = True
        for name in result.selected_names:
            key = (name or "").strip().lower()
            mid = name_to_id.get(key)
            if mid is None:
                all_valid = False
                break
            seen_ids.add(mid)

        safe = all_valid and bool(seen_ids)
        return replace(result, safe_to_apply=safe)


def build_modifier_selections_from_names(
    *,
    selected_names: tuple[str, ...],
    group: "PendingModifierGroup",
    existing_ids: set[str] | None = None,
) -> list["ModifierSelection"]:
    """Map GPT-returned option names to ModifierSelection instances.

    Parameters
    ----------
    selected_names:
        Option names returned by GPT (from OptionResolverResult.selected_names).
    group:
        The PendingModifierGroup whose choices are the allowed options.
    existing_ids:
        Modifier IDs already selected in this group — duplicates are skipped.

    Returns
    -------
    list[ModifierSelection]
        Empty list if ANY name fails to resolve to a valid modifier_id in
        the current group (all-or-nothing policy: partial results are rejected
        to prevent partially-applied GPT decisions).
    """
    from app.state_machine.models.pending_item_models import ModifierSelection

    # Build case-insensitive name → choice lookup (group-scoped only).
    name_to_choice: dict[str, object] = {}
    for choice in group.choices:
        key = (choice.name or "").strip().lower()
        if key:
            name_to_choice[key] = choice

    _existing = existing_ids or set()
    selections: list[ModifierSelection] = []
    seen: set[str] = set()

    for name in selected_names:
        key = (name or "").strip().lower()
        choice = name_to_choice.get(key)
        if choice is None:
            # Unresolvable name → abort the entire batch (all-or-nothing).
            return []
        mid = choice.modifier_id  # type: ignore[attr-defined]
        if mid in _existing or mid in seen:
            continue  # skip already-selected or within-batch duplicate
        seen.add(mid)
        selections.append(
            ModifierSelection(
                modifier_id=mid,
                name=choice.name,  # type: ignore[attr-defined]
                action="add",
                instruction=None,
            )
        )

    return selections
