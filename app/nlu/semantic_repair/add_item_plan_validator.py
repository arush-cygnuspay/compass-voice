# app/nlu/semantic_repair/add_item_plan_validator.py
"""Local menu validator for GPT ADD_ITEM extracted plans.

Shadow-only — validates GptAddItemPlan.items[] against live menu data and
produces a ValidatedAddItemPlan that categorises each item and its children
as valid, warning, or rejected.

Contract
--------
* The validator is NEVER applied to cart, session state, or live response.
* Validator exceptions must never surface — the service wrapper catches all.
* Menu lookup uses pre-built O(1) indexes; overhead is < 1 ms per item.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.menu.models import MenuItem, SideGroup
    from app.menu.store import MenuStore
    from app.nlu.semantic_repair.add_item_extractor import GptAddItem, GptAddItemPlan


# ---------------------------------------------------------------------------
# Warning code constants
# ---------------------------------------------------------------------------

BLOCKING_WARNING_CODES: frozenset[str] = frozenset({
    "item_not_on_menu",       # GPT named an item not in the menu
    "item_ambiguous",         # item name maps to multiple menu items
    "invalid_item_size",      # named size/variant does not exist for this item
    "invalid_item_variant",   # alias for invalid_item_size when variant field used
    "side_not_valid_for_item",  # side not found in any side group for this item
    "invalid_side_size",      # named size/variant does not exist for this side
    "invalid_side_variant",   # alias for invalid_side_size
    "ambiguous_size_scope",   # size cannot be assigned to a specific item or side
    "over_max_selector",      # sides in a group exceed max_selector
})

NON_BLOCKING_WARNING_CODES: frozenset[str] = frozenset({
    "modifier_size_unsupported",  # modifier doesn't support size/variant
    "modifier_not_found",         # modifier not in any group (logged only)
    "duplicate_dropped",          # duplicate side dropped (allow_duplicate=False)
    "required_group_missing",     # required side group has no selection
    "quantity_clamped",           # quantity was out of safe range
})


# ---------------------------------------------------------------------------
# Output dataclasses (frozen — shadow only, never applied)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ValidationWarning:
    """A single validation warning for an item or one of its children."""

    code: str
    entity_kind: str     # "item" | "side" | "modifier"
    entity_name: str
    detail: str = ""

    @property
    def is_blocking(self) -> bool:
        return self.code in BLOCKING_WARNING_CODES


@dataclass(frozen=True, slots=True)
class ValidatedModifier:
    """A modifier successfully resolved against the item's modifier groups."""

    group_id: str
    modifier_id: str
    name: str
    operation: str = "add"    # "add" | "remove" | "replace"
    quantity: int = 1


@dataclass(frozen=True, slots=True)
class ValidatedSide:
    """A side successfully resolved against the item's side groups."""

    group_id: str
    side_item_id: str
    name: str
    quantity: int = 1
    variant_id: str | None = None
    variant_label: str | None = None


@dataclass(frozen=True, slots=True)
class ValidatedAddItem:
    """Validation result for one GPT-extracted item."""

    item_id: str
    item_name: str
    quantity: int = 1
    variant_id: str | None = None
    variant_label: str | None = None
    sides: tuple[ValidatedSide, ...] = ()
    modifiers: tuple[ValidatedModifier, ...] = ()
    missing_required_groups: tuple[str, ...] = ()   # group_ids with no selection
    warnings: tuple[ValidationWarning, ...] = ()

    @property
    def has_blocking_warnings(self) -> bool:
        return any(w.is_blocking for w in self.warnings)


@dataclass(frozen=True, slots=True)
class ValidatedAddItemPlan:
    """Full validator result for one GptAddItemPlan (shadow only)."""

    items: tuple[ValidatedAddItem, ...] = ()        # resolved items (no blocking warnings)
    rejected_items: tuple[str, ...] = ()            # item names that could not be resolved
    warnings: tuple[ValidationWarning, ...] = ()    # cross-item plan-level warnings
    validator_ms: float = 0.0
    has_blocking_warnings: bool = False

    @staticmethod
    def empty(validator_ms: float = 0.0) -> "ValidatedAddItemPlan":
        return ValidatedAddItemPlan(validator_ms=validator_ms)


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

_MAX_SAFE_QUANTITY: int = 20
_MIN_SAFE_QUANTITY: int = 1


class AddItemPlanValidator:
    """Validates a GptAddItemPlan against live menu data.

    Stateless — safe to instantiate once and reuse across turns.

    ``validate()`` never raises; all exceptions produce an empty result.
    """

    def validate(
        self,
        *,
        plan: "GptAddItemPlan",
        menu_store: "MenuStore | None" = None,
        menu_repo: Any = None,
    ) -> ValidatedAddItemPlan:
        """Validate *plan* against menu data.

        At least one of *menu_store* / *menu_repo* must be provided.
        If neither is available an empty result is returned immediately.
        """
        t_start = time.perf_counter()
        try:
            store = self._resolve_store(menu_store=menu_store, menu_repo=menu_repo)
            if store is None:
                return ValidatedAddItemPlan.empty(validator_ms=0.0)

            if not plan.items:
                return ValidatedAddItemPlan.empty(
                    validator_ms=round((time.perf_counter() - t_start) * 1000.0, 3),
                )

            validated_items: list[ValidatedAddItem] = []
            rejected_items: list[str] = []

            for gpt_item in plan.items:
                try:
                    result = self._validate_item(gpt_item, store)
                    if result is None:
                        rejected_items.append(gpt_item.item or "")
                    else:
                        validated_items.append(result)
                except Exception:
                    rejected_items.append(gpt_item.item or "")

            any_blocking = (
                any(vi.has_blocking_warnings for vi in validated_items)
                or bool(rejected_items)
            )
            validator_ms = round((time.perf_counter() - t_start) * 1000.0, 3)

            return ValidatedAddItemPlan(
                items=tuple(validated_items),
                rejected_items=tuple(rejected_items),
                warnings=(),
                validator_ms=validator_ms,
                has_blocking_warnings=any_blocking,
            )

        except Exception:
            return ValidatedAddItemPlan.empty(
                validator_ms=round((time.perf_counter() - t_start) * 1000.0, 3),
            )

    # ------------------------------------------------------------------
    # Item-level validation
    # ------------------------------------------------------------------

    def _validate_item(
        self,
        gpt_item: "GptAddItem",
        store: "MenuStore",
    ) -> ValidatedAddItem | None:
        """Return a ValidatedAddItem or None when the item has blocking errors."""
        from app.nlu.query_normalization.text_preprocessor import normalize_text

        warnings: list[ValidationWarning] = []
        item_name_raw = (gpt_item.item or "").strip()
        if not item_name_raw:
            return None

        norm_name = normalize_text(item_name_raw)

        # 1. Resolve item
        menu_item = self._resolve_menu_item(norm_name, store, item_name_raw, warnings)
        if menu_item is None:
            return None  # blocking warning already appended; caller adds to rejected

        # 2. Quantity
        quantity = self._clamp_quantity(gpt_item.quantity, item_name_raw, warnings)

        # 3. Item size / variant
        variant_id, variant_label = self._resolve_item_size_variant(
            gpt_item=gpt_item,
            menu_item=menu_item,
            warnings=warnings,
        )

        # 4. Sides
        validated_sides = self._validate_sides(
            gpt_item=gpt_item,
            menu_item=menu_item,
            store=store,
            warnings=warnings,
        )

        # 5. Modifiers
        validated_modifiers = self._validate_modifiers(
            gpt_item=gpt_item,
            menu_item=menu_item,
            store=store,
            warnings=warnings,
        )

        # 6. Required side-group coverage
        missing_required = self._check_required_groups(
            menu_item=menu_item,
            validated_sides=validated_sides,
            warnings=warnings,
        )

        # If any blocking warning, treat item as rejected
        if any(w.is_blocking for w in warnings):
            return None

        return ValidatedAddItem(
            item_id=menu_item.item_id,
            item_name=menu_item.name,
            quantity=quantity,
            variant_id=variant_id,
            variant_label=variant_label,
            sides=tuple(validated_sides),
            modifiers=tuple(validated_modifiers),
            missing_required_groups=tuple(missing_required),
            warnings=tuple(warnings),
        )

    # ------------------------------------------------------------------
    # Step helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_menu_item(
        norm_name: str,
        store: "MenuStore",
        raw_name: str,
        warnings: list[ValidationWarning],
    ) -> "MenuItem | None":
        # 1. Exact normalized name (O(1))
        item = store.find_item_exact(norm_name)
        if item is not None:
            return item

        # 2. Alias index
        alias_ids = store.find_item_ids_by_alias(norm_name)
        if alias_ids:
            if len(alias_ids) == 1:
                found = store.items.get(alias_ids[0])
                if found is not None:
                    return found
            warnings.append(ValidationWarning(
                code="item_ambiguous",
                entity_kind="item",
                entity_name=raw_name,
                detail=f"alias matched {len(alias_ids)} items",
            ))
            return None

        # 3. Voice label index
        vl_ids = store.find_item_ids_by_voice_label(norm_name)
        if vl_ids:
            if len(vl_ids) == 1:
                found = store.items.get(vl_ids[0])
                if found is not None:
                    return found
            warnings.append(ValidationWarning(
                code="item_ambiguous",
                entity_kind="item",
                entity_name=raw_name,
                detail=f"voice label matched {len(vl_ids)} items",
            ))
            return None

        # 4. Not found — blocking
        warnings.append(ValidationWarning(
            code="item_not_on_menu",
            entity_kind="item",
            entity_name=raw_name,
            detail="no exact, alias, or voice label match in menu",
        ))
        return None

    @staticmethod
    def _clamp_quantity(
        raw: int | None,
        entity_name: str,
        warnings: list[ValidationWarning],
    ) -> int:
        if raw is None:
            return 1
        clamped = max(_MIN_SAFE_QUANTITY, min(_MAX_SAFE_QUANTITY, raw))
        if clamped != raw:
            warnings.append(ValidationWarning(
                code="quantity_clamped",
                entity_kind="item",
                entity_name=entity_name,
                detail=f"quantity {raw!r} clamped to {clamped}",
            ))
        return clamped

    @staticmethod
    def _resolve_item_size_variant(
        *,
        gpt_item: "GptAddItem",
        menu_item: "MenuItem",
        warnings: list[ValidationWarning],
    ) -> tuple[str | None, str | None]:
        from app.nlu.query_normalization.text_preprocessor import normalize_text

        # Prefer explicit size; fall back to variant field
        size_raw = ((gpt_item.size or "") or (gpt_item.variant or "")).strip()
        if not size_raw:
            return None, None

        pricing = menu_item.pricing

        if pricing.mode in ("fixed", "unit"):
            # Item has no variants — size specification is invalid
            warnings.append(ValidationWarning(
                code="invalid_item_size",
                entity_kind="item",
                entity_name=menu_item.name,
                detail=f"item has {pricing.mode} pricing; size '{size_raw}' not valid",
            ))
            return None, None

        if pricing.mode == "variant" and pricing.variants:
            norm_size = normalize_text(size_raw)
            for pv in pricing.variants:
                if pv.normalized_label == norm_size or normalize_text(pv.label) == norm_size:
                    return pv.variant_id, pv.label
            # No match — blocking
            labels = [v.label for v in pricing.variants]
            warnings.append(ValidationWarning(
                code="invalid_item_size",
                entity_kind="item",
                entity_name=menu_item.name,
                detail=f"size '{size_raw}' not in {labels}",
            ))

        return None, None

    def _validate_sides(
        self,
        *,
        gpt_item: "GptAddItem",
        menu_item: "MenuItem",
        store: "MenuStore",
        warnings: list[ValidationWarning],
    ) -> list[ValidatedSide]:
        from app.nlu.query_normalization.text_preprocessor import normalize_text

        validated: list[ValidatedSide] = []
        group_selection_counts: dict[str, int] = {}
        selected_side_keys: set[tuple[str, str]] = set()  # (group_id, side_item_id)

        for child in gpt_item.sides:
            child_name_raw = (child.name or "").strip()
            if not child_name_raw:
                continue
            norm_child = normalize_text(child_name_raw)

            # Find which side group(s) contain this side
            matched_groups: list[tuple["SideGroup", str]] = []
            for sg in menu_item.side_groups:
                ids = store.find_side_ids_for_group_by_label(sg.group_id, norm_child)
                if ids:
                    matched_groups.append((sg, ids[0]))

            if not matched_groups:
                warnings.append(ValidationWarning(
                    code="side_not_valid_for_item",
                    entity_kind="side",
                    entity_name=child_name_raw,
                    detail=f"not in any side group for '{menu_item.name}'",
                ))
                continue

            # Pick first matching group
            sg, side_item_id = matched_groups[0]

            # Duplicate guard
            dup_key = (sg.group_id, side_item_id)
            if dup_key in selected_side_keys and not sg.allow_duplicate_selections:
                warnings.append(ValidationWarning(
                    code="duplicate_dropped",
                    entity_kind="side",
                    entity_name=child_name_raw,
                    detail=f"group '{sg.name}' disallows duplicate selections",
                ))
                continue
            selected_side_keys.add(dup_key)

            # Max-selector guard
            new_count = group_selection_counts.get(sg.group_id, 0) + 1
            group_selection_counts[sg.group_id] = new_count
            if new_count > sg.max_selector:
                warnings.append(ValidationWarning(
                    code="over_max_selector",
                    entity_kind="side",
                    entity_name=child_name_raw,
                    detail=f"group '{sg.name}' max_selector={sg.max_selector}, count={new_count}",
                ))
                continue

            # Side size / variant
            size_raw = ((child.size or "") or (child.variant or "")).strip()
            variant_id: str | None = None
            variant_label: str | None = None

            if size_raw:
                side_choice = next(
                    (c for c in sg.choices if c.item_id == side_item_id), None
                )
                if side_choice is not None:
                    sp = side_choice.pricing
                    if sp.mode in ("fixed", "unit"):
                        warnings.append(ValidationWarning(
                            code="invalid_side_size",
                            entity_kind="side",
                            entity_name=child_name_raw,
                            detail=f"side has {sp.mode} pricing; size '{size_raw}' not valid",
                        ))
                    elif sp.mode == "variant" and sp.variants:
                        norm_size = normalize_text(size_raw)
                        for pv in sp.variants:
                            if pv.normalized_label == norm_size or normalize_text(pv.label) == norm_size:
                                variant_id = pv.variant_id
                                variant_label = pv.label
                                break
                        if variant_id is None:
                            side_labels = [v.label for v in sp.variants]
                            warnings.append(ValidationWarning(
                                code="invalid_side_size",
                                entity_kind="side",
                                entity_name=child_name_raw,
                                detail=f"size '{size_raw}' not in {side_labels}",
                            ))

            validated.append(ValidatedSide(
                group_id=sg.group_id,
                side_item_id=side_item_id,
                name=child_name_raw,
                quantity=self._clamp_quantity(child.quantity, child_name_raw, []),
                variant_id=variant_id,
                variant_label=variant_label,
            ))

        return validated

    def _validate_modifiers(
        self,
        *,
        gpt_item: "GptAddItem",
        menu_item: "MenuItem",
        store: "MenuStore",
        warnings: list[ValidationWarning],
    ) -> list[ValidatedModifier]:
        from app.nlu.query_normalization.text_preprocessor import normalize_text

        validated: list[ValidatedModifier] = []

        for child in gpt_item.modifiers:
            child_name_raw = (child.name or "").strip()
            if not child_name_raw:
                continue
            norm_child = normalize_text(child_name_raw)

            # Find first modifier group containing this modifier
            matched: tuple[str, str] | None = None  # (group_id, modifier_id)
            for mg in menu_item.modifier_groups:
                ids = store.find_modifier_ids_for_group_by_label(mg.group_id, norm_child)
                if ids:
                    matched = (mg.group_id, ids[0])
                    break

            if matched is None:
                warnings.append(ValidationWarning(
                    code="modifier_not_found",
                    entity_kind="modifier",
                    entity_name=child_name_raw,
                    detail=f"not in any modifier group for '{menu_item.name}'",
                ))
                continue

            # Size on modifier — non-blocking
            if (child.size or child.variant or "").strip():
                warnings.append(ValidationWarning(
                    code="modifier_size_unsupported",
                    entity_kind="modifier",
                    entity_name=child_name_raw,
                    detail="modifiers do not support size/variant in current schema",
                ))

            group_id, modifier_id = matched
            validated.append(ValidatedModifier(
                group_id=group_id,
                modifier_id=modifier_id,
                name=child_name_raw,
                operation=(child.operation or "add"),
                quantity=self._clamp_quantity(child.quantity, child_name_raw, []),
            ))

        return validated

    @staticmethod
    def _check_required_groups(
        *,
        menu_item: "MenuItem",
        validated_sides: list[ValidatedSide],
        warnings: list[ValidationWarning],
    ) -> list[str]:
        """Return group_ids of required side groups that have no selection."""
        selected_group_ids: set[str] = {vs.group_id for vs in validated_sides}
        missing: list[str] = []

        for sg in menu_item.side_groups:
            if (
                sg.is_required
                and sg.min_selector > 0
                and not sg.is_suggested_addon
                and sg.group_id not in selected_group_ids
            ):
                missing.append(sg.group_id)
                warnings.append(ValidationWarning(
                    code="required_group_missing",
                    entity_kind="side",
                    entity_name=sg.name,
                    detail=f"required group '{sg.name}' has no selection (min={sg.min_selector})",
                ))

        return missing

    # ------------------------------------------------------------------
    # Store resolution helper
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_store(*, menu_store: Any, menu_repo: Any) -> Any:
        if menu_store is not None:
            return menu_store
        if menu_repo is not None:
            store = getattr(menu_repo, "store", None)
            if store is not None:
                return store
            getter = getattr(menu_repo, "get_store", None)
            if callable(getter):
                return getter()
        return None

    def validate_planner_items(
        self,
        *,
        planner_items: "tuple[Any, ...]",
        menu_store: "MenuStore | None" = None,
        menu_repo: Any = None,
    ) -> ValidatedAddItemPlan:
        """Validate Phase 4 PlannerGptItem list against live menu data.

        Adapts PlannerGptItem objects into GptAddItem-compatible objects and
        delegates to the existing validate() method via a synthetic plan.

        Parameters
        ----------
        planner_items:
            Tuple of PlannerGptItem from the Phase 4 planner output.
        menu_store / menu_repo:
            Menu data source (same as validate()).
        """
        from app.nlu.semantic_repair.add_item_extractor import (
            GptAddItem,
            GptAddItemChild,
            GptAddItemPlan,
        )

        # Convert PlannerGptItem → GptAddItem (drop candidate_item_id, adapt operations)
        gpt_items: list[GptAddItem] = []
        for pi in (planner_items or ()):
            mods = tuple(
                GptAddItemChild(
                    name=getattr(m, "name", ""),
                    operation=getattr(m, "operation", "add"),
                    quantity=getattr(m, "quantity", 1),
                )
                for m in (getattr(pi, "modifiers", ()) or ())
            )
            sides = tuple(
                GptAddItemChild(
                    name=getattr(s, "name", ""),
                    operation="add",
                    quantity=getattr(s, "quantity", 1),
                    size=getattr(s, "size", None),
                )
                for s in (getattr(pi, "sides", ()) or ())
            )
            gpt_items.append(GptAddItem(
                item=getattr(pi, "item_name", "") or "",
                quantity=getattr(pi, "quantity", None),
                size=getattr(pi, "size", None),
                variant=getattr(pi, "variant", None),
                modifiers=mods,
                sides=sides,
            ))

        synthetic_plan = GptAddItemPlan(
            decision="add_items",
            items=tuple(gpt_items),
        )
        return self.validate(
            plan=synthetic_plan,
            menu_store=menu_store,
            menu_repo=menu_repo,
        )


# ---------------------------------------------------------------------------
# Phase 4 Apply Gate
# ---------------------------------------------------------------------------


class PlannerApplyGate:
    """Decides whether a validated Phase 4 plan is safe to apply.

    The apply gate is the single authority that controls whether GPT planner
    results are allowed to feed into the FSM / cart mutation path.

    Contract
    --------
    * should_apply() returns (safe: bool, reason: str).
    * Returns (False, reason) whenever ANY blocking condition is true.
    * Returns (True, "approved") only when ALL conditions pass.
    * Never raises — all exceptions produce (False, "gate_exception").
    * Mode "shadow" always produces (False, "shadow_mode_never_applies").
    * Mode "disabled" always produces (False, "mode_disabled").
    """

    def should_apply(
        self,
        *,
        route_mode: str,
        decision: str,
        validated_plan: "ValidatedAddItemPlan | None",
        confidence: float | None,
        min_confidence: float,
        parse_error: str | None = None,
        gpt_called: bool = True,
        timed_out: bool = False,
    ) -> tuple[bool, str]:
        """Return (safe_to_apply, reason).

        Parameters
        ----------
        route_mode:
            "no_gpt" | "shadow_gpt" | "inline_gpt"
        decision:
            GPT decision string (must be "add_items" to proceed).
        validated_plan:
            ValidatedAddItemPlan from AddItemPlanValidator (None → not run).
        confidence:
            GPT confidence (0.0–1.0).  None → treated as 0.0.
        min_confidence:
            Minimum threshold for apply gate approval.
        parse_error:
            Non-None → gate rejects.
        gpt_called:
            False → gate rejects (nothing to apply).
        timed_out:
            True → gate rejects.
        """
        try:
            return self._evaluate(
                route_mode=route_mode,
                decision=decision,
                validated_plan=validated_plan,
                confidence=confidence,
                min_confidence=min_confidence,
                parse_error=parse_error,
                gpt_called=gpt_called,
                timed_out=timed_out,
            )
        except Exception:
            return False, "gate_exception"

    @staticmethod
    def _evaluate(
        *,
        route_mode: str,
        decision: str,
        validated_plan: "ValidatedAddItemPlan | None",
        confidence: float | None,
        min_confidence: float,
        parse_error: str | None,
        gpt_called: bool,
        timed_out: bool,
    ) -> tuple[bool, str]:
        # Gate 1: mode must be inline
        if route_mode != "inline_gpt":
            return False, "shadow_mode_never_applies" if route_mode == "shadow_gpt" else "not_inline_mode"

        # Gate 2: GPT must have been called
        if not gpt_called:
            return False, "gpt_not_called"

        # Gate 3: no timeout
        if timed_out:
            return False, "gpt_timeout"

        # Gate 4: no parse error
        if parse_error:
            return False, f"parse_error:{parse_error[:80]}"

        # Gate 5: decision must be "add_items"
        if decision != "add_items":
            return False, f"decision_not_add_items:{decision}"

        # Gate 6: confidence >= threshold
        conf = float(confidence) if confidence is not None else 0.0
        if conf < min_confidence:
            return False, f"confidence_too_low:{conf:.3f}<{min_confidence:.3f}"

        # Gate 7: validated plan must exist and pass
        if validated_plan is None:
            return False, "validator_not_run"

        if validated_plan.has_blocking_warnings:
            return False, "validator_blocking_warnings"

        if not validated_plan.items:
            return False, "no_valid_items"

        # Gate 8: all validated items must have a resolved item_id
        for vi in validated_plan.items:
            if not getattr(vi, "item_id", ""):
                return False, f"unresolved_item_id:{getattr(vi, 'item_name', '?')}"

        return True, "approved"
