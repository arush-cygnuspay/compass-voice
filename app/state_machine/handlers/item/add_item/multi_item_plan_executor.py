# app/state_machine/handlers/item/add_item/multi_item_plan_executor.py
"""apply_multi_item_plan — convert any validated multi-item plan into a HandlerResult.

Fixes the critical collapse bug where both SmartTurnPlanner and the GPT add-item
planner discarded items[1..N] from validated multi-item plans:

  _apply_smart_plan_add_item  — read only items[0]
  _apply_planner_result       — returned None when len(validated_items) != 1

This module provides a single function that converts any plan object with an
`items` attribute (duck-typed — works with SmartTurnPlan, GptAddItemPlan,
ParsedMultiItemPlan, or any other plan with item-like objects) into
`ParsedItemSegment` objects and routes them through the existing deterministic
`MultiItemQueueCoordinator` path.

Design
------
* Pure function — never raises (exceptions → returns None → caller falls through).
* Does NOT touch GPT, HTTP, DB, or any I/O.
* Does NOT mutate cart directly — all mutations go through the coordinator.
* Builds synthetic SlotValue objects to preserve size/variant/modifier info.
* Each item is resolved independently by name via MenuRepository — no cross-item
  slot leakage.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Sequence

if TYPE_CHECKING:
    from app.menu.repository import MenuRepository
    from app.nlu.multi_item_parser import ParsedItemSegment
    from app.nlu.nlu_result import SlotValue
    from app.state_machine.handler_result import HandlerResult
    from app.state_machine.handlers.item.add_item.multi_item_queue_coordinator import (
        MultiItemQueueCoordinator,
    )
    from app.state_machine.models.conversation_context import ConversationContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_multi_item_plan(
    plan: Any,
    context: "ConversationContext",
    menu_repo: "MenuRepository",
    multi_item_coordinator: "MultiItemQueueCoordinator",
    *,
    get_last_slots: "Callable[[ConversationContext], Sequence[SlotValue]]",
) -> "HandlerResult | None":
    """Convert a multi-item plan into a HandlerResult via the deterministic coordinator.

    Parameters
    ----------
    plan:
        Any plan object with an ``items`` attribute containing item-like objects.
        Compatible with:
          - ``SmartTurnPlan`` (items: tuple[SmartTurnItem, ...])
          - ``GptAddItemPlan`` validated plan (items: tuple[GptValidatedItem, ...])
          - ``ParsedMultiItemPlan`` (items: tuple[ParsedOrderItem, ...])
        Item objects are duck-typed — any combination of the following attributes
        is used when present:
          ``item_name``   — display name for menu resolution
          ``item_id``     — direct menu item ID (bypasses text resolution when set)
          ``quantity``    — order quantity (default 1)
          ``size``        — size label (SmartTurnItem)
          ``size_name``   — size label (ParsedOrderItem / GptValidatedItem)
          ``variant``     — variant label (SmartTurnItem)
          ``variant_name``— variant label (ParsedOrderItem)
          ``modifiers``   — sequence of modifier objects with a ``name`` attribute
          ``sides``       — sequence of side objects with a ``name`` attribute
          ``raw_span``    — raw text segment (ParsedOrderItem)
    context:
        Current conversation context.
    menu_repo:
        Live MenuRepository for item resolution.
    multi_item_coordinator:
        The ``MultiItemQueueCoordinator`` instance from AddItemHandler.
    get_last_slots:
        Callable returning the context's last slots (passed through to coordinator).

    Returns
    -------
    HandlerResult when the plan was successfully routed.
    None when the plan is invalid, empty, has fewer than 2 items, or an
    exception occurred (caller should fall through to the local path).

    Never raises.
    """
    try:
        return _apply(plan, context, menu_repo, multi_item_coordinator, get_last_slots)
    except Exception as exc:
        logger.warning("apply_multi_item_plan_error: %s", exc, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Internal implementation
# ---------------------------------------------------------------------------


def _apply(
    plan: Any,
    context: "ConversationContext",
    menu_repo: "MenuRepository",
    multi_item_coordinator: "MultiItemQueueCoordinator",
    get_last_slots: "Callable[[ConversationContext], Sequence[SlotValue]]",
) -> "HandlerResult | None":
    """Core logic — may raise (caller wraps in try/except)."""
    from app.nlu.multi_item_parser import ParsedItemSegment
    from app.nlu.nlu_result import SlotValue

    items = getattr(plan, "items", None) or ()
    if len(items) < 2:
        logger.debug(
            "apply_multi_item_plan: skipped — fewer than 2 items (%d)", len(items)
        )
        return None

    # item[0] → ParsedItemSegment (activates immediately through coordinator)
    first_seg = _item_to_segment(items[0], SlotValue)
    if first_seg is None:
        logger.warning("apply_multi_item_plan: first item has no usable name — falling through")
        return None

    # items[1..N] → StagedItemPlan (preserved in staged_item_queue, no re-entry)
    staged_items: list = []
    for plan_item in items[1:]:
        staged = _item_to_staged_plan(plan_item)
        if staged is not None:
            staged_items.append(staged)

    # Guard: if no staged items could be produced (all items[1..N] dropped),
    # fall through — this mirrors the old "< 2 valid segments → None" behaviour.
    if not staged_items:
        logger.warning(
            "apply_multi_item_plan: no valid staged items from %d plan items[1..N] — falling through",
            len(items) - 1,
        )
        return None

    logger.info(
        "apply_multi_item_plan.routing",
        extra={
            "segment_count": 1,
            "staged_count": len(staged_items),
            "item_names": [first_seg.item_slot_value or first_seg.raw_text]
                          + [s.item_name for s in staged_items],
            "quantities": [first_seg.quantity]
                          + [s.quantity for s in staged_items],
        },
    )

    return multi_item_coordinator.handle(
        context=context,
        segments=[first_seg],
        get_last_slots=get_last_slots,
        staged_items=staged_items if staged_items else None,
    )


def _item_to_staged_plan(plan_item: Any) -> "Any | None":
    """Convert a duck-typed plan item to a StagedItemPlan for structured queue staging.

    Returns None if the item has no usable name.
    Unlike _item_to_segment, this preserves full structured data (sides with sizes,
    modifiers with operations) rather than flattening to raw slot values.
    """
    from app.state_machine.models.pending_item_models import (
        StagedItemPlan, StagedModifier, StagedSide,
    )

    item_name: str = (
        getattr(plan_item, "item_name", "")
        or getattr(plan_item, "raw_span", "")
        or ""
    ).strip()
    if not item_name:
        return None

    item_id: str = str(getattr(plan_item, "item_id", "") or "").strip()

    qty: int = int(getattr(plan_item, "quantity", 1) or 1)
    qty = max(1, min(qty, 99))

    raw_span: str = (getattr(plan_item, "raw_span", "") or item_name).strip()

    # Item-level size/variant
    variant_id: str | None = str(getattr(plan_item, "variant_id", "") or "").strip() or None
    variant_label: str | None = (
        getattr(plan_item, "size", "")
        or getattr(plan_item, "size_name", "")
        or getattr(plan_item, "variant", "")
        or getattr(plan_item, "variant_name", "")
        or None
    )
    if variant_label:
        variant_label = str(variant_label).strip() or None

    # Requested sides — preserve side-level size/variant
    staged_sides: list[StagedSide] = []
    for sid in (getattr(plan_item, "sides", ()) or ()):
        sid_name = str(getattr(sid, "name", sid) or "").strip()
        if not sid_name:
            continue
        # Side size/variant: SmartTurnSide has .size and .variant after Step 7
        side_variant_label: str | None = (
            getattr(sid, "size", None)
            or getattr(sid, "variant", None)
            or getattr(sid, "variant_label", None)
            or None
        )
        if side_variant_label:
            side_variant_label = str(side_variant_label).strip() or None
        side_qty = max(1, min(int(getattr(sid, "quantity", 1) or 1), 99))
        staged_sides.append(StagedSide(
            name=sid_name,
            variant_label=side_variant_label,
            quantity=side_qty,
        ))

    # Requested modifiers
    staged_mods: list[StagedModifier] = []
    for mod in (getattr(plan_item, "modifiers", ()) or ()):
        mod_name = str(getattr(mod, "name", mod) or "").strip()
        if not mod_name:
            continue
        mod_op = str(getattr(mod, "operation", "add") or "add")
        staged_mods.append(StagedModifier(name=mod_name, operation=mod_op))

    # Determine plan_source
    plan_source: str = str(getattr(plan_item, "plan_source", "") or "").strip()
    if not plan_source:
        # Infer from item type name
        plan_source = type(plan_item).__name__.lower()

    return StagedItemPlan(
        item_id=item_id,
        item_name=item_name,
        quantity=qty,
        variant_id=variant_id,
        variant_label=variant_label,
        requested_sides=tuple(staged_sides),
        requested_modifiers=tuple(staged_mods),
        raw_span=raw_span,
        plan_source=plan_source,
    )


def _item_to_segment(
    plan_item: Any,
    SlotValue: Any,
) -> "ParsedItemSegment | None":
    """Convert a duck-typed plan item to a ParsedItemSegment.

    Returns None if the item has no usable name.
    """
    from app.nlu.multi_item_parser import ParsedItemSegment

    # Resolve display name — prefer item_name, fall back to raw_span
    item_name: str = (
        getattr(plan_item, "item_name", "")
        or getattr(plan_item, "raw_span", "")
        or ""
    ).strip()

    if not item_name:
        return None

    # Quantity — default 1, clamp to [1, 99]
    qty: int = int(getattr(plan_item, "quantity", 1) or 1)
    qty = max(1, min(qty, 99))

    # Raw span text for logging / coordinator ack
    raw_text: str = (
        getattr(plan_item, "raw_span", "")
        or item_name
    ).strip()

    # Build synthetic slot values
    synthetic_slots: list[Any] = []

    # ITEM slot — enables slot-based resolution in MultiItemQueueCoordinator
    synthetic_slots.append(SlotValue(name="ITEM", value=item_name))

    # SIZE / VARIANT slot — from whichever attribute is present
    size_or_variant: str = (
        getattr(plan_item, "size", "")
        or getattr(plan_item, "size_name", "")
        or getattr(plan_item, "variant", "")
        or getattr(plan_item, "variant_name", "")
        or ""
    ).strip()
    if size_or_variant:
        synthetic_slots.append(SlotValue(name="VARIANT", value=size_or_variant))

    # MODIFIER slots
    for mod in (getattr(plan_item, "modifiers", ()) or ()):
        mod_name = str(getattr(mod, "name", mod) or "").strip()
        if mod_name:
            synthetic_slots.append(SlotValue(name="MODIFIER", value=mod_name))

    # SIDE slots
    for sid in (getattr(plan_item, "sides", ()) or ()):
        sid_name = str(getattr(sid, "name", sid) or "").strip()
        if sid_name:
            synthetic_slots.append(SlotValue(name="SIDE", value=sid_name))

    return ParsedItemSegment(
        raw_text=raw_text,
        item_slot_value=item_name,
        quantity=qty if qty > 1 else None,  # None for qty=1 (coordinator treats None as 1)
        slots=tuple(synthetic_slots),
    )
