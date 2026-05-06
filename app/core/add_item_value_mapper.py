# app/core/add_item_value_mapper.py
"""Maps raw command/context data to human-readable display values for diagnostics.

Extracted from TurnEngine so the orchestrator doesn't own ID→display-name
translation logic.  All output from this module is consumed only by the
diagnostics layer (TurnEvent logging) — never by business logic.
"""
from __future__ import annotations

from typing import Any

from app.menu.repository import MenuRepository
from app.session.session import Session
from app.state_machine.handler_result import HandlerResult
from app.state_machine.handlers.item.add_item.group_collection_utils import (
    effective_group_selector_bounds,
)


def format_modifier_selection_name(selection: Any) -> str:
    """Convert a modifier selection object/dict to a spoken name fragment."""
    name = str(getattr(selection, "name", "") or "").strip()
    action = str(getattr(selection, "action", "add") or "add").strip()
    instruction = getattr(selection, "instruction", None)
    if not name:
        return ""
    if action == "remove":
        return f"no {name}"
    if instruction == "extra":
        return f"extra {name}"
    if instruction == "less":
        return f"less {name}"
    if instruction == "on_side":
        return f"{name} on the side"
    return name


def build_normalized_values(
    session: Session,
    menu_repo: MenuRepository,
    result: HandlerResult | None,
) -> dict[str, Any]:
    """Return display-friendly values (item name, variant, sides, modifiers) for logging.

    Tries the ADD_ITEM_TO_CART command payload first; falls back to the
    ConversationContext when no command is present.
    """
    command = (result.command or {}) if result is not None else {}
    payload = command.get("payload") or {}

    if command.get("type") == "ADD_ITEM_TO_CART" and payload:
        return _map_from_command_payload(payload, menu_repo)

    return _map_from_context(session, menu_repo)


def build_missing_required_fields(session: Session) -> list[str]:
    """Return names of required item fields that are not yet filled.

    Used exclusively to populate the ``missing_required_fields`` diagnostic
    column on TurnEvent.
    """
    ctx = session.conversation_context
    pending = ctx.pending_add_item
    if pending is None:
        return []

    missing: list[str] = []

    if pending.item_variants and not ctx.selected_variant_id:
        missing.append("size")

    for group in pending.side_groups:
        selected_ids = ctx.selected_side_groups.get(group.group_id, ())
        min_selector, _ = effective_group_selector_bounds(group)
        if bool(getattr(group, "is_required", False)) and len(selected_ids) < min_selector:
            missing.append(group.name)

    for group in pending.modifier_groups:
        selections = ctx.selected_modifier_groups.get(group.group_id, ())
        min_selector, _ = effective_group_selector_bounds(group)
        if bool(getattr(group, "is_required", False)) and len(selections) < min_selector:
            missing.append(group.name)

    if not (isinstance(ctx.quantity, int) and ctx.quantity > 0):
        missing.append("quantity")

    return missing


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _map_from_command_payload(
    payload: dict[str, Any],
    menu_repo: MenuRepository,
) -> dict[str, Any]:
    item_id = payload.get("item_id")
    item = menu_repo.store.get_item(item_id) if item_id else None

    side_choice_by_id: dict[str, Any] = {}
    side_group_by_id: dict[str, Any] = {}
    modifier_group_by_id: dict[str, Any] = {}

    if item is not None:
        for group in getattr(item, "side_groups", ()) or ():
            side_group_by_id[group.group_id] = group
            for choice in getattr(group, "choices", ()) or ():
                side_choice_by_id[choice.item_id] = choice
        for group in getattr(item, "modifier_groups", ()) or ():
            modifier_group_by_id[group.group_id] = group

    mapped_sides: dict[str, list[str]] = {}
    for group_id, side_ids in (payload.get("sides") or {}).items():
        group = side_group_by_id.get(group_id)
        group_name = getattr(group, "name", None) or str(group_id)
        mapped_sides[group_name] = [
            getattr(side_choice_by_id.get(side_id), "name", side_id)
            for side_id in side_ids
        ]

    mapped_side_variants: dict[str, str] = {}
    for side_id, variant_id in (payload.get("side_variants") or {}).items():
        choice = side_choice_by_id.get(side_id)
        side_name = getattr(choice, "name", None) or str(side_id)
        variant_label = str(variant_id)
        if choice is not None:
            for variant in getattr(getattr(choice, "pricing", None), "variants", ()) or ():
                if getattr(variant, "variant_id", None) == variant_id:
                    variant_label = getattr(variant, "label", variant_label)
                    break
        mapped_side_variants[side_name] = variant_label

    mapped_modifiers: dict[str, list[str]] = {}
    for group_id, selections in (payload.get("modifiers") or {}).items():
        group = modifier_group_by_id.get(group_id)
        group_name = getattr(group, "name", None) or str(group_id)
        names: list[str] = []
        for selection in selections or ():
            if isinstance(selection, dict):
                names.append(
                    format_modifier_selection_name(
                        type("ModifierPayload", (), selection)()
                    )
                )
            else:
                names.append(format_modifier_selection_name(selection))
        mapped_modifiers[group_name] = [n for n in names if n]

    mapped_variant = None
    variant_id = payload.get("variant_id")
    if item is not None and variant_id:
        for variant in getattr(getattr(item, "pricing", None), "variants", ()) or ():
            if getattr(variant, "variant_id", None) == variant_id:
                mapped_variant = getattr(variant, "label", None) or variant_id
                break
    elif variant_id:
        mapped_variant = variant_id

    return {
        "item_name": getattr(item, "name", None) or str(item_id or ""),
        "quantity": payload.get("quantity"),
        "variant": mapped_variant,
        "sides": mapped_sides,
        "side_variants": mapped_side_variants,
        "modifiers": mapped_modifiers,
    }


def _map_from_context(
    session: Session,
    menu_repo: MenuRepository,
) -> dict[str, Any]:
    ctx = session.conversation_context
    pending = ctx.pending_add_item
    if pending is None:
        return {
            "item_name": ctx.current_item_name,
            "quantity": ctx.quantity,
        }

    mapped_sides: dict[str, list[str]] = {}
    for group in pending.side_groups:
        selected_ids = ctx.selected_side_groups.get(group.group_id, ())
        if not selected_ids:
            continue
        mapped_sides[group.name] = [
            group.choices_by_item_id[item_id].name
            for item_id in selected_ids
            if item_id in group.choices_by_item_id
        ]

    mapped_modifiers: dict[str, list[str]] = {}
    for group in pending.modifier_groups:
        selections = ctx.selected_modifier_groups.get(group.group_id, ())
        if not selections:
            continue
        mapped_modifiers[group.name] = [
            format_modifier_selection_name(selection)
            for selection in selections
            if format_modifier_selection_name(selection)
        ]

    mapped_variant = None
    if ctx.selected_variant_id:
        variant = pending.item_variants_by_id.get(ctx.selected_variant_id)
        mapped_variant = getattr(variant, "name", None) or ctx.selected_variant_id

    mapped_side_variants: dict[str, str] = {}
    for side_item_id, variant_id in ctx.selected_side_variants.items():
        choice = pending.side_choice_by_item_id.get(side_item_id)
        side_name = getattr(choice, "name", None) or str(side_item_id)
        variant_name = variant_id
        if choice is not None:
            variant = choice.variants_by_id.get(variant_id)
            if variant is not None:
                variant_name = variant.name
        mapped_side_variants[side_name] = variant_name

    return {
        "item_name": pending.item_name,
        "quantity": ctx.quantity,
        "variant": mapped_variant,
        "sides": mapped_sides,
        "side_variants": mapped_side_variants,
        "modifiers": mapped_modifiers,
    }
