# app/state_machine/handlers/item/add_item/add_item_flow.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.state_machine.models.conversation_context import ConversationContext
from app.state_machine.models.pending_item_models import PendingAddItem
from app.state_machine.models.conversation_state import ConversationState
from app.state_machine.handlers.item.add_item.group_collection_utils import (
    effective_group_selector_bounds,
)


@dataclass(frozen=True, slots=True)
class AddItemNextStep:
    next_state: ConversationState
    response_key: str
    response_payload: dict | None = None


@dataclass(frozen=True, slots=True)
class AddItemCommand:
    """Immutable snapshot of the finalized item data ready to be sent to the cart."""

    item_id: str
    quantity: int
    variant_id: Optional[str]
    sides: dict
    side_variants: dict
    modifiers: dict

    def to_dict(self) -> dict:
        return {
            "type": "ADD_ITEM_TO_CART",
            "payload": {
                "item_id": self.item_id,
                "quantity": self.quantity,
                "variant_id": self.variant_id,
                "sides": self.sides,
                "side_variants": self.side_variants,
                "modifiers": self.modifiers,
            },
        }


@dataclass(frozen=True, slots=True)
class ReadyToFinalize:
    """
    Explicit flow outcome returned by determine_next_add_item_step when all
    required item attributes are resolved and the item is ready to be added to
    the cart.  Callers convert this into a terminal HandlerResult directly —
    no router lookup or pseudo-state transition occurs.
    """

    command: AddItemCommand


def determine_next_add_item_step(context: ConversationContext) -> AddItemNextStep | ReadyToFinalize:
    pending = context.pending_add_item
    if pending is None:
        return AddItemNextStep(
            next_state=ConversationState.ERROR_RECOVERY,
            response_key="item_context_missing",
        )

    side_variant_step = _find_pending_side_variant_step(context, pending)
    if side_variant_step is not None:
        return side_variant_step

    if pending.item_variants and not _has_valid_variant_selected(context, pending):
        context.size_target = {"type": "item"}
        context.current_prompt_field = "size"
        context.available_choices_kind = "size"
        context.available_choices_values = pending.item_variant_names

        return AddItemNextStep(
            next_state=ConversationState.WAITING_FOR_SIZE,
            response_key="ask_for_size",
            response_payload={
                "item_name": pending.item_name,
                "available_sizes": list(pending.item_variant_names),
            },
        )

    next_side_index = _find_next_unresolved_side_group_index(context, pending)
    if next_side_index is not None:
        context.current_side_group_index = next_side_index
        group = pending.side_groups[next_side_index]
        selected_ids = list(context.selected_side_groups.get(group.group_id, []))

        context.current_prompt_field = "side"
        context.available_choices_kind = "side"
        context.available_choices_values = group.choice_names

        return AddItemNextStep(
            next_state=ConversationState.WAITING_FOR_SIDE,
            response_key="ask_for_side",
            response_payload={
                "item_name": pending.item_name,
                "group_name": group.name,
                "top_choices": list(group.top_choice_names),
                "min_selector": effective_group_selector_bounds(group)[0],
                "max_selector": effective_group_selector_bounds(group)[1],
                "selected_count": len(selected_ids),
            },
        )

    next_modifier_index = _find_next_unresolved_modifier_group_index(context, pending)
    if next_modifier_index is not None:
        context.current_modifier_group_index = next_modifier_index
        group = pending.modifier_groups[next_modifier_index]

        selected_ids = list(context.selected_modifier_groups.get(group.group_id, []))

        context.current_prompt_field = "modifier"
        context.available_choices_kind = "modifier"
        context.available_choices_values = group.choice_names

        return AddItemNextStep(
            next_state=ConversationState.WAITING_FOR_MODIFIER,
            response_key="ask_for_modifier",
            response_payload={
                "item_name": pending.item_name,
                "group_name": group.name,
                "top_choices": list(group.top_choice_names),
                "min_selector": effective_group_selector_bounds(group)[0],
                "max_selector": effective_group_selector_bounds(group)[1],
                "selected_count": len(selected_ids),
            },
        )

    if not _has_valid_quantity(context):
        context.current_prompt_field = "quantity"
        context.available_choices_kind = None
        context.available_choices_values = ()

        return AddItemNextStep(
            next_state=ConversationState.WAITING_FOR_QUANTITY,
            response_key="ask_for_quantity",
            response_payload={"item_name": pending.item_name},
        )

    return ReadyToFinalize(
        command=AddItemCommand(
            item_id=pending.item_id,
            quantity=context.quantity or 1,
            variant_id=context.selected_variant_id,
            sides=dict(context.selected_side_groups),
            side_variants=dict(context.selected_side_variants),
            modifiers={
                group_id: [
                    {
                        "modifier_id": sel.modifier_id,
                        "name": sel.name,
                        "action": sel.action,
                        "instruction": sel.instruction,
                    }
                    for sel in selections
                ]
                for group_id, selections in context.selected_modifier_groups.items()
            },
        )
    )


def _has_valid_variant_selected(context: ConversationContext, pending: PendingAddItem) -> bool:
    variant_id = context.selected_variant_id
    if not variant_id:
        return False
    return variant_id in pending.item_variants_by_id


def _find_pending_side_variant_step(
    context: ConversationContext,
    pending: PendingAddItem,
) -> AddItemNextStep | None:
    selected_side_groups = context.selected_side_groups
    selected_side_variants = context.selected_side_variants

    for group in pending.side_groups:
        selected_ids = selected_side_groups.get(group.group_id, ())
        if not selected_ids:
            continue

        for selected_item_id in selected_ids:
            if selected_item_id in selected_side_variants:
                continue

            choice = group.choices_by_item_id.get(selected_item_id)
            if choice is None:
                continue

            if choice.pricing_mode != "variant":
                continue

            context.pending_side_item_id = choice.item_id
            context.pending_side_item_name = choice.name
            context.pending_side_group_id = group.group_id
            context.current_prompt_field = "side_size"
            context.available_choices_kind = "side_size"
            context.available_choices_values = choice.variant_names

            return AddItemNextStep(
                next_state=ConversationState.WAITING_FOR_SIDE_SIZE,
                response_key="ask_for_side_size",
                response_payload={
                    "item_name": pending.item_name,
                    "side_item_name": choice.name,
                    "group_name": group.name,
                    "available_sizes": list(choice.variant_names),
                },
            )

    return None


def _find_next_unresolved_side_group_index(
    context: ConversationContext,
    pending: PendingAddItem,
) -> int | None:
    selected_side_groups = context.selected_side_groups
    skipped_side_groups = context.skipped_side_groups

    for idx, group in enumerate(pending.side_groups):
        group_id = group.group_id
        selected = selected_side_groups.get(group_id, ())
        skipped = group_id in skipped_side_groups

        if _side_group_satisfied(group, selected, skipped):
            continue

        return idx

    return None


def _side_group_satisfied(group, selected_item_ids: list[str] | tuple[str, ...], skipped: bool) -> bool:
    selected_count = len(selected_item_ids)
    min_selector, _ = effective_group_selector_bounds(group)
    if bool(getattr(group, "is_required", False)):
        return selected_count >= min_selector
    return selected_count > 0 or skipped


def _find_next_unresolved_modifier_group_index(
    context: ConversationContext,
    pending: PendingAddItem,
) -> int | None:
    selected_modifier_groups = context.selected_modifier_groups
    skipped_modifier_groups = context.skipped_modifier_groups

    for idx, group in enumerate(pending.modifier_groups):
        group_id = group.group_id
        selected = selected_modifier_groups.get(group_id, ())
        skipped = group_id in skipped_modifier_groups

        if _modifier_group_satisfied(group, selected, skipped):
            continue

        return idx

    return None


def _modifier_group_satisfied(group, selected_modifier_selections, skipped: bool) -> bool:
    """
    A modifier group is satisfied when:
    - required group: selected count >= min_selector
    - optional group: at least one selection exists OR the whole group was skipped
    """
    selected_count = len(selected_modifier_selections or ())
    min_selector, _ = effective_group_selector_bounds(group)
    if bool(getattr(group, "is_required", False)):
        return selected_count >= min_selector

    return selected_count > 0 or skipped


def _has_valid_quantity(context: ConversationContext) -> bool:
    return isinstance(context.quantity, int) and context.quantity > 0
