# app/state_machine/handlers/item/add_item/add_item_flow.py
from __future__ import annotations

from dataclasses import dataclass

from app.state_machine.conversation_context import ConversationContext, PendingAddItem
from app.state_machine.conversation_state import ConversationState


@dataclass(frozen=True, slots=True)
class AddItemNextStep:
    next_state: ConversationState
    response_key: str
    response_payload: dict | None = None


def determine_next_add_item_step(context: ConversationContext) -> AddItemNextStep:
    pending = context.pending_add_item
    if pending is None:
        return AddItemNextStep(
            next_state=ConversationState.ERROR_RECOVERY,
            response_key="item_context_missing",
        )

    side_variant_step = _find_pending_side_variant_step(context, pending)
    if side_variant_step is not None:
        return side_variant_step

    next_side_index = _find_next_unresolved_side_group_index(context, pending)
    if next_side_index is not None:
        context.current_side_group_index = next_side_index
        group = pending.side_groups[next_side_index]

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
            },
        )

    next_modifier_index = _find_next_unresolved_modifier_group_index(context, pending)
    if next_modifier_index is not None:
        context.current_modifier_group_index = next_modifier_index
        group = pending.modifier_groups[next_modifier_index]

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
            },
        )

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

    if not _has_valid_quantity(context):
        context.current_prompt_field = "quantity"
        context.available_choices_kind = None
        context.available_choices_values = ()

        return AddItemNextStep(
            next_state=ConversationState.WAITING_FOR_QUANTITY,
            response_key="ask_for_quantity",
            response_payload={"item_name": pending.item_name},
        )

    return AddItemNextStep(
        next_state=ConversationState.FINALIZING_ADD_ITEM,
        response_key="finalize_add_item",
        response_payload={"item_name": pending.item_name},
    )


def build_add_item_command(context: ConversationContext) -> dict:
    pending = context.pending_add_item
    if pending is None:
        raise ValueError("pending_add_item is missing")

    return {
        "type": "ADD_ITEM_TO_CART",
        "payload": {
            "item_id": pending.item_id,
            "quantity": context.quantity or 1,
            "variant_id": context.selected_variant_id,
            "sides": dict(context.selected_side_groups),
            "side_variants": dict(context.selected_side_variants),
            "modifiers": dict(context.selected_modifier_groups),
        },
    }


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
    if bool(group.is_required):
        return selected_count >= int(group.min_selector or 1)
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


def _modifier_group_satisfied(group, selected_modifier_ids: list[str] | tuple[str, ...], skipped: bool) -> bool:
    selected_count = len(selected_modifier_ids)
    if bool(group.is_required):
        return selected_count >= int(group.min_selector or 1)
    return selected_count > 0 or skipped


def _has_valid_quantity(context: ConversationContext) -> bool:
    return isinstance(context.quantity, int) and context.quantity > 0