# app/state_machine/handlers/item/add_item/confirmation_decision_helper.py
from __future__ import annotations

from app.menu.models import MenuItem
from app.nlu.modifier_instructions import speak as _speak_modifier
from app.state_machine.handler_result import HandlerResult
from app.state_machine.handlers.item.add_item.add_item_flow import (
    ReadyToFinalize,
    determine_next_add_item_step,
)
from app.state_machine.models.conversation_context import ConversationContext
from app.state_machine.models.conversation_state import ConversationState


def _apply_group_defaults(context: ConversationContext) -> None:
    """Silently apply default_item_ids for any side group the customer skipped.

    Called just before ReadyToFinalize so finalized cart data includes the
    menu-defined defaults (e.g. a sauce that comes with the item unless
    the customer explicitly refused it).
    """
    pending = context.pending_add_item
    if pending is None:
        return
    for group in pending.side_groups:
        if not group.default_item_ids:
            continue
        group_id = group.group_id
        if group_id in context.skipped_side_groups:
            continue
        existing = list(context.selected_side_groups.get(group_id) or [])
        if existing:
            continue
        context.selected_side_groups[group_id] = list(group.default_item_ids)


def _spoken_modifiers_for(context: ConversationContext) -> list[str]:
    """Render the customer's selected modifiers as spoken phrases.

    Used by the success-message path so "Chicken Burger added" becomes
    "Chicken Burger with no onions and extra cheese added." The action
    (add/remove) and instruction (extra/less/on_side) come straight
    from each ModifierSelection — no inline branching here, all rules
    live in app.nlu.modifier_instructions.
    """
    pending = context.pending_add_item
    if pending is None:
        return []

    spoken: list[str] = []
    for group in pending.modifier_groups:
        for selection in context.selected_modifier_groups.get(group.group_id, []):
            phrase = _speak_modifier(
                selection.name,
                action=selection.action,
                instruction=selection.instruction,
            )
            if phrase:
                spoken.append(phrase)
    return spoken


class ConfirmationDecisionHelper:
    """Interprets the next-step determination and builds the final HandlerResult.

    Receives the post-prefill context and the prefill summaries, calls
    determine_next_add_item_step(), and returns the appropriate HandlerResult:
    either an immediate add-to-cart (ReadyToFinalize) or a waiting-state
    prompt for the next unresolved group.
    """

    def build_handler_result(
        self,
        *,
        context: ConversationContext,
        item: MenuItem,
        prefilled_summary: str,
        prefill_feedback: str,
        prefill_debug: dict,
    ) -> HandlerResult:
        _apply_group_defaults(context)
        step = determine_next_add_item_step(context)

        if isinstance(step, ReadyToFinalize):
            spoken_modifiers = _spoken_modifiers_for(context)
            payload: dict = {
                "item_name": item.name,
                "quantity": context.quantity or 1,
                "prefill_debug": prefill_debug,
                # NEW: render modifiers with their action/instruction so the
                # success line voices "with no onions and extra cheese" back.
                "spoken_modifiers": spoken_modifiers,
            }
            # Suppress the legacy prefilled_summary prefix when we already
            # have a spoken_modifiers clause — the success line owns the
            # listing, no need to double-confirm it.
            if not spoken_modifiers and prefilled_summary:
                payload["prefilled_summary"] = prefilled_summary
            if prefill_feedback:
                payload["prefill_feedback"] = prefill_feedback
            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="item_added_successfully",
                response_payload=payload,
                command=step.command.to_dict(),
                reset_context=True,
            )

        payload = dict(step.response_payload or {})
        if prefilled_summary:
            payload["prefilled_summary"] = prefilled_summary
            payload["prefilled_item_name"] = item.name
        if prefill_feedback:
            payload["prefill_feedback"] = prefill_feedback
        payload["prefill_debug"] = prefill_debug

        return HandlerResult(
            next_state=step.next_state,
            response_key=step.response_key,
            response_payload=payload,
        )
