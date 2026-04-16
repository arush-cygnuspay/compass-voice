# app/state_machine/handlers/item/add_item/waiting_for_modifier_handler.py
from __future__ import annotations

from app.menu.repository import MenuRepository
from app.nlu.intent_resolution.intent import Intent
from app.session.session import Session
from app.state_machine.models.conversation_context import (
    ConversationContext,
    InterruptProposal,
    PendingModifierGroup,
)
from app.state_machine.models.conversation_state import ConversationState
from app.state_machine.models.pending_item_models import ModifierSelection
from app.state_machine.handler_result import HandlerResult
from app.state_machine.handlers.base_handler import BaseHandler
from app.state_machine.handlers.item.add_item.add_item_flow import (
    build_add_item_command,
    determine_next_add_item_step,
)
from app.state_machine.handlers.item.add_item.modifier_group_resolver import (
    ModifierGroupResolver,
    extract_modifier_slot_values_normalized,
)
from app.state_machine.handlers.item.add_item.group_collection_utils import (
    effective_group_selector_bounds,
)

SOFT_SWITCH_INTENTS: set[Intent] = {
    Intent.ADD_ITEM,
    Intent.REMOVE_ITEM,
    Intent.MODIFY_ITEM,
    Intent.SHOW_MENU,
    Intent.ASK_MENU_INFO,
    Intent.ASK_PRICE,
    Intent.SHOW_CART,
    Intent.SHOW_TOTAL,
    Intent.START_ORDER,
    Intent.END_ADDING,
    Intent.CHECKOUT,
    Intent.CONFIRM_ORDER,
    Intent.FINISH_ORDER,
    Intent.REVIEW_ORDER,
    Intent.PAYMENT_REQUEST,
    Intent.CANCEL_ORDER,
}

DONE_WORDS = {
    "done",
    "thats all",
    "that's all",
    "finished",
    "continue",
    "next",
}

SKIP_WORDS = {
    "no",
    "none",
    "nothing",
    "skip",
    "skip it",
    "no thanks",
}


def _looks_like_done_answer(normalized_user_text: str) -> bool:
    return (normalized_user_text or "").strip() in DONE_WORDS


def _looks_like_skip_modifier_answer(normalized_user_text: str, group: PendingModifierGroup) -> bool:
    text = (normalized_user_text or "").strip()
    if not text:
        return False

    # whole-group skip only; specific "no onions" is handled by the resolver
    return text in SKIP_WORDS


class WaitingForModifierHandler(BaseHandler):
    """
    Resolve modifier selections strictly from the active pending item snapshot.

    Important:
    - no broad menu resolution during waiting state
    - only choices from the current active modifier group can match
    - supports multi-modifier capture in one utterance
    - supports structured selections like:
        - bacon
        - extra bacon
        - no onions
        - less mayo
    - keeps group open when min is met but more are still allowed
    """

    def __init__(self, menu_repo: MenuRepository | None = None) -> None:
        self.menu_repo = menu_repo
        self.modifier_resolver = ModifierGroupResolver()

    def handle(
        self,
        intent: Intent,
        context: ConversationContext,
        user_text: str,
        session: Session | None = None,
    ) -> HandlerResult:
        pending = context.pending_add_item
        if pending is None:
            return HandlerResult(
                next_state=ConversationState.ERROR_RECOVERY,
                response_key="item_context_missing",
            )

        normalized_user_text = user_text or ""
        groups = pending.modifier_groups
        idx = context.current_modifier_group_index

        if idx >= len(groups):
            step = determine_next_add_item_step(context)
            return self._step_to_result(context, step)

        group = groups[idx]
        existing_selections = list(context.selected_modifier_groups.get(group.group_id, []))
        existing_ids = [sel.modifier_id for sel in existing_selections]

        min_selector, max_selector = effective_group_selector_bounds(group)

        if intent == Intent.ASK_OPTIONS:
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_MODIFIER,
                response_key="list_modifier_options",
                response_payload=self._choice_payload(group, existing_selections),
            )

        if intent == Intent.DENY or _looks_like_skip_modifier_answer(normalized_user_text, group):
            if len(existing_selections) < min_selector:
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_MODIFIER,
                    response_key="required_modifier_cannot_skip",
                    response_payload=self._choice_payload(group, existing_selections),
                )

            if not existing_selections:
                context.skipped_modifier_groups.add(group.group_id)
                context.selected_modifier_groups.pop(group.group_id, None)

            step = determine_next_add_item_step(context)
            return self._step_to_result(context, step)

        if _looks_like_done_answer(normalized_user_text):
            if len(existing_selections) < min_selector:
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_MODIFIER,
                    response_key="repeat_modifier_options",
                    response_payload={
                        **self._choice_payload(group, existing_selections),
                        "repeat_reason": "need_more",
                    },
                )

            step = determine_next_add_item_step(context)
            return self._step_to_result(context, step)

        resolution = self.modifier_resolver.resolve(
            group=group,
            normalized_user_text=normalized_user_text,
            normalized_slot_values=extract_modifier_slot_values_normalized(context),
            already_selected_ids=existing_ids,
        )

        if resolution.selections:
            return self._apply_modifier_selection(
                context=context,
                group=group,
                matched_selections=resolution.selections,
            )

        if intent in SOFT_SWITCH_INTENTS:
            context.awaiting_flow_confirmation = True
            context.return_state = ConversationState.WAITING_FOR_MODIFIER
            context.interrupt_proposal = InterruptProposal(
                text=normalized_user_text,
                predicted_main_intent=None,
                predicted_sub_intent=intent.value,
            )
            return HandlerResult(
                next_state=ConversationState.CANCELLATION_CONFIRMATION,
                response_key="confirm_cancel_current_item_for_new_request",
                response_payload={"item_name": pending.item_name},
            )

        return HandlerResult(
            next_state=ConversationState.WAITING_FOR_MODIFIER,
            response_key="repeat_modifier_options",
            response_payload={
                **self._choice_payload(group, existing_selections),
                "repeat_reason": "invalid",
            },
        )

    def _apply_modifier_selection(
        self,
        *,
        context: ConversationContext,
        group: PendingModifierGroup,
        matched_selections: list[ModifierSelection],
    ) -> HandlerResult:
        existing = list(context.selected_modifier_groups.get(group.group_id, []))
        proposed = list(existing)

        existing_ids = {sel.modifier_id for sel in existing}
        for selection in matched_selections:
            if selection.modifier_id not in existing_ids:
                proposed.append(selection)
                existing_ids.add(selection.modifier_id)

        min_selector, max_selector = effective_group_selector_bounds(group)

        if max_selector > 0 and len(proposed) > max_selector:
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_MODIFIER,
                response_key="too_many_modifier_choices",
                response_payload=self._choice_payload(group, proposed),
            )

        context.selected_modifier_groups[group.group_id] = proposed
        context.skipped_modifier_groups.discard(group.group_id)

        if len(proposed) < min_selector:
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_MODIFIER,
                response_key="repeat_modifier_options",
                response_payload={
                    **self._choice_payload(group, proposed),
                    "repeat_reason": "need_more",
                },
            )

        if max_selector > 1 and len(proposed) < max_selector:
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_MODIFIER,
                response_key="repeat_modifier_options",
                response_payload={
                    **self._choice_payload(group, proposed),
                    "repeat_reason": "optional_more",
                },
            )

        step = determine_next_add_item_step(context)
        return self._step_to_result(context, step)

    def _choice_payload(
        self,
        group: PendingModifierGroup,
        selections: list[ModifierSelection] | None = None,
    ) -> dict:
        selections = selections or []
        selected_ids = {sel.modifier_id for sel in selections}

        selected_names: list[str] = []
        for sel in selections:
            if sel.action == "remove":
                selected_names.append(f"no {sel.name}")
            elif sel.instruction == "extra":
                selected_names.append(f"extra {sel.name}")
            elif sel.instruction == "less":
                selected_names.append(f"less {sel.name}")
            else:
                selected_names.append(sel.name)

        selected_count = len(selections)
        min_selector, max_selector = effective_group_selector_bounds(group)
        remaining_choice_names = [
            choice.name
            for choice in group.choices
            if choice.modifier_id not in selected_ids
        ]

        return {
            "group_name": group.name,
            "top_choices": remaining_choice_names[:4],
            "all_choices": remaining_choice_names,
            "selected_names": selected_names,
            "selected_count": selected_count,
            "min_selector": min_selector,
            "max_selector": max_selector,
            "remaining_to_min": max(min_selector - selected_count, 0),
            "remaining_to_max": max(max_selector - selected_count, 0),
        }

    def _step_to_result(self, context: ConversationContext, step) -> HandlerResult:
        pending = context.pending_add_item
        if pending is None:
            return HandlerResult(
                next_state=ConversationState.ERROR_RECOVERY,
                response_key="item_context_missing",
            )

        if step.next_state == ConversationState.FINALIZING_ADD_ITEM:
            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="item_added_successfully",
                response_payload={
                    "item_name": pending.item_name,
                    "quantity": context.quantity or 1,
                },
                command=build_add_item_command(context),
                reset_context=True,
            )

        return HandlerResult(
            next_state=step.next_state,
            response_key=step.response_key,
            response_payload=step.response_payload,
        )
