# app/state_machine/handlers/item/add_item/waiting_for_side_handler.py
from __future__ import annotations

from dataclasses import dataclass

from app.menu.repository import MenuRepository
from app.nlu.intent_resolution.intent import Intent
from app.session.session import Session
from app.state_machine.models.conversation_context import (
    ConversationContext,
    InterruptProposal,
    PendingSideGroup,
)
from app.state_machine.models.conversation_state import ConversationState
from app.state_machine.handler_result import HandlerResult
from app.state_machine.handlers.base_handler import BaseHandler
from app.state_machine.handlers.item.add_item.add_item_flow import (
    build_add_item_command,
    determine_next_add_item_step,
)
from app.state_machine.handlers.item.add_item.side_group_resolver import (
    SideGroupResolver,
    extract_side_slot_values_normalized,
    dedupe_keep_order,
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


@dataclass(frozen=True, slots=True)
class _ScoredSideChoice:
    item_id: str
    choice_name: str
    confidence: float


def _looks_like_done_answer(normalized_user_text: str) -> bool:
    return (normalized_user_text or "").strip() in DONE_WORDS


def _looks_like_skip_side_answer(normalized_user_text: str, group: PendingSideGroup) -> bool:
    text = (normalized_user_text or "").strip()
    if not text:
        return False

    return text in SKIP_WORDS


class WaitingForSideHandler(BaseHandler):
    """
    Resolve side selections strictly from the active pending item snapshot.

    Important:
    - no broad menu resolution during waiting state
    - only choices from the current active side group can match
    - supports multi-side capture in one utterance
    - keeps group open when min is met but more are still allowed
    - preserves existing side-size handoff for newly selected variant-priced sides
    """

    def __init__(self, menu_repo: MenuRepository | None = None) -> None:
        self.menu_repo = menu_repo
        self.side_resolver = SideGroupResolver()

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
        groups = pending.side_groups
        idx = context.current_side_group_index

        if idx >= len(groups):
            step = determine_next_add_item_step(context)
            return self._step_to_result(context, step)

        group = groups[idx]
        existing_ids = list(context.selected_side_groups.get(group.group_id, []))
        min_selector, max_selector = effective_group_selector_bounds(group)

        if intent == Intent.ASK_OPTIONS:
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_SIDE,
                response_key="list_side_options",
                response_payload=self._choice_payload(context, group),
            )

        if intent == Intent.DENY or _looks_like_skip_side_answer(normalized_user_text, group):
            if len(existing_ids) < min_selector:
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_SIDE,
                    response_key="required_side_cannot_skip",
                    response_payload=self._choice_payload(context, group),
                )

            if not existing_ids:
                context.skipped_side_groups.add(group.group_id)
                context.selected_side_groups.pop(group.group_id, None)

            step = determine_next_add_item_step(context)
            return self._step_to_result(context, step)

        if _looks_like_done_answer(normalized_user_text):
            if len(existing_ids) < min_selector:
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_SIDE,
                    response_key="repeat_side_options",
                    response_payload={
                        **self._choice_payload(context, group),
                        "repeat_reason": "need_more",
                    },
                )

            step = determine_next_add_item_step(context)
            return self._step_to_result(context, step)

        resolution = self.side_resolver.resolve(
            group=group,
            normalized_user_text=normalized_user_text,
            normalized_slot_values=extract_side_slot_values_normalized(context),
            already_selected_ids=existing_ids,
        )

        if resolution.matched_item_ids:
            return self._apply_side_selection(
                context=context,
                pending_item_name=pending.item_name,
                group=group,
                matched_ids=resolution.matched_item_ids,
            )

        if intent in SOFT_SWITCH_INTENTS:
            context.awaiting_flow_confirmation = True
            context.return_state = ConversationState.WAITING_FOR_SIDE
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
            next_state=ConversationState.WAITING_FOR_SIDE,
            response_key="repeat_side_options",
            response_payload={
                **self._choice_payload(context, group),
                "repeat_reason": "invalid",
            },
        )

    def _apply_side_selection(
        self,
        *,
        context: ConversationContext,
        pending_item_name: str,
        group: PendingSideGroup,
        matched_ids: list[str],
    ) -> HandlerResult:
        existing_ids = list(context.selected_side_groups.get(group.group_id, []))
        proposed_ids = dedupe_keep_order(existing_ids + matched_ids)

        min_selector, max_selector = effective_group_selector_bounds(group)

        if max_selector > 0 and len(proposed_ids) > max_selector:
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_SIDE,
                response_key="too_many_side_choices",
                response_payload=self._choice_payload(context, group),
            )

        context.selected_side_groups[group.group_id] = proposed_ids
        context.skipped_side_groups.discard(group.group_id)

        newly_added_ids = [item_id for item_id in proposed_ids if item_id not in existing_ids]
        for selected_item_id in newly_added_ids:
            choice = group.choices_by_item_id.get(selected_item_id)
            if choice and choice.pricing_mode == "variant":
                context.pending_side_item_id = choice.item_id
                context.pending_side_item_name = choice.name
                context.pending_side_group_id = group.group_id
                context.current_prompt_field = "side_size"
                context.available_choices_kind = "side_size"
                context.available_choices_values = choice.variant_names

                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_SIDE_SIZE,
                    response_key="ask_for_side_size",
                    response_payload={
                        "item_name": pending_item_name,
                        "side_item_name": choice.name,
                        "group_name": group.name,
                        "available_sizes": list(choice.variant_names),
                    },
                )

        if len(proposed_ids) < min_selector:
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_SIDE,
                response_key="repeat_side_options",
                response_payload={
                    **self._choice_payload(context, group),
                    "repeat_reason": "need_more",
                },
            )

        if max_selector > 1 and len(proposed_ids) < max_selector:
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_SIDE,
                response_key="repeat_side_options",
                response_payload={
                    **self._choice_payload(context, group),
                    "repeat_reason": "optional_more",
                },
            )

        step = determine_next_add_item_step(context)
        return self._step_to_result(context, step)

    def _choice_payload(self, context: ConversationContext, group: PendingSideGroup) -> dict:
        selected_ids = list(context.selected_side_groups.get(group.group_id, []))
        selected_names = [
            group.choices_by_item_id[item_id].name
            for item_id in selected_ids
            if item_id in group.choices_by_item_id
        ]

        selected_count = len(selected_ids)
        min_selector, max_selector = effective_group_selector_bounds(group)
        selected_id_set = set(selected_ids)
        remaining_choice_names = [
            choice.name
            for choice in group.choices
            if choice.item_id not in selected_id_set
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
