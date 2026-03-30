# app/state_machine/handlers/item/add_item/waiting_for_side_handler.py
from __future__ import annotations

from app.menu.repository import MenuRepository
from app.nlu.intent_resolution.intent import Intent
from app.nlu.query_normalization.text_preprocessor import normalize_text
from app.session.session import Session
from app.state_machine.conversation_context import (
    ConversationContext,
    InterruptProposal,
    PendingSideChoice,
    PendingSideGroup,
)
from app.state_machine.conversation_state import ConversationState
from app.state_machine.handler_result import HandlerResult
from app.state_machine.handlers.base_handler import BaseHandler
from app.state_machine.handlers.item.add_item.add_item_flow import (
    build_add_item_command,
    determine_next_add_item_step,
)
from app.utils.candidate_texts import build_candidate_texts_normalized


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


def _extract_side_slot_values_normalized(context: ConversationContext) -> list[str]:
    slots = context.last_slots or ()
    values: list[str] = []
    seen: set[str] = set()

    for slot in slots:
        name = str(slot.name).upper()
        if name not in {"SIDE", "ITEM", "MENU_ITEM"}:
            continue

        value = slot.value
        if not isinstance(value, str):
            continue

        normalized = normalize_text(value)
        if not normalized or normalized in seen:
            continue

        seen.add(normalized)
        values.append(normalized)

    return values


def _looks_like_pure_side_answer(
    normalized_user_text: str,
    normalized_choice_names: tuple[str, ...],
) -> bool:
    """
    Conservative direct-answer detector for side selection.

    Accept:
    - fries
    - coke
    - with fries
    - a coke
    - the fries
    - coke please

    Reject:
    - how much is coke
    - add a coke
    - show me drinks
    - i want burger
    """
    if not normalized_user_text:
        return False

    filler_words = {
        "the",
        "a",
        "an",
        "with",
        "side",
        "please",
        "thanks",
        "thank",
        "you",
        "and",
        "um",
        "uh",
        "okay",
        "ok",
        "ill",
        "i",
        "want",
        "take",
        "have",
        "get",
        "like",
        "would",
    }

    blocked_phrases = {
        "how much",
        "price",
        "cost",
        "add",
        "show",
        "menu",
        "checkout",
        "cart",
        "total",
        "remove",
        "change",
        "modify",
        "instead",
    }

    if any(phrase in normalized_user_text for phrase in blocked_phrases):
        return False

    tokens = [token for token in normalized_user_text.split() if token not in filler_words]
    compact = " ".join(tokens).strip()
    if not compact:
        return False

    for choice_name in normalized_choice_names:
        if compact == choice_name:
            return True
        if len(compact) >= 3 and (compact in choice_name or choice_name in compact):
            return True

    return False


class WaitingForSideHandler(BaseHandler):
    """
    Resolve side selections strictly from the active pending item snapshot.

    Important:
    - no broad menu resolution during waiting state
    - only choices from the current active side group can match
    - interruption is considered before broad free-text matching
    - only explicit slot values or short direct answers may satisfy the side step
    """

    def __init__(self, menu_repo: MenuRepository | None = None) -> None:
        self.menu_repo = menu_repo

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

        if intent == Intent.ASK_OPTIONS:
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_SIDE,
                response_key="list_side_options",
                response_payload=self._choice_payload(group),
            )

        if intent == Intent.DENY:
            if group.is_required:
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_SIDE,
                    response_key="required_side_cannot_skip",
                    response_payload=self._choice_payload(group),
                )

            context.skipped_side_groups.add(group.group_id)
            context.selected_side_groups.pop(group.group_id, None)

            step = determine_next_add_item_step(context)
            return self._step_to_result(context, step)

        normalized_slot_values = _extract_side_slot_values_normalized(context)
        if normalized_slot_values:
            matched_ids = self._match_side_choices_from_values(
                normalized_values=normalized_slot_values,
                group=group,
            )
            if matched_ids:
                return self._apply_side_selection(
                    context=context,
                    pending_item_name=pending.item_name,
                    group=group,
                    matched_ids=matched_ids,
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

        if _looks_like_pure_side_answer(normalized_user_text, group.normalized_choice_names):
            matched_ids = self._match_side_choices_from_values(
                normalized_values=[normalized_user_text],
                group=group,
            )
            if matched_ids:
                return self._apply_side_selection(
                    context=context,
                    pending_item_name=pending.item_name,
                    group=group,
                    matched_ids=matched_ids,
                )

        return HandlerResult(
            next_state=ConversationState.WAITING_FOR_SIDE,
            response_key="repeat_side_options",
            response_payload={
                **self._choice_payload(group),
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
        proposed_ids = list(existing_ids)

        for item_id in matched_ids:
            if item_id not in proposed_ids:
                proposed_ids.append(item_id)

        max_selector = int(group.max_selector or 1)
        min_selector = max(int(group.min_selector or 1), 1)

        if max_selector > 0 and len(proposed_ids) > max_selector:
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_SIDE,
                response_key="too_many_side_choices",
                response_payload=self._choice_payload(group),
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
                    **self._choice_payload(group),
                    "repeat_reason": "options",
                },
            )

        step = determine_next_add_item_step(context)
        return self._step_to_result(context, step)

    def _choice_payload(self, group: PendingSideGroup) -> dict:
        return {
            "group_name": group.name,
            "top_choices": list(group.top_choice_names),
        }

    def _match_side_choices_from_values(
        self,
        *,
        normalized_values: list[str],
        group: PendingSideGroup,
    ) -> list[str]:
        matched_ids: list[str] = []
        seen_ids: set[str] = set()

        candidate_texts = build_candidate_texts_normalized(
            normalized_user_text="",
            normalized_slot_values=normalized_values,
            allow_split=True,
        )

        for candidate in candidate_texts:
            exact_choices = group.choices_by_normalized_name.get(candidate, ())
            for choice in exact_choices:
                if choice.item_id not in seen_ids:
                    matched_ids.append(choice.item_id)
                    seen_ids.add(choice.item_id)

            if len(candidate) < 3:
                continue

            for choice in group.choices:
                if choice.item_id in seen_ids:
                    continue

                choice_name = choice.normalized_name
                if candidate in choice_name or choice_name in candidate:
                    matched_ids.append(choice.item_id)
                    seen_ids.add(choice.item_id)

        return matched_ids

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