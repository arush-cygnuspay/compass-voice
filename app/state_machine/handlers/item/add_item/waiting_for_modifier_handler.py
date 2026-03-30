# app/state_machine/handlers/item/add_item/waiting_for_modifier_handler.py
from __future__ import annotations

from app.menu.repository import MenuRepository
from app.nlu.intent_resolution.intent import Intent
from app.nlu.query_normalization.text_preprocessor import normalize_text
from app.session.session import Session
from app.state_machine.conversation_context import (
    ConversationContext,
    InterruptProposal,
    PendingModifierGroup,
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


def _extract_modifier_slot_values_normalized(context: ConversationContext) -> list[str]:
    slots = context.last_slots or ()
    values: list[str] = []
    seen: set[str] = set()

    for slot in slots:
        name = str(slot.name).upper()
        if name not in {"MODIFIER", "ITEM", "MENU_ITEM"}:
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


def _looks_like_pure_modifier_answer(
    normalized_user_text: str,
    normalized_choice_names: tuple[str, ...],
) -> bool:
    """
    Conservative direct-answer detector for modifier selection.

    Accept:
    - cheese
    - extra cheese
    - sausage please
    - i want bacon

    Reject:
    - how much is cheese
    - add fries
    - show menu
    - checkout
    """
    if not normalized_user_text:
        return False

    blocked_phrases = {
        "how much",
        "price",
        "cost",
        "show menu",
        "show me",
        "menu",
        "checkout",
        "check out",
        "cart",
        "total",
        "remove item",
        "change item",
        "modify item",
        "start order",
        "finish order",
        "pay now",
        "payment",
        "add another",
        "add item",
        "remove",
        "change",
        "modify",
        "instead",
    }
    if any(phrase in normalized_user_text for phrase in blocked_phrases):
        return False

    filler_words = {
        "the",
        "a",
        "an",
        "with",
        "add",
        "please",
        "thanks",
        "thank",
        "you",
        "and",
        "extra",
        "only",
        "just",
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


class WaitingForModifierHandler(BaseHandler):
    """
    Resolve modifier selections strictly from the active pending item snapshot.

    Important:
    - no broad menu resolution during waiting state
    - only choices from the current active modifier group can match
    - interruption is considered before broader free-text matching
    - only explicit slot values or short direct answers may satisfy the modifier step
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
        groups = pending.modifier_groups
        idx = context.current_modifier_group_index

        if idx >= len(groups):
            step = determine_next_add_item_step(context)
            return self._step_to_result(context, step)

        group = groups[idx]

        if intent == Intent.DENY:
            if group.is_required:
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_MODIFIER,
                    response_key="required_modifier_cannot_skip",
                    response_payload=self._choice_payload(group),
                )

            context.skipped_modifier_groups.add(group.group_id)
            context.current_modifier_group_index += 1

            step = determine_next_add_item_step(context)
            return self._step_to_result(context, step)

        if intent == Intent.ASK_OPTIONS:
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_MODIFIER,
                response_key="list_modifier_options",
                response_payload=self._choice_payload(group),
            )

        normalized_slot_values = _extract_modifier_slot_values_normalized(context)
        if normalized_slot_values:
            matched_ids = self._match_modifier_choices_from_values(
                group=group,
                normalized_values=normalized_slot_values,
            )
            if matched_ids:
                return self._apply_modifier_selection(
                    context=context,
                    group=group,
                    matched_ids=matched_ids,
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

        if _looks_like_pure_modifier_answer(normalized_user_text, group.normalized_choice_names):
            matched_ids = self._match_modifier_choices_from_values(
                group=group,
                normalized_values=[normalized_user_text],
            )
            if matched_ids:
                return self._apply_modifier_selection(
                    context=context,
                    group=group,
                    matched_ids=matched_ids,
                )

        return HandlerResult(
            next_state=ConversationState.WAITING_FOR_MODIFIER,
            response_key="repeat_modifier_options",
            response_payload={
                **self._choice_payload(group),
                "repeat_reason": "invalid",
            },
        )

    def _apply_modifier_selection(
        self,
        *,
        context: ConversationContext,
        group: PendingModifierGroup,
        matched_ids: list[str],
    ) -> HandlerResult:
        existing_ids = list(context.selected_modifier_groups.get(group.group_id, []))
        proposed_ids = list(existing_ids)

        for modifier_id in matched_ids:
            if modifier_id not in proposed_ids:
                proposed_ids.append(modifier_id)

        max_selector = int(group.max_selector or 1)
        min_selector = int(group.min_selector or 1)

        if max_selector > 0 and len(proposed_ids) > max_selector:
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_MODIFIER,
                response_key="too_many_modifier_choices",
                response_payload=self._choice_payload(group),
            )

        context.selected_modifier_groups[group.group_id] = proposed_ids
        context.skipped_modifier_groups.discard(group.group_id)

        if len(proposed_ids) < min_selector:
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_MODIFIER,
                response_key="repeat_modifier_options",
                response_payload={
                    **self._choice_payload(group),
                    "repeat_reason": "options",
                },
            )

        context.current_modifier_group_index += 1
        step = determine_next_add_item_step(context)
        return self._step_to_result(context, step)

    def _choice_payload(self, group: PendingModifierGroup) -> dict:
        return {
            "group_name": group.name,
            "top_choices": list(group.top_choice_names),
        }

    def _match_modifier_choices_from_values(
        self,
        *,
        group: PendingModifierGroup,
        normalized_values: list[str],
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
                if choice.modifier_id not in seen_ids:
                    matched_ids.append(choice.modifier_id)
                    seen_ids.add(choice.modifier_id)

            if len(candidate) < 3:
                continue

            for choice in group.choices:
                if choice.modifier_id in seen_ids:
                    continue

                choice_name = choice.normalized_name
                if candidate in choice_name or choice_name in candidate:
                    matched_ids.append(choice.modifier_id)
                    seen_ids.add(choice.modifier_id)

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