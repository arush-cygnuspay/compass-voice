# app/state_machine/handlers/item/add_item/waiting_for_side_size_handler.py
from __future__ import annotations

from dataclasses import dataclass

from app.menu.repository import MenuRepository
from app.nlu.intent_resolution.intent import Intent
from app.nlu.nlu_result import SlotValue
from app.nlu.query_normalization.text_preprocessor import normalize_text
from app.session.session import Session
from app.state_machine.conversation_context import ConversationContext, InterruptProposal
from app.state_machine.conversation_state import ConversationState
from app.state_machine.handler_result import HandlerResult
from app.state_machine.handlers.base_handler import BaseHandler
from app.state_machine.handlers.item.add_item.add_item_flow import (
    build_add_item_command,
    determine_next_add_item_step,
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


@dataclass(frozen=True, slots=True)
class _VariantChoice:
    variant_id: str
    name: str
    normalized_name: str


def _first_size_slot_normalized(slots: list[SlotValue] | tuple[SlotValue, ...]) -> str | None:
    for slot in slots:
        if str(slot.name).lower() != "size":
            continue

        value = slot.value
        if not isinstance(value, str):
            continue

        normalized = normalize_text(value)
        if normalized:
            return normalized

    return None


def _match_variant_value_normalized(
    normalized_value: str,
    choices: list[_VariantChoice],
) -> _VariantChoice | None:
    if not normalized_value:
        return None

    for choice in choices:
        if choice.normalized_name == normalized_value:
            return choice

    return None


def _looks_like_pure_size_answer(
    normalized_user_text: str,
    choices: list[_VariantChoice],
) -> bool:
    if not normalized_user_text:
        return False

    filler_words = {
        "please",
        "the",
        "a",
        "an",
        "one",
        "size",
        "with",
        "thanks",
        "thank",
        "you",
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
    }

    if any(phrase in normalized_user_text for phrase in blocked_phrases):
        return False

    tokens = [token for token in normalized_user_text.split() if token not in filler_words]
    compact = " ".join(tokens).strip()
    if not compact:
        return False

    for choice in choices:
        if choice.normalized_name == compact:
            return True

    return False


class WaitingForSideSizeHandler(BaseHandler):
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
        if pending is None or not context.pending_side_item_id:
            return HandlerResult(
                next_state=ConversationState.ERROR_RECOVERY,
                response_key="item_context_missing",
            )

        normalized_user_text = user_text or ""

        side_choice = self._find_pending_side_choice(context)
        if side_choice is None:
            return HandlerResult(
                next_state=ConversationState.ERROR_RECOVERY,
                response_key="item_context_missing",
            )

        if side_choice.pricing_mode != "variant":
            self._clear_pending_side_size(context)
            step = determine_next_add_item_step(context)
            return self._step_to_result(context, step)

        choices = [
            _VariantChoice(
                variant_id=variant.variant_id,
                name=variant.name,
                normalized_name=variant.normalized_name,
            )
            for variant in (side_choice.variants or [])
            if variant.name
        ]

        if not choices:
            self._clear_pending_side_size(context)
            step = determine_next_add_item_step(context)
            return self._step_to_result(context, step)

        if intent == Intent.DENY:
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_SIDE_SIZE,
                response_key="required_side_size_cannot_skip",
                response_payload={
                    "item_name": pending.item_name,
                    "side_item_name": side_choice.name,
                    "available_sizes": [choice.name for choice in choices],
                },
            )

        if intent == Intent.ASK_OPTIONS:
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_SIDE_SIZE,
                response_key="repeat_side_size_options",
                response_payload={
                    "item_name": pending.item_name,
                    "side_item_name": side_choice.name,
                    "available_sizes": [choice.name for choice in choices],
                },
            )

        slot_value = _first_size_slot_normalized(context.last_slots or ())
        if slot_value:
            matched = _match_variant_value_normalized(slot_value, choices)
            if matched is not None:
                context.selected_side_variants[side_choice.item_id] = matched.variant_id
                self._clear_pending_side_size(context)
                step = determine_next_add_item_step(context)
                return self._step_to_result(context, step)

        if intent in SOFT_SWITCH_INTENTS:
            context.awaiting_flow_confirmation = True
            context.return_state = ConversationState.WAITING_FOR_SIDE_SIZE
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

        if _looks_like_pure_size_answer(normalized_user_text, choices):
            matched = _match_variant_value_normalized(normalized_user_text, choices)
            if matched is not None:
                context.selected_side_variants[side_choice.item_id] = matched.variant_id
                self._clear_pending_side_size(context)
                step = determine_next_add_item_step(context)
                return self._step_to_result(context, step)

        return HandlerResult(
            next_state=ConversationState.WAITING_FOR_SIDE_SIZE,
            response_key="invalid_side_size_option",
            response_payload={
                "item_name": pending.item_name,
                "side_item_name": side_choice.name,
                "available_sizes": [choice.name for choice in choices],
            },
        )

    def _find_pending_side_choice(self, context: ConversationContext):
        pending = context.pending_add_item
        pending_side_item_id = context.pending_side_item_id
        pending_side_group_id = context.pending_side_group_id

        if pending is None or not pending_side_item_id:
            return None

        if pending_side_group_id:
            for group in pending.side_groups:
                if group.group_id != pending_side_group_id:
                    continue
                for choice in group.choices:
                    if choice.item_id == pending_side_item_id:
                        return choice

        for group in pending.side_groups:
            for choice in group.choices:
                if choice.item_id == pending_side_item_id:
                    return choice

        return None

    def _clear_pending_side_size(self, context: ConversationContext) -> None:
        context.pending_side_item_id = None
        context.pending_side_item_name = None
        context.pending_side_group_id = None

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