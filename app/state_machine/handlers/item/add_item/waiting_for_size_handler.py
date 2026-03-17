# app/state_machine/handlers/item/add_item/waiting_for_size_handler.py
from __future__ import annotations

from app.menu.repository import MenuRepository
from app.nlu.intent_resolution.intent import Intent
from app.nlu.nlu_result import SlotValue
from app.nlu.query_normalization.text_preprocessor import normalize_text
from app.session.session import Session
from app.state_machine.conversation_context import (
    ConversationContext,
    InterruptProposal,
    PendingVariantChoice,
)
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


def _normalize_answer_text(normalized_user_text: str) -> str:
    if not normalized_user_text:
        return ""

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
        "i",
        "want",
        "id",
        "i'd",
        "ill",
        "i'll",
        "give",
        "me",
    }

    tokens = [token for token in normalized_user_text.split() if token not in filler_words]
    return " ".join(tokens).strip()


def _match_variant_value_normalized(
    normalized_value: str,
    choices_by_normalized_name: dict[str, PendingVariantChoice],
    choices: list[PendingVariantChoice],
) -> PendingVariantChoice | None:
    if not normalized_value:
        return None

    compact_value = _normalize_answer_text(normalized_value)
    if not compact_value:
        return None

    exact = choices_by_normalized_name.get(compact_value)
    if exact is not None:
        return exact

    if len(compact_value) < 3:
        return None

    for choice in choices:
        choice_name = choice.normalized_name
        if compact_value in choice_name or choice_name in compact_value:
            return choice

    return None


def _looks_like_pure_size_answer(
    normalized_user_text: str,
    normalized_choice_names: tuple[str, ...],
) -> bool:
    """
    Conservative answer detector.

    Accept:
    - small
    - medium please
    - large one
    - the large

    Reject:
    - how much is medium coke
    - add large coke
    - show me menu
    """
    compact = _normalize_answer_text(normalized_user_text)
    if not compact:
        return False

    for choice_name in normalized_choice_names:
        if compact == choice_name:
            return True
        if len(compact) >= 3 and (compact in choice_name or choice_name in compact):
            return True

    return False


class WaitingForSizeHandler(BaseHandler):
    """
    Resolve main item size/variant from the active pending item snapshot.

    Critical rules:
    - do NOT consume new requests as size answers
    - only accept explicit SIZE slot or short direct size answers
    - after read-only interrupts, current add-item flow must remain resumable
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
        if pending is None or not pending.item_id:
            return HandlerResult(
                next_state=ConversationState.ERROR_RECOVERY,
                response_key="item_context_missing",
            )

        normalized_user_text = user_text or ""
        choices = pending.item_variants

        if not choices:
            context.size_target = None
            step = determine_next_add_item_step(context)
            return self._step_to_result(context, step)

        available_sizes = list(pending.item_variant_names)
        normalized_choice_names = tuple(pending.item_variants_by_normalized_name.keys())

        if intent == Intent.DENY:
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_SIZE,
                response_key="required_size_cannot_skip",
                response_payload={"item_name": pending.item_name},
            )

        if intent == Intent.ASK_OPTIONS:
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_SIZE,
                response_key="repeat_size_options",
                response_payload={
                    "item_name": pending.item_name,
                    "available_sizes": available_sizes,
                },
            )

        slots = context.last_slots or ()
        slot_value = _first_size_slot_normalized(slots)

        if slot_value:
            matched = _match_variant_value_normalized(
                slot_value,
                pending.item_variants_by_normalized_name,
                choices,
            )
            if matched is not None:
                context.selected_variant_id = matched.variant_id
                context.size_target = None

                step = determine_next_add_item_step(context)
                return self._step_to_result(context, step)

        if intent in SOFT_SWITCH_INTENTS:
            context.awaiting_flow_confirmation = True
            context.return_state = ConversationState.WAITING_FOR_SIZE
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

        if _looks_like_pure_size_answer(normalized_user_text, normalized_choice_names):
            matched = _match_variant_value_normalized(
                normalized_user_text,
                pending.item_variants_by_normalized_name,
                choices,
            )
            if matched is not None:
                context.selected_variant_id = matched.variant_id
                context.size_target = None

                step = determine_next_add_item_step(context)
                return self._step_to_result(context, step)

        return HandlerResult(
            next_state=ConversationState.WAITING_FOR_SIZE,
            response_key="invalid_size_option",
            response_payload={
                "item_name": pending.item_name,
                "available_sizes": available_sizes,
            },
        )

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