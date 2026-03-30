# app/state_machine/handlers/item/add_item/waiting_for_size_handler.py
from __future__ import annotations

from app.menu.repository import MenuRepository
from app.nlu.intent_resolution.intent import Intent
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


def _extract_size_slot_values_normalized(context: ConversationContext) -> list[str]:
    """
    Extract size-like values from slots.

    Keep this tolerant because different slot models may emit SIZE,
    VARIANT, or even ITEM for utterances like 'small coke'.
    """
    slots = context.last_slots or ()
    values: list[str] = []
    seen: set[str] = set()

    for slot in slots:
        name = str(slot.name).upper()
        if name not in {"SIZE", "VARIANT"}:
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


def _looks_like_pure_size_answer(
    normalized_user_text: str,
    normalized_choice_names: tuple[str, ...],
) -> bool:
    """
    Conservative direct-answer detector for size selection.

    Accept:
    - small
    - large please
    - make it medium
    - i want small

    Reject:
    - how much is small
    - show menu
    - checkout
    - add burger
    """
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
        "make",
        "it",
        "i",
        "want",
        "would",
        "like",
        "to",
        "have",
        "my",
        "um",
        "uh",
        "okay",
        "ok",
        "ill",
        "get",
        "take",
    }
    blocked_phrases = {
        "how much",
        "price",
        "cost",
        "show menu",
        "show me",
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

    tokens = [token for token in normalized_user_text.split() if token not in filler_words]
    compact = " ".join(tokens).strip()
    if not compact:
        return False

    if compact in normalized_choice_names:
        return True

    for choice_name in normalized_choice_names:
        if len(compact) >= 3 and (compact in choice_name or choice_name in compact):
            return True

    return False


class WaitingForSizeHandler(BaseHandler):
    """
    Resolve the size / variant for the main pending item.

    Important:
    - waiting state owns the turn
    - size resolution is slot-first, then direct-text fallback
    - accepts short natural answers like 'small' and mixed answers like 'make it large'
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

        if not pending.item_variants:
            step = determine_next_add_item_step(context)
            return self._step_to_result(context, step)

        normalized_user_text = user_text or ""
        available_sizes = list(pending.item_variant_names)
        choices_by_normalized_name = pending.item_variants_by_normalized_name
        normalized_choice_names = tuple(choices_by_normalized_name.keys())

        if intent == Intent.DENY:
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_SIZE,
                response_key="required_size_cannot_skip",
                response_payload={
                    "item_name": pending.item_name,
                    "available_sizes": available_sizes,
                },
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

        normalized_slot_values = _extract_size_slot_values_normalized(context)
        if normalized_slot_values:
            matched = self._match_variant_from_values(
                normalized_values=normalized_slot_values,
                choices_by_normalized_name=choices_by_normalized_name,
            )
            if matched is not None:
                context.selected_variant_id = matched.variant_id
                self._clear_pending_size_prompt(context)

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
            matched = self._match_variant_from_values(
                normalized_values=[normalized_user_text],
                choices_by_normalized_name=choices_by_normalized_name,
            )
            if matched is not None:
                context.selected_variant_id = matched.variant_id
                self._clear_pending_size_prompt(context)

                step = determine_next_add_item_step(context)
                return self._step_to_result(context, step)

        return HandlerResult(
            next_state=ConversationState.WAITING_FOR_SIZE,
            response_key="repeat_size_options",
            response_payload={
                "item_name": pending.item_name,
                "available_sizes": available_sizes,
            },
        )

    def _match_variant_from_values(
        self,
        *,
        normalized_values: list[str],
        choices_by_normalized_name: dict[str, PendingVariantChoice],
    ) -> PendingVariantChoice | None:
        candidate_texts = build_candidate_texts_normalized(
            normalized_user_text="",
            normalized_slot_values=normalized_values,
            allow_split=True,
        )

        for candidate in candidate_texts:
            matched = choices_by_normalized_name.get(candidate)
            if matched is not None:
                return matched

        for candidate in candidate_texts:
            if len(candidate) < 3:
                continue

            for choice_name, choice in choices_by_normalized_name.items():
                if candidate in choice_name or choice_name in candidate:
                    return choice

        return None

    def _clear_pending_size_prompt(self, context: ConversationContext) -> None:
        if context.current_prompt_field == "size":
            context.current_prompt_field = None

        if context.available_choices_kind == "size":
            context.available_choices_kind = None
            context.available_choices_values = ()

        context.size_target = None

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