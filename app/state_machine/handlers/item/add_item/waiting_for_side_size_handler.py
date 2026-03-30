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
from app.utils.token_matcher import is_controlled_partial_match, is_strong_token_match

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


def _first_size_slot_normalized(slots) -> str | None:
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
    choices_by_normalized_name: dict[str, PendingVariantChoice],
) -> PendingVariantChoice | None:
    if not normalized_value:
        return None

    exact = choices_by_normalized_name.get(normalized_value)
    if exact:
        return exact

    for name, choice in choices_by_normalized_name.items():
        if is_strong_token_match(normalized_value, name):
            return choice

    for name, choice in choices_by_normalized_name.items():
        if is_controlled_partial_match(normalized_value, name):
            return choice

    return None


def _looks_like_pure_size_answer(
    normalized_user_text: str,
    normalized_choice_names: tuple[str, ...],
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
        "um",
        "uh",
        "okay",
        "ok",
        "ill",
        "i",
        "want",
        "would",
        "like",
        "to",
        "have",
        "my",
        "make",
        "it",
        "get",
        "take",
    }
    blocked_phrases = {
        "how much",
        "price",
        "cost",
        "add",
        "show",
        "menu",
        "checkout",
        "check out",
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

    if compact in normalized_choice_names:
        return True

    for choice_name in normalized_choice_names:
        if len(compact) >= 3 and (compact in choice_name or choice_name in compact):
            return True

    return False


class WaitingForSideSizeHandler(BaseHandler):
    """
    Resolve the size / variant for the currently selected side item.

    Important:
    - waiting state owns the turn
    - size resolution is slot-first, then direct-text fallback
    - only sizes for the active pending side item may match
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
        pending_side_item_id = context.pending_side_item_id

        if pending is None or not pending_side_item_id:
            return HandlerResult(
                next_state=ConversationState.ERROR_RECOVERY,
                response_key="item_context_missing",
            )

        normalized_user_text = user_text or ""
        side_choice = pending.side_choice_by_item_id.get(pending_side_item_id)

        if side_choice is None:
            return HandlerResult(
                next_state=ConversationState.ERROR_RECOVERY,
                response_key="item_context_missing",
            )

        if side_choice.pricing_mode != "variant" or not side_choice.variants:
            self._clear_pending_side_size(context)
            step = determine_next_add_item_step(context)
            return self._step_to_result(context, step)

        available_sizes = list(side_choice.variant_names)
        normalized_choice_names = tuple(side_choice.variants_by_normalized_name.keys())

        if intent == Intent.DENY:
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_SIDE_SIZE,
                response_key="required_side_size_cannot_skip",
                response_payload={
                    "item_name": pending.item_name,
                    "side_item_name": side_choice.name,
                    "available_sizes": available_sizes,
                },
            )

        if intent == Intent.ASK_OPTIONS:
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_SIDE_SIZE,
                response_key="repeat_side_size_options",
                response_payload={
                    "item_name": pending.item_name,
                    "side_item_name": side_choice.name,
                    "available_sizes": available_sizes,
                },
            )

        slot_value = _first_size_slot_normalized(context.last_slots or ())
        if slot_value:
            matched = _match_variant_value_normalized(
                slot_value,
                side_choice.variants_by_normalized_name,
            )
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

        if _looks_like_pure_size_answer(normalized_user_text, normalized_choice_names):
            matched = _match_variant_value_normalized(
                normalized_user_text,
                side_choice.variants_by_normalized_name,
            )
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
                "available_sizes": available_sizes,
            },
        )

    def _clear_pending_side_size(self, context: ConversationContext) -> None:
        context.pending_side_item_id = None
        context.pending_side_item_name = None
        context.pending_side_group_id = None

        if context.current_prompt_field == "side_size":
            context.current_prompt_field = None

        if context.available_choices_kind == "side_size":
            context.available_choices_kind = None
            context.available_choices_values = ()

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