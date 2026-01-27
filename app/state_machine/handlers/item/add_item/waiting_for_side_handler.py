# app/state_machine/handlers/item/waiting_for_side_handler.py
from app.nlu.intent_resolution.choice_signal_resolver import resolve_choice_signal
from app.nlu.intent_resolution.choice_signals import ChoiceSignal
from app.session.session import Session
from app.state_machine.base_handler import BaseHandler
from app.state_machine.handler_result import HandlerResult
from app.state_machine.conversation_state import ConversationState
from app.state_machine.context import ConversationContext
from app.nlu.intent_resolution.intent import Intent
from app.menu.repository import MenuRepository
from app.utils.choice_matching import match_choice
from app.utils.text_utils import split_candidates
from app.utils.top_k_choices import get_top_k_choices
import re


class WaitingForSideHandler(BaseHandler):
    """
    Handles side selection for items that require sides.
    Enforces selection one side group at a time.
    """

    def __init__(self, menu_repo: MenuRepository) -> None:
        self.menu_repo = menu_repo

    def handle(
        self,
        intent: Intent,  # ignored in locked state
        context: ConversationContext,
        user_text: str,
        session: Session = None,
    ) -> HandlerResult:

        signal = resolve_choice_signal(user_text)

        # -------------------------
        # CANCEL
        # -------------------------
        if signal == ChoiceSignal.CANCEL:
            context.reset()
            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="action_cancelled",
            )

        item = self.menu_repo.get_item(context.current_item_id)
        if not item:
            return HandlerResult(
                next_state=ConversationState.ERROR_RECOVERY,
                response_key="item_context_missing",
            )

        idx = context.current_side_group_index

        # -------------------------
        # End of side groups
        # -------------------------
        if idx >= len(item.side_groups):
            if item.modifier_groups:
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_MODIFIER,
                    response_key="ask_for_modifier",
                )

            if item.pricing.mode == "variant":
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_SIZE,
                    response_key="ask_for_size",
                )

            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_QUANTITY,
                response_key="ask_for_quantity",
            )

        group = item.side_groups[idx]

        # -------------------------
        # ASK OPTIONS
        # -------------------------
        if signal == ChoiceSignal.ASK_OPTIONS:
            top_choices = [c.name for c in get_top_k_choices(group.choices, k=3)]
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_SIDE,
                response_key="list_side_options",
                response_payload={
                    "top_choices": top_choices,
                    "group_name": group.name,
                },
            )

        # -------------------------
        # DENY / SKIP
        # -------------------------
        if signal == ChoiceSignal.DENY:
            if group.is_required:
                top_choices = [c.name for c in get_top_k_choices(group.choices, k=3)]
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_SIDE,
                    response_key="required_side_cannot_skip",
                    response_payload={
                        "top_choices": top_choices,
                        "group_name": group.name,
                    },
                )

            context.current_side_group_index = len(item.side_groups)
            return self._advance_after_side(context, item)

        # -------------------------
        # OPTIONAL SILENCE SKIP
        # -------------------------
        if not user_text.strip() and not group.is_required:
            context.current_side_group_index += 1
            return self._advance_after_side(context, item)

        # -------------------------
        # MATCH SIDE CHOICE
        # -------------------------
        matched_choices = []

        for chunk in split_candidates(user_text):
            choice = match_choice(chunk, group.choices)
            if choice and choice not in matched_choices:
                matched_choices.append(choice)

        if not matched_choices:
            top_choices = [c.name for c in get_top_k_choices(group.choices, k=3)]
            if group.is_required:
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_SIDE,
                    response_key="required_side_cannot_skip",
                    response_payload={
                        "top_choices": top_choices,
                        "group_name": group.name,
                    },
                )

            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_SIDE,
                response_key="repeat_side_options",
                response_payload={
                    "top_choices": top_choices,
                    "group_name": group.name,
                },
            )

        if group.max_selector == 1 and len(matched_choices) > 1:
            top_choices = [c.name for c in get_top_k_choices(group.choices, k=3)]
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_SIDE,
                response_key="too_many_side_choices",
                response_payload={
                    "top_choices": top_choices,
                    "group_name": group.name,
                },
            )

        # -------------------------
        # Commit side
        # -------------------------
        chosen = matched_choices[0]
        context.selected_side_groups.setdefault(group.group_id, []).append(
            chosen.item_id
        )

        context.current_side_group_index += 1
        return self._advance_after_side(context, item)

    def _advance_after_side(self, context: ConversationContext, item):
        if context.current_side_group_index < len(item.side_groups):
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_SIDE,
                response_key="ask_for_side",
            )

        if item.modifier_groups:
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_MODIFIER,
                response_key="ask_for_modifier",
            )

        if item.pricing.mode == "variant":
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_SIZE,
                response_key="ask_for_size",
            )

        return HandlerResult(
            next_state=ConversationState.WAITING_FOR_QUANTITY,
            response_key="ask_for_quantity",
        )

