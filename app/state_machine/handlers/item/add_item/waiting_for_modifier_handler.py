# app/state_machine/handlers/item/add_item/waiting_for_modifier_handler.py
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


class WaitingForModifierHandler(BaseHandler):
    """
    Handles modifier selection for the current item.
    Processes one modifier group at a time.
    """

    def __init__(self, menu_repo: MenuRepository) -> None:
        self.menu_repo = menu_repo

    def handle(
        self,
        intent: Intent,
        context: ConversationContext,
        user_text: str,
        session: Session = None,
    ) -> HandlerResult:

        signal = resolve_choice_signal(user_text)

        # -------- CANCEL --------
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

        idx = context.current_modifier_group_index
        total = len(item.modifier_groups)

        # -------- End of modifier groups --------
        if idx >= total:
            return self._advance_after_modifiers(context, item)

        group = item.modifier_groups[idx]

        # -------- ASK OPTIONS --------
        if signal == ChoiceSignal.ASK_OPTIONS:
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_MODIFIER,
                response_key="list_modifier_options",
                response_payload={
                    "top_choices": [c.name for c in get_top_k_choices(group.choices, 3)],
                    "group_name": group.name,
                },
            )

        # -------- DENY --------
        if signal == ChoiceSignal.DENY:
            if group.is_required:
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_MODIFIER,
                    response_key="required_modifier_cannot_skip",
                    response_payload={
                        "top_choices": [c.name for c in get_top_k_choices(group.choices, 3)],
                        "group_name": group.name,
                    },
                )

            context.current_modifier_group_index += 1
            return self._advance_after_modifiers(context, item)

        # -------- MATCH MODIFIERS --------
        matched_ids = []
        invalid_terms = []

        for chunk in split_candidates(user_text):
            choice = match_choice(chunk, group.choices)
            if choice:
                matched_ids.append(choice.modifier_id)
            else:
                invalid_terms.append(chunk)

        if invalid_terms:
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_MODIFIER,
                response_key="repeat_modifier_options",
                response_payload={
                    "top_choices": [c.name for c in get_top_k_choices(group.choices, 3)],
                    "group_name": group.name,
                },
            )

        context.selected_modifier_groups.setdefault(group.group_id, []).extend(
            matched_ids
        )

        context.current_modifier_group_index += 1
        return self._advance_after_modifiers(context, item)

    def _advance_after_modifiers(self, context: ConversationContext, item):
        if context.current_modifier_group_index < len(item.modifier_groups):
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_MODIFIER,
                response_key="ask_for_modifier",
            )

        return HandlerResult(
            next_state=ConversationState.WAITING_FOR_QUANTITY,
            response_key="ask_for_quantity",
            response_payload={"item_name": item.name},
        )


