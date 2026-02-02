# app/state_machine/handlers/item/add_item/waiting_for_modifier_handler.py
from app.nlu.choice_signals.resolver import resolve_choice_signal
from app.nlu.choice_signals.choice_signals import ChoiceSignal
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
        session: Session | None = None,
    ) -> HandlerResult:

        session_id = session.session_id if session else "n/a"
        signal = resolve_choice_signal(user_text)

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
        if idx >= len(item.modifier_groups):
            return self._advance_after_modifiers(context, item)

        group = item.modifier_groups[idx]

        if signal == ChoiceSignal.ASK_OPTIONS:
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_MODIFIER,
                response_key="repeat_modifier_options",
                response_payload={
                    "group_name": group.name,
                    "top_choices": [c.name for c in get_top_k_choices(group.choices, 3)],
                    "repeat_reason": "options",
                },
            )

        if signal == ChoiceSignal.DENY:
            if group.is_required:
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_MODIFIER,
                    response_key="required_modifier_cannot_skip",
                    response_payload={
                        "group_name": group.name,
                        "top_choices": [c.name for c in get_top_k_choices(group.choices, 3)],
                    },
                )

            context.current_modifier_group_index += 1
            return self._advance_after_modifiers(context, item)

        matched_ids, invalid_terms = [], []

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
                    "group_name": group.name,
                    "top_choices": [c.name for c in get_top_k_choices(group.choices, 4)],
                    "invalid_terms": invalid_terms,
                    "repeat_reason": "invalid",
                },
            )

        context.selected_modifier_groups.setdefault(group.group_id, []).extend(matched_ids)
        context.current_modifier_group_index += 1

        return self._advance_after_modifiers(context, item)

    def _advance_after_modifiers(
            self,
            context: ConversationContext,
            item,
    ) -> HandlerResult:
        """
        Advances the flow after completing modifier selection.

        Order of operations:
        1. More modifier groups → continue modifiers
        2. Otherwise → quantity
        """

        # More modifier groups remaining
        if context.current_modifier_group_index < len(item.modifier_groups):
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_MODIFIER,
                response_key="ask_for_modifier",
            )

        # Proceed to quantity
        return HandlerResult(
            next_state=ConversationState.WAITING_FOR_QUANTITY,
            response_key="ask_for_quantity",
            response_payload={"item_name": item.name},
        )


