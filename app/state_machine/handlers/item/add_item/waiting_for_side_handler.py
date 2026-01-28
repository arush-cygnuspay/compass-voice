# app/state_machine/handlers/item/waiting_for_side_handler.py
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


class WaitingForSideHandler(BaseHandler):
    """
    Handles side selection for items that require sides.
    Enforces selection one side group at a time.
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

        # -------------------------
        # Local cancellation
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
        # All side groups complete
        # -------------------------
        if idx >= len(item.side_groups):
            return self._advance_after_sides(context, item)

        group = item.side_groups[idx]

        # -------------------------
        # ASK_OPTIONS → informational
        # -------------------------
        if signal == ChoiceSignal.ASK_OPTIONS:
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_SIDE,
                response_key="repeat_side_options",
                response_payload={
                    "group_name": group.name,
                    "top_choices": [c.name for c in get_top_k_choices(group.choices, 4)],
                    "repeat_reason": "options",
                },
            )

        # -------------------------
        # DENY / SKIP
        # -------------------------
        if signal == ChoiceSignal.DENY:
            if group.is_required:
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_SIDE,
                    response_key="required_side_cannot_skip",
                    response_payload={
                        "group_name": group.name,
                        "top_choices": [c.name for c in get_top_k_choices(group.choices, 3)],
                    },
                )

            context.current_side_group_index += 1
            return self._advance_after_sides(context, item)

        # -------------------------
        # Match side choice
        # -------------------------
        matched = [
            choice
            for chunk in split_candidates(user_text)
            if (choice := match_choice(chunk, group.choices))
        ]

        if not matched:
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_SIDE,
                response_key="repeat_side_options",
                response_payload={
                    "group_name": group.name,
                    "top_choices": [c.name for c in get_top_k_choices(group.choices, 3)],
                    "repeat_reason": "invalid",
                },
            )

        if group.max_selector == 1 and len(matched) > 1:
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_SIDE,
                response_key="too_many_side_choices",
                response_payload={
                    "group_name": group.name,
                    "top_choices": [c.name for c in get_top_k_choices(group.choices, 3)],
                },
            )

        # -------------------------
        # Commit selection
        # -------------------------
        context.selected_side_groups.setdefault(group.group_id, []).append(
            matched[0].item_id
        )
        context.current_side_group_index += 1

        return self._advance_after_sides(context, item)

    def _advance_after_sides(
            self,
            context: ConversationContext,
            item,
    ) -> HandlerResult:
        """
        Advances the flow after completing side selection.

        Order of operations:
        1. More side groups → continue sides
        2. Modifier groups → move to modifiers
        3. Otherwise → quantity
        """

        # More side groups remaining
        if context.current_side_group_index < len(item.side_groups):
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_SIDE,
                response_key="ask_for_side",
            )

        # Proceed to modifiers if any
        if item.modifier_groups:
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_MODIFIER,
                response_key="ask_for_modifier",
            )

        # Otherwise go to quantity
        return HandlerResult(
            next_state=ConversationState.WAITING_FOR_QUANTITY,
            response_key="ask_for_quantity",
            response_payload={"item_name": item.name},
        )


