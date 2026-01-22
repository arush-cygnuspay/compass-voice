# app/state_machine/handlers/item/waiting_for_side_handler.py
from app.session.session import Session
from app.state_machine.base_handler import BaseHandler
from app.state_machine.handler_result import HandlerResult
from app.state_machine.conversation_state import ConversationState
from app.state_machine.context import ConversationContext
from app.nlu.intent_resolution.intent import Intent
from app.menu.repository import MenuRepository
from app.utils.choice_matching import match_choice


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
        session: Session = None,
    ) -> HandlerResult:

        # -------------------------
        # Global cancel
        # -------------------------
        if intent == Intent.CANCEL:
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
        # DENY on REQUIRED side
        # -------------------------
        if intent == Intent.DENY and group.is_required:
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_SIDE,
                response_key="required_side_cannot_skip",
            )

        # -------------------------
        # Skip OPTIONAL side
        # -------------------------
        if intent == Intent.DENY and not group.is_required:
            context.current_side_group_index += 1
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_SIDE,
                response_key="ask_for_side",
            )

        # -------------------------
        # Match side choice
        # -------------------------
        matched_choice = match_choice(user_text, group.choices)

        if not matched_choice:
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_SIDE,
                response_key="repeat_side_options",
            )

        # -------------------------
        # Commit side
        # -------------------------
        context.selected_side_groups.setdefault(group.group_id, []).append(
            matched_choice.item_id
        )

        context.current_side_group_index += 1

        # ---------------------------------
        # Decide next step AFTER side commit
        # ---------------------------------
        if context.current_side_group_index < len(item.side_groups):
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_SIDE,
                response_key="ask_for_side",
            )

        # Side groups finished → move on
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


