# app/state_machine/handlers/item/waiting_for_modifier_handler.py

from app.state_machine.base_handler import BaseHandler
from app.state_machine.handler_result import HandlerResult
from app.state_machine.conversation_state import ConversationState
from app.state_machine.context import ConversationContext
from app.nlu.intent import Intent
from app.menu.menu_repository import MenuRepository


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
    ) -> HandlerResult:

        # Global cancel
        if intent == Intent.CANCEL:
            context.reset()
            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="action_cancelled",
            )

        item = self.menu_repo.items.get(context.current_item_id)
        if not item:
            return HandlerResult(
                next_state=ConversationState.ERROR_RECOVERY,
                response_key="item_context_missing",
            )

        # Find next modifier group
        next_group = None
        for group in item.modifier_groups:
            if group.group_id in context.skipped_modifier_groups:
                continue

            selected = context.selected_modifier_groups.get(group.group_id, [])

            if group.is_required and len(selected) < group.min_selector:
                next_group = group
                break

            if not group.is_required and group.group_id not in context.selected_modifier_groups:
                next_group = group
                break

        if not next_group:
            # No more modifiers needed
            if item.pricing.mode == "variant":
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_SIZE,
                    response_key="ask_for_size",
                )

            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_QUANTITY,
                response_key="ask_for_quantity",
            )

        # Handle skip / deny
        if intent == Intent.DENY:
            if next_group.is_required:
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_MODIFIER,
                    response_key="modifier_required",
                )

            context.skipped_modifier_groups.add(next_group.group_id)
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_MODIFIER,
                response_key="ask_for_modifier",
            )

        # Try to match modifier choice
        user_text_lower = user_text.lower()
        matched = None

        for choice in next_group.choices:
            if choice.name.lower() in user_text_lower:
                matched = choice
                break

        if not matched:
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_MODIFIER,
                response_key="repeat_modifier_options",
            )

        # Ask for confirmation
        context.awaiting_confirmation_for = {
            "type": "modifier",
            "group_id": next_group.group_id,
            "value_id": matched.modifier_id,
        }

        return HandlerResult(
            next_state=ConversationState.CONFIRMING,
            response_key="confirm_modifier_selection",
        )
