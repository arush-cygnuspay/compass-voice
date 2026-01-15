# app/state_machine/handlers/item/confirming_handler.py

from app.state_machine.base_handler import BaseHandler
from app.state_machine.handler_result import HandlerResult
from app.state_machine.conversation_state import ConversationState
from app.state_machine.context import ConversationContext
from app.nlu.intent import Intent
from app.menu.menu_repository import MenuRepository


class ConfirmingHandler(BaseHandler):
    """
    Handles confirmation steps for item, side, modifier, size, quantity.
    """

    def __init__(self, menu_repo: MenuRepository) -> None:
        self.menu_repo = menu_repo

    def handle(
        self,
        intent: Intent,
        context: ConversationContext,
    ) -> HandlerResult:

        confirmation = context.awaiting_confirmation_for

        if not confirmation:
            return HandlerResult(
                next_state=ConversationState.ERROR_RECOVERY,
                response_key="confirmation_state_error",
            )

        # Global cancellation
        if intent == Intent.CANCEL:
            context.reset()
            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="action_cancelled",
            )

        # Item confirmation flow
        if confirmation["type"] == "item":

            if intent == Intent.DENY:
                context.candidate_item_id = None
                context.awaiting_confirmation_for = None
                return HandlerResult(
                    next_state=ConversationState.IDLE,
                    response_key="item_confirmation_denied",
                )

            if intent == Intent.CONFIRM:
                item_id = context.candidate_item_id
                context.current_item_id = item_id
                context.candidate_item_id = None
                context.awaiting_confirmation_for = None

                item = self.menu_repo.items.get(item_id)

                # Decide next step based on item structure
                if item.side_groups:
                    return HandlerResult(
                        next_state=ConversationState.WAITING_FOR_SIDE,
                        response_key="ask_for_side",
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

        # Any other case → re-ask confirmation
        return HandlerResult(
            next_state=ConversationState.CONFIRMING,
            response_key="repeat_confirmation",
        )
