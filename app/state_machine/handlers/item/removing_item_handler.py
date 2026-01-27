# app/state_machine/handlers/item/removing_item_handler.py
from app.session.session import Session
from app.state_machine.base_handler import BaseHandler
from app.state_machine.handler_result import HandlerResult
from app.state_machine.conversation_state import ConversationState
from app.state_machine.context import ConversationContext
from app.nlu.intent_resolution.intent import Intent


class RemovingItemHandler(BaseHandler):
    """
    Handles confirmation for item removal.
    User confirms or denies removing the item from cart.
    """

    def handle(
        self,
        intent: Intent,
        context: ConversationContext,
        user_text: str,
        session: Session = None,
    ) -> HandlerResult:

        confirmation = context.awaiting_confirmation_for

        if not confirmation or confirmation.get("type") != "remove_item":
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

        # User denies removal
        if intent == Intent.DENY:
            context.reset()
            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="item_removal_cancelled",
            )

        # User confirms removal
        if intent == Intent.CONFIRM:
            cart_item_id = confirmation.get("cart_item_id")
            
            if not cart_item_id:
                return HandlerResult(
                    next_state=ConversationState.ERROR_RECOVERY,
                    response_key="cart_item_id_missing",
                )

            # Clean up context BEFORE command execution
            item_name = confirmation.get("item_name", "item")
            context.reset()

            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="item_removed_successfully",
                response_payload={
                    "item_name": item_name,
                },
                command={
                    "type": "REMOVE_ITEM_FROM_CART",
                    "payload": {
                        "cart_item_id": cart_item_id,
                    },
                },
            )

        # Fallback: repeat confirmation
        return HandlerResult(
            next_state=ConversationState.REMOVING_ITEM,
            response_key="repeat_remove_confirmation",
        )
