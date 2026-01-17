# app/state_machine/handlers/item/waiting_for_size_handler.py

from app.state_machine.base_handler import BaseHandler
from app.state_machine.handler_result import HandlerResult
from app.state_machine.conversation_state import ConversationState
from app.state_machine.context import ConversationContext
from app.nlu.intent_resolution.intent import Intent
from app.menu.repository import MenuRepository


class WaitingForSizeHandler(BaseHandler):
    """
    Handles pricing variant (size) selection for items with variant pricing.
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

        pricing = item.pricing
        if pricing.mode != "variant" or not pricing.variants:
            # Defensive: size not applicable
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_QUANTITY,
                response_key="ask_for_quantity",
            )

        user_text_lower = user_text.lower()
        matched_variant = None

        for variant in pricing.variants:
            if variant.label.lower() in user_text_lower:
                matched_variant = variant
                break

        if not matched_variant:
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_SIZE,
                response_key="repeat_size_options",
            )

        # Ask for confirmation
        context.awaiting_confirmation_for = {
            "type": "size",
            "value_id": matched_variant.variant_id,
        }

        return HandlerResult(
            next_state=ConversationState.CONFIRMING,
            response_key="confirm_size_selection",
        )
