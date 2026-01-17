# app/state_machine/handlers/item/waiting_for_size_handler.py

from app.state_machine.base_handler import BaseHandler
from app.state_machine.handler_result import HandlerResult
from app.state_machine.conversation_state import ConversationState
from app.state_machine.context import ConversationContext
from app.nlu.intent_resolution.intent import Intent
from app.menu.repository import MenuRepository


class WaitingForSizeHandler(BaseHandler):
    """
    Resolves size for the current size_target (item for now).
    """

    def __init__(self, menu_repo: MenuRepository) -> None:
        self.menu_repo = menu_repo

    def handle(
        self,
        intent: Intent,
        context: ConversationContext,
        user_text: str,
    ) -> HandlerResult:

        if intent == Intent.CANCEL:
            context.reset()
            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="action_cancelled",
            )

        if not context.size_target or context.size_target["type"] != "item":
            # Defensive guard
            return HandlerResult(
                next_state=ConversationState.ERROR_RECOVERY,
                response_key="size_target_missing",
            )

        item = self.menu_repo.get_item(context.current_item_id)
        if not item or item.pricing.mode != "variant":
            return HandlerResult(
                next_state=ConversationState.ERROR_RECOVERY,
                response_key="item_context_missing",
            )

        matched_variant = None
        for v in item.pricing.variants:
            if v.label.lower() in user_text.lower():
                matched_variant = v
                break

        if not matched_variant:
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_SIZE,
                response_key="repeat_size_options",
            )

        # ✅ Commit
        context.selected_variant_id = matched_variant.variant_id
        context.size_target = None  # 🔑 reset size mode

        # ✅ Decide next step
        if item.side_groups:
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_SIDE,
                response_key="ask_for_side",
            )

        if item.modifier_groups:
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_MODIFIER,
                response_key="ask_for_modifier",
            )

        # ✅ SIZE-ONLY ITEM → FINALIZE
        return HandlerResult(
            next_state=ConversationState.IDLE,
            response_key="item_added_successfully",
            command={
                "type": "ADD_ITEM_TO_CART",
                "payload": {
                    "item_id": context.current_item_id,
                    "quantity": 1,
                    "variant_id": context.selected_variant_id,
                    "sides": {},
                    "modifiers": {},
                },
            },
            response_payload={
                "item_id": context.current_item_id,
                "item_name": item.name,
                "quantity": 1,
            },
        )


