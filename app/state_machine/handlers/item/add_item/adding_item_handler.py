# app/state_machine/handlers/item/add_item_handler.py
from app.session.session import Session
from app.state_machine.base_handler import BaseHandler
from app.state_machine.handler_result import HandlerResult
from app.state_machine.conversation_state import ConversationState
from app.state_machine.context import ConversationContext
from app.nlu.intent_resolution.intent import Intent
from app.menu.repository import MenuRepository
from app.state_machine.handlers.item.add_item.add_item_flow import _response_key_for_state, \
    determine_next_add_item_state


class AddItemHandler(BaseHandler):
    """
    Handles initial item addition intent.
    Resolves item using MenuRepository (entity-first).
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

        if intent != Intent.ADD_ITEM:
            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="unhandled_intent",
            )

        # -----------------------------
        # Resolve item from menu
        # -----------------------------
        item = self.menu_repo.resolve_item(user_text)

        if not item:
            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="item_not_found",
            )

        # -----------------------------
        # Ask for confirmation
        # -----------------------------
        # After resolving item
        context.current_item_id = item.item_id
        context.current_item_name = item.name
        context.pending_action = "add"
        context.current_side_group_index = 0

        context.candidate_item_id = None
        context.awaiting_confirmation_for = None
        context.selected_side_groups.clear()
        context.selected_modifier_groups.clear()
        context.selected_variant_id = None
        context.quantity = None

        next_state = determine_next_add_item_state(item, context)

        print("✅ SETTING SIZE TARGET:", {
            "type": "item",
            "item_id": item.item_id
        })

        # after resolving item and before returning
        if item.pricing.mode == "variant":
            context.size_target = {
                "type": "item",
                "item_id": item.item_id,
            }

        return HandlerResult(
            next_state=next_state,
            response_key=_response_key_for_state(next_state),
            response_payload=None
        )
