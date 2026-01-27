# app/state_machine/handlers/item/add_item_handler.py
from typing import Optional

from app.nlu.intent_patterns.quantity import *
from app.session.session import Session
from app.state_machine.base_handler import BaseHandler
from app.state_machine.handler_result import HandlerResult
from app.state_machine.conversation_state import ConversationState
from app.state_machine.context import ConversationContext
from app.nlu.intent_resolution.intent import Intent
from app.menu.repository import MenuRepository
from app.utils.quantity_detection import normalize_quantity


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

        resolution = self.menu_repo.resolve_item(user_text)

        if not resolution:
            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="item_not_found",
            )

        item = resolution.item
        score = resolution.score

        # ---------- Ambiguity ----------
        if score < 6.0:
            context.reset()
            context.awaiting_confirmation_for = {
                "type": "item",
                "value_id": item.item_id,
                "value_name": item.name,
            }
            context.candidate_item_id = item.item_id
            context.candidate_item_name = item.name

            return HandlerResult(
                next_state=ConversationState.CONFIRMING_ITEM,
                response_key="confirm_item",
            )

        # ---------- Initialize context ----------
        context.reset()
        context.current_item_id = item.item_id
        context.current_item_name = item.name
        context.pending_action = "add"

        explicit_qty = self.extract_explicit_quantity(user_text)
        if explicit_qty:
            context.quantity = explicit_qty

        # ---------- STRUCTURE FIRST ----------
        if item.pricing.mode == "variant":
            context.size_target = {"type": "item"}
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_SIZE,
                response_key="ask_for_size",
            )

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

        # ---------- ALWAYS funnel to quantity ----------
        return HandlerResult(
            next_state=ConversationState.WAITING_FOR_QUANTITY,
            response_key="ask_for_quantity",
            response_payload={"item_name": item.name},
        )

    def extract_explicit_quantity(self, text: str) -> Optional[int]:
        if PURE_QUANTITY_PAT.match(text):
            return normalize_quantity(text)
        if QUANTITY_NOUN_PAT.search(text):
            return normalize_quantity(text)
        return None

