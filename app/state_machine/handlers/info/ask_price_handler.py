# app/state_machine/handlers/info/ask_price_handler.py
from app.menu.query_result import MenuQueryType
from app.menu.repository import MenuRepository
from app.nlu.intent_resolution.intent import Intent
from app.session.session import Session
from app.state_machine.base_handler import BaseHandler
from app.state_machine.context import ConversationContext
from app.state_machine.conversation_state import ConversationState
from app.state_machine.handler_result import HandlerResult


class AskPriceHandler(BaseHandler):
    """
    Handles item price inquiries.
    Read-only. No cart or state mutation.
    """

    def __init__(self, menu_repo: MenuRepository):
        self.menu_repo = menu_repo

    def handle(
        self,
        intent: Intent,
        context: ConversationContext,
        user_text: str,
        session: Session,
    ) -> HandlerResult:

        result = self.menu_repo.resolve_menu_query(user_text)

        # Only item price queries are supported
        if result.type != MenuQueryType.ITEM:
            return HandlerResult(
                response_key="price_not_found",
                next_state=ConversationState.IDLE,
            )

        item = result.item
        pricing = item.pricing

        return HandlerResult(
            response_key="show_item_price",
            response_payload={
                "item_name": item.name,
                "pricing": pricing,
            },
            next_state=ConversationState.IDLE,
        )
