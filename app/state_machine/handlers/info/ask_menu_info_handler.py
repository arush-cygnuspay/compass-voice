# app/state_machine/handlers/info/ask_menu_info_handler.py

from app.menu.query_result import MenuQueryType
from app.state_machine.base_handler import BaseHandler
from app.state_machine.handler_result import HandlerResult
from app.state_machine.conversation_state import ConversationState
from app.nlu.intent_resolution.intent import Intent


class AskMenuInfoHandler(BaseHandler):
    """
    Handles informational menu queries (items or categories).

    IMPORTANT:
    - Read-only
    - No state mutation
    - No cart mutation
    - No scoring or inference
    """

    def __init__(self, menu_repo):
        self.menu_repo = menu_repo

    def handle(
        self,
        intent: Intent,
        context,
        user_text: str,
        session,
    ) -> HandlerResult:

        result = self.menu_repo.resolve_menu_query(user_text)

        # --------------------------------------------------
        # CATEGORY (multiple items)
        # --------------------------------------------------
        if result.type == MenuQueryType.CATEGORY:
            return HandlerResult(
                response_key="show_category",
                response_payload={
                    "category_name": result.category_name,
                    "items": [item.name for item in (result.items or [])],
                },
                next_state=ConversationState.IDLE,
            )

        # --------------------------------------------------
        # ITEM (single resolved item)
        # --------------------------------------------------
        if result.type == MenuQueryType.ITEM:
            return HandlerResult(
                response_key="show_item_info",
                response_payload={
                    "item_name": result.item.name,
                },
                next_state=ConversationState.IDLE,
            )

        # --------------------------------------------------
        # ITEM AMBIGUOUS
        # --------------------------------------------------
        if result.type == MenuQueryType.ITEM_AMBIGUOUS:
            return HandlerResult(
                response_key="menu_ambiguity",
                response_payload={
                    "options": [item.name for item in (result.matched_items or [])],
                },
                next_state=ConversationState.IDLE,
            )

        # --------------------------------------------------
        # CATEGORY AMBIGUOUS
        # --------------------------------------------------
        if result.type == MenuQueryType.CATEGORY_AMBIGUOUS:
            return HandlerResult(
                response_key="menu_ambiguity",
                response_payload={
                    "options": [
                        cat["name"] for cat in (result.matched_categories or [])
                    ],
                },
                next_state=ConversationState.IDLE,
            )

        # --------------------------------------------------
        # CATEGORY WITH SINGLE ITEM
        # (Optional UX improvement — still dumb)
        # --------------------------------------------------
        if result.type == MenuQueryType.CATEGORY_SINGLE_ITEM:
            single_item = result.items[0]
            return HandlerResult(
                response_key="show_item_info",
                response_payload={
                    "item_name": single_item.name,
                },
                next_state=ConversationState.IDLE,
            )

        # --------------------------------------------------
        # NOT FOUND
        # --------------------------------------------------
        return HandlerResult(
            response_key="menu_not_found",
            next_state=ConversationState.IDLE,
        )
