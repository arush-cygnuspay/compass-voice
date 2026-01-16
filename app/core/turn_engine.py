# app/core/turn_engine.py

from app.session.session import Session
from app.state_machine.handlers.item.adding_item_handler import AddItemHandler
from app.state_machine.state_router import StateRouter
from app.nlu.intent import Intent
from app.menu.repository import MenuRepository


class TurnEngine:
    """
    Stateless turn processor.
    Session is the single source of truth.
    """

    def __init__(self, router: StateRouter, menu_repo: MenuRepository):
        self.router = router

        # Explicit handler registry
        self.handlers = {
            "idle_handler": AddItemHandler(menu_repo),
        }

    def process_turn(
        self,
        session: Session,
        intent: Intent,
        user_text: str,
    ) -> str:

        # Route based on SESSION state
        route = self.router.route(
            state=session.conversation_state,
            intent=intent,
        )

        if not route.allowed:
            return "intent_not_allowed"

        handler = self.handlers.get(route.handler_name)
        if not handler:
            return "handler_not_implemented"

        result = handler.handle(
            intent=intent,
            context=session.conversation_context,
            user_text=user_text,
        )

        # 🔑 FSM update lives on Session
        session.conversation_state = result.next_state
        session.last_intent = intent
        session.last_response_key = result.response_key
        session.turn_count += 1

        return result.response_key
