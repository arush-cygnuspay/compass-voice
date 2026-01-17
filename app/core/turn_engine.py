# app/core/turn_engine.py
from app.nlu.intent_resolution.intent_resolver import resolve_intent
from app.session.session import Session
from app.state_machine.conversation_state import ConversationState
from app.state_machine.handlers.item.add_item.adding_item_handler import AddItemHandler
from app.state_machine.handlers.item.add_item.waiting_for_side_handler import WaitingForSideHandler
from app.state_machine.handlers.item.confirming_handler import ConfirmingHandler
from app.state_machine.state_router import StateRouter
from app.nlu.intent_resolution.intent import Intent
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
            "waiting_for_side_handler": WaitingForSideHandler(menu_repo),
            "confirming_handler": ConfirmingHandler(menu_repo),
            # add later:
            # "waiting_for_modifier_handler": ...
            # "waiting_for_size_handler": ...
        }

    def process_turn(
            self,
            session: Session,
            user_text: str,
    ) -> str:

        # 1️⃣ Pure NLU
        intent_result = resolve_intent(user_text)

        # 2️⃣ Route based on STATE + intent
        route = self.router.route(
            state=session.conversation_state,
            intent_result=intent_result,
        )

        if not route.allowed:
            return "intent_not_allowed"

        handler = self.handlers.get(route.handler_name)
        if not handler:
            return "handler_not_implemented"

        # 3️⃣ Execute handler
        result = handler.handle(
            intent=intent_result.intent,  # ← pass original intent
            context=session.conversation_context,
            user_text=user_text,
        )

        # 4️⃣ Persist FSM changes
        session.conversation_state = result.next_state
        session.last_intent = intent_result.intent
        session.last_response_key = result.response_key
        session.turn_count += 1

        return result.response_key

