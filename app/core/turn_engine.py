# app/core/turn_engine.py

from app.state_machine.state_router import StateRouter
from app.state_machine.context import ConversationContext
from app.state_machine.conversation_state import ConversationState
from app.nlu.intent import Intent


class TurnEngine:
    """
    Coordinates a single conversational turn.
    """

    def __init__(self, router: StateRouter) -> None:
        self.router = router

    def process_turn(
        self,
        state: ConversationState,
        intent: Intent,
        context: ConversationContext,
        handler,
    ):
        route_result = self.router.route(state, intent)

        if not route_result.allowed:
            return state, context, route_result.reason

        result = handler.handle(intent, context)

        return result.next_state, context, result.response_key
