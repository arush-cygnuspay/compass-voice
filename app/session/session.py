# app/session/session.py

from dataclasses import dataclass, field
from typing import Optional

from app.state_machine.conversation_state import ConversationState
from app.state_machine.context import ConversationContext
from app.cart.cart import Cart
from app.nlu.intent_resolution.intent import Intent


@dataclass
class Session:
    """
    Represents a persisted conversational session.
    This object is the single source of truth across turns.
    """

    # Identity
    session_id: str
    restaurant_id: str

    # FSM
    conversation_state: ConversationState = ConversationState.IDLE
    conversation_context: ConversationContext = field(
        default_factory=ConversationContext
    )

    # Cart
    cart: Cart = field(default_factory=Cart)

    # Meta / Debug
    turn_count: int = 0
    last_intent: Optional[Intent] = None
    last_response_key: Optional[str] = None
