# app/session/session.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from app.cart.cart import Cart
from app.nlu.intent_resolution.intent import Intent
from app.state_machine.models.conversation_context import ConversationContext
from app.state_machine.models.conversation_state import ConversationState


@dataclass(slots=True)
class Session:
    """
    Persisted conversational session.
    This is the single source of truth across turns.
    """

    session_id: str
    restaurant_id: str

    conversation_state: ConversationState = ConversationState.WAITING_FOR_ORDER_TYPE
    conversation_context: ConversationContext = field(default_factory=ConversationContext)

    cart: Cart = field(default_factory=Cart)

    turn_count: int = 0
    last_intent: Optional[Intent] = None
    last_response_key: Optional[str] = None
    last_response_payload: Optional[Dict[str, Any]] = None

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "restaurant_id": self.restaurant_id,
            "conversation_state": self.conversation_state.value,
            "conversation_context": self.conversation_context.to_dict(),
            "cart": self.cart.to_dict(),
            "turn_count": self.turn_count,
            "last_intent": self.last_intent.value if self.last_intent else None,
            "last_response_key": self.last_response_key,
            "last_response_payload": self.last_response_payload,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Session":
        session = cls(
            session_id=data["session_id"],
            restaurant_id=data["restaurant_id"],
            conversation_state=ConversationState(data["conversation_state"]),
        )
        session.conversation_context = ConversationContext.from_dict(
            data.get("conversation_context")
        )
        session.cart = Cart.from_dict(data["cart"])
        session.turn_count = data.get("turn_count", 0)

        last_intent = data.get("last_intent")
        session.last_intent = Intent(last_intent) if last_intent else None

        session.last_response_key = data.get("last_response_key")
        session.last_response_payload = data.get("last_response_payload")
        return session
