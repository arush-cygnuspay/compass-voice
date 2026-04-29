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

    conversation_state: ConversationState = ConversationState.WAITING_FOR_CALLER_DEVICE_TYPE
    conversation_context: ConversationContext = field(default_factory=ConversationContext)

    cart: Cart = field(default_factory=Cart)

    turn_count: int = 0
    last_intent: Optional[Intent] = None
    last_response_key: Optional[str] = None
    last_response_payload: Optional[Dict[str, Any]] = None
    last_response_at_epoch: Optional[float] = None
    last_normalized_user_text: Optional[str] = None
    repeated_user_turn_count: int = 0
    fallback_count: int = 0
    reprompt_escalation_count: int = 0
    slot_extraction_failure_count: int = 0
    invalid_modifier_count: int = 0
    reprompt_count_by_field: Dict[str, int] = field(default_factory=dict)

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
            "last_response_at_epoch": self.last_response_at_epoch,
            "last_normalized_user_text": self.last_normalized_user_text,
            "repeated_user_turn_count": self.repeated_user_turn_count,
            "fallback_count": self.fallback_count,
            "reprompt_escalation_count": self.reprompt_escalation_count,
            "slot_extraction_failure_count": self.slot_extraction_failure_count,
            "invalid_modifier_count": self.invalid_modifier_count,
            "reprompt_count_by_field": dict(self.reprompt_count_by_field),
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
        session.last_response_at_epoch = data.get("last_response_at_epoch")
        session.last_normalized_user_text = data.get("last_normalized_user_text")
        session.repeated_user_turn_count = int(data.get("repeated_user_turn_count", 0) or 0)
        session.fallback_count = int(data.get("fallback_count", 0) or 0)
        session.reprompt_escalation_count = int(data.get("reprompt_escalation_count", 0) or 0)
        session.slot_extraction_failure_count = int(data.get("slot_extraction_failure_count", 0) or 0)
        session.invalid_modifier_count = int(data.get("invalid_modifier_count", 0) or 0)
        session.reprompt_count_by_field = dict(data.get("reprompt_count_by_field") or {})
        return session
