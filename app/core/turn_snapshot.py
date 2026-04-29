# app/core/turn_snapshot.py
"""Lightweight snapshot of the mutable session fields that TurnEngine owns.

Used to restore pre-turn state when an unhandled exception occurs mid-turn,
preventing partial state from persisting to the next caller turn.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from app.nlu.intent_resolution.intent import Intent
    from app.session.session import Session
    from app.state_machine.models.conversation_state import ConversationState


@dataclass(frozen=True, slots=True)
class TurnSnapshot:
    conversation_state: "ConversationState"
    turn_count: int
    last_response_key: Optional[str]
    last_response_payload: Optional[Dict[str, Any]]
    last_intent: Optional["Intent"]

    @classmethod
    def capture(cls, session: "Session") -> "TurnSnapshot":
        return cls(
            conversation_state=session.conversation_state,
            turn_count=session.turn_count,
            last_response_key=session.last_response_key,
            last_response_payload=session.last_response_payload,
            last_intent=session.last_intent,
        )

    def restore(self, session: "Session") -> None:
        session.conversation_state = self.conversation_state
        session.turn_count = self.turn_count
        session.last_response_key = self.last_response_key
        session.last_response_payload = self.last_response_payload
        session.last_intent = self.last_intent
