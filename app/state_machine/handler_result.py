# app/state_machine/handler_result.py
from dataclasses import dataclass
from typing import Any, Dict, Optional

from app.state_machine.models.conversation_state import ConversationState


@dataclass(slots=True)
class HandlerResult:
    """
    Standard output of every conversation handler.
    """

    next_state: ConversationState
    response_key: str
    end_turn: bool = True
    command: Optional[Dict[str, Any]] = None
    response_payload: Optional[Dict[str, Any]] = None
    reset_context: bool = False