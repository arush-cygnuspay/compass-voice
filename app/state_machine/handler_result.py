# app/state_machine/handler_result.py

from dataclasses import dataclass
from typing import Optional

from app.state_machine.conversation_state import ConversationState


@dataclass
class HandlerResult:
    """
    Standard output of every conversation handler.
    """

    next_state: ConversationState
    response_key: str                 # used by response_builder
    end_turn: bool = True             # whether to wait for next user input
