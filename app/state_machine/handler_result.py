# app/state_machine/handler_result.py
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional

from app.state_machine.models.conversation_state import ConversationState

if TYPE_CHECKING:
    from app.state_machine.models.pending_item_models import InterruptProposal


@dataclass(slots=True)
class HandlerResult:
    """
    Standard output of every conversation handler.

    Engine-applied fields (prompt_field, interrupt_proposal, awaiting_flow_confirmation)
    are written to ConversationContext by TurnEngine after the handler returns.  Handlers
    must NOT mutate those context fields directly; they should carry the values here so
    TurnEngine remains the single owner of context mutations.
    """

    next_state: ConversationState
    response_key: str
    end_turn: bool = True
    command: Optional[Dict[str, Any]] = None
    response_payload: Optional[Dict[str, Any]] = None
    reset_context: bool = False
    # Engine-applied state fields — set these instead of mutating context directly.
    # None means "leave the current context value unchanged".
    prompt_field: Optional[str] = None
    interrupt_proposal: Optional[InterruptProposal] = None
    awaiting_flow_confirmation: Optional[bool] = None