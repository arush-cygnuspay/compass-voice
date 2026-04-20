# app/state_machine/handlers/base_handler.py
from __future__ import annotations

from abc import ABC, abstractmethod

from app.nlu.intent_resolution.intent import Intent
from app.session.session import Session
from app.state_machine.handler_result import HandlerResult
from app.state_machine.models.conversation_context import ConversationContext


class BaseHandler(ABC):
    """
    Base class for all conversational state handlers.

    Each handler receives the resolved intent, the current conversation
    context (mutable), the raw user text, and optionally the full session.
    It returns a HandlerResult describing the next state, response key, and
    any commands to execute.
    """

    @abstractmethod
    def handle(
        self,
        intent: Intent,
        context: ConversationContext,
        user_text: str,
        session: Session | None = None,
    ) -> HandlerResult:
        pass
