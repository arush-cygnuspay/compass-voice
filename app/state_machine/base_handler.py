# app/state_machine/base_handler.py

from abc import ABC, abstractmethod

from app.session.session import Session
from app.state_machine.context import ConversationContext
from app.state_machine.handler_result import HandlerResult
from app.nlu.intent_resolution.intent import Intent


class BaseHandler(ABC):
    """
    Base class for all conversational handlers.
    """

    @abstractmethod
    def handle(
    self,
    session: Session,
    intent: Intent,
    user_text: str,
) -> HandlerResult:
        pass
