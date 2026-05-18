from app.cart.cart_item import CartItem
from app.nlu.intent_resolution.intent import Intent
from app.nlu.nlu_result import NLUResult
from app.session.session import Session
from app.state_machine.handlers.common.cancellation_confirmation_handler import (
    CancellationConfirmationHandler,
)
from app.state_machine.models.conversation_state import ConversationState


def _session() -> Session:
    session = Session(session_id="cancel-1", restaurant_id="steves_grill")
    session.conversation_state = ConversationState.CANCELLATION_CONFIRMATION
    session.cart.add_item(
        CartItem.create(
            item_id="burger",
            quantity=1,
            variant_id=None,
            sides={},
            side_variants={},
            modifiers={},
        )
    )
    return session


def _set_last_nlu(session: Session, intent: Intent, confidence: float = 0.2) -> None:
    session.conversation_context.set_last_nlu(
        user_text="",
        nlu=NLUResult(
            effective_intent=intent,
            intent_confidence=confidence,
            raw_text="",
            normalized_text="",
        ),
    )


def test_clear_cart_accepts_natural_confirmation_phrase() -> None:
    handler = CancellationConfirmationHandler()
    session = _session()
    session.conversation_context.return_state = ConversationState.IDLE
    session.conversation_context.awaiting_flow_confirmation = True
    session.conversation_context.awaiting_confirmation_for = {"type": "clear_cart"}
    _set_last_nlu(session, Intent.UNKNOWN)

    result = handler.handle(
        intent=Intent.UNKNOWN,
        context=session.conversation_context,
        user_text="yeah go ahead",
        session=session,
    )

    assert result.response_key == "cart_cleared"
    assert result.command == {"type": "CLEAR_CART"}


def test_clear_cart_denial_keeps_cart_without_yes_no_requirement() -> None:
    handler = CancellationConfirmationHandler()
    session = _session()
    session.conversation_context.return_state = ConversationState.IDLE
    session.conversation_context.awaiting_flow_confirmation = True
    session.conversation_context.awaiting_confirmation_for = {"type": "clear_cart"}
    _set_last_nlu(session, Intent.UNKNOWN)

    result = handler.handle(
        intent=Intent.UNKNOWN,
        context=session.conversation_context,
        user_text="no keep it",
        session=session,
    )

    assert result.response_key == "clear_cart_cancelled"
    assert result.next_state == ConversationState.IDLE
