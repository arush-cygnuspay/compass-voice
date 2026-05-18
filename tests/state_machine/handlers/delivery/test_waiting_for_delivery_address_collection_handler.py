from app.cart.cart_item import CartItem
from app.nlu.intent_resolution.intent import Intent
from app.nlu.nlu_result import NLUResult
from app.session.session import Session
from app.state_machine.handlers.delivery.waiting_for_delivery_address_collection_handler import (
    WaitingForDeliveryAddressCollectionHandler,
)
from app.state_machine.models.conversation_context import ConversationContext
from app.state_machine.models.conversation_state import ConversationState


class _StubCartSummaryBuilder:
    def build(self, cart):
        return {"items": [{"name": "Burger", "quantity": 1}], "total": "$10.00"}


class _StubCheckoutService:
    def ensure_payment_link_for_voice_session(self, *args, **kwargs):
        raise NotImplementedError


def _session() -> Session:
    session = Session(session_id="delivery-1", restaurant_id="steves_grill")
    session.conversation_state = ConversationState.WAITING_FOR_DELIVERY_ADDRESS_COLLECTION
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


def _context() -> ConversationContext:
    context = ConversationContext()
    context.current_prompt_field = "delivery_seed_confirmation"
    context.delivery_address.area = "Downtown"
    context.delivery_address.postal_code = "12345"
    return context


def _set_last_nlu(context: ConversationContext, intent: Intent, confidence: float = 0.2) -> None:
    context.set_last_nlu(
        user_text="",
        nlu=NLUResult(
            effective_intent=intent,
            intent_confidence=confidence,
            raw_text="",
            normalized_text="",
        ),
    )


def test_delivery_seed_confirmation_accepts_natural_affirmation() -> None:
    handler = WaitingForDeliveryAddressCollectionHandler(
        cart_summary_builder=_StubCartSummaryBuilder(),
        checkout_service=_StubCheckoutService(),
    )
    session = _session()
    context = _context()
    _set_last_nlu(context, Intent.UNKNOWN)

    result = handler.handle(
        intent=Intent.UNKNOWN,
        context=context,
        user_text="yeah go ahead",
        session=session,
    )

    assert result.response_key == "ask_for_delivery_house_number"
    assert context.current_prompt_field == "delivery_house_number"


def test_delivery_seed_confirmation_cancel_returns_to_order_review() -> None:
    handler = WaitingForDeliveryAddressCollectionHandler(
        cart_summary_builder=_StubCartSummaryBuilder(),
        checkout_service=_StubCheckoutService(),
    )
    session = _session()
    context = _context()
    _set_last_nlu(context, Intent.UNKNOWN)

    result = handler.handle(
        intent=Intent.UNKNOWN,
        context=context,
        user_text="wait hold on",
        session=session,
    )

    assert result.response_key == "confirm_order_summary"
    assert result.next_state == ConversationState.CONFIRMING_ORDER
