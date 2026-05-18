# app/api/chat_demo.py
"""
Internal testing chat endpoint.
Used by the browser-based UI to talk to the same TurnEngine
that Twilio voice uses.

POST-only by design.
"""

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from app.config.restaurant import DEFAULT_RESTAURANT_ID
from app.session.repository import load_session, save_session
from app.core.turn_engine import TurnEngine
from app.core.response_builder import ResponseBuilder
from app.services.sms_service import DEFAULT_SMS_OVERRIDE_TO
from app.session.session import Session
from app.state_machine.models.conversation_state import ConversationState

router = APIRouter(prefix="/test", tags=["testing"])


CHAT_WAITING_STATES = {
    ConversationState.WAITING_FOR_PAYMENT,
    ConversationState.WAITING_FOR_CHECKOUT_COMPLETION,
}


# -------------------------
# Request / Response models
# -------------------------

class ChatRequest(BaseModel):
    """
    Payload sent from the browser UI.
    """
    session_id: str
    text: str


class ChatLink(BaseModel):
    kind: str
    label: str
    url: str


class ChatResponse(BaseModel):
    """
    Structured response returned to the UI.
    """
    response: str
    response_key: str
    state: str
    last_intent: str | None
    waiting_external: bool = False
    auto_check_recommended: bool = False
    order_type: str | None = None
    order_number: str | None = None
    sms_phone_number: str | None = None
    quick_replies: list[str] = Field(default_factory=list)
    quick_reply_mode: str = "single"   # "single" = click sends, "multi" = toggle + Done
    links: list[ChatLink] = Field(default_factory=list)


def _prepare_chat_session(session: Session) -> None:
    context = session.conversation_context

    context.caller_device_type = "chat"

    if not context.delivery_address.customer_phone_number:
        context.delivery_address.customer_phone_number = DEFAULT_SMS_OVERRIDE_TO


def _default_response_key_for_state(session: Session) -> str:
    state = session.conversation_state
    context = session.conversation_context

    if state == ConversationState.WAITING_FOR_ORDER_TYPE:
        return "ask_for_order_type"
    if state == ConversationState.WAITING_FOR_DELIVERY_ELIGIBILITY:
        if context.current_prompt_field == "delivery_postal_code":
            return "ask_for_delivery_zip"
        return "ask_for_delivery_area"
    if state == ConversationState.WAITING_FOR_DELIVERY_ADDRESS_COLLECTION:
        prompt_field = context.current_prompt_field
        if prompt_field == "delivery_street":
            return "ask_for_delivery_street"
        if prompt_field == "delivery_secondary_address":
            return "ask_for_delivery_secondary_address"
        return "ask_for_delivery_house_number"
    if state == ConversationState.WAITING_FOR_PAYMENT:
        return "waiting_for_payment"
    if state == ConversationState.WAITING_FOR_CHECKOUT_COMPLETION:
        return "waiting_for_checkout_completion"
    if state == ConversationState.CONFIRMING_ORDER:
        return "confirm_order_summary"
    if state == ConversationState.IDLE:
        return "resume_shopping"
    return session.last_response_key or "handler_not_implemented"


def _quick_replies_for_session(session: Session) -> tuple[list[str], str]:
    """Return (quick_reply_labels, mode).

    mode is "single" (click sends immediately) or "multi" (toggle + Done).
    """
    state = session.conversation_state
    context = session.conversation_context

    # ── Multi-select states: show ALL options + Done ──
    if context.available_choices_values and context.available_choices_kind in {
        "side", "modifier",
    }:
        choices = list(context.available_choices_values)
        choices.append("Done")
        choices.append("Skip")
        return choices, "multi"

    # ── Single-select states: show all options (no Done needed) ──
    if context.available_choices_values:
        return list(context.available_choices_values), "single"

    if state == ConversationState.CONFIRMING_ITEM:
        return ["Yes", "No", "Cancel"], "single"
    if state == ConversationState.CANCELLATION_CONFIRMATION:
        return ["Yes", "No"], "single"
    if state == ConversationState.WAITING_FOR_ORDER_TYPE:
        return ["Pickup", "Delivery"], "single"
    if state == ConversationState.CONFIRMING_ORDER:
        return ["Checkout", "Show cart", "Cancel order"], "single"
    if state == ConversationState.WAITING_FOR_CHECKOUT_COMPLETION:
        return ["I completed it", "Send the link again", "Check status"], "single"
    if state == ConversationState.WAITING_FOR_PAYMENT:
        return ["I completed payment", "Send the link again", "Check status"], "single"
    if state == ConversationState.IDLE and not session.cart.is_empty():
        return ["Show cart", "Checkout"], "single"
    return [], "single"


def _links_for_session(session: Session) -> list[ChatLink]:
    delivery = session.conversation_context.delivery_address
    links: list[ChatLink] = []

    if delivery.address_form_link:
        links.append(
            ChatLink(
                kind="checkout",
                label="Open secure checkout",
                url=delivery.address_form_link,
            )
        )

    if delivery.payment_link:
        links.append(
            ChatLink(
                kind="payment",
                label="Open payment link",
                url=delivery.payment_link,
            )
        )

    # Prefer the real checkout/payment link over the static
    # restaurant confirmation_link for order confirmation.
    confirmation_url = delivery.confirmation_link or delivery.payment_link
    if confirmation_url:
        links.append(
            ChatLink(
                kind="confirmation",
                label="Open order confirmation",
                url=confirmation_url,
            )
        )

    return links


def _build_chat_response(
    *,
    session: Session,
    responder: ResponseBuilder,
    response_key: str,
    response_payload: dict | None,
) -> ChatResponse:
    response_text = responder.build(
        response_key=response_key,
        context=session.conversation_context,
        payload=response_payload,
    )
    delivery = session.conversation_context.delivery_address
    quick_reply_labels, quick_reply_mode = _quick_replies_for_session(session)

    return ChatResponse(
        response=response_text,
        response_key=response_key,
        state=session.conversation_state.name,
        last_intent=session.last_intent.name if session.last_intent else None,
        waiting_external=session.conversation_state in CHAT_WAITING_STATES,
        auto_check_recommended=(
            session.conversation_state in CHAT_WAITING_STATES
            and bool(delivery.order_number)
        ),
        order_type=session.conversation_context.order_type,
        order_number=delivery.order_number,
        sms_phone_number=delivery.customer_phone_number,
        quick_replies=quick_reply_labels,
        quick_reply_mode=quick_reply_mode,
        links=_links_for_session(session),
    )


# -------------------------
# Test chat endpoint
# -------------------------

@router.post("/chat", response_model=ChatResponse)
def test_chat(req: ChatRequest, request: Request):
    """
    Handles a single chat turn from the browser UI.

    - Uses the same TurnEngine as Twilio
    - Session is keyed by session_id
    """

    # Pull shared engine + responder from app state
    engine: TurnEngine = request.app.state.engine
    responder: ResponseBuilder = request.app.state.responder

    # Load (or create) session
    session = load_session(req.session_id, restaurant_id=DEFAULT_RESTAURANT_ID)
    _prepare_chat_session(session)

    user_text = (req.text or "").strip()

    if not user_text:
        response_key = session.last_response_key or _default_response_key_for_state(session)
        response_payload = (
            session.last_response_payload
            if session.last_response_key == response_key
            else None
        )
        save_session(session)
        return _build_chat_response(
            session=session,
            responder=responder,
            response_key=response_key,
            response_payload=response_payload,
        )

    # Run core FSM pipeline
    turn_output = engine.process_turn(
        session=session,
        user_text=user_text,
    )

    # Persist session
    save_session(session)
    return _build_chat_response(
        session=session,
        responder=responder,
        response_key=turn_output.response_key,
        response_payload=turn_output.response_payload,
    )
