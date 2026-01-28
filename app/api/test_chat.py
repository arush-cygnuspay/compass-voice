# app/api/test_chat.py
"""
Internal testing chat endpoint.
Used by the browser-based UI to talk to the same TurnEngine
that Twilio voice uses.

POST-only by design.
"""

from fastapi import APIRouter, Request
from pydantic import BaseModel

from app.session.repository import load_session, save_session
from app.core.turn_engine import TurnEngine
from app.core.response_builder import ResponseBuilder

router = APIRouter(prefix="/test", tags=["testing"])


# -------------------------
# Request / Response models
# -------------------------

class ChatRequest(BaseModel):
    """
    Payload sent from the browser UI.
    """
    session_id: str
    text: str


class ChatResponse(BaseModel):
    """
    Structured response returned to the UI.
    """
    response: str
    state: str
    last_intent: str | None


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
    session = load_session(req.session_id, restaurant_id="demo")

    # Run core FSM pipeline
    turn_output = engine.process_turn(
        session=session,
        user_text=req.text,
    )

    # Persist session
    save_session(session)

    # Build user-facing text
    response_text = responder.build(
        response_key=turn_output.response_key,
        context=session.conversation_context,
        payload=turn_output.response_payload,
    )

    return ChatResponse(
        response=response_text,
        state=session.conversation_state.name,
        last_intent=session.last_intent.name if session.last_intent else None,
    )
