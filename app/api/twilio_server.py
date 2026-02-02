# app/api/twilio_server.py
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Form
from fastapi.responses import Response

from twilio.twiml.voice_response import VoiceResponse, Gather

#  IMPORT ROUTERS
from app.api.ui.ui import router as ui_router
from app.api.test_chat import router as test_chat_router

from app.core.turn_engine import TurnEngine
from app.core.response_builder import ResponseBuilder
from app.menu.repository import MenuRepository
from app.menu.store import MenuStore
from app.menu.exceptions import MenuLoadError
from app.state_machine.state_router import StateRouter
from app.session.repository import load_session, save_session


# ----------------------------------------------------------
# Lifespan
# ----------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---------- INIT ----------
    restaurant_id = "demo"

    project_root = Path(__file__).resolve().parents[2]
    data_root = project_root / "app" / "data" / "restaurants" / restaurant_id

    try:
        store = MenuStore(
            data_root / "menu.json",
            data_root / "entity_index.json",
        )
    except MenuLoadError as e:
        raise RuntimeError(f"Failed to load menu: {e}")

    menu_repo = MenuRepository(store)
    router = StateRouter()

    engine = TurnEngine(router, menu_repo)
    responder = ResponseBuilder(menu_repo)

    # Attach to app state
    app.state.engine = engine
    app.state.responder = responder

    # ✅ REGISTER ROUTERS
    app.include_router(test_chat_router)  # /test/chat
    app.include_router(ui_router)         # /ui

    print("Twilio server initialized with Compass Voice v2 pipeline")

    yield

    # ---------- SHUTDOWN ----------
    print("Shutting down Compass Voice v2")


# ----------------------------------------------------------
# App
# ----------------------------------------------------------

app = FastAPI(lifespan=lifespan)


# ----------------------------------------------------------
# Helper
# ----------------------------------------------------------

def gather(action_url: str, say: str | None = None) -> Gather:
    g = Gather(
        input="speech",
        action=action_url,
        method="POST",
        timeout=15,
        speechTimeout="auto",
        bargeIn=True,
    )
    if say:
        g.say(say)
    return g


# ==========================================================
# ENTRY POINT — /voice
# ==========================================================

@app.post("/voice")
async def voice(request: Request):
    form = await request.form()
    call_sid = form.get("CallSid")
    restaurant_id = "demo"

    session = load_session(call_sid, restaurant_id)

    # 🔒 HARD GUARD
    if session.turn_count > 0:
        return Response("", media_type="application/xml")

    print(f"[CALL START] SID={call_sid}")

    vr = VoiceResponse()
    vr.append(
        gather(
            action_url=str(request.url_for("process_speech")),
            say="Thank you for calling Compass. What would you like to order?",
        )
    )

    return Response(str(vr), media_type="application/xml")


# ==========================================================
# PROCESS SPEECH — /process_speech
# ==========================================================

@app.post("/process_speech")
async def process_speech(
    request: Request,
    SpeechResult: str = Form(default=""),
):
    engine: TurnEngine = request.app.state.engine
    responder: ResponseBuilder = request.app.state.responder

    form = await request.form()
    call_sid = form.get("CallSid")
    restaurant_id = "demo"

    user_text = SpeechResult or ""

    print(f"[TWILIO SPEECH] SID={call_sid} TEXT={user_text}")

    session = load_session(call_sid, restaurant_id)

    vr = VoiceResponse()

    if not user_text.strip():
        vr.say("Sorry, I didn’t catch that. Could you repeat?")
        vr.append(gather(str(request.url_for("process_speech"))))
        return Response(str(vr), media_type="application/xml")

    # -------------------------
    # CORE V2 PIPELINE
    # -------------------------
    turn_output = engine.process_turn(
        session=session,
        user_text=user_text,
    )

    save_session(session)

    response_text = responder.build(
        response_key=turn_output.response_key,
        context=session.conversation_context,
        payload=turn_output.response_payload,
    )

    print(f"[BOT → CALLER] {response_text}")

    vr.say(response_text)
    vr.append(gather(str(request.url_for("process_speech"))))

    return Response(str(vr), media_type="application/xml")


# ==========================================================
# RUN SERVER (DEV)
# ==========================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.api.twilio_server:app",
        host="0.0.0.0",
        port=5000,
        reload=True,
    )
