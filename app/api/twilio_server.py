# app/api/twilio_server.py
from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Form, Request
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from twilio.twiml.voice_response import Gather, VoiceResponse

from app.api.chat_demo import router as test_chat_router
from app.api.checkout_routes import router as checkout_api_router, page_router as checkout_page_router
from app.api.payment_links_webhook import router as payment_links_webhook_router
from app.api.ui.ui import router as ui_router
from app.bootstrap.runtime import build_runtime
from app.config.restaurant import DEFAULT_RESTAURANT_ID
from app.core.response_builder import ResponseBuilder
from app.core.turn_engine import TurnEngine
from app.session.repository import load_session, save_session


@asynccontextmanager
async def lifespan(app: FastAPI):
    runtime = build_runtime(restaurant_id=DEFAULT_RESTAURANT_ID)

    app.state.runtime = runtime
    app.state.engine = runtime.engine
    app.state.responder = runtime.responder

    print("Twilio server initialized with Compass Voice v2 pipeline")
    yield
    print("Shutting down Compass Voice v2")


app = FastAPI(lifespan=lifespan)

app.mount("/static", StaticFiles(directory="app/static", check_dir=False), name="static")
app.include_router(test_chat_router)
app.include_router(ui_router)
app.include_router(checkout_api_router)
app.include_router(checkout_page_router)
app.include_router(payment_links_webhook_router)


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


@app.post("/voice")
async def voice(request: Request):
    form = await request.form()
    call_sid = form.get("CallSid")
    restaurant_id = DEFAULT_RESTAURANT_ID

    session = load_session(call_sid, restaurant_id)

    if session.turn_count > 0:
        return Response("", media_type="application/xml")

    print(f"[CALL START] SID={call_sid}")

    responder: ResponseBuilder = request.app.state.responder
    vr = VoiceResponse()
    vr.append(
        gather(
            action_url=str(request.url_for("process_speech")),
            say=responder.build("ask_for_order_type", None, None),
        )
    )

    return Response(str(vr), media_type="application/xml")


@app.post("/process_speech")
async def process_speech(
    request: Request,
    SpeechResult: str = Form(default=""),
):
    engine: TurnEngine = request.app.state.engine
    responder: ResponseBuilder = request.app.state.responder

    form = await request.form()
    call_sid = form.get("CallSid")
    restaurant_id = DEFAULT_RESTAURANT_ID

    # Empty / whitespace-only STT result: still route through TurnEngine
    # so that the same NoInputEscalationPolicy applies as for unintelligible
    # utterances. Never branch on empty text in the transport layer - that
    # historically caused infinite "Sorry, I didn't catch that" loops because
    # the global UNKNOWN counter was bypassed.
    user_text = (SpeechResult or "").strip()

    print(f"[TWILIO SPEECH] SID={call_sid} TEXT={user_text!r}")

    session = load_session(call_sid, restaurant_id)

    vr = VoiceResponse()

    turn_start = time.perf_counter()

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

    turn_elapsed_seconds = time.perf_counter() - turn_start
    turn_elapsed_milliseconds = turn_elapsed_seconds * 1000.0

    print(
        "[TURN_RESPONSE_TIME]",
        {
            "sid": call_sid,
            "seconds": round(turn_elapsed_seconds, 3),
            "milliseconds": round(turn_elapsed_milliseconds, 3),
        },
    )

    print(f"[BOT → CALLER] {response_text}")

    vr.say(response_text)

    transfer_number = getattr(turn_output, "transfer_call_to_number", None)
    if transfer_number:
        vr.dial(transfer_number)
        return Response(str(vr), media_type="application/xml")

    if getattr(turn_output, "end_call_after_playback", False):
        vr.hangup()
        return Response(str(vr), media_type="application/xml")

    vr.append(gather(str(request.url_for("process_speech"))))

    return Response(str(vr), media_type="application/xml")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.api.twilio_server:app",
        host="0.0.0.0",
        port=5000,
        reload=False,
    )
