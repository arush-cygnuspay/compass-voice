# app/api/voice_stream_server.py
from __future__ import annotations

import asyncio
import base64
import json
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass

from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from twilio.twiml.voice_response import Connect, VoiceResponse

from app.api.chat_demo import router as test_chat_router
from app.api.ui.ui import router as ui_router
from app.bootstrap.runtime import build_runtime
from app.realtime.deepgram_stt_client import (
    DeepgramSTTCallbacks,
    DeepgramSTTClient,
    DeepgramSTTConfig,
)
from app.realtime.deepgram_tts_client import DeepgramTTSClient
from app.realtime.realtime_conversation_state import RealtimePhase
from app.realtime.turn_commit_controller import TurnCommitController
from app.session.repository import load_session, save_session
from app.session.session import Session

load_dotenv()


# Twilio telephony audio is μ-law 8k mono. 20 ms at 8 kHz = 160 samples = 160 bytes.
# Sending in 20 ms frames keeps playback responsive and avoids large buffered bursts.
TWILIO_MULAW_FRAME_BYTES = 160
TWILIO_SEND_PACING_SECONDS = 0.02


@dataclass(slots=True)
class StreamSession:
    call_sid: str | None = None
    stream_sid: str | None = None
    account_sid: str | None = None
    tracks: list[str] | None = None
    media_format: dict[str, object] | None = None
    inbound_chunks: int = 0
    inbound_bytes_b64: int = 0
    inbound_audio_bytes: int = 0
    outbound_audio_bytes: int = 0
    outbound_media_messages: int = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    runtime = build_runtime(restaurant_id="demo")

    app.state.runtime = runtime
    app.state.engine = runtime.engine
    app.state.responder = runtime.responder

    print("Voice stream server initialized with Twilio Media Streams + Deepgram STT/TTS")
    yield
    print("Shutting down voice stream server")


app = FastAPI(lifespan=lifespan)

app.include_router(test_chat_router)
app.include_router(ui_router)


def _build_stream_url(request: Request) -> str:
    host = request.headers.get("host", "").strip()
    if not host:
        raise ValueError("Missing Host header; cannot build stream URL.")

    explicit_public_base = os.getenv("PUBLIC_WSS_BASE_URL", "").strip()
    if explicit_public_base:
        return f"{explicit_public_base.rstrip('/')}/ws/twilio-media"

    return f"wss://{host}/ws/twilio-media"


@app.post("/voice")
async def voice(request: Request):
    form = await request.form()
    call_sid = form.get("CallSid", "")
    from_number = form.get("From", "")
    to_number = form.get("To", "")

    stream_url = _build_stream_url(request)

    print(
        "[CALL START]",
        {
            "call_sid": call_sid,
            "from": from_number,
            "to": to_number,
            "stream_url": stream_url,
        },
    )

    vr = VoiceResponse()
    vr.say("Thank you for calling Compass. What would you like to order today?")

    connect = Connect()
    connect.stream(
        url=stream_url,
        name="compass-voice-stream",
    )
    vr.append(connect)

    return Response(content=str(vr), media_type="application/xml")


@app.websocket("/ws/twilio-media")
async def twilio_media_ws(websocket: WebSocket):
    await websocket.accept()

    stream_session = StreamSession()
    dg_stt_client: DeepgramSTTClient | None = None
    dg_stt_started = False
    dg_tts_client = DeepgramTTSClient()

    restaurant_id = "demo"
    app_session: Session | None = None

    controller = TurnCommitController()
    phase = RealtimePhase.LISTENING
    pending_interrupt_text: str | None = None
    processing_lock = asyncio.Lock()

    active_mark_name: str | None = None
    mark_counter = 0

    bot_playback_started_at: float | None = None
    bot_barge_in_guard_seconds = 1.0
    disable_barge_in = True

    async def send_twilio_media(audio_bytes: bytes) -> None:
        if not audio_bytes or not stream_session.stream_sid:
            return

        payload_b64 = base64.b64encode(audio_bytes).decode("ascii")
        message = {
            "event": "media",
            "streamSid": stream_session.stream_sid,
            "media": {
                "payload": payload_b64,
            },
        }
        await websocket.send_text(json.dumps(message))
        stream_session.outbound_audio_bytes += len(audio_bytes)
        stream_session.outbound_media_messages += 1

    async def send_twilio_mark(name: str) -> None:
        if not stream_session.stream_sid:
            return

        message = {
            "event": "mark",
            "streamSid": stream_session.stream_sid,
            "mark": {"name": name},
        }
        await websocket.send_text(json.dumps(message))

    async def send_twilio_clear() -> None:
        if not stream_session.stream_sid:
            return

        message = {
            "event": "clear",
            "streamSid": stream_session.stream_sid,
        }
        await websocket.send_text(json.dumps(message))

    async def stream_audio_to_twilio(audio_chunk_stream) -> int:
        """
        Streams Deepgram TTS output to Twilio as 20 ms μ-law frames.

        This starts playback as soon as the first audio bytes arrive instead of waiting
        for the full utterance to be synthesized.
        """
        buffered = bytearray()
        total_bytes_sent = 0

        async for tts_chunk in audio_chunk_stream:
            if not tts_chunk:
                continue

            buffered.extend(tts_chunk)

            while len(buffered) >= TWILIO_MULAW_FRAME_BYTES:
                frame = bytes(buffered[:TWILIO_MULAW_FRAME_BYTES])
                del buffered[:TWILIO_MULAW_FRAME_BYTES]

                await send_twilio_media(frame)
                total_bytes_sent += len(frame)

                # Light pacing prevents dumping a large burst into Twilio all at once.
                await asyncio.sleep(TWILIO_SEND_PACING_SECONDS)

        if buffered:
            # Pad final partial frame with silence (μ-law silence is 0xFF).
            padded = bytes(buffered) + (b"\xFF" * (TWILIO_MULAW_FRAME_BYTES - len(buffered)))
            await send_twilio_media(padded)
            total_bytes_sent += len(padded)

        return total_bytes_sent

    async def speak_response_text(response_text: str) -> None:
        nonlocal phase, active_mark_name, mark_counter, bot_playback_started_at

        cleaned = " ".join((response_text or "").split()).strip()
        if not cleaned:
            phase = RealtimePhase.LISTENING
            return

        phase = RealtimePhase.SPEAKING
        bot_playback_started_at = time.monotonic()

        streamed_bytes = await stream_audio_to_twilio(
            dg_tts_client.stream_mulaw_8k(cleaned)
        )

        if streamed_bytes <= 0:
            print(
                "[TTS EMPTY AUDIO]",
                {
                    "stream_sid": stream_session.stream_sid,
                    "text": cleaned,
                },
            )
            phase = RealtimePhase.LISTENING
            bot_playback_started_at = None
            return

        mark_counter += 1
        active_mark_name = f"bot-playback-{mark_counter}"
        await send_twilio_mark(active_mark_name)

        print(
            "[TWILIO OUTBOUND AUDIO SENT]",
            {
                "stream_sid": stream_session.stream_sid,
                "bytes": streamed_bytes,
                "mark_name": active_mark_name,
                "barge_in_disabled": disable_barge_in,
            },
        )

    async def process_committed_turn(user_text: str) -> None:
        nonlocal phase, pending_interrupt_text, app_session

        if app_session is None:
            return

        cleaned = " ".join(user_text.split()).strip()
        if not cleaned:
            return

        if phase == RealtimePhase.PROCESSING:
            pending_interrupt_text = cleaned
            print(
                "[INTERRUPT BUFFERED]",
                {
                    "stream_sid": stream_session.stream_sid,
                    "text": cleaned,
                },
            )
            return

        async with processing_lock:
            phase = RealtimePhase.PROCESSING

            print(
                "[COMMITTED USER TURN]",
                {
                    "stream_sid": stream_session.stream_sid,
                    "text": cleaned,
                },
            )

            turn_output = app.state.engine.process_turn(
                session=app_session,
                user_text=cleaned,
            )

            save_session(app_session)

            response_text = app.state.responder.build(
                response_key=turn_output.response_key,
                context=app_session.conversation_context,
                payload=turn_output.response_payload,
            )

            print(
                "[BOT RESPONSE TEXT]",
                {
                    "stream_sid": stream_session.stream_sid,
                    "text": response_text,
                },
            )

            await speak_response_text(response_text)

            # Do not process buffered interrupt here unless playback is already complete.
            # Normal flow is: wait for Twilio mark -> phase becomes LISTENING -> replay buffered text.
            if pending_interrupt_text and phase == RealtimePhase.LISTENING:
                buffered = pending_interrupt_text
                pending_interrupt_text = None
                await process_committed_turn(buffered)

    async def on_dg_transcript(transcript: str, is_final: bool, payload: dict) -> None:
        controller.on_transcript(transcript, is_final)

        print(
            "[DEEPGRAM TRANSCRIPT]",
            {
                "stream_sid": stream_session.stream_sid,
                "type": "FINAL" if is_final else "INTERIM",
                "text": transcript,
            },
        )

        speech_final = bool(payload.get("speech_final", False))
        if speech_final:
            committed = controller.on_speech_final()
            if committed is not None:
                await process_committed_turn(committed.text)

    async def on_dg_event(name: str, payload: dict) -> None:
        nonlocal phase, active_mark_name, bot_playback_started_at

        print(
            "[DEEPGRAM EVENT]",
            {
                "stream_sid": stream_session.stream_sid,
                "event": name,
                "payload": payload,
            },
        )

        if name == "message:SpeechStarted":
            controller.on_speech_started()

            if phase == RealtimePhase.SPEAKING:
                if disable_barge_in:
                    print(
                        "[BARGE_IN IGNORED]",
                        {
                            "stream_sid": stream_session.stream_sid,
                            "reason": "disabled_for_playback_verification",
                        },
                    )
                    return

                now = time.monotonic()
                if (
                    bot_playback_started_at is not None
                    and now - bot_playback_started_at < bot_barge_in_guard_seconds
                ):
                    print(
                        "[BARGE_IN IGNORED]",
                        {
                            "stream_sid": stream_session.stream_sid,
                            "reason": "guard_window",
                            "elapsed_ms": round((now - bot_playback_started_at) * 1000.0, 1),
                        },
                    )
                    return

                print(
                    "[BARGE_IN]",
                    {
                        "stream_sid": stream_session.stream_sid,
                        "active_mark_name": active_mark_name,
                    },
                )
                await send_twilio_clear()
                active_mark_name = None
                bot_playback_started_at = None
                phase = RealtimePhase.LISTENING

            elif phase == RealtimePhase.PROCESSING:
                print(
                    "[USER_SPEECH_DURING_PROCESSING]",
                    {
                        "stream_sid": stream_session.stream_sid,
                    },
                )

        elif name == "message:UtteranceEnd":
            committed = controller.on_utterance_end()
            if committed is not None:
                await process_committed_turn(committed.text)

    async def on_dg_error(name: str, payload: dict) -> None:
        print(
            "[DEEPGRAM ERROR]",
            {
                "stream_sid": stream_session.stream_sid,
                "event": name,
                "payload": payload,
            },
        )

    print("[WS OPEN] Twilio media WebSocket connected")

    try:
        while True:
            raw_message = await websocket.receive_json()
            event = raw_message.get("event")

            if event == "connected":
                print("[TWILIO WS CONNECTED]", raw_message)

            elif event == "start":
                start = raw_message.get("start", {})
                stream_session.call_sid = start.get("callSid")
                stream_session.stream_sid = start.get("streamSid")
                stream_session.account_sid = start.get("accountSid")
                stream_session.tracks = start.get("tracks")
                stream_session.media_format = start.get("mediaFormat")

                print(
                    "[TWILIO STREAM START]",
                    {
                        "call_sid": stream_session.call_sid,
                        "stream_sid": stream_session.stream_sid,
                        "tracks": stream_session.tracks,
                        "media_format": stream_session.media_format,
                    },
                )

                if stream_session.call_sid:
                    app_session = load_session(stream_session.call_sid, restaurant_id)

                dg_stt_client = DeepgramSTTClient(
                    config=DeepgramSTTConfig(
                        model="nova-3",
                        language="en-US",
                        encoding="mulaw",
                        sample_rate=8000,
                        channels=1,
                        interim_results=True,
                        smart_format=True,
                        punctuate=True,
                        vad_events=True,
                        endpointing=300,
                        utterance_end_ms=1000,
                        keepalive_interval_seconds=4.0,
                    ),
                    callbacks=DeepgramSTTCallbacks(
                        on_transcript=on_dg_transcript,
                        on_event=on_dg_event,
                        on_error=on_dg_error,
                    ),
                )

                await dg_stt_client.start()
                dg_stt_started = True

                print(
                    "[DEEPGRAM CONNECTED]",
                    {
                        "stream_sid": stream_session.stream_sid,
                        "url": dg_stt_client.config.websocket_url(),
                    },
                )

            elif event == "media":
                media = raw_message.get("media", {})
                payload_b64 = media.get("payload", "")

                stream_session.inbound_chunks += 1
                stream_session.inbound_bytes_b64 += len(payload_b64)

                if not payload_b64:
                    continue

                audio_bytes = base64.b64decode(payload_b64)
                stream_session.inbound_audio_bytes += len(audio_bytes)

                if dg_stt_started and dg_stt_client is not None:
                    await dg_stt_client.send_audio(audio_bytes)

                if stream_session.inbound_chunks % 50 == 0:
                    print(
                        "[TWILIO MEDIA PROGRESS]",
                        {
                            "stream_sid": stream_session.stream_sid,
                            "chunks": stream_session.inbound_chunks,
                            "payload_b64_chars": stream_session.inbound_bytes_b64,
                            "audio_bytes": stream_session.inbound_audio_bytes,
                            "track": media.get("track"),
                            "chunk": media.get("chunk"),
                            "timestamp": media.get("timestamp"),
                        },
                    )

            elif event == "mark":
                mark = raw_message.get("mark", {})
                mark_name = mark.get("name")

                print(
                    "[TWILIO MARK RECEIVED]",
                    {
                        "stream_sid": stream_session.stream_sid,
                        "mark_name": mark_name,
                    },
                )

                if active_mark_name and mark_name == active_mark_name:
                    active_mark_name = None
                    bot_playback_started_at = None

                    if phase == RealtimePhase.SPEAKING:
                        phase = RealtimePhase.LISTENING

                    if pending_interrupt_text:
                        buffered = pending_interrupt_text
                        pending_interrupt_text = None
                        await process_committed_turn(buffered)

            elif event == "stop":
                stop = raw_message.get("stop", {})
                print(
                    "[TWILIO STREAM STOP]",
                    {
                        "call_sid": stream_session.call_sid,
                        "stream_sid": stream_session.stream_sid,
                        "reason": stop,
                        "chunks_received": stream_session.inbound_chunks,
                        "payload_b64_chars": stream_session.inbound_bytes_b64,
                        "audio_bytes": stream_session.inbound_audio_bytes,
                        "outbound_audio_bytes": stream_session.outbound_audio_bytes,
                        "outbound_media_messages": stream_session.outbound_media_messages,
                    },
                )

                if dg_stt_client is not None:
                    await dg_stt_client.finalize()

                break

            else:
                print("[TWILIO UNKNOWN EVENT]", raw_message)

    except WebSocketDisconnect:
        print(
            "[WS CLOSED]",
            {
                "call_sid": stream_session.call_sid,
                "stream_sid": stream_session.stream_sid,
                "chunks_received": stream_session.inbound_chunks,
            },
        )
    except Exception as exc:
        print(
            "[WS ERROR]",
            {
                "call_sid": stream_session.call_sid,
                "stream_sid": stream_session.stream_sid,
                "error": str(exc),
            },
        )
        try:
            await websocket.close()
        except Exception:
            pass
    finally:
        if dg_stt_client is not None:
            await dg_stt_client.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.api.voice_stream_server:app",
        host="0.0.0.0",
        port=5000,
        reload=False,
    )