# app/api/voice_stream_server.py
from __future__ import annotations

import asyncio
import audioop
import base64
import json
import os
import re
import time
import wave
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Callable

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from twilio.rest import Client
from twilio.twiml.voice_response import Connect, VoiceResponse

from app.api.chat_demo import router as test_chat_router
from app.api.ui.ui import router as ui_router
from app.api.checkout_routes import router as checkout_api_router, page_router as checkout_page_router
from app.api.payment_links_webhook import router as payment_links_webhook_router
from app.bootstrap.runtime import build_runtime
from app.logging.realtime_latency_logger import (
    RealtimeLatencyLogger,
    RealtimeTurnTrace,
    utc_now_iso,
)
from app.realtime.deepgram_stt_client import (
    DeepgramSTTCallbacks,
    DeepgramSTTClient,
    DeepgramSTTConfig,
)
from app.realtime.deepgram_tts_client import DeepgramTTSClient
from app.realtime.realtime_conversation_state import RealtimePhase
from app.realtime.turn_commit_controller import TurnCommitController
from app.session.repository import load_existing_session, load_session, save_session
from app.session.session import Session
from app.state_machine.handlers.system.waiting_for_caller_device_type_handler import (
    HUMAN_AGENT_TRANSFER_NUMBER,
)
from app.state_machine.models.conversation_state import ConversationState

TWILIO_MULAW_FRAME_BYTES = 160
TWILIO_FRAME_DURATION_SECONDS = 0.02

TWILIO_BURST_FRAMES = int(float(os.getenv("TWILIO_BURST_FRAMES", "5")))
if TWILIO_BURST_FRAMES <= 0:
    TWILIO_BURST_FRAMES = 5

TWILIO_BURST_BYTES = int(TWILIO_MULAW_FRAME_BYTES * TWILIO_BURST_FRAMES)
TWILIO_BURST_PACING_SECONDS = TWILIO_FRAME_DURATION_SECONDS * TWILIO_BURST_FRAMES

WELCOME_AUDIO_WAV_PATH = os.getenv(
    "COMPASS_WELCOME_AUDIO_WAV_PATH",
    "app/static/audio/compass_welcome_8k_0.9x_order_type.wav",
).strip()

VOICE_DEBUG_ENABLED = os.getenv("COMPASS_VOICE_DEBUG_ENABLED", "0") == "1"
VOICE_TRANSCRIPT_DEBUG_ENABLED = os.getenv(
    "COMPASS_VOICE_TRANSCRIPT_DEBUG_ENABLED",
    "0",
) == "1"
VOICE_MEDIA_PROGRESS_DEBUG_ENABLED = os.getenv(
    "COMPASS_VOICE_MEDIA_PROGRESS_DEBUG_ENABLED",
    "0",
) == "1"

DYNAMIC_RESPONSE_KEYS = {
    "ask_for_side",
    "ask_for_modifier",
    "ask_for_size",
    "ask_for_quantity",
    "ask_for_side_size",
    "ask_for_caller_device_type",
    "repeat_caller_device_type",
    "confirm_landline_pickup_only",
    "repeat_landline_pickup_only",
    "transferring_to_human_agent",
    "ask_for_order_type",
    "repeat_order_type",
    "order_type_captured_pickup",
    "order_type_captured_delivery",
    "confirm_order_summary",
}

# Response keys emitted right after sending a checkout/payment link.
# Any of these arms the auto-confirm timer so the agent silently probes
# payment status ~50 s later without needing a user prompt.
PAYMENT_LINK_SENT_KEYS: frozenset[str] = frozenset({
    "ask_delivery_address_method",
    "checkout_link_sent",
    "payment_link_sent",
})

# States where an automatic payment-status probe is meaningful.
_PAYMENT_AWAITING_STATES: frozenset[ConversationState] = frozenset({
    ConversationState.WAITING_FOR_PAYMENT,
    ConversationState.WAITING_FOR_CHECKOUT_COMPLETION,
})

# Seconds to wait before the first (and each repeated) auto-confirm probe.
# Mid-point of the requested 45-60 s window.
PAYMENT_AUTO_CHECK_DELAY_SECONDS: int = 50

_WELCOME_AUDIO_BYTES_CACHE: bytes | None = None


def _debug_log(event: str, payload: dict[str, Any]) -> None:
    if not VOICE_DEBUG_ENABLED:
        return
    print(event, payload)


def _debug_transcript_log(event: str, payload: dict[str, Any]) -> None:
    if not VOICE_TRANSCRIPT_DEBUG_ENABLED:
        return
    print(event, payload)


def _debug_media_log(event: str, payload: dict[str, Any]) -> None:
    if not VOICE_MEDIA_PROGRESS_DEBUG_ENABLED:
        return
    print(event, payload)


def _normalize_response_text(text: str | None) -> str:
    return " ".join((text or "").split()).strip()


def _split_for_progressive_tts(text: str) -> list[str]:
    cleaned = _normalize_response_text(text)
    if not cleaned:
        return []

    sentence_parts = re.split(r"(?<=[.!?])\s+", cleaned)
    sentence_parts = [part.strip() for part in sentence_parts if part.strip()]
    if len(sentence_parts) > 1:
        return sentence_parts

    clause_parts = re.split(
        r"(?<=,)\s+|\s+(?=but\s|and\s|then\s|so\s)",
        cleaned,
        flags=re.IGNORECASE,
    )
    clause_parts = [part.strip() for part in clause_parts if part.strip()]
    if len(clause_parts) > 1:
        return clause_parts

    return [cleaned]


def _load_welcome_audio_as_mulaw_8k(path: str) -> bytes:
    with wave.open(path, "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        frame_count = wf.getnframes()
        pcm_bytes = wf.readframes(frame_count)

    if channels not in {1, 2}:
        raise ValueError(f"Unsupported welcome WAV channel count: {channels}")

    if sample_width not in {1, 2}:
        raise ValueError(f"Unsupported welcome WAV sample width: {sample_width}")

    if channels == 2:
        pcm_bytes = audioop.tomono(pcm_bytes, sample_width, 0.5, 0.5)

    if sample_rate != 8000:
        pcm_bytes, _ = audioop.ratecv(
            pcm_bytes,
            sample_width,
            1,
            sample_rate,
            8000,
            None,
        )

    return audioop.lin2ulaw(pcm_bytes, sample_width)


def _get_welcome_audio_bytes() -> bytes:
    global _WELCOME_AUDIO_BYTES_CACHE

    if _WELCOME_AUDIO_BYTES_CACHE is None:
        _WELCOME_AUDIO_BYTES_CACHE = _load_welcome_audio_as_mulaw_8k(
            WELCOME_AUDIO_WAV_PATH
        )

    return _WELCOME_AUDIO_BYTES_CACHE


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

    turn_index: int = 0
    current_utterance_first_media_monotonic: float | None = None
    current_utterance_last_media_monotonic: float | None = None
    current_utterance_inbound_audio_bytes: int = 0

    last_dg_speech_started_monotonic: float | None = None
    last_dg_final_transcript_monotonic: float | None = None

    active_trace: RealtimeTurnTrace | None = None
    active_trace_mark_name: str | None = None


def _safe_session_id(session: Session | None) -> str:
    if session is None:
        return ""

    value = getattr(session, "session_id", None)
    if value is not None:
        return str(value)

    value = getattr(session, "id", None)
    if value is not None:
        return str(value)

    return ""


def _trace_set_attr(
    trace: RealtimeTurnTrace | None,
    attr_name: str,
    value: Any,
) -> None:
    if trace is None:
        return
    try:
        setattr(trace, attr_name, value)
    except Exception:
        return


def _trace_add_note(trace: RealtimeTurnTrace | None, key: str, value: Any) -> None:
    if trace is None:
        return
    try:
        notes = getattr(trace, "notes", None)
        if isinstance(notes, dict):
            notes[key] = value
    except Exception:
        return


def _reset_utterance_tracking(stream_session: StreamSession) -> None:
    stream_session.current_utterance_first_media_monotonic = None
    stream_session.current_utterance_last_media_monotonic = None
    stream_session.current_utterance_inbound_audio_bytes = 0
    stream_session.last_dg_speech_started_monotonic = None
    stream_session.last_dg_final_transcript_monotonic = None


def _build_turn_trace(
    *,
    stream_session: StreamSession,
    app_session: Session,
    user_text: str,
) -> RealtimeTurnTrace:
    return RealtimeTurnTrace(
        call_sid=stream_session.call_sid or "",
        stream_sid=stream_session.stream_sid or "",
        session_id=_safe_session_id(app_session),
        turn_index=stream_session.turn_index,
        turn_started_at_utc=utc_now_iso(),
        turn_committed_at_utc=utc_now_iso(),
        first_inbound_media_monotonic=stream_session.current_utterance_first_media_monotonic,
        last_inbound_media_monotonic=stream_session.current_utterance_last_media_monotonic,
        dg_speech_started_monotonic=stream_session.last_dg_speech_started_monotonic,
        dg_final_transcript_monotonic=stream_session.last_dg_final_transcript_monotonic,
        turn_commit_monotonic=time.perf_counter(),
        user_text=user_text,
        inbound_audio_bytes=stream_session.current_utterance_inbound_audio_bytes,
    )


def _finalize_active_trace(
    *,
    app: FastAPI,
    stream_session: StreamSession,
    extra_notes: dict[str, Any] | None = None,
    reset_utterance_tracking: bool = True,
) -> None:
    trace = stream_session.active_trace
    if trace is None:
        return

    if extra_notes:
        for key, value in extra_notes.items():
            _trace_add_note(trace, key, value)

    try:
        app.state.realtime_latency_logger.write(trace)
    except Exception as exc:
        print(
            "[REALTIME_LATENCY_LOGGER_ERROR]",
            {
                "stream_sid": stream_session.stream_sid,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
    finally:
        stream_session.active_trace = None
        stream_session.active_trace_mark_name = None
        if reset_utterance_tracking:
            _reset_utterance_tracking(stream_session)


@asynccontextmanager
async def lifespan(app: FastAPI):
    runtime = build_runtime(restaurant_id="demo")

    app.state.runtime = runtime
    app.state.engine = runtime.engine
    app.state.responder = runtime.responder
    app.state.realtime_latency_logger = RealtimeLatencyLogger(
        file_path=os.getenv(
            "COMPASS_REALTIME_LATENCY_LOG_PATH",
            "app/logs/realtime_turn_latency.jsonl",
        ),
        csv_file_path=os.getenv(
            "COMPASS_REALTIME_LATENCY_CSV_PATH",
            "app/logs/realtime_turn_latency.csv",
        ),
        enabled=os.getenv("COMPASS_REALTIME_LATENCY_ENABLED", "1") == "1",
        write_csv=os.getenv("COMPASS_REALTIME_LATENCY_WRITE_CSV", "1") == "1",
    )

    print("Voice stream server initialized with Twilio Media Streams + Deepgram STT/TTS")
    try:
        yield
    finally:
        try:
            app.state.realtime_latency_logger.shutdown()
        except Exception:
            pass

        try:
            runtime.engine.nlu_logger.shutdown()
        except Exception:
            pass

        print("Shutting down voice stream server")


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="app/static", check_dir=False), name="static")

app.include_router(test_chat_router)
app.include_router(ui_router)
app.include_router(checkout_api_router)
app.include_router(checkout_page_router)
app.include_router(payment_links_webhook_router)


def _build_stream_url(request: Request) -> str:
    host = request.headers.get("host", "").strip()
    if not host:
        raise ValueError("Missing Host header; cannot build stream URL.")

    explicit_public_base = os.getenv("PUBLIC_WSS_BASE_URL", "").strip()
    if explicit_public_base:
        base = explicit_public_base.rstrip("/")
        if base.startswith("https://"):
            base = "wss://" + base[len("https://") :]
        elif base.startswith("http://"):
            base = "ws://" + base[len("http://") :]
        elif not base.startswith(("wss://", "ws://")):
            raise ValueError(
                "PUBLIC_WSS_BASE_URL must start with wss://, ws://, https://, or http://"
            )
        return f"{base}/ws/twilio-media"

    forwarded_proto = request.headers.get("x-forwarded-proto", "").strip().lower()
    if forwarded_proto == "https":
        return f"wss://{host}/ws/twilio-media"
    if forwarded_proto == "http":
        return f"ws://{host}/ws/twilio-media"

    return f"wss://{host}/ws/twilio-media"


def _build_action_url(request: Request) -> str:
    """Return the HTTPS URL Twilio should POST to when the stream ends.

    Mirrors the logic in ``_build_stream_url`` but produces an HTTP(S) URL
    instead of a WebSocket URL.  Twilio calls this endpoint after the
    ``<Connect><Stream>`` block exits so we can return ``<Dial>`` TwiML
    for landline-to-human-agent transfers.
    """
    host = request.headers.get("host", "localhost")
    explicit_public_base = os.getenv("PUBLIC_WSS_BASE_URL", "").strip()
    if explicit_public_base:
        base = explicit_public_base.rstrip("/")
        # Convert a WebSocket base URL to HTTP(S) if needed.
        if base.startswith("wss://"):
            base = "https://" + base[len("wss://"):]
        elif base.startswith("ws://"):
            base = "http://" + base[len("ws://"):]
        return f"{base}/stream-ended"

    forwarded_proto = request.headers.get("x-forwarded-proto", "").strip().lower()
    scheme = "https" if forwarded_proto != "http" else "http"
    return f"{scheme}://{host}/stream-ended"


@app.post("/stream-ended")
async def stream_ended(request: Request):
    """Twilio action callback fired when a ``<Connect><Stream>`` session ends.

    When a landline caller is being transferred to a human agent, closing
    the WebSocket releases the ``<Connect>`` block and Twilio immediately
    POSTs here.  We check the persisted session state and return
    ``<Dial>`` TwiML to bridge the caller to the human-agent number.
    For all other call endings (normal completion, errors) we return an
    empty response and let Twilio hang up.
    """
    form = await request.form()
    raw_form = dict(form)
    call_sid = str(form.get("CallSid", "")).strip()

    print(
        "[STREAM-ENDED RECEIVED]",
        {
            "call_sid": call_sid,
            "raw_form": raw_form,
        },
    )

    vr = VoiceResponse()

    if not call_sid:
        print("[STREAM-ENDED] No CallSid in request — returning empty TwiML (hang up)")
        return Response(content=str(vr), media_type="application/xml")

    session = load_existing_session(call_sid, "demo")

    if session is None:
        print(
            "[STREAM-ENDED] Session not found in Redis",
            {"call_sid": call_sid},
        )
        return Response(content=str(vr), media_type="application/xml")

    print(
        "[STREAM-ENDED SESSION]",
        {
            "call_sid": call_sid,
            "conversation_state": session.conversation_state,
            "caller_device_type": session.conversation_context.caller_device_type,
        },
    )

    if session.conversation_state == ConversationState.TRANSFERRING_TO_HUMAN_AGENT:
        transfer_target = (
            (
                (session.last_response_payload or {}).get("transfer_number")
                if isinstance(session.last_response_payload, dict)
                else None
            )
            or HUMAN_AGENT_TRANSFER_NUMBER
        )
        vr.say("Okay. Connecting you to a team member now. One moment please.")
        vr.dial(transfer_target)
        twiml_out = str(vr)
        print(
            "[STREAM-ENDED TRANSFER]",
            {
                "call_sid": call_sid,
                "target": transfer_target,
                "twiml": twiml_out,
            },
        )
        session.conversation_state = ConversationState.COMPLETED
        save_session(session)
        return Response(content=twiml_out, media_type="application/xml")

    print(
        "[STREAM-ENDED] State is not TRANSFERRING_TO_HUMAN_AGENT — returning empty TwiML",
        {"state": session.conversation_state},
    )
    return Response(content=str(vr), media_type="application/xml")


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

    if call_sid:
        session = load_session(call_sid, "demo")
        session.conversation_context.delivery_address.customer_phone_number = (
            from_number or None
        )
        save_session(session)

    action_url = _build_action_url(request)

    print(
        "[VOICE TWIML]",
        {
            "call_sid": call_sid,
            "stream_url": stream_url,
            "action_url": action_url,
        },
    )

    vr = VoiceResponse()
    connect = Connect(action=action_url)
    connect.stream(url=stream_url, name="compass-voice-stream")
    vr.append(connect)

    twiml_str = str(vr)
    print("[VOICE TWIML RESPONSE]", {"twiml": twiml_str})

    return Response(content=twiml_str, media_type="application/xml")


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
    bot_barge_in_guard_seconds = 0.35
    disable_barge_in = True
    playback_generation = 0
    welcome_sent = False

    twilio_account_sid = os.getenv("TWILIO_ACCOUNT_SID", "").strip()
    twilio_auth_token = os.getenv("TWILIO_AUTH_TOKEN", "").strip()
    twilio_client: Client | None = None
    if twilio_account_sid and twilio_auth_token:
        twilio_client = Client(twilio_account_sid, twilio_auth_token)

    should_end_call_after_playback = False
    pending_transfer_number: str | None = None
    websocket_close_requested = False
    _payment_check_task: asyncio.Task | None = None

    async def _transfer_live_call(target_number: str) -> None:
        """Initiate a transfer to ``target_number`` by closing the WebSocket.

        ``<Connect><Stream>`` blocks all TwiML that follows it for as long
        as the WebSocket stays open.  Calling the Twilio REST API with a
        new ``<Dial>`` TwiML while the stream is active is silently queued
        and never executes.

        The correct approach: close the WebSocket from our side.  Twilio
        then releases the ``<Connect>`` block and immediately POSTs to the
        ``action`` URL we registered on ``<Connect>`` in the ``/voice``
        handler.  That endpoint (``/stream-ended``) checks the session
        state and returns ``<Dial>{target_number}</Dial>`` TwiML, bridging
        the caller to the human-agent line.
        """
        nonlocal websocket_close_requested

        print(
            "[TRANSFER] _transfer_live_call called",
            {
                "call_sid": stream_session.call_sid,
                "stream_sid": stream_session.stream_sid,
                "target_number": target_number,
                "session_state": getattr(app_session, "conversation_state", None),
            },
        )
        # Closing the WebSocket releases the <Connect><Stream> block.
        # Twilio then POSTs to the action URL (/stream-ended) where we
        # return <Dial> TwiML to bridge the caller to the human agent.
        try:
            websocket_close_requested = True
            await websocket.close()
            print(
                "[TRANSFER] WebSocket closed — Twilio will POST to /stream-ended",
                {
                    "call_sid": stream_session.call_sid,
                    "stream_sid": stream_session.stream_sid,
                },
            )
        except Exception as exc:
            print(
                "[TRANSFER] WebSocket close failed",
                {
                    "call_sid": stream_session.call_sid,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )

    async def _end_live_call() -> None:
        if not stream_session.call_sid:
            print(
                "[TWILIO CALL END SKIPPED]",
                {
                    "reason": "missing_call_sid",
                    "stream_sid": stream_session.stream_sid,
                },
            )
            return

        if twilio_client is None:
            print(
                "[TWILIO CALL END SKIPPED]",
                {
                    "call_sid": stream_session.call_sid,
                    "reason": "missing_twilio_client",
                },
            )
            return

        try:
            result = await asyncio.to_thread(
                twilio_client.calls(stream_session.call_sid).update,
                status="completed",
            )
            print(
                "[TWILIO CALL END REQUESTED]",
                {
                    "call_sid": stream_session.call_sid,
                    "stream_sid": stream_session.stream_sid,
                    "twilio_status": getattr(result, "status", None),
                },
            )
        except Exception as exc:
            print(
                "[TWILIO CALL END FAILED]",
                {
                    "call_sid": stream_session.call_sid,
                    "stream_sid": stream_session.stream_sid,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )

    async def send_twilio_media(
        audio_bytes: bytes,
        trace: RealtimeTurnTrace | None = None,
    ) -> None:
        if not audio_bytes or not stream_session.stream_sid:
            return

        payload_b64 = base64.b64encode(audio_bytes).decode("ascii")
        message = {
            "event": "media",
            "streamSid": stream_session.stream_sid,
            "media": {"payload": payload_b64},
        }
        await websocket.send_text(json.dumps(message))

        now = time.perf_counter()
        stream_session.outbound_audio_bytes += len(audio_bytes)
        stream_session.outbound_media_messages += 1

        if trace is not None:
            if getattr(trace, "twilio_first_outbound_media_monotonic", None) is None:
                _trace_set_attr(trace, "twilio_first_outbound_media_monotonic", now)
                _trace_set_attr(trace, "response_first_audio_sent_at_utc", utc_now_iso())

            _trace_set_attr(trace, "twilio_last_outbound_media_monotonic", now)
            _trace_set_attr(
                trace,
                "outbound_audio_bytes",
                int(getattr(trace, "outbound_audio_bytes", 0) or 0) + len(audio_bytes),
            )

    async def send_twilio_mark(
        name: str,
        trace: RealtimeTurnTrace | None = None,
    ) -> None:
        if not stream_session.stream_sid:
            return

        message = {
            "event": "mark",
            "streamSid": stream_session.stream_sid,
            "mark": {"name": name},
        }
        await websocket.send_text(json.dumps(message))

        if trace is not None:
            _trace_set_attr(trace, "twilio_mark_sent_monotonic", time.perf_counter())

    async def send_twilio_clear() -> None:
        if not stream_session.stream_sid:
            return

        message = {
            "event": "clear",
            "streamSid": stream_session.stream_sid,
        }
        await websocket.send_text(json.dumps(message))

    async def _interrupt_bot_playback(reason: str) -> None:
        nonlocal phase, active_mark_name, bot_playback_started_at, playback_generation

        if phase != RealtimePhase.SPEAKING:
            return

        _debug_log(
            "[BOT PLAYBACK INTERRUPT]",
            {
                "stream_sid": stream_session.stream_sid,
                "reason": reason,
                "active_mark_name": active_mark_name,
            },
        )

        playback_generation += 1
        await send_twilio_clear()

        try:
            await dg_tts_client.clear()
        except Exception as exc:
            _debug_log(
                "[DEEPGRAM TTS CLEAR ERROR]",
                {
                    "stream_sid": stream_session.stream_sid,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )

        if stream_session.active_trace is not None:
            _trace_add_note(stream_session.active_trace, "barge_in", True)
            _trace_add_note(stream_session.active_trace, "barge_in_reason", reason)
            _trace_add_note(
                stream_session.active_trace,
                "barge_in_active_mark_name",
                active_mark_name,
            )

        active_mark_name = None
        stream_session.active_trace_mark_name = None
        bot_playback_started_at = None
        phase = RealtimePhase.LISTENING

    async def _send_burst_frames(
        burst_bytes: bytes,
        trace: RealtimeTurnTrace | None = None,
    ) -> int:
        sent = 0

        for offset in range(0, len(burst_bytes), TWILIO_MULAW_FRAME_BYTES):
            frame = burst_bytes[offset : offset + TWILIO_MULAW_FRAME_BYTES]
            if not frame:
                continue
            await send_twilio_media(frame, trace=trace)
            sent += len(frame)

        await asyncio.sleep(TWILIO_BURST_PACING_SECONDS)
        return sent

    async def stream_audio_to_twilio(
        audio_chunk_stream: AsyncGenerator[bytes, None],
        trace: RealtimeTurnTrace | None = None,
        should_abort: Callable[[], bool] | None = None,
    ) -> int:
        buffered = bytearray()
        total_bytes_sent = 0

        async for tts_chunk in audio_chunk_stream:
            if should_abort and should_abort():
                buffered.clear()
                break

            if not tts_chunk:
                continue

            now = time.perf_counter()
            if trace is not None:
                if getattr(trace, "tts_first_chunk_monotonic", None) is None:
                    _trace_set_attr(trace, "tts_first_chunk_monotonic", now)
                _trace_set_attr(trace, "tts_last_chunk_monotonic", now)

            buffered.extend(tts_chunk)

            while len(buffered) >= TWILIO_BURST_BYTES:
                if should_abort and should_abort():
                    buffered.clear()
                    return total_bytes_sent

                burst = bytes(buffered[:TWILIO_BURST_BYTES])
                del buffered[:TWILIO_BURST_BYTES]
                total_bytes_sent += await _send_burst_frames(burst, trace=trace)

        if should_abort and should_abort():
            return total_bytes_sent

        remaining_full_frame_bytes = (
            len(buffered) // TWILIO_MULAW_FRAME_BYTES
        ) * TWILIO_MULAW_FRAME_BYTES

        if remaining_full_frame_bytes > 0:
            burst = bytes(buffered[:remaining_full_frame_bytes])
            del buffered[:remaining_full_frame_bytes]
            total_bytes_sent += await _send_burst_frames(burst, trace=trace)

        if buffered:
            padded = bytes(buffered) + (
                b"\xFF" * (TWILIO_MULAW_FRAME_BYTES - len(buffered))
            )
            await send_twilio_media(padded, trace=trace)
            total_bytes_sent += len(padded)

        return total_bytes_sent

    async def _stream_progressive_tts_text(
        spoken_text: str,
        trace: RealtimeTurnTrace | None = None,
        generation: int = 0,
    ) -> int:
        chunks = _split_for_progressive_tts(spoken_text)
        if not chunks:
            return 0

        if trace is not None:
            _trace_add_note(trace, "tts_progressive_chunks", chunks)
            _trace_add_note(trace, "tts_progressive_chunk_count", len(chunks))

        async def audio_stream() -> AsyncGenerator[bytes, None]:
            for chunk in chunks:
                if generation != playback_generation:
                    return
                await dg_tts_client.send_text(chunk)

            if generation != playback_generation:
                return

            await dg_tts_client.flush()

            async for audio in dg_tts_client.iter_audio_until_flushed():
                if generation != playback_generation:
                    return
                yield audio

        return await stream_audio_to_twilio(
            audio_stream(),
            trace=trace,
            should_abort=lambda: generation != playback_generation,
        )

    async def _stream_static_mulaw_audio(
        audio_bytes: bytes,
        trace: RealtimeTurnTrace | None = None,
        generation: int = 0,
    ) -> int:
        if not audio_bytes:
            return 0

        async def audio_stream() -> AsyncGenerator[bytes, None]:
            for offset in range(0, len(audio_bytes), TWILIO_BURST_BYTES):
                if generation != playback_generation:
                    return
                yield audio_bytes[offset : offset + TWILIO_BURST_BYTES]

        return await stream_audio_to_twilio(
            audio_stream(),
            trace=trace,
            should_abort=lambda: generation != playback_generation,
        )

    def _build_response_texts(turn_output: Any) -> tuple[str, str]:
        internal_text = _normalize_response_text(
            getattr(turn_output, "internal_response_text", None)
        )
        spoken_text = _normalize_response_text(
            getattr(turn_output, "spoken_response_text", None)
        )

        if not internal_text:
            raise ValueError(
                f"TurnEngine returned empty internal_response_text for response_key="
                f"{getattr(turn_output, 'response_key', 'unknown')}"
            )

        if not spoken_text:
            spoken_text = internal_text

        return internal_text, spoken_text

    async def speak_response_text(
        spoken_text: str,
        trace: RealtimeTurnTrace | None = None,
        end_call_after_playback: bool = False,
    ) -> None:
        nonlocal phase, active_mark_name, mark_counter, bot_playback_started_at, playback_generation, pending_transfer_number

        cleaned = _normalize_response_text(spoken_text)
        if not cleaned:
            phase = RealtimePhase.LISTENING
            return

        phase = RealtimePhase.SPEAKING
        bot_playback_started_at = time.monotonic()
        playback_generation += 1
        generation = playback_generation

        if trace is not None:
            _trace_set_attr(trace, "spoken_response_text", cleaned)
            _trace_set_attr(trace, "tts_text_chars", len(cleaned))
            _trace_set_attr(trace, "tts_request_start_monotonic", time.perf_counter())
            _trace_set_attr(trace, "end_call_after_playback", end_call_after_playback)

        streamed_bytes = await _stream_progressive_tts_text(
            cleaned,
            trace=trace,
            generation=generation,
        )

        if generation != playback_generation:
            _debug_log(
                "[TTS PLAYBACK ABORTED]",
                {
                    "stream_sid": stream_session.stream_sid,
                    "generation": generation,
                },
            )
            return

        if streamed_bytes <= 0:
            print(
                "[TTS EMPTY AUDIO]",
                {
                    "stream_sid": stream_session.stream_sid,
                    "text": cleaned,
                },
            )
            _trace_add_note(trace, "tts_empty_audio", True)
            phase = RealtimePhase.LISTENING
            bot_playback_started_at = None

            # Transfer takes precedence over a plain hangup when both
            # have been requested for the same turn.
            if pending_transfer_number:
                target = pending_transfer_number
                pending_transfer_number = None
                await _transfer_live_call(target)
            elif end_call_after_playback:
                await _end_live_call()

            return

        if trace is not None:
            _trace_set_attr(
                trace,
                "outbound_audio_duration_ms",
                round((streamed_bytes / 8000.0) * 1000.0, 3),
            )

        # If a transfer is pending we must NOT end the call here — that
        # would tear down the leg before Twilio can dial the human agent.
        # The mark handler will perform the transfer once playback ends.
        if end_call_after_playback and not pending_transfer_number:
            _debug_log(
                "[TWILIO CALL END AFTER PLAYBACK]",
                {
                    "stream_sid": stream_session.stream_sid,
                    "call_sid": stream_session.call_sid,
                },
            )
            await _end_live_call()

        mark_counter += 1
        active_mark_name = f"bot-playback-{mark_counter}"
        stream_session.active_trace_mark_name = active_mark_name
        await send_twilio_mark(active_mark_name, trace=trace)

        _debug_log(
            "[TWILIO OUTBOUND AUDIO SENT]",
            {
                "stream_sid": stream_session.stream_sid,
                "bytes": streamed_bytes,
                "mark_name": active_mark_name,
                "barge_in_disabled": disable_barge_in,
                "burst_frames": TWILIO_BURST_FRAMES,
                "generation": generation,
                "end_call_after_playback": end_call_after_playback,
            },
        )

    async def speak_cached_welcome_audio() -> None:
        nonlocal phase, active_mark_name, mark_counter, bot_playback_started_at, playback_generation

        audio_bytes = _get_welcome_audio_bytes()
        if not audio_bytes:
            raise ValueError("Welcome audio is empty.")

        phase = RealtimePhase.SPEAKING
        bot_playback_started_at = time.monotonic()
        playback_generation += 1
        generation = playback_generation

        streamed_bytes = await _stream_static_mulaw_audio(
            audio_bytes,
            trace=None,
            generation=generation,
        )

        if generation != playback_generation:
            return

        if streamed_bytes <= 0:
            phase = RealtimePhase.LISTENING
            bot_playback_started_at = None
            raise ValueError("Welcome audio produced no outbound media.")

        mark_counter += 1
        active_mark_name = f"bot-playback-{mark_counter}"
        stream_session.active_trace_mark_name = active_mark_name
        await send_twilio_mark(active_mark_name, trace=None)

        _debug_log(
            "[TWILIO WELCOME AUDIO SENT]",
            {
                "stream_sid": stream_session.stream_sid,
                "bytes": streamed_bytes,
                "mark_name": active_mark_name,
                "burst_frames": TWILIO_BURST_FRAMES,
            },
        )

    async def _schedule_payment_auto_check() -> None:
        """Arm (or re-arm) a background timer that probes payment status.

        Fires once after PAYMENT_AUTO_CHECK_DELAY_SECONDS of silence and
        silently re-arms itself while the order is still unpaid, so the
        agent keeps checking every ~50 s until payment clears or the call
        ends.  Cancelled automatically when the WebSocket closes.
        """
        nonlocal _payment_check_task

        if _payment_check_task is not None and not _payment_check_task.done():
            _payment_check_task.cancel()

        async def _probe() -> None:
            try:
                await asyncio.sleep(PAYMENT_AUTO_CHECK_DELAY_SECONDS)
            except asyncio.CancelledError:
                return
            # Only probe when the call is idle and still in a payment state.
            if (
                app_session is None
                or app_session.conversation_state not in _PAYMENT_AWAITING_STATES
                or phase != RealtimePhase.LISTENING
            ):
                return
            await process_committed_turn("__auto_payment_check__")
            # Re-arm if payment is still pending after the probe.
            if (
                app_session is not None
                and app_session.conversation_state in _PAYMENT_AWAITING_STATES
            ):
                await _schedule_payment_auto_check()

        _payment_check_task = asyncio.create_task(_probe())

    async def process_committed_turn(user_text: str) -> None:
        nonlocal phase, pending_interrupt_text, app_session, should_end_call_after_playback, pending_transfer_number

        if app_session is None:
            return

        cleaned = " ".join(user_text.split()).strip()
        if not cleaned:
            return

        if phase == RealtimePhase.PROCESSING:
            pending_interrupt_text = cleaned
            _debug_log(
                "[INTERRUPT BUFFERED]",
                {
                    "stream_sid": stream_session.stream_sid,
                    "text": cleaned,
                },
            )
            return

        async with processing_lock:
            phase = RealtimePhase.PROCESSING
            stream_session.turn_index += 1

            if stream_session.active_trace is not None:
                _finalize_active_trace(
                    app=app,
                    stream_session=stream_session,
                    extra_notes={"interrupted_before_mark": True},
                    reset_utterance_tracking=False,
                )

            trace = _build_turn_trace(
                stream_session=stream_session,
                app_session=app_session,
                user_text=cleaned,
            )
            stream_session.active_trace = trace

            turn_output = app.state.engine.process_turn(
                session=app_session,
                user_text=cleaned,
                trace=trace,
            )

            save_session(app_session)

            _trace_set_attr(trace, "responder_start_monotonic", time.perf_counter())
            internal_response_text, spoken_response_text = _build_response_texts(turn_output)
            _trace_set_attr(trace, "responder_end_monotonic", time.perf_counter())
            _trace_set_attr(trace, "response_key", turn_output.response_key)
            _trace_set_attr(trace, "response_text", internal_response_text)
            _trace_set_attr(trace, "spoken_response_text", spoken_response_text)
            _trace_set_attr(
                trace,
                "end_call_after_playback",
                bool(getattr(turn_output, "end_call_after_playback", False)),
            )

            should_end_call_after_playback = bool(
                getattr(turn_output, "end_call_after_playback", False)
            )
            pending_transfer_number = (
                getattr(turn_output, "transfer_call_to_number", None) or None
            )

            print(
                "[TURN OUTPUT]",
                {
                    "response_key": turn_output.response_key,
                    "session_state": getattr(app_session, "conversation_state", None),
                    "end_call_after_playback": should_end_call_after_playback,
                    "pending_transfer_number": pending_transfer_number,
                },
            )

            if pending_transfer_number:
                target = pending_transfer_number
                pending_transfer_number = None
                should_end_call_after_playback = False
                print(
                    "[TURN OUTPUT] Initiating immediate transfer handoff",
                    {
                        "target": target,
                        "call_sid": stream_session.call_sid,
                    },
                )
                await _transfer_live_call(target)
                return

            await speak_response_text(
                spoken_response_text,
                trace=trace,
                end_call_after_playback=should_end_call_after_playback,
            )

            # Arm the silent auto-confirm probe when a payment/checkout
            # link has just been spoken.  The probe fires after
            # PAYMENT_AUTO_CHECK_DELAY_SECONDS and re-arms itself while
            # the order remains unpaid — no user input required.
            if turn_output.response_key in PAYMENT_LINK_SENT_KEYS:
                await _schedule_payment_auto_check()

            if pending_interrupt_text and phase == RealtimePhase.LISTENING:
                buffered = pending_interrupt_text
                pending_interrupt_text = None
                await process_committed_turn(buffered)

    async def on_dg_transcript(transcript: str, is_final: bool, payload: dict) -> None:
        committed = controller.on_transcript(transcript, is_final)

        if is_final:
            stream_session.last_dg_final_transcript_monotonic = time.perf_counter()

        if committed is not None:
            await process_committed_turn(committed.text)
            return

        speech_final = bool(payload.get("speech_final", False))
        if speech_final:
            committed = controller.on_speech_final()
            if committed is not None:
                await process_committed_turn(committed.text)

    async def on_dg_event(name: str, payload: dict) -> None:
        nonlocal phase, bot_playback_started_at

        if name == "message:SpeechStarted":
            controller.on_speech_started()
            now = time.perf_counter()
            stream_session.last_dg_speech_started_monotonic = now

            if stream_session.current_utterance_first_media_monotonic is None:
                stream_session.current_utterance_first_media_monotonic = now
            stream_session.current_utterance_last_media_monotonic = now

            if phase == RealtimePhase.SPEAKING:
                if disable_barge_in:
                    return

                guard_now = time.monotonic()
                if (
                    bot_playback_started_at is not None
                    and guard_now - bot_playback_started_at < bot_barge_in_guard_seconds
                ):
                    return

                await _interrupt_bot_playback(reason="user_speech_started")

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

    async def _start_stt_client_in_background() -> bool:
        nonlocal dg_stt_started

        if dg_stt_client is None:
            return False

        try:
            await dg_stt_client.start()
            dg_stt_started = True
            print("[DEEPGRAM STT CONNECTED]", {"stream_sid": stream_session.stream_sid})
            return True
        except Exception as exc:
            print(
                "[DEEPGRAM STT CONNECT FAILED]",
                {
                    "stream_sid": stream_session.stream_sid,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            return False

    async def _connect_tts_in_background() -> bool:
        try:
            await dg_tts_client.connect()
            print("[DEEPGRAM TTS CONNECTED]", {"stream_sid": stream_session.stream_sid})
            return True
        except Exception as exc:
            print(
                "[DEEPGRAM TTS CONNECT FAILED]",
                {
                    "stream_sid": stream_session.stream_sid,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            return False

    print("[WS OPEN] Twilio media WebSocket connected")

    try:
        while True:
            try:
                raw_message = await websocket.receive_json()
            except RuntimeError as exc:
                if websocket_close_requested and "WebSocket is not connected" in str(exc):
                    print(
                        "[WS CLOSED AFTER TRANSFER HANDOFF]",
                        {
                            "call_sid": stream_session.call_sid,
                            "stream_sid": stream_session.stream_sid,
                        },
                    )
                    break
                raise

            event = raw_message.get("event")

            if event == "start":
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
                        endpointing=160,
                        keepalive_interval_seconds=4.0,
                    ),
                    callbacks=DeepgramSTTCallbacks(
                        on_transcript=on_dg_transcript,
                        on_event=on_dg_event,
                        on_error=on_dg_error,
                    ),
                )

                stt_connect_task = asyncio.create_task(_start_stt_client_in_background())
                tts_connect_task = asyncio.create_task(_connect_tts_in_background())

                if not welcome_sent:
                    # Always ask the device-type question first via TTS.
                    # The cached welcome audio still says "pickup or
                    # delivery" and is no longer the right opener — the
                    # very first thing we need to know is whether the
                    # caller is on a landline (so we can hand off to a
                    # human) or a mobile (so we can text payment links).
                    try:
                        await speak_response_text(
                            "Welcome to Compass. Before we get started, "
                            "are you calling from a landline or a mobile phone?"
                        )
                    finally:
                        welcome_sent = True
                        disable_barge_in = False

                await asyncio.gather(stt_connect_task, tts_connect_task)

            elif event == "media":
                media = raw_message.get("media", {})
                payload_b64 = media.get("payload", "")

                stream_session.inbound_chunks += 1
                stream_session.inbound_bytes_b64 += len(payload_b64)

                if not payload_b64:
                    continue

                audio_bytes = base64.b64decode(payload_b64)
                stream_session.inbound_audio_bytes += len(audio_bytes)

                now = time.perf_counter()
                if phase in {RealtimePhase.LISTENING, RealtimePhase.PROCESSING}:
                    if stream_session.current_utterance_first_media_monotonic is None:
                        stream_session.current_utterance_first_media_monotonic = now
                    stream_session.current_utterance_last_media_monotonic = now
                    stream_session.current_utterance_inbound_audio_bytes += len(audio_bytes)

                if dg_stt_started and dg_stt_client is not None:
                    await dg_stt_client.send_audio(audio_bytes)

            elif event == "mark":
                mark = raw_message.get("mark", {})
                mark_name = mark.get("name")

                if active_mark_name and mark_name == active_mark_name:
                    active_mark_name = None
                    bot_playback_started_at = None

                    if stream_session.active_trace is not None:
                        _trace_set_attr(
                            stream_session.active_trace,
                            "twilio_mark_received_monotonic",
                            time.perf_counter(),
                        )
                        _trace_set_attr(
                            stream_session.active_trace,
                            "playback_completed_at_utc",
                            utc_now_iso(),
                        )

                    if phase == RealtimePhase.SPEAKING:
                        phase = RealtimePhase.LISTENING

                    _finalize_active_trace(app=app, stream_session=stream_session)

                    print(
                        "[MARK HANDLER]",
                        {
                            "mark_name": mark_name,
                            "pending_transfer_number": pending_transfer_number,
                            "should_end_call_after_playback": should_end_call_after_playback,
                            "session_state": getattr(app_session, "conversation_state", None),
                        },
                    )

                    if pending_transfer_number:
                        target = pending_transfer_number
                        pending_transfer_number = None
                        should_end_call_after_playback = False
                        print(
                            "[MARK HANDLER] Triggering transfer",
                            {"target": target},
                        )
                        await _transfer_live_call(target)
                        continue

                    if should_end_call_after_playback:
                        should_end_call_after_playback = False
                        await _end_live_call()
                        continue

                    if pending_interrupt_text:
                        buffered = pending_interrupt_text
                        pending_interrupt_text = None
                        await process_committed_turn(buffered)

            elif event == "stop":
                print(
                    "[TWILIO STREAM STOP]",
                    {
                        "call_sid": stream_session.call_sid,
                        "stream_sid": stream_session.stream_sid,
                    },
                )

                if stream_session.active_trace is not None:
                    _finalize_active_trace(app=app, stream_session=stream_session)

                if dg_stt_client is not None:
                    await dg_stt_client.finalize()

                break

    except WebSocketDisconnect:
        if stream_session.active_trace is not None:
            _finalize_active_trace(app=app, stream_session=stream_session)
    finally:
        if _payment_check_task is not None and not _payment_check_task.done():
            _payment_check_task.cancel()
        if dg_stt_client is not None:
            await dg_stt_client.close()
        await dg_tts_client.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.api.voice_stream_server:app",
        host="0.0.0.0",
        port=5000,
        reload=False,
    )
