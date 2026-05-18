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
from app.api.health_routes import router as health_router
from app.api.payment_links_webhook import router as payment_links_webhook_router
from app.config.realtime import get_realtime_turn_config
from app.config.required_env import assert_required_env_or_die
from app.config.restaurant import DEFAULT_RESTAURANT_ID

# Fail fast at import time if any required secret/config is missing.
# With gunicorn --preload, this runs in the master process before workers
# fork, so a misconfigured deploy crashes `docker compose up` instead of
# the first WebSocket request. See app/config/required_env.py.
assert_required_env_or_die()
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
from app.realtime.conversation_session import ConversationSession
from app.realtime.tts_exceptions import TTSFailureError
from app.core.session_task_manager import SessionTaskManager
from app.realtime.turn_commit_controller import TurnCommitController
from app.session.repository import load_existing_session, load_session, save_session
from app.session.session import Session
from app.config.voice_transfer import HUMAN_AGENT_TRANSFER_NUMBER
from app.state_machine.models.conversation_state import ConversationState

TWILIO_MULAW_FRAME_BYTES = 160
TWILIO_FRAME_DURATION_SECONDS = 0.02

TWILIO_BURST_FRAMES = int(float(os.getenv("TWILIO_BURST_FRAMES", "5")))
if TWILIO_BURST_FRAMES <= 0:
    TWILIO_BURST_FRAMES = 5

TWILIO_BURST_BYTES = int(TWILIO_MULAW_FRAME_BYTES * TWILIO_BURST_FRAMES)
TWILIO_BURST_PACING_SECONDS = TWILIO_FRAME_DURATION_SECONDS * TWILIO_BURST_FRAMES
TTS_EMPTY_AUDIO_MAX_ATTEMPTS = max(
    1,
    int(os.getenv("COMPASS_TTS_EMPTY_AUDIO_MAX_ATTEMPTS", "3")),
)
TTS_EMPTY_AUDIO_RETRY_SETTLE_SECONDS = max(
    0.0,
    float(os.getenv("COMPASS_TTS_EMPTY_AUDIO_RETRY_SETTLE_SECONDS", "0.12")),
)

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
    "repeat_landline_pickup_only",
    "transferring_to_human_agent",
    "ask_for_order_type",
    "repeat_order_type",
    "order_type_captured_pickup",
    "order_type_captured_delivery",
    "confirm_order_summary",
}

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
    runtime = build_runtime(restaurant_id=DEFAULT_RESTAURANT_ID)

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

app.include_router(health_router)
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

    session = load_existing_session(call_sid, DEFAULT_RESTAURANT_ID)

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
        session = load_session(call_sid, DEFAULT_RESTAURANT_ID)
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
    session_mgr = SessionTaskManager()
    session_id = str(id(stream_session))
    dg_stt_client: DeepgramSTTClient | None = None
    dg_stt_started = False
    dg_tts_client = DeepgramTTSClient()

    restaurant_id = DEFAULT_RESTAURANT_ID

    controller = TurnCommitController()

    active_mark_name: str | None = None
    mark_counter = 0

    bot_playback_started_at: float | None = None
    disable_barge_in = True
    playback_generation = 0
    welcome_sent = False

    # Debounce timer — fires USER_TURN_COMMIT_DELAY_MS after the last STT final
    # when speech_final / UtteranceEnd have not already triggered an immediate
    # commit.  Replaced (not stacked) on each new final fragment.
    _commit_debounce_task: asyncio.Task | None = None

    # Monotonic timestamp when the current barge-in candidate speech started.
    # Reset on each SpeechStarted event; used to measure candidate audio duration.
    _barge_in_speech_started_at: float | None = None

    twilio_account_sid = os.getenv("TWILIO_ACCOUNT_SID", "").strip()
    twilio_auth_token = os.getenv("TWILIO_AUTH_TOKEN", "").strip()
    twilio_client: Client | None = None
    if twilio_account_sid and twilio_auth_token:
        twilio_client = Client(twilio_account_sid, twilio_auth_token)

    websocket_close_requested = False

    # ConversationSession + transport adapter are constructed below, after
    # the local closures they wrap (``speak_response_text``,
    # ``_interrupt_bot_playback``, etc.) are defined.
    conv_session: ConversationSession | None = None

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
                "session_state": getattr(
                    conv_session.app_session if conv_session else None,
                    "conversation_state",
                    None,
                ),
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
        nonlocal active_mark_name, bot_playback_started_at, playback_generation

        if conv_session is None or not conv_session.is_speaking():
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
        conv_session.set_phase_listening()

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
                    # Close generator to cancel any pending asyncio queue-watcher
                    # tasks inside iter_audio_until_flushed().
                    await audio_chunk_stream.aclose()
                    return total_bytes_sent

                burst = bytes(buffered[:TWILIO_BURST_BYTES])
                del buffered[:TWILIO_BURST_BYTES]
                total_bytes_sent += await _send_burst_frames(burst, trace=trace)

        if should_abort and should_abort():
            await audio_chunk_stream.aclose()
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

            # Hold an explicit reference so we can aclose() it on early exit.
            # Without this, the abandoned generator's internal asyncio.create_task()
            # queue-watcher tasks stay alive and can steal the next utterance's
            # Flushed/Cleared events from _event_queue.
            audio_iter = dg_tts_client.iter_audio_until_flushed()
            try:
                async for audio in audio_iter:
                    if generation != playback_generation:
                        return
                    yield audio
            finally:
                await audio_iter.aclose()

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

    async def _stream_single_tts_text(
        spoken_text: str,
        trace: RealtimeTurnTrace | None = None,
        generation: int = 0,
    ) -> int:
        cleaned = _normalize_response_text(spoken_text)
        if not cleaned:
            return 0

        if trace is not None:
            _trace_add_note(trace, "tts_retry_mode", "single_chunk")

        async def audio_stream() -> AsyncGenerator[bytes, None]:
            await dg_tts_client.send_text(cleaned)

            if generation != playback_generation:
                return

            await dg_tts_client.flush()

            audio_iter = dg_tts_client.iter_audio_until_flushed()
            try:
                async for audio in audio_iter:
                    if generation != playback_generation:
                        return
                    yield audio
            finally:
                await audio_iter.aclose()

        return await stream_audio_to_twilio(
            audio_stream(),
            trace=trace,
            should_abort=lambda: generation != playback_generation,
        )

    async def _reconnect_tts_client(reason: str) -> bool:
        nonlocal dg_tts_client

        _debug_log(
            "[DEEPGRAM TTS RECONNECT]",
            {
                "stream_sid": stream_session.stream_sid,
                "reason": reason,
            },
        )

        try:
            await dg_tts_client.close()
        except Exception as exc:
            _debug_log(
                "[DEEPGRAM TTS CLOSE ERROR]",
                {
                    "stream_sid": stream_session.stream_sid,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )

        try:
            dg_tts_client = DeepgramTTSClient()
            await dg_tts_client.connect()
            print(
                "[DEEPGRAM TTS RECONNECTED]",
                {
                    "stream_sid": stream_session.stream_sid,
                    "reason": reason,
                },
            )
            return True
        except Exception as exc:
            print(
                "[DEEPGRAM TTS RECONNECT FAILED]",
                {
                    "stream_sid": stream_session.stream_sid,
                    "reason": reason,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            return False

    async def speak_response_text(
        spoken_text: str,
        trace: RealtimeTurnTrace | None = None,
        end_call_after_playback: bool = False,
    ) -> None:
        nonlocal active_mark_name, mark_counter, bot_playback_started_at, playback_generation

        cleaned = _normalize_response_text(spoken_text)
        if not cleaned:
            if conv_session is not None:
                conv_session.set_phase_listening()
            return

        if trace is not None:
            _trace_set_attr(trace, "spoken_response_text", cleaned)
            _trace_set_attr(trace, "tts_text_chars", len(cleaned))
            _trace_set_attr(trace, "end_call_after_playback", end_call_after_playback)

        max_attempts = TTS_EMPTY_AUDIO_MAX_ATTEMPTS
        streamed_bytes = 0
        generation = 0

        for attempt in range(1, max_attempts + 1):
            if conv_session is not None:
                conv_session.set_phase_speaking()
            bot_playback_started_at = time.monotonic()
            playback_generation += 1
            generation = playback_generation

            if trace is not None:
                if attempt == 1:
                    _trace_set_attr(trace, "tts_request_start_monotonic", time.perf_counter())
                _trace_add_note(trace, "tts_attempt", attempt)

            if attempt == 1:
                streamed_bytes = await _stream_progressive_tts_text(
                    cleaned,
                    trace=trace,
                    generation=generation,
                )
            else:
                streamed_bytes = await _stream_single_tts_text(
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
                        "attempt": attempt,
                    },
                )
                return

            if streamed_bytes > 0:
                break

            print(
                "[TTS EMPTY AUDIO]",
                {
                    "stream_sid": stream_session.stream_sid,
                    "text": cleaned,
                    "attempt": attempt,
                },
            )
            _trace_add_note(trace, "tts_empty_audio", True)
            _trace_add_note(trace, "tts_empty_audio_attempt", attempt)

            if attempt >= max_attempts:
                bot_playback_started_at = None
                raise TTSFailureError(
                    f"TTS returned empty audio after {attempt} attempts",
                    attempts=attempt,
                    provider="deepgram",
                )

            if conv_session is not None:
                conv_session.set_phase_listening()
            bot_playback_started_at = None

            reconnected = await _reconnect_tts_client(reason="empty_audio_retry")
            _trace_add_note(trace, "tts_empty_audio_reconnected", reconnected)
            if not reconnected:
                raise TTSFailureError(
                    f"TTS reconnect failed on attempt {attempt} — no audio delivered",
                    attempts=attempt,
                    provider="deepgram",
                )

            if TTS_EMPTY_AUDIO_RETRY_SETTLE_SECONDS > 0:
                await asyncio.sleep(TTS_EMPTY_AUDIO_RETRY_SETTLE_SECONDS)

        if trace is not None:
            _trace_set_attr(
                trace,
                "outbound_audio_duration_ms",
                round((streamed_bytes / 8000.0) * 1000.0, 3),
            )

        # If a transfer is pending we must NOT end the call here — that
        # would tear down the leg before Twilio can dial the human agent.
        # The mark handler will perform the transfer once playback ends.
        pending_transfer_number = (
            conv_session.pending_transfer_number if conv_session else None
        )
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
        nonlocal active_mark_name, mark_counter, bot_playback_started_at, playback_generation

        audio_bytes = _get_welcome_audio_bytes()
        if not audio_bytes:
            raise ValueError("Welcome audio is empty.")

        if conv_session is not None:
            conv_session.set_phase_speaking()
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
            if conv_session is not None:
                conv_session.set_phase_listening()
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

    class _TwilioVoiceTransport:
        """Adapter exposing the WebSocket-handler closures to
        :class:`ConversationSession` via the
        :class:`app.realtime.voice_transport.VoiceTransport` interface.

        Holds no state of its own; every method delegates to the local
        closure variables of ``twilio_media_ws`` so the conversation
        layer remains transport-agnostic.
        """

        def is_barge_in_disabled(self) -> bool:
            return disable_barge_in

        def debug_log(self, event: str, payload: dict[str, Any]) -> None:
            _debug_log(event, payload)

        def begin_turn_trace(self, *, user_text: str) -> RealtimeTurnTrace | None:
            stream_session.turn_index += 1

            if stream_session.active_trace is not None:
                _finalize_active_trace(
                    app=app,
                    stream_session=stream_session,
                    extra_notes={"interrupted_before_mark": True},
                    reset_utterance_tracking=False,
                )

            if conv_session is None or conv_session.app_session is None:
                return None

            trace = _build_turn_trace(
                stream_session=stream_session,
                app_session=conv_session.app_session,
                user_text=user_text,
            )
            stream_session.active_trace = trace
            return trace

        def annotate_response_trace(
            self,
            trace: RealtimeTurnTrace | None,
            *,
            responder_start_monotonic: float | None,
            responder_end_monotonic: float | None,
            response_key: str,
            internal_response_text: str,
            spoken_response_text: str,
            end_call_after_playback: bool,
        ) -> None:
            if trace is None:
                return
            _trace_set_attr(trace, "responder_start_monotonic", responder_start_monotonic)
            _trace_set_attr(trace, "responder_end_monotonic", responder_end_monotonic)
            _trace_set_attr(trace, "response_key", response_key)
            _trace_set_attr(trace, "response_text", internal_response_text)
            _trace_set_attr(trace, "spoken_response_text", spoken_response_text)
            _trace_set_attr(trace, "end_call_after_playback", bool(end_call_after_playback))

        async def speak_response(
            self,
            spoken_text: str,
            *,
            trace: RealtimeTurnTrace | None,
            end_call_after_playback: bool,
        ) -> None:
            await speak_response_text(
                spoken_text,
                trace=trace,
                end_call_after_playback=end_call_after_playback,
            )

        async def interrupt_playback(self, reason: str) -> None:
            await _interrupt_bot_playback(reason)

        async def transfer_call(self, target_number: str) -> None:
            await _transfer_live_call(target_number)

        async def end_call(self) -> None:
            await _end_live_call()

    conv_session = ConversationSession(
        app_session=None,
        engine=app.state.engine,
        transport=_TwilioVoiceTransport(),
    )

    async def _cancel_debounce() -> None:
        nonlocal _commit_debounce_task
        if _commit_debounce_task is not None and not _commit_debounce_task.done():
            _commit_debounce_task.cancel()
            _commit_debounce_task = None

    async def _schedule_debounce_commit(reason: str) -> None:
        """Schedule a delayed commit for USER_TURN_COMMIT_DELAY_MS.

        Called after each STT final that doesn't satisfy the early-commit
        whitelist.  Replaces any existing pending timer so multiple finals
        within the debounce window merge into one logical turn.
        """
        nonlocal _commit_debounce_task
        rt_cfg = get_realtime_turn_config()
        delay_s = rt_cfg.user_turn_commit_delay_ms / 1000.0

        await _cancel_debounce()

        _debug_log(
            "[user_turn_commit_scheduled]",
            {
                "stream_sid": stream_session.stream_sid,
                "delay_ms": rt_cfg.user_turn_commit_delay_ms,
                "reason": reason,
                "session_id": _safe_session_id(conv_session.app_session if conv_session else None),
            },
        )

        async def _debounce_fire() -> None:
            nonlocal _commit_debounce_task
            await asyncio.sleep(delay_s)
            _commit_debounce_task = None
            committed = controller.commit()
            if committed is not None:
                _debug_log(
                    "[user_turn_committed]",
                    {
                        "stream_sid": stream_session.stream_sid,
                        "turn_id": committed.turn_id,
                        "text": committed.text,
                        "trigger": "debounce_timer",
                        "session_id": _safe_session_id(
                            conv_session.app_session if conv_session else None
                        ),
                    },
                )
                barge_audio_ms = (
                    (time.monotonic() - _barge_in_speech_started_at) * 1000.0
                    if _barge_in_speech_started_at is not None
                    else None
                )
                await conv_session.process_committed_turn(
                    committed.text,
                    turn_id=committed.turn_id,
                    barge_in_audio_ms=barge_audio_ms,
                )

        _commit_debounce_task = session_mgr.create_task(
            session_id, _debounce_fire(), name="commit_debounce"
        )

    async def on_dg_transcript(transcript: str, is_final: bool, payload: dict) -> None:
        nonlocal _commit_debounce_task

        if is_final:
            stream_session.last_dg_final_transcript_monotonic = time.perf_counter()
            _debug_log(
                "[stt_final_received]",
                {
                    "stream_sid": stream_session.stream_sid,
                    "transcript": transcript,
                    "speech_final": payload.get("speech_final", False),
                    "session_id": _safe_session_id(
                        conv_session.app_session if conv_session else None
                    ),
                },
            )
        else:
            _debug_log(
                "[stt_partial_received]",
                {
                    "stream_sid": stream_session.stream_sid,
                    "transcript": transcript,
                    "session_id": _safe_session_id(
                        conv_session.app_session if conv_session else None
                    ),
                },
            )

        committed = controller.on_transcript(transcript, is_final)

        if committed is not None:
            # Early-committed via whitelist (yes/no/ok/checkout…)
            await _cancel_debounce()
            _debug_log(
                "[user_turn_committed]",
                {
                    "stream_sid": stream_session.stream_sid,
                    "turn_id": committed.turn_id,
                    "text": committed.text,
                    "trigger": "early_commit_whitelist",
                    "session_id": _safe_session_id(
                        conv_session.app_session if conv_session else None
                    ),
                },
            )
            barge_audio_ms = (
                (time.monotonic() - _barge_in_speech_started_at) * 1000.0
                if _barge_in_speech_started_at is not None
                else None
            )
            await conv_session.process_committed_turn(
                committed.text,
                turn_id=committed.turn_id,
                barge_in_audio_ms=barge_audio_ms,
            )
            return

        if not is_final:
            return

        speech_final = bool(payload.get("speech_final", False))
        if speech_final:
            # Deepgram signals end of utterance — commit immediately.
            await _cancel_debounce()
            committed = controller.on_speech_final()
            if committed is not None:
                _debug_log(
                    "[user_turn_committed]",
                    {
                        "stream_sid": stream_session.stream_sid,
                        "turn_id": committed.turn_id,
                        "text": committed.text,
                        "trigger": "speech_final",
                        "session_id": _safe_session_id(
                            conv_session.app_session if conv_session else None
                        ),
                    },
                )
                barge_audio_ms = (
                    (time.monotonic() - _barge_in_speech_started_at) * 1000.0
                    if _barge_in_speech_started_at is not None
                    else None
                )
                await conv_session.process_committed_turn(
                    committed.text,
                    turn_id=committed.turn_id,
                    barge_in_audio_ms=barge_audio_ms,
                )
        else:
            # Non-speech_final final — schedule debounce to wait for more fragments.
            _debug_log(
                "[user_turn_commit_scheduled]",
                {
                    "stream_sid": stream_session.stream_sid,
                    "transcript": transcript,
                    "session_id": _safe_session_id(
                        conv_session.app_session if conv_session else None
                    ),
                },
            )
            await _schedule_debounce_commit(reason="stt_final_no_speech_final")

    async def on_dg_event(name: str, payload: dict) -> None:
        nonlocal _barge_in_speech_started_at

        if name == "message:SpeechStarted":
            controller.on_speech_started()
            now = time.perf_counter()
            stream_session.last_dg_speech_started_monotonic = now
            _barge_in_speech_started_at = time.monotonic()

            if stream_session.current_utterance_first_media_monotonic is None:
                stream_session.current_utterance_first_media_monotonic = now
            stream_session.current_utterance_last_media_monotonic = now

            if conv_session.is_speaking():
                if disable_barge_in:
                    return

                rt_cfg = get_realtime_turn_config()
                guard_s = rt_cfg.post_playback_guard_ms / 1000.0
                from app.realtime.barge_in_policy import is_within_barge_in_guard_window
                if is_within_barge_in_guard_window(bot_playback_started_at, guard_s):
                    _debug_log(
                        "[barge_in_candidate]",
                        {
                            "stream_sid": stream_session.stream_sid,
                            "status": "suppressed_guard_window",
                            "playback_age_ms": round(
                                (time.monotonic() - bot_playback_started_at) * 1000
                            ) if bot_playback_started_at else None,
                        },
                    )
                    return

                _debug_log(
                    "[barge_in_candidate]",
                    {
                        "stream_sid": stream_session.stream_sid,
                        "status": "pending",
                        "active_mark_name": active_mark_name,
                    },
                )

        elif name == "message:UtteranceEnd":
            # UtteranceEnd is the most reliable commit signal — cancel debounce
            # and commit the merged finals immediately.
            await _cancel_debounce()
            committed = controller.on_utterance_end()
            if committed is not None:
                _debug_log(
                    "[user_turn_committed]",
                    {
                        "stream_sid": stream_session.stream_sid,
                        "turn_id": committed.turn_id,
                        "text": committed.text,
                        "trigger": "utterance_end",
                        "session_id": _safe_session_id(
                            conv_session.app_session if conv_session else None
                        ),
                    },
                )
                barge_audio_ms = (
                    (time.monotonic() - _barge_in_speech_started_at) * 1000.0
                    if _barge_in_speech_started_at is not None
                    else None
                )
                await conv_session.process_committed_turn(
                    committed.text,
                    turn_id=committed.turn_id,
                    barge_in_audio_ms=barge_audio_ms,
                )

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
                    conv_session.load_app_session(
                        stream_session.call_sid,
                        restaurant_id,
                    )

                _rt_cfg = get_realtime_turn_config()
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
                        endpointing=_rt_cfg.deepgram_stt_endpointing_ms,
                        utterance_end_ms=_rt_cfg.deepgram_stt_utterance_end_ms,
                        keepalive_interval_seconds=4.0,
                    ),
                    callbacks=DeepgramSTTCallbacks(
                        on_transcript=on_dg_transcript,
                        on_event=on_dg_event,
                        on_error=on_dg_error,
                    ),
                )

                stt_connect_task = session_mgr.create_task(
                    session_id,
                    _start_stt_client_in_background(),
                    name="stt_connect",
                )
                tts_connect_task = session_mgr.create_task(
                    session_id,
                    _connect_tts_in_background(),
                    name="tts_connect",
                )

                if not welcome_sent:
                    try:
                        await speak_response_text(
                            app.state.responder.build("ask_for_order_type", None, None)
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
                if conv_session.is_listening() or conv_session.is_processing():
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

                    _finalize_active_trace(app=app, stream_session=stream_session)

                    print(
                        "[MARK HANDLER]",
                        {
                            "mark_name": mark_name,
                            "pending_transfer_number": conv_session.pending_transfer_number,
                            "should_end_call_after_playback": conv_session.should_end_call_after_playback,
                            "session_state": getattr(
                                conv_session.app_session,
                                "conversation_state",
                                None,
                            ),
                        },
                    )

                    await conv_session.on_playback_completed()

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
        playback_generation += 1
        await session_mgr.cleanup(session_id)
        await conv_session.cancel_payment_auto_check()
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

