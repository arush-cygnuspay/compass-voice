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
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from twilio.rest import Client
from twilio.twiml.voice_response import Connect, VoiceResponse

from app.api.chat_demo import router as test_chat_router
from app.api.ui.ui import router as ui_router
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
from app.session.repository import load_session, save_session
from app.session.session import Session

load_dotenv()

TWILIO_MULAW_FRAME_BYTES = 160
TWILIO_FRAME_DURATION_SECONDS = 0.02

TWILIO_BURST_FRAMES = int(float(os.getenv("TWILIO_BURST_FRAMES", "5")))
if TWILIO_BURST_FRAMES <= 0:
    TWILIO_BURST_FRAMES = 5

TWILIO_BURST_BYTES = int(TWILIO_MULAW_FRAME_BYTES * TWILIO_BURST_FRAMES)
TWILIO_BURST_PACING_SECONDS = TWILIO_FRAME_DURATION_SECONDS * TWILIO_BURST_FRAMES

WELCOME_AUDIO_WAV_PATH = os.getenv(
    "COMPASS_WELCOME_AUDIO_WAV_PATH",
    # "app/static/audio/compass_welcome_ulaw_8k_0.9x.wav",
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
    connect = Connect()
    connect.stream(url=stream_url, name="compass-voice-stream")
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

    async def _end_live_call() -> None:
        if not stream_session.call_sid:
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
            await asyncio.to_thread(
                twilio_client.calls(stream_session.call_sid).update,
                status="completed",
            )
            print(
                "[TWILIO CALL END REQUESTED]",
                {
                    "call_sid": stream_session.call_sid,
                    "stream_sid": stream_session.stream_sid,
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
            frame = burst_bytes[offset: offset + TWILIO_MULAW_FRAME_BYTES]
            if not frame:
                continue
            await send_twilio_media(frame, trace=trace)
            sent += len(frame)

        # pace one burst = 100 ms when TWILIO_BURST_FRAMES = 5
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

    def _compact_spoken_response(response_key: str, text: str) -> str:
        compact_map = {
            "item_added_successfully": "Added. Anything else?",
            "payment_link_sent": "Payment link sent. Tell me when done.",
            "order_completed": "Payment confirmed. Your order is placed. Ready in 25 minutes. Thank you.",
            "show_cart": text,
            "show_total": text,
            "confirm_clear_cart": "Clear your cart? Yes or no.",
            "cart_cleared": "Cart cleared.",
            "clear_cart_cancelled": "Okay, keeping your cart.",
        }
        return _normalize_response_text(compact_map.get(response_key, text))

    def _build_response_texts(
            turn_output: Any,
    ) -> tuple[str, str]:
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

        response_key = turn_output.response_key

        if response_key in DYNAMIC_RESPONSE_KEYS:
            spoken_text = internal_text
        else:
            spoken_text = getattr(turn_output, "spoken_response_text", None)
            if not spoken_text:
                spoken_text = _compact_spoken_response(
                    response_key,
                    internal_text,
                )

        return (
            _normalize_response_text(internal_text),
            _normalize_response_text(spoken_text),
        )

    async def speak_response_text(
        spoken_text: str,
        trace: RealtimeTurnTrace | None = None,
    ) -> None:
        nonlocal phase, active_mark_name, mark_counter, bot_playback_started_at, playback_generation

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

            if trace is not None and getattr(trace, "response_key", None) == "order_completed":
                await _end_live_call()

            return

        if trace is not None:
            _trace_set_attr(
                trace,
                "outbound_audio_duration_ms",
                round((streamed_bytes / 8000.0) * 1000.0, 3),
            )

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

    async def process_committed_turn(user_text: str) -> None:
        nonlocal phase, pending_interrupt_text, app_session, should_end_call_after_playback

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

            _debug_log(
                "[COMMITTED USER TURN]",
                {
                    "stream_sid": stream_session.stream_sid,
                    "text": cleaned,
                    "turn_index": stream_session.turn_index,
                },
            )

            turn_output = app.state.engine.process_turn(
                session=app_session,
                user_text=cleaned,
                trace=trace,
            )

            save_session(app_session)

            _trace_set_attr(trace, "responder_start_monotonic", time.perf_counter())
            internal_response_text, spoken_response_text = _build_response_texts(
                turn_output=turn_output,
            )
            _trace_set_attr(trace, "responder_end_monotonic", time.perf_counter())
            _trace_set_attr(trace, "response_key", turn_output.response_key)
            _trace_set_attr(trace, "response_text", internal_response_text)
            _trace_set_attr(trace, "spoken_response_text", spoken_response_text)

            should_end_call_after_playback = turn_output.response_key == "order_completed"

            _debug_log(
                "[BOT RESPONSE TEXT]",
                {
                    "stream_sid": stream_session.stream_sid,
                    "internal_text": internal_response_text,
                    "spoken_text": spoken_response_text,
                    "end_call_after_playback": should_end_call_after_playback,
                },
            )

            await speak_response_text(spoken_response_text, trace=trace)

            if pending_interrupt_text and phase == RealtimePhase.LISTENING:
                buffered = pending_interrupt_text
                pending_interrupt_text = None
                await process_committed_turn(buffered)

    async def on_dg_transcript(transcript: str, is_final: bool, payload: dict) -> None:
        committed = controller.on_transcript(transcript, is_final)

        _debug_transcript_log(
            "[DEEPGRAM TRANSCRIPT]",
            {
                "stream_sid": stream_session.stream_sid,
                "type": "FINAL" if is_final else "INTERIM",
                "text": transcript,
            },
        )

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

        _debug_log(
            "[DEEPGRAM EVENT]",
            {
                "stream_sid": stream_session.stream_sid,
                "event": name,
                "payload": payload,
            },
        )

        if name == "message:SpeechStarted":
            controller.on_speech_started()
            now = time.perf_counter()
            stream_session.last_dg_speech_started_monotonic = now

            if stream_session.current_utterance_first_media_monotonic is None:
                stream_session.current_utterance_first_media_monotonic = now
            stream_session.current_utterance_last_media_monotonic = now

            if phase == RealtimePhase.SPEAKING:
                if disable_barge_in:
                    _debug_log(
                        "[BARGE_IN IGNORED]",
                        {
                            "stream_sid": stream_session.stream_sid,
                            "reason": "disabled",
                        },
                    )
                    return

                guard_now = time.monotonic()
                if (
                    bot_playback_started_at is not None
                    and guard_now - bot_playback_started_at < bot_barge_in_guard_seconds
                ):
                    _debug_log(
                        "[BARGE_IN IGNORED]",
                        {
                            "stream_sid": stream_session.stream_sid,
                            "reason": "guard_window",
                            "elapsed_ms": round(
                                (guard_now - bot_playback_started_at) * 1000.0,
                                1,
                            ),
                        },
                    )
                    return

                await _interrupt_bot_playback(reason="user_speech_started")

            elif phase == RealtimePhase.PROCESSING:
                _debug_log(
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

        if stream_session.active_trace is not None:
            _trace_add_note(
                trace=stream_session.active_trace,
                key="deepgram_error_event",
                value=name,
            )
            _trace_add_note(
                trace=stream_session.active_trace,
                key="deepgram_error_payload",
                value=payload,
            )

    async def _start_stt_client_in_background() -> bool:
        nonlocal dg_stt_started

        if dg_stt_client is None:
            return False

        try:
            await dg_stt_client.start()
            dg_stt_started = True
            print(
                "[DEEPGRAM STT CONNECTED]",
                {
                    "stream_sid": stream_session.stream_sid,
                },
            )
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
            print(
                "[DEEPGRAM TTS CONNECTED]",
                {
                    "stream_sid": stream_session.stream_sid,
                },
            )
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
            raw_message = await websocket.receive_json()
            event = raw_message.get("event")

            if event == "connected":
                _debug_log("[TWILIO WS CONNECTED]", raw_message)

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
                    try:
                        await speak_cached_welcome_audio()
                        welcome_sent = True
                    except Exception as exc:
                        print(
                            "[WELCOME STATIC AUDIO FAILED]",
                            {
                                "stream_sid": stream_session.stream_sid,
                                "error": f"{type(exc).__name__}: {exc}",
                            },
                        )
                        try:
                            await speak_response_text(
                                "Welcome to Compass. Is this for pickup or delivery today?"
                            )
                            welcome_sent = True
                        except Exception as fallback_exc:
                            print(
                                "[WELCOME TTS FAILED]",
                                {
                                    "stream_sid": stream_session.stream_sid,
                                    "error": f"{type(fallback_exc).__name__}: {fallback_exc}",
                                },
                            )
                            phase = RealtimePhase.LISTENING
                    finally:
                        disable_barge_in = False

                stt_ready, tts_ready = await asyncio.gather(
                    stt_connect_task,
                    tts_connect_task,
                )

                _debug_log(
                    "[VOICE STARTUP READY]",
                    {
                        "stream_sid": stream_session.stream_sid,
                        "stt_connected": stt_ready,
                        "tts_connected": tts_ready,
                        "welcome_sent": welcome_sent,
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

                now = time.perf_counter()
                if phase in {RealtimePhase.LISTENING, RealtimePhase.PROCESSING}:
                    if stream_session.current_utterance_first_media_monotonic is None:
                        stream_session.current_utterance_first_media_monotonic = now
                    stream_session.current_utterance_last_media_monotonic = now
                    stream_session.current_utterance_inbound_audio_bytes += len(audio_bytes)

                if dg_stt_started and dg_stt_client is not None:
                    await dg_stt_client.send_audio(audio_bytes)

                if stream_session.inbound_chunks % 50 == 0:
                    _debug_media_log(
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

                _debug_log(
                    "[TWILIO MARK RECEIVED]",
                    {
                        "stream_sid": stream_session.stream_sid,
                        "mark_name": mark_name,
                    },
                )

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

                    if should_end_call_after_playback:
                        should_end_call_after_playback = False
                        await _end_live_call()
                        continue

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

                if stream_session.active_trace is not None:
                    _trace_add_note(stream_session.active_trace, "stream_stop", stop)
                    _trace_add_note(
                        stream_session.active_trace,
                        "finalized_without_mark",
                        True,
                    )
                    _finalize_active_trace(app=app, stream_session=stream_session)

                if dg_stt_client is not None:
                    await dg_stt_client.finalize()

                break

            else:
                _debug_log("[TWILIO UNKNOWN EVENT]", raw_message)

    except WebSocketDisconnect:
        print(
            "[WS CLOSED]",
            {
                "call_sid": stream_session.call_sid,
                "stream_sid": stream_session.stream_sid,
                "chunks_received": stream_session.inbound_chunks,
            },
        )

        if stream_session.active_trace is not None:
            _trace_add_note(stream_session.active_trace, "websocket_disconnect", True)
            _trace_add_note(
                stream_session.active_trace,
                "finalized_without_mark",
                True,
            )
            _finalize_active_trace(app=app, stream_session=stream_session)

    except Exception as exc:
        print(
            "[WS ERROR]",
            {
                "call_sid": stream_session.call_sid,
                "stream_sid": stream_session.stream_sid,
                "error": str(exc),
            },
        )

        if stream_session.active_trace is not None:
            _trace_add_note(stream_session.active_trace, "websocket_error", str(exc))
            _trace_add_note(
                stream_session.active_trace,
                "finalized_without_mark",
                True,
            )
            _finalize_active_trace(app=app, stream_session=stream_session)

        try:
            await websocket.close()
        except Exception:
            pass
    finally:
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