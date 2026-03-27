# app/api/voice_stream_server.py
from __future__ import annotations

import asyncio
import base64
import json
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
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
from app.realtime.local_stt_client import (
    LocalWhisperSTTConfig,
    LocalWhisperSTTEngine,
)
from app.realtime.realtime_conversation_state import RealtimePhase
from app.realtime.turn_commit_controller import TurnCommitController
from app.session.repository import load_session, save_session
from app.session.session import Session

# Always load .env before reading any provider/config variables.
load_dotenv()

# Twilio telephony audio is μ-law 8k mono.
# 20 ms at 8 kHz = 160 samples = 160 bytes.
TWILIO_MULAW_FRAME_BYTES = 160
TWILIO_FRAME_DURATION_SECONDS = 0.02

# Send a small burst instead of sleeping after every single frame.
# 5 frames = 100 ms of audio.
TWILIO_BURST_FRAMES = 5
TWILIO_BURST_BYTES = TWILIO_MULAW_FRAME_BYTES * TWILIO_BURST_FRAMES
TWILIO_BURST_PACING_SECONDS = TWILIO_FRAME_DURATION_SECONDS * TWILIO_BURST_FRAMES

VOICE_DEBUG_ENABLED = os.getenv("COMPASS_VOICE_DEBUG_ENABLED", "0") == "1"
VOICE_TRANSCRIPT_DEBUG_ENABLED = (
    os.getenv("COMPASS_VOICE_TRANSCRIPT_DEBUG_ENABLED", "0") == "1"
)
VOICE_MEDIA_PROGRESS_DEBUG_ENABLED = (
    os.getenv("COMPASS_VOICE_MEDIA_PROGRESS_DEBUG_ENABLED", "0") == "1"
)

STT_PROVIDER = os.getenv("COMPASS_STT_PROVIDER", "deepgram").strip().lower()
TTS_PROVIDER = os.getenv("COMPASS_TTS_PROVIDER", "deepgram").strip().lower()


def _mask_secret(value: str | None, *, visible_prefix: int = 4, visible_suffix: int = 2) -> str:
    if not value:
        return ""
    value = str(value)
    if len(value) <= visible_prefix + visible_suffix:
        return "*" * len(value)
    return f"{value[:visible_prefix]}{'*' * (len(value) - visible_prefix - visible_suffix)}{value[-visible_suffix:]}"


def _print_loaded_env_config() -> None:
    print(
        "[VOICE CONFIG LOADED]",
        {
            "COMPASS_STT_PROVIDER": STT_PROVIDER,
            "COMPASS_TTS_PROVIDER": TTS_PROVIDER,
            "COMPASS_LOCAL_STT_MODEL": os.getenv("COMPASS_LOCAL_STT_MODEL", ""),
            "COMPASS_LOCAL_STT_DEVICE": os.getenv("COMPASS_LOCAL_STT_DEVICE", ""),
            "COMPASS_LOCAL_STT_COMPUTE_TYPE": os.getenv("COMPASS_LOCAL_STT_COMPUTE_TYPE", ""),
            "COMPASS_LOCAL_STT_LANGUAGE": os.getenv("COMPASS_LOCAL_STT_LANGUAGE", ""),
            "COMPASS_LOCAL_STT_SAMPLE_RATE": os.getenv("COMPASS_LOCAL_STT_SAMPLE_RATE", ""),
            "COMPASS_LOCAL_STT_VAD_ENABLED": os.getenv("COMPASS_LOCAL_STT_VAD_ENABLED", ""),
            "COMPASS_LOCAL_STT_ENDPOINTING_MS": os.getenv("COMPASS_LOCAL_STT_ENDPOINTING_MS", ""),
            "COMPASS_LOCAL_STT_UTTERANCE_END_MS": os.getenv("COMPASS_LOCAL_STT_UTTERANCE_END_MS", ""),
            "COMPASS_LOCAL_TTS_MODEL_PATH": os.getenv("COMPASS_LOCAL_TTS_MODEL_PATH", ""),
            "COMPASS_LOCAL_TTS_CONFIG_PATH": os.getenv("COMPASS_LOCAL_TTS_CONFIG_PATH", ""),
            "COMPASS_LOCAL_TTS_SPEAKER_ID": os.getenv("COMPASS_LOCAL_TTS_SPEAKER_ID", ""),
            "COMPASS_LOCAL_TTS_TARGET_SAMPLE_RATE": os.getenv("COMPASS_LOCAL_TTS_TARGET_SAMPLE_RATE", ""),
            "COMPASS_NLU_CSV_LOGGER_ENABLED": os.getenv("COMPASS_NLU_CSV_LOGGER_ENABLED", ""),
            "COMPASS_NLU_CSV_LOG_DIR": os.getenv("COMPASS_NLU_CSV_LOG_DIR", ""),
            "COMPASS_REALTIME_LATENCY_LOG_PATH": os.getenv(
                "COMPASS_REALTIME_LATENCY_LOG_PATH",
                "app/logs/realtime_turn_latency.jsonl",
            ),
            "COMPASS_REALTIME_LATENCY_CSV_PATH": os.getenv(
                "COMPASS_REALTIME_LATENCY_CSV_PATH",
                "app/logs/realtime_turn_latency.csv",
            ),
            "DEEPGRAM_TTS_MODEL": os.getenv("DEEPGRAM_TTS_MODEL", ""),
            "DEEPGRAM_API_KEY": _mask_secret(os.getenv("DEEPGRAM_API_KEY", "")),
        },
    )


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


def _trace_set_attr(trace: RealtimeTurnTrace | None, attr_name: str, value: Any) -> None:
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
        notes={
            "stt_provider": STT_PROVIDER,
            "tts_provider": TTS_PROVIDER,
        },
    )


def _finalize_active_trace(
    *,
    app: FastAPI,
    stream_session: StreamSession,
    extra_notes: dict[str, Any] | None = None,
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
        _reset_utterance_tracking(stream_session)


def _build_stt_client(app: FastAPI, callbacks: DeepgramSTTCallbacks) -> Any:
    provider = STT_PROVIDER

    if provider == "deepgram":
        return DeepgramSTTClient(
            config=DeepgramSTTConfig(
                model=os.getenv("COMPASS_DEEPGRAM_STT_MODEL", "nova-3"),
                language=os.getenv("COMPASS_DEEPGRAM_STT_LANGUAGE", "en-US"),
                encoding="mulaw",
                sample_rate=8000,
                channels=1,
                interim_results=True,
                smart_format=True,
                punctuate=True,
                vad_events=True,
                endpointing=int(os.getenv("COMPASS_DEEPGRAM_ENDPOINTING_MS", "300")),
                utterance_end_ms=int(os.getenv("COMPASS_DEEPGRAM_UTTERANCE_END_MS", "1000")),
                keepalive_interval_seconds=float(
                    os.getenv("COMPASS_DEEPGRAM_KEEPALIVE_SECONDS", "4.0")
                ),
            ),
            callbacks=callbacks,
        )

    if provider == "local":
        local_stt_engine = getattr(app.state, "local_stt_engine", None)
        if local_stt_engine is None:
            raise RuntimeError("Shared local STT engine is not initialized.")

        # IMPORTANT: session comes from shared model
        return local_stt_engine.create_session(callbacks=callbacks)

    raise ValueError(
        f"Unsupported COMPASS_STT_PROVIDER='{provider}'. Supported values: deepgram, local."
    )


def _build_tts_client(app: FastAPI) -> Any:
    provider = TTS_PROVIDER

    if provider == "deepgram":
        shared_tts_client = getattr(app.state, "shared_tts_client", None)
        if shared_tts_client is None:
            raise RuntimeError("Shared Deepgram TTS client is not initialized.")
        return shared_tts_client

    if provider == "local":
        shared_tts_client = getattr(app.state, "shared_tts_client", None)
        if shared_tts_client is None:
            raise RuntimeError("Shared local TTS client is not initialized.")
        return shared_tts_client

    raise ValueError(
        f"Unsupported COMPASS_TTS_PROVIDER='{provider}'. "
        "Supported values: deepgram, local."
    )


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

    app.state.local_stt_engine = None
    app.state.shared_tts_client = None

    if STT_PROVIDER == "local":
        from app.realtime.local_stt_client import (
            LocalWhisperSTTConfig,
            LocalWhisperSTTEngine,
        )

        local_stt_config = LocalWhisperSTTConfig(
            model_name=os.getenv("COMPASS_LOCAL_STT_MODEL", "medium.en"),
            device=os.getenv("COMPASS_LOCAL_STT_DEVICE", "cuda"),
            compute_type=os.getenv("COMPASS_LOCAL_STT_COMPUTE_TYPE", "float16"),
            language=os.getenv("COMPASS_LOCAL_STT_LANGUAGE", "en"),
            sample_rate=int(os.getenv("COMPASS_LOCAL_STT_SAMPLE_RATE", "16000")),
            vad_enabled=os.getenv("COMPASS_LOCAL_STT_VAD_ENABLED", "1") == "1",
            endpointing_ms=int(os.getenv("COMPASS_LOCAL_STT_ENDPOINTING_MS", "300")),
            utterance_end_ms=int(os.getenv("COMPASS_LOCAL_STT_UTTERANCE_END_MS", "1000")),
            rms_threshold=int(os.getenv("COMPASS_LOCAL_STT_RMS_THRESHOLD", "400")),
            min_utterance_ms=int(os.getenv("COMPASS_LOCAL_STT_MIN_UTTERANCE_MS", "250")),
            speech_start_min_ms=int(os.getenv("COMPASS_LOCAL_STT_SPEECH_START_MIN_MS", "180")),
        )
        app.state.local_stt_engine = LocalWhisperSTTEngine(local_stt_config)
        app.state.local_stt_engine.load()
        print("[LOCAL STT] shared model loaded")

    if TTS_PROVIDER == "deepgram":
        app.state.shared_tts_client = DeepgramTTSClient()
        print("[DEEPGRAM TTS] shared client initialized")

    elif TTS_PROVIDER == "local":
        from app.realtime.local_tts_client import LocalPiperTTSClient

        app.state.shared_tts_client = LocalPiperTTSClient(
            model_path=os.getenv("COMPASS_LOCAL_TTS_MODEL_PATH", "").strip(),
            config_path=os.getenv("COMPASS_LOCAL_TTS_CONFIG_PATH", "").strip(),
            speaker_id=int(os.getenv("COMPASS_LOCAL_TTS_SPEAKER_ID", "0")),
            target_sample_rate=int(
                os.getenv("COMPASS_LOCAL_TTS_TARGET_SAMPLE_RATE", "8000")
            ),
        )
        print("[LOCAL TTS] shared client initialized")

    print(
        "Voice stream server initialized",
        {
            "stt_provider": STT_PROVIDER,
            "tts_provider": TTS_PROVIDER,
        },
    )

    try:
        yield
    finally:
        try:
            if getattr(app.state, "realtime_latency_logger", None) is not None:
                app.state.realtime_latency_logger.shutdown()
        except Exception:
            pass

        try:
            if getattr(runtime.engine, "nlu_logger", None) is not None:
                runtime.engine.nlu_logger.shutdown()
        except Exception:
            pass

        try:
            if getattr(app.state, "local_stt_engine", None) is not None:
                app.state.local_stt_engine.unload()
        except Exception:
            pass

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
            "stt_provider": STT_PROVIDER,
            "tts_provider": TTS_PROVIDER,
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
    stt_client: Any | None = None
    stt_started = False
    tts_client = _build_tts_client(app)

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
            "media": {
                "payload": payload_b64,
            },
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

        return sent

    async def stream_audio_to_twilio(
        audio_chunk_stream,
        trace: RealtimeTurnTrace | None = None,
    ) -> int:
        buffered = bytearray()
        total_bytes_sent = 0

        async for tts_chunk in audio_chunk_stream:
            if not tts_chunk:
                continue

            now = time.perf_counter()
            if trace is not None:
                if getattr(trace, "tts_first_chunk_monotonic", None) is None:
                    _trace_set_attr(trace, "tts_first_chunk_monotonic", now)
                _trace_set_attr(trace, "tts_last_chunk_monotonic", now)

            buffered.extend(tts_chunk)

            while len(buffered) >= TWILIO_BURST_BYTES:
                burst = bytes(buffered[:TWILIO_BURST_BYTES])
                del buffered[:TWILIO_BURST_BYTES]

                total_bytes_sent += await _send_burst_frames(burst, trace=trace)
                await asyncio.sleep(TWILIO_BURST_PACING_SECONDS)

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

    async def speak_response_text(
        response_text: str,
        trace: RealtimeTurnTrace | None = None,
    ) -> None:
        nonlocal phase, active_mark_name, mark_counter, bot_playback_started_at

        cleaned = " ".join((response_text or "").split()).strip()
        if not cleaned:
            phase = RealtimePhase.LISTENING
            return

        phase = RealtimePhase.SPEAKING
        bot_playback_started_at = time.monotonic()

        if trace is not None:
            _trace_set_attr(trace, "response_text", cleaned)
            _trace_set_attr(trace, "tts_text_chars", len(cleaned))
            _trace_set_attr(trace, "tts_request_start_monotonic", time.perf_counter())
            _trace_add_note(trace, "tts_provider", TTS_PROVIDER)

        streamed_bytes = await stream_audio_to_twilio(
            tts_client.stream_mulaw_8k(cleaned),
            trace=trace,
        )

        if streamed_bytes <= 0:
            print(
                "[TTS EMPTY AUDIO]",
                {
                    "stream_sid": stream_session.stream_sid,
                    "text": cleaned,
                    "tts_provider": TTS_PROVIDER,
                },
            )
            _trace_add_note(trace, "tts_empty_audio", True)
            phase = RealtimePhase.LISTENING
            bot_playback_started_at = None
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
                "tts_provider": TTS_PROVIDER,
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
                    "stt_provider": STT_PROVIDER,
                    "tts_provider": TTS_PROVIDER,
                },
            )

            turn_output = app.state.engine.process_turn(
                session=app_session,
                user_text=cleaned,
                trace=trace,
            )

            save_session(app_session)

            _trace_set_attr(trace, "responder_start_monotonic", time.perf_counter())
            response_text = app.state.responder.build(
                response_key=turn_output.response_key,
                context=app_session.conversation_context,
                payload=turn_output.response_payload,
            )
            _trace_set_attr(trace, "responder_end_monotonic", time.perf_counter())
            _trace_set_attr(trace, "response_key", turn_output.response_key)
            _trace_set_attr(trace, "response_text", response_text)

            _debug_log(
                "[BOT RESPONSE TEXT]",
                {
                    "stream_sid": stream_session.stream_sid,
                    "text": response_text,
                },
            )

            await speak_response_text(response_text, trace=trace)

            if pending_interrupt_text and phase == RealtimePhase.LISTENING:
                buffered = pending_interrupt_text
                pending_interrupt_text = None
                await process_committed_turn(buffered)

    async def on_stt_transcript(transcript: str, is_final: bool, payload: dict) -> None:
        controller.on_transcript(transcript, is_final)

        _debug_transcript_log(
            "[STT TRANSCRIPT]",
            {
                "stream_sid": stream_session.stream_sid,
                "provider": STT_PROVIDER,
                "type": "FINAL" if is_final else "INTERIM",
                "text": transcript,
            },
        )

        if is_final:
            stream_session.last_dg_final_transcript_monotonic = time.perf_counter()

        speech_final = bool(payload.get("speech_final", False))
        if speech_final:
            committed = controller.on_speech_final()
            if committed is not None:
                await process_committed_turn(committed.text)

    async def on_stt_event(name: str, payload: dict) -> None:
        nonlocal phase, active_mark_name, bot_playback_started_at

        _debug_log(
            "[STT EVENT]",
            {
                "stream_sid": stream_session.stream_sid,
                "provider": STT_PROVIDER,
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
                            "reason": "disabled_for_playback_verification",
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

                _debug_log(
                    "[BARGE_IN]",
                    {
                        "stream_sid": stream_session.stream_sid,
                        "active_mark_name": active_mark_name,
                    },
                )
                await send_twilio_clear()

                if stream_session.active_trace is not None:
                    _trace_add_note(stream_session.active_trace, "barge_in", True)
                    _trace_add_note(
                        stream_session.active_trace,
                        "barge_in_active_mark_name",
                        active_mark_name,
                    )

                active_mark_name = None
                stream_session.active_trace_mark_name = None
                bot_playback_started_at = None
                phase = RealtimePhase.LISTENING

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

    async def on_stt_error(name: str, payload: dict) -> None:
        print(
            "[STT ERROR]",
            {
                "stream_sid": stream_session.stream_sid,
                "provider": STT_PROVIDER,
                "event": name,
                "payload": payload,
            },
        )

        if stream_session.active_trace is not None:
            _trace_add_note(stream_session.active_trace, "stt_error_event", name)
            _trace_add_note(stream_session.active_trace, "stt_error_payload", payload)
            _trace_add_note(stream_session.active_trace, "stt_provider", STT_PROVIDER)

    print(
        "[WS OPEN] Twilio media WebSocket connected",
        {
            "stt_provider": STT_PROVIDER,
            "tts_provider": TTS_PROVIDER,
        },
    )

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
                        "stt_provider": STT_PROVIDER,
                        "tts_provider": TTS_PROVIDER,
                    },
                )

                if stream_session.call_sid:
                    app_session = load_session(stream_session.call_sid, restaurant_id)

                stt_callbacks = DeepgramSTTCallbacks(
                    on_transcript=on_stt_transcript,
                    on_event=on_stt_event,
                    on_error=on_stt_error,
                )

                stt_client = _build_stt_client(app, callbacks=stt_callbacks)

                await stt_client.start()
                stt_started = True

                print(
                    "[STT CONNECTED]",
                    {
                        "stream_sid": stream_session.stream_sid,
                        "stt_provider": STT_PROVIDER,
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

                if stt_started and stt_client is not None:
                    await stt_client.send_audio(audio_bytes)

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

                if stt_client is not None:
                    await stt_client.finalize()

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
        if stt_client is not None:
            await stt_client.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.api.voice_stream_server:app",
        host="0.0.0.0",
        port=5000,
        reload=False,
    )