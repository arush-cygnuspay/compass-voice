# app/realtime/deepgram_stt_client.py
from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional
from urllib.parse import urlencode

from dotenv import load_dotenv
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed


TranscriptCallback = Callable[[str, bool, dict], Awaitable[None]]
EventCallback = Callable[[str, dict], Awaitable[None]]


@dataclass(slots=True)
class DeepgramSTTConfig:
    model: str = "nova-3"
    language: str = "en-US"
    encoding: str = "mulaw"
    sample_rate: int = 8000
    channels: int = 1
    interim_results: bool = True
    smart_format: bool = True
    punctuate: bool = True
    vad_events: bool = True
    endpointing: int = 300
    utterance_end_ms: int = 1000
    keepalive_interval_seconds: float = 4.0

    def websocket_url(self) -> str:
        params = {
            "model": self.model,
            "language": self.language,
            "encoding": self.encoding,
            "sample_rate": str(self.sample_rate),
            "channels": str(self.channels),
            "interim_results": str(self.interim_results).lower(),
            "smart_format": str(self.smart_format).lower(),
            "punctuate": str(self.punctuate).lower(),
            "vad_events": str(self.vad_events).lower(),
            "endpointing": str(self.endpointing),
            "utterance_end_ms": str(self.utterance_end_ms),
        }
        return f"wss://api.deepgram.com/v1/listen?{urlencode(params)}"


@dataclass(slots=True)
class DeepgramSTTCallbacks:
    on_transcript: Optional[TranscriptCallback] = None
    on_event: Optional[EventCallback] = None
    on_error: Optional[EventCallback] = None


class DeepgramSTTClient:
    """
    Direct WebSocket client for Deepgram Nova streaming STT.

    Why direct WS here:
    - avoids SDK version/signature issues
    - uses documented /v1/listen query params directly
    - easier to debug 400 errors and raw server responses
    """

    def __init__(
        self,
        config: DeepgramSTTConfig,
        callbacks: DeepgramSTTCallbacks,
    ) -> None:
        self.config = config
        self.callbacks = callbacks
        self._ws = None
        self._recv_task: asyncio.Task | None = None
        self._keepalive_task: asyncio.Task | None = None
        self._started = False
        self._closed = False
        self._last_audio_at = time.monotonic()

    async def start(self) -> None:
        if self._started:
            return

        load_dotenv()
        api_key = os.getenv("DEEPGRAM_API_KEY", "416dca8bd948ca8dcb5d81a5cb6b52d160cfd4bf").strip()
        if not api_key:
            raise RuntimeError("DEEPGRAM_API_KEY is not set in environment or .env file.")

        ws_url = self.config.websocket_url()

        self._ws = await connect(
            ws_url,
            additional_headers={
                "Authorization": f"Token {api_key}",
            },
            max_size=None,
        )

        self._started = True
        self._closed = False
        self._last_audio_at = time.monotonic()

        self._recv_task = asyncio.create_task(self._recv_loop(), name="deepgram-recv")
        self._keepalive_task = asyncio.create_task(
            self._keepalive_loop(),
            name="deepgram-keepalive",
        )

    async def _emit_event(self, name: str, payload: dict) -> None:
        if self.callbacks.on_event is not None:
            await self.callbacks.on_event(name, payload)

    async def _emit_error(self, name: str, payload: dict) -> None:
        if self.callbacks.on_error is not None:
            await self.callbacks.on_error(name, payload)

    async def _recv_loop(self) -> None:
        assert self._ws is not None

        try:
            async for raw_message in self._ws:
                if isinstance(raw_message, bytes):
                    await self._emit_event("binary_message", {"size": len(raw_message)})
                    continue

                try:
                    payload = json.loads(raw_message)
                except json.JSONDecodeError:
                    await self._emit_event("text_message", {"raw": raw_message})
                    continue

                msg_type = payload.get("type", "Unknown")
                await self._emit_event(f"message:{msg_type}", payload)

                if msg_type == "Error":
                    await self._emit_error("error", payload)
                    continue

                transcript = ""
                is_final = False

                # Nova /v1/listen Results shape
                if msg_type == "Results":
                    channel = payload.get("channel", {})
                    alternatives = channel.get("alternatives", [])
                    if alternatives:
                        transcript = (alternatives[0].get("transcript", "") or "").strip()
                    is_final = bool(payload.get("is_final", False))

                if transcript and self.callbacks.on_transcript is not None:
                    await self.callbacks.on_transcript(transcript, is_final, payload)

        except ConnectionClosed as exc:
            await self._emit_event(
                "closed",
                {
                    "code": exc.code,
                    "reason": exc.reason,
                },
            )
        except Exception as exc:
            await self._emit_error("recv_exception", {"error": str(exc)})

    async def _keepalive_loop(self) -> None:
        while not self._closed:
            await asyncio.sleep(self.config.keepalive_interval_seconds)

            if self._ws is None or self._closed:
                return

            idle_for = time.monotonic() - self._last_audio_at
            if idle_for >= self.config.keepalive_interval_seconds:
                try:
                    # Deepgram expects KeepAlive as a TEXT frame containing JSON.
                    await self._ws.send(json.dumps({"type": "KeepAlive"}))
                except Exception:
                    return

    async def send_audio(self, audio_bytes: bytes) -> None:
        if not audio_bytes or self._ws is None or self._closed:
            return

        await self._ws.send(audio_bytes)
        self._last_audio_at = time.monotonic()

    async def finalize(self) -> None:
        if self._ws is None or self._closed:
            return

        try:
            await self._ws.send(json.dumps({"type": "Finalize"}))
        except Exception:
            pass

    async def close(self) -> None:
        self._closed = True

        if self._keepalive_task is not None:
            self._keepalive_task.cancel()

        if self._recv_task is not None:
            self._recv_task.cancel()

        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            finally:
                self._ws = None