# app/realtime/deepgram_tts_client.py
from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncGenerator
from typing import Any
from urllib.parse import urlencode

from dotenv import load_dotenv
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed


class DeepgramTTSClient:
    """
    Persistent Deepgram TTS websocket client.

    Barge-in guarantee (the hard part):
    ─────────────────────────────────────────────────────────────────────────
    When the user speaks over the bot we call clear(), which:
      1. Sets _clear_event  ← internal asyncio.Event
      2. Sends {"type":"Clear"} to Deepgram over the wire

    The running iter_audio_until_flushed() is watching _clear_event via
    a clear_task inside _wait_for_audio_or_event().  It exits the moment
    _clear_event fires — WITHOUT consuming the "Cleared" acknowledgment that
    Deepgram sends back.

    begin_utterance() for the new response then runs as the SOLE consumer of
    the Cleared ack.  It waits briefly for it to arrive and drains it along
    with any stale audio before starting fresh.

    The earlier approach of waiting in begin_utterance() WITHOUT _clear_event
    caused a fatal race: both the old iter AND begin_utterance() competed for
    the single Cleared item.  Whichever lost would spin on 1-second timeouts
    for up to 4 seconds, causing the repetition/latency bug.
    ─────────────────────────────────────────────────────────────────────────
    """

    def __init__(self) -> None:
        load_dotenv()

        api_key = os.getenv("DEEPGRAM_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("DEEPGRAM_API_KEY is not set.")

        self._api_key = api_key
        self._model = os.getenv("DEEPGRAM_TTS_MODEL", "aura-2-thalia-en").strip()
        self._encoding = "mulaw"
        self._sample_rate = 8000
        self._container = "none"

        self._ws: Any | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._write_lock = asyncio.Lock()

        self._connected = False
        self._closed = False

        self._audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._event_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

        self._utterance_open = False
        self._clear_requested = False

        # Fired by clear() so that iter_audio_until_flushed() exits immediately
        # without consuming the Deepgram "Cleared" ack from _event_queue.
        # begin_utterance() resets this after draining the ack.
        self._clear_event: asyncio.Event = asyncio.Event()

    def _websocket_url(self) -> str:
        params = {
            "model": self._model,
            "encoding": self._encoding,
            "sample_rate": str(self._sample_rate),
            "container": self._container,
        }
        return f"wss://api.deepgram.com/v1/speak?{urlencode(params)}"

    async def connect(self) -> None:
        if self._connected and self._ws is not None:
            return

        try:
            self._ws = await connect(
                self._websocket_url(),
                additional_headers={
                    "Authorization": f"Token {self._api_key}",
                },
                max_size=None,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to connect to Deepgram TTS websocket. "
                f"url={self._websocket_url()} error={type(exc).__name__}: {exc}"
            ) from exc

        self._connected = True
        self._closed = False
        self._reader_task = asyncio.create_task(
            self._reader_loop(),
            name="deepgram-tts-reader",
        )

    async def _reader_loop(self) -> None:
        assert self._ws is not None

        try:
            async for message in self._ws:
                if isinstance(message, bytes):
                    if message:
                        await self._audio_queue.put(message)
                    continue

                try:
                    payload = json.loads(message)
                except json.JSONDecodeError:
                    continue

                msg_type = str(payload.get("type", ""))

                if msg_type == "Warning":
                    print("[DEEPGRAM TTS WARNING]", payload)
                    continue

                if msg_type == "Metadata":
                    print("[DEEPGRAM TTS METADATA]", payload)
                    continue

                if msg_type == "Error":
                    await self._event_queue.put(payload)
                    continue

                if msg_type in {"Flushed", "Cleared"}:
                    await self._event_queue.put(payload)
                    continue

        except ConnectionClosed as exc:
            await self._event_queue.put(
                {
                    "type": "Error",
                    "error": (
                        "Deepgram TTS websocket closed unexpectedly: "
                        f"code={exc.code}, reason={exc.reason}"
                    ),
                }
            )
        except Exception as exc:
            await self._event_queue.put(
                {
                    "type": "Error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        finally:
            self._connected = False
            await self._audio_queue.put(None)

    async def _drain_queue_nowait(self, queue: asyncio.Queue[Any]) -> None:
        while not queue.empty():
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def _wait_for_cleared_ack(self, timeout: float = 0.35) -> None:
        """
        Wait until a Cleared (or Error) event arrives in _event_queue and discard it.

        Called by begin_utterance() when _clear_requested is True and the queue is
        currently empty, meaning the Cleared ack from Deepgram is still in-flight.
        Because iter_audio_until_flushed() exits via _clear_event (not by consuming
        Cleared), this is now the sole consumer — no race condition.

        Discards any non-Cleared events seen while waiting; they belong to the
        interrupted utterance.  Returns on timeout so the caller always makes progress.
        """
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                return
            try:
                event = await asyncio.wait_for(
                    self._event_queue.get(), timeout=remaining
                )
                if str(event.get("type", "")) in ("Cleared", "Error"):
                    return
                # Discard stale Flushed or other events from the prior utterance.
            except asyncio.TimeoutError:
                return

    async def begin_utterance(self) -> None:
        """
        Start a clean logical utterance on the persistent socket.

        After barge-in:
          - _clear_event is set → iter_audio_until_flushed() already exited (or is
            about to) without touching _event_queue.
          - _clear_requested is True → Deepgram will send a "Cleared" ack.
          - We wait briefly for that ack so drain_queue_nowait can remove it before
            the new iter_audio_until_flushed() starts.  If it never arrives within
            the deadline we proceed anyway (ack was consumed elsewhere or DG is slow).
          - _clear_event is reset so the new utterance starts with a clean signal.
        """
        await self.connect()

        if self._clear_requested:
            # Always wait briefly for the Deepgram Cleared ack after barge-in.
            # A stale Flushed event may already be queued while the Cleared ack is
            # still in-flight; skipping this wait lets that late Cleared poison the
            # new utterance and causes the classic zero-audio retry failure.
            await self._wait_for_cleared_ack(timeout=0.35)

        await self._drain_queue_nowait(self._audio_queue)
        await self._drain_queue_nowait(self._event_queue)

        # Reset the clear signal so the new utterance's iter won't exit immediately.
        self._clear_event.clear()
        self._utterance_open = True
        self._clear_requested = False

    async def send_text(self, text: str) -> None:
        cleaned = " ".join((text or "").split()).strip()
        if not cleaned:
            return

        if not self._utterance_open:
            await self.begin_utterance()

        async with self._write_lock:
            assert self._ws is not None
            try:
                await self._ws.send(json.dumps({"type": "Speak", "text": cleaned}))
            except Exception as exc:
                self._connected = False
                raise RuntimeError(
                    f"Failed to send Speak to Deepgram TTS: {type(exc).__name__}: {exc}"
                ) from exc

    async def flush(self) -> None:
        if not self._utterance_open:
            await self.begin_utterance()

        async with self._write_lock:
            assert self._ws is not None
            try:
                await self._ws.send(json.dumps({"type": "Flush"}))
            except Exception as exc:
                self._connected = False
                raise RuntimeError(
                    f"Failed to send Flush to Deepgram TTS: {type(exc).__name__}: {exc}"
                ) from exc

    async def clear(self) -> None:
        """
        Interrupt the current utterance immediately.

        Sets _clear_event first so that any running iter_audio_until_flushed()
        exits on its next _wait_for_audio_or_event() call, WITHOUT consuming the
        Deepgram "Cleared" ack from _event_queue.  begin_utterance() then drains
        that ack as sole consumer.
        """
        self._clear_requested = True
        self._clear_event.set()  # Wake up iter_audio_until_flushed() immediately

        if not self._connected or self._ws is None:
            self._utterance_open = False
            return

        async with self._write_lock:
            try:
                await self._ws.send(json.dumps({"type": "Clear"}))
            except Exception:
                self._connected = False
            finally:
                self._utterance_open = False

    async def _drain_ready_audio(self) -> AsyncGenerator[bytes, None]:
        while not self._audio_queue.empty():
            chunk = await self._audio_queue.get()
            if chunk is None:
                return
            yield chunk

    async def _get_next_event_nowait(self) -> dict[str, Any] | None:
        if self._event_queue.empty():
            return None
        return await self._event_queue.get()

    async def _wait_for_audio_or_event(
        self,
        timeout_seconds: float,
    ) -> tuple[str, bytes | dict[str, Any] | None]:
        """
        Wait for the next audio chunk, event, or barge-in signal.

        Returns one of:
          ("clear",   None)   — _clear_event fired; caller should exit immediately
          ("audio",   bytes)  — audio chunk from Deepgram
          ("event",   dict)   — Flushed / Cleared / Error JSON from Deepgram
          ("timeout", None)   — nothing arrived within timeout_seconds

        clear_task is checked FIRST so barge-in always takes priority over
        in-flight audio or events.
        """
        audio_task = asyncio.create_task(self._audio_queue.get())
        event_task = asyncio.create_task(self._event_queue.get())
        clear_task = asyncio.create_task(self._clear_event.wait())

        done, pending = await asyncio.wait(
            {audio_task, event_task, clear_task},
            timeout=timeout_seconds,
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            task.cancel()

        if not done:
            return "timeout", None

        # Priority: clear > audio > event
        if clear_task in done:
            return "clear", None
        if audio_task in done:
            return "audio", audio_task.result()
        return "event", event_task.result()

    async def iter_audio_until_flushed(self) -> AsyncGenerator[bytes, None]:
        """
        Yield audio for the current utterance until it is complete.

        Exits immediately when _clear_event fires (barge-in) without consuming
        the Deepgram "Cleared" ack — that is left for begin_utterance() to drain.

        Other exit conditions:
        - Flushed received  → short grace period to drain any trailing audio
        - Cleared received  → utterance was cancelled on the wire (fallback path)
        - Error received    → raises RuntimeError
        - WS closed         → audio_queue receives None sentinel
        """
        if not self._utterance_open and not self._clear_requested:
            return

        # Fast exit if barge-in already happened before we even started.
        if self._clear_event.is_set():
            self._utterance_open = False
            return

        flush_seen = False
        audio_emitted = False

        while True:
            async for ready_chunk in self._drain_ready_audio():
                audio_emitted = True
                yield ready_chunk

            event = await self._get_next_event_nowait()
            if event is not None:
                msg_type = str(event.get("type", ""))

                if msg_type == "Error":
                    self._utterance_open = False
                    raise RuntimeError(f"Deepgram TTS error: {event}")

                if msg_type == "Cleared":
                    self._utterance_open = False
                    return

                if msg_type == "Flushed":
                    flush_seen = True

            if flush_seen:
                grace_deadline = asyncio.get_running_loop().time() + 0.15
                while asyncio.get_running_loop().time() < grace_deadline:
                    kind, payload = await self._wait_for_audio_or_event(0.03)

                    if kind == "clear":
                        self._utterance_open = False
                        return

                    if kind == "audio":
                        if payload is None:
                            self._utterance_open = False
                            return
                        audio_emitted = True
                        yield payload  # type: ignore[misc]
                        grace_deadline = asyncio.get_running_loop().time() + 0.05
                        continue

                    if kind == "event" and isinstance(payload, dict):
                        msg_type = str(payload.get("type", ""))
                        if msg_type == "Error":
                            self._utterance_open = False
                            raise RuntimeError(f"Deepgram TTS error: {payload}")
                        if msg_type == "Cleared":
                            self._utterance_open = False
                            return

                self._utterance_open = False
                return

            kind, payload = await self._wait_for_audio_or_event(1.0)

            if kind == "clear":
                self._utterance_open = False
                return

            if kind == "timeout":
                if audio_emitted:
                    self._utterance_open = False
                    return
                continue

            if kind == "audio":
                if payload is None:
                    self._utterance_open = False
                    return
                audio_emitted = True
                yield payload  # type: ignore[misc]
                continue

            if kind == "event" and isinstance(payload, dict):
                msg_type = str(payload.get("type", ""))

                if msg_type == "Error":
                    self._utterance_open = False
                    raise RuntimeError(f"Deepgram TTS error: {payload}")

                if msg_type == "Cleared":
                    self._utterance_open = False
                    return

                if msg_type == "Flushed":
                    flush_seen = True
                    continue

    async def speak_once_progressive(
        self,
        chunks: list[str],
    ) -> AsyncGenerator[bytes, None]:
        await self.begin_utterance()

        for chunk in chunks:
            await self.send_text(chunk)

        await self.flush()

        async for audio in self.iter_audio_until_flushed():
            yield audio

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._utterance_open = False

        try:
            if self._ws is not None:
                async with self._write_lock:
                    await self._ws.send(json.dumps({"type": "Close"}))
        except Exception:
            pass

        try:
            if self._ws is not None:
                await self._ws.close()
        except Exception:
            pass

        try:
            if self._reader_task is not None:
                await self._reader_task
        except Exception:
            pass

        self._connected = False
        self._ws = None
        self._reader_task = None
