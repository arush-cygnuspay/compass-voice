# app/realtime/deepgram_tts_client.py
from __future__ import annotations

import json
import os
from collections.abc import AsyncGenerator
from urllib.parse import urlencode

from dotenv import load_dotenv
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed


class DeepgramTTSClient:
    """
    Streaming Deepgram TTS client for Twilio Media Streams.

    Uses Deepgram's /v1/speak WebSocket so audio can be forwarded to Twilio
    incrementally as it is generated, instead of waiting for one full REST blob.
    """

    def __init__(self) -> None:
        load_dotenv()

        api_key = os.getenv("DEEPGRAM_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("DEEPGRAM_API_KEY is not set in environment or .env file.")

        self._api_key = api_key
        self._model = os.getenv("DEEPGRAM_TTS_MODEL", "aura-2-thalia-en").strip()
        self._encoding = "mulaw"
        self._sample_rate = 8000

    def _websocket_url(self) -> str:
        params = {
            "model": self._model,
            "encoding": self._encoding,
            "sample_rate": str(self._sample_rate),
        }
        return f"wss://api.deepgram.com/v1/speak?{urlencode(params)}"

    async def stream_mulaw_8k(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        Streams Deepgram audio chunks for a single utterance.

        Flow:
        1. open WS
        2. send Speak
        3. send Flush
        4. yield binary audio frames as they arrive
        5. stop when Flushed is received
        """
        cleaned = " ".join((text or "").split()).strip()
        if not cleaned:
            return

        ws_url = self._websocket_url()

        async with connect(
            ws_url,
            additional_headers={
                "Authorization": f"Token {self._api_key}",
            },
            max_size=None,
        ) as ws:
            await ws.send(json.dumps({"type": "Speak", "text": cleaned}))
            await ws.send(json.dumps({"type": "Flush"}))

            try:
                async for message in ws:
                    if isinstance(message, bytes):
                        if message:
                            yield message
                        continue

                    try:
                        payload = json.loads(message)
                    except json.JSONDecodeError:
                        continue

                    msg_type = payload.get("type", "")

                    if msg_type == "Warning":
                        print("[DEEPGRAM TTS WARNING]", payload)
                        continue

                    if msg_type == "Metadata":
                        print("[DEEPGRAM TTS METADATA]", payload)
                        continue

                    if msg_type == "Flushed":
                        break

                    if msg_type == "Cleared":
                        break

                    if msg_type == "Error":
                        raise RuntimeError(f"Deepgram TTS error: {payload}")

            except ConnectionClosed as exc:
                raise RuntimeError(
                    f"Deepgram TTS websocket closed unexpectedly: code={exc.code}, reason={exc.reason}"
                ) from exc
            finally:
                try:
                    await ws.send(json.dumps({"type": "Close"}))
                except Exception:
                    pass