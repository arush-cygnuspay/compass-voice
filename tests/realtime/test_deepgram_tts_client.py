import asyncio
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

dotenv_module = types.ModuleType("dotenv")
dotenv_module.load_dotenv = lambda *args, **kwargs: None
sys.modules.setdefault("dotenv", dotenv_module)

websockets_module = types.ModuleType("websockets")
websockets_asyncio_module = types.ModuleType("websockets.asyncio")
websockets_asyncio_client_module = types.ModuleType("websockets.asyncio.client")
websockets_exceptions_module = types.ModuleType("websockets.exceptions")


async def _unused_connect(*args, **kwargs):
    raise AssertionError("connect() should not be called in this unit test")


class _ConnectionClosed(Exception):
    def __init__(self, code: int = 1000, reason: str = "") -> None:
        super().__init__(reason)
        self.code = code
        self.reason = reason


websockets_asyncio_client_module.connect = _unused_connect
websockets_exceptions_module.ConnectionClosed = _ConnectionClosed

sys.modules.setdefault("websockets", websockets_module)
sys.modules.setdefault("websockets.asyncio", websockets_asyncio_module)
sys.modules.setdefault("websockets.asyncio.client", websockets_asyncio_client_module)
sys.modules.setdefault("websockets.exceptions", websockets_exceptions_module)

from app.realtime.deepgram_tts_client import DeepgramTTSClient


def test_iter_audio_until_flushed_leaves_cleared_ack_for_next_utterance(
    monkeypatch,
) -> None:
    monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key")

    async def scenario() -> None:
        client = DeepgramTTSClient()
        client._utterance_open = True
        client._clear_requested = True
        client._clear_event.set()
        await client._event_queue.put({"type": "Cleared", "sequence_id": 0})

        chunks = [chunk async for chunk in client.iter_audio_until_flushed()]

        assert chunks == []
        assert client._event_queue.qsize() == 1

    asyncio.run(scenario())


def test_begin_utterance_drains_stale_clear_state(monkeypatch) -> None:
    monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key")

    async def scenario() -> None:
        client = DeepgramTTSClient()

        async def fake_connect() -> None:
            client._connected = True

        client.connect = fake_connect  # type: ignore[method-assign]
        client._clear_requested = True
        client._clear_event.set()
        await client._audio_queue.put(b"stale-audio")
        await client._event_queue.put({"type": "Cleared", "sequence_id": 0})

        await client.begin_utterance()

        assert client._utterance_open is True
        assert client._clear_requested is False
        assert client._clear_event.is_set() is False
        assert client._audio_queue.empty()
        assert client._event_queue.empty()

    asyncio.run(scenario())


def test_begin_utterance_waits_for_late_cleared_ack_after_stale_flushed(
    monkeypatch,
) -> None:
    monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key")

    async def scenario() -> None:
        client = DeepgramTTSClient()

        async def fake_connect() -> None:
            client._connected = True

        client.connect = fake_connect  # type: ignore[method-assign]
        client._clear_requested = True
        client._clear_event.set()
        await client._event_queue.put({"type": "Flushed", "sequence_id": 0})

        async def push_cleared() -> None:
            await asyncio.sleep(0.01)
            await client._event_queue.put({"type": "Cleared", "sequence_id": 1})

        task = asyncio.create_task(push_cleared())
        await client.begin_utterance()
        await task

        assert client._utterance_open is True
        assert client._clear_requested is False
        assert client._clear_event.is_set() is False
        assert client._event_queue.empty()

    asyncio.run(scenario())
