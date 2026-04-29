# tests/realtime/test_deepgram_stt_client.py
"""Tests for DeepgramSTTClient._recv_loop and DeepgramEventParser.

Coverage:
- Fast-lane event routing (SpeechStarted, UtteranceEnd, Metadata, KeepAlive)
- Full-parse path for Results events (transcript + is_final)
- Full-parse path for Error events
- Malformed JSON handling
- Binary frame handling
- peek_event_type: bytes and str input, whitespace tolerance, 256-byte limit
- parse: valid JSON, invalid JSON raises ValueError
- Downstream semantics unchanged: on_transcript receives (text, is_final, payload)
"""
from __future__ import annotations

import asyncio
import json
import sys
import types
import unittest

# Stub dotenv so deepgram_stt_client can be imported without the package.
_dotenv_stub = types.ModuleType("dotenv")
_dotenv_stub.load_dotenv = lambda *a, **k: None
sys.modules.setdefault("dotenv", _dotenv_stub)

# Stub websockets so the import succeeds without the package in test env.
for _mod in ("websockets", "websockets.asyncio", "websockets.asyncio.client", "websockets.exceptions"):
    sys.modules.setdefault(_mod, types.ModuleType(_mod))
if not hasattr(sys.modules["websockets.asyncio.client"], "connect"):
    sys.modules["websockets.asyncio.client"].connect = None
if not hasattr(sys.modules["websockets.exceptions"], "ConnectionClosed"):
    sys.modules["websockets.exceptions"].ConnectionClosed = Exception

from app.realtime.deepgram_event_parser import DeepgramEventParser, TYPE_ONLY_EVENTS


# ---------------------------------------------------------------------------
# DeepgramEventParser unit tests
# ---------------------------------------------------------------------------

class PeekEventTypeTests(unittest.TestCase):
    def test_extracts_type_from_str(self):
        raw = '{"type":"Results","is_final":true}'
        self.assertEqual(DeepgramEventParser.peek_event_type(raw), "Results")

    def test_extracts_type_from_bytes(self):
        raw = b'{"type":"SpeechStarted","timestamp":1.0}'
        self.assertEqual(DeepgramEventParser.peek_event_type(raw), "SpeechStarted")

    def test_handles_whitespace_around_colon(self):
        raw = '{"type" : "UtteranceEnd"}'
        self.assertEqual(DeepgramEventParser.peek_event_type(raw), "UtteranceEnd")

    def test_returns_none_for_missing_type(self):
        raw = '{"channel":{"alternatives":[]}}'
        self.assertIsNone(DeepgramEventParser.peek_event_type(raw))

    def test_returns_none_for_empty_string(self):
        self.assertIsNone(DeepgramEventParser.peek_event_type(""))

    def test_returns_none_for_empty_bytes(self):
        self.assertIsNone(DeepgramEventParser.peek_event_type(b""))

    def test_ignores_type_beyond_256_bytes(self):
        # "type" field placed after 256-byte prefix should not be found
        prefix = "x" * 260
        raw = f'{{"padding":"{prefix}","type":"Results"}}'
        # peek only scans first 256 bytes of the encoded string
        result = DeepgramEventParser.peek_event_type(raw)
        # type is beyond scan window — should be None
        self.assertIsNone(result)

    def test_all_type_only_events_extractable(self):
        for event in TYPE_ONLY_EVENTS:
            raw = f'{{"type":"{event}","timestamp":0.0}}'
            self.assertEqual(DeepgramEventParser.peek_event_type(raw), event)

    def test_results_event_extractable(self):
        raw = json.dumps({"type": "Results", "channel": {"alternatives": []}})
        self.assertEqual(DeepgramEventParser.peek_event_type(raw), "Results")

    def test_error_event_extractable(self):
        raw = json.dumps({"type": "Error", "description": "bad request"})
        self.assertEqual(DeepgramEventParser.peek_event_type(raw), "Error")


class ParseTests(unittest.TestCase):
    def test_parses_valid_json_str(self):
        raw = '{"type":"Results","is_final":false}'
        result = DeepgramEventParser.parse(raw)
        self.assertEqual(result["type"], "Results")
        self.assertFalse(result["is_final"])

    def test_parses_valid_json_bytes(self):
        raw = b'{"type":"SpeechStarted"}'
        result = DeepgramEventParser.parse(raw)
        self.assertEqual(result["type"], "SpeechStarted")

    def test_raises_value_error_for_invalid_json(self):
        with self.assertRaises(ValueError):
            DeepgramEventParser.parse("not json at all")

    def test_raises_value_error_for_truncated_json(self):
        with self.assertRaises(ValueError):
            DeepgramEventParser.parse('{"type":"Results"')

    def test_parses_nested_results_payload(self):
        payload = {
            "type": "Results",
            "channel": {"alternatives": [{"transcript": "hello", "confidence": 0.99}]},
            "is_final": True,
        }
        raw = json.dumps(payload)
        result = DeepgramEventParser.parse(raw)
        self.assertEqual(result["channel"]["alternatives"][0]["transcript"], "hello")
        self.assertTrue(result["is_final"])


# ---------------------------------------------------------------------------
# _recv_loop behavioral tests
# ---------------------------------------------------------------------------

def _make_ws(frames: list):
    """Async iterator stub that yields frames then stops."""
    class _FakeWS:
        def __aiter__(self):
            return self._gen()
        async def _gen(self):
            for f in frames:
                yield f
    return _FakeWS()


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


class RecvLoopTests(unittest.TestCase):
    """Drive _recv_loop through its frame-handling paths via a fake WebSocket."""

    def _build_client(self):
        from app.realtime.deepgram_stt_client import DeepgramSTTClient, DeepgramSTTCallbacks, DeepgramSTTConfig
        callbacks = DeepgramSTTCallbacks()
        client = DeepgramSTTClient(config=DeepgramSTTConfig(), callbacks=callbacks)
        return client, callbacks

    # --- binary frames --------------------------------------------------

    def test_binary_frame_emits_binary_message_event(self):
        client, callbacks = self._build_client()
        events: list[tuple[str, dict]] = []
        async def on_event(name, payload): events.append((name, payload))
        callbacks.on_event = on_event
        client._ws = _make_ws([b"\x00\x01\x02"])
        _run(client._recv_loop())
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0][0], "binary_message")
        self.assertEqual(events[0][1]["size"], 3)

    # --- fast-lane type-only events ------------------------------------

    def test_speech_started_fast_lane_no_transcript(self):
        client, callbacks = self._build_client()
        events: list[tuple[str, dict]] = []
        transcripts: list = []
        async def on_event(name, payload): events.append((name, payload))
        async def on_transcript(text, is_final, payload): transcripts.append(text)
        callbacks.on_event = on_event
        callbacks.on_transcript = on_transcript
        raw = json.dumps({"type": "SpeechStarted", "timestamp": 1.5})
        client._ws = _make_ws([raw])
        _run(client._recv_loop())
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0][0], "message:SpeechStarted")
        self.assertEqual(events[0][1], {"type": "SpeechStarted"})
        self.assertEqual(transcripts, [])  # no transcript triggered

    def test_utterance_end_fast_lane(self):
        client, callbacks = self._build_client()
        events: list[tuple[str, dict]] = []
        async def on_event(name, payload): events.append((name, payload))
        callbacks.on_event = on_event
        raw = json.dumps({"type": "UtteranceEnd", "last_word_end": 3.2})
        client._ws = _make_ws([raw])
        _run(client._recv_loop())
        self.assertEqual(events[0][0], "message:UtteranceEnd")

    def test_metadata_fast_lane(self):
        client, callbacks = self._build_client()
        events: list[tuple[str, dict]] = []
        async def on_event(name, payload): events.append((name, payload))
        callbacks.on_event = on_event
        raw = json.dumps({"type": "Metadata", "request_id": "abc-123"})
        client._ws = _make_ws([raw])
        _run(client._recv_loop())
        self.assertEqual(events[0][0], "message:Metadata")

    def test_keepalive_fast_lane(self):
        client, callbacks = self._build_client()
        events: list[tuple[str, dict]] = []
        async def on_event(name, payload): events.append((name, payload))
        callbacks.on_event = on_event
        raw = json.dumps({"type": "KeepAlive"})
        client._ws = _make_ws([raw])
        _run(client._recv_loop())
        self.assertEqual(events[0][0], "message:KeepAlive")

    # --- Results (full parse required) ---------------------------------

    def test_results_event_calls_on_transcript(self):
        client, callbacks = self._build_client()
        transcripts: list[tuple[str, bool, dict]] = []
        events: list[tuple[str, dict]] = []
        async def on_transcript(text, is_final, payload):
            transcripts.append((text, is_final, payload))
        async def on_event(name, payload): events.append((name, payload))
        callbacks.on_transcript = on_transcript
        callbacks.on_event = on_event
        payload = {
            "type": "Results",
            "channel": {"alternatives": [{"transcript": "hello world", "confidence": 0.99}]},
            "is_final": True,
            "speech_final": False,
        }
        client._ws = _make_ws([json.dumps(payload)])
        _run(client._recv_loop())
        self.assertEqual(len(transcripts), 1)
        text, is_final, full_payload = transcripts[0]
        self.assertEqual(text, "hello world")
        self.assertTrue(is_final)
        self.assertIn("channel", full_payload)
        # on_event also fired for Results
        self.assertEqual(events[0][0], "message:Results")

    def test_results_with_empty_transcript_does_not_call_on_transcript(self):
        client, callbacks = self._build_client()
        transcripts: list = []
        async def on_transcript(text, is_final, payload): transcripts.append(text)
        callbacks.on_transcript = on_transcript
        payload = {
            "type": "Results",
            "channel": {"alternatives": [{"transcript": "", "confidence": 0.0}]},
            "is_final": False,
        }
        client._ws = _make_ws([json.dumps(payload)])
        _run(client._recv_loop())
        self.assertEqual(transcripts, [])

    def test_results_interim_is_final_false(self):
        client, callbacks = self._build_client()
        transcripts: list[tuple[str, bool, dict]] = []
        async def on_transcript(text, is_final, payload):
            transcripts.append((text, is_final, payload))
        callbacks.on_transcript = on_transcript
        payload = {
            "type": "Results",
            "channel": {"alternatives": [{"transcript": "hel", "confidence": 0.5}]},
            "is_final": False,
        }
        client._ws = _make_ws([json.dumps(payload)])
        _run(client._recv_loop())
        self.assertFalse(transcripts[0][1])

    # --- Error events --------------------------------------------------

    def test_error_event_calls_on_error_not_on_transcript(self):
        client, callbacks = self._build_client()
        errors: list[tuple[str, dict]] = []
        transcripts: list = []
        async def on_error(name, payload): errors.append((name, payload))
        async def on_transcript(text, is_final, payload): transcripts.append(text)
        callbacks.on_error = on_error
        callbacks.on_transcript = on_transcript
        raw = json.dumps({"type": "Error", "description": "invalid credentials", "variant": "TOKEN"})
        client._ws = _make_ws([raw])
        _run(client._recv_loop())
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0][0], "error")
        self.assertEqual(transcripts, [])

    # --- Malformed JSON ------------------------------------------------

    def test_malformed_json_emits_text_message_event(self):
        client, callbacks = self._build_client()
        events: list[tuple[str, dict]] = []
        async def on_event(name, payload): events.append((name, payload))
        callbacks.on_event = on_event
        client._ws = _make_ws(["not json {{"])
        _run(client._recv_loop())
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0][0], "text_message")
        self.assertEqual(events[0][1]["raw"], "not json {{")

    def test_truncated_json_treated_as_malformed(self):
        client, callbacks = self._build_client()
        events: list[tuple[str, dict]] = []
        async def on_event(name, payload): events.append((name, payload))
        callbacks.on_event = on_event
        client._ws = _make_ws(['{"type":"Results"'])
        _run(client._recv_loop())
        self.assertEqual(events[0][0], "text_message")

    # --- Multiple frames in sequence ----------------------------------

    def test_multiple_frames_processed_independently(self):
        client, callbacks = self._build_client()
        events: list[str] = []
        transcripts: list[str] = []
        async def on_event(name, payload): events.append(name)
        async def on_transcript(text, is_final, payload): transcripts.append(text)
        callbacks.on_event = on_event
        callbacks.on_transcript = on_transcript
        frames = [
            json.dumps({"type": "SpeechStarted"}),
            json.dumps({
                "type": "Results",
                "channel": {"alternatives": [{"transcript": "order a burger"}]},
                "is_final": True,
            }),
            json.dumps({"type": "UtteranceEnd"}),
        ]
        client._ws = _make_ws(frames)
        _run(client._recv_loop())
        self.assertEqual(events, [
            "message:SpeechStarted",
            "message:Results",
            "message:UtteranceEnd",
        ])
        self.assertEqual(transcripts, ["order a burger"])

    # --- No callbacks set (no AttributeError) -------------------------

    def test_recv_loop_stable_with_no_callbacks(self):
        client, callbacks = self._build_client()
        frames = [
            json.dumps({"type": "SpeechStarted"}),
            json.dumps({
                "type": "Results",
                "channel": {"alternatives": [{"transcript": "hi"}]},
                "is_final": True,
            }),
        ]
        client._ws = _make_ws(frames)
        _run(client._recv_loop())  # must not raise


class TypeOnlyEventsMembershipTests(unittest.TestCase):
    """Guard: ensure expected events are in TYPE_ONLY_EVENTS."""

    def test_speech_started_in_set(self):
        self.assertIn("SpeechStarted", TYPE_ONLY_EVENTS)

    def test_utterance_end_in_set(self):
        self.assertIn("UtteranceEnd", TYPE_ONLY_EVENTS)

    def test_metadata_in_set(self):
        self.assertIn("Metadata", TYPE_ONLY_EVENTS)

    def test_results_not_in_set(self):
        self.assertNotIn("Results", TYPE_ONLY_EVENTS)

    def test_error_not_in_set(self):
        self.assertNotIn("Error", TYPE_ONLY_EVENTS)


if __name__ == "__main__":
    unittest.main()
