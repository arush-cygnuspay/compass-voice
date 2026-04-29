# app/realtime/deepgram_event_parser.py
"""Minimal, fast parser for Deepgram streaming WebSocket frames.

Two-tier strategy
-----------------
* peek_event_type  — regex scan on the first 256 bytes, extracts "type"
                     without a full JSON parse.  O(1) on Deepgram envelopes
                     because "type" is always the first key.
* parse            — full JSON decode via orjson (falls back to stdlib json
                     when orjson is not installed, e.g. in dev environments).

Events in _TYPE_ONLY_EVENTS are fully handled by their type name alone:
no downstream consumer inspects their payload, so full parsing is skipped
on the hot path.
"""
from __future__ import annotations

import re

try:
    import orjson as _orjson

    def _loads(data: bytes | str) -> dict:
        return _orjson.loads(data)

    # orjson raises orjson.JSONDecodeError which is a subclass of ValueError.
    _JSON_DECODE_ERRORS: tuple[type[Exception], ...] = (ValueError,)

except ModuleNotFoundError:  # pragma: no cover
    import json as _stdlib_json

    def _loads(data: bytes | str) -> dict:
        return _stdlib_json.loads(data)

    import json as _json_mod
    _JSON_DECODE_ERRORS = (_json_mod.JSONDecodeError,)


# Regex that extracts the "type" value from a Deepgram JSON envelope.
# Deepgram always emits "type" as the first key; searching only the first
# 256 bytes keeps this O(1) regardless of full message size.
_TYPE_PATTERN: re.Pattern[bytes] = re.compile(
    rb'"type"\s*:\s*"([^"]{1,64})"'
)

# Events whose entire semantic is captured by their type name.
# The voice_stream_server callbacks for these events never read the payload
# dict — they act solely on the event-name string passed to on_event.
# Skipping full JSON parse for these removes ~60 µs per frame on CPython.
TYPE_ONLY_EVENTS: frozenset[str] = frozenset({
    "SpeechStarted",
    "UtteranceEnd",
    "Metadata",
    "KeepAlive",
})


class DeepgramEventParser:
    """Stateless parser — no shared mutable state, safe for concurrent use."""

    @staticmethod
    def peek_event_type(raw: bytes | str) -> str | None:
        """Extract the Deepgram event type without a full JSON parse.

        Returns the ``"type"`` field value, or ``None`` when not found in the
        first 256 bytes (which covers every known Deepgram envelope format).
        """
        if isinstance(raw, str):
            # Encode only the prefix needed for the scan; avoid encoding the
            # entire (potentially large) Results payload.
            data: bytes = raw[:256].encode("utf-8", errors="replace")
        else:
            data = raw[:256]

        m = _TYPE_PATTERN.search(data)
        if m is None:
            return None
        return m.group(1).decode("utf-8", errors="replace")

    @staticmethod
    def parse(raw: bytes | str) -> dict:
        """Full JSON decode using orjson when available, else stdlib json.

        Raises ``ValueError`` (or a subclass) on malformed input — callers
        should not swallow this broadly; handle it as a decode error.
        """
        return _loads(raw)
