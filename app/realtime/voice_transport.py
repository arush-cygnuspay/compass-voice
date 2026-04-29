# app/realtime/voice_transport.py
"""Transport adapter interface used by :class:`ConversationSession`.

The WebSocket handler in ``app/api/voice_stream_server.py`` implements
these hooks so :class:`ConversationSession` can drive audio I/O,
call-control side effects, and turn-trace lifecycle without holding any
direct reference to Twilio frames or Deepgram clients.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class VoiceTransport(Protocol):
    """Hooks the conversation orchestrator calls into the transport.

    Every method is expected to be safe to call from inside the
    ``twilio_media_ws`` task.  No method is allowed to block on user
    input or perform synchronous I/O on the audio path; latency-
    sensitive behavior remains owned by the WebSocket handler.
    """

    def is_barge_in_disabled(self) -> bool: ...

    def debug_log(self, event: str, payload: dict[str, Any]) -> None: ...

    def begin_turn_trace(self, *, user_text: str) -> Any | None:
        """Finalize the previous trace (if any), bump turn_index, and
        return a freshly constructed :class:`RealtimeTurnTrace`.  May
        return ``None`` if tracing is unavailable for this turn."""
        ...

    def annotate_response_trace(
        self,
        trace: Any | None,
        *,
        responder_start_monotonic: float | None,
        responder_end_monotonic: float | None,
        response_key: str,
        internal_response_text: str,
        spoken_response_text: str,
        end_call_after_playback: bool,
    ) -> None: ...

    async def speak_response(
        self,
        spoken_text: str,
        *,
        trace: Any | None,
        end_call_after_playback: bool,
    ) -> None: ...

    async def interrupt_playback(self, reason: str) -> None: ...

    async def transfer_call(self, target_number: str) -> None: ...

    async def end_call(self) -> None: ...
