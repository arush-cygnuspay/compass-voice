# app/realtime/realtime_conversation_state.py
from __future__ import annotations

from enum import StrEnum


class RealtimePhase(StrEnum):
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"