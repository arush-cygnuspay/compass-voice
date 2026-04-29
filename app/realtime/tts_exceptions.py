# app/realtime/tts_exceptions.py
"""Typed exception for TTS synthesis failures.

Raised by ``speak_response_text`` after all retry attempts are exhausted
so the conversation layer can apply a deterministic fallback instead of
returning to LISTENING with no audio (dead air).
"""
from __future__ import annotations


class TTSFailureError(Exception):
    """TTS synthesis failed after all configured retry attempts.

    Attributes:
        attempts: Number of synthesis attempts made before giving up.
        provider: Name of the TTS provider (e.g. ``"deepgram"``).
    """

    def __init__(
        self,
        message: str,
        *,
        attempts: int,
        provider: str | None = None,
    ) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.provider = provider
