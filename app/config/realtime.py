# app/config/realtime.py
"""Typed realtime / diagnostics settings loaded once from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional


@dataclass(frozen=True)
class RealtimeConfig:
    """Immutable realtime / diagnostics settings snapshot."""

    route_debug_enabled: bool
    turn_timing_enabled: bool
    nlu_json_log_path: Optional[str]


@lru_cache(maxsize=1)
def get_realtime_config() -> RealtimeConfig:
    """Return the singleton RealtimeConfig, loading env vars on first call."""
    return RealtimeConfig(
        route_debug_enabled=os.getenv("COMPASS_ROUTE_DEBUG_ENABLED", "0") == "1",
        turn_timing_enabled=os.getenv("COMPASS_TURN_TIMING_ENABLED", "0") == "1",
        nlu_json_log_path=os.getenv("COMPASS_NLU_JSON_LOG_PATH") or None,
    )


# ---------------------------------------------------------------------------
# Turn-taking / barge-in config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RealtimeTurnConfig:
    """Turn-taking and barge-in configuration for realtime voice sessions.

    All values are loaded from environment variables with safe, demo-ready
    defaults.  The object is frozen so callers can cache the reference.
    """

    # How long (ms) to wait after the last STT final before committing the turn.
    user_turn_commit_delay_ms: int = 700

    # Minimum accumulated audio duration (ms) for a barge-in candidate.
    min_barge_in_audio_ms: int = 700

    # Minimum word count for a barge-in to be accepted (unless known command).
    min_barge_in_words: int = 2

    # Minimum STT confidence for barge-in acceptance (0.0–1.0).
    min_barge_in_confidence: float = 0.60

    # Suppress barge-in for this many ms after TTS playback starts.
    post_playback_guard_ms: int = 250

    # Natural pause (ms) after turn commit before speaking the response.
    post_user_turn_response_delay_ms: int = 250

    # Master barge-in enable/disable switch.
    barge_in_enabled: bool = True

    # When True, no FSM turn fires while assistant is speaking unless barge-in
    # passes all acceptance criteria.
    strict_turn_taking: bool = True

    # asyncio.Lock acquisition timeout (seconds).
    turn_lock_timeout_s: float = 8.0

    # Maximum entries in the pending-interrupt queue.
    max_pending_interrupt_queue: int = 2

    # Maximum recursive drain depth after a turn completes.
    max_drain_depth: int = 4

    # Deepgram endpointing_ms (silence window before emitting a final).
    deepgram_stt_endpointing_ms: int = 300

    # Deepgram utterance_end_ms (silence window before emitting UtteranceEnd).
    deepgram_stt_utterance_end_ms: int = 1000


# ---------------------------------------------------------------------------
# Internal env-var helpers
# ---------------------------------------------------------------------------

def _get_bool(key: str, default: bool) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes")


def _get_int(key: str, default: int, min_val: int, max_val: int) -> int:
    val = os.getenv(key)
    if val is None:
        return default
    try:
        parsed = int(val.strip())
    except ValueError:
        return default
    return max(min_val, min(max_val, parsed))


def _get_float(key: str, default: float, min_val: float, max_val: float) -> float:
    val = os.getenv(key)
    if val is None:
        return default
    try:
        parsed = float(val.strip())
    except ValueError:
        return default
    return max(min_val, min(max_val, parsed))


@lru_cache(maxsize=1)
def get_realtime_turn_config() -> RealtimeTurnConfig:
    """Return singleton RealtimeTurnConfig with validated env-var overrides."""
    return RealtimeTurnConfig(
        user_turn_commit_delay_ms=_get_int("USER_TURN_COMMIT_DELAY_MS", 700, 100, 3000),
        min_barge_in_audio_ms=_get_int("MIN_BARGE_IN_AUDIO_MS", 700, 100, 5000),
        min_barge_in_words=_get_int("MIN_BARGE_IN_WORDS", 2, 1, 20),
        min_barge_in_confidence=_get_float("MIN_BARGE_IN_CONFIDENCE", 0.60, 0.0, 1.0),
        post_playback_guard_ms=_get_int("POST_PLAYBACK_GUARD_MS", 250, 0, 2000),
        post_user_turn_response_delay_ms=_get_int(
            "POST_USER_TURN_RESPONSE_DELAY_MS", 250, 0, 2000
        ),
        barge_in_enabled=_get_bool("BARGE_IN_ENABLED", True),
        strict_turn_taking=_get_bool("STRICT_TURN_TAKING", True),
        turn_lock_timeout_s=_get_float("TURN_LOCK_TIMEOUT_S", 8.0, 1.0, 60.0),
        max_pending_interrupt_queue=_get_int("MAX_PENDING_INTERRUPT_QUEUE", 2, 1, 10),
        max_drain_depth=_get_int("MAX_DRAIN_DEPTH", 4, 1, 20),
        deepgram_stt_endpointing_ms=_get_int("DEEPGRAM_STT_ENDPOINTING_MS", 300, 100, 5000),
        deepgram_stt_utterance_end_ms=_get_int(
            "DEEPGRAM_STT_UTTERANCE_END_MS", 1000, 100, 10000
        ),
    )
