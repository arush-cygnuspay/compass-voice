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
