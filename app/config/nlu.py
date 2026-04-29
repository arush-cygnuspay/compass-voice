# app/config/nlu.py
"""Typed NLU settings loaded once from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class NluConfig:
    """Immutable NLU/intent-classification settings snapshot."""

    intent_conf_threshold: float
    max_item_queue_depth: int


@lru_cache(maxsize=1)
def get_nlu_config() -> NluConfig:
    """Return the singleton NluConfig, loading env vars on first call."""
    return NluConfig(
        intent_conf_threshold=float(
            os.getenv("COMPASS_INTENT_CONF_THRESHOLD", "0.55")
        ),
        max_item_queue_depth=int(
            os.getenv("COMPASS_MAX_ITEM_QUEUE_DEPTH", "20")
        ),
    )
