# app/config/logging.py
"""Logging and log-rotation settings loaded once from the environment."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class LoggingConfig:
    """Immutable snapshot of logging settings.

    rotate_realtime_logs_on_start  – rotate realtime CSV on process boot.
    rotate_gpt_logs_on_start       – rotate GPT repair CSV on process boot.
    gpt_csv_log_path               – where to write per-turn GPT repair CSV records.
    gpt_jsonl_log_path             – where to write per-turn GPT repair JSONL records
                                     (source of truth for nested GPT training data).
    realtime_log_path              – path of the realtime CSV written by the
                                     voice/Twilio server (used for rotation only).
    """

    rotate_realtime_logs_on_start: bool
    rotate_gpt_logs_on_start: bool
    gpt_csv_log_path: str
    gpt_jsonl_log_path: str
    realtime_log_path: str


@lru_cache(maxsize=1)
def get_logging_config() -> LoggingConfig:
    """Return singleton LoggingConfig from environment variables."""
    return LoggingConfig(
        rotate_realtime_logs_on_start=os.getenv(
            "COMPASS_ROTATE_REALTIME_LOGS_ON_START", "false"
        ).lower()
        == "true",
        rotate_gpt_logs_on_start=os.getenv(
            "COMPASS_ROTATE_GPT_LOGS_ON_START", "false"
        ).lower()
        == "true",
        gpt_csv_log_path=os.getenv(
            "COMPASS_GPT_CSV_LOG_PATH", "app/logs/gpt_repair_turns.csv"
        ),
        gpt_jsonl_log_path=os.getenv(
            "COMPASS_GPT_JSONL_LOG_PATH", "app/logs/gpt_repair_turns.jsonl"
        ),
        realtime_log_path=os.getenv(
            "COMPASS_REALTIME_LOG_PATH", "app/logs/realtime_turn_latency.csv"
        ),
    )
