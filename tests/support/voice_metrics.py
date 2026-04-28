# D:/Working/Cygnus/compass-voice/tests/support/voice_metrics.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MetricSummary:
    intent_accuracy: float
    slot_accuracy: float
    flow_completion_success_rate: float
    average_turns_per_successful_order: float
    fallback_frequency: float


def safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator
