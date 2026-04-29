# app/nlu/matching/index.py
"""Reusable slot-value accessors for NLU results.

Provides helpers for extracting typed values from ``SlotValue`` sequences
without scattering the same lookup boilerplate across handlers.
"""
from __future__ import annotations

from typing import Iterable

from app.nlu.nlu_result import SlotValue

__all__ = [
    "slot_values",
    "first_slot_value",
]


def slot_values(slots: Iterable[SlotValue], *labels: str) -> list[str]:
    """Return all non-empty string slot values matching any of the given labels.

    Values are returned in slot order and de-duplicated.
    """
    want = {label.upper() for label in labels}
    values: list[str] = []
    seen: set[str] = set()

    for slot in slots:
        if str(slot.name).upper() not in want:
            continue

        value = slot.value
        if not isinstance(value, str):
            continue

        value = value.strip()
        if not value or value in seen:
            continue

        seen.add(value)
        values.append(value)

    return values


def first_slot_value(slots: Iterable[SlotValue], *labels: str) -> str | None:
    """Return the first non-empty slot.value matching any of the given labels.

    Labels are matched case-insensitively.
    """
    want = {label.upper() for label in labels}

    for slot in slots:
        if str(slot.name).upper() not in want:
            continue

        value = slot.value
        if isinstance(value, str):
            value = value.strip()
            if value:
                return value

    return None
