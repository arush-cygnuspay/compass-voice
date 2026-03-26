# app/menu/slot_helpers.py

from __future__ import annotations

from typing import Iterable, Optional

from app.nlu.nlu_result import SlotValue


def first_slot_value(slots: Iterable[SlotValue], *labels: str) -> str | None:
    """
    Returns the first non-empty slot.value matching any of the given labels.
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