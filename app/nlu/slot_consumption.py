# app/nlu/slot_consumption.py
"""Reusable contract for handlers that prefer NLU slots over regex.

Pattern: read the relevant slot from the latest NLU result; if present and
well-formed, use it. Otherwise, fall back to the existing regex parser.
Whenever the fallback fires we emit a structured log event so we can
measure slot-extraction reliability per slot type per consumer in
production.

The signature deviates slightly from the original spec: the helper takes
the slot tuple directly (matching the existing :func:`first_slot_value`
API in :mod:`app.menu.slot_helpers`) instead of a full
``ConversationContext``. Handler call sites already have the slot tuple
in scope, so this is the path of least coupling.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Generic, Iterable, Optional, TypeVar

from app.nlu.nlu_result import SlotValue


T = TypeVar("T")

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SlotResolution(Generic[T]):
    value: Optional[T]
    source: str  # "slot" | "regex_fallback" | "missing"


def log_slot_event(event_name: str, **data: Any) -> None:
    """Emit a structured log event mirroring ``log_control_intent_event``."""
    logger.info(event_name, extra={"event_name": event_name, **data})


def _coerce_slot_value(slot: SlotValue) -> Optional[str]:
    raw = slot.value
    if raw is None:
        return None
    if isinstance(raw, str):
        stripped = raw.strip()
        return stripped or None
    text = str(raw).strip()
    return text or None


def consume_slot_or_fallback(
    *,
    slots: Iterable[SlotValue],
    slot_labels: tuple[str, ...],
    fallback: Callable[[], Optional[T]],
    parse: Callable[[str], Optional[T]] = lambda s: s,  # type: ignore[assignment,return-value]
    consumer_site: str,
) -> SlotResolution[T]:
    """Read a slot from the latest NLU result; fall back to a regex parser.

    Emits ``nlu_slot_consumed`` when slot is used, ``nlu_slot_fallback``
    when the fallback fires, ``nlu_slot_missing`` when both fail.
    """
    wanted_labels = {label.upper() for label in slot_labels}

    for slot in slots or ():
        name = str(getattr(slot, "name", "")).upper()
        if name not in wanted_labels:
            continue
        text = _coerce_slot_value(slot)
        if text is None:
            continue
        parsed = parse(text)
        if parsed is None:
            continue
        log_slot_event(
            "nlu_slot_consumed",
            consumer_site=consumer_site,
            slot=name,
        )
        return SlotResolution(parsed, "slot")

    fallback_value = fallback()
    if fallback_value is not None:
        log_slot_event(
            "nlu_slot_fallback",
            consumer_site=consumer_site,
            attempted_slots=list(slot_labels),
        )
        return SlotResolution(fallback_value, "regex_fallback")

    log_slot_event(
        "nlu_slot_missing",
        consumer_site=consumer_site,
        attempted_slots=list(slot_labels),
    )
    return SlotResolution(None, "missing")
