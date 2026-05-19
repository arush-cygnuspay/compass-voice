# app/state_machine/services/turn_memory_service.py
"""Structured helpers for writing and reading TurnMemoryEntry objects.

These helpers are the preferred write path when callers have rich context
(state, intent, slots). The simpler ctx.append_turn_memory(role, text)
API remains for backward-compatible callers.

PII contract: sanitize user-facing text before appending.
No API keys, payment links, card data, or unnecessary PII stored.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from app.state_machine.models.turn_memory import TurnMemoryEntry

if TYPE_CHECKING:
    from app.state_machine.models.conversation_context import ConversationContext

_MAX_TEXT_LEN = 512  # cap stored text to prevent unbounded memory


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _cap(text: str | None) -> str | None:
    if text is None:
        return None
    return text[:_MAX_TEXT_LEN] if len(text) > _MAX_TEXT_LEN else text


def append_user_turn_memory(
    context: "ConversationContext",
    text: str,
    normalized_text: str | None = None,
    state: str | None = None,
    intent: str | None = None,
    slots: tuple | None = None,
) -> None:
    """Append a user turn entry with optional rich context fields."""
    if not text or not text.strip():
        return
    entry = TurnMemoryEntry(
        role="user",
        text=(_cap(text) or "").strip(),
        normalized_text=_cap(normalized_text),
        response_key=None,
        state=state,
        intent=intent,
        slots=slots,
        timestamp_utc=_utc_now_iso(),
    )
    context.append_turn_memory_entry(entry)


def append_assistant_turn_memory(
    context: "ConversationContext",
    response_text: str,
    response_key: str | None = None,
    state: str | None = None,
) -> None:
    """Append an assistant turn entry with response_key and state."""
    if not response_text or not response_text.strip():
        return
    entry = TurnMemoryEntry(
        role="assistant",
        text=(_cap(response_text) or "").strip(),
        normalized_text=None,
        response_key=response_key,
        state=state,
        intent=None,
        slots=None,
        timestamp_utc=_utc_now_iso(),
    )
    context.append_turn_memory_entry(entry)


def get_recent_turns(
    context: "ConversationContext",
    max_entries: int = 6,
) -> tuple[TurnMemoryEntry, ...]:
    """Return the last *max_entries* TurnMemoryEntry objects from the context."""
    getter = getattr(context, "get_turn_memory_entries", None)
    if callable(getter):
        return getter(max_entries)
    # Fallback: read simple tuples and wrap as entries
    getter2 = getattr(context, "get_turn_memory", None)
    if not callable(getter2):
        return ()
    raw = getter2(max_entries)
    return tuple(
        TurnMemoryEntry(role=r, text=t)  # type: ignore[arg-type]
        for r, t in raw
        if r and t and str(t).strip()
    )
