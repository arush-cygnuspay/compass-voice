# app/state_machine/models/turn_memory.py
"""Typed turn memory entry for per-session GPT context tracking.

Stored in ConversationContext.turn_memory (bounded deque, maxlen=6).
These entries are runtime-only: never persisted to disk, never fetched from logs.
PII contract: no API keys, payment links, card data, or raw phone numbers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class TurnMemoryEntry:
    """Immutable record of one conversation turn for GPT context building.

    Fields
    ------
    role:
        "user" for customer utterances, "assistant" for bot responses.
    text:
        Spoken/displayed text for this turn (normalized for user, rendered for assistant).
    normalized_text:
        Preprocessed user input (user turns only).
    response_key:
        Internal response key (assistant turns only).
    state:
        Conversation state value string at turn time.
    intent:
        Effective intent string at turn time (user turns).
    slots:
        Compact slot representation captured this turn (user turns).
    timestamp_utc:
        ISO 8601 UTC timestamp string.
    """

    role: Literal["user", "assistant"]
    text: str
    normalized_text: str | None = None
    response_key: str | None = None
    state: str | None = None
    intent: str | None = None
    slots: tuple | None = None
    timestamp_utc: str | None = None
