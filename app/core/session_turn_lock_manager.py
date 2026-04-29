# app/core/session_turn_lock_manager.py
"""Per-session asyncio lock registry.

Callers (e.g. the voice transport layer) acquire the lock before invoking
TurnEngine.process_turn so that concurrent events for the same session are
serialised without blocking events for different sessions.

Usage:
    lock = turn_lock_manager.get_lock(session.session_id)
    async with lock:
        output = engine.process_turn(session, user_text)
"""
from __future__ import annotations

import asyncio
import weakref


class SessionTurnLockManager:
    """Maps session_id → asyncio.Lock.  Locks are created on demand and
    garbage-collected automatically when the last reference drops."""

    def __init__(self) -> None:
        self._locks: weakref.WeakValueDictionary[str, asyncio.Lock] = (
            weakref.WeakValueDictionary()
        )

    def get_lock(self, session_id: str) -> asyncio.Lock:
        lock = self._locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[session_id] = lock
        return lock
