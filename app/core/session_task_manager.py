# app/core/session_task_manager.py
"""Per-session asyncio task ownership for WebSocket call sessions.

One SessionTaskManager is instantiated per WebSocket call. It wraps
asyncio.create_task, keeps every spawned task in a set, and exposes
cancel + cleanup so the WebSocket close path can drain all tasks
without leaks.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Coroutine

logger = logging.getLogger(__name__)


class SessionTaskManager:
    """Own, cancel, and await background tasks for a single session lifetime.

    Usage pattern (one instance per WebSocket handler call):
        manager = SessionTaskManager()
        session_id = str(id(some_session_object))

        task = manager.create_task(session_id, my_coro(), name="stt_connect")

        # on disconnect:
        manager.cancel_all(session_id)
        errors = await manager.cleanup(session_id)
    """

    def __init__(self) -> None:
        self._tasks: set[asyncio.Task] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_task(
        self,
        session_id: str,
        coro: Coroutine[Any, Any, Any],
        *,
        name: str,
    ) -> asyncio.Task:
        """Create and track a background task.

        The task is automatically removed from the tracked set when it
        finishes (whether it completes, is cancelled, or raises).

        Parameters
        ----------
        session_id:
            Opaque identifier for the owning session.  Used to scope the
            task name for logging; no shared state is keyed on this value.
        coro:
            Awaitable coroutine to run as a background task.
        name:
            Short label for the task (embedded in the asyncio task name).
        """
        task = asyncio.create_task(coro, name=f"{session_id}/{name}")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    def cancel_all(self, session_id: str = "") -> None:
        """Request cancellation of every live task owned by this manager."""
        for task in list(self._tasks):
            task.cancel()

    async def cleanup(self, session_id: str = "") -> list[BaseException]:
        """Cancel all tasks and await their termination.

        Returns a list of exceptions from tasks that raised (including
        CancelledError).  Never raises itself — safe to call from a
        ``finally`` block.
        """
        if not self._tasks:
            return []
        self.cancel_all(session_id)
        results = await asyncio.gather(*list(self._tasks), return_exceptions=True)
        return [r for r in results if isinstance(r, BaseException)]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def active_count(self) -> int:
        """Number of tasks that have not yet finished."""
        return len(self._tasks)
