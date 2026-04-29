# tests/core/test_session_task_manager.py
"""Tests for SessionTaskManager — per-session asyncio task tracking.

Validates:
- Tasks are registered and auto-removed when complete.
- cancel_all() cancels every live task.
- cleanup() awaits all tasks and returns exceptions (including CancelledError).
- Completed tasks are not double-awaited.
- cleanup() is safe to call repeatedly (idempotent after first drain).
- active_count reflects live tasks only.
- session_id scopes the task name but does not share state across instances.
- Concurrent sessions are fully isolated.
- Edge cases: task already done before cancel, task raises during run,
  cleanup on empty manager.
"""
from __future__ import annotations

import asyncio
import unittest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


async def _noop():
    pass


async def _sleep_forever():
    await asyncio.sleep(9999)


async def _raises(exc: type[BaseException] = ValueError):
    raise exc("boom")


# ---------------------------------------------------------------------------
# Import under test
# ---------------------------------------------------------------------------

from app.core.session_task_manager import SessionTaskManager

SID = "test-session-001"


# ---------------------------------------------------------------------------
# create_task — registration and auto-removal
# ---------------------------------------------------------------------------

class CreateTaskTests(unittest.TestCase):

    def test_task_is_returned(self):
        async def go():
            mgr = SessionTaskManager()
            task = mgr.create_task(SID, _noop(), name="t")
            self.assertIsInstance(task, asyncio.Task)
            await task
        _run(go())

    def test_task_name_includes_session_id_and_name(self):
        async def go():
            mgr = SessionTaskManager()
            task = mgr.create_task(SID, _noop(), name="stt_connect")
            self.assertIn(SID, task.get_name())
            self.assertIn("stt_connect", task.get_name())
            await task
        _run(go())

    def test_active_count_increments_on_create(self):
        async def go():
            mgr = SessionTaskManager()
            mgr.create_task(SID, _sleep_forever(), name="a")
            self.assertEqual(mgr.active_count, 1)
            mgr.create_task(SID, _sleep_forever(), name="b")
            self.assertEqual(mgr.active_count, 2)
            mgr.cancel_all()
            await mgr.cleanup()
        _run(go())

    def test_completed_task_is_auto_removed(self):
        async def go():
            mgr = SessionTaskManager()
            task = mgr.create_task(SID, _noop(), name="t")
            await task
            # done callback runs synchronously after await
            self.assertEqual(mgr.active_count, 0)
        _run(go())

    def test_raising_task_is_auto_removed(self):
        async def go():
            mgr = SessionTaskManager()
            task = mgr.create_task(SID, _raises(RuntimeError), name="t")
            try:
                await task
            except RuntimeError:
                pass
            self.assertEqual(mgr.active_count, 0)
        _run(go())


# ---------------------------------------------------------------------------
# cancel_all
# ---------------------------------------------------------------------------

class CancelAllTests(unittest.TestCase):

    def test_cancel_all_cancels_live_tasks(self):
        async def go():
            mgr = SessionTaskManager()
            task = mgr.create_task(SID, _sleep_forever(), name="t")
            mgr.cancel_all(SID)
            with self.assertRaises(asyncio.CancelledError):
                await task
        _run(go())

    def test_cancel_all_on_empty_manager_is_safe(self):
        async def go():
            mgr = SessionTaskManager()
            mgr.cancel_all(SID)   # must not raise
        _run(go())

    def test_cancel_all_does_not_raise_on_already_done_task(self):
        async def go():
            mgr = SessionTaskManager()
            task = mgr.create_task(SID, _noop(), name="t")
            await task  # finishes, auto-removed
            mgr.cancel_all(SID)  # set is now empty — must not raise
        _run(go())


# ---------------------------------------------------------------------------
# cleanup
# ---------------------------------------------------------------------------

class CleanupTests(unittest.TestCase):

    def test_cleanup_returns_empty_list_when_no_tasks(self):
        async def go():
            mgr = SessionTaskManager()
            result = await mgr.cleanup(SID)
            self.assertEqual(result, [])
        _run(go())

    def test_cleanup_returns_cancelled_errors(self):
        async def go():
            mgr = SessionTaskManager()
            mgr.create_task(SID, _sleep_forever(), name="t")
            errors = await mgr.cleanup(SID)
            self.assertEqual(len(errors), 1)
            self.assertIsInstance(errors[0], asyncio.CancelledError)
        _run(go())

    def test_cleanup_returns_task_exceptions(self):
        async def go():
            mgr = SessionTaskManager()
            mgr.create_task(SID, _raises(RuntimeError), name="t")
            # Yield once so the task runs and raises (done callback is
            # scheduled via call_soon and fires on the next iteration).
            await asyncio.sleep(0)
            # Task is done but still in the set (discard callback pending).
            # cleanup() gathers it and surfaces the RuntimeError.
            errors = await mgr.cleanup(SID)
            self.assertEqual(len(errors), 1)
            self.assertIsInstance(errors[0], RuntimeError)
        _run(go())

    def test_cleanup_drains_all_tasks(self):
        async def go():
            mgr = SessionTaskManager()
            mgr.create_task(SID, _sleep_forever(), name="a")
            mgr.create_task(SID, _sleep_forever(), name="b")
            self.assertEqual(mgr.active_count, 2)
            await mgr.cleanup(SID)
            self.assertEqual(mgr.active_count, 0)
        _run(go())

    def test_cleanup_is_safe_to_call_twice(self):
        async def go():
            mgr = SessionTaskManager()
            mgr.create_task(SID, _sleep_forever(), name="t")
            await mgr.cleanup(SID)
            result = await mgr.cleanup(SID)  # second call — set is empty
            self.assertEqual(result, [])
        _run(go())

    def test_cleanup_does_not_raise_on_task_exception(self):
        async def go():
            mgr = SessionTaskManager()
            mgr.create_task(SID, _raises(ValueError), name="t")
            # cleanup must swallow the ValueError via return_exceptions=True
            try:
                await mgr.cleanup(SID)
            except ValueError:
                self.fail("cleanup must not propagate task exceptions")
        _run(go())


# ---------------------------------------------------------------------------
# Session isolation
# ---------------------------------------------------------------------------

class SessionIsolationTests(unittest.TestCase):

    def test_two_manager_instances_are_independent(self):
        async def go():
            mgr_a = SessionTaskManager()
            mgr_b = SessionTaskManager()
            mgr_a.create_task("session-a", _sleep_forever(), name="t")
            mgr_b.create_task("session-b", _sleep_forever(), name="t")

            # cancel A only
            await mgr_a.cleanup("session-a")
            self.assertEqual(mgr_a.active_count, 0)
            self.assertEqual(mgr_b.active_count, 1)   # B unaffected

            await mgr_b.cleanup("session-b")
        _run(go())

    def test_concurrent_sessions_do_not_share_tasks(self):
        async def go():
            mgr1 = SessionTaskManager()
            mgr2 = SessionTaskManager()
            mgr1.create_task("s1", _sleep_forever(), name="x")
            mgr1.create_task("s1", _sleep_forever(), name="y")
            mgr2.create_task("s2", _sleep_forever(), name="z")

            self.assertEqual(mgr1.active_count, 2)
            self.assertEqual(mgr2.active_count, 1)

            await mgr1.cleanup()
            await mgr2.cleanup()

            self.assertEqual(mgr1.active_count, 0)
            self.assertEqual(mgr2.active_count, 0)
        _run(go())


# ---------------------------------------------------------------------------
# active_count
# ---------------------------------------------------------------------------

class ActiveCountTests(unittest.TestCase):

    def test_active_count_is_zero_on_fresh_manager(self):
        mgr = SessionTaskManager()
        self.assertEqual(mgr.active_count, 0)

    def test_active_count_drops_to_zero_after_cleanup(self):
        async def go():
            mgr = SessionTaskManager()
            for i in range(5):
                mgr.create_task(SID, _sleep_forever(), name=f"t{i}")
            self.assertEqual(mgr.active_count, 5)
            await mgr.cleanup(SID)
            self.assertEqual(mgr.active_count, 0)
        _run(go())


if __name__ == "__main__":
    unittest.main()
