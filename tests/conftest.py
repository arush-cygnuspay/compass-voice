# tests/conftest.py
"""Shared pytest fixtures.

Hot path: ``build_menu_repo()`` parses a ~430KB JSON menu fixture every
call. With ~1500 tests this is the single largest constant overhead in
the suite. The session-scoped fixture below loads it once and hands the
SAME repo instance to every test - safe because ``MenuRepository`` is
immutable from the test's perspective (no test mutates the menu data).

Tests that previously called ``build_menu_repo()`` directly continue to
work; new tests should prefer the ``menu_repo`` fixture.
"""
from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def menu_repo():
    """Session-scoped MenuRepository.

    Reuses one parsed menu across every test in the run. ~430KB JSON is
    loaded exactly once instead of ~1500 times.
    """
    from tests.support.voice_test_harness import build_menu_repo as _build

    return _build()


@pytest.fixture
def fresh_session():
    """Factory for a clean Session at WAITING_FOR_ORDER_TYPE."""
    from tests.support.voice_test_harness import new_session as _new

    return _new


@pytest.fixture
def engine_factory(menu_repo):
    """Factory: ``engine_factory(checkout_service=...)`` -> TurnEngine.

    Reuses the session-scoped ``menu_repo``. Allows per-test overrides
    of stub services without re-parsing the menu.
    """
    from tests.support.voice_test_harness import build_engine as _build

    def _factory(**kwargs):
        kwargs.setdefault("menu_repo", menu_repo)
        return _build(**kwargs)

    return _factory
