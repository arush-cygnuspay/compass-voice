# tests/config/test_voice_transfer_config.py
"""Guards for the centralized human-agent transfer config module."""
from __future__ import annotations

import importlib
import os


def test_human_agent_transfer_number_importable_from_config() -> None:
    """HUMAN_AGENT_TRANSFER_NUMBER must be importable from app.config.voice_transfer."""
    from app.config.voice_transfer import HUMAN_AGENT_TRANSFER_NUMBER
    assert isinstance(HUMAN_AGENT_TRANSFER_NUMBER, str)
    assert HUMAN_AGENT_TRANSFER_NUMBER.strip()


def test_human_agent_transfer_number_default_value() -> None:
    """Default value must be +17036371136 when env var is unset."""
    env_key = "COMPASS_HUMAN_AGENT_TRANSFER_NUMBER"
    original = os.environ.pop(env_key, None)
    try:
        import app.config.voice_transfer as _mod
        importlib.reload(_mod)
        assert _mod.HUMAN_AGENT_TRANSFER_NUMBER == "+17036371136"
    finally:
        if original is not None:
            os.environ[env_key] = original
        importlib.reload(importlib.import_module("app.config.voice_transfer"))


def test_human_agent_transfer_number_env_override() -> None:
    """Env var COMPASS_HUMAN_AGENT_TRANSFER_NUMBER must override the default."""
    env_key = "COMPASS_HUMAN_AGENT_TRANSFER_NUMBER"
    original = os.environ.get(env_key)
    os.environ[env_key] = "+19995550001"
    try:
        import app.config.voice_transfer as _mod
        importlib.reload(_mod)
        assert _mod.HUMAN_AGENT_TRANSFER_NUMBER == "+19995550001"
    finally:
        if original is None:
            del os.environ[env_key]
        else:
            os.environ[env_key] = original
        importlib.reload(importlib.import_module("app.config.voice_transfer"))
