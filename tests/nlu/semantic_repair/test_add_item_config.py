# tests/nlu/semantic_repair/test_add_item_config.py
"""Tests for ADD_ITEM extractor config fields in SemanticRepairConfig."""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from app.config.semantic_repair import SemanticRepairConfig


def _cfg(**kwargs) -> SemanticRepairConfig:
    defaults = dict(phase=0, model="gpt-4o-mini", timeout_seconds=0.35)
    defaults.update(kwargs)
    return SemanticRepairConfig(**defaults)


# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------

class TestAddItemConfigDefaults:
    def test_add_item_mode_defaults_disabled(self):
        cfg = _cfg()
        assert cfg.add_item_mode == "disabled"

    def test_add_item_timeout_ms_default(self):
        cfg = _cfg()
        assert cfg.add_item_timeout_ms == 350

    def test_add_item_min_text_len_default(self):
        cfg = _cfg()
        assert cfg.add_item_min_text_len == 3

    def test_add_item_max_items_per_turn_default(self):
        cfg = _cfg()
        assert cfg.add_item_max_items_per_turn == 8


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestAddItemConfigValidation:
    def test_shadow_mode_accepted(self):
        cfg = _cfg(add_item_mode="shadow")
        assert cfg.add_item_mode == "shadow"

    def test_disabled_mode_accepted(self):
        cfg = _cfg(add_item_mode="disabled")
        assert cfg.add_item_mode == "disabled"

    def test_invalid_mode_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid add_item_mode"):
            _cfg(add_item_mode="apply")

    def test_invalid_mode_unknown_raises(self):
        with pytest.raises(ValueError):
            _cfg(add_item_mode="live")

    def test_immutable_frozen_dataclass(self):
        cfg = _cfg()
        with pytest.raises(Exception):  # FrozenInstanceError
            cfg.add_item_mode = "shadow"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Environment variable loading
# ---------------------------------------------------------------------------

class TestAddItemConfigFromEnv:
    def test_env_add_item_mode_shadow(self):
        from app.config.semantic_repair import get_semantic_repair_config
        get_semantic_repair_config.cache_clear()
        try:
            with patch.dict(os.environ, {"COMPASS_GPT_ADD_ITEM_MODE": "shadow"}):
                cfg = get_semantic_repair_config()
                assert cfg.add_item_mode == "shadow"
        finally:
            get_semantic_repair_config.cache_clear()

    def test_env_add_item_mode_disabled_default(self):
        from app.config.semantic_repair import get_semantic_repair_config
        get_semantic_repair_config.cache_clear()
        try:
            env = {k: v for k, v in os.environ.items() if k != "COMPASS_GPT_ADD_ITEM_MODE"}
            with patch.dict(os.environ, env, clear=True):
                cfg = get_semantic_repair_config()
                assert cfg.add_item_mode == "disabled"
        finally:
            get_semantic_repair_config.cache_clear()

    def test_env_add_item_timeout_ms(self):
        from app.config.semantic_repair import get_semantic_repair_config
        get_semantic_repair_config.cache_clear()
        try:
            with patch.dict(os.environ, {"COMPASS_GPT_ADD_ITEM_TIMEOUT_MS": "500"}):
                cfg = get_semantic_repair_config()
                assert cfg.add_item_timeout_ms == 500
        finally:
            get_semantic_repair_config.cache_clear()

    def test_env_add_item_min_text_len(self):
        from app.config.semantic_repair import get_semantic_repair_config
        get_semantic_repair_config.cache_clear()
        try:
            with patch.dict(os.environ, {"COMPASS_GPT_ADD_ITEM_MIN_TEXT_LEN": "5"}):
                cfg = get_semantic_repair_config()
                assert cfg.add_item_min_text_len == 5
        finally:
            get_semantic_repair_config.cache_clear()

    def test_env_add_item_max_items(self):
        from app.config.semantic_repair import get_semantic_repair_config
        get_semantic_repair_config.cache_clear()
        try:
            with patch.dict(os.environ, {"COMPASS_GPT_ADD_ITEM_MAX_ITEMS_PER_TURN": "4"}):
                cfg = get_semantic_repair_config()
                assert cfg.add_item_max_items_per_turn == 4
        finally:
            get_semantic_repair_config.cache_clear()


# ---------------------------------------------------------------------------
# effective_call_mode — legacy phase-based fallback when CALL_MODE is unset
# ---------------------------------------------------------------------------

class TestEffectiveCallModeDefault:
    """COMPASS_GPT_CALL_MODE unset must use legacy phase-based effective_call_mode."""

    def _load(self, env_overrides: dict) -> "SemanticRepairConfig":
        """Load config with cache_clear, environment patched with no CALL_MODE."""
        from app.config.semantic_repair import get_semantic_repair_config
        get_semantic_repair_config.cache_clear()
        # Ensure COMPASS_GPT_CALL_MODE is absent from the env
        clean = {k: v for k, v in os.environ.items() if k != "COMPASS_GPT_CALL_MODE"}
        clean.update(env_overrides)
        try:
            with patch.dict(os.environ, clean, clear=True):
                return get_semantic_repair_config()
        finally:
            get_semantic_repair_config.cache_clear()

    def test_phase0_unset_callmode_is_disabled(self):
        """PHASE=0 + unset CALL_MODE → effective_call_mode == 'disabled'."""
        cfg = self._load({"COMPASS_GPT_REPAIR_PHASE": "0"})
        assert cfg.call_mode is None, "call_mode must be None when env var is unset"
        assert cfg.effective_call_mode == "disabled"

    def test_phase2_unset_callmode_is_eligible_only(self):
        """PHASE=2 + unset CALL_MODE → effective_call_mode == 'eligible_only'."""
        cfg = self._load({"COMPASS_GPT_REPAIR_PHASE": "2"})
        assert cfg.call_mode is None, "call_mode must be None when env var is unset"
        assert cfg.effective_call_mode == "eligible_only"

    def test_explicit_disabled_overrides_phase2(self):
        """Explicit CALL_MODE=disabled beats phase=2 eligible_only fallback."""
        from app.config.semantic_repair import get_semantic_repair_config
        get_semantic_repair_config.cache_clear()
        try:
            with patch.dict(
                os.environ,
                {"COMPASS_GPT_REPAIR_PHASE": "2", "COMPASS_GPT_CALL_MODE": "disabled"},
            ):
                cfg = get_semantic_repair_config()
            assert cfg.call_mode == "disabled"
            assert cfg.effective_call_mode == "disabled"
        finally:
            get_semantic_repair_config.cache_clear()

    def test_explicit_eligible_only_stored_directly(self):
        """Explicit CALL_MODE=eligible_only is stored as-is."""
        from app.config.semantic_repair import get_semantic_repair_config
        get_semantic_repair_config.cache_clear()
        try:
            with patch.dict(os.environ, {"COMPASS_GPT_CALL_MODE": "eligible_only"}):
                cfg = get_semantic_repair_config()
            assert cfg.call_mode == "eligible_only"
            assert cfg.effective_call_mode == "eligible_only"
        finally:
            get_semantic_repair_config.cache_clear()

    def test_explicit_all_shadow_stored_directly(self):
        """Explicit CALL_MODE=all_shadow is stored as-is."""
        from app.config.semantic_repair import get_semantic_repair_config
        get_semantic_repair_config.cache_clear()
        try:
            with patch.dict(os.environ, {"COMPASS_GPT_CALL_MODE": "all_shadow"}):
                cfg = get_semantic_repair_config()
            assert cfg.call_mode == "all_shadow"
            assert cfg.effective_call_mode == "all_shadow"
        finally:
            get_semantic_repair_config.cache_clear()

    def test_invalid_callmode_treated_as_unset_uses_phase(self):
        """An unrecognised CALL_MODE value falls back to None → legacy phase."""
        from app.config.semantic_repair import get_semantic_repair_config
        get_semantic_repair_config.cache_clear()
        try:
            with patch.dict(
                os.environ,
                {"COMPASS_GPT_REPAIR_PHASE": "2", "COMPASS_GPT_CALL_MODE": "unknown_mode"},
            ):
                cfg = get_semantic_repair_config()
            # Unknown value is silently discarded; phase=2 → eligible_only
            assert cfg.call_mode is None
            assert cfg.effective_call_mode == "eligible_only"
        finally:
            get_semantic_repair_config.cache_clear()
