# tests/nlu/semantic_repair/test_gpt_call_mode_config.py
"""Tests for COMPASS_GPT_CALL_MODE config additions (Step 1 rollout)."""
from __future__ import annotations

import pytest

from app.config.semantic_repair import (
    VALID_CALL_MODES,
    SemanticRepairConfig,
    get_semantic_repair_config,
)


# ---------------------------------------------------------------------------
# VALID_CALL_MODES constant
# ---------------------------------------------------------------------------


class TestValidCallModes:
    def test_all_required_modes_present(self):
        assert "disabled" in VALID_CALL_MODES
        assert "eligible_only" in VALID_CALL_MODES
        assert "all_shadow" in VALID_CALL_MODES
        assert "all_apply_safe" in VALID_CALL_MODES

    def test_no_extra_modes(self):
        assert len(VALID_CALL_MODES) == 4


# ---------------------------------------------------------------------------
# SemanticRepairConfig — call_mode field
# ---------------------------------------------------------------------------


class TestSemanticRepairConfigCallMode:
    def _cfg(self, call_mode: str | None = None, **kw) -> SemanticRepairConfig:
        return SemanticRepairConfig(
            phase=2,
            model="gpt-4o-mini",
            timeout_seconds=0.35,
            call_mode=call_mode,
            **kw,
        )

    def test_call_mode_defaults_none(self):
        cfg = SemanticRepairConfig(phase=2, model="gpt-4o-mini", timeout_seconds=0.35)
        assert cfg.call_mode is None

    def test_call_mode_disabled_accepted(self):
        cfg = self._cfg(call_mode="disabled")
        assert cfg.call_mode == "disabled"

    def test_call_mode_eligible_only_accepted(self):
        cfg = self._cfg(call_mode="eligible_only")
        assert cfg.call_mode == "eligible_only"

    def test_call_mode_all_shadow_accepted(self):
        cfg = self._cfg(call_mode="all_shadow")
        assert cfg.call_mode == "all_shadow"

    def test_call_mode_all_apply_safe_accepted(self):
        cfg = self._cfg(call_mode="all_apply_safe")
        assert cfg.call_mode == "all_apply_safe"

    def test_invalid_call_mode_raises(self):
        with pytest.raises(ValueError, match="Invalid call_mode"):
            self._cfg(call_mode="bad_mode")

    def test_invalid_call_mode_uppercase_raises(self):
        with pytest.raises(ValueError):
            self._cfg(call_mode="ALL_SHADOW")

    def test_none_call_mode_no_error(self):
        cfg = self._cfg(call_mode=None)
        assert cfg.call_mode is None


# ---------------------------------------------------------------------------
# effective_call_mode property
# ---------------------------------------------------------------------------


class TestEffectiveCallMode:
    def test_explicit_disabled_returns_disabled(self):
        cfg = SemanticRepairConfig(phase=2, model="m", timeout_seconds=0.1, call_mode="disabled")
        assert cfg.effective_call_mode == "disabled"

    def test_explicit_all_shadow_returns_all_shadow(self):
        cfg = SemanticRepairConfig(phase=2, model="m", timeout_seconds=0.1, call_mode="all_shadow")
        assert cfg.effective_call_mode == "all_shadow"

    def test_explicit_eligible_only_returns_eligible_only(self):
        cfg = SemanticRepairConfig(phase=2, model="m", timeout_seconds=0.1, call_mode="eligible_only")
        assert cfg.effective_call_mode == "eligible_only"

    def test_legacy_phase0_resolves_to_disabled(self):
        cfg = SemanticRepairConfig(phase=0, model="m", timeout_seconds=0.1)
        assert cfg.effective_call_mode == "disabled"

    def test_legacy_phase1_resolves_to_disabled(self):
        cfg = SemanticRepairConfig(phase=1, model="m", timeout_seconds=0.1)
        assert cfg.effective_call_mode == "disabled"

    def test_legacy_phase2_resolves_to_eligible_only(self):
        cfg = SemanticRepairConfig(phase=2, model="m", timeout_seconds=0.1)
        assert cfg.effective_call_mode == "eligible_only"

    def test_legacy_phase3_resolves_to_eligible_only(self):
        cfg = SemanticRepairConfig(phase=3, model="m", timeout_seconds=0.1)
        assert cfg.effective_call_mode == "eligible_only"


# ---------------------------------------------------------------------------
# New fields: shadow_timeout_ms, apply_fallbacks
# ---------------------------------------------------------------------------


class TestNewConfigFields:
    def test_shadow_timeout_ms_default(self):
        cfg = SemanticRepairConfig(phase=0, model="m", timeout_seconds=0.1)
        assert cfg.shadow_timeout_ms == 2000

    def test_shadow_timeout_ms_custom(self):
        cfg = SemanticRepairConfig(phase=0, model="m", timeout_seconds=0.1, shadow_timeout_ms=1500)
        assert cfg.shadow_timeout_ms == 1500

    def test_apply_fallbacks_default_false(self):
        cfg = SemanticRepairConfig(phase=0, model="m", timeout_seconds=0.1)
        assert cfg.apply_fallbacks is False

    def test_apply_fallbacks_can_be_true(self):
        cfg = SemanticRepairConfig(phase=2, model="m", timeout_seconds=0.1, apply_fallbacks=True)
        assert cfg.apply_fallbacks is True


# ---------------------------------------------------------------------------
# get_semantic_repair_config() env loading
# ---------------------------------------------------------------------------


class TestGetSemanticRepairConfigEnv:
    def test_default_call_mode_is_none_when_unset(self, monkeypatch):
        """When COMPASS_GPT_CALL_MODE is unset, call_mode must be None (legacy phase behavior)."""
        monkeypatch.delenv("COMPASS_GPT_CALL_MODE", raising=False)
        get_semantic_repair_config.cache_clear()
        cfg = get_semantic_repair_config()
        # call_mode=None → effective_call_mode derived from phase (phase=0 → "disabled")
        assert cfg.call_mode is None, (
            "call_mode must be None when COMPASS_GPT_CALL_MODE is unset so "
            "effective_call_mode can use legacy phase-based behavior"
        )
        assert cfg.effective_call_mode == "disabled"
        get_semantic_repair_config.cache_clear()

    def test_all_shadow_loaded_from_env(self, monkeypatch):
        monkeypatch.setenv("COMPASS_GPT_CALL_MODE", "all_shadow")
        get_semantic_repair_config.cache_clear()
        cfg = get_semantic_repair_config()
        assert cfg.call_mode == "all_shadow"
        get_semantic_repair_config.cache_clear()

    def test_invalid_env_falls_back_to_none_uses_phase(self, monkeypatch):
        """An invalid COMPASS_GPT_CALL_MODE value silently maps to None → legacy phase."""
        monkeypatch.setenv("COMPASS_GPT_CALL_MODE", "totally_invalid")
        get_semantic_repair_config.cache_clear()
        cfg = get_semantic_repair_config()
        # Invalid value is discarded; call_mode=None → effective from phase.
        assert cfg.call_mode is None
        assert cfg.effective_call_mode == "disabled"  # default phase=0
        get_semantic_repair_config.cache_clear()

    def test_shadow_timeout_loaded_from_env(self, monkeypatch):
        monkeypatch.setenv("COMPASS_GPT_SHADOW_TIMEOUT_MS", "3000")
        get_semantic_repair_config.cache_clear()
        cfg = get_semantic_repair_config()
        assert cfg.shadow_timeout_ms == 3000
        get_semantic_repair_config.cache_clear()

    def test_apply_fallbacks_loaded_from_env(self, monkeypatch):
        monkeypatch.setenv("COMPASS_GPT_APPLY_FALLBACKS", "true")
        get_semantic_repair_config.cache_clear()
        cfg = get_semantic_repair_config()
        assert cfg.apply_fallbacks is True
        get_semantic_repair_config.cache_clear()

    def test_apply_fallbacks_false_by_default(self, monkeypatch):
        monkeypatch.delenv("COMPASS_GPT_APPLY_FALLBACKS", raising=False)
        get_semantic_repair_config.cache_clear()
        cfg = get_semantic_repair_config()
        assert cfg.apply_fallbacks is False
        get_semantic_repair_config.cache_clear()
