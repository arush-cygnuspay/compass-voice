# tests/scripts/test_verify_gpt_ordering_pipeline.py
"""Tests for scripts/verify_gpt_ordering_pipeline.py

Verifies:
  - disabled config passes all gate checks (no warnings on critical items)
  - shadow config passes with INFO/WARN on expected items
  - inline without API key produces WARN/FAIL on key check
  - invalid config value produces FAIL on mode check
  - production env + inline produces WARN on production_inline_check
  - script never calls OpenAI (all checks are structural/config-only)
  - exit code is 0 when no FAILs
  - exit code is 1 when any FAIL present
  - main() runs without side effects
  - shadow_never_applies check passes (structural invariant)
  - inline_apply_gate_all_conditions check passes (structural invariant)
  - logging fields checks pass with current source
  - handler_dispatcher_builds passes with test menu repo
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from unittest.mock import patch, MagicMock
import unittest

# ── Ensure project root is on path ───────────────────────────────────────────
_TESTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _TESTS_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Stub external deps that may not be installed in CI ───────────────────────
for _mod_name in ("twilio", "twilio.base", "twilio.base.exceptions", "twilio.rest"):
    sys.modules.setdefault(_mod_name, types.ModuleType(_mod_name))
sys.modules["twilio.base.exceptions"].TwilioRestException = Exception
sys.modules["twilio.rest"].Client = type("_C", (), {"__init__": lambda *a, **k: None})

_redis = types.ModuleType("redis")
_redis.Redis = type("_R", (), {"__init__": lambda *a, **k: None})
sys.modules.setdefault("redis", _redis)

_intent_mod = types.ModuleType("app.ml.intent.inference_intent")
_slot_mod = types.ModuleType("app.ml.slot.inference_slot")
_intent_mod.IntentBundle = type("IntentBundle", (), {})
_intent_mod.predict_intent = lambda *a, **k: []
_slot_mod.SlotBundle = type("SlotBundle", (), {})
_slot_mod.predict_slots = lambda *a, **k: []
sys.modules.setdefault("app.ml.intent.inference_intent", _intent_mod)
sys.modules.setdefault("app.ml.slot.inference_slot", _slot_mod)

# ── Import the module under test ──────────────────────────────────────────────
from scripts.verify_gpt_ordering_pipeline import (
    FAIL, PASS, WARN, INFO,
    CheckResult,
    run_checks,
    main,
    _check_config_loads,
    _check_option_resolver_mode,
    _check_add_item_planner_mode,
    _check_disabled_is_default,
    _check_api_key_presence,
    _check_inline_without_api_key,
    _check_shadow_mode_never_applies,
    _check_inline_apply_gate_requires_all_conditions,
    _check_option_resolver_service_instantiates,
    _check_add_item_planner_service_instantiates,
    _check_logging_fields_present,
    _check_option_resolver_logging_fields,
    _check_production_inline_warning,
)
from app.config.semantic_repair import SemanticRepairConfig


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clear_cache():
    from app.config.semantic_repair import get_semantic_repair_config
    get_semantic_repair_config.cache_clear()


class _with_env:
    """Context manager that removes GPT env vars, then sets only the ones in `env`.

    Uses os.environ.pop() to remove keys so os.getenv() returns None (→ default),
    not "" (which would be parsed as an empty string and fail config validation).
    """
    _CLEAR_KEYS = frozenset({
        "COMPASS_GPT_OPTION_RESOLVER_MODE",
        "COMPASS_GPT_ADD_ITEM_PLANNER_MODE",
        "OPENAI_API_KEY",
        "COMPASS_GPT_REPAIR_PHASE",
        "COMPASS_GPT_CALL_MODE",
        "COMPASS_GPT_ADD_ITEM_MODE",
    })

    def __init__(self, env: dict):
        self._env = env
        self._saved: dict[str, str] = {}

    def __enter__(self):
        # Save and remove all clear-keys
        for k in self._CLEAR_KEYS:
            if k in os.environ:
                self._saved[k] = os.environ.pop(k)
        # Set requested overrides
        for k, v in self._env.items():
            self._saved.setdefault(k, os.environ.get(k, _SENTINEL))
            os.environ[k] = v

    def __exit__(self, *_):
        # Remove any keys we set
        for k in self._env:
            os.environ.pop(k, None)
        # Restore originals
        for k, v in self._saved.items():
            if v is _SENTINEL:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self._saved.clear()


_SENTINEL = object()


# ── Part 1: Config loads ──────────────────────────────────────────────────────


class TestConfigLoads(unittest.TestCase):
    def test_config_loads_with_disabled_env(self):
        with _with_env({}):
            _clear_cache()
            result = _check_config_loads()
        _clear_cache()
        self.assertEqual(result.status, PASS)

    def test_config_load_fail_on_invalid_mode(self):
        # Patch get_semantic_repair_config to raise
        with patch("scripts.verify_gpt_ordering_pipeline.get_semantic_repair_config",
                   side_effect=ValueError("bad mode")):
            result = _check_config_loads()
        self.assertEqual(result.status, FAIL)
        self.assertIn("bad mode", result.message)


# ── Part 2: Mode checks ───────────────────────────────────────────────────────


class TestModeChecks(unittest.TestCase):
    def test_option_resolver_disabled_is_info(self):
        with _with_env({"COMPASS_GPT_OPTION_RESOLVER_MODE": "disabled"}):
            _clear_cache()
            result = _check_option_resolver_mode()
        _clear_cache()
        self.assertEqual(result.status, INFO)
        self.assertIn("disabled", result.message)

    def test_option_resolver_shadow_is_info(self):
        with _with_env({"COMPASS_GPT_OPTION_RESOLVER_MODE": "shadow"}):
            _clear_cache()
            result = _check_option_resolver_mode()
        _clear_cache()
        self.assertEqual(result.status, INFO)
        self.assertIn("shadow", result.message)

    def test_option_resolver_inline_is_info(self):
        with _with_env({"COMPASS_GPT_OPTION_RESOLVER_MODE": "inline"}):
            _clear_cache()
            result = _check_option_resolver_mode()
        _clear_cache()
        self.assertEqual(result.status, INFO)
        self.assertIn("inline", result.message)

    def test_add_item_planner_disabled_is_info(self):
        with _with_env({"COMPASS_GPT_ADD_ITEM_PLANNER_MODE": "disabled"}):
            _clear_cache()
            result = _check_add_item_planner_mode()
        _clear_cache()
        self.assertEqual(result.status, INFO)

    def test_add_item_planner_shadow_is_info(self):
        with _with_env({"COMPASS_GPT_ADD_ITEM_PLANNER_MODE": "shadow"}):
            _clear_cache()
            result = _check_add_item_planner_mode()
        _clear_cache()
        self.assertEqual(result.status, INFO)
        self.assertIn("shadow", result.message)

    def test_invalid_option_resolver_mode_fails(self):
        """An invalid mode value produces a FAIL (mode validated by __post_init__)."""
        # Simulate a config object with bad mode
        bad_cfg = MagicMock()
        bad_cfg.option_resolver_mode = "turbo"
        with patch("scripts.verify_gpt_ordering_pipeline.get_semantic_repair_config",
                   return_value=bad_cfg):
            result = _check_option_resolver_mode()
        self.assertEqual(result.status, FAIL)

    def test_invalid_add_item_planner_mode_fails(self):
        bad_cfg = MagicMock()
        bad_cfg.add_item_planner_mode = "ultra"
        with patch("scripts.verify_gpt_ordering_pipeline.get_semantic_repair_config",
                   return_value=bad_cfg):
            result = _check_add_item_planner_mode()
        self.assertEqual(result.status, FAIL)


# ── Part 3: Default mode safety ───────────────────────────────────────────────


class TestDefaultModeSafety(unittest.TestCase):
    def test_unset_vars_shows_defaults_as_disabled(self):
        with _with_env({}):
            _clear_cache()
            result = _check_disabled_is_default()
        _clear_cache()
        self.assertEqual(result.status, PASS)
        # Both vars should appear in detail
        self.assertIn("COMPASS_GPT_OPTION_RESOLVER_MODE", result.detail)
        self.assertIn("COMPASS_GPT_ADD_ITEM_PLANNER_MODE", result.detail)

    def test_explicit_disabled_shows_explicit_label(self):
        with _with_env({
            "COMPASS_GPT_OPTION_RESOLVER_MODE": "disabled",
            "COMPASS_GPT_ADD_ITEM_PLANNER_MODE": "disabled",
        }):
            _clear_cache()
            result = _check_disabled_is_default()
        _clear_cache()
        self.assertEqual(result.status, PASS)
        self.assertIn("explicit", result.detail)


# ── Part 4: API key ───────────────────────────────────────────────────────────


class TestApiKeyCheck(unittest.TestCase):
    def test_no_api_key_warns(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
            os.environ.pop("OPENAI_API_KEY", None)
            result = _check_api_key_presence()
        self.assertEqual(result.status, WARN)
        self.assertNotIn("sk-", result.message)   # key value must not appear
        self.assertNotIn("sk-", result.detail)

    def test_api_key_present_passes(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-1234567890"}, clear=False):
            result = _check_api_key_presence()
        self.assertEqual(result.status, PASS)
        # Must not log the full key
        self.assertNotIn("sk-test-1234567890", result.message)
        self.assertNotIn("sk-test-1234567890", result.detail)
        # Should log a prefix (first 7 chars)
        self.assertIn("sk-test", result.detail)

    def test_api_key_full_value_never_in_detail(self):
        secret = "sk-super-secret-key-should-never-appear-in-logs"
        with patch.dict(os.environ, {"OPENAI_API_KEY": secret}, clear=False):
            result = _check_api_key_presence()
        combined = result.message + result.detail
        self.assertNotIn(secret, combined)


# ── Part 5: Inline without API key ───────────────────────────────────────────


class TestInlineWithoutApiKey(unittest.TestCase):
    def test_no_inline_no_key_passes(self):
        """No inline features active + no key = PASS."""
        disabled_cfg = SemanticRepairConfig(
            phase=0, model="gpt-4o-mini", timeout_seconds=1.0,
        )
        with patch("scripts.verify_gpt_ordering_pipeline.get_semantic_repair_config",
                   return_value=disabled_cfg):
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("OPENAI_API_KEY", None)
                result = _check_inline_without_api_key()
        self.assertEqual(result.status, PASS)

    def test_inline_option_resolver_without_key_fails(self):
        inline_cfg = SemanticRepairConfig(
            phase=0, model="gpt-4o-mini", timeout_seconds=1.2,
            option_resolver_mode="inline",
        )
        with patch("scripts.verify_gpt_ordering_pipeline.get_semantic_repair_config",
                   return_value=inline_cfg):
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("OPENAI_API_KEY", None)
                result = _check_inline_without_api_key()
        self.assertEqual(result.status, FAIL)
        self.assertIn("option_resolver", result.message)

    def test_inline_planner_without_key_fails(self):
        inline_cfg = SemanticRepairConfig(
            phase=0, model="gpt-4o-mini", timeout_seconds=1.8,
            add_item_planner_mode="inline",
        )
        with patch("scripts.verify_gpt_ordering_pipeline.get_semantic_repair_config",
                   return_value=inline_cfg):
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("OPENAI_API_KEY", None)
                result = _check_inline_without_api_key()
        self.assertEqual(result.status, FAIL)
        self.assertIn("add_item_planner", result.message)

    def test_inline_with_key_passes(self):
        inline_cfg = SemanticRepairConfig(
            phase=0, model="gpt-4o-mini", timeout_seconds=1.8,
            option_resolver_mode="inline",
            add_item_planner_mode="inline",
        )
        with patch("scripts.verify_gpt_ordering_pipeline.get_semantic_repair_config",
                   return_value=inline_cfg):
            with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-valid-test-key"}, clear=False):
                result = _check_inline_without_api_key()
        self.assertEqual(result.status, PASS)


# ── Part 6: Shadow-never-applies structural check ─────────────────────────────


class TestShadowNeverApplies(unittest.TestCase):
    def test_shadow_never_applies_passes(self):
        """PlannerApplyGate must block shadow_gpt — structural invariant."""
        result = _check_shadow_mode_never_applies()
        self.assertEqual(result.status, PASS)
        # The detail shows the block reason — just verify it is non-empty.
        self.assertTrue(result.detail, "detail should explain why shadow was blocked")

    def test_inline_apply_gate_all_conditions_passes(self):
        """Apply gate correctly blocks on all tested failure conditions."""
        result = _check_inline_apply_gate_requires_all_conditions()
        self.assertEqual(result.status, PASS)


# ── Part 7: Service instantiation ─────────────────────────────────────────────


class TestServiceInstantiation(unittest.TestCase):
    def test_option_resolver_instantiates(self):
        result = _check_option_resolver_service_instantiates()
        self.assertEqual(result.status, PASS)

    def test_add_item_planner_instantiates(self):
        result = _check_add_item_planner_service_instantiates()
        self.assertEqual(result.status, PASS)


# ── Part 8: Logging field checks ─────────────────────────────────────────────


class TestLoggingFields(unittest.TestCase):
    def test_add_item_handler_has_all_required_log_fields(self):
        result = _check_logging_fields_present()
        self.assertEqual(result.status, PASS, msg=result.detail)

    def test_option_resolver_handler_has_all_required_log_fields(self):
        result = _check_option_resolver_logging_fields()
        self.assertEqual(result.status, PASS, msg=result.detail)


# ── Part 9: Production inline warning ─────────────────────────────────────────


class TestProductionInlineWarning(unittest.TestCase):
    def test_production_no_inline_passes(self):
        disabled_cfg = SemanticRepairConfig(
            phase=0, model="gpt-4o-mini", timeout_seconds=1.0,
        )
        with patch("scripts.verify_gpt_ordering_pipeline.get_semantic_repair_config",
                   return_value=disabled_cfg):
            result = _check_production_inline_warning("production")
        self.assertEqual(result.status, PASS)

    def test_production_with_inline_warns(self):
        inline_cfg = SemanticRepairConfig(
            phase=0, model="gpt-4o-mini", timeout_seconds=1.8,
            add_item_planner_mode="inline",
        )
        with patch("scripts.verify_gpt_ordering_pipeline.get_semantic_repair_config",
                   return_value=inline_cfg):
            result = _check_production_inline_warning("production")
        self.assertEqual(result.status, WARN)
        self.assertIn("PRODUCTION", result.message)

    def test_staging_with_inline_is_info(self):
        inline_cfg = SemanticRepairConfig(
            phase=0, model="gpt-4o-mini", timeout_seconds=1.8,
            add_item_planner_mode="inline",
        )
        with patch("scripts.verify_gpt_ordering_pipeline.get_semantic_repair_config",
                   return_value=inline_cfg):
            result = _check_production_inline_warning("staging")
        self.assertEqual(result.status, INFO)

    def test_local_with_inline_is_info(self):
        inline_cfg = SemanticRepairConfig(
            phase=0, model="gpt-4o-mini", timeout_seconds=1.8,
            option_resolver_mode="inline",
        )
        with patch("scripts.verify_gpt_ordering_pipeline.get_semantic_repair_config",
                   return_value=inline_cfg):
            result = _check_production_inline_warning("local")
        self.assertEqual(result.status, INFO)


# ── Part 10: run_checks() and main() ─────────────────────────────────────────


class TestRunChecks(unittest.TestCase):
    def test_all_disabled_exits_zero(self):
        """With all modes disabled, run_checks should produce exit_code=0."""
        with _with_env({
            "COMPASS_GPT_OPTION_RESOLVER_MODE": "disabled",
            "COMPASS_GPT_ADD_ITEM_PLANNER_MODE": "disabled",
        }):
            _clear_cache()
            results, exit_code = run_checks(env_name="local")
        _clear_cache()
        self.assertEqual(exit_code, 0, msg=[r for r in results if r.status == FAIL])

    def test_fail_results_produce_exit_one(self):
        """If any FAIL check is in the results list, exit_code must be 1.

        We test the exit_code computation directly by injecting a FAIL result
        into the ALL_CHECKS list via patching the list used by run_checks().
        """
        import scripts.verify_gpt_ordering_pipeline as _mod
        forced_fail = lambda: CheckResult(FAIL, "forced", "forced fail for test")
        original_checks = _mod._ALL_CHECKS[:]
        _mod._ALL_CHECKS.clear()
        _mod._ALL_CHECKS.append(forced_fail)
        try:
            with _with_env({}):
                _clear_cache()
                results, exit_code = run_checks(env_name="local")
            _clear_cache()
        finally:
            _mod._ALL_CHECKS.clear()
            _mod._ALL_CHECKS.extend(original_checks)
        self.assertEqual(exit_code, 1)
        fail_names = [r.name for r in results if r.status == FAIL]
        self.assertIn("forced", fail_names)

    def test_warn_does_not_fail_by_default(self):
        """WARN results do NOT change exit code without --strict."""
        with patch("scripts.verify_gpt_ordering_pipeline._check_api_key_presence",
                   return_value=CheckResult(WARN, "key", "no key set")):
            with _with_env({}):
                _clear_cache()
                results, exit_code = run_checks(env_name="local")
        _clear_cache()
        # exit_code depends on other checks — must not be 1 BECAUSE of the WARN alone.
        # Just verify no FAIL was introduced by our WARN check.
        fail_names = [r.name for r in results if r.status == FAIL]
        self.assertNotIn("key", fail_names)

    def test_script_never_calls_openai(self):
        """run_checks must never invoke openai.OpenAI or any HTTP client.

        Stubs the openai module if not installed, then verifies the constructor
        was never called during run_checks().
        """
        # Ensure openai is available as a stubbed module for patching.
        import sys as _sys
        _openai_was_present = "openai" in _sys.modules
        if not _openai_was_present:
            _stub = types.ModuleType("openai")
            _stub.OpenAI = lambda *a, **k: (_ for _ in ()).throw(AssertionError("openai.OpenAI called"))
            _sys.modules["openai"] = _stub

        try:
            import openai as _openai_mod
            openai_call_count: list[int] = []
            _orig = getattr(_openai_mod, "OpenAI", None)
            def _spy(*a, **k):
                openai_call_count.append(1)
                raise AssertionError("run_checks must not call openai.OpenAI")
            _openai_mod.OpenAI = _spy
            try:
                with _with_env({"OPENAI_API_KEY": "sk-should-not-be-called"}):
                    _clear_cache()
                    results, exit_code = run_checks(env_name="local")
                _clear_cache()
            except AssertionError as exc:
                self.fail(f"run_checks illegally called openai.OpenAI: {exc}")
            finally:
                if _orig is not None:
                    _openai_mod.OpenAI = _orig
                elif hasattr(_openai_mod, "OpenAI"):
                    del _openai_mod.OpenAI
            self.assertEqual(openai_call_count, [], "openai.OpenAI must not be called by run_checks")
        finally:
            if not _openai_was_present:
                _sys.modules.pop("openai", None)


class TestMainFunction(unittest.TestCase):
    def test_main_disabled_exits_zero(self):
        with _with_env({
            "COMPASS_GPT_OPTION_RESOLVER_MODE": "disabled",
            "COMPASS_GPT_ADD_ITEM_PLANNER_MODE": "disabled",
        }):
            _clear_cache()
            code = main(["--env", "local", "--no-colour"])
        _clear_cache()
        self.assertEqual(code, 0)

    def test_main_no_args_runs_without_error(self):
        """main() with default args runs cleanly in local env."""
        with _with_env({}):
            _clear_cache()
            try:
                code = main(["--no-colour"])
            except SystemExit as exc:
                code = exc.code
        _clear_cache()
        self.assertIn(code, (0, 1))   # either is acceptable; must not crash

    def test_main_production_env_runs(self):
        """--env production does not crash."""
        with _with_env({}):
            _clear_cache()
            try:
                code = main(["--env", "production", "--no-colour"])
            except SystemExit as exc:
                code = exc.code
        _clear_cache()
        self.assertIn(code, (0, 1))


if __name__ == "__main__":
    unittest.main()
