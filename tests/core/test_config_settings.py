# tests/core/test_config_settings.py
"""
Tests for Task 9: typed config settings and flow_sets package split.

Coverage:
- PaymentConfig defaults match the old inline os.getenv defaults
- NluConfig defaults match the old inline os.getenv defaults
- RealtimeConfig defaults match the old inline os.getenv defaults
- get_*_config() returns the same object on repeated calls (lru_cache)
- flow_sets package re-exports everything the old flat module exported
- flow_sets.state_groups contains only state constants
- flow_sets.intent_policy contains intent sets and signal functions
- bootstrap.runtime loads all configs without error
- checkout_service module-level constants are still present (test-patchable)
- INTENT_MIN_CONF, TURN_TIMING_ENABLED, MAX_QUEUE_DEPTH still importable
  from their original modules
"""
from __future__ import annotations

import sys
import types


# ── stub heavy deps ───────────────────────────────────────────────────────────
def _stub_heavy_deps() -> None:
    for _n in ("twilio", "twilio.base", "twilio.base.exceptions", "twilio.rest"):
        sys.modules.setdefault(_n, types.ModuleType(_n))
    sys.modules["twilio.base.exceptions"].TwilioRestException = Exception
    sys.modules["twilio.rest"].Client = type(
        "_C", (), {"__init__": lambda *a, **k: None}
    )
    _im = types.ModuleType("app.ml.intent.inference_intent")
    _sm = types.ModuleType("app.ml.slot.inference_slot")
    _im.IntentBundle = type("IB", (), {})  # type: ignore[attr-defined]
    _im.predict_intent = lambda *a, **k: []  # type: ignore[attr-defined]
    _sm.SlotBundle = type("SB", (), {})  # type: ignore[attr-defined]
    _sm.predict_slots = lambda *a, **k: []  # type: ignore[attr-defined]
    sys.modules.setdefault("app.ml.intent.inference_intent", _im)
    sys.modules.setdefault("app.ml.slot.inference_slot", _sm)
    sys.modules.setdefault("torch", types.ModuleType("torch"))


_stub_heavy_deps()
# ─────────────────────────────────────────────────────────────────────────────

from app.config.nlu import NluConfig, get_nlu_config
from app.config.payment import PaymentConfig, get_payment_config
from app.config.realtime import RealtimeConfig, get_realtime_config


# ── PaymentConfig ─────────────────────────────────────────────────────────────

def test_payment_config_is_frozen():
    cfg = get_payment_config()
    try:
        cfg.payment_poll_interval_seconds = 999  # type: ignore
        assert False, "Should have raised"
    except (AttributeError, TypeError):
        pass


def test_payment_config_poll_interval_default():
    cfg = get_payment_config()
    assert cfg.payment_poll_interval_seconds == 6.0


def test_payment_config_poll_max_duration_default():
    cfg = get_payment_config()
    assert cfg.payment_poll_max_duration_seconds == 900.0


def test_payment_config_checkout_dir_default():
    from pathlib import Path
    cfg = get_payment_config()
    assert cfg.checkout_data_dir == Path("app/data/checkout_sessions")


def test_payment_config_payment_link_dir_default():
    from pathlib import Path
    cfg = get_payment_config()
    assert cfg.payment_link_session_data_dir == Path("app/data/payment_link_sessions")


def test_payment_config_checkout_pending_interval_default():
    cfg = get_payment_config()
    assert cfg.checkout_pending_reminder_interval_seconds == 30.0


def test_payment_config_payment_pending_interval_default():
    cfg = get_payment_config()
    assert cfg.payment_pending_reminder_interval_seconds == 30.0


def test_payment_config_reverse_geocode_url_default():
    cfg = get_payment_config()
    assert "nominatim.openstreetmap.org" in cfg.reverse_geocode_url


def test_payment_config_cached():
    assert get_payment_config() is get_payment_config()


# ── NluConfig ─────────────────────────────────────────────────────────────────

def test_nlu_config_is_frozen():
    cfg = get_nlu_config()
    try:
        cfg.intent_conf_threshold = 0.99  # type: ignore
        assert False, "Should have raised"
    except (AttributeError, TypeError):
        pass


def test_nlu_config_intent_threshold_default():
    cfg = get_nlu_config()
    assert cfg.intent_conf_threshold == 0.55


def test_nlu_config_max_queue_depth_default():
    cfg = get_nlu_config()
    assert cfg.max_item_queue_depth == 20


def test_nlu_config_cached():
    assert get_nlu_config() is get_nlu_config()


# ── RealtimeConfig ────────────────────────────────────────────────────────────

def test_realtime_config_is_frozen():
    cfg = get_realtime_config()
    try:
        cfg.route_debug_enabled = True  # type: ignore
        assert False, "Should have raised"
    except (AttributeError, TypeError):
        pass


def test_realtime_config_route_debug_default():
    cfg = get_realtime_config()
    assert cfg.route_debug_enabled is False


def test_realtime_config_turn_timing_default():
    cfg = get_realtime_config()
    assert cfg.turn_timing_enabled is False


def test_realtime_config_nlu_log_path_default():
    cfg = get_realtime_config()
    assert cfg.nlu_json_log_path is None


def test_realtime_config_cached():
    assert get_realtime_config() is get_realtime_config()


# ── bootstrap.runtime ─────────────────────────────────────────────────────────

def test_bootstrap_runtime_exposes_all_configs():
    import bootstrap.runtime as rt
    assert isinstance(rt.payment_config, PaymentConfig)
    assert isinstance(rt.nlu_config, NluConfig)
    assert isinstance(rt.realtime_config, RealtimeConfig)


def test_bootstrap_runtime_configs_are_same_cached_instances():
    import bootstrap.runtime as rt
    assert rt.payment_config is get_payment_config()
    assert rt.nlu_config is get_nlu_config()
    assert rt.realtime_config is get_realtime_config()


# ── flow_sets package backward-compat ─────────────────────────────────────────

def test_flow_sets_state_group_imports():
    from app.state_machine.flow_sets import (
        ADD_ITEM_FLOW_STATES,
        ACTIVE_TASK_STATES,
        DELIVERY_GATING_STATES,
        MID_ITEM_BLOCKING_STATES,
        ORDER_FLOW_STATES,
    )
    assert len(ADD_ITEM_FLOW_STATES) == 6
    assert len(ORDER_FLOW_STATES) == 2
    assert len(DELIVERY_GATING_STATES) == 2
    assert len(MID_ITEM_BLOCKING_STATES) == 8
    assert len(ACTIVE_TASK_STATES) == 14


def test_flow_sets_intent_policy_imports():
    from app.state_machine.flow_sets import (
        DELIVERY_GATING_ALLOWED_CONTROL_INTENTS,
        GROUP_DONE_INTENTS,
        ORDERING_INTENTS,
        SOFT_SWITCH_INTENTS,
        SOFT_SWITCH_INTENTS_REDUCED,
        WAITING_STATE_ALLOWED_CONTROL_INTENTS,
    )
    assert len(GROUP_DONE_INTENTS) == 5
    assert len(ORDERING_INTENTS) == 8
    assert SOFT_SWITCH_INTENTS_REDUCED == SOFT_SWITCH_INTENTS - GROUP_DONE_INTENTS


def test_flow_sets_signal_functions():
    from app.state_machine.flow_sets import (
        looks_like_done_answer,
        looks_like_skip_answer,
        looks_like_more_options_answer,
    )
    assert looks_like_done_answer("done")
    assert looks_like_done_answer("ok done please")
    assert looks_like_skip_answer("no thanks")
    assert looks_like_more_options_answer("what else")
    assert not looks_like_done_answer("add a burger")


def test_flow_sets_word_sets():
    from app.state_machine.flow_sets import DONE_WORDS, SKIP_WORDS, MORE_OPTIONS_WORDS
    assert "done" in DONE_WORDS
    assert "skip" in SKIP_WORDS
    assert "options" in MORE_OPTIONS_WORDS


def test_flow_sets_state_groups_submodule():
    from app.state_machine.flow_sets.state_groups import (
        ADD_ITEM_FLOW_STATES,
        ACTIVE_TASK_STATES,
    )
    assert ADD_ITEM_FLOW_STATES  # non-empty


def test_flow_sets_intent_policy_submodule():
    from app.state_machine.flow_sets.intent_policy import (
        GROUP_DONE_INTENTS,
        looks_like_done_answer,
    )
    assert GROUP_DONE_INTENTS
    assert looks_like_done_answer("done")


# ── migrated module constants still importable ────────────────────────────────

def test_intent_min_conf_still_importable():
    from app.core.nlu_orchestrator import INTENT_MIN_CONF
    assert INTENT_MIN_CONF == 0.55


def test_turn_timing_enabled_still_importable():
    from app.core.turn_diagnostics import TURN_TIMING_ENABLED
    assert TURN_TIMING_ENABLED is False


def test_max_queue_depth_still_importable():
    from app.core.item_queue_service import MAX_QUEUE_DEPTH
    assert MAX_QUEUE_DEPTH == 20


def test_checkout_service_module_level_constants():
    import app.services.checkout_service as cs
    from pathlib import Path
    assert isinstance(cs.CHECKOUT_DATA_DIR, Path)
    assert cs.PAYMENT_POLL_INTERVAL_SECONDS == 6.0
    assert "nominatim" in cs.REVERSE_GEOCODE_URL


def test_payment_flow_orchestrator_module_constants():
    import app.core.payment_flow_orchestrator as pfo
    assert pfo.CHECKOUT_PENDING_REMINDER_INTERVAL_SECONDS == 30.0
    assert pfo.PAYMENT_PENDING_REMINDER_INTERVAL_SECONDS == 30.0
