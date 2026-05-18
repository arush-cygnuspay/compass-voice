# tests/core/test_turn_engine_phase3_policy.py
"""Phase 3 option resolver integration tests through TurnEngine.

These tests replaced the old stale assertions for removed TurnEvent fields
(gpt_policy_mode, gpt_prompt_bucket, gpt_used_inline, gpt_execution_policy_ms,
gpt_allowed_intents_json, gpt_top_intents_json, gpt_result_applied) which were
cleaned out of TurnEvent in the Phase 2 audit remediation.

What is covered here:
  - Phase 3 config defaults are safe (option_resolver_mode == "disabled")
  - Phase 3 option resolver service is importable and usable standalone
  - GPT option resolver in disabled mode does not block local FSM behavior
  - Invalid GPT JSON continues to fall back to local FSM (existing Phase 2 behavior)
  - Existing Phase 2 TurnEvent fields are still populated correctly
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

from app.config.semantic_repair import SemanticRepairConfig
from app.nlu.semantic_repair.gpt_repair_result import GPT_NOT_CALLED
from app.nlu.semantic_repair.repair_service import GptRepairService
from tests.core.test_turn_engine_phase2_shadow import (
    CapturingBackend,
    _build_engine,
    _build_menu_repo,
    _idle_session,
    _make_openai_response,
    _turn,
)


def _repair_response(intent: str = "add_item") -> object:
    payload = json.dumps({
        "decision": "repair",
        "repaired_intent": intent,
        "repaired_control_intent": None,
        "slot_corrections": {},
        "confidence": 0.91,
        "reason": "phase3 repair",
        "requires_handler_validation": True,
    })
    return _make_openai_response(payload)


def _build_phase3_engine(cb: CapturingBackend) -> object:
    menu_repo = _build_menu_repo()
    engine = _build_engine(menu_repo, cb, gpt_phase=2)  # Phase 2 is current stable
    engine.gpt_repair = GptRepairService(
        config=SemanticRepairConfig(
            phase=2,
            model="gpt-4o-mini",
            timeout_seconds=3.0,
        )
    )
    return engine


# ---------------------------------------------------------------------------
# Phase 3 config defaults
# ---------------------------------------------------------------------------


def test_phase3_option_resolver_config_defaults_are_safe() -> None:
    """Phase 3 config defaults must all be safe (disabled mode, conservative thresholds)."""
    cfg = SemanticRepairConfig(phase=3, model="gpt-4o-mini", timeout_seconds=3.0)
    assert cfg.option_resolver_mode == "disabled"
    assert cfg.option_resolver_min_confidence == 0.75
    assert cfg.option_resolver_repeat_threshold == 2
    assert cfg.option_resolver_timeout_ms == 1200


def test_phase3_option_resolver_mode_validation() -> None:
    """Only valid option_resolver_mode values are accepted."""
    import pytest

    with pytest.raises(ValueError, match="option_resolver_mode"):
        SemanticRepairConfig(
            phase=3,
            model="gpt-4o-mini",
            timeout_seconds=3.0,
            option_resolver_mode="invalid_mode",
        )


def test_phase3_option_resolver_shadow_mode_accepted() -> None:
    cfg = SemanticRepairConfig(
        phase=3, model="gpt-4o-mini", timeout_seconds=3.0,
        option_resolver_mode="shadow",
    )
    assert cfg.option_resolver_mode == "shadow"


def test_phase3_option_resolver_inline_mode_accepted() -> None:
    cfg = SemanticRepairConfig(
        phase=3, model="gpt-4o-mini", timeout_seconds=3.0,
        option_resolver_mode="inline",
    )
    assert cfg.option_resolver_mode == "inline"


# ---------------------------------------------------------------------------
# Phase 3 option resolver service — standalone smoke test
# ---------------------------------------------------------------------------


def test_phase3_option_resolver_service_disabled_returns_skipped() -> None:
    """Option resolver with mode=disabled immediately returns skipped result."""
    from app.nlu.semantic_repair.option_resolver_service import GptOptionResolverService
    from app.state_machine.models.pending_item_models import (
        PendingModifierChoice,
        PendingModifierGroup,
    )

    cfg = SemanticRepairConfig(
        phase=3, model="gpt-4o-mini", timeout_seconds=3.0,
        option_resolver_mode="disabled",
    )
    svc = GptOptionResolverService(config=cfg)
    group = PendingModifierGroup(
        group_id="g1",
        name="Cheese",
        is_required=True,
        min_selector=1,
        max_selector=1,
        choices=[
            PendingModifierChoice(
                modifier_id="m1", name="Mozzarella Cheese",
                group_id="g1", normalized_name="mozzarella cheese",
            )
        ],
    )
    result = svc.run(
        user_text="macarola cheese",
        item_name="Burger",
        group=group,
        existing_selections=[],
        local_resolved=False,
    )
    assert result.gpt_called is False
    assert result.safe_to_apply is False
    assert result.decision == "skipped"


def test_phase3_removed_event_fields_do_not_exist() -> None:
    """Confirm removed Phase 2-scope-creep TurnEvent fields are gone."""
    from app.diagnostics.turn_event import TurnEvent

    te = TurnEvent(
        session_id="s",
        turn_index=1,
        state_before="idle", state_after="idle", next_state="idle",
        pending_action="", current_prompt_field="", current_item_id="",
        current_item_name="", raw_user_text="", user_text="", normalized_text="",
        pred_main_intent="", pred_sub_intent="", pred_intent="",
        pred_intent_confidence=None, slot_model_ran=False, slots=(),
        response_key="", response_text="", command=None,
        normalized_values={}, missing_required_fields=(),
        reprompt_field="", reprompt_count=0, reprompt_escalated=False,
        reprompt_escalation_count=0, fallback_triggered=False,
        fallback_reason="", fallback_count=0,
        slot_extraction_failed=False, slot_extraction_failure_count=0,
        invalid_modifier=False, invalid_modifier_count=0,
        user_repeated=False, repeated_user_turn_count=0,
    )
    # These fields were REMOVED in Phase 2 cleanup; they must not exist.
    assert not hasattr(te, "gpt_policy_mode"), "gpt_policy_mode must be removed"
    assert not hasattr(te, "gpt_prompt_bucket"), "gpt_prompt_bucket must be removed"
    assert not hasattr(te, "gpt_used_inline"), "gpt_used_inline must be removed"
    assert not hasattr(te, "gpt_used_shadow"), "gpt_used_shadow must be removed"
    assert not hasattr(te, "gpt_result_applied"), "gpt_result_applied must be removed"
    assert not hasattr(te, "gpt_result_rejected"), "gpt_result_rejected must be removed"
    assert not hasattr(te, "gpt_execution_policy_ms"), "gpt_execution_policy_ms must be removed"
    assert not hasattr(te, "gpt_allowed_intents_json"), "gpt_allowed_intents_json must be removed"
    assert not hasattr(te, "gpt_top_intents_json"), "gpt_top_intents_json must be removed"


# ---------------------------------------------------------------------------
# Fallback behavior with invalid GPT JSON (Phase 2 regression guard)
# ---------------------------------------------------------------------------


def test_phase3_invalid_gpt_json_falls_back_to_local_fsm() -> None:
    """Invalid GPT JSON response must not crash the turn and must fall back to local FSM."""
    cb = CapturingBackend()
    engine = _build_phase3_engine(cb)

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_openai_response("not json")
    engine.gpt_repair._client = mock_client

    output = _turn(engine, _idle_session(), "I want something good to eat")

    event = cb.last
    assert event is not None
    # Turn must produce a valid response key.
    assert output.response_key
    # GPT parse error must be captured in the event.
    assert event.gpt_parse_error is not None
    # GPT result must never be applied (Phase 2 shadow contract).
    assert event.gpt_applied is False
    # Local FSM intent is preserved.
    assert event.final_intent_after_gpt == event.local_intent_before_gpt
