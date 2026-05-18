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


def _repair_response(intent: str = "add_item"):
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


def _build_phase3_engine(cb: CapturingBackend):
    menu_repo = _build_menu_repo()
    engine = _build_engine(menu_repo, cb, gpt_phase=3)
    engine.gpt_repair = GptRepairService(
        config=SemanticRepairConfig(
            phase=3,
            model="gpt-4o-mini",
            timeout_seconds=3.0,
        )
    )
    return engine


def test_phase3_turn_event_contains_policy_fields() -> None:
    cb = CapturingBackend()
    engine = _build_phase3_engine(cb)

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _repair_response("checkout")
    engine.gpt_repair._client = mock_client

    _turn(engine, _idle_session(), "I want something good to eat")

    event = cb.last
    assert event is not None
    assert event.gpt_policy_mode == "inline_with_timeout"
    assert event.gpt_prompt_bucket == "add_item_plan"
    assert event.gpt_used_inline is True
    assert event.gpt_execution_policy_ms is not None
    assert event.gpt_allowed_intents_json is not None
    assert event.gpt_top_intents_json is not None


def test_phase3_invalid_gpt_json_falls_back_to_local_fsm() -> None:
    cb = CapturingBackend()
    engine = _build_phase3_engine(cb)

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_openai_response("not json")
    engine.gpt_repair._client = mock_client

    output = _turn(engine, _idle_session(), "I want something good to eat")

    event = cb.last
    assert event is not None
    assert output.response_key
    assert event.gpt_parse_error is not None
    assert event.gpt_result_applied is False
    assert event.final_intent_after_gpt == event.local_intent_before_gpt
