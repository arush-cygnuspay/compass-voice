# tests/tools/test_replay_phase3_option_resolver.py
"""Tests for the Phase 3.5 Option Resolver Offline Replay Harness.

Covers:
  - disabled mode never calls GPT
  - shadow mode calls GPT but never marks would_apply=True
  - inline mode can mark would_apply=True when validator passes
  - malformed log rows are skipped with error captured
  - missing optional fields do not crash replay
  - summary counts are correct
  - report JSONL is parseable
  - markdown summary is generated
  - built-in fixtures are complete and valid
  - parse_jsonl_row handles nested JSONL format correctly
  - ReplaySummaryBuilder aggregates correctly
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from unittest.mock import patch

from app.config.semantic_repair import SemanticRepairConfig
from app.nlu.semantic_repair.option_resolver_replay import (
    BUILT_IN_FIXTURES,
    Phase3OptionResolverReplayHarness,
    ReplayInputTurn,
    ReplayResult,
    ReplaySummaryBuilder,
    _build_synthetic_group,
    parse_jsonl_row,
)

# Fake API key used across all mock-GPT tests so the service's env check passes.
_FAKE_API_KEY = {"OPENAI_API_KEY": "sk-test-replay-harness"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cfg(mode: str = "inline") -> SemanticRepairConfig:
    return SemanticRepairConfig(
        phase=3,
        model="gpt-4o-mini",
        timeout_seconds=1.0,
        option_resolver_mode=mode,
        option_resolver_timeout_ms=500,
        option_resolver_min_confidence=0.75,
        option_resolver_repeat_threshold=2,
    )


def _mock_client(responses: dict[str, dict]) -> Any:
    """Deterministic fake OpenAI client keyed by substring of user_text."""
    client = MagicMock()

    def _create(*, messages, **kwargs):
        user_msg = next(
            (m["content"] for m in messages if m.get("role") == "user"), ""
        )
        try:
            payload = json.loads(user_msg)
            text = payload.get("text", "").lower()
        except Exception:
            text = user_msg.lower()

        resp_dict = {
            "decision": "no_match",
            "selected_names": [],
            "confidence": 0.05,
            "reason_code": "no_match",
        }
        for pattern, resp in responses.items():
            if pattern.lower() in text:
                resp_dict = resp
                break

        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = json.dumps(resp_dict)
        return mock_resp

    client.chat.completions.create.side_effect = _create
    return client


def _harness(mode: str = "inline", client: Any = None) -> Phase3OptionResolverReplayHarness:
    return Phase3OptionResolverReplayHarness(
        config=_cfg(mode),
        use_live_gpt=True,  # allow calls; client is overridden by mock
        mock_client=client,
    )


def _fixture_turn(
    *,
    user_text: str = "macarola cheese",
    choice_names: tuple[str, ...] = ("American Cheese", "Mozzarella Cheese", "Cheddar Cheese"),
    repeat_count: int = 0,
    has_correction_signal: bool | None = None,
) -> ReplayInputTurn:
    return ReplayInputTurn(
        user_text=user_text,
        state_before="waiting_for_modifier",
        choice_names=choice_names,
        group_name="Cheese",
        item_name="Burger",
        repeat_count=repeat_count,
        has_correction_signal=has_correction_signal,
    )


# ---------------------------------------------------------------------------
# parse_jsonl_row tests
# ---------------------------------------------------------------------------


class TestParseJsonlRow:
    def test_minimal_valid_row(self) -> None:
        row = {
            "state_before": "waiting_for_modifier",
            "normalized_text": "macarola cheese",
            "local": {"intent": "add_item", "confidence": 0.72, "slots": [], "top_intents": []},
            "allowed": {"choices": ["Mozzarella Cheese", "American Cheese"]},
        }
        turn = parse_jsonl_row(row)
        assert turn is not None
        assert turn.user_text == "macarola cheese"
        assert turn.state_before == "waiting_for_modifier"
        assert "Mozzarella Cheese" in turn.choice_names
        assert turn.local_intent == "add_item"
        assert turn.local_confidence == pytest.approx(0.72)

    def test_filters_by_state(self) -> None:
        row = {
            "state_before": "idle",
            "normalized_text": "burger",
            "local": {},
            "allowed": {"choices": []},
        }
        result = parse_jsonl_row(row, filter_state="waiting_for_modifier")
        assert result is None

    def test_matching_state_filter_passes(self) -> None:
        row = {
            "state_before": "waiting_for_modifier",
            "normalized_text": "mozarella",
            "local": {},
            "allowed": {"choices": ["Mozzarella Cheese"]},
        }
        turn = parse_jsonl_row(row, filter_state="waiting_for_modifier")
        assert turn is not None

    def test_empty_user_text_returns_none(self) -> None:
        row = {
            "state_before": "waiting_for_modifier",
            "normalized_text": "",
            "local": {},
            "allowed": {"choices": ["A"]},
        }
        assert parse_jsonl_row(row) is None

    def test_missing_fields_return_none_gracefully(self) -> None:
        row = {
            "state_before": "waiting_for_modifier",
            "normalized_text": "test",
        }
        turn = parse_jsonl_row(row)
        assert turn is not None
        assert turn.choice_names == ()
        assert turn.local_intent is None
        assert turn.local_confidence is None

    def test_source_turn_id_built_from_session_and_index(self) -> None:
        row = {
            "session_id": "sess1",
            "turn_index": 5,
            "state_before": "waiting_for_modifier",
            "normalized_text": "test",
            "local": {},
            "allowed": {},
        }
        turn = parse_jsonl_row(row)
        assert turn is not None
        assert turn.source_turn_id == "sess1:5"

    def test_response_key_parsed(self) -> None:
        row = {
            "state_before": "waiting_for_modifier",
            "normalized_text": "test",
            "response_key": "ask_for_modifier",
            "local": {},
            "allowed": {},
        }
        turn = parse_jsonl_row(row)
        assert turn is not None
        assert turn.response_key_before == "ask_for_modifier"

    def test_malformed_confidence_defaults_to_none(self) -> None:
        row = {
            "state_before": "waiting_for_modifier",
            "normalized_text": "test",
            "local": {"confidence": "not-a-float"},
            "allowed": {},
        }
        turn = parse_jsonl_row(row)
        assert turn is not None
        assert turn.local_confidence is None

    def test_slots_parsed_as_tuple_of_dicts(self) -> None:
        row = {
            "state_before": "waiting_for_modifier",
            "normalized_text": "test",
            "local": {"slots": [{"name": "MODIFIER", "value": "mozarella"}]},
            "allowed": {},
        }
        turn = parse_jsonl_row(row)
        assert turn is not None
        assert len(turn.local_slots) == 1
        assert turn.local_slots[0]["name"] == "MODIFIER"

    def test_non_dict_choices_ignored(self) -> None:
        row = {
            "state_before": "waiting_for_modifier",
            "normalized_text": "test",
            "local": {},
            "allowed": {"choices": [123, None, "Valid Choice"]},
        }
        turn = parse_jsonl_row(row)
        assert turn is not None
        assert "Valid Choice" in turn.choice_names


# ---------------------------------------------------------------------------
# Phase3OptionResolverReplayHarness tests
# ---------------------------------------------------------------------------


class TestReplayHarnessDisabled:
    """disabled mode — GPT must never be called."""

    def test_disabled_single_turn_no_gpt(self) -> None:
        harness = Phase3OptionResolverReplayHarness(config=_cfg("disabled"))
        turn = _fixture_turn()
        result = harness.replay_turn(turn, mode="disabled")
        assert result.gpt_called is False
        assert result.would_apply is False
        assert result.actual_applied is False
        assert result.route_mode == "no_gpt"

    def test_disabled_fixtures_never_call_gpt(self) -> None:
        harness = Phase3OptionResolverReplayHarness(config=_cfg("disabled"))
        for result in harness.replay_fixtures(BUILT_IN_FIXTURES, mode="disabled"):
            assert result.gpt_called is False, (
                f"disabled mode called GPT for: {result.user_text!r}"
            )

    def test_disabled_result_never_marks_would_apply(self) -> None:
        harness = Phase3OptionResolverReplayHarness(config=_cfg("disabled"))
        for result in harness.replay_fixtures(BUILT_IN_FIXTURES, mode="disabled"):
            assert result.would_apply is False

    def test_disabled_actual_applied_always_false(self) -> None:
        harness = Phase3OptionResolverReplayHarness(config=_cfg("disabled"))
        result = harness.replay_turn(_fixture_turn(), mode="disabled")
        assert result.actual_applied is False


class TestReplayHarnessShadow:
    """shadow mode — GPT may be called but would_apply is always False."""

    def _harness_with_mock(self, responses: dict) -> Phase3OptionResolverReplayHarness:
        client = _mock_client(responses)
        return Phase3OptionResolverReplayHarness(
            config=_cfg("shadow"), use_live_gpt=True, mock_client=client
        )

    def test_shadow_gpt_called_for_phonetic_mismatch(self) -> None:
        harness = self._harness_with_mock({
            "macarola": {
                "decision": "select_option",
                "selected_names": ["Mozzarella Cheese"],
                "confidence": 0.92,
                "reason_code": "phonetic_match",
            }
        })
        turn = _fixture_turn(user_text="macarola cheese")
        with patch.dict("os.environ", _FAKE_API_KEY):
            result = harness.replay_turn(turn, mode="shadow")
        assert result.gpt_called is True
        assert result.route_mode == "shadow_gpt"

    def test_shadow_never_marks_would_apply(self) -> None:
        harness = self._harness_with_mock({
            "macarola": {
                "decision": "select_option",
                "selected_names": ["Mozzarella Cheese"],
                "confidence": 0.99,
                "reason_code": "phonetic_match",
            }
        })
        turn = _fixture_turn(user_text="macarola cheese")
        with patch.dict("os.environ", _FAKE_API_KEY):
            result = harness.replay_turn(turn, mode="shadow")
        # Shadow mode: validator always returns safe_to_apply=False
        assert result.would_apply is False
        assert result.safe_to_apply is False

    def test_shadow_actual_applied_always_false(self) -> None:
        harness = self._harness_with_mock({})
        result = harness.replay_turn(_fixture_turn(), mode="shadow")
        assert result.actual_applied is False

    def test_shadow_all_fixtures_never_would_apply(self) -> None:
        harness = self._harness_with_mock({
            "macarola": {"decision": "select_option", "selected_names": ["Mozzarella Cheese"], "confidence": 0.95, "reason_code": "phonetic_match"},
            "mozarella": {"decision": "select_option", "selected_names": ["Mozzarella Cheese"], "confidence": 0.90, "reason_code": "fuzzy_match"},
        })
        with patch.dict("os.environ", _FAKE_API_KEY):
            for result in harness.replay_fixtures(BUILT_IN_FIXTURES, mode="shadow"):
                assert result.would_apply is False, (
                    f"shadow mode marked would_apply for: {result.user_text!r}"
                )


class TestReplayHarnessInline:
    """inline mode — would_apply=True is allowed when validator passes."""

    def _harness_with_mock(self, responses: dict) -> Phase3OptionResolverReplayHarness:
        client = _mock_client(responses)
        return Phase3OptionResolverReplayHarness(
            config=_cfg("inline"), use_live_gpt=True, mock_client=client
        )

    def test_inline_would_apply_true_when_validator_passes(self) -> None:
        harness = self._harness_with_mock({
            "macarola": {
                "decision": "select_option",
                "selected_names": ["Mozzarella Cheese"],
                "confidence": 0.92,
                "reason_code": "phonetic_match",
            }
        })
        turn = _fixture_turn(user_text="macarola cheese")
        with patch.dict("os.environ", _FAKE_API_KEY):
            result = harness.replay_turn(turn, mode="inline")
        assert result.gpt_called is True
        assert result.route_mode == "inline_gpt"
        assert result.would_apply is True
        assert result.actual_applied is False  # always False in replay

    def test_inline_would_apply_false_when_hallucinated_name(self) -> None:
        harness = self._harness_with_mock({
            "macarola": {
                "decision": "select_option",
                "selected_names": ["Hallucinated Option"],
                "confidence": 0.99,
                "reason_code": "exact_match",
            }
        })
        turn = _fixture_turn(user_text="macarola cheese")
        with patch.dict("os.environ", _FAKE_API_KEY):
            result = harness.replay_turn(turn, mode="inline")
        assert result.gpt_called is True
        assert result.would_apply is False  # hallucinated name rejected
        assert result.safe_to_apply is False

    def test_inline_would_apply_false_when_no_match(self) -> None:
        harness = self._harness_with_mock({})  # default: no_match
        turn = _fixture_turn(user_text="zibblequark nonsense")
        result = harness.replay_turn(turn, mode="inline")
        assert result.would_apply is False

    def test_inline_would_apply_false_when_mode_shadow(self) -> None:
        """would_apply uses the mode parameter, not the config mode."""
        harness = self._harness_with_mock({
            "test": {
                "decision": "select_option",
                "selected_names": ["Mozzarella Cheese"],
                "confidence": 0.90,
                "reason_code": "fuzzy_match",
            }
        })
        # Override mode on replay_turn call to shadow even though config is inline
        turn = _fixture_turn(user_text="test macarola")
        result = harness.replay_turn(turn, mode="shadow")
        assert result.would_apply is False  # shadow never would_apply

    def test_inline_actual_applied_always_false(self) -> None:
        harness = self._harness_with_mock({
            "macarola": {"decision": "select_option", "selected_names": ["Mozzarella Cheese"], "confidence": 0.95, "reason_code": "phonetic_match"},
        })
        turn = _fixture_turn(user_text="macarola cheese")
        result = harness.replay_turn(turn, mode="inline")
        assert result.actual_applied is False


# ---------------------------------------------------------------------------
# Robustness tests
# ---------------------------------------------------------------------------


class TestReplayRobustness:
    def test_missing_optional_fields_do_not_crash(self) -> None:
        """ReplayInputTurn with only required fields must not crash the harness."""
        harness = Phase3OptionResolverReplayHarness(config=_cfg("disabled"))
        turn = ReplayInputTurn(
            user_text="test",
            state_before="waiting_for_modifier",
        )
        result = harness.replay_turn(turn, mode="disabled")
        assert result is not None
        assert result.error is None

    def test_empty_choice_names_returns_no_gpt(self) -> None:
        """No options → routing always returns NO_GPT."""
        harness = Phase3OptionResolverReplayHarness(config=_cfg("inline"), use_live_gpt=True)
        turn = ReplayInputTurn(
            user_text="mozarella",
            state_before="waiting_for_modifier",
            choice_names=(),  # no options available
        )
        result = harness.replay_turn(turn, mode="inline")
        assert result.gpt_called is False
        assert result.route_mode == "no_gpt"

    def test_empty_text_returns_no_gpt(self) -> None:
        """Empty text is guarded at routing level."""
        harness = Phase3OptionResolverReplayHarness(config=_cfg("inline"), use_live_gpt=True)
        turn = ReplayInputTurn(
            user_text="",
            state_before="waiting_for_modifier",
            choice_names=("American Cheese",),
        )
        result = harness.replay_turn(turn, mode="inline")
        assert result.gpt_called is False

    def test_replay_turn_never_raises(self) -> None:
        """Even with a crashing mock client, replay_turn returns a result."""
        crash_client = MagicMock()
        crash_client.chat.completions.create.side_effect = RuntimeError("boom")
        harness = Phase3OptionResolverReplayHarness(
            config=_cfg("inline"), use_live_gpt=True, mock_client=crash_client
        )
        turn = _fixture_turn()
        result = harness.replay_turn(turn, mode="inline")
        assert result is not None
        # Either error is captured or GPT call failed gracefully
        assert result.actual_applied is False

    def test_result_to_dict_is_json_serializable(self) -> None:
        harness = Phase3OptionResolverReplayHarness(config=_cfg("disabled"))
        result = harness.replay_turn(_fixture_turn(), mode="disabled")
        d = result.to_dict()
        # Should not raise
        json_str = json.dumps(d)
        assert len(json_str) > 0

    def test_result_to_dict_has_required_keys(self) -> None:
        harness = Phase3OptionResolverReplayHarness(config=_cfg("disabled"))
        result = harness.replay_turn(_fixture_turn(), mode="disabled")
        d = result.to_dict()
        for key in (
            "replay_id", "source_turn_id", "user_text", "state_before", "mode",
            "route_mode", "gpt_called", "gpt_decision", "gpt_selected_names_or_ids",
            "gpt_confidence", "validator_passed", "validator_reject_reason",
            "safe_to_apply", "would_apply", "actual_applied", "error", "latency_ms",
        ):
            assert key in d, f"Missing key in to_dict(): {key!r}"

    def test_actual_applied_always_false_in_dict(self) -> None:
        client = _mock_client({
            "macarola": {"decision": "select_option", "selected_names": ["Mozzarella Cheese"], "confidence": 0.92, "reason_code": "phonetic_match"},
        })
        harness = Phase3OptionResolverReplayHarness(
            config=_cfg("inline"), use_live_gpt=True, mock_client=client
        )
        result = harness.replay_turn(_fixture_turn(), mode="inline")
        d = result.to_dict()
        assert d["actual_applied"] is False


# ---------------------------------------------------------------------------
# JSONL replay tests
# ---------------------------------------------------------------------------


class TestJsonlReplay:
    def _make_jsonl(self, tmp_path: Path, rows: list[dict]) -> Path:
        p = tmp_path / "test_turns.jsonl"
        with p.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        return p

    def _valid_row(
        self,
        user_text: str = "macarola cheese",
        state: str = "waiting_for_modifier",
        choices: list[str] | None = None,
    ) -> dict:
        return {
            "session_id": "s1",
            "turn_index": 1,
            "state_before": state,
            "normalized_text": user_text,
            "response_key": "ask_for_modifier",
            "local": {"intent": "unknown", "confidence": 0.18, "slots": [], "top_intents": []},
            "allowed": {"choices": choices or ["Mozzarella Cheese", "American Cheese"]},
        }

    def test_valid_rows_are_replayed(self, tmp_path: Path) -> None:
        p = self._make_jsonl(tmp_path, [
            self._valid_row("macarola cheese"),
            self._valid_row("mozarella"),
        ])
        harness = Phase3OptionResolverReplayHarness(config=_cfg("disabled"))
        results = list(harness.replay_jsonl(p, mode="disabled"))
        assert len(results) == 2

    def test_malformed_json_line_produces_error_result(self, tmp_path: Path) -> None:
        p = self._make_jsonl(tmp_path, [])
        with p.open("w", encoding="utf-8") as fh:
            fh.write("{broken json\n")
            fh.write(json.dumps(self._valid_row()) + "\n")
        harness = Phase3OptionResolverReplayHarness(config=_cfg("disabled"))
        results = list(harness.replay_jsonl(p, mode="disabled"))
        error_results = [r for r in results if r.error]
        assert len(error_results) >= 1

    def test_filter_by_state(self, tmp_path: Path) -> None:
        p = self._make_jsonl(tmp_path, [
            self._valid_row(state="waiting_for_modifier"),
            self._valid_row(state="idle"),  # should be filtered out
            self._valid_row(state="waiting_for_modifier"),
        ])
        harness = Phase3OptionResolverReplayHarness(config=_cfg("disabled"))
        results = list(harness.replay_jsonl(
            p, mode="disabled", filter_state="waiting_for_modifier"
        ))
        assert len(results) == 2

    def test_max_turns_limits_output(self, tmp_path: Path) -> None:
        p = self._make_jsonl(tmp_path, [
            self._valid_row(f"turn {i}") for i in range(10)
        ])
        harness = Phase3OptionResolverReplayHarness(config=_cfg("disabled"))
        results = list(harness.replay_jsonl(p, mode="disabled", max_turns=3))
        assert len(results) == 3

    def test_silence_rows_skipped(self, tmp_path: Path) -> None:
        p = self._make_jsonl(tmp_path, [
            {
                "state_before": "waiting_for_modifier",
                "normalized_text": "",  # silence
                "local": {},
                "allowed": {"choices": ["A"]},
            },
            self._valid_row("macarola cheese"),
        ])
        harness = Phase3OptionResolverReplayHarness(config=_cfg("disabled"))
        results = list(harness.replay_jsonl(
            p, mode="disabled", filter_state="waiting_for_modifier"
        ))
        # Silence row is skipped by parse_jsonl_row
        assert len(results) == 1

    def test_output_jsonl_parseable(self, tmp_path: Path) -> None:
        p = self._make_jsonl(tmp_path, [
            self._valid_row("macarola cheese"),
            self._valid_row("mozarella"),
        ])
        harness = Phase3OptionResolverReplayHarness(config=_cfg("disabled"))
        results = list(harness.replay_jsonl(p, mode="disabled"))
        # All results should be serializable
        for result in results:
            parsed = json.loads(json.dumps(result.to_dict()))
            assert parsed["mode"] == "disabled"


# ---------------------------------------------------------------------------
# ReplaySummaryBuilder tests
# ---------------------------------------------------------------------------


class TestReplaySummaryBuilder:
    def _make_result(
        self,
        *,
        mode: str = "inline",
        gpt_called: bool = False,
        gpt_decision: str | None = None,
        route_mode: str = "no_gpt",
        validator_passed: bool = False,
        would_apply: bool = False,
        error: str | None = None,
        latency_ms: float | None = 50.0,
    ) -> ReplayResult:
        return ReplayResult(
            replay_id="test",
            source_turn_id=None,
            user_text="test text",
            state_before="waiting_for_modifier",
            mode=mode,
            route_mode=route_mode,
            gpt_called=gpt_called,
            gpt_decision=gpt_decision,
            validator_passed=validator_passed,
            would_apply=would_apply,
            error=error,
            latency_ms=latency_ms,
        )

    def test_total_count(self) -> None:
        builder = ReplaySummaryBuilder(mode="inline")
        for _ in range(5):
            builder.add(self._make_result())
        assert builder.to_dict()["total_turns"] == 5

    def test_gpt_called_count(self) -> None:
        builder = ReplaySummaryBuilder(mode="shadow")
        builder.add(self._make_result(gpt_called=True, route_mode="shadow_gpt"))
        builder.add(self._make_result(gpt_called=True, route_mode="shadow_gpt"))
        builder.add(self._make_result(gpt_called=False))
        assert builder.to_dict()["gpt_called"] == 2

    def test_validator_pass_count(self) -> None:
        builder = ReplaySummaryBuilder(mode="inline")
        builder.add(self._make_result(gpt_called=True, validator_passed=True, route_mode="inline_gpt"))
        builder.add(self._make_result(gpt_called=True, validator_passed=False, route_mode="inline_gpt"))
        builder.add(self._make_result(gpt_called=False))
        d = builder.to_dict()
        assert d["validator_pass_count"] == 1
        assert d["validator_reject_count"] == 1

    def test_would_apply_count(self) -> None:
        builder = ReplaySummaryBuilder(mode="inline")
        builder.add(self._make_result(would_apply=True, gpt_called=True, route_mode="inline_gpt"))
        builder.add(self._make_result(would_apply=False))
        assert builder.to_dict()["would_apply_count"] == 1

    def test_error_count(self) -> None:
        builder = ReplaySummaryBuilder(mode="inline")
        builder.add(self._make_result(error="something broke"))
        builder.add(self._make_result(error=None))
        assert builder.to_dict()["error_count"] == 1

    def test_markdown_generated(self) -> None:
        builder = ReplaySummaryBuilder(mode="shadow")
        builder.add(self._make_result(gpt_called=True, route_mode="shadow_gpt"))
        md = builder.to_markdown()
        assert "Phase 3.5" in md
        assert "shadow" in md.lower()

    def test_markdown_non_empty(self) -> None:
        builder = ReplaySummaryBuilder(mode="disabled")
        md = builder.to_markdown()
        assert len(md) > 50

    def test_route_mode_distribution(self) -> None:
        builder = ReplaySummaryBuilder(mode="inline")
        builder.add(self._make_result(route_mode="no_gpt"))
        builder.add(self._make_result(route_mode="inline_gpt", gpt_called=True))
        builder.add(self._make_result(route_mode="inline_gpt", gpt_called=True))
        d = builder.to_dict()
        assert d["route_modes"]["no_gpt"] == 1
        assert d["route_modes"]["inline_gpt"] == 2


# ---------------------------------------------------------------------------
# Built-in fixture tests
# ---------------------------------------------------------------------------


class TestBuiltInFixtures:
    def test_7_fixtures_defined(self) -> None:
        assert len(BUILT_IN_FIXTURES) == 7

    def test_all_fixtures_are_waiting_for_modifier(self) -> None:
        for f in BUILT_IN_FIXTURES:
            assert f.state_before == "waiting_for_modifier"

    def test_all_fixtures_have_user_text(self) -> None:
        for f in BUILT_IN_FIXTURES:
            assert f.user_text.strip(), f"Fixture {f.source_turn_id} has empty user_text"

    def test_phonetic_fixture_has_cheese_choices(self) -> None:
        fix = BUILT_IN_FIXTURES[0]
        assert "macarola" in fix.user_text.lower()
        assert "Mozzarella Cheese" in fix.choice_names

    def test_correction_signal_fixture_has_flag(self) -> None:
        # fixture:4 is the correction signal case
        fix = next(f for f in BUILT_IN_FIXTURES if f.source_turn_id == "fixture:4")
        assert fix.has_correction_signal is True

    def test_repeat_loop_fixture_has_repeat_count(self) -> None:
        fix = next(f for f in BUILT_IN_FIXTURES if f.source_turn_id == "fixture:7")
        assert fix.repeat_count >= 2

    def test_disabled_mode_all_fixtures_pass(self) -> None:
        harness = Phase3OptionResolverReplayHarness(config=_cfg("disabled"))
        for result in harness.replay_fixtures(BUILT_IN_FIXTURES, mode="disabled"):
            assert result.error is None, (
                f"Fixture {result.source_turn_id} errored: {result.error}"
            )


# ---------------------------------------------------------------------------
# Synthetic group builder tests
# ---------------------------------------------------------------------------


class TestBuildSyntheticGroup:
    def test_builds_group_from_names(self) -> None:
        group = _build_synthetic_group(("American Cheese", "Mozzarella Cheese"))
        assert len(group.choices) == 2
        names = {c.name for c in group.choices}
        assert "American Cheese" in names
        assert "Mozzarella Cheese" in names

    def test_empty_names_gives_empty_group(self) -> None:
        group = _build_synthetic_group(())
        assert len(group.choices) == 0

    def test_modifier_ids_are_synthetic(self) -> None:
        group = _build_synthetic_group(("A", "B"))
        for choice in group.choices:
            assert choice.modifier_id.startswith("synthetic_")

    def test_normalized_names_are_lowercase(self) -> None:
        group = _build_synthetic_group(("American Cheese",))
        assert group.choices[0].normalized_name == "american cheese"


# ---------------------------------------------------------------------------
# CLI integration (dry-run path, no file I/O)
# ---------------------------------------------------------------------------


class TestCliDryRun:
    def test_fixtures_only_dry_run_returns_zero(self) -> None:
        from tools.replay_phase3_option_resolver import main

        result = main(["--fixtures-only", "--mode", "disabled", "--dry-run"])
        assert result == 0

    def test_missing_input_file_returns_nonzero(self) -> None:
        from tools.replay_phase3_option_resolver import main

        result = main(["--input", "/nonexistent/path.jsonl", "--mode", "disabled"])
        assert result != 0

    def test_dry_run_shadow_fixtures(self) -> None:
        from tools.replay_phase3_option_resolver import main

        result = main(["--fixtures-only", "--mode", "shadow", "--dry-run"])
        assert result == 0

    def test_dry_run_inline_fixtures(self) -> None:
        from tools.replay_phase3_option_resolver import main

        result = main(["--fixtures-only", "--mode", "inline", "--dry-run"])
        assert result == 0

    def test_file_written_when_not_dry_run(self, tmp_path: Path) -> None:
        from tools.replay_phase3_option_resolver import main

        out = tmp_path / "test_report.jsonl"
        result = main([
            "--fixtures-only", "--mode", "disabled",
            "--output", str(out),
        ])
        assert result == 0
        assert out.exists()
        # Each line must be valid JSON
        lines = [l for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == len(BUILT_IN_FIXTURES)
        for line in lines:
            obj = json.loads(line)
            assert "replay_id" in obj
            assert obj["actual_applied"] is False

    def test_summary_markdown_written_alongside_jsonl(self, tmp_path: Path) -> None:
        from tools.replay_phase3_option_resolver import main

        out = tmp_path / "report.jsonl"
        result = main([
            "--fixtures-only", "--mode", "disabled",
            "--output", str(out),
        ])
        assert result == 0
        summary_path = out.with_suffix(".summary.md")
        assert summary_path.exists()
        content = summary_path.read_text(encoding="utf-8")
        assert "Phase 3.5" in content
