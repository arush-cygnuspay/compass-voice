# tests/tools/test_replay_phase4_add_item_planner.py
"""Tests for Phase 4 Add-Item Planner replay harness.

Safety invariants verified:
  - actual_applied is always False
  - GPT is never called by default (use_live_gpt=False)
  - Shadow mode never would_apply
  - Inline mode would_apply only when apply gate approves
"""
from __future__ import annotations

import json
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Stub heavy ML / infrastructure dependencies
# ---------------------------------------------------------------------------
_intent_inference = types.ModuleType("app.ml.intent.inference_intent")
_intent_inference.IntentBundle = type("IntentBundle", (), {})
_intent_inference.predict_intent = lambda *a, **kw: []
sys.modules.setdefault("app.ml.intent.inference_intent", _intent_inference)

_slot_inference = types.ModuleType("app.ml.slot.inference_slot")
_slot_inference.SlotBundle = type("SlotBundle", (), {})
_slot_inference.predict_slots = lambda *a, **kw: []
sys.modules.setdefault("app.ml.slot.inference_slot", _slot_inference)

for _mod_name in ("twilio", "twilio.base", "twilio.base.exceptions", "twilio.rest", "redis"):
    sys.modules.setdefault(_mod_name, types.ModuleType(_mod_name))
if not hasattr(sys.modules.get("twilio.base.exceptions"), "TwilioRestException"):
    sys.modules["twilio.base.exceptions"].TwilioRestException = type(
        "TwilioRestException", (Exception,), {}
    )
if not hasattr(sys.modules.get("twilio.rest"), "Client"):
    sys.modules["twilio.rest"].Client = type(
        "Client", (), {"__init__": lambda s, *a, **kw: None}
    )
if not hasattr(sys.modules.get("redis"), "Redis"):
    sys.modules["redis"].Redis = type(
        "Redis", (), {"__init__": lambda s, *a, **kw: None}
    )

# ---------------------------------------------------------------------------
# Subject under test imports
# ---------------------------------------------------------------------------
from app.nlu.semantic_repair.add_item_planner_replay import (
    BUILT_IN_FIXTURES,
    Phase4AddItemPlannerReplayHarness,
    PlannerReplayInputTurn,
    PlannerReplayResult,
    PlannerReplaySummaryBuilder,
    parse_jsonl_row,
)

_FAKE_API_KEY = {"OPENAI_API_KEY": "sk-test-phase4-replay"}
_PROJECT_ROOT = Path(__file__).parents[2]

# ---------------------------------------------------------------------------
# Mock GPT response helpers
# ---------------------------------------------------------------------------


def _mock_client(response_json: str = None) -> MagicMock:
    """Return a mock OpenAI client that returns a fixed JSON response."""
    default = json.dumps({
        "decision": "add_items",
        "items": [{"item_name": "chicken burger", "quantity": 1, "modifiers": [], "sides": []}],
        "unresolved": [],
        "confidence": 0.85,
        "reason_code": "complex_with_phrase",
        "safe_to_apply": False,
    })
    content = response_json or default
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    client = MagicMock()
    client.chat.completions.create.return_value = resp
    return client


# ===========================================================================
# Part 1: parse_jsonl_row
# ===========================================================================


class TestParseJsonlRow:
    """parse_jsonl_row() extracts PlannerReplayInputTurn from JSONL records."""

    def _row(self, **overrides) -> dict:
        base = {
            "normalized_text": "chicken burger with cheese",
            "state_before": "IDLE",
            "response_key": "ask_for_item",
            "session_id": "sess_abc",
            "turn_index": 3,
            "local": {
                "intent": "add_item",
                "confidence": 0.72,
                "slots": [{"n": "ITEM", "v": "chicken burger"}],
                "top_intents": [{"i": "add_item", "c": 0.72}],
            },
            "allowed": {"choices": ["American Cheese", "Swiss"]},
        }
        base.update(overrides)
        return base

    def test_basic_row_parsed(self) -> None:
        turn = parse_jsonl_row(self._row())
        assert turn is not None
        assert turn.user_text == "chicken burger with cheese"
        # state_before is stored normalised to lowercase
        assert turn.state_before.lower() == "idle"

    def test_local_intent_extracted(self) -> None:
        turn = parse_jsonl_row(self._row())
        assert turn.local_intent == "add_item"
        assert turn.local_confidence == pytest.approx(0.72, abs=0.01)

    def test_local_slots_extracted(self) -> None:
        turn = parse_jsonl_row(self._row())
        assert len(turn.local_slots) == 1
        assert turn.local_slots[0]["n"] == "ITEM"

    def test_session_info_extracted(self) -> None:
        turn = parse_jsonl_row(self._row())
        assert turn.session_id == "sess_abc"
        assert turn.turn_index == 3

    def test_filter_state_match_passes(self) -> None:
        turn = parse_jsonl_row(self._row(state_before="IDLE"), filter_state="idle")
        assert turn is not None

    def test_filter_state_mismatch_returns_none(self) -> None:
        turn = parse_jsonl_row(
            self._row(state_before="WAITING_FOR_MODIFIER"), filter_state="idle"
        )
        assert turn is None

    def test_empty_text_returns_none(self) -> None:
        turn = parse_jsonl_row(self._row(normalized_text=""))
        assert turn is None

    def test_missing_local_block_uses_defaults(self) -> None:
        row = {
            "normalized_text": "burger with cheese",
            "state_before": "IDLE",
        }
        turn = parse_jsonl_row(row)
        assert turn is not None
        assert turn.local_intent is None

    def test_malformed_row_returns_none(self) -> None:
        turn = parse_jsonl_row("not a dict")  # type: ignore[arg-type]
        assert turn is None

    def test_response_key_before_extracted(self) -> None:
        turn = parse_jsonl_row(self._row())
        assert turn.response_key_before == "ask_for_item"


# ===========================================================================
# Part 2: Disabled Mode
# ===========================================================================


class TestReplayHarnessDisabled:
    """Disabled mode never calls GPT and never would_apply."""

    @pytest.fixture(autouse=True)
    def harness(self) -> None:
        self.harness = Phase4AddItemPlannerReplayHarness(use_live_gpt=False)

    def test_disabled_mode_gpt_not_called(self) -> None:
        turn = PlannerReplayInputTurn(
            user_text="chicken burger with cheese and coke",
            state_before="IDLE",
            source_turn_id="test:1",
        )
        result = self.harness.replay_turn(turn, mode="disabled")
        assert result.gpt_called is False

    def test_disabled_mode_would_apply_false(self) -> None:
        turn = PlannerReplayInputTurn(
            user_text="burger with extra cheese",
            state_before="IDLE",
            source_turn_id="test:2",
        )
        result = self.harness.replay_turn(turn, mode="disabled")
        assert result.would_apply is False

    def test_disabled_mode_actual_applied_always_false(self) -> None:
        turn = PlannerReplayInputTurn(user_text="anything", state_before="IDLE")
        result = self.harness.replay_turn(turn, mode="disabled")
        assert result.actual_applied is False

    def test_disabled_mode_result_serialisable(self) -> None:
        turn = PlannerReplayInputTurn(user_text="anything", state_before="IDLE")
        result = self.harness.replay_turn(turn, mode="disabled")
        d = result.to_dict()
        json.dumps(d)  # must not raise


# ===========================================================================
# Part 3: Shadow Mode
# ===========================================================================


class TestReplayHarnessShadow:
    """Shadow mode can call GPT but never would_apply."""

    @pytest.fixture(autouse=True)
    def harness(self) -> None:
        self.client = _mock_client()
        self.harness = Phase4AddItemPlannerReplayHarness(
            use_live_gpt=False,
            mock_client=self.client,
        )

    def test_shadow_complex_utterance_calls_gpt(self) -> None:
        turn = PlannerReplayInputTurn(
            user_text="chicken burger with mozzarella and coke",
            state_before="IDLE",
            source_turn_id="shadow:1",
            local_intent="add_item",
            local_confidence=0.55,
        )
        with patch.dict("os.environ", _FAKE_API_KEY):
            result = self.harness.replay_turn(turn, mode="shadow")
        assert result.gpt_called is True

    def test_shadow_never_would_apply(self) -> None:
        turn = PlannerReplayInputTurn(
            user_text="chicken burger with mozzarella and coke",
            state_before="IDLE",
            source_turn_id="shadow:2",
            local_intent="add_item",
            local_confidence=0.55,
        )
        with patch.dict("os.environ", _FAKE_API_KEY):
            result = self.harness.replay_turn(turn, mode="shadow")
        assert result.would_apply is False

    def test_shadow_actual_applied_always_false(self) -> None:
        turn = PlannerReplayInputTurn(
            user_text="burger with extra cheese",
            state_before="IDLE",
        )
        with patch.dict("os.environ", _FAKE_API_KEY):
            result = self.harness.replay_turn(turn, mode="shadow")
        assert result.actual_applied is False

    def test_all_fixtures_shadow_never_would_apply(self) -> None:
        with patch.dict("os.environ", _FAKE_API_KEY):
            for result in self.harness.replay_fixtures(BUILT_IN_FIXTURES, mode="shadow"):
                assert result.would_apply is False


# ===========================================================================
# Part 4: Inline Mode
# ===========================================================================


class TestReplayHarnessInline:
    """Inline mode would_apply only when apply gate approves."""

    def test_inline_simple_no_would_apply(self) -> None:
        harness = Phase4AddItemPlannerReplayHarness(use_live_gpt=False)
        turn = PlannerReplayInputTurn(
            user_text="just a coke",
            state_before="IDLE",
            local_intent="add_item",
            local_confidence=0.95,
            local_slots=({"n": "ITEM", "v": "coke"},),
        )
        result = harness.replay_turn(turn, mode="inline")
        # Simple + high confidence → NO_GPT → would_apply=False
        assert result.would_apply is False

    def test_inline_no_menu_store_prevents_would_apply(self) -> None:
        """Without menu store, validator not run, apply gate returns False."""
        client = _mock_client()
        harness = Phase4AddItemPlannerReplayHarness(use_live_gpt=False, mock_client=client)
        turn = PlannerReplayInputTurn(
            user_text="burger with extra cheese and coke",
            state_before="IDLE",
            local_intent="add_item",
            local_confidence=0.55,
        )
        with patch.dict("os.environ", _FAKE_API_KEY):
            result = harness.replay_turn(turn, mode="inline")
        # No menu_store → validator not run → apply gate gate 7 fails → would_apply=False
        assert result.would_apply is False

    def test_inline_would_apply_false_for_no_repair(self) -> None:
        client = _mock_client(response_json=json.dumps({
            "decision": "no_repair", "items": [], "unresolved": [],
            "confidence": 0.1, "reason_code": "unclear",
        }))
        harness = Phase4AddItemPlannerReplayHarness(use_live_gpt=False, mock_client=client)
        turn = PlannerReplayInputTurn(
            user_text="burger with extra cheese",
            state_before="IDLE",
        )
        with patch.dict("os.environ", _FAKE_API_KEY):
            result = harness.replay_turn(turn, mode="inline")
        assert result.would_apply is False

    def test_inline_actual_applied_always_false(self) -> None:
        harness = Phase4AddItemPlannerReplayHarness(use_live_gpt=False)
        for fixture in BUILT_IN_FIXTURES:
            result = harness.replay_turn(fixture, mode="inline")
            assert result.actual_applied is False


# ===========================================================================
# Part 5: Robustness
# ===========================================================================


class TestReplayRobustness:
    """Harness never raises, handles edge cases gracefully."""

    @pytest.fixture(autouse=True)
    def harness(self) -> None:
        self.harness = Phase4AddItemPlannerReplayHarness(use_live_gpt=False)

    def test_empty_text_does_not_raise(self) -> None:
        turn = PlannerReplayInputTurn(user_text="", state_before="IDLE")
        result = self.harness.replay_turn(turn, mode="shadow")
        assert result is not None
        assert result.gpt_called is False

    def test_noise_text_does_not_raise(self) -> None:
        turn = PlannerReplayInputTurn(user_text="   um   ", state_before="IDLE")
        result = self.harness.replay_turn(turn, mode="inline")
        assert result is not None

    def test_all_fixtures_produce_valid_json(self) -> None:
        for fixture in BUILT_IN_FIXTURES:
            result = self.harness.replay_turn(fixture, mode="shadow")
            d = result.to_dict()
            json.dumps(d)  # must not raise

    def test_replay_id_unique_per_turn(self) -> None:
        turn = PlannerReplayInputTurn(user_text="burger with cheese", state_before="IDLE")
        ids = {self.harness.replay_turn(turn, mode="shadow").replay_id for _ in range(5)}
        assert len(ids) == 5  # all unique

    def test_empty_candidate_items_no_crash(self) -> None:
        turn = PlannerReplayInputTurn(
            user_text="blarbqux frizzlestick",
            state_before="IDLE",
            candidate_items=(),
        )
        result = self.harness.replay_turn(turn, mode="shadow")
        assert result is not None


# ===========================================================================
# Part 6: JSONL Replay
# ===========================================================================


class TestJsonlReplay:
    """replay_jsonl() reads files and applies filters correctly."""

    @pytest.fixture(autouse=True)
    def harness(self) -> None:
        self.harness = Phase4AddItemPlannerReplayHarness(use_live_gpt=False)

    def _write_jsonl(self, rows: list[dict], tmp_dir: Path) -> Path:
        path = tmp_dir / "turns.jsonl"
        with path.open("w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        return path

    def _row(self, text: str, state: str = "IDLE") -> dict:
        return {
            "normalized_text": text,
            "state_before": state,
            "session_id": "s1",
            "turn_index": 0,
            "local": {"intent": "add_item", "confidence": 0.55},
        }

    def test_replay_jsonl_returns_results(self, tmp_path) -> None:
        path = self._write_jsonl([
            self._row("chicken burger with cheese"),
            self._row("pizza and coke"),
        ], tmp_path)
        results = list(self.harness.replay_jsonl(str(path), mode="shadow"))
        assert len(results) == 2

    def test_filter_state_skips_non_matching(self, tmp_path) -> None:
        path = self._write_jsonl([
            self._row("burger with cheese", state="IDLE"),
            self._row("mozzarella", state="WAITING_FOR_MODIFIER"),
        ], tmp_path)
        results = list(self.harness.replay_jsonl(
            str(path), mode="shadow", filter_state="idle"
        ))
        assert len(results) == 1

    def test_max_turns_limit_respected(self, tmp_path) -> None:
        path = self._write_jsonl([self._row(f"item {i} with cheese") for i in range(10)], tmp_path)
        results = list(self.harness.replay_jsonl(str(path), mode="shadow", max_turns=3))
        assert len(results) == 3

    def test_empty_jsonl_returns_empty(self, tmp_path) -> None:
        path = self._write_jsonl([], tmp_path)
        results = list(self.harness.replay_jsonl(str(path), mode="shadow"))
        assert results == []

    def test_malformed_line_skipped_gracefully(self, tmp_path) -> None:
        # Malformed lines may produce an error result or be silently skipped.
        # The important invariant is the harness does not raise.
        path = tmp_path / "turns.jsonl"
        with path.open("w") as f:
            f.write("not valid json\n")
            f.write(json.dumps(self._row("burger with cheese")) + "\n")
        results = list(self.harness.replay_jsonl(str(path), mode="shadow"))
        # At least the valid row is processed; total >= 1
        assert len(results) >= 1


# ===========================================================================
# Part 7: Summary Builder
# ===========================================================================


class TestPlannerReplaySummaryBuilder:
    """PlannerReplaySummaryBuilder aggregates results correctly."""

    def _result(self, **kw) -> PlannerReplayResult:
        defaults = dict(
            replay_id="r1",
            source_turn_id=None,
            user_text="burger with cheese",
            state_before="IDLE",
            mode="shadow",
            response_key_before=None,
            local_intent="add_item",
            local_confidence=0.70,
            local_slots=[],
            route_mode="shadow_gpt",
            route_reason="shadow_complex_or_evidence",
            gpt_called=True,
            decision="add_items",
            items_count=1,
            unresolved_count=0,
            confidence=0.82,
            validator_passed=False,
            validator_reject_reason="shadow_mode_never_applies",
            safe_to_apply=False,
            would_apply=False,
            actual_applied=False,
        )
        defaults.update(kw)
        return PlannerReplayResult(**defaults)

    def test_total_turns_counted(self) -> None:
        sb = PlannerReplaySummaryBuilder(mode="shadow")
        for _ in range(5):
            sb.add(self._result())
        assert sb.to_dict()["total_turns"] == 5

    def test_gpt_called_counted(self) -> None:
        sb = PlannerReplaySummaryBuilder(mode="shadow")
        sb.add(self._result(gpt_called=True))
        sb.add(self._result(gpt_called=False))
        assert sb.to_dict()["gpt_called"] == 1

    def test_would_apply_counted(self) -> None:
        sb = PlannerReplaySummaryBuilder(mode="inline")
        sb.add(self._result(mode="inline", would_apply=True))
        sb.add(self._result(mode="inline", would_apply=False))
        assert sb.to_dict()["would_apply_count"] == 1

    def test_error_counted(self) -> None:
        sb = PlannerReplaySummaryBuilder(mode="shadow")
        sb.add(self._result(error="gpt_timeout"))
        sb.add(self._result(error=None))
        assert sb.to_dict()["error_count"] == 1

    def test_decision_distribution(self) -> None:
        sb = PlannerReplaySummaryBuilder(mode="shadow")
        sb.add(self._result(decision="add_items"))
        sb.add(self._result(decision="add_items"))
        sb.add(self._result(decision="no_repair"))
        dist = sb.to_dict()["decision_distribution"]
        assert dist.get("add_items") == 2
        assert dist.get("no_repair") == 1

    def test_zero_turns_safe(self) -> None:
        sb = PlannerReplaySummaryBuilder(mode="shadow")
        d = sb.to_dict()
        assert d["total_turns"] == 0
        assert d["would_apply_count"] == 0

    def test_to_markdown_returns_string(self) -> None:
        sb = PlannerReplaySummaryBuilder(mode="shadow")
        sb.add(self._result())
        md = sb.to_markdown()
        assert isinstance(md, str)
        assert "shadow" in md.lower()

    def test_to_markdown_includes_summary_counts(self) -> None:
        sb = PlannerReplaySummaryBuilder(mode="shadow")
        for _ in range(3):
            sb.add(self._result(gpt_called=True))
        md = sb.to_markdown()
        assert "3" in md


# ===========================================================================
# Part 8: Built-In Fixtures
# ===========================================================================


class TestBuiltInFixtures:
    """BUILT_IN_FIXTURES are complete and valid."""

    def test_fixtures_not_empty(self) -> None:
        assert len(BUILT_IN_FIXTURES) == 7

    def test_all_fixtures_have_required_fields(self) -> None:
        for f in BUILT_IN_FIXTURES:
            assert f.user_text
            assert f.state_before
            assert f.source_turn_id is not None

    def test_fixture_ids_unique(self) -> None:
        ids = [f.source_turn_id for f in BUILT_IN_FIXTURES]
        assert len(set(ids)) == len(ids)

    def test_complex_fixture_has_candidate_items(self) -> None:
        # Fixtures 1-5 should have candidate items for context
        fixtures_with_candidates = [f for f in BUILT_IN_FIXTURES if f.candidate_items]
        assert len(fixtures_with_candidates) >= 3

    def test_nonsense_fixture_has_no_candidates(self) -> None:
        # Fixture 6 (nonsense text) should have no candidates
        nonsense = next(f for f in BUILT_IN_FIXTURES if f.source_turn_id == "fixture:6")
        assert len(nonsense.candidate_items) == 0

    def test_all_fixtures_replay_without_crash(self) -> None:
        harness = Phase4AddItemPlannerReplayHarness(use_live_gpt=False)
        for fixture in BUILT_IN_FIXTURES:
            result = harness.replay_turn(fixture, mode="shadow")
            assert result is not None

    def test_all_fixtures_actual_applied_false(self) -> None:
        harness = Phase4AddItemPlannerReplayHarness(use_live_gpt=False)
        for fixture in BUILT_IN_FIXTURES:
            result = harness.replay_turn(fixture, mode="disabled")
            assert result.actual_applied is False


# ===========================================================================
# Part 9: CLI Integration (dry-run)
# ===========================================================================


class TestCliDryRun:
    """CLI runs correctly in dry-run mode without writing files."""

    def test_fixtures_only_dry_run_exits_zero(self) -> None:
        from tools.replay_phase4_add_item_planner import main
        rc = main(["--fixtures-only", "--mode", "shadow", "--dry-run"])
        assert rc == 0

    def test_fixtures_only_disabled_dry_run_exits_zero(self) -> None:
        from tools.replay_phase4_add_item_planner import main
        rc = main(["--fixtures-only", "--mode", "disabled", "--dry-run"])
        assert rc == 0

    def test_use_live_gpt_without_key_exits_one(self) -> None:
        from tools.replay_phase4_add_item_planner import main
        with patch.dict("os.environ", {}, clear=True):
            rc = main(["--fixtures-only", "--use-live-gpt", "--dry-run"])
        assert rc == 1

    def test_missing_input_file_exits_one(self) -> None:
        from tools.replay_phase4_add_item_planner import main
        rc = main(["--input", "/nonexistent/path.jsonl", "--mode", "shadow"])
        assert rc == 1

    def test_dry_run_does_not_write_files(self, tmp_path) -> None:
        from tools.replay_phase4_add_item_planner import main
        output = tmp_path / "out.jsonl"
        rc = main([
            "--fixtures-only", "--mode", "shadow", "--dry-run",
            "--output", str(output),
        ])
        assert rc == 0
        assert not output.exists()

    def test_verbose_flag_accepted(self) -> None:
        from tools.replay_phase4_add_item_planner import main
        rc = main(["--fixtures-only", "--mode", "disabled", "--dry-run", "--verbose"])
        assert rc == 0
