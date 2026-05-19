# tests/nlu/turn_resolver/test_idle_item_resolver.py
"""Tests for Priority 4: Idle Natural Item Resolver (Bucket 0).

Coverage:
- Policy: should_call_idle_item_resolver() trigger conditions (Tests 1–12)
- Validator: validate_idle_item_resolution() safety gates (Tests 13–20)
- Resolver: parsing, async resolve, config defaults (Tests 21–34)
- Config: new bucket-0 fields (Tests 35–38)
- Integration: to_gpt_turn_resolution() adapter (Tests 39–42)
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

# ── Policy ────────────────────────────────────────────────────────────────────

from app.nlu.turn_resolver.idle_item_policy import (
    should_call_idle_item_resolver,
    _is_meaningful,
    _has_item_slots,
)


class TestIdleItemPolicy:
    """Tests 1–12: should_call_idle_item_resolver() trigger policy."""

    def _call(self, **kwargs) -> tuple[bool, str]:
        defaults = dict(
            state="idle",
            user_text="hamburger",
            normalized_text="hamburger",
            local_intent="UNKNOWN",
            local_confidence=0.0,
            local_slots=[],
            menu_candidates=None,
            previous_assistant_prompt=None,
        )
        defaults.update(kwargs)
        return should_call_idle_item_resolver(**defaults)

    # ── Non-trigger conditions ────────────────────────────────────────────────

    def test_non_idle_state_returns_false(self):
        called, reason = self._call(state="waiting_for_modifier")
        assert called is False
        assert reason == "not_idle_state"

    def test_empty_text_returns_false(self):
        called, reason = self._call(user_text="  ", normalized_text="")
        assert called is False
        assert reason == "empty_text"

    def test_noise_only_text_returns_false(self):
        called, reason = self._call(user_text="um uh", normalized_text="um uh")
        assert called is False
        assert reason == "empty_text"

    def test_checkout_phrase_returns_false(self):
        for phrase in ["checkout", "let me checkout", "place my order", "I'm done"]:
            called, reason = self._call(user_text=phrase, normalized_text=phrase)
            assert called is False, f"Expected no-trigger for: {phrase!r}"
            assert reason == "checkout_phrase"

    def test_cancel_phrase_returns_false(self):
        for phrase in ["cancel", "never mind", "start over", "forget it"]:
            called, reason = self._call(user_text=phrase, normalized_text=phrase)
            assert called is False, f"Expected no-trigger for: {phrase!r}"
            assert reason == "cancel_phrase"

    def test_high_confidence_add_item_returns_false(self):
        called, reason = self._call(
            user_text="I want a hamburger",
            local_intent="ADD_ITEM",
            local_confidence=0.92,
        )
        assert called is False
        assert reason == "high_confidence_local"

    # ── Trigger conditions ────────────────────────────────────────────────────

    def test_previous_prompt_asked_order_triggers(self):
        called, reason = self._call(
            user_text="hamburger",
            previous_assistant_prompt="What would you like to order?",
            menu_candidates=[{"name": "Hamburger", "item_id": "h1"}],
        )
        assert called is True
        assert reason == "previous_prompt_asked_order"

    def test_bare_menu_phrase_triggers(self):
        """Bare item name with no 'I want' preamble."""
        called, reason = self._call(
            user_text="tuna melt",
            normalized_text="tuna melt",
            menu_candidates=[{"name": "Tuna Melt", "item_id": "t1"}],
        )
        assert called is True
        assert reason == "bare_menu_phrase"

    def test_hamburger_with_small_coke_triggers(self):
        called, reason = self._call(
            user_text="hamburger with small coke",
            normalized_text="hamburger with small coke",
            menu_candidates=[
                {"name": "Hamburger", "item_id": "h1"},
                {"name": "Coke", "item_id": "c1"},
            ],
        )
        assert called is True
        assert reason in {"bare_menu_phrase", "with_related_item_phrase"}

    def test_large_fries_triggers(self):
        called, reason = self._call(
            user_text="large fries",
            normalized_text="large fries",
            menu_candidates=[{"name": "Fries", "item_id": "f1", "available_sizes": ["Small", "Medium", "Large"]}],
        )
        assert called is True
        assert reason in {"bare_menu_phrase", "size_or_variant_menu_phrase"}

    def test_six_piece_wings_triggers(self):
        called, reason = self._call(
            user_text="six piece wings",
            normalized_text="six piece wings",
            menu_candidates=[{"name": "Wings", "item_id": "w1", "available_variants": ["6 pc", "12 pc"]}],
        )
        assert called is True
        assert reason in {"bare_menu_phrase", "size_or_variant_menu_phrase"}

    def test_low_confidence_add_item_triggers(self):
        called, reason = self._call(
            user_text="I want a hamburger",
            local_intent="ADD_ITEM",
            local_confidence=0.55,
        )
        assert called is True
        assert reason == "low_confidence_add_item"


# ── Validator ─────────────────────────────────────────────────────────────────

from app.nlu.turn_resolver.idle_item_validator import validate_idle_item_resolution
from app.nlu.turn_resolver.waiting_option_validator import ValidationResult
from app.nlu.turn_resolver.idle_item_resolver import (
    IdleItemDecision,
    IdleItemResolution,
    IdleResolvedItem,
    IdleResolvedSide,
    IdleResolvedModifier,
    IDLE_ITEM_NOT_CALLED,
)


def _make_resolution(
    decision: str = "execute",
    intent: str = "add_item",
    confidence: float = 0.90,
    item_plan: tuple = (),
    unresolved_spans: tuple = (),
    reason: str = "test",
) -> IdleItemResolution:
    ok = decision == "execute" and bool(item_plan)
    return IdleItemResolution(
        ok=ok,
        intent=intent,
        confidence=confidence,
        item_plan=item_plan,
        unresolved_spans=unresolved_spans,
        decision=decision,
        reason=reason,
    )


def _make_candidates(names: list[str]) -> tuple[dict, ...]:
    return tuple(
        {"item_id": f"item_{i}", "name": name}
        for i, name in enumerate(names)
    )


class TestIdleItemValidator:
    """Tests 13–20: validate_idle_item_resolution()."""

    def test_fallback_decision_is_always_valid(self):
        res = _make_resolution(decision="fallback", confidence=0.0)
        result = validate_idle_item_resolution(res, (), None)
        assert result.is_valid

    def test_clarify_decision_is_always_valid(self):
        res = _make_resolution(decision="clarify", confidence=0.0)
        result = validate_idle_item_resolution(res, (), None)
        assert result.is_valid

    def test_reject_decision_is_always_valid(self):
        res = _make_resolution(decision="reject", confidence=0.0)
        result = validate_idle_item_resolution(res, (), None)
        assert result.is_valid

    def test_hamburger_plus_coke_small_passes(self):
        """Hamburger + Coke Small — valid when both in candidates."""
        candidates = _make_candidates(["Hamburger", "Coke"])
        item = IdleResolvedItem(
            item_id="h1",
            item_name="Hamburger",
            quantity=1,
            sides=(IdleResolvedSide(item_id="c1", name="Coke", size="Small"),),
        )
        res = _make_resolution(item_plan=(item,))
        result = validate_idle_item_resolution(res, candidates, None)
        assert result.is_valid

    def test_fries_large_passes(self):
        """Fries Large, quantity 1 — valid."""
        candidates = _make_candidates(["Fries"])
        item = IdleResolvedItem(item_id="f1", item_name="Fries", quantity=1, size_name="Large")
        res = _make_resolution(item_plan=(item,))
        result = validate_idle_item_resolution(res, candidates, None)
        assert result.is_valid

    def test_wings_6pc_quantity_1_passes(self):
        """Wings, variant='6 pc', quantity=1 — valid."""
        candidates = _make_candidates(["Wings"])
        item = IdleResolvedItem(item_id="w1", item_name="Wings", quantity=1, variant_name="6 pc")
        res = _make_resolution(item_plan=(item,))
        result = validate_idle_item_resolution(res, candidates, None)
        assert result.is_valid

    def test_two_6pc_wings_quantity_2_variant_passes(self):
        """Wings, variant='6 pc', quantity=2 — valid (user ordered two)."""
        candidates = _make_candidates(["Wings"])
        item = IdleResolvedItem(item_id="w1", item_name="Wings", quantity=2, variant_name="6 pc")
        res = _make_resolution(item_plan=(item,))
        result = validate_idle_item_resolution(res, candidates, None)
        assert result.is_valid

    def test_6pc_wings_as_quantity_6_rejected(self):
        """Wings, variant='6 pc', quantity=6 — REJECTED (piece count as quantity)."""
        candidates = _make_candidates(["Wings"])
        # GPT incorrectly sets quantity=6 for "6 piece wings" — must be caught
        item = IdleResolvedItem(item_id="w1", item_name="Wings", quantity=6, variant_name="6 pc")
        res = _make_resolution(item_plan=(item,))
        result = validate_idle_item_resolution(res, candidates, None)
        assert not result.is_valid
        assert result.reason == "quantity_is_variant_not_count"

    def test_item_not_in_candidates_is_rejected(self):
        """GPT invents an item not in candidates."""
        candidates = _make_candidates(["Hamburger"])
        item = IdleResolvedItem(item_id=None, item_name="Unicorn Burger", quantity=1)
        res = _make_resolution(item_plan=(item,))
        result = validate_idle_item_resolution(res, candidates, None)
        assert not result.is_valid
        assert result.reason == "item_not_in_candidates"

    def test_low_confidence_is_rejected(self):
        candidates = _make_candidates(["Fries"])
        item = IdleResolvedItem(item_id="f1", item_name="Fries", quantity=1)
        res = _make_resolution(item_plan=(item,), confidence=0.50)
        result = validate_idle_item_resolution(res, candidates, None, min_confidence=0.70)
        assert not result.is_valid
        assert result.reason == "low_confidence"

    def test_empty_item_plan_with_execute_is_rejected(self):
        candidates = _make_candidates(["Fries"])
        res = IdleItemResolution(
            ok=True,
            intent="add_item",
            confidence=0.90,
            item_plan=(),
            unresolved_spans=(),
            decision="execute",
            reason="test",
        )
        result = validate_idle_item_resolution(res, candidates, None)
        assert not result.is_valid
        assert result.reason == "empty_item_plan"

    def test_validator_never_raises_on_broken_input(self):
        """Broken input must not raise."""
        result = validate_idle_item_resolution(
            IDLE_ITEM_NOT_CALLED,
            (None, "bad", 42),  # type: ignore[arg-type]
            None,
        )
        assert isinstance(result, ValidationResult)


# ── Resolver: parsing helpers ─────────────────────────────────────────────────

from app.nlu.turn_resolver.idle_item_resolver import (
    IdleItemResolver,
    _parse_json_response,
    _format_menu_candidates,
    _format_previous_turns,
    to_gpt_turn_resolution,
)
from app.nlu.turn_resolver.gpt_safe_client import GptCallStatus, GptSafeResult


def _make_resolver(gpt_client=None, config=None) -> IdleItemResolver:
    return IdleItemResolver(gpt_client=gpt_client, config=config)


class TestIdleItemResolverParsing:
    """Tests 21–30: parsing helpers and _parse_resolution_from_dict."""

    def _parse(self, data: dict) -> IdleItemResolution:
        resolver = _make_resolver()
        return resolver._parse_resolution_from_dict(data)

    def test_execute_single_item_parsed(self):
        data = {
            "decision": "execute",
            "intent": "add_item",
            "confidence": 0.92,
            "items": [{"item_name": "Hamburger", "quantity": 1, "raw_span": "hamburger"}],
            "unresolved_spans": [],
            "reason": "matched",
        }
        res = self._parse(data)
        assert res.decision == "execute"
        assert res.intent == "add_item"
        assert len(res.item_plan) == 1
        assert res.item_plan[0].item_name == "Hamburger"
        assert res.item_plan[0].quantity == 1
        assert res.ok is True

    def test_hamburger_with_coke_small_parsed(self):
        data = {
            "decision": "execute",
            "intent": "add_item",
            "confidence": 0.88,
            "items": [{
                "item_name": "Hamburger",
                "quantity": 1,
                "sides": [{"name": "Coke", "size": "Small", "quantity": 1}],
            }],
            "unresolved_spans": [],
        }
        res = self._parse(data)
        assert len(res.item_plan) == 1
        assert len(res.item_plan[0].sides) == 1
        assert res.item_plan[0].sides[0].name == "Coke"
        assert res.item_plan[0].sides[0].size == "Small"

    def test_wings_6pc_parsed_with_variant_and_qty_1(self):
        data = {
            "decision": "execute",
            "intent": "add_item",
            "confidence": 0.95,
            "items": [{"item_name": "Wings", "quantity": 1, "variant": "6 pc"}],
        }
        res = self._parse(data)
        assert res.item_plan[0].item_name == "Wings"
        assert res.item_plan[0].quantity == 1
        assert res.item_plan[0].variant_name == "6 pc"

    def test_two_6pc_wings_parsed_qty_2(self):
        data = {
            "decision": "execute",
            "intent": "add_item",
            "confidence": 0.93,
            "items": [{"item_name": "Wings", "quantity": 2, "variant": "6 pc"}],
        }
        res = self._parse(data)
        assert res.item_plan[0].quantity == 2
        assert res.item_plan[0].variant_name == "6 pc"

    def test_fries_large_parsed(self):
        data = {
            "decision": "execute",
            "intent": "add_item",
            "confidence": 0.91,
            "items": [{"item_name": "Fries", "quantity": 1, "size": "Large"}],
        }
        res = self._parse(data)
        assert res.item_plan[0].item_name == "Fries"
        assert res.item_plan[0].size_name == "Large"

    def test_clarify_decision_parsed(self):
        data = {"decision": "clarify", "intent": "unknown", "confidence": 0.5, "items": []}
        res = self._parse(data)
        assert res.decision == "clarify"
        assert res.ok is False

    def test_unknown_decision_maps_to_fallback(self):
        data = {"decision": "teleport", "confidence": 0.99}
        res = self._parse(data)
        assert res.decision == "fallback"

    def test_confidence_clamped_0_to_1(self):
        data = {"decision": "execute", "items": [{"item_name": "X"}], "confidence": 2.5}
        res = self._parse(data)
        assert res.confidence == 1.0

    def test_parse_json_response_plain(self):
        raw = '{"decision": "execute", "confidence": 0.9}'
        result = _parse_json_response(raw)
        assert result["decision"] == "execute"

    def test_parse_json_response_markdown_fenced(self):
        raw = "```json\n{\"decision\": \"clarify\", \"confidence\": 0.7}\n```"
        result = _parse_json_response(raw)
        assert result["decision"] == "clarify"

    def test_parse_json_response_empty_raises(self):
        with pytest.raises(Exception):
            _parse_json_response("")

    def test_format_menu_candidates_empty(self):
        result = _format_menu_candidates(())
        assert "no candidates" in result.lower()

    def test_format_menu_candidates_with_entries(self):
        candidates = (
            {"item_id": "h1", "name": "Hamburger", "available_sizes": ["Small", "Large"]},
            {"item_id": "w1", "name": "Wings", "available_variants": ["6 pc", "12 pc"]},
        )
        result = _format_menu_candidates(candidates)
        assert "Hamburger" in result
        assert "Wings" in result
        assert "6 pc" in result

    def test_format_previous_turns_empty(self):
        result = _format_previous_turns([])
        assert result == ""

    def test_format_previous_turns_dict_entries(self):
        turns = [
            {"role": "assistant", "text": "What would you like?"},
            {"role": "user", "text": "hamburger"},
        ]
        result = _format_previous_turns(turns)
        assert "Bot" in result
        assert "Customer" in result

    def test_format_previous_turns_tuple_entries(self):
        turns = [("bot", "What would you like?"), ("user", "large fries")]
        result = _format_previous_turns(turns)
        assert "Bot" in result
        assert "large fries" in result


# ── Resolver: async resolve() ─────────────────────────────────────────────────

class TestIdleItemResolverAsync:
    """Tests 31–37: async resolve() with mocked GptSafeClient."""

    def _make_ok_client(
        self,
        item_name: str = "Hamburger",
        decision: str = "execute",
        confidence: float = 0.92,
    ) -> MagicMock:
        parsed_data = {
            "decision": decision,
            "intent": "add_item",
            "confidence": confidence,
            "items": [{"item_name": item_name, "quantity": 1, "raw_span": item_name.lower()}],
            "unresolved_spans": [],
            "reason": "fuzzy match",
        }
        ok_result = GptSafeResult(
            ok=True,
            status=GptCallStatus.OK,
            task_mode="idle_add_item_or_menu_query",
            parsed=parsed_data,
            latency_ms=120.0,
            model="gpt-4o-mini",
        )
        client = MagicMock()
        client.call = AsyncMock(return_value=ok_result)
        return client

    def _make_fail_client(self, status: str = GptCallStatus.TIMEOUT) -> MagicMock:
        fail_result = GptSafeResult(
            ok=False,
            status=status,
            task_mode="idle_add_item_or_menu_query",
            latency_ms=700.0,
        )
        client = MagicMock()
        client.call = AsyncMock(return_value=fail_result)
        return client

    def _make_config(self, mode: str = "inline") -> MagicMock:
        cfg = MagicMock()
        cfg.model = "gpt-4o-mini"
        cfg.bucket_0_mode = mode
        cfg.bucket_0_timeout_ms = 700
        cfg.bucket_0_min_confidence = 0.70
        cfg.idle_item_menu_candidate_limit = 12
        return cfg

    def test_disabled_mode_returns_not_called(self):
        cfg = self._make_config("disabled")
        resolver = IdleItemResolver(config=cfg)
        result = asyncio.run(
            resolver.resolve(
                user_text="hamburger",
                state="idle",
                local_intent="UNKNOWN",
                local_confidence=0.0,
                local_slots=[],
            )
        )
        assert result.ok is False
        assert result.decision == IdleItemDecision.FALLBACK
        assert result.reason == "not_called"

    def test_gpt_failure_returns_fallback(self):
        cfg = self._make_config("inline")
        client = self._make_fail_client(GptCallStatus.TIMEOUT)
        resolver = IdleItemResolver(gpt_client=client, config=cfg)
        result = asyncio.run(
            resolver.resolve(
                user_text="hamburger",
                state="idle",
                local_intent="UNKNOWN",
                local_confidence=0.0,
                local_slots=[],
                menu_candidates=(),
            )
        )
        assert result.ok is False
        assert result.decision == IdleItemDecision.FALLBACK
        assert "gpt_failed" in result.reason

    def test_shadow_mode_logs_but_returns_fallback(self):
        cfg = self._make_config("shadow")
        client = self._make_ok_client("Hamburger", "execute", 0.92)
        resolver = IdleItemResolver(gpt_client=client, config=cfg)
        result = asyncio.run(
            resolver.resolve(
                user_text="hamburger",
                state="idle",
                local_intent="UNKNOWN",
                local_confidence=0.0,
                local_slots=[],
                menu_candidates=({"item_id": "h1", "name": "Hamburger"},),
            )
        )
        # Shadow mode → ok=False even when GPT succeeds
        assert result.ok is False
        assert result.decision == IdleItemDecision.FALLBACK
        assert result.reason == "shadow_mode_not_applied"
        assert result.metadata.get("shadow_decision") == "execute"

    def test_inline_mode_returns_ok_when_valid(self):
        cfg = self._make_config("inline")
        client = self._make_ok_client("Hamburger", "execute", 0.92)
        resolver = IdleItemResolver(gpt_client=client, config=cfg)
        result = asyncio.run(
            resolver.resolve(
                user_text="hamburger",
                state="idle",
                local_intent="UNKNOWN",
                local_confidence=0.0,
                local_slots=[],
                menu_candidates=({"item_id": "h1", "name": "Hamburger"},),
            )
        )
        assert result.ok is True
        assert result.decision == IdleItemDecision.EXECUTE
        assert any(it.item_name == "Hamburger" for it in result.item_plan)

    def test_unsupported_state_returns_fallback(self):
        cfg = self._make_config("inline")
        resolver = IdleItemResolver(config=cfg)
        result = asyncio.run(
            resolver.resolve(
                user_text="hamburger",
                state="waiting_for_modifier",  # not idle
                local_intent=None,
                local_confidence=None,
                local_slots=None,
            )
        )
        assert result.ok is False
        assert result.decision == IdleItemDecision.FALLBACK

    def test_resolve_sync_never_raises(self):
        """resolve_sync() must return IdleItemResolution even on internal crash."""
        broken_client = MagicMock()
        broken_client.call = AsyncMock(side_effect=RuntimeError("boom"))
        resolver = IdleItemResolver(
            gpt_client=broken_client,
            config=self._make_config("inline"),
        )
        result = resolver.resolve_sync(
            user_text="hamburger",
            state="idle",
            local_intent="UNKNOWN",
            local_confidence=0.0,
            local_slots=[],
            menu_candidates=(),
        )
        assert isinstance(result, IdleItemResolution)
        assert result.ok is False

    def test_invalid_item_id_rejected_by_validator(self):
        """GPT returns item not in candidates → validation fails → ok=False."""
        cfg = self._make_config("inline")
        # GPT claims 'Unicorn Burger' which is NOT in candidates
        parsed_data = {
            "decision": "execute",
            "intent": "add_item",
            "confidence": 0.95,
            "items": [{"item_name": "Unicorn Burger", "quantity": 1}],
            "unresolved_spans": [],
        }
        ok_result = GptSafeResult(
            ok=True,
            status=GptCallStatus.OK,
            task_mode="idle_add_item_or_menu_query",
            parsed=parsed_data,
            latency_ms=100.0,
            model="gpt-4o-mini",
        )
        client = MagicMock()
        client.call = AsyncMock(return_value=ok_result)
        resolver = IdleItemResolver(gpt_client=client, config=cfg)
        result = asyncio.run(
            resolver.resolve(
                user_text="unicorn burger",
                state="idle",
                local_intent="UNKNOWN",
                local_confidence=0.0,
                local_slots=[],
                menu_candidates=({"item_id": "h1", "name": "Hamburger"},),  # no unicorn
            )
        )
        assert result.ok is False
        assert "validation_failed" in result.reason


# ── Config additions ──────────────────────────────────────────────────────────

class TestConfigAdditions:
    """Tests 35–38: new bucket-0 config fields."""

    def test_bucket_0_timeout_ms_default(self):
        from app.config.semantic_repair import SemanticRepairConfig
        cfg = SemanticRepairConfig(phase=0, model="gpt-4o-mini", timeout_seconds=1.0)
        assert cfg.bucket_0_timeout_ms == 700

    def test_bucket_0_min_confidence_default(self):
        from app.config.semantic_repair import SemanticRepairConfig
        cfg = SemanticRepairConfig(phase=0, model="gpt-4o-mini", timeout_seconds=1.0)
        assert cfg.bucket_0_min_confidence == pytest.approx(0.70)

    def test_idle_item_menu_candidate_limit_default(self):
        from app.config.semantic_repair import SemanticRepairConfig
        cfg = SemanticRepairConfig(phase=0, model="gpt-4o-mini", timeout_seconds=1.0)
        assert cfg.idle_item_menu_candidate_limit == 12

    def test_idle_item_high_conf_threshold_default(self):
        from app.config.semantic_repair import SemanticRepairConfig
        cfg = SemanticRepairConfig(phase=0, model="gpt-4o-mini", timeout_seconds=1.0)
        assert cfg.idle_item_high_conf_threshold == pytest.approx(0.85)

    def test_bucket_0_timeout_ms_overridable(self):
        from app.config.semantic_repair import SemanticRepairConfig
        cfg = SemanticRepairConfig(
            phase=0, model="gpt-4o-mini", timeout_seconds=1.0,
            bucket_0_timeout_ms=500,
        )
        assert cfg.bucket_0_timeout_ms == 500

    def test_bucket_0_min_confidence_overridable(self):
        from app.config.semantic_repair import SemanticRepairConfig
        cfg = SemanticRepairConfig(
            phase=0, model="gpt-4o-mini", timeout_seconds=1.0,
            bucket_0_min_confidence=0.80,
        )
        assert cfg.bucket_0_min_confidence == pytest.approx(0.80)


# ── to_gpt_turn_resolution() adapter ─────────────────────────────────────────

class TestToGptTurnResolution:
    """Tests 39–42: to_gpt_turn_resolution() adapter."""

    def test_fallback_resolution_maps_to_skipped(self):
        from app.nlu.turn_resolver.schemas import GptTurnResolution
        res = IDLE_ITEM_NOT_CALLED
        gpt = to_gpt_turn_resolution(res)
        assert isinstance(gpt, GptTurnResolution)
        assert gpt.decision == "skipped"
        assert gpt.gpt_called is False

    def test_execute_resolution_maps_to_add_items(self):
        from app.nlu.turn_resolver.schemas import GptTurnResolution
        item = IdleResolvedItem(item_id="h1", item_name="Hamburger", quantity=1)
        res = IdleItemResolution(
            ok=True,
            intent="add_item",
            confidence=0.92,
            item_plan=(item,),
            unresolved_spans=(),
            decision=IdleItemDecision.EXECUTE,
            reason="matched",
        )
        gpt = to_gpt_turn_resolution(res)
        assert isinstance(gpt, GptTurnResolution)
        assert gpt.decision == "add_items"
        assert gpt.gpt_called is True
        assert len(gpt.items) == 1
        assert gpt.items[0].item_name == "Hamburger"

    def test_clarify_resolution_maps_to_clarify(self):
        from app.nlu.turn_resolver.schemas import GptTurnResolution
        res = IdleItemResolution(
            ok=False,
            intent="unknown",
            confidence=0.5,
            item_plan=(),
            unresolved_spans=(),
            decision=IdleItemDecision.CLARIFY,
            reason="ambiguous",
        )
        gpt = to_gpt_turn_resolution(res)
        assert gpt.decision == "clarify"

    def test_execute_with_sides_maps_correctly(self):
        from app.nlu.turn_resolver.schemas import GptTurnResolution
        side = IdleResolvedSide(item_id="c1", name="Coke", size="Small")
        item = IdleResolvedItem(
            item_id="h1", item_name="Hamburger", quantity=1, sides=(side,)
        )
        res = IdleItemResolution(
            ok=True,
            intent="add_item",
            confidence=0.91,
            item_plan=(item,),
            unresolved_spans=(),
            decision=IdleItemDecision.EXECUTE,
            reason="matched",
        )
        gpt = to_gpt_turn_resolution(res)
        assert len(gpt.items) == 1
        assert len(gpt.items[0].sides) == 1
        assert gpt.items[0].sides[0].name == "Coke"
        assert gpt.items[0].sides[0].size == "Small"


# ── Sentinel / constants ──────────────────────────────────────────────────────

class TestSentinelAndConstants:
    """Tests 43–45: sentinel and action constants."""

    def test_not_called_sentinel_is_fallback(self):
        assert IDLE_ITEM_NOT_CALLED.ok is False
        assert IDLE_ITEM_NOT_CALLED.decision == IdleItemDecision.FALLBACK
        assert IDLE_ITEM_NOT_CALLED.reason == "not_called"
        assert IDLE_ITEM_NOT_CALLED.raw_gpt_status == GptCallStatus.DISABLED

    def test_decision_constants_are_strings(self):
        assert IdleItemDecision.EXECUTE == "execute"
        assert IdleItemDecision.CLARIFY == "clarify"
        assert IdleItemDecision.REJECT == "reject"
        assert IdleItemDecision.FALLBACK == "fallback"

    def test_idle_item_resolution_is_frozen(self):
        res = IdleItemResolution(
            ok=True,
            intent="add_item",
            confidence=0.9,
            item_plan=(),
            unresolved_spans=(),
            decision="execute",
            reason="test",
        )
        with pytest.raises((AttributeError, TypeError)):
            res.ok = False  # type: ignore[misc]
