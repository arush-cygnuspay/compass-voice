# tests/nlu/turn_resolver/test_turn_resolver.py
"""Integration tests for the unified GPT turn-resolution layer.

25 tests across 5 categories:
  A  — Idle item resolution (Bucket 0 triggering)
  B  — Waiting state option resolution (Bucket 2 triggering)
  C  — Multi-item / staged planning (Bucket 3 triggering)
  D  — Safety: GPT never mutates cart / state / FSM
  E  — Logging / schema: GptTurnResolution and FinalTurnDecision structure

Hard constraints verified:
  * pick_bucket() returns None for terminal states
  * pick_bucket() returns None when bucket mode is "disabled"
  * FinalTurnDecision.source == "local" in shadow mode
  * FinalTurnDecision.apply_gpt == False unless mode == "inline" and validation passes
  * GptTurnResolution.safe_to_apply is never set by GPT output — only by validators
  * No cart / state / FSM mutation anywhere in the resolver path
"""
from __future__ import annotations

import pytest
from collections import deque
from dataclasses import replace
from unittest.mock import MagicMock, patch

from app.config.semantic_repair import SemanticRepairConfig
from app.nlu.intent_resolution.intent import Intent
from app.nlu.nlu_result import IntentCandidate, NLUResult, SlotValue
from app.nlu.turn_resolver.bucket_policy import (
    BUCKET_IDLE_ITEM,
    BUCKET_MULTI_ITEM,
    BUCKET_OPTION,
    LOW_CONFIDENCE_THRESHOLD,
    pick_bucket,
)
from app.nlu.turn_resolver.context_builder import build_context_packet
from app.nlu.turn_resolver.final_turn_decision_resolver import (
    FinalTurnDecision,
    resolve,
)
from app.nlu.turn_resolver.schemas import (
    GPT_TURN_RESOLUTION_SKIPPED,
    GptTurnResolution,
    ResolvedItemPlan,
    ResolvedModifierPlan,
    ResolvedSidePlan,
)
from app.nlu.turn_resolver.validators import (
    ValidationResult,
    validate_bucket0_result,
    validate_bucket2_result,
    validate_bucket3_result,
    validate_gpt_result,
)
from app.state_machine.models.conversation_context import ConversationContext
from app.state_machine.models.conversation_state import ConversationState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_IDLE = ConversationState.IDLE
_WAITING_MOD = ConversationState.WAITING_FOR_MODIFIER
_WAITING_SIDE = ConversationState.WAITING_FOR_SIDE
_WAITING_SIZE = ConversationState.WAITING_FOR_SIZE
_COMPLETED = ConversationState.COMPLETED
_ERROR = ConversationState.ERROR_RECOVERY
_TRANSFER = ConversationState.TRANSFERRING_TO_HUMAN_AGENT


def _nlu(
    intent: Intent = Intent.UNKNOWN,
    confidence: float = 0.3,
    text: str = "a burger please",
    slots: tuple = (),
    candidates: tuple = (),
) -> NLUResult:
    return NLUResult(
        effective_intent=intent,
        intent_confidence=confidence,
        raw_text=text,
        normalized_text=text,
        slots=slots,
        intent_candidates=candidates,
    )


def _slot(name: str, value: str) -> SlotValue:
    return SlotValue(name=name, value=value)


def _cfg(
    b0: str = "disabled",
    b2: str = "disabled",
    b3: str = "disabled",
    **kwargs,
) -> SemanticRepairConfig:
    return SemanticRepairConfig(
        phase=2,
        model="gpt-4o-mini",
        timeout_seconds=0.35,
        bucket_0_mode=b0,
        bucket_2_mode=b2,
        bucket_3_mode=b3,
        **kwargs,
    )


def _ctx() -> ConversationContext:
    ctx = ConversationContext()
    ctx.current_prompt_field = None
    return ctx


def _gpt_resolution(
    bucket: str = BUCKET_IDLE_ITEM,
    decision: str = "add_items",
    items: tuple = (),
    selected_option_names: tuple = (),
    confidence: float | None = 0.9,
    safe_to_apply: bool = False,
    gpt_called: bool = True,
) -> GptTurnResolution:
    return GptTurnResolution(
        bucket=bucket,
        decision=decision,
        items=items,
        selected_option_names=selected_option_names,
        confidence=confidence,
        safe_to_apply=safe_to_apply,
        gpt_called=gpt_called,
    )


# ---------------------------------------------------------------------------
# Category A — Idle item resolution (Bucket 0)
# ---------------------------------------------------------------------------

class TestA_IdleItemResolution:
    """Bucket 0 triggering: state==IDLE, low-confidence/UNKNOWN, item-like text."""

    def test_A1_unknown_intent_idle_triggers_bucket0(self) -> None:
        result = pick_bucket(
            state=_IDLE,
            local_intent=Intent.UNKNOWN,
            local_confidence=0.2,
            local_slots=(),
            user_text="a cheeseburger please",
            config=_cfg(b0="shadow"),
        )
        assert result == BUCKET_IDLE_ITEM

    def test_A2_low_confidence_idle_triggers_bucket0(self) -> None:
        result = pick_bucket(
            state=_IDLE,
            local_intent=Intent.ADD_ITEM,
            local_confidence=LOW_CONFIDENCE_THRESHOLD - 0.01,
            local_slots=(),
            user_text="um large fries",
            config=_cfg(b0="shadow"),
        )
        assert result == BUCKET_IDLE_ITEM

    def test_A3_high_confidence_does_not_trigger_bucket0(self) -> None:
        result = pick_bucket(
            state=_IDLE,
            local_intent=Intent.ADD_ITEM,
            local_confidence=0.95,
            local_slots=(),
            user_text="large fries",
            config=_cfg(b0="shadow"),
        )
        assert result is None

    def test_A4_bucket0_disabled_does_not_trigger(self) -> None:
        result = pick_bucket(
            state=_IDLE,
            local_intent=Intent.UNKNOWN,
            local_confidence=0.1,
            local_slots=(),
            user_text="burger please",
            config=_cfg(b0="disabled"),
        )
        assert result is None

    def test_A5_noise_only_does_not_trigger_bucket0(self) -> None:
        # "um" alone is only 1 meaningful token — below minimum
        result = pick_bucket(
            state=_IDLE,
            local_intent=Intent.UNKNOWN,
            local_confidence=0.1,
            local_slots=(),
            user_text="um",
            config=_cfg(b0="shadow"),
        )
        assert result is None

    def test_A6_resolve_returns_local_in_shadow_mode(self) -> None:
        """In shadow mode the source is always 'local' even when GPT resolves."""
        nlu = _nlu(intent=Intent.UNKNOWN, confidence=0.2, text="burger please")
        cfg = _cfg(b0="shadow")
        ctx = _ctx()

        # Patch _call_gpt_for_bucket to avoid real GPT call
        mock_gpt = GptTurnResolution(
            bucket=BUCKET_IDLE_ITEM,
            decision="add_items",
            items=(ResolvedItemPlan(item_name="Burger"),),
            confidence=0.9,
            gpt_called=True,
        )
        with patch(
            "app.nlu.turn_resolver.final_turn_decision_resolver._call_gpt_for_bucket",
            return_value=mock_gpt,
        ):
            decision = resolve(
                state=_IDLE,
                local_nlu=nlu,
                context=ctx,
                config=cfg,
            )

        assert decision.source == "local"
        assert decision.apply_gpt is False
        assert decision.bucket == BUCKET_IDLE_ITEM
        assert decision.gpt_result is not None

    def test_A7_resolve_local_when_bucket0_disabled(self) -> None:
        nlu = _nlu(intent=Intent.UNKNOWN, confidence=0.2, text="burger please")
        cfg = _cfg(b0="disabled")
        ctx = _ctx()
        decision = resolve(state=_IDLE, local_nlu=nlu, context=ctx, config=cfg)
        assert decision.source == "local"
        assert decision.bucket is None
        assert decision.apply_gpt is False


# ---------------------------------------------------------------------------
# Category B — Waiting state option resolution (Bucket 2)
# ---------------------------------------------------------------------------

class TestB_WaitingStateOptionResolution:
    """Bucket 2 triggering: waiting states with option_match_failed=True."""

    def test_B1_waiting_modifier_failed_match_triggers_bucket2(self) -> None:
        result = pick_bucket(
            state=_WAITING_MOD,
            local_intent=Intent.ADD_ITEM,
            local_confidence=0.8,
            local_slots=(),
            user_text="ched ar",
            option_match_failed=True,
            config=_cfg(b2="shadow"),
        )
        assert result == BUCKET_OPTION

    def test_B2_waiting_side_failed_match_triggers_bucket2(self) -> None:
        result = pick_bucket(
            state=_WAITING_SIDE,
            local_intent=Intent.ADD_ITEM,
            local_confidence=0.7,
            local_slots=(),
            user_text="frys",
            option_match_failed=True,
            config=_cfg(b2="shadow"),
        )
        assert result == BUCKET_OPTION

    def test_B3_waiting_state_no_failed_match_does_not_trigger(self) -> None:
        result = pick_bucket(
            state=_WAITING_MOD,
            local_intent=Intent.ADD_ITEM,
            local_confidence=0.8,
            local_slots=(),
            user_text="cheddar",
            option_match_failed=False,  # local matched OK
            config=_cfg(b2="shadow"),
        )
        assert result is None

    def test_B4_bucket2_disabled_does_not_trigger(self) -> None:
        result = pick_bucket(
            state=_WAITING_MOD,
            local_intent=Intent.ADD_ITEM,
            local_confidence=0.5,
            local_slots=(),
            user_text="ched ar",
            option_match_failed=True,
            config=_cfg(b2="disabled"),
        )
        assert result is None

    def test_B5_validate_bucket2_all_names_in_choices(self) -> None:
        gpt = _gpt_resolution(
            bucket=BUCKET_OPTION,
            decision="select_option",
            selected_option_names=("Cheddar",),
            confidence=0.9,
        )
        result = validate_bucket2_result(gpt, choice_names=["Cheddar", "Swiss", "American"])
        assert result.is_safe is True

    def test_B6_validate_bucket2_name_not_in_choices_rejects(self) -> None:
        gpt = _gpt_resolution(
            bucket=BUCKET_OPTION,
            decision="select_option",
            selected_option_names=("Provolone",),
            confidence=0.9,
        )
        result = validate_bucket2_result(gpt, choice_names=["Cheddar", "Swiss"])
        assert result.is_safe is False
        assert "provolone" in (result.reject_reason or "").lower()

    def test_B7_shadow_mode_bucket2_never_applies(self) -> None:
        nlu = _nlu(
            intent=Intent.ADD_ITEM,
            confidence=0.8,
            text="ched ar",
        )
        cfg = _cfg(b2="shadow")
        ctx = _ctx()

        mock_gpt = GptTurnResolution(
            bucket=BUCKET_OPTION,
            decision="select_option",
            selected_option_names=("Cheddar",),
            confidence=0.92,
            gpt_called=True,
        )
        with patch(
            "app.nlu.turn_resolver.final_turn_decision_resolver._call_gpt_for_bucket",
            return_value=mock_gpt,
        ):
            decision = resolve(
                state=_WAITING_MOD,
                local_nlu=nlu,
                context=ctx,
                config=cfg,
                option_match_failed=True,
                choices=["Cheddar", "Swiss"],
            )

        assert decision.source == "local"
        assert decision.apply_gpt is False


# ---------------------------------------------------------------------------
# Category C — Multi-item / staged (Bucket 3)
# ---------------------------------------------------------------------------

class TestC_MultiItemPlanning:
    """Bucket 3 triggering: idle state with compound markers / multiple slots."""

    def test_C1_and_conjunction_triggers_bucket3(self) -> None:
        result = pick_bucket(
            state=_IDLE,
            local_intent=Intent.ADD_ITEM,
            local_confidence=0.6,
            local_slots=(),
            user_text="a burger and fries",
            config=_cfg(b3="shadow"),
        )
        assert result == BUCKET_MULTI_ITEM

    def test_C2_comma_list_triggers_bucket3(self) -> None:
        result = pick_bucket(
            state=_IDLE,
            local_intent=Intent.ADD_ITEM,
            local_confidence=0.7,
            local_slots=(),
            user_text="burger, fries, coke",
            config=_cfg(b3="shadow"),
        )
        assert result == BUCKET_MULTI_ITEM

    def test_C3_two_item_slots_triggers_bucket3(self) -> None:
        slots = (
            _slot("ITEM", "burger"),
            _slot("ITEM", "fries"),
        )
        result = pick_bucket(
            state=_IDLE,
            local_intent=Intent.ADD_ITEM,
            local_confidence=0.75,
            local_slots=slots,
            user_text="burger fries",
            config=_cfg(b3="shadow"),
        )
        assert result == BUCKET_MULTI_ITEM

    def test_C4_single_simple_item_does_not_trigger_bucket3(self) -> None:
        result = pick_bucket(
            state=_IDLE,
            local_intent=Intent.ADD_ITEM,
            local_confidence=0.9,
            local_slots=(_slot("ITEM", "burger"),),
            user_text="a burger",
            config=_cfg(b3="shadow"),
        )
        assert result is None

    def test_C5_bucket3_disabled_does_not_trigger(self) -> None:
        result = pick_bucket(
            state=_IDLE,
            local_intent=Intent.ADD_ITEM,
            local_confidence=0.7,
            local_slots=(),
            user_text="burger and fries",
            config=_cfg(b3="disabled"),
        )
        assert result is None

    def test_C6_bucket3_validation_requires_add_items_decision(self) -> None:
        gpt = _gpt_resolution(
            bucket=BUCKET_MULTI_ITEM,
            decision="clarify",
            items=(),
            confidence=0.8,
        )
        result = validate_bucket3_result(gpt, known_item_names=[])
        assert result.is_safe is False

    def test_C7_bucket3_validation_passes_with_valid_items(self) -> None:
        gpt = _gpt_resolution(
            bucket=BUCKET_MULTI_ITEM,
            decision="add_items",
            items=(
                ResolvedItemPlan(item_name="Burger"),
                ResolvedItemPlan(item_name="Fries"),
            ),
            confidence=0.88,
        )
        result = validate_bucket3_result(gpt, known_item_names=[])
        assert result.is_safe is True

    def test_C8_shadow_mode_bucket3_never_applies(self) -> None:
        nlu = _nlu(
            intent=Intent.ADD_ITEM,
            confidence=0.7,
            text="burger and fries",
        )
        cfg = _cfg(b3="shadow")
        ctx = _ctx()

        mock_gpt = GptTurnResolution(
            bucket=BUCKET_MULTI_ITEM,
            decision="add_items",
            items=(
                ResolvedItemPlan(item_name="Burger"),
                ResolvedItemPlan(item_name="Fries"),
            ),
            confidence=0.9,
            gpt_called=True,
        )
        with patch(
            "app.nlu.turn_resolver.final_turn_decision_resolver._call_gpt_for_bucket",
            return_value=mock_gpt,
        ):
            decision = resolve(
                state=_IDLE,
                local_nlu=nlu,
                context=ctx,
                config=cfg,
            )

        assert decision.source == "local"
        assert decision.apply_gpt is False
        assert decision.bucket == BUCKET_MULTI_ITEM


# ---------------------------------------------------------------------------
# Category D — Safety: GPT never mutates cart / state / FSM
# ---------------------------------------------------------------------------

class TestD_Safety:
    """Safety: GPT result never mutates cart, state, or FSM directly."""

    def test_D1_terminal_state_completed_no_bucket(self) -> None:
        for terminal in (_COMPLETED, _ERROR, _TRANSFER):
            result = pick_bucket(
                state=terminal,
                local_intent=Intent.UNKNOWN,
                local_confidence=0.1,
                local_slots=(),
                user_text="a burger",
                config=_cfg(b0="shadow", b2="shadow", b3="shadow"),
            )
            assert result is None, f"Expected None for terminal state {terminal}"

    def test_D2_empty_text_no_bucket(self) -> None:
        for text in ("", "   ", "\t"):
            result = pick_bucket(
                state=_IDLE,
                local_intent=Intent.UNKNOWN,
                local_confidence=0.1,
                local_slots=(),
                user_text=text,
                config=_cfg(b0="shadow"),
            )
            assert result is None

    def test_D3_safe_to_apply_never_set_by_gpt_parser(self) -> None:
        # GptTurnResolution.safe_to_apply defaults False; only validators change it
        gpt = GptTurnResolution(
            bucket=BUCKET_IDLE_ITEM,
            decision="add_items",
            items=(ResolvedItemPlan(item_name="Burger"),),
            confidence=0.99,
            gpt_called=True,
        )
        assert gpt.safe_to_apply is False

    def test_D4_validate_dispatch_unknown_bucket_rejects(self) -> None:
        gpt = _gpt_resolution(decision="add_items", confidence=0.9)
        result = validate_gpt_result("unknown_bucket_xyz", gpt)
        assert result.is_safe is False

    def test_D5_resolver_exception_falls_back_to_local(self) -> None:
        """When the GPT service raises, resolver silently falls back to local."""
        nlu = _nlu(intent=Intent.UNKNOWN, confidence=0.2, text="large burger")
        cfg = _cfg(b0="inline")
        ctx = _ctx()

        with patch(
            "app.nlu.turn_resolver.final_turn_decision_resolver._call_gpt_for_bucket",
            side_effect=RuntimeError("simulated GPT failure"),
        ):
            decision = resolve(
                state=_IDLE,
                local_nlu=nlu,
                context=ctx,
                config=cfg,
            )

        assert decision.source == "local"
        assert decision.apply_gpt is False

    def test_D6_context_not_mutated_by_resolve(self) -> None:
        """resolve() must not mutate ConversationContext."""
        nlu = _nlu(intent=Intent.UNKNOWN, confidence=0.2, text="burger please")
        cfg = _cfg(b0="shadow")
        ctx = _ctx()

        original_state_field = ctx.current_prompt_field
        original_pending = ctx.pending_add_item
        original_item_id = ctx.current_item_id

        with patch(
            "app.nlu.turn_resolver.final_turn_decision_resolver._call_gpt_for_bucket",
            return_value=GPT_TURN_RESOLUTION_SKIPPED,
        ):
            resolve(state=_IDLE, local_nlu=nlu, context=ctx, config=cfg)

        assert ctx.current_prompt_field == original_state_field
        assert ctx.pending_add_item == original_pending
        assert ctx.current_item_id == original_item_id

    def test_D7_inline_validation_failure_stays_local(self) -> None:
        """When inline validation fails, source is 'local' and apply_gpt=False."""
        nlu = _nlu(intent=Intent.UNKNOWN, confidence=0.2, text="large burger")
        cfg = _cfg(b0="inline")
        ctx = _ctx()

        # GPT returns low confidence that fails validation gate
        mock_gpt = GptTurnResolution(
            bucket=BUCKET_IDLE_ITEM,
            decision="add_items",
            items=(ResolvedItemPlan(item_name="Burger"),),
            confidence=0.3,  # below 0.75 threshold
            gpt_called=True,
        )
        with patch(
            "app.nlu.turn_resolver.final_turn_decision_resolver._call_gpt_for_bucket",
            return_value=mock_gpt,
        ):
            decision = resolve(
                state=_IDLE,
                local_nlu=nlu,
                context=ctx,
                config=cfg,
            )

        assert decision.source == "local"
        assert decision.apply_gpt is False
        assert decision.gpt_result is not None
        assert "confidence" in (decision.reason or "")

    def test_D8_inline_bucket0_applies_when_validation_passes(self) -> None:
        """In inline mode, apply_gpt=True when validation passes."""
        nlu = _nlu(intent=Intent.UNKNOWN, confidence=0.2, text="large burger")
        cfg = _cfg(b0="inline")
        ctx = _ctx()

        mock_gpt = GptTurnResolution(
            bucket=BUCKET_IDLE_ITEM,
            decision="add_items",
            items=(ResolvedItemPlan(item_name="Burger"),),
            confidence=0.92,
            gpt_called=True,
        )
        with patch(
            "app.nlu.turn_resolver.final_turn_decision_resolver._call_gpt_for_bucket",
            return_value=mock_gpt,
        ):
            decision = resolve(
                state=_IDLE,
                local_nlu=nlu,
                context=ctx,
                config=cfg,
                allowed_intents=["add_item"],
            )

        assert decision.source == "gpt"
        assert decision.apply_gpt is True
        assert decision.gpt_result is not None
        assert decision.gpt_result.safe_to_apply is True


# ---------------------------------------------------------------------------
# Category E — Logging / schema structure
# ---------------------------------------------------------------------------

class TestE_LoggingAndSchema:
    """Schema integrity and logging structure for GptTurnResolution."""

    def test_E1_gpt_turn_resolution_skipped_sentinel(self) -> None:
        s = GPT_TURN_RESOLUTION_SKIPPED
        assert s.decision == "skipped"
        assert s.gpt_called is False
        assert s.bucket == "none"
        assert s.safe_to_apply is False

    def test_E2_to_log_dict_contains_required_fields(self) -> None:
        gpt = GptTurnResolution(
            bucket=BUCKET_IDLE_ITEM,
            decision="add_items",
            items=(ResolvedItemPlan(item_name="Burger", quantity=2),),
            confidence=0.88,
            gpt_called=True,
            model="gpt-4o-mini",
            prompt_chars=120,
            completion_chars=40,
        )
        d = gpt.to_log_dict()

        assert d["bucket"] == BUCKET_IDLE_ITEM
        assert d["decision"] == "add_items"
        assert d["gpt_called"] is True
        assert d["safe_to_apply"] is False
        assert len(d["items"]) == 1
        assert d["items"][0]["item_name"] == "Burger"
        assert d["items"][0]["quantity"] == 2
        assert d["model"] == "gpt-4o-mini"
        assert "prompt_chars" in d
        assert "completion_chars" in d

    def test_E3_to_log_dict_no_api_key_or_pii(self) -> None:
        gpt = GptTurnResolution(
            bucket=BUCKET_OPTION,
            decision="select_option",
            selected_option_names=("Cheddar",),
            confidence=0.9,
            gpt_called=True,
        )
        d = gpt.to_log_dict()
        serialized = str(d)
        assert "api_key" not in serialized.lower()
        assert "phone" not in serialized.lower()
        assert "password" not in serialized.lower()

    def test_E4_context_packet_to_dict_is_json_serialisable(self) -> None:
        import json
        nlu = _nlu(
            intent=Intent.UNKNOWN,
            confidence=0.3,
            text="burger",
            candidates=(
                IntentCandidate(
                    intent_main="food",
                    intent_sub_intent="add_item",
                    canonical_intent="add_item",
                    confidence=0.3,
                ),
            ),
        )
        ctx = _ctx()
        packet = build_context_packet(
            bucket=BUCKET_IDLE_ITEM,
            user_text="burger",
            state=_IDLE,
            context=ctx,
            local_nlu=nlu,
            choices=("Cheddar", "Swiss"),
            cart_item_names=("Burger",),
            allowed_intents=("add_item",),
        )
        d = packet.to_dict()
        serialized = json.dumps(d)  # must not raise
        assert isinstance(serialized, str)
        assert "burger" in serialized

    def test_E5_final_turn_decision_has_gpt_result_in_shadow(self) -> None:
        """In shadow mode the gpt_result is logged even though source=='local'."""
        nlu = _nlu(intent=Intent.UNKNOWN, confidence=0.2, text="large burger")
        cfg = _cfg(b0="shadow")
        ctx = _ctx()

        mock_gpt = GptTurnResolution(
            bucket=BUCKET_IDLE_ITEM,
            decision="add_items",
            items=(ResolvedItemPlan(item_name="Burger"),),
            confidence=0.88,
            gpt_called=True,
        )
        with patch(
            "app.nlu.turn_resolver.final_turn_decision_resolver._call_gpt_for_bucket",
            return_value=mock_gpt,
        ):
            decision = resolve(
                state=_IDLE,
                local_nlu=nlu,
                context=ctx,
                config=cfg,
            )

        assert decision.gpt_result is not None
        assert decision.gpt_result.gpt_called is True
        assert decision.gpt_result.decision == "add_items"
        # In shadow mode: local intent / slots are preserved
        assert decision.local_intent == Intent.UNKNOWN
        assert decision.source == "local"


# ---------------------------------------------------------------------------
# Bonus: config validation
# ---------------------------------------------------------------------------

class TestBucketConfig:
    """SemanticRepairConfig validates bucket mode values."""

    def test_default_bucket_modes_are_disabled(self) -> None:
        cfg = SemanticRepairConfig(phase=2, model="gpt-4o-mini", timeout_seconds=0.35)
        assert cfg.bucket_0_mode == "disabled"
        assert cfg.bucket_2_mode == "disabled"
        assert cfg.bucket_3_mode == "disabled"
        assert cfg.bucket_timeout_ms == 1200

    def test_valid_shadow_mode_accepted(self) -> None:
        cfg = SemanticRepairConfig(
            phase=2,
            model="gpt-4o-mini",
            timeout_seconds=0.35,
            bucket_0_mode="shadow",
            bucket_2_mode="shadow",
            bucket_3_mode="shadow",
        )
        assert cfg.bucket_0_mode == "shadow"

    def test_valid_inline_mode_accepted(self) -> None:
        cfg = SemanticRepairConfig(
            phase=2,
            model="gpt-4o-mini",
            timeout_seconds=0.35,
            bucket_0_mode="inline",
        )
        assert cfg.bucket_0_mode == "inline"

    def test_invalid_bucket0_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="bucket_0_mode"):
            SemanticRepairConfig(
                phase=2,
                model="gpt-4o-mini",
                timeout_seconds=0.35,
                bucket_0_mode="always_on",
            )

    def test_invalid_bucket2_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="bucket_2_mode"):
            SemanticRepairConfig(
                phase=2,
                model="gpt-4o-mini",
                timeout_seconds=0.35,
                bucket_2_mode="enabled",
            )

    def test_invalid_bucket3_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="bucket_3_mode"):
            SemanticRepairConfig(
                phase=2,
                model="gpt-4o-mini",
                timeout_seconds=0.35,
                bucket_3_mode="on",
            )
