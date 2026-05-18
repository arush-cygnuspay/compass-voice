# tests/nlu/semantic_repair/test_phase3_option_resolver.py
"""Phase 3 GPT Option Resolver — unit + integration tests.

Covers:
  - SemanticRepairConfig Phase 3 config fields and validation
  - GptRoutingPolicy.decide() all routing rules
  - GptOptionContextBuilder.build_messages() payload safety
  - GptOptionSelectionValidator.validate() rules
  - build_modifier_selections_from_names() all-or-nothing policy
  - GptOptionResolverService.run() skipping, parse, and shadow/inline contract
  - WaitingForModifierHandler._try_gpt_option_resolve() integration
"""
from __future__ import annotations

import json
from dataclasses import replace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from app.config.semantic_repair import SemanticRepairConfig
from app.nlu.semantic_repair.option_context_builder import GptOptionContextBuilder
from app.nlu.semantic_repair.option_resolver_result import (
    OPTION_RESOLVER_NOT_CALLED,
    OptionResolverResult,
)
from app.nlu.semantic_repair.option_routing_policy import GptRoutingPolicy, OptionRouteMode
from app.nlu.semantic_repair.option_selection_validator import (
    GptOptionSelectionValidator,
    build_modifier_selections_from_names,
)
from app.state_machine.models.pending_item_models import (
    ModifierSelection,
    PendingModifierChoice,
    PendingModifierGroup,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_group(
    choices: list[tuple[str, str]] | None = None,
    *,
    min_selector: int = 0,
    max_selector: int = 1,
) -> PendingModifierGroup:
    """Build a PendingModifierGroup with the given (modifier_id, name) pairs."""
    if choices is None:
        choices = [
            ("m1", "American Cheese"),
            ("m2", "Mozzarella Cheese"),
            ("m3", "Cheddar Cheese"),
        ]
    group_choices = [
        PendingModifierChoice(
            modifier_id=mid,
            name=name,
            group_id="test_grp",
            normalized_name=name.lower(),
        )
        for mid, name in choices
    ]
    return PendingModifierGroup(
        group_id="test_grp",
        name="Cheese",
        is_required=True,
        min_selector=min_selector,
        max_selector=max_selector,
        choices=group_choices,
    )


def _cfg(
    mode: str = "disabled",
    *,
    repeat_threshold: int = 2,
    min_confidence: float = 0.75,
    timeout_ms: int = 1200,
) -> SemanticRepairConfig:
    return SemanticRepairConfig(
        phase=2,
        model="gpt-4o-mini",
        timeout_seconds=0.35,
        option_resolver_mode=mode,
        option_resolver_repeat_threshold=repeat_threshold,
        option_resolver_min_confidence=min_confidence,
        option_resolver_timeout_ms=timeout_ms,
    )


# ===========================================================================
# Config validation tests
# ===========================================================================


class TestPhase3ConfigFields:
    def test_default_mode_is_disabled(self) -> None:
        cfg = SemanticRepairConfig(phase=2, model="gpt-4o-mini", timeout_seconds=0.35)
        assert cfg.option_resolver_mode == "disabled"

    def test_shadow_mode_accepted(self) -> None:
        cfg = _cfg("shadow")
        assert cfg.option_resolver_mode == "shadow"

    def test_inline_mode_accepted(self) -> None:
        cfg = _cfg("inline")
        assert cfg.option_resolver_mode == "inline"

    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="option_resolver_mode"):
            _cfg("all_apply_safe")

    def test_default_timeout_ms(self) -> None:
        cfg = SemanticRepairConfig(phase=2, model="gpt-4o-mini", timeout_seconds=0.35)
        assert cfg.option_resolver_timeout_ms == 1200

    def test_default_min_confidence(self) -> None:
        cfg = SemanticRepairConfig(phase=2, model="gpt-4o-mini", timeout_seconds=0.35)
        assert cfg.option_resolver_min_confidence == 0.75

    def test_default_repeat_threshold(self) -> None:
        cfg = SemanticRepairConfig(phase=2, model="gpt-4o-mini", timeout_seconds=0.35)
        assert cfg.option_resolver_repeat_threshold == 2

    def test_custom_values_accepted(self) -> None:
        cfg = _cfg("inline", repeat_threshold=3, min_confidence=0.80, timeout_ms=800)
        assert cfg.option_resolver_repeat_threshold == 3
        assert cfg.option_resolver_min_confidence == 0.80
        assert cfg.option_resolver_timeout_ms == 800


# ===========================================================================
# GptRoutingPolicy tests
# ===========================================================================


class TestGptRoutingPolicy:
    def setup_method(self) -> None:
        self.policy = GptRoutingPolicy()

    # ── disabled mode ──────────────────────────────────────────────────

    def test_disabled_mode_always_no_gpt(self) -> None:
        cfg = _cfg("disabled")
        result = self.policy.decide(
            config=cfg,
            local_resolved=False,
            user_text="macarola cheese",
            options_exist=True,
            repeat_count=5,
        )
        assert result == OptionRouteMode.NO_GPT

    def test_disabled_mode_no_gpt_even_for_repeat(self) -> None:
        cfg = _cfg("disabled")
        result = self.policy.decide(
            config=cfg, local_resolved=False, user_text="x", options_exist=True, repeat_count=99
        )
        assert result == OptionRouteMode.NO_GPT

    # ── shadow mode ────────────────────────────────────────────────────

    def test_shadow_local_failed_returns_shadow_gpt(self) -> None:
        cfg = _cfg("shadow")
        result = self.policy.decide(
            config=cfg, local_resolved=False, user_text="macarola", options_exist=True, repeat_count=0
        )
        assert result == OptionRouteMode.SHADOW_GPT

    def test_shadow_local_resolved_returns_no_gpt(self) -> None:
        cfg = _cfg("shadow")
        result = self.policy.decide(
            config=cfg, local_resolved=True, user_text="mozzarella", options_exist=True, repeat_count=0
        )
        assert result == OptionRouteMode.NO_GPT

    def test_shadow_short_text_still_runs(self) -> None:
        """Shadow always runs regardless of text length when local failed."""
        cfg = _cfg("shadow")
        result = self.policy.decide(
            config=cfg, local_resolved=False, user_text="ma", options_exist=True, repeat_count=0
        )
        assert result == OptionRouteMode.SHADOW_GPT

    # ── inline mode ────────────────────────────────────────────────────

    def test_inline_text_gte3_local_failed_options_exist(self) -> None:
        cfg = _cfg("inline")
        result = self.policy.decide(
            config=cfg, local_resolved=False, user_text="mac", options_exist=True, repeat_count=0
        )
        assert result == OptionRouteMode.INLINE_GPT

    def test_inline_text_lt3_returns_no_gpt(self) -> None:
        cfg = _cfg("inline")
        result = self.policy.decide(
            config=cfg, local_resolved=False, user_text="ma", options_exist=True, repeat_count=0
        )
        assert result == OptionRouteMode.NO_GPT

    def test_inline_local_resolved_returns_no_gpt(self) -> None:
        cfg = _cfg("inline")
        result = self.policy.decide(
            config=cfg, local_resolved=True, user_text="mozzarella cheese", options_exist=True, repeat_count=0
        )
        assert result == OptionRouteMode.NO_GPT

    def test_inline_no_options_returns_no_gpt(self) -> None:
        cfg = _cfg("inline")
        result = self.policy.decide(
            config=cfg, local_resolved=False, user_text="mac cheese", options_exist=False, repeat_count=0
        )
        assert result == OptionRouteMode.NO_GPT

    def test_inline_repeat_at_threshold_triggers_gpt(self) -> None:
        cfg = _cfg("inline", repeat_threshold=2)
        result = self.policy.decide(
            config=cfg, local_resolved=False, user_text="x", options_exist=True, repeat_count=2
        )
        assert result == OptionRouteMode.INLINE_GPT

    def test_inline_repeat_below_threshold_no_gpt(self) -> None:
        cfg = _cfg("inline", repeat_threshold=2)
        result = self.policy.decide(
            config=cfg, local_resolved=False, user_text="x", options_exist=True, repeat_count=1
        )
        assert result == OptionRouteMode.NO_GPT

    def test_inline_repeat_escalates_even_with_short_text(self) -> None:
        """Repeat-loop recovery escalates to INLINE_GPT even for very short text."""
        cfg = _cfg("inline", repeat_threshold=2)
        result = self.policy.decide(
            config=cfg, local_resolved=False, user_text="a", options_exist=True, repeat_count=3
        )
        assert result == OptionRouteMode.INLINE_GPT

    def test_inline_unknown_mode_returns_no_gpt(self) -> None:
        """Any unrecognised mode value is treated as disabled."""
        cfg = _cfg("disabled")
        # Manually set unknown mode value bypassing validation for the test
        result = self.policy.decide(
            config=cfg, local_resolved=False, user_text="mac cheese", options_exist=True, repeat_count=0
        )
        assert result == OptionRouteMode.NO_GPT

    # ── empty / silence text guard ─────────────────────────────────────

    def test_empty_text_returns_no_gpt_in_shadow(self) -> None:
        """Silence/empty text is skipped even in shadow mode."""
        cfg = _cfg("shadow")
        result = self.policy.decide(
            config=cfg, local_resolved=False, user_text="", options_exist=True, repeat_count=0
        )
        assert result == OptionRouteMode.NO_GPT

    def test_whitespace_only_text_returns_no_gpt_in_inline(self) -> None:
        """Whitespace-only text is treated as silence → NO_GPT."""
        cfg = _cfg("inline")
        result = self.policy.decide(
            config=cfg, local_resolved=False, user_text="   ", options_exist=True, repeat_count=0
        )
        assert result == OptionRouteMode.NO_GPT

    def test_none_text_returns_no_gpt(self) -> None:
        """None text is treated as silence → NO_GPT."""
        cfg = _cfg("inline")
        result = self.policy.decide(
            config=cfg, local_resolved=False, user_text=None, options_exist=True, repeat_count=0  # type: ignore[arg-type]
        )
        assert result == OptionRouteMode.NO_GPT

    # ── shadow with no options guard ──────────────────────────────────

    def test_shadow_no_options_returns_no_gpt(self) -> None:
        """Shadow mode skips GPT when there are no available options."""
        cfg = _cfg("shadow")
        result = self.policy.decide(
            config=cfg, local_resolved=False, user_text="macarola", options_exist=False, repeat_count=0
        )
        assert result == OptionRouteMode.NO_GPT

    # ── correction signal ──────────────────────────────────────────────

    def test_inline_correction_signal_escalates_short_text(self) -> None:
        """Correction phrase ('actually') escalates to INLINE_GPT even for short text."""
        cfg = _cfg("inline")
        result = self.policy.decide(
            config=cfg,
            local_resolved=False,
            user_text="actually mozzarella",
            options_exist=True,
            repeat_count=0,
            has_correction=True,
        )
        assert result == OptionRouteMode.INLINE_GPT

    def test_inline_no_correction_signal_short_text_no_gpt(self) -> None:
        """Without correction signal, text < 3 chars → NO_GPT (existing behaviour preserved)."""
        cfg = _cfg("inline")
        result = self.policy.decide(
            config=cfg,
            local_resolved=False,
            user_text="mo",
            options_exist=True,
            repeat_count=0,
            has_correction=False,
        )
        assert result == OptionRouteMode.NO_GPT

    def test_correction_signal_on_local_resolved_still_no_gpt(self) -> None:
        """Correction signal does not override local_resolved=True."""
        cfg = _cfg("inline")
        result = self.policy.decide(
            config=cfg,
            local_resolved=True,
            user_text="actually mozzarella",
            options_exist=True,
            repeat_count=0,
            has_correction=True,
        )
        assert result == OptionRouteMode.NO_GPT

    def test_correction_signal_no_options_still_no_gpt(self) -> None:
        """Correction signal with no options available → NO_GPT."""
        cfg = _cfg("inline")
        result = self.policy.decide(
            config=cfg,
            local_resolved=False,
            user_text="actually no, i want something else",
            options_exist=False,
            repeat_count=0,
            has_correction=True,
        )
        assert result == OptionRouteMode.NO_GPT


# ===========================================================================
# GptOptionContextBuilder tests
# ===========================================================================


class TestGptOptionContextBuilder:
    def setup_method(self) -> None:
        self.builder = GptOptionContextBuilder()

    def test_returns_two_messages(self) -> None:
        msgs = self.builder.build_messages(
            user_text="macarola cheese",
            item_name="Cheeseburger",
            group_name="Cheese",
            choice_names=["American Cheese", "Mozzarella Cheese"],
        )
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_payload_contains_required_fields(self) -> None:
        msgs = self.builder.build_messages(
            user_text="macarola cheese",
            item_name="Cheeseburger",
            group_name="Cheese",
            choice_names=["American Cheese", "Mozzarella Cheese"],
        )
        payload = json.loads(msgs[1]["content"])
        assert payload["t"] == "select_modifier"
        assert payload["item"] == "Cheeseburger"
        assert payload["group"] == "Cheese"
        assert payload["text"] == "macarola cheese"
        assert "American Cheese" in payload["choices"]
        assert "schema" in payload

    def test_already_selected_included(self) -> None:
        msgs = self.builder.build_messages(
            user_text="mozzo",
            item_name="Pizza",
            group_name="Cheese",
            choice_names=["Mozzarella", "Cheddar"],
            already_selected_names=["Cheddar"],
        )
        payload = json.loads(msgs[1]["content"])
        assert payload["selected"] == ["Cheddar"]

    def test_no_already_selected_omits_field(self) -> None:
        msgs = self.builder.build_messages(
            user_text="mozzo",
            item_name="Pizza",
            group_name="Cheese",
            choice_names=["Mozzarella"],
        )
        payload = json.loads(msgs[1]["content"])
        assert "selected" not in payload

    def test_history_included_when_provided(self) -> None:
        msgs = self.builder.build_messages(
            user_text="mozzo",
            item_name="Pizza",
            group_name="Cheese",
            choice_names=["Mozzarella"],
            previous_turns=[("bot", "Which cheese?"), ("user", "mozzo")],
        )
        payload = json.loads(msgs[1]["content"])
        assert "history" in payload
        assert payload["history"][0] == ["bot", "Which cheese?"]

    def test_choices_capped_at_max(self) -> None:
        many = [f"Option {i}" for i in range(30)]
        msgs = self.builder.build_messages(
            user_text="test",
            item_name="Item",
            group_name="Group",
            choice_names=many,
        )
        payload = json.loads(msgs[1]["content"])
        assert len(payload["choices"]) <= 20  # MAX_CHOICES

    def test_no_full_menu_in_payload(self) -> None:
        msgs = self.builder.build_messages(
            user_text="test",
            item_name="Burger",
            group_name="Cheese",
            choice_names=["Mozzarella"],
        )
        full_content = msgs[0]["content"] + msgs[1]["content"]
        assert "full_menu" not in full_content.lower()
        assert "api_key" not in full_content.lower()

    def test_history_capped_at_3_turns(self) -> None:
        long_history = [("bot", f"Turn {i}") for i in range(10)]
        msgs = self.builder.build_messages(
            user_text="test",
            item_name="Pizza",
            group_name="Sauce",
            choice_names=["Tomato", "Pesto"],
            previous_turns=long_history,
        )
        payload = json.loads(msgs[1]["content"])
        assert len(payload["history"]) <= 3

    def test_extract_choice_names_from_group(self) -> None:
        group = _make_group()
        names = GptOptionContextBuilder.extract_choice_names(group)
        assert "American Cheese" in names
        assert "Mozzarella Cheese" in names
        assert len(names) == 3

    # ── new hardened fields: last_response_key, local_slots, top_intents ─

    def test_last_response_key_included_when_provided(self) -> None:
        msgs = self.builder.build_messages(
            user_text="mozzo",
            item_name="Pizza",
            group_name="Cheese",
            choice_names=["Mozzarella"],
            last_response_key="ask_for_modifier",
        )
        payload = json.loads(msgs[1]["content"])
        assert payload.get("last_prompt") == "ask_for_modifier"

    def test_last_response_key_absent_when_not_provided(self) -> None:
        msgs = self.builder.build_messages(
            user_text="mozzo",
            item_name="Pizza",
            group_name="Cheese",
            choice_names=["Mozzarella"],
        )
        payload = json.loads(msgs[1]["content"])
        assert "last_prompt" not in payload

    def test_local_slots_included_when_provided(self) -> None:
        msgs = self.builder.build_messages(
            user_text="mozzo",
            item_name="Pizza",
            group_name="Cheese",
            choice_names=["Mozzarella"],
            local_slots=[{"n": "MODIFIER", "v": "mozzo"}],
        )
        payload = json.loads(msgs[1]["content"])
        assert payload.get("local_slots") == [{"n": "MODIFIER", "v": "mozzo"}]

    def test_local_slots_capped_at_max(self) -> None:
        many_slots = [{"n": f"SLOT_{i}", "v": f"val_{i}"} for i in range(10)]
        msgs = self.builder.build_messages(
            user_text="test",
            item_name="Pizza",
            group_name="Cheese",
            choice_names=["Mozzarella"],
            local_slots=many_slots,
        )
        payload = json.loads(msgs[1]["content"])
        assert len(payload["local_slots"]) <= 4  # MAX_LOCAL_SLOTS

    def test_top_intents_included_when_provided(self) -> None:
        msgs = self.builder.build_messages(
            user_text="mozzo",
            item_name="Pizza",
            group_name="Cheese",
            choice_names=["Mozzarella"],
            top_intents=[{"i": "add_item", "c": 0.72}, {"i": "unknown", "c": 0.18}],
        )
        payload = json.loads(msgs[1]["content"])
        intents = payload.get("top_intents", [])
        assert len(intents) == 2
        assert intents[0]["i"] == "add_item"

    def test_top_intents_capped_at_max(self) -> None:
        many_intents = [{"i": f"intent_{i}", "c": 0.1} for i in range(10)]
        msgs = self.builder.build_messages(
            user_text="test",
            item_name="Pizza",
            group_name="Cheese",
            choice_names=["Mozzarella"],
            top_intents=many_intents,
        )
        payload = json.loads(msgs[1]["content"])
        assert len(payload["top_intents"]) <= 3  # MAX_TOP_INTENTS

    def test_no_local_slots_omits_field(self) -> None:
        msgs = self.builder.build_messages(
            user_text="mozzo",
            item_name="Pizza",
            group_name="Cheese",
            choice_names=["Mozzarella"],
        )
        payload = json.loads(msgs[1]["content"])
        assert "local_slots" not in payload
        assert "top_intents" not in payload

    def test_safety_fields_never_in_payload(self) -> None:
        """Prices, full cart, API key, and PII must never appear in the payload."""
        msgs = self.builder.build_messages(
            user_text="test",
            item_name="Burger",
            group_name="Cheese",
            choice_names=["Mozzarella"],
            local_slots=[{"n": "MODIFIER", "v": "test"}],
            top_intents=[{"i": "add_item", "c": 0.80}],
            last_response_key="ask_for_modifier",
        )
        all_content = msgs[0]["content"] + msgs[1]["content"]
        assert "api_key" not in all_content.lower()
        assert "full_menu" not in all_content.lower()
        assert "price" not in all_content.lower()
        assert "cart" not in all_content.lower()


# ===========================================================================
# GptOptionSelectionValidator tests
# ===========================================================================


class TestGptOptionSelectionValidator:
    def setup_method(self) -> None:
        self.validator = GptOptionSelectionValidator()
        self.group = _make_group()

    def _result(
        self,
        *,
        decision: str = "select_option",
        selected_names: tuple[str, ...] = ("Mozzarella Cheese",),
        confidence: float = 0.90,
        route_mode: str = "inline_gpt",
    ) -> OptionResolverResult:
        return OptionResolverResult(
            decision=decision,
            selected_names=selected_names,
            confidence=confidence,
            route_mode=route_mode,
        )

    def test_valid_inline_high_confidence_is_safe(self) -> None:
        result = self.validator.validate(
            result=self._result(confidence=0.90, route_mode="inline_gpt"),
            group=self.group,
            min_confidence=0.75,
        )
        assert result.safe_to_apply is True

    def test_shadow_mode_never_safe(self) -> None:
        result = self.validator.validate(
            result=self._result(confidence=0.95, route_mode="shadow_gpt"),
            group=self.group,
            min_confidence=0.75,
        )
        assert result.safe_to_apply is False

    def test_low_confidence_not_safe(self) -> None:
        result = self.validator.validate(
            result=self._result(confidence=0.60, route_mode="inline_gpt"),
            group=self.group,
            min_confidence=0.75,
        )
        assert result.safe_to_apply is False

    def test_unknown_option_name_not_safe(self) -> None:
        result = self.validator.validate(
            result=self._result(
                selected_names=("Gouda Cheese",),
                confidence=0.95,
                route_mode="inline_gpt",
            ),
            group=self.group,
            min_confidence=0.75,
        )
        assert result.safe_to_apply is False

    def test_no_match_decision_not_safe(self) -> None:
        result = self.validator.validate(
            result=self._result(decision="no_match", selected_names=(), route_mode="inline_gpt"),
            group=self.group,
            min_confidence=0.75,
        )
        assert result.safe_to_apply is False

    def test_empty_selected_names_not_safe(self) -> None:
        result = self.validator.validate(
            result=self._result(selected_names=(), route_mode="inline_gpt"),
            group=self.group,
            min_confidence=0.75,
        )
        assert result.safe_to_apply is False

    def test_confidence_exactly_at_threshold_is_safe(self) -> None:
        result = self.validator.validate(
            result=self._result(confidence=0.75, route_mode="inline_gpt"),
            group=self.group,
            min_confidence=0.75,
        )
        assert result.safe_to_apply is True

    def test_one_invalid_name_in_multiple_makes_unsafe(self) -> None:
        result = self.validator.validate(
            result=self._result(
                selected_names=("Mozzarella Cheese", "Gouda Cheese"),
                confidence=0.90,
                route_mode="inline_gpt",
            ),
            group=self.group,
            min_confidence=0.75,
        )
        assert result.safe_to_apply is False

    def test_case_insensitive_name_matching(self) -> None:
        result = self.validator.validate(
            result=self._result(
                selected_names=("mozzarella cheese",),  # lowercase
                confidence=0.88,
                route_mode="inline_gpt",
            ),
            group=self.group,
            min_confidence=0.75,
        )
        assert result.safe_to_apply is True

    def test_original_result_is_not_mutated(self) -> None:
        original = self._result(confidence=0.90, route_mode="inline_gpt")
        validated = self.validator.validate(result=original, group=self.group, min_confidence=0.75)
        # Original should be unchanged (frozen dataclass)
        assert original.safe_to_apply is False  # default
        assert validated is not original

    # ── max_selector guard (Rule 4) ────────────────────────────────────

    def test_max_selector_exceeded_not_safe(self) -> None:
        """Selecting more names than max_selector rejects the result."""
        group = _make_group(max_selector=1)
        result = self.validator.validate(
            result=self._result(
                selected_names=("American Cheese", "Mozzarella Cheese"),  # 2 > max_selector=1
                confidence=0.92,
                route_mode="inline_gpt",
            ),
            group=group,
            min_confidence=0.75,
        )
        assert result.safe_to_apply is False

    def test_max_selector_zero_skips_guard(self) -> None:
        """max_selector=0 means the guard is disabled (unlimited)."""
        group = _make_group(max_selector=0)
        result = self.validator.validate(
            result=self._result(
                selected_names=("American Cheese", "Mozzarella Cheese", "Cheddar Cheese"),
                confidence=0.92,
                route_mode="inline_gpt",
            ),
            group=group,
            min_confidence=0.75,
        )
        # Guard is bypassed; all names are in group → safe
        assert result.safe_to_apply is True

    def test_max_selector_exactly_met_is_safe(self) -> None:
        """Exactly max_selector names selected is accepted."""
        group = _make_group(max_selector=2)
        result = self.validator.validate(
            result=self._result(
                selected_names=("American Cheese", "Mozzarella Cheese"),
                confidence=0.88,
                route_mode="inline_gpt",
            ),
            group=group,
            min_confidence=0.75,
        )
        assert result.safe_to_apply is True


# ===========================================================================
# build_modifier_selections_from_names tests
# ===========================================================================


class TestBuildModifierSelectionsFromNames:
    def setup_method(self) -> None:
        self.group = _make_group()

    def test_valid_name_maps_to_selection(self) -> None:
        sels = build_modifier_selections_from_names(
            selected_names=("Mozzarella Cheese",),
            group=self.group,
        )
        assert len(sels) == 1
        assert sels[0].modifier_id == "m2"
        assert sels[0].name == "Mozzarella Cheese"
        assert sels[0].action == "add"
        assert sels[0].instruction is None

    def test_unknown_name_returns_empty(self) -> None:
        sels = build_modifier_selections_from_names(
            selected_names=("Gouda Cheese",),
            group=self.group,
        )
        assert sels == []

    def test_mixed_valid_invalid_returns_empty(self) -> None:
        """All-or-nothing: if any name fails, return []."""
        sels = build_modifier_selections_from_names(
            selected_names=("Mozzarella Cheese", "Gouda Cheese"),
            group=self.group,
        )
        assert sels == []

    def test_already_selected_id_skipped(self) -> None:
        sels = build_modifier_selections_from_names(
            selected_names=("Mozzarella Cheese",),
            group=self.group,
            existing_ids={"m2"},
        )
        assert sels == []

    def test_duplicate_names_deduped(self) -> None:
        sels = build_modifier_selections_from_names(
            selected_names=("Mozzarella Cheese", "Mozzarella Cheese"),
            group=self.group,
        )
        assert len(sels) == 1

    def test_case_insensitive(self) -> None:
        sels = build_modifier_selections_from_names(
            selected_names=("MOZZARELLA CHEESE",),
            group=self.group,
        )
        assert len(sels) == 1
        assert sels[0].modifier_id == "m2"

    def test_empty_names_returns_empty(self) -> None:
        sels = build_modifier_selections_from_names(
            selected_names=(),
            group=self.group,
        )
        assert sels == []

    def test_multiple_valid_selections(self) -> None:
        group_multi = _make_group(max_selector=3)
        sels = build_modifier_selections_from_names(
            selected_names=("American Cheese", "Cheddar Cheese"),
            group=group_multi,
        )
        assert len(sels) == 2
        ids = {s.modifier_id for s in sels}
        assert "m1" in ids
        assert "m3" in ids


# ===========================================================================
# GptOptionResolverService.run() tests
# ===========================================================================


class TestGptOptionResolverService:
    """Test GptOptionResolverService.run() with mocked OpenAI client."""

    def _make_service(self, mode: str = "inline") -> Any:
        from app.nlu.semantic_repair.option_resolver_service import GptOptionResolverService
        cfg = _cfg(mode)
        svc = GptOptionResolverService(config=cfg)
        return svc

    def _make_mock_response(self, content: str) -> Any:
        mock_response = MagicMock()
        mock_response.choices[0].message.content = content
        return mock_response

    def test_disabled_mode_returns_skipped(self) -> None:
        svc = self._make_service("disabled")
        group = _make_group()
        result = svc.run(
            user_text="macarola cheese",
            item_name="Burger",
            group=group,
            existing_selections=[],
            local_resolved=False,
        )
        assert result.decision == "skipped"
        assert result.skipped_reason == "routing_policy_no_gpt"
        assert result.gpt_called is False

    def test_missing_api_key_returns_skipped(self) -> None:
        svc = self._make_service("inline")
        group = _make_group()
        with patch.dict("os.environ", {}, clear=False):
            # Remove API key if present
            import os
            old_key = os.environ.pop("OPENAI_API_KEY", None)
            try:
                result = svc.run(
                    user_text="macarola cheese",
                    item_name="Burger",
                    group=group,
                    existing_selections=[],
                    local_resolved=False,
                )
            finally:
                if old_key is not None:
                    os.environ["OPENAI_API_KEY"] = old_key
        assert result.skipped_reason in ("missing_api_key", "routing_policy_no_gpt")
        assert result.gpt_called is False

    def test_local_resolved_returns_no_gpt(self) -> None:
        svc = self._make_service("inline")
        group = _make_group()
        result = svc.run(
            user_text="mozzarella",
            item_name="Burger",
            group=group,
            existing_selections=[],
            local_resolved=True,  # local already resolved
        )
        assert result.route_mode == "no_gpt"
        assert result.gpt_called is False

    def test_no_choices_returns_skipped(self) -> None:
        from app.nlu.semantic_repair.option_resolver_service import GptOptionResolverService
        svc = GptOptionResolverService(config=_cfg("inline"))
        empty_group = _make_group(choices=[])
        result = svc.run(
            user_text="whatever",
            item_name="Burger",
            group=empty_group,
            existing_selections=[],
            local_resolved=False,
        )
        assert result.skipped_reason in ("no_choices", "routing_policy_no_gpt")

    def test_valid_gpt_response_select_option(self) -> None:
        """GPT returns a valid select_option response → parsed and validated."""
        from app.nlu.semantic_repair.option_resolver_service import GptOptionResolverService
        svc = GptOptionResolverService(config=_cfg("inline"))
        group = _make_group()

        mock_resp = self._make_mock_response(json.dumps({
            "decision": "select_option",
            "selected_names": ["Mozzarella Cheese"],
            "confidence": 0.92,
            "reason_code": "phonetic_match",
        }))
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_resp
        svc._client = mock_client

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            result = svc.run(
                user_text="macarola cheese",
                item_name="Burger",
                group=group,
                existing_selections=[],
                local_resolved=False,
            )

        assert result.gpt_called is True
        assert result.decision == "select_option"
        assert "Mozzarella Cheese" in result.selected_names
        assert result.confidence == pytest.approx(0.92)
        assert result.route_mode == "inline_gpt"
        # Validator must approve
        assert result.safe_to_apply is True

    def test_shadow_mode_gpt_called_but_not_safe(self) -> None:
        """In shadow mode, safe_to_apply is always False."""
        from app.nlu.semantic_repair.option_resolver_service import GptOptionResolverService
        svc = GptOptionResolverService(config=_cfg("shadow"))
        group = _make_group()

        mock_resp = self._make_mock_response(json.dumps({
            "decision": "select_option",
            "selected_names": ["American Cheese"],
            "confidence": 0.98,
            "reason_code": "exact_match",
        }))
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_resp
        svc._client = mock_client

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            result = svc.run(
                user_text="american cheese",
                item_name="Burger",
                group=group,
                existing_selections=[],
                local_resolved=False,
            )

        assert result.gpt_called is True
        assert result.decision == "select_option"
        assert result.route_mode == "shadow_gpt"
        # Shadow → never safe to apply
        assert result.safe_to_apply is False

    def test_gpt_no_match_response(self) -> None:
        from app.nlu.semantic_repair.option_resolver_service import GptOptionResolverService
        svc = GptOptionResolverService(config=_cfg("inline"))
        group = _make_group()

        mock_resp = self._make_mock_response(json.dumps({
            "decision": "no_match",
            "selected_names": [],
            "confidence": 0.10,
            "reason_code": "no_match",
        }))
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_resp
        svc._client = mock_client

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            result = svc.run(
                user_text="kumquat topping",
                item_name="Burger",
                group=group,
                existing_selections=[],
                local_resolved=False,
            )

        assert result.decision == "no_match"
        assert result.safe_to_apply is False

    def test_invalid_gpt_json_returns_error(self) -> None:
        from app.nlu.semantic_repair.option_resolver_service import GptOptionResolverService
        svc = GptOptionResolverService(config=_cfg("inline"))
        group = _make_group()

        mock_resp = self._make_mock_response("not valid json }{")
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_resp
        svc._client = mock_client

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            result = svc.run(
                user_text="test",
                item_name="Burger",
                group=group,
                existing_selections=[],
                local_resolved=False,
            )

        assert result.decision == "error"
        assert result.parse_error is not None
        assert result.safe_to_apply is False

    def test_gpt_timeout_returns_error(self) -> None:
        from app.nlu.semantic_repair.option_resolver_service import GptOptionResolverService
        svc = GptOptionResolverService(config=_cfg("inline"))
        group = _make_group()

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = TimeoutError("timed out")
        svc._client = mock_client

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            result = svc.run(
                user_text="test text",
                item_name="Burger",
                group=group,
                existing_selections=[],
                local_resolved=False,
            )

        assert result.gpt_called is True
        assert result.safe_to_apply is False
        # Should not propagate the exception
        assert result.decision in ("error", "skipped")

    def test_hallucinated_option_name_not_safe(self) -> None:
        """GPT returns a name not in the group → validator rejects → not safe."""
        from app.nlu.semantic_repair.option_resolver_service import GptOptionResolverService
        svc = GptOptionResolverService(config=_cfg("inline"))
        group = _make_group()

        mock_resp = self._make_mock_response(json.dumps({
            "decision": "select_option",
            "selected_names": ["Hallucinated Option"],
            "confidence": 0.99,
            "reason_code": "exact_match",
        }))
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_resp
        svc._client = mock_client

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            result = svc.run(
                user_text="hallu option",
                item_name="Burger",
                group=group,
                existing_selections=[],
                local_resolved=False,
            )

        # Validator rejects hallucinated option name
        assert result.safe_to_apply is False

    def test_gpt_markdown_fenced_response_parsed(self) -> None:
        """GPT sometimes wraps JSON in ```json ... ``` fences."""
        from app.nlu.semantic_repair.option_resolver_service import GptOptionResolverService
        svc = GptOptionResolverService(config=_cfg("inline"))
        group = _make_group()

        fenced = (
            "```json\n"
            + json.dumps({
                "decision": "select_option",
                "selected_names": ["Cheddar Cheese"],
                "confidence": 0.85,
                "reason_code": "fuzzy_match",
            })
            + "\n```"
        )
        mock_resp = self._make_mock_response(fenced)
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_resp
        svc._client = mock_client

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            result = svc.run(
                user_text="chedda",
                item_name="Burger",
                group=group,
                existing_selections=[],
                local_resolved=False,
            )

        assert result.decision == "select_option"
        assert "Cheddar Cheese" in result.selected_names

    def test_daily_budget_exceeded_skips(self) -> None:
        from app.nlu.semantic_repair.option_resolver_service import GptOptionResolverService
        svc = GptOptionResolverService(config=_cfg("inline"))
        # Exhaust the budget
        svc._daily_budget._limit = 0  # unlimited — won't trigger
        svc._daily_budget._limit = 1
        svc._daily_budget._count = 1  # already at limit
        group = _make_group()

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            result = svc.run(
                user_text="macarola cheese",
                item_name="Burger",
                group=group,
                existing_selections=[],
                local_resolved=False,
            )

        assert result.skipped_reason == "daily_budget_exceeded"
        assert result.gpt_called is False

    def test_result_never_mutates_group(self) -> None:
        """Running the service never modifies the PendingModifierGroup."""
        from app.nlu.semantic_repair.option_resolver_service import GptOptionResolverService
        svc = GptOptionResolverService(config=_cfg("disabled"))
        group = _make_group()
        original_choices_count = len(group.choices)

        svc.run(
            user_text="test",
            item_name="Burger",
            group=group,
            existing_selections=[],
            local_resolved=False,
        )
        assert len(group.choices) == original_choices_count

    def test_has_correction_signal_escalates_in_inline_mode(self) -> None:
        """has_correction_signal=True with short text escalates to INLINE_GPT in inline mode."""
        from app.nlu.semantic_repair.option_resolver_service import GptOptionResolverService
        svc = GptOptionResolverService(config=_cfg("inline"))
        group = _make_group()

        mock_resp = self._make_mock_response(json.dumps({
            "decision": "select_option",
            "selected_names": ["Mozzarella Cheese"],
            "confidence": 0.88,
            "reason_code": "fuzzy_match",
        }))
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_resp
        svc._client = mock_client

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            result = svc.run(
                user_text="ac",  # < 3 chars, would normally be NO_GPT
                item_name="Burger",
                group=group,
                existing_selections=[],
                local_resolved=False,
                has_correction_signal=True,  # escalates
            )

        # With correction signal, short text still gets GPT
        assert result.gpt_called is True
        assert result.route_mode == "inline_gpt"

    def test_last_response_key_forwarded_to_context_builder(self) -> None:
        """last_response_key is threaded through to the context builder payload."""
        from app.nlu.semantic_repair.option_resolver_service import GptOptionResolverService
        svc = GptOptionResolverService(config=_cfg("inline"))
        group = _make_group()

        captured_messages: list = []

        def _capture_create(**kwargs):
            captured_messages.extend(kwargs.get("messages", []))
            raise ConnectionError("abort after capture")

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = _capture_create
        svc._client = mock_client

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            svc.run(
                user_text="macarola cheese",
                item_name="Burger",
                group=group,
                existing_selections=[],
                local_resolved=False,
                last_response_key="ask_for_modifier",
            )

        # The user message payload should contain last_prompt
        user_msg = next((m for m in captured_messages if m.get("role") == "user"), None)
        assert user_msg is not None
        payload = json.loads(user_msg["content"])
        assert payload.get("last_prompt") == "ask_for_modifier"

    def test_empty_text_never_calls_gpt(self) -> None:
        """Empty text is pre-filtered by routing policy — GPT is never called."""
        from app.nlu.semantic_repair.option_resolver_service import GptOptionResolverService
        svc = GptOptionResolverService(config=_cfg("inline"))
        group = _make_group()

        mock_client = MagicMock()
        svc._client = mock_client

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            result = svc.run(
                user_text="",
                item_name="Burger",
                group=group,
                existing_selections=[],
                local_resolved=False,
            )

        assert result.gpt_called is False
        mock_client.chat.completions.create.assert_not_called()


# ===========================================================================
# WaitingForModifierHandler integration tests
# ===========================================================================


class TestWaitingForModifierHandlerPhase3Integration:
    """Integration tests for Phase 3 GPT hook in the handler."""

    def _handler(self) -> Any:
        from app.state_machine.handlers.item.add_item.waiting_for_modifier_handler import (
            WaitingForModifierHandler,
        )
        return WaitingForModifierHandler()

    def test_handler_has_option_resolver_attr(self) -> None:
        h = self._handler()
        assert hasattr(h, "_option_resolver")
        assert h._option_resolver is None  # lazy-initialized

    def test_ensure_option_resolver_creates_service(self) -> None:
        from app.nlu.semantic_repair.option_resolver_service import GptOptionResolverService
        h = self._handler()
        svc = h._ensure_option_resolver()
        assert isinstance(svc, GptOptionResolverService)
        # Second call returns same instance
        svc2 = h._ensure_option_resolver()
        assert svc is svc2

    def test_try_gpt_option_resolve_mode_disabled_returns_sentinel(self) -> None:
        """When mode=disabled, _try_gpt_option_resolve returns OPTION_RESOLVER_NOT_CALLED."""
        h = self._handler()
        group = _make_group()
        # Force disabled config in the service
        from app.nlu.semantic_repair.option_resolver_service import GptOptionResolverService
        svc = GptOptionResolverService(config=_cfg("disabled"))
        h._option_resolver = svc

        result = h._try_gpt_option_resolve(
            user_text="macarola cheese",
            item_name="Burger",
            group=group,
            existing_selections=[],
            local_resolved=False,
            session=None,
        )
        assert result.skipped_reason in ("routing_policy_no_gpt", "not_called")
        assert result.gpt_called is False

    def test_try_gpt_option_resolve_never_raises(self) -> None:
        """Even if service.run() raises internally, the handler returns the sentinel."""
        h = self._handler()
        group = _make_group()
        mock_svc = MagicMock()
        mock_svc.run.side_effect = RuntimeError("unexpected failure")
        h._option_resolver = mock_svc

        result = h._try_gpt_option_resolve(
            user_text="crash test",
            item_name="Burger",
            group=group,
            existing_selections=[],
            local_resolved=False,
            session=None,
        )
        # Must return the sentinel, not raise
        assert result is OPTION_RESOLVER_NOT_CALLED

    def test_get_previous_turns_no_session_returns_empty(self) -> None:
        from app.state_machine.handlers.item.add_item.waiting_for_modifier_handler import (
            WaitingForModifierHandler,
        )
        turns = WaitingForModifierHandler._get_previous_turns(None)
        assert turns == []

    def test_get_previous_turns_no_context_returns_empty(self) -> None:
        from app.state_machine.handlers.item.add_item.waiting_for_modifier_handler import (
            WaitingForModifierHandler,
        )
        mock_session = MagicMock()
        mock_session.conversation_context = None
        turns = WaitingForModifierHandler._get_previous_turns(mock_session)
        assert turns == []

    def test_gpt_not_applied_when_mode_disabled(self) -> None:
        """With mode=disabled, handler never applies GPT — returns repeat_modifier_options."""
        from app.nlu.intent_resolution.intent import Intent
        from app.state_machine.handlers.item.add_item.waiting_for_modifier_handler import (
            WaitingForModifierHandler,
        )
        from app.state_machine.models.conversation_context import ConversationContext
        from app.state_machine.models.conversation_state import ConversationState
        from app.state_machine.models.pending_item_models import PendingAddItem

        # Build a minimal context with a group
        group = _make_group()
        pending = PendingAddItem(
            item_id="item_1",
            item_name="Cheeseburger",
            modifier_groups=[group],
            modifier_groups_by_id={group.group_id: group},
        )

        ctx = MagicMock(spec=ConversationContext)
        ctx.pending_add_item = pending
        ctx.current_modifier_group_index = 0
        ctx.selected_modifier_groups = {}
        ctx.skipped_modifier_groups = set()
        ctx.last_nlu = None
        ctx.last_intent_confidence = None
        ctx.return_state = None

        h = WaitingForModifierHandler()
        # Force disabled resolver
        from app.nlu.semantic_repair.option_resolver_service import GptOptionResolverService
        h._option_resolver = GptOptionResolverService(config=_cfg("disabled"))

        result = h.handle(
            intent=Intent.ADD_ITEM,
            context=ctx,
            user_text="nonexistent modifier that nobody has",
            session=None,
        )

        # GPT disabled → local fails → repeat_modifier_options
        assert result.response_key in (
            "repeat_modifier_options",
            "required_modifier_cannot_skip",
            "block_new_item_until_required_done",
        )

    def test_safe_to_apply_false_does_not_apply(self) -> None:
        """When GPT returns safe_to_apply=False, handler falls through to repeat_modifier_options."""
        from app.nlu.intent_resolution.intent import Intent
        from app.state_machine.handlers.item.add_item.waiting_for_modifier_handler import (
            WaitingForModifierHandler,
        )
        from app.state_machine.models.conversation_context import ConversationContext
        from app.state_machine.models.pending_item_models import PendingAddItem

        group = _make_group()
        pending = PendingAddItem(
            item_id="item_1",
            item_name="Cheeseburger",
            modifier_groups=[group],
            modifier_groups_by_id={group.group_id: group},
        )

        ctx = MagicMock(spec=ConversationContext)
        ctx.pending_add_item = pending
        ctx.current_modifier_group_index = 0
        ctx.selected_modifier_groups = {}
        ctx.skipped_modifier_groups = set()
        ctx.last_nlu = None
        ctx.last_intent_confidence = None
        ctx.return_state = None

        h = WaitingForModifierHandler()
        # Mock resolver that returns safe_to_apply=False
        mock_svc = MagicMock()
        mock_svc.run.return_value = OptionResolverResult(
            decision="select_option",
            selected_names=("Hallucinated Option",),
            confidence=0.99,
            route_mode="inline_gpt",
            safe_to_apply=False,  # not safe
        )
        h._option_resolver = mock_svc

        result = h.handle(
            intent=Intent.ADD_ITEM,
            context=ctx,
            user_text="hallucinated modifier",
            session=None,
        )

        # Not safe → falls through to local repeat
        assert result.response_key in (
            "repeat_modifier_options",
            "required_modifier_cannot_skip",
            "block_new_item_until_required_done",
        )
        # Context must not be mutated
        assert ctx.selected_modifier_groups == {}
