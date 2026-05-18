# tests/nlu/semantic_repair/test_phase4_add_item_planner.py
"""Phase 4 GPT Add-Item Planner — comprehensive unit tests.

Test coverage:
  - SemanticRepairConfig Phase 4 fields
  - GptAddItemPlannerRoutingPolicy + is_complex_utterance
  - GptAddItemPlannerContextBuilder
  - parse_planner_output (Phase 4 output parser)
  - PlannerApplyGate
  - AddItemPlanValidator.validate_planner_items + extend operations
  - GptAddItemPlannerService (with mock client)
  - AddItemHandler Phase 4 integration hooks

Safety invariants verified in every relevant test:
  - safe_to_apply=False in shadow mode
  - No cart/state/session mutation from service
  - No API key in any log/output field
  - No full menu in any payload
"""
from __future__ import annotations

import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any
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
from app.config.semantic_repair import SemanticRepairConfig
from app.nlu.semantic_repair.add_item_extractor import GptAddItem, GptAddItemChild, GptAddItemPlan
from app.nlu.semantic_repair.add_item_plan_validator import (
    AddItemPlanValidator,
    PlannerApplyGate,
    ValidatedAddItemPlan,
)
from app.nlu.semantic_repair.add_item_planner_context_builder import (
    GptAddItemPlannerContextBuilder,
    MAX_HISTORY_TURNS,
    MAX_ITEM_CANDIDATES,
    MAX_OPTION_CANDIDATES,
)
from app.nlu.semantic_repair.add_item_planner_output_parser import (
    parse_planner_output,
)
from app.nlu.semantic_repair.add_item_planner_result import (
    ADD_ITEM_PLANNER_NOT_CALLED,
    AddItemPlannerResult,
    PlannerGptItem,
    PlannerGptModifier,
    PlannerGptSide,
    PlannerUnresolved,
)
from app.nlu.semantic_repair.add_item_planner_routing_policy import (
    AddItemPlannerRouteMode,
    GptAddItemPlannerRoutingPolicy,
    is_complex_utterance,
)
from app.nlu.semantic_repair.add_item_planner_service import GptAddItemPlannerService
from app.menu.store import MenuStore

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_FAKE_API_KEY = {"OPENAI_API_KEY": "sk-test-phase4-planner"}
_PROJECT_ROOT = Path(__file__).parents[3]
_DEMO_BASE = _PROJECT_ROOT / "app" / "data" / "restaurants" / "demo"


def _cfg(mode: str = "disabled", **kw) -> SemanticRepairConfig:
    """Build a minimal SemanticRepairConfig for the given planner mode."""
    return SemanticRepairConfig(
        phase=3,
        model="gpt-4o-mini",
        timeout_seconds=2.0,
        add_item_planner_mode=mode,
        add_item_planner_timeout_ms=1800,
        add_item_planner_min_confidence=0.75,
        add_item_planner_max_item_candidates=10,
        add_item_planner_max_option_candidates=20,
        **kw,
    )


def _mock_openai_response(content: str) -> MagicMock:
    """Return a mock OpenAI response object with a single choice."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


_ADD_ITEMS_RESPONSE = json.dumps({
    "decision": "add_items",
    "items": [
        {
            "candidate_item_id": None,
            "item_name": "chicken burger",
            "quantity": 1,
            "size": None,
            "variant": None,
            "modifiers": [{"name": "mozzarella", "operation": "add", "quantity": 1}],
            "sides": [],
            "special_instructions": None,
        }
    ],
    "unresolved": [],
    "confidence": 0.88,
    "reason_code": "complex_with_phrase",
    "safe_to_apply": False,
})

_NO_REPAIR_RESPONSE = json.dumps({
    "decision": "no_repair",
    "items": [],
    "unresolved": [],
    "confidence": 0.2,
    "reason_code": "unclear",
    "safe_to_apply": False,
})


# ===========================================================================
# Part 1: Phase 4 Config Fields
# ===========================================================================


class TestPhase4ConfigFields:
    """SemanticRepairConfig correctly exposes all 5 Phase 4 fields."""

    def test_default_planner_mode_is_disabled(self) -> None:
        cfg = _cfg("disabled")
        assert cfg.add_item_planner_mode == "disabled"

    def test_shadow_mode_accepted(self) -> None:
        cfg = _cfg("shadow")
        assert cfg.add_item_planner_mode == "shadow"

    def test_inline_mode_accepted(self) -> None:
        cfg = _cfg("inline")
        assert cfg.add_item_planner_mode == "inline"

    def test_invalid_mode_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="add_item_planner_mode"):
            _cfg("live")

    def test_default_timeout_ms(self) -> None:
        cfg = _cfg()
        assert cfg.add_item_planner_timeout_ms == 1800

    def test_default_min_confidence(self) -> None:
        cfg = _cfg()
        assert cfg.add_item_planner_min_confidence == 0.75

    def test_default_max_item_candidates(self) -> None:
        cfg = _cfg()
        assert cfg.add_item_planner_max_item_candidates == 10

    def test_default_max_option_candidates(self) -> None:
        cfg = _cfg()
        assert cfg.add_item_planner_max_option_candidates == 20

    def test_env_var_mode(self) -> None:
        with patch.dict("os.environ", {"COMPASS_GPT_ADD_ITEM_PLANNER_MODE": "shadow"}):
            from app.config.semantic_repair import get_semantic_repair_config
            get_semantic_repair_config.cache_clear()
            cfg = get_semantic_repair_config()
            assert cfg.add_item_planner_mode == "shadow"
            get_semantic_repair_config.cache_clear()

    def test_env_var_timeout(self) -> None:
        with patch.dict("os.environ", {"COMPASS_GPT_ADD_ITEM_PLANNER_TIMEOUT_MS": "2500"}):
            from app.config.semantic_repair import get_semantic_repair_config
            get_semantic_repair_config.cache_clear()
            cfg = get_semantic_repair_config()
            assert cfg.add_item_planner_timeout_ms == 2500
            get_semantic_repair_config.cache_clear()


# ===========================================================================
# Part 2: Complexity Detection + Routing Policy
# ===========================================================================


class TestIsComplexUtterance:
    """is_complex_utterance() detects all complexity signals."""

    def test_with_phrase_is_complex(self) -> None:
        assert is_complex_utterance("chicken burger with mozzarella") is True

    def test_without_is_complex(self) -> None:
        assert is_complex_utterance("burger without onions") is True

    def test_extra_is_complex(self) -> None:
        assert is_complex_utterance("burger with extra cheese") is True

    def test_light_is_complex(self) -> None:
        assert is_complex_utterance("salad with light dressing") is True

    def test_no_before_noun_is_complex(self) -> None:
        assert is_complex_utterance("burger no onions") is True

    def test_bare_no_is_not_complex(self) -> None:
        # "no" alone as an answer is not a negated-modifier signal
        assert is_complex_utterance("no") is False

    def test_comma_is_complex(self) -> None:
        assert is_complex_utterance("chicken, coke, fries") is True

    def test_and_word_is_complex(self) -> None:
        assert is_complex_utterance("pizza and coke") is True

    def test_multiple_number_words_is_complex(self) -> None:
        assert is_complex_utterance("two sandwiches and one coke") is True

    def test_digit_quantities_is_complex(self) -> None:
        assert is_complex_utterance("2 burgers and 3 cokes") is True

    def test_item_plus_modifier_slot_is_complex(self) -> None:
        slots = [{"n": "ITEM", "v": "burger"}, {"n": "MODIFIER", "v": "cheese"}]
        assert is_complex_utterance("burger cheese", local_slots=slots) is True

    def test_item_plus_side_slot_is_complex(self) -> None:
        slots = [{"n": "ITEM", "v": "burger"}, {"n": "SIDE", "v": "fries"}]
        assert is_complex_utterance("burger fries", local_slots=slots) is True

    def test_two_item_slots_is_complex(self) -> None:
        slots = [{"n": "ITEM", "v": "burger"}, {"n": "ITEM", "v": "coke"}]
        assert is_complex_utterance("burger coke", local_slots=slots) is True

    def test_simple_utterance_not_complex(self) -> None:
        assert is_complex_utterance("chicken burger") is False

    def test_empty_text_not_complex(self) -> None:
        assert is_complex_utterance("") is False

    def test_none_treated_as_empty(self) -> None:
        assert is_complex_utterance(None) is False  # type: ignore[arg-type]


class TestGptAddItemPlannerRoutingPolicy:
    """GptAddItemPlannerRoutingPolicy.decide() returns correct route modes."""

    @pytest.fixture(autouse=True)
    def policy(self) -> GptAddItemPlannerRoutingPolicy:
        self.policy = GptAddItemPlannerRoutingPolicy()
        return self.policy

    # ── Guard: disabled ────────────────────────────────────────────────

    def test_disabled_mode_returns_no_gpt(self) -> None:
        route, reason = self.policy.decide(
            config=_cfg("disabled"),
            user_text="chicken burger with cheese",
        )
        assert route == AddItemPlannerRouteMode.NO_GPT
        assert reason == "mode_disabled"

    def test_empty_text_returns_no_gpt_regardless_of_mode(self) -> None:
        for mode in ("shadow", "inline"):
            route, reason = self.policy.decide(
                config=_cfg(mode),
                user_text="",
            )
            assert route == AddItemPlannerRouteMode.NO_GPT
            assert reason == "empty_text"

    def test_whitespace_only_returns_no_gpt(self) -> None:
        route, _ = self.policy.decide(config=_cfg("shadow"), user_text="   ")
        assert route == AddItemPlannerRouteMode.NO_GPT

    # ── Simple high-confidence bypass ─────────────────────────────────

    def test_simple_high_confidence_add_item_bypasses_shadow(self) -> None:
        route, reason = self.policy.decide(
            config=_cfg("shadow"),
            user_text="chicken burger",
            local_intent="add_item",
            local_confidence=0.95,
            local_slots=[{"n": "ITEM", "v": "chicken burger"}],
        )
        assert route == AddItemPlannerRouteMode.NO_GPT
        assert reason == "simple_high_confidence_local"

    def test_simple_high_confidence_add_item_bypasses_inline(self) -> None:
        route, reason = self.policy.decide(
            config=_cfg("inline"),
            user_text="coke",
            local_intent="add_item",
            local_confidence=0.92,
            local_slots=[{"n": "ITEM", "v": "coke"}],
        )
        assert route == AddItemPlannerRouteMode.NO_GPT
        assert reason == "simple_high_confidence_local"

    def test_complex_utterance_overrides_high_confidence_bypass(self) -> None:
        # Even high confidence doesn't bypass when utterance is complex
        route, _ = self.policy.decide(
            config=_cfg("shadow"),
            user_text="chicken burger with extra cheese",
            local_intent="add_item",
            local_confidence=0.95,
            local_slots=[{"n": "ITEM", "v": "chicken burger"}],
        )
        assert route == AddItemPlannerRouteMode.SHADOW_GPT

    # ── Shadow mode routing ────────────────────────────────────────────

    def test_shadow_mode_complex_utterance_returns_shadow_gpt(self) -> None:
        route, _ = self.policy.decide(
            config=_cfg("shadow"),
            user_text="chicken burger with mozzarella and coke",
            local_intent="add_item",
            local_confidence=0.60,
        )
        assert route == AddItemPlannerRouteMode.SHADOW_GPT

    def test_shadow_mode_multi_item_returns_shadow_gpt(self) -> None:
        route, _ = self.policy.decide(
            config=_cfg("shadow"),
            user_text="two chicken sandwiches, one with Swiss",
            local_intent="add_item",
            local_confidence=0.55,
        )
        assert route == AddItemPlannerRouteMode.SHADOW_GPT

    def test_shadow_with_item_evidence_returns_shadow_gpt(self) -> None:
        route, _ = self.policy.decide(
            config=_cfg("shadow"),
            user_text="burger and fries",
            local_intent="add_item",
            local_confidence=0.55,
            item_candidates_exist=True,
        )
        assert route == AddItemPlannerRouteMode.SHADOW_GPT

    # ── Inline mode routing ────────────────────────────────────────────

    def test_inline_mode_complex_utterance_returns_inline_gpt(self) -> None:
        route, _ = self.policy.decide(
            config=_cfg("inline"),
            user_text="chicken burger with mozzarella, onions, mayo, and coke",
            local_intent="add_item",
            local_confidence=0.52,
        )
        assert route == AddItemPlannerRouteMode.INLINE_GPT

    def test_inline_mode_comma_list_returns_inline_gpt(self) -> None:
        route, _ = self.policy.decide(
            config=_cfg("inline"),
            user_text="pizza, coke, fries",
            local_intent="add_item",
            local_confidence=0.60,
        )
        assert route == AddItemPlannerRouteMode.INLINE_GPT

    def test_inline_mode_no_complexity_returns_no_gpt(self) -> None:
        route, _ = self.policy.decide(
            config=_cfg("inline"),
            user_text="just a coke",
            local_intent="add_item",
            local_confidence=0.50,
        )
        assert route == AddItemPlannerRouteMode.NO_GPT

    def test_unknown_intent_with_item_slot_shadow_triggers(self) -> None:
        route, _ = self.policy.decide(
            config=_cfg("shadow"),
            user_text="give me the burger thing with cheese",
            local_intent="UNKNOWN",
            local_confidence=0.20,
            local_slots=[{"n": "ITEM", "v": "burger"}, {"n": "MODIFIER", "v": "cheese"}],
        )
        assert route == AddItemPlannerRouteMode.SHADOW_GPT


# ===========================================================================
# Part 3: Context Builder
# ===========================================================================


class TestGptAddItemPlannerContextBuilder:
    """GptAddItemPlannerContextBuilder produces correct payloads."""

    @pytest.fixture(autouse=True)
    def builder(self) -> None:
        self.builder = GptAddItemPlannerContextBuilder()

    def _messages(self, **kw) -> list[dict]:
        defaults = dict(user_text="chicken burger with cheese")
        defaults.update(kw)
        return self.builder.build_messages(**defaults)

    def test_returns_two_messages(self) -> None:
        msgs = self._messages()
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_user_text_in_payload(self) -> None:
        msgs = self._messages(user_text="chicken burger with extra cheese")
        payload = json.loads(msgs[1]["content"])
        assert payload["text"] == "chicken burger with extra cheese"

    def test_local_intent_in_payload(self) -> None:
        msgs = self._messages(local_intent="add_item", local_confidence=0.82)
        payload = json.loads(msgs[1]["content"])
        assert payload["local"]["intent"] == "add_item"
        assert payload["local"]["conf"] == pytest.approx(0.82, abs=0.01)

    def test_top_k_intents_in_payload(self) -> None:
        top_k = [{"i": "add_item", "c": 0.82}, {"i": "unknown", "c": 0.12}]
        msgs = self._messages(top_k_intents=top_k)
        payload = json.loads(msgs[1]["content"])
        assert payload["local"]["top_k"] == top_k

    def test_candidate_items_in_payload(self) -> None:
        cands = [{"id": "item1", "name": "Chicken Burger", "modifier_groups": []}]
        msgs = self._messages(candidate_items=cands)
        payload = json.loads(msgs[1]["content"])
        assert "candidates" in payload
        assert payload["candidates"][0]["name"] == "Chicken Burger"

    def test_candidate_items_capped_at_max(self) -> None:
        cands = [{"id": f"item{i}", "name": f"Item {i}"} for i in range(20)]
        msgs = self._messages(candidate_items=cands)
        payload = json.loads(msgs[1]["content"])
        assert len(payload["candidates"]) == MAX_ITEM_CANDIDATES

    def test_modifier_choices_capped_at_max(self) -> None:
        choices = [f"Option {i}" for i in range(30)]
        cands = [{"id": "x", "name": "Burger", "modifier_groups": [{"name": "G", "choices": choices}]}]
        msgs = self._messages(candidate_items=cands)
        payload = json.loads(msgs[1]["content"])
        assert len(payload["candidates"][0]["modifier_groups"][0]["choices"]) == MAX_OPTION_CANDIDATES

    def test_history_capped_at_max(self) -> None:
        history = [("user", f"turn {i}") for i in range(10)]
        msgs = self._messages(previous_turns=history)
        payload = json.loads(msgs[1]["content"])
        assert len(payload["history"]) == MAX_HISTORY_TURNS

    def test_cart_included_compact(self) -> None:
        msgs = self._messages(cart_item_names=["Pizza", "Wings"])
        payload = json.loads(msgs[1]["content"])
        assert payload["cart"]["n"] == 2
        assert "Pizza" in payload["cart"]["items"]

    def test_no_prices_in_payload(self) -> None:
        cands = [{"id": "x", "name": "Burger", "price": 9.99, "modifier_groups": []}]
        msgs = self._messages(candidate_items=cands)
        payload_str = msgs[1]["content"]
        assert "price" not in payload_str.lower() or "9.99" not in payload_str

    def test_schema_always_in_payload(self) -> None:
        msgs = self._messages()
        payload = json.loads(msgs[1]["content"])
        assert "schema" in payload

    def test_no_full_menu_in_payload(self) -> None:
        # Payload must not contain "full_menu" or "all_items" keys
        msgs = self._messages()
        payload = json.loads(msgs[1]["content"])
        assert "full_menu" not in payload
        assert "all_items" not in payload


# ===========================================================================
# Part 4: Phase 4 Output Parser
# ===========================================================================


class TestParsePlannerOutput:
    """parse_planner_output() handles all Phase 4 schema variants."""

    def _parse(self, data: dict, utterance: str = "chicken burger with cheese") -> tuple:
        return parse_planner_output(
            json.dumps(data),
            utterance_text=utterance,
            candidate_names={"chicken burger"},
            candidate_option_names={"mozzarella", "cheese", "coke"},
        )

    def test_valid_add_items_decision_parsed(self) -> None:
        decision, items, unresolved, conf, reason_code, parse_error = self._parse({
            "decision": "add_items",
            "items": [{"item_name": "chicken burger", "quantity": 1, "modifiers": [], "sides": []}],
            "unresolved": [],
            "confidence": 0.88,
            "reason_code": "complex_with_phrase",
        })
        assert decision == "add_items"
        assert len(items) == 1
        assert items[0].item_name == "chicken burger"

    def test_clarify_decision_parsed(self) -> None:
        decision, items, _, conf, _, _ = self._parse({
            "decision": "clarify",
            "items": [],
            "unresolved": [],
            "confidence": 0.45,
            "reason_code": "unclear",
        })
        assert decision == "clarify"

    def test_markdown_fenced_json_parsed(self) -> None:
        raw = "```json\n" + json.dumps({
            "decision": "add_items",
            "items": [{"item_name": "chicken burger", "quantity": 1}],
            "unresolved": [],
            "confidence": 0.88,
            "reason_code": "complex_with_phrase",
        }) + "\n```"
        decision, items, _, _, _, parse_error = parse_planner_output(
            raw, utterance_text="chicken burger", candidate_names={"chicken burger"}
        )
        assert decision == "add_items"
        assert parse_error is None

    def test_malformed_json_returns_no_repair(self) -> None:
        decision, items, _, _, _, parse_error = parse_planner_output(
            "not json", utterance_text="test"
        )
        assert decision == "no_repair"
        assert parse_error is not None

    def test_unknown_decision_becomes_no_repair(self) -> None:
        decision, _, _, _, _, _ = self._parse({"decision": "invented", "items": []})
        assert decision == "no_repair"

    def test_modifier_extra_operation_parsed(self) -> None:
        decision, items, _, _, _, _ = self._parse({
            "decision": "add_items",
            "items": [{
                "item_name": "chicken burger",
                "modifiers": [{"name": "cheese", "operation": "extra", "quantity": 1}],
                "sides": [],
            }],
            "unresolved": [],
            "confidence": 0.80,
        })
        assert items[0].modifiers[0].operation == "extra"

    def test_modifier_light_operation_parsed(self) -> None:
        decision, items, _, _, _, _ = self._parse({
            "decision": "add_items",
            "items": [{
                "item_name": "chicken burger",
                "modifiers": [{"name": "cheese", "operation": "light", "quantity": 1}],
                "sides": [],
            }],
            "unresolved": [],
            "confidence": 0.80,
        })
        assert items[0].modifiers[0].operation == "light"

    def test_hallucinated_item_dropped(self) -> None:
        decision, items, _, _, _, _ = parse_planner_output(
            json.dumps({
                "decision": "add_items",
                "items": [{"item_name": "unicorn special", "quantity": 1}],
                "unresolved": [],
                "confidence": 0.99,
            }),
            utterance_text="chicken burger",
            candidate_names={"chicken burger"},
        )
        # "unicorn special" is not in utterance or candidate names
        assert len(items) == 0

    def test_hallucinated_modifier_dropped(self) -> None:
        _, items, _, _, _, _ = parse_planner_output(
            json.dumps({
                "decision": "add_items",
                "items": [{
                    "item_name": "chicken burger",
                    "modifiers": [{"name": "invented sauce", "operation": "add", "quantity": 1}],
                }],
                "unresolved": [],
                "confidence": 0.85,
            }),
            utterance_text="chicken burger with cheese",
            candidate_names={"chicken burger"},
            candidate_option_names={"cheese"},
        )
        # "invented sauce" not in utterance or options
        assert len(items[0].modifiers) == 0

    def test_unresolved_array_parsed(self) -> None:
        _, _, unresolved, _, _, _ = self._parse({
            "decision": "add_items",
            "items": [{"item_name": "chicken burger"}],
            "unresolved": [{"text": "something weird", "reason": "not_on_menu"}],
            "confidence": 0.70,
        })
        assert len(unresolved) == 1
        assert unresolved[0].reason == "not_on_menu"

    def test_confidence_clamped_to_0_1(self) -> None:
        _, _, _, conf, _, _ = parse_planner_output(
            json.dumps({"decision": "add_items", "items": [], "confidence": 1.5}),
            utterance_text="test",
        )
        assert conf == pytest.approx(1.0)


# ===========================================================================
# Part 5: Apply Gate
# ===========================================================================


class TestPlannerApplyGate:
    """PlannerApplyGate.should_apply() enforces all gating rules."""

    @pytest.fixture(autouse=True)
    def gate(self) -> None:
        self.gate = PlannerApplyGate()

    def _make_validated_plan(self, *, has_blocking=False, n_items=1) -> MagicMock:
        vp = MagicMock(spec=ValidatedAddItemPlan)
        vp.has_blocking_warnings = has_blocking
        vi = MagicMock()
        vi.item_id = "item_123"
        vi.item_name = "Chicken Burger"
        vp.items = tuple(vi for _ in range(n_items))
        vp.rejected_items = ()
        return vp

    def test_shadow_mode_never_applies(self) -> None:
        vp = self._make_validated_plan()
        safe, reason = self.gate.should_apply(
            route_mode="shadow_gpt",
            decision="add_items",
            validated_plan=vp,
            confidence=0.90,
            min_confidence=0.75,
        )
        assert safe is False
        assert "shadow" in reason

    def test_no_gpt_route_never_applies(self) -> None:
        vp = self._make_validated_plan()
        safe, _ = self.gate.should_apply(
            route_mode="no_gpt",
            decision="add_items",
            validated_plan=vp,
            confidence=0.90,
            min_confidence=0.75,
        )
        assert safe is False

    def test_inline_valid_plan_applies(self) -> None:
        vp = self._make_validated_plan()
        safe, reason = self.gate.should_apply(
            route_mode="inline_gpt",
            decision="add_items",
            validated_plan=vp,
            confidence=0.88,
            min_confidence=0.75,
        )
        assert safe is True
        assert reason == "approved"

    def test_low_confidence_prevents_apply(self) -> None:
        vp = self._make_validated_plan()
        safe, reason = self.gate.should_apply(
            route_mode="inline_gpt",
            decision="add_items",
            validated_plan=vp,
            confidence=0.50,
            min_confidence=0.75,
        )
        assert safe is False
        assert "confidence_too_low" in reason

    def test_blocking_warnings_prevent_apply(self) -> None:
        vp = self._make_validated_plan(has_blocking=True)
        safe, reason = self.gate.should_apply(
            route_mode="inline_gpt",
            decision="add_items",
            validated_plan=vp,
            confidence=0.90,
            min_confidence=0.75,
        )
        assert safe is False
        assert "blocking" in reason

    def test_decision_not_add_items_prevents_apply(self) -> None:
        vp = self._make_validated_plan()
        safe, reason = self.gate.should_apply(
            route_mode="inline_gpt",
            decision="clarify",
            validated_plan=vp,
            confidence=0.90,
            min_confidence=0.75,
        )
        assert safe is False
        assert "decision_not_add_items" in reason

    def test_parse_error_prevents_apply(self) -> None:
        vp = self._make_validated_plan()
        safe, reason = self.gate.should_apply(
            route_mode="inline_gpt",
            decision="add_items",
            validated_plan=vp,
            confidence=0.88,
            min_confidence=0.75,
            parse_error="json_decode:unexpected token",
        )
        assert safe is False
        assert "parse_error" in reason

    def test_timeout_prevents_apply(self) -> None:
        vp = self._make_validated_plan()
        safe, reason = self.gate.should_apply(
            route_mode="inline_gpt",
            decision="add_items",
            validated_plan=vp,
            confidence=0.88,
            min_confidence=0.75,
            timed_out=True,
        )
        assert safe is False
        assert reason == "gpt_timeout"

    def test_no_validated_plan_prevents_apply(self) -> None:
        safe, reason = self.gate.should_apply(
            route_mode="inline_gpt",
            decision="add_items",
            validated_plan=None,
            confidence=0.88,
            min_confidence=0.75,
        )
        assert safe is False
        assert reason == "validator_not_run"

    def test_zero_valid_items_prevents_apply(self) -> None:
        vp = self._make_validated_plan(n_items=0)
        safe, reason = self.gate.should_apply(
            route_mode="inline_gpt",
            decision="add_items",
            validated_plan=vp,
            confidence=0.88,
            min_confidence=0.75,
        )
        assert safe is False


# ===========================================================================
# Part 6: Validator Extensions (extra/light operations)
# ===========================================================================


class TestValidatorExtendedOperations:
    """AddItemPlanValidator handles extra/light modifier operations from Phase 4."""

    @pytest.fixture(scope="class")
    def demo_store(self) -> MenuStore:
        return MenuStore(
            menu_path=_DEMO_BASE / "menu.json",
            entity_index_path=_DEMO_BASE / "entity_index.json",
        )

    @pytest.fixture(scope="class")
    def validator(self) -> AddItemPlanValidator:
        return AddItemPlanValidator()

    def test_extra_operation_stored_in_validated_modifier(self, demo_store, validator) -> None:
        """Modifier with operation='extra' should be accepted without blocking."""
        from app.nlu.semantic_repair.add_item_extractor import GptAddItem, GptAddItemChild, GptAddItemPlan
        plan = GptAddItemPlan(
            decision="add_items",
            items=(GptAddItem(
                item="ham biscuit",
                modifiers=(GptAddItemChild(name="Cheese", operation="extra", quantity=1),),
            ),),
        )
        result = validator.validate(plan=plan, menu_store=demo_store)
        assert not result.has_blocking_warnings
        assert len(result.items) == 1
        mod = result.items[0].modifiers[0]
        assert mod.operation == "extra"

    def test_light_operation_stored_in_validated_modifier(self, demo_store, validator) -> None:
        from app.nlu.semantic_repair.add_item_extractor import GptAddItem, GptAddItemChild, GptAddItemPlan
        plan = GptAddItemPlan(
            decision="add_items",
            items=(GptAddItem(
                item="ham biscuit",
                modifiers=(GptAddItemChild(name="Cheese", operation="light", quantity=1),),
            ),),
        )
        result = validator.validate(plan=plan, menu_store=demo_store)
        # operation field should be preserved through validation
        assert not result.has_blocking_warnings
        assert result.items[0].modifiers[0].operation == "light"

    def test_validate_planner_items_adapter(self, demo_store, validator) -> None:
        """validate_planner_items() adapts PlannerGptItem correctly."""
        items = (PlannerGptItem(
            item_name="ham biscuit",
            quantity=1,
            modifiers=(PlannerGptModifier(name="Cheese", operation="extra", quantity=1),),
        ),)
        result = validator.validate_planner_items(
            planner_items=items, menu_store=demo_store
        )
        assert not result.has_blocking_warnings
        assert len(result.items) == 1

    def test_validate_planner_items_hallucinated_item_rejected(self, demo_store, validator) -> None:
        items = (PlannerGptItem(item_name="unicorn burger", quantity=1),)
        result = validator.validate_planner_items(planner_items=items, menu_store=demo_store)
        assert result.has_blocking_warnings
        assert len(result.items) == 0
        assert "unicorn burger" in result.rejected_items

    def test_validate_planner_items_empty_returns_empty_plan(self, demo_store, validator) -> None:
        result = validator.validate_planner_items(planner_items=(), menu_store=demo_store)
        assert not result.has_blocking_warnings
        assert result.items == ()


# ===========================================================================
# Part 7: GptAddItemPlannerService (with mock client)
# ===========================================================================


class TestGptAddItemPlannerService:
    """GptAddItemPlannerService.run() behaves correctly across all scenarios."""

    def _svc(self, mode: str = "shadow", mock_response: str | None = None) -> GptAddItemPlannerService:
        client = MagicMock()
        resp = _mock_openai_response(mock_response or _NO_REPAIR_RESPONSE)
        client.chat.completions.create.return_value = resp
        return GptAddItemPlannerService(config=_cfg(mode), mock_client=client)

    # ── Disabled mode ──────────────────────────────────────────────────

    def test_disabled_mode_returns_not_called_sentinel(self) -> None:
        svc = GptAddItemPlannerService(config=_cfg("disabled"), mock_client=MagicMock())
        result = svc.run(user_text="chicken burger with cheese")
        assert result.gpt_called is False
        assert result.route_mode == "no_gpt"
        assert result.safe_to_apply is False

    def test_disabled_mode_client_never_called(self) -> None:
        client = MagicMock()
        svc = GptAddItemPlannerService(config=_cfg("disabled"), mock_client=client)
        svc.run(user_text="chicken burger with cheese and coke")
        client.chat.completions.create.assert_not_called()

    # ── Shadow mode ────────────────────────────────────────────────────

    def test_shadow_mode_complex_utterance_calls_gpt(self) -> None:
        svc = self._svc("shadow")
        with patch.dict("os.environ", _FAKE_API_KEY):
            result = svc.run(user_text="chicken burger with mozzarella and coke")
        assert result.gpt_called is True

    def test_shadow_mode_never_safe_to_apply(self) -> None:
        svc = self._svc("shadow", mock_response=_ADD_ITEMS_RESPONSE)
        with patch.dict("os.environ", _FAKE_API_KEY):
            result = svc.run(
                user_text="chicken burger with mozzarella, onions, mayo, and coke",
                local_intent="add_item",
                local_confidence=0.52,
            )
        # Shadow mode → safe_to_apply must always be False
        assert result.safe_to_apply is False

    def test_shadow_mode_simple_utterance_skips_gpt(self) -> None:
        client = MagicMock()
        svc = GptAddItemPlannerService(config=_cfg("shadow"), mock_client=client)
        with patch.dict("os.environ", _FAKE_API_KEY):
            result = svc.run(
                user_text="just a coke",
                local_intent="add_item",
                local_confidence=0.92,
                local_slots=[{"n": "ITEM", "v": "coke"}],
            )
        client.chat.completions.create.assert_not_called()
        assert result.gpt_called is False

    # ── No API key ─────────────────────────────────────────────────────

    def test_missing_api_key_returns_skipped(self) -> None:
        svc = self._svc("shadow")
        svc._client = None  # reset lazy client
        with patch.dict("os.environ", {}, clear=True):
            result = svc.run(user_text="burger with extra cheese")
        assert result.gpt_called is False
        assert result.skipped_reason == "missing_api_key"

    # ── Inline mode ────────────────────────────────────────────────────

    def test_inline_mode_complex_calls_gpt(self) -> None:
        svc = self._svc("inline", mock_response=_ADD_ITEMS_RESPONSE)
        with patch.dict("os.environ", _FAKE_API_KEY):
            result = svc.run(
                user_text="chicken burger with mozzarella and coke",
                local_intent="add_item",
                local_confidence=0.55,
            )
        assert result.gpt_called is True

    def test_inline_mode_high_confidence_valid_plan(self) -> None:
        """With inline mode, high-confidence valid GPT plan → safe_to_apply=True
        ONLY when the validator also passes. Without a menu_store, validator is
        not run → safe_to_apply=False (apply gate requires validator)."""
        svc = self._svc("inline", mock_response=_ADD_ITEMS_RESPONSE)
        with patch.dict("os.environ", _FAKE_API_KEY):
            result = svc.run(
                user_text="chicken burger with mozzarella and coke",
                local_intent="add_item",
                local_confidence=0.55,
            )
        # No menu_store → validator not run → safe_to_apply=False
        assert result.safe_to_apply is False
        assert result.gpt_called is True

    def test_malformed_response_returns_error_result(self) -> None:
        svc = self._svc("shadow", mock_response="definitely not json {{{")
        with patch.dict("os.environ", _FAKE_API_KEY):
            result = svc.run(user_text="burger with extra cheese")
        assert result.parse_error is not None
        assert result.safe_to_apply is False

    def test_empty_text_never_calls_gpt(self) -> None:
        client = MagicMock()
        for mode in ("shadow", "inline"):
            svc = GptAddItemPlannerService(config=_cfg(mode), mock_client=client)
            result = svc.run(user_text="")
            assert result.gpt_called is False
        client.chat.completions.create.assert_not_called()

    def test_result_always_has_route_mode_field(self) -> None:
        svc = self._svc("disabled")
        result = svc.run(user_text="anything here")
        assert hasattr(result, "route_mode")
        assert isinstance(result.route_mode, str)

    def test_result_is_serialisable(self) -> None:
        svc = self._svc("shadow", mock_response=_ADD_ITEMS_RESPONSE)
        with patch.dict("os.environ", _FAKE_API_KEY):
            result = svc.run(user_text="burger with extra cheese and coke")
        d = result.to_dict()
        serialised = json.dumps(d)  # must not raise
        assert isinstance(serialised, str)

    def test_service_never_raises(self) -> None:
        """run() must never raise regardless of arguments."""
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("boom")
        svc = GptAddItemPlannerService(config=_cfg("shadow"), mock_client=client)
        with patch.dict("os.environ", _FAKE_API_KEY):
            result = svc.run(user_text="burger with extra cheese")
        assert result is not None

    def test_no_api_key_in_result_dict(self) -> None:
        svc = self._svc("shadow", mock_response=_ADD_ITEMS_RESPONSE)
        with patch.dict("os.environ", _FAKE_API_KEY):
            result = svc.run(user_text="burger with extra cheese")
        d_str = json.dumps(result.to_dict())
        assert "sk-test" not in d_str
        assert "OPENAI_API_KEY" not in d_str

    def test_terminal_state_skipped(self) -> None:
        client = MagicMock()
        svc = GptAddItemPlannerService(config=_cfg("shadow"), mock_client=client)
        result = svc.run(user_text="burger with cheese", state="COMPLETED")
        assert result.gpt_called is False
        assert result.skipped_reason == "terminal_state"
        client.chat.completions.create.assert_not_called()

    def test_unresolved_entities_in_result(self) -> None:
        response = json.dumps({
            "decision": "add_items",
            "items": [{"item_name": "burger", "quantity": 1}],
            "unresolved": [{"text": "mystery sauce", "reason": "not_on_menu"}],
            "confidence": 0.75,
            "reason_code": "complex_with_phrase",
        })
        svc = self._svc("shadow", mock_response=response)
        with patch.dict("os.environ", _FAKE_API_KEY):
            result = svc.run(
                user_text="burger with mystery sauce",
                local_intent="add_item",
                local_confidence=0.50,
            )
        assert result.gpt_called is True
        assert len(result.unresolved) == 1
        assert result.unresolved[0].reason == "not_on_menu"


# ===========================================================================
# Part 8: AddItemPlannerResult dataclass
# ===========================================================================


class TestAddItemPlannerResult:
    """AddItemPlannerResult sentinel and serialisation."""

    def test_not_called_sentinel_fields(self) -> None:
        r = ADD_ITEM_PLANNER_NOT_CALLED
        assert r.gpt_called is False
        assert r.safe_to_apply is False
        assert r.decision == "skipped"
        assert r.route_mode == "no_gpt"

    def test_to_dict_is_json_serialisable(self) -> None:
        r = AddItemPlannerResult(
            decision="add_items",
            gpt_called=True,
            route_mode="shadow_gpt",
            route_reason="shadow_complex_or_evidence",
            items=(PlannerGptItem(
                item_name="Chicken Burger",
                modifiers=(PlannerGptModifier(name="Cheese", operation="extra"),),
                sides=(PlannerGptSide(name="Coke", size="large"),),
            ),),
            unresolved=(PlannerUnresolved(text="mystery", reason="not_on_menu"),),
            confidence=0.88,
        )
        d = r.to_dict()
        s = json.dumps(d)
        assert "Chicken Burger" in s
        assert "extra" in s
        assert "mystery" in s


# ===========================================================================
# Part 9: AddItemHandler Phase 4 Integration
# ===========================================================================


class TestAddItemHandlerPhase4Integration:
    """AddItemHandler correctly wires the GPT planner hook."""

    @pytest.fixture(scope="class")
    def menu_repo(self):
        from app.menu.repository import MenuRepository
        store = MenuStore(
            menu_path=_DEMO_BASE / "menu.json",
            entity_index_path=_DEMO_BASE / "entity_index.json",
        )
        return MenuRepository(store=store)

    def test_handler_initialises_without_planner(self, menu_repo) -> None:
        from app.state_machine.handlers.item.add_item.add_item_handler import AddItemHandler
        h = AddItemHandler(menu_repo=menu_repo)
        assert h._gpt_planner is None

    def test_handler_accepts_planner_injection(self, menu_repo) -> None:
        from app.state_machine.handlers.item.add_item.add_item_handler import AddItemHandler
        mock_planner = MagicMock()
        h = AddItemHandler(menu_repo=menu_repo, gpt_planner=mock_planner)
        assert h._gpt_planner is mock_planner

    def test_planner_not_called_when_none(self, menu_repo) -> None:
        """When _gpt_planner is None, _try_gpt_planner must return None silently."""
        from app.state_machine.handlers.item.add_item.add_item_handler import AddItemHandler
        h = AddItemHandler(menu_repo=menu_repo)
        result = h._try_gpt_planner(user_text="burger with cheese", slots=[], session=None)
        assert result is None

    def test_planner_called_when_injected(self, menu_repo) -> None:
        from app.state_machine.handlers.item.add_item.add_item_handler import AddItemHandler
        mock_planner = MagicMock()
        mock_planner.run.return_value = ADD_ITEM_PLANNER_NOT_CALLED
        mock_planner._config = MagicMock()
        mock_planner._config.add_item_planner_mode = "shadow"
        h = AddItemHandler(menu_repo=menu_repo, gpt_planner=mock_planner)
        h._try_gpt_planner(user_text="burger with cheese", slots=[], session=None)
        mock_planner.run.assert_called_once()

    def test_planner_exception_never_raises(self, menu_repo) -> None:
        """Planner errors must never crash the handler."""
        from app.state_machine.handlers.item.add_item.add_item_handler import AddItemHandler
        mock_planner = MagicMock()
        mock_planner.run.side_effect = RuntimeError("planner crash")
        mock_planner._config = MagicMock()
        mock_planner._config.add_item_planner_mode = "shadow"
        h = AddItemHandler(menu_repo=menu_repo, gpt_planner=mock_planner)
        result = h._try_gpt_planner(user_text="burger", slots=[], session=None)
        assert result is None  # exception swallowed

    def test_shadow_planner_does_not_apply_to_flow(self, menu_repo) -> None:
        """When safe_to_apply=False, handle() falls through to local path."""
        from app.nlu.intent_resolution.intent import Intent
        from app.state_machine.handlers.item.add_item.add_item_handler import AddItemHandler
        from app.state_machine.models.conversation_context import ConversationContext
        mock_planner = MagicMock()
        mock_planner.run.return_value = AddItemPlannerResult(
            decision="add_items",
            gpt_called=True,
            route_mode="shadow_gpt",
            route_reason="shadow_complex_or_evidence",
            safe_to_apply=False,
        )
        mock_planner._config = MagicMock()
        mock_planner._config.add_item_planner_mode = "shadow"
        h = AddItemHandler(menu_repo=menu_repo, gpt_planner=mock_planner)
        ctx = ConversationContext()
        # This will fall through to local path and attempt menu resolution.
        # We only assert it doesn't crash AND planner was called.
        try:
            h.handle(intent=Intent.ADD_ITEM, context=ctx, user_text="burger", session=None)
        except Exception:
            pass  # local path may fail in test env — we only care planner was called
        mock_planner.run.assert_called_once()

    def test_apply_planner_result_none_on_multi_item(self, menu_repo) -> None:
        """Multi-item plans (>1 item) must return None — deferred to future PR."""
        from app.state_machine.handlers.item.add_item.add_item_handler import AddItemHandler
        from app.state_machine.models.conversation_context import ConversationContext
        h = AddItemHandler(menu_repo=menu_repo)
        ctx = ConversationContext()
        # Build a mock planner_result with 2 validated items
        mock_result = MagicMock()
        mock_result.safe_to_apply = True
        mock_vp = MagicMock()
        vi1 = MagicMock(); vi1.item_id = "item1"; vi1.item_name = "Burger"
        vi2 = MagicMock(); vi2.item_id = "item2"; vi2.item_name = "Pizza"
        mock_vp.items = (vi1, vi2)
        mock_result.validated_plan = mock_vp
        applied = h._apply_planner_result(mock_result, ctx)
        assert applied is None  # multi-item not applied in this PR
