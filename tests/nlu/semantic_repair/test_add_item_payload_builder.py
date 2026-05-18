# tests/nlu/semantic_repair/test_add_item_payload_builder.py
"""Tests for AddItemPayloadBuilder and AddItemEligibilityGate."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from app.nlu.semantic_repair.add_item_extractor import (
    AddItemEligibilityGate,
    AddItemPayloadBuilder,
)
from app.state_machine.models.conversation_state import ConversationState


def _mock_config(
    *,
    add_item_mode: str = "shadow",
    add_item_min_text_len: int = 3,
) -> MagicMock:
    cfg = MagicMock()
    cfg.add_item_mode = add_item_mode
    cfg.add_item_min_text_len = add_item_min_text_len
    return cfg


def _mock_intent_result(intent_value: str = "ADD_ITEM") -> MagicMock:
    from app.nlu.intent_resolution.intent import Intent
    ir = MagicMock()
    if intent_value == "ADD_ITEM":
        ir.intent = Intent.ADD_ITEM
    elif intent_value == "CHECKOUT":
        ir.intent = Intent.CHECKOUT
    else:
        ir.intent = Intent.UNKNOWN
    return ir


# ---------------------------------------------------------------------------
# AddItemEligibilityGate
# ---------------------------------------------------------------------------

class TestAddItemEligibilityGate:
    def setup_method(self):
        self.gate = AddItemEligibilityGate()

    def _check(self, *, state=ConversationState.IDLE, intent="ADD_ITEM", text="burger", **kwargs):
        return self.gate.check(
            intent_result=_mock_intent_result(intent),
            state=state,
            normalized_text=text,
            config=_mock_config(**kwargs),
        )

    def test_disabled_mode_returns_not_eligible(self):
        eligible, reason = self._check(add_item_mode="disabled")
        assert not eligible
        assert reason == "mode_disabled"

    def test_terminal_state_not_eligible(self):
        eligible, reason = self._check(state=ConversationState.COMPLETED)
        assert not eligible
        assert reason == "terminal_state"

    def test_transferring_state_not_eligible(self):
        eligible, reason = self._check(state=ConversationState.TRANSFERRING_TO_HUMAN_AGENT)
        assert not eligible
        assert reason == "terminal_state"

    def test_wrong_intent_not_eligible(self):
        eligible, reason = self._check(intent="CHECKOUT")
        assert not eligible
        assert reason == "intent_not_add_item"

    def test_unsupported_state_not_eligible(self):
        eligible, reason = self._check(state=ConversationState.CONFIRMING_ORDER)
        assert not eligible
        assert reason == "state_not_supported"

    def test_text_too_short_not_eligible(self):
        eligible, reason = self._check(text="b", add_item_min_text_len=3)
        assert not eligible
        assert reason == "text_too_short"

    def test_coalesce_existing_repair_not_eligible(self):
        gate = AddItemEligibilityGate()
        eligible, reason = gate.check(
            intent_result=_mock_intent_result("ADD_ITEM"),
            state=ConversationState.IDLE,
            normalized_text="burger",
            gpt_shadow_decision="repair",
            gpt_shadow_repaired_intent="add_item",
            config=_mock_config(),
        )
        assert not eligible
        assert reason == "coalesce_existing_repair"

    def test_eligible_on_happy_path(self):
        eligible, reason = self._check()
        assert eligible
        assert reason == "eligible_add_item"

    def test_eligible_waiting_for_side(self):
        eligible, reason = self._check(state=ConversationState.WAITING_FOR_SIDE)
        assert eligible

    def test_eligible_waiting_for_modifier(self):
        eligible, reason = self._check(state=ConversationState.WAITING_FOR_MODIFIER)
        assert eligible

    def test_eligible_confirming_item(self):
        eligible, reason = self._check(state=ConversationState.CONFIRMING_ITEM)
        assert eligible

    def test_non_add_item_repair_does_not_coalesce(self):
        """Repair to a different intent should NOT coalesce."""
        gate = AddItemEligibilityGate()
        eligible, reason = gate.check(
            intent_result=_mock_intent_result("ADD_ITEM"),
            state=ConversationState.IDLE,
            normalized_text="burger",
            gpt_shadow_decision="repair",
            gpt_shadow_repaired_intent="checkout",
            config=_mock_config(),
        )
        assert eligible


# ---------------------------------------------------------------------------
# AddItemPayloadBuilder
# ---------------------------------------------------------------------------

class TestAddItemPayloadBuilder:
    def setup_method(self):
        self.builder = AddItemPayloadBuilder()

    def _build(self, **kwargs) -> list[dict]:
        defaults = dict(
            state=ConversationState.IDLE,
            normalized_text="I want a burger",
            current_item="",
            prompt_field="",
            local_intent="ADD_ITEM",
            local_confidence=0.9,
            top_k_intents=[],
            local_slots=[],
            choices=[],
            required_missing=[],
            cart_item_names=[],
            previous_turns=[],
        )
        defaults.update(kwargs)
        return self.builder.build_messages(**defaults)

    def test_returns_system_and_user_messages(self):
        msgs = self._build()
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_system_prompt_non_empty(self):
        msgs = self._build()
        assert len(msgs[0]["content"]) > 10

    def test_user_content_is_valid_json(self):
        msgs = self._build()
        payload = json.loads(msgs[1]["content"])
        assert isinstance(payload, dict)

    def test_payload_contains_task_key(self):
        payload = json.loads(self._build()[1]["content"])
        assert payload.get("t") == "extract_add_items"

    def test_payload_contains_state(self):
        payload = json.loads(self._build(state=ConversationState.IDLE)[1]["content"])
        assert payload["state"] == ConversationState.IDLE.value

    def test_payload_contains_text(self):
        payload = json.loads(self._build(normalized_text="double cheeseburger")[1]["content"])
        assert "double cheeseburger" in payload["text"]

    def test_no_full_menu_in_payload(self):
        # Payload must not contain "menu" top-level key with item list
        payload = json.loads(self._build()[1]["content"])
        assert "menu" not in payload
        assert "full_menu" not in payload

    def test_no_api_key_in_payload(self):
        payload_str = self._build()[1]["content"]
        assert "sk-" not in payload_str

    def test_cart_capped_at_10_items(self):
        cart_names = [f"item{i}" for i in range(20)]
        payload = json.loads(self._build(cart_item_names=cart_names)[1]["content"])
        # Cart payload structure: {"n": total_count, "items": [capped_names]}
        cart_in_payload = payload.get("cart", {})
        items_in_payload = cart_in_payload.get("items", []) if isinstance(cart_in_payload, dict) else cart_in_payload
        assert len(items_in_payload) <= 10

    def test_history_capped_at_3_turns(self):
        history = [("user", f"msg{i}") for i in range(10)]
        payload = json.loads(self._build(previous_turns=history)[1]["content"])
        hist_in_payload = payload.get("history", [])
        assert len(hist_in_payload) <= 3

    def test_top_k_capped_at_4(self):
        top_k = [{"intent": f"INTENT_{i}", "conf": 0.1} for i in range(10)]
        payload = json.loads(self._build(top_k_intents=top_k)[1]["content"])
        local = payload.get("local", {})
        top_k_out = local.get("top_k", [])
        assert len(top_k_out) <= 4

    def test_choices_capped_at_12(self):
        choices = [f"choice_{i}" for i in range(20)]
        payload = json.loads(
            self._build(
                state=ConversationState.WAITING_FOR_SIDE,
                choices=choices,
            )[1]["content"]
        )
        choices_out = payload.get("choices", [])
        assert len(choices_out) <= 12

    def test_choices_only_included_for_waiting_states(self):
        # IDLE should not include choices even if provided
        payload = json.loads(
            self._build(state=ConversationState.IDLE, choices=["choice1"])[1]["content"]
        )
        # choices key may be absent or empty for non-waiting states
        choices_out = payload.get("choices", [])
        assert len(choices_out) == 0

    def test_choices_included_for_waiting_for_side(self):
        payload = json.loads(
            self._build(
                state=ConversationState.WAITING_FOR_SIDE,
                choices=["fries", "salad"],
            )[1]["content"]
        )
        assert "fries" in payload.get("choices", [])

    def test_prompt_field_included_when_non_empty(self):
        payload = json.loads(self._build(prompt_field="size")[1]["content"])
        assert payload.get("prompt") == "size"

    def test_prompt_field_absent_when_empty(self):
        payload = json.loads(self._build(prompt_field="")[1]["content"])
        assert "prompt" not in payload

    def test_current_item_included_when_non_empty(self):
        payload = json.loads(self._build(current_item="Hawaiian pizza")[1]["content"])
        assert payload.get("current_item") == "Hawaiian pizza"

    def test_schema_included_in_payload(self):
        payload = json.loads(self._build()[1]["content"])
        assert "schema" in payload or "rules" in payload

    def test_no_prices_in_payload(self):
        payload_str = self._build()[1]["content"]
        assert "price" not in payload_str.lower()
        assert "tax" not in payload_str.lower()
