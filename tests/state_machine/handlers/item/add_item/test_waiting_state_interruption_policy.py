# tests/state_machine/handlers/item/add_item/test_waiting_state_interruption_policy.py
"""Tests for Phase 6+7: waiting-state interruption policy and response template.

Validates:
- InterruptionDecision.BLOCK returned when group is required, min not met,
  and user utterance looks like a new item request.
- InterruptionDecision.ALLOW returned for legitimate option answers.
- InterruptionDecision.ALLOW returned when group is optional.
- InterruptionDecision.ALLOW returned when min is already met.
- block_new_item_until_required_done renders correctly.
"""
from __future__ import annotations

import pytest

from app.state_machine.handlers.item.add_item.waiting_state_interruption_policy import (
    InterruptionDecision,
    evaluate_waiting_state_interruption,
)


# ---------------------------------------------------------------------------
# evaluate_waiting_state_interruption
# ---------------------------------------------------------------------------

class TestInterruptionPolicyBlock:
    def _check(
        self,
        text,
        *,
        group_is_required=True,
        selected_count=0,
        min_selector=1,
        group_prompt_noun="side",
        pending_item_name="Cheese Burger",
    ):
        return evaluate_waiting_state_interruption(
            normalized_user_text=text,
            pending_item_name=pending_item_name,
            group_is_required=group_is_required,
            group_prompt_noun=group_prompt_noun,
            selected_count=selected_count,
            min_selector=min_selector,
        )

    @pytest.mark.parametrize("text", [
        "can i get a coke",
        "can i get coke",
        "i want a coke",
        "add a coke",
        "i would like a burger",
        "give me a large fries",
        "let me get a sprite",
        "i also want a salad",
        "could i get a water",
    ])
    def test_ordering_prefix_triggers_block_when_required(self, text):
        result = self._check(text)
        assert result.decision == InterruptionDecision.BLOCK, (
            f"Expected BLOCK for {text!r}, got {result.decision}"
        )
        assert result.pending_item_name == "Cheese Burger"
        assert result.group_prompt_noun == "side"

    def test_block_remaining_set_correctly(self):
        result = self._check("can i get a coke", min_selector=2, selected_count=0)
        assert result.remaining_to_min == 2

    def test_block_with_partial_selection(self):
        """Still BLOCK when required group min not yet met."""
        result = self._check("i want a sprite", min_selector=2, selected_count=1)
        assert result.decision == InterruptionDecision.BLOCK


class TestInterruptionPolicyAllow:
    def _check(
        self,
        text,
        *,
        group_is_required=True,
        selected_count=0,
        min_selector=1,
    ):
        return evaluate_waiting_state_interruption(
            normalized_user_text=text,
            pending_item_name="Cheese Burger",
            group_is_required=group_is_required,
            group_prompt_noun="side",
            selected_count=selected_count,
            min_selector=min_selector,
        )

    def test_option_answer_not_blocked(self):
        """Plain option phrases are ALLOW — not new-item requests."""
        for text in ["fries", "coleslaw", "no side", "none", "skip"]:
            result = self._check(text)
            assert result.decision == InterruptionDecision.ALLOW, (
                f"Expected ALLOW for '{text}', got {result.decision}"
            )

    def test_optional_group_not_blocked(self):
        """Optional groups never block — user can pivot freely."""
        result = self._check(
            "can i get a coke",
            group_is_required=False,
        )
        assert result.decision == InterruptionDecision.ALLOW

    def test_min_already_met_not_blocked(self):
        """When min_selector already satisfied, ALLOW even for new-item phrases."""
        result = self._check(
            "i want a coke",
            min_selector=1,
            selected_count=1,  # min already met
        )
        assert result.decision == InterruptionDecision.ALLOW

    def test_empty_text_not_blocked(self):
        result = self._check("")
        assert result.decision == InterruptionDecision.ALLOW

    def test_min_zero_not_blocked(self):
        """min_selector=0 effectively means optional — never block."""
        result = self._check("can i get a coke", min_selector=0)
        assert result.decision == InterruptionDecision.ALLOW


# ---------------------------------------------------------------------------
# block_new_item_until_required_done response template
# ---------------------------------------------------------------------------

class TestBlockNewItemResponse:
    def _render(self, pending_item_name="", group_prompt_noun=""):
        from app.responses.item.sides import block_new_item_until_required_done
        from unittest.mock import MagicMock
        ctx = MagicMock()
        repo = MagicMock()
        payload = {}
        if pending_item_name:
            payload["pending_item_name"] = pending_item_name
        if group_prompt_noun:
            payload["group_prompt_noun"] = group_prompt_noun
        return block_new_item_until_required_done(ctx, repo, payload)

    def test_response_mentions_pending_item(self):
        text = self._render("Cheese Burger", "side")
        assert "Cheese Burger" in text

    def test_response_mentions_group_noun(self):
        text = self._render("Burger", "sauce")
        assert "sauce" in text

    def test_response_tells_user_to_cancel(self):
        text = self._render("Burger", "side")
        assert "cancel" in text.lower()

    def test_no_pending_item_name_fallback(self):
        """Without item name, graceful fallback message."""
        text = self._render("", "side")
        assert "side" in text
        assert "cancel" in text.lower()

    def test_no_group_noun_fallback(self):
        text = self._render("Coke", "")
        assert "Coke" in text
        # Falls back to "option" as noun
        assert "option" in text or "cancel" in text.lower()

    def test_no_x_notation(self):
        """Response must not contain 'x' multiplier notation (voice-safe)."""
        text = self._render("Cheese Burger", "side")
        assert " x " not in text
        assert "x2" not in text
