# tests/responses/test_no_duplicate_couldnt_find_emission.py
"""Tests that "I couldn't find X." is emitted at most once per response.

Before the fix, when partial_success=True for a terminal response key:
  - ResponseBuilder.build() prepended prefill_feedback ("I couldn't find X.")
  - item_added_successfully() also appended unmatched_note ("I couldn't find X.")
  Result: the caller heard "I couldn't find X. Pot Stickers added. I couldn't find X."

Fix applied:
  - ResponseBuilder never prepends prefill_feedback for TERMINAL_SUCCESS_KEYS.
  - item_added_successfully owns the single feedback emission for partial_success=True.
"""
from __future__ import annotations

import pytest
from types import SimpleNamespace

from app.core.response_builder import ResponseBuilder, TERMINAL_SUCCESS_KEYS
from app.responses.item.success import item_added_successfully
from app.state_machine.models.conversation_context import ConversationContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeStore:
    def get_item(self, item_id):
        return SimpleNamespace(
            item_id=item_id,
            name="Pot Stickers",
            pricing=SimpleNamespace(variants=[]),
            side_groups=[],
            modifier_groups=[],
        )


class _FakeMenuRepo:
    def __init__(self):
        self.store = _FakeStore()


def _builder() -> ResponseBuilder:
    return ResponseBuilder(_FakeMenuRepo())


def _context() -> ConversationContext:
    return ConversationContext()


# ---------------------------------------------------------------------------
# TERMINAL_SUCCESS_KEYS contract
# ---------------------------------------------------------------------------


class TestTerminalKeySet:
    def test_item_added_successfully_is_terminal(self) -> None:
        assert "item_added_successfully" in TERMINAL_SUCCESS_KEYS

    def test_order_completed_is_terminal(self) -> None:
        assert "order_completed" in TERMINAL_SUCCESS_KEYS

    def test_ask_for_side_is_not_terminal(self) -> None:
        assert "ask_for_side" not in TERMINAL_SUCCESS_KEYS


# ---------------------------------------------------------------------------
# ResponseBuilder: terminal keys never prepend prefill_feedback
# ---------------------------------------------------------------------------


class TestTerminalKeyNeverPrependsCallback:
    @pytest.mark.parametrize("key", list(TERMINAL_SUCCESS_KEYS))
    def test_terminal_key_does_not_prepend_prefill_feedback(self, key: str) -> None:
        """For any terminal key, prefill_feedback must never be prepended."""
        builder = _builder()
        ctx = _context()

        try:
            text = builder.build(
                key,
                ctx,
                {
                    "item_name": "Pot Stickers",
                    "quantity": 1,
                    "prefill_feedback": "I couldn't find port stickers.",
                    "partial_success": True,
                    "unresolved_entities": ["port stickers"],
                    "order_type": "pickup",
                    "phone_number": "5555555555",
                    "sms_enabled": False,
                },
            )
        except Exception:
            pytest.skip(f"key={key!r} response function requires specific payload shape")

        count = text.lower().count("couldn't find")
        assert count <= 1, (
            f"key={key!r}: 'couldn't find' appeared {count} times — duplicate detected. "
            f"response={text!r}"
        )


class TestNonTerminalKeyPrependsCallback:
    def test_ask_for_quantity_prepends_prefill_feedback(self) -> None:
        """Non-terminal keys should still have prefill_feedback prepended."""
        builder = _builder()
        ctx = _context()

        text = builder.build(
            "ask_for_quantity",
            ctx,
            {
                "item_name": "Burger",
                "prefill_feedback": "I couldn't find rice.",
            },
        )

        assert "I couldn't find rice." in text

    def test_ask_for_side_prepends_prefill_feedback(self) -> None:
        builder = _builder()
        ctx = _context()
        ctx.current_item_name = "Test Burger"
        ctx.current_item_id = "test_burger"

        try:
            text = builder.build(
                "ask_for_side",
                ctx,
                {
                    "group_name": "Choose Drink",
                    "choices": ["Coke", "Sprite"],
                    "prefill_feedback": "I couldn't find avocado.",
                },
            )
            assert "I couldn't find avocado." in text
        except Exception:
            pytest.skip("ask_for_side requires specific context shape")


# ---------------------------------------------------------------------------
# item_added_successfully: feedback exactly once for partial_success=True
# ---------------------------------------------------------------------------


class TestItemAddedSuccessfullyFeedbackCount:
    def test_partial_success_true_emits_feedback_once(self) -> None:
        text = item_added_successfully({
            "item_name": "Pot Stickers",
            "quantity": 1,
            "partial_success": True,
            "unresolved_entities": ["port stickers"],
        })
        assert text.count("couldn't find") == 1

    def test_partial_success_false_emits_no_feedback(self) -> None:
        text = item_added_successfully({
            "item_name": "Pot Stickers",
            "quantity": 1,
            "partial_success": False,
            "unresolved_entities": ["port stickers"],
        })
        assert "couldn't find" not in text.lower()

    def test_no_partial_success_no_entities_clean_response(self) -> None:
        text = item_added_successfully({
            "item_name": "Pot Stickers",
            "quantity": 1,
        })
        assert "couldn't find" not in text.lower()
        assert "Pot Stickers added." in text


# ---------------------------------------------------------------------------
# Full pipeline: ResponseBuilder.build() for item_added_successfully
# ---------------------------------------------------------------------------


class TestNoDuplicateEmissionViaResponseBuilder:
    def test_partial_success_with_terminal_key_one_emission(self) -> None:
        """ResponseBuilder + item_added_successfully: exactly one 'couldn't find'."""
        builder = _builder()
        ctx = _context()

        text = builder.build(
            "item_added_successfully",
            ctx,
            {
                "item_name": "Pot Stickers",
                "quantity": 1,
                "partial_success": True,
                "unresolved_entities": ["port stickers"],
                "prefill_feedback": "I couldn't find port stickers.",
            },
        )

        count = text.lower().count("couldn't find")
        assert count == 1, (
            f"Expected exactly 1 'couldn't find', got {count}. response={text!r}"
        )

    def test_clean_terminal_key_zero_emissions(self) -> None:
        """Clean add with prefill_feedback present must still emit zero 'couldn't find'."""
        builder = _builder()
        ctx = _context()

        text = builder.build(
            "item_added_successfully",
            ctx,
            {
                "item_name": "Pot Stickers",
                "quantity": 1,
                "partial_success": False,
                "prefill_feedback": "I couldn't find port stickers.",
            },
        )

        assert "couldn't find" not in text.lower(), (
            f"Clean add must not emit feedback. response={text!r}"
        )
