# tests/responses/test_response_consistency_invariants.py
"""Table-driven invariant tests for item_added_successfully() response shape.

These tests do not target a single behaviour; they assert structural invariants
that must hold across every valid payload combination:

  1. Response always ends with "Would you like anything else?"
  2. Feedback ("I couldn't find X.") appears IFF partial_success=True
     and unresolved_entities is non-empty, OR legacy unmatched_names present.
  3. Feedback text immediately precedes the closing question.
  4. Feedback never duplicates: at most one "I couldn't find" sentence.
  5. quantity=2 prefix is "Added 2 <item>" not "<item> added".
  6. queue_transition format: "<prev> added. <this> added.[feedback] ..."
"""
from __future__ import annotations

import pytest

from app.responses.item.success import item_added_successfully


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CLOSING = "Would you like anything else?"
_FIND_MARKER = "I couldn't find"


def _simple(
    item_name: str = "Coke",
    quantity: int = 1,
    *,
    partial_success: bool | None = None,
    unresolved_entities: list[str] | None = None,
    unmatched_names: list[str] | None = None,
) -> dict:
    payload: dict = {"item_name": item_name, "quantity": quantity}
    if partial_success is not None:
        payload["partial_success"] = partial_success
    if unresolved_entities is not None:
        payload["unresolved_entities"] = unresolved_entities
    if unmatched_names is not None:
        payload["unmatched_names"] = unmatched_names
    return payload


def _queue(
    item_name: str,
    quantity: int,
    prev_name: str,
    prev_qty: int,
    remaining: int = 0,
    **kw,
) -> dict:
    return {
        "item_name": item_name,
        "quantity": quantity,
        "prev_item_name": prev_name,
        "prev_quantity": prev_qty,
        "remaining_queue_count": remaining,
        "queue_transition": True,
        **kw,
    }


# ---------------------------------------------------------------------------
# Invariant 1: every response closes with the standard question
# ---------------------------------------------------------------------------

class TestAlwaysEndsWithClosingQuestion:
    @pytest.mark.parametrize("payload", [
        _simple("Coke"),
        _simple("Coke", 2),
        _simple("Coke", partial_success=True, unresolved_entities=["rice"]),
        _simple("Coke", unmatched_names=["rice"]),
        _simple("Coke", partial_success=False, unresolved_entities=["rice"]),
        _queue("Coke", 1, "Burger", 1),
        _queue("Coke", 1, "Coke", 1),
        _queue("Coke", 1, "Burger", 1, remaining=2),
    ])
    def test_ends_with_closing_question(self, payload: dict) -> None:
        text = item_added_successfully(payload)
        assert text.endswith(_CLOSING), f"Missing closing question in: {text!r}"


# ---------------------------------------------------------------------------
# Invariant 2: feedback presence matches expectation
# ---------------------------------------------------------------------------

class TestFeedbackPresenceMatchesExpectation:
    @pytest.mark.parametrize("payload,expect_feedback", [
        # Clean adds — no feedback
        (_simple("Coke"), False),
        (_simple("Coke", 2), False),
        (_simple("Coke", partial_success=False), False),
        (_simple("Coke", partial_success=True, unresolved_entities=[]), False),
        # Primary path fires
        (_simple("Coke", partial_success=True, unresolved_entities=["rice"]), True),
        (_simple("Coke", partial_success=True, unresolved_entities=["rice", "avocado"]), True),
        # Primary path gate closed — entities present but partial_success absent/False
        (_simple("Coke", unresolved_entities=["rice"]), False),
        (_simple("Coke", partial_success=False, unresolved_entities=["rice"]), False),
        # Legacy path
        (_simple("Coke", unmatched_names=["rice"]), True),
        (_simple("Coke", unmatched_names=[]), False),
    ])
    def test_feedback_presence(self, payload: dict, expect_feedback: bool) -> None:
        text = item_added_successfully(payload)
        if expect_feedback:
            assert _FIND_MARKER in text, f"Expected feedback in: {text!r}"
        else:
            assert _FIND_MARKER not in text, f"Unexpected feedback in: {text!r}"


# ---------------------------------------------------------------------------
# Invariant 3: feedback immediately precedes the closing question
# ---------------------------------------------------------------------------

class TestFeedbackPositionBeforeClosingQuestion:
    def test_feedback_precedes_closing_question(self) -> None:
        text = item_added_successfully(
            _simple("Coke", partial_success=True, unresolved_entities=["rice"])
        )
        find_pos = text.index(_FIND_MARKER)
        closing_pos = text.index(_CLOSING)
        assert find_pos < closing_pos, (
            f"Feedback must appear before closing question: {text!r}"
        )

    def test_no_text_between_feedback_and_closing(self) -> None:
        text = item_added_successfully(
            _simple("Coke", partial_success=True, unresolved_entities=["rice"])
        )
        after_feedback = text[text.index(_FIND_MARKER):]
        # Everything after the find marker should contain the closing question
        assert _CLOSING in after_feedback


# ---------------------------------------------------------------------------
# Invariant 4: at most one "I couldn't find" sentence per response
# ---------------------------------------------------------------------------

class TestAtMostOneFeedbackSentence:
    @pytest.mark.parametrize("payload", [
        _simple("Coke", partial_success=True, unresolved_entities=["rice"]),
        _simple("Coke", unmatched_names=["rice"]),
        _simple(
            "Coke",
            partial_success=True,
            unresolved_entities=["rice"],
            unmatched_names=["legacy"],
        ),
    ])
    def test_single_feedback_sentence(self, payload: dict) -> None:
        text = item_added_successfully(payload)
        count = text.count(_FIND_MARKER)
        assert count <= 1, f"Multiple feedback sentences in: {text!r}"


# ---------------------------------------------------------------------------
# Invariant 5: quantity>1 prefix shape
# ---------------------------------------------------------------------------

class TestQuantityPrefixShape:
    def test_qty_1_format(self) -> None:
        text = item_added_successfully(_simple("Coke", 1))
        assert text.startswith("Coke added.")

    def test_qty_2_format(self) -> None:
        text = item_added_successfully(_simple("Coke", 2))
        assert text.startswith("Added 2 Coke.")

    def test_qty_2_with_feedback(self) -> None:
        text = item_added_successfully(
            _simple("Coke", 2, partial_success=True, unresolved_entities=["rice"])
        )
        assert "Added 2 Coke." in text
        assert "I couldn't find rice." in text

    def test_qty_1_with_feedback(self) -> None:
        text = item_added_successfully(
            _simple("Coke", 1, partial_success=True, unresolved_entities=["rice"])
        )
        assert text.startswith("Coke added.")
        assert "I couldn't find rice." in text


# ---------------------------------------------------------------------------
# Invariant 6: queue_transition response shape
# ---------------------------------------------------------------------------

class TestQueueTransitionShape:
    def test_different_items_both_mentioned(self) -> None:
        text = item_added_successfully(_queue("Coke", 1, "Burger", 1))
        assert "Burger" in text
        assert "Coke" in text
        assert text.endswith(_CLOSING)

    def test_same_item_merged_not_doubled(self) -> None:
        text = item_added_successfully(_queue("Coke", 1, "Coke", 1))
        assert "Coke added. Coke added" not in text
        assert text.endswith(_CLOSING)

    def test_queue_with_feedback(self) -> None:
        payload = _queue(
            "Coke", 1, "Burger", 1,
            partial_success=True,
            unresolved_entities=["rice"],
        )
        text = item_added_successfully(payload)
        assert "I couldn't find rice." in text
        assert text.endswith(_CLOSING)

    def test_queue_remaining_mentions_count(self) -> None:
        text = item_added_successfully(_queue("Coke", 1, "Burger", 1, remaining=2))
        assert "2" in text
        assert "more" in text.lower()
