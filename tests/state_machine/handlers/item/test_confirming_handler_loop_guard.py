# tests/state_machine/handlers/item/test_confirming_handler_loop_guard.py
"""
Tests for bounded item-disambiguation loops in ConfirmingHandler.

Spec:
- Unresolved disambiguation is capped at 2 failed attempts.
- On attempt >= 2: IDLE + item_clarification_limit_reached (not another CONFIRMING_ITEM re-prompt).
- "yes" / "no" in the multiple_matches stage count as failed attempts (they give no item selection).
- Unresolved text input also counts.
- On successful candidate selection, counter does NOT escalate.
- Rejected candidates (after DENY on candidate_selected) are not offered again.
"""
from __future__ import annotations

import pytest

from app.menu.models import MenuItem, Pricing
from app.menu.query_result import MenuQueryResult, MenuQueryType
from app.nlu.intent_resolution.intent import Intent
from app.nlu.query_normalization.text_preprocessor import normalize_text
from app.state_machine.handlers.item.confirming_handler import ConfirmingHandler
from app.state_machine.models.conversation_context import ConversationContext
from app.state_machine.models.conversation_state import ConversationState


# ── Stubs ─────────────────────────────────────────────────────────────────────

class _StubMenuRepo:
    def __init__(self, *, query_result: MenuQueryResult | None = None):
        self._query_result = query_result or MenuQueryResult(type=MenuQueryType.NOT_FOUND)

    def get_item(self, item_id: str) -> MenuItem:
        names = {
            "burger_1": "Zinger Burger",
            "burger_2": "Chicken Burger",
            "burger_3": "Classic Burger",
        }
        name = names.get(item_id, "Unknown Item")
        return MenuItem(
            item_id=item_id,
            name=name,
            normalized_name=normalize_text(name),
            aliases=(),
            normalized_aliases=(),
            voice_labels=(),
            pricing=Pricing(mode="fixed", price_cents=1000),
            side_groups=[],
            modifier_groups=[],
            available=True,
        )

    def resolve_item_within_candidates_normalized(
        self, normalized_text: str, candidate_item_ids: list[str]
    ) -> MenuItem | None:
        return None

    def resolve_menu_query(self, normalized_text: str, limit: int = 5) -> MenuQueryResult:
        return self._query_result

    def resolve_menu_query_from_slots(self, **kwargs) -> MenuQueryResult:
        return self._query_result


class _Session:
    conversation_state = ConversationState.CONFIRMING_ITEM


# ── Helpers ───────────────────────────────────────────────────────────────────

def _multiple_matches_confirmation(*, rejected_candidate_ids: list[str] | None = None) -> dict:
    base = {
        "type": "item",
        "reason": "multiple_matches",
        "query": "burger",
        "candidate_item_ids": ["burger_1", "burger_2"],
        "candidate_item_names": ["Zinger Burger", "Chicken Burger"],
    }
    if rejected_candidate_ids:
        base["rejected_candidate_ids"] = rejected_candidate_ids
    return base


def _handle(handler, context, user_text, intent=Intent.UNKNOWN):
    return handler.handle(
        intent=intent,
        context=context,
        user_text=user_text,
        session=_Session(),
    )


# ── Scenario 1: two unresolved text inputs → escalate ─────────────────────────

def test_first_unresolved_input_repeats_prompt() -> None:
    handler = ConfirmingHandler(_StubMenuRepo())
    context = ConversationContext()
    context.awaiting_confirmation_for = _multiple_matches_confirmation()

    result = _handle(handler, context, "something completely different")

    assert result.next_state == ConversationState.CONFIRMING_ITEM
    assert result.response_key == "confirm_item_ambiguous"
    assert context.reprompt_count("item_disambiguation") == 1


def test_second_unresolved_input_escalates_to_idle() -> None:
    handler = ConfirmingHandler(_StubMenuRepo())
    context = ConversationContext()
    context.awaiting_confirmation_for = _multiple_matches_confirmation()

    _handle(handler, context, "something completely different")
    result = _handle(handler, context, "still not matching")

    assert result.next_state == ConversationState.IDLE
    assert result.response_key == "item_clarification_limit_reached"


def test_escalation_clears_item_scope() -> None:
    handler = ConfirmingHandler(_StubMenuRepo())
    context = ConversationContext()
    context.awaiting_confirmation_for = _multiple_matches_confirmation()
    context.current_item_name = "Burger"

    _handle(handler, context, "no match")
    _handle(handler, context, "still no match")

    assert context.awaiting_confirmation_for is None
    assert context.current_item_name is None


# ── Scenario 2: "yes" / "no" in disambiguation stage count as failed attempts ─

def test_yes_in_disambiguation_stage_first_attempt_repeats() -> None:
    handler = ConfirmingHandler(_StubMenuRepo())
    context = ConversationContext()
    context.awaiting_confirmation_for = _multiple_matches_confirmation()

    result = _handle(handler, context, "yes", intent=Intent.CONFIRM)

    assert result.next_state == ConversationState.CONFIRMING_ITEM
    assert context.reprompt_count("item_disambiguation") == 1


def test_yes_twice_in_disambiguation_stage_escalates() -> None:
    handler = ConfirmingHandler(_StubMenuRepo())
    context = ConversationContext()
    context.awaiting_confirmation_for = _multiple_matches_confirmation()

    _handle(handler, context, "yes", intent=Intent.CONFIRM)
    result = _handle(handler, context, "yes", intent=Intent.CONFIRM)

    assert result.next_state == ConversationState.IDLE
    assert result.response_key == "item_clarification_limit_reached"


def test_no_twice_in_disambiguation_stage_escalates() -> None:
    handler = ConfirmingHandler(_StubMenuRepo())
    context = ConversationContext()
    context.awaiting_confirmation_for = _multiple_matches_confirmation()

    _handle(handler, context, "no")
    result = _handle(handler, context, "no")

    assert result.next_state == ConversationState.IDLE
    assert result.response_key == "item_clarification_limit_reached"


# ── Scenario 3: successful match resets counter, no premature escalation ──────

def test_successful_candidate_match_does_not_escalate() -> None:
    class _MatchingRepo(_StubMenuRepo):
        def resolve_item_within_candidates_normalized(self, normalized_text, candidate_item_ids):
            if "zinger" in normalized_text:
                return self.get_item("burger_1")
            return None

    handler = ConfirmingHandler(_MatchingRepo())
    context = ConversationContext()
    context.awaiting_confirmation_for = _multiple_matches_confirmation()

    result = _handle(handler, context, "I want the zinger burger")

    assert result.next_state == ConversationState.CONFIRMING_ITEM
    assert result.response_key == "confirm_item"
    # Attempt counter not bumped on success
    assert context.reprompt_count("item_disambiguation") == 0


# ── Scenario 4: counter resets on cancel ──────────────────────────────────────

def test_cancel_in_disambiguation_stage_does_not_leave_stale_counter() -> None:
    handler = ConfirmingHandler(_StubMenuRepo())
    context = ConversationContext()
    context.awaiting_confirmation_for = _multiple_matches_confirmation()

    _handle(handler, context, "no match")
    result = _handle(handler, context, "cancel", intent=Intent.CANCEL_ORDER)

    assert result.next_state == ConversationState.IDLE
    assert result.response_key == "item_cancelled_successfully"


# ── Scenario 5: rejected candidate tracking after DENY on candidate_selected ──

def test_deny_on_candidate_selected_adds_to_rejected_list() -> None:
    handler = ConfirmingHandler(_StubMenuRepo())
    context = ConversationContext()
    context.awaiting_confirmation_for = {
        "type": "item",
        "reason": "candidate_selected",
        "value_id": "burger_1",
        "value_name": "Zinger Burger",
        "previous_confirmation": _multiple_matches_confirmation(),
    }

    result = _handle(handler, context, "no", intent=Intent.UNKNOWN)

    assert result.next_state == ConversationState.CONFIRMING_ITEM
    assert result.response_key == "confirm_item_ambiguous"
    restored = context.awaiting_confirmation_for
    assert "burger_1" in (restored.get("rejected_candidate_ids") or [])


def test_rejected_candidate_not_presented_again() -> None:
    """After rejecting burger_1, the candidate list passed to the prompt excludes it."""
    handler = ConfirmingHandler(_StubMenuRepo())
    context = ConversationContext()
    # Simulate already-rejected state restored by DENY handling
    context.awaiting_confirmation_for = _multiple_matches_confirmation(
        rejected_candidate_ids=["burger_1"]
    )

    # Handler should try to resolve within the non-rejected candidates only.
    # A fresh resolution returning NOT_FOUND means it falls back to repeat.
    result = _handle(handler, context, "still not sure")

    # Should still repeat (attempt 1), but the payload must exclude rejected candidate.
    assert result.next_state == ConversationState.CONFIRMING_ITEM
    candidate_ids = (result.response_payload or {}).get("candidate_item_ids") or []
    assert "burger_1" not in candidate_ids
