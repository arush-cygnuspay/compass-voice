# tests/state_machine/policy/test_contextual_control_resolver.py
"""Unit tests for the contextual control resolver.

All tests call resolve_contextual_control() directly — no TurnEngine needed.
Tests are grouped by behavior contract:
  A. IDLE + ANYTHING_ELSE context — finish-adding coercion
  B. CONFIRMING_ORDER — payment-status coercion
  C. Guard: required item step states — resolver must return NONE
  D. Guard: no-context cases — resolver must return NONE
"""
import pytest

from app.nlu.intent_resolution.intent import Intent
from app.nlu.prompt_type import PromptType
from app.state_machine.models.conversation_state import ConversationState
from app.state_machine.policy.contextual_control_resolver import (
    ContextualControlKind,
    resolve_contextual_control,
)

_ANYTHING_ELSE = PromptType.ANYTHING_ELSE.value
_CONFIRM_ORDER = PromptType.CONFIRM_ORDER.value
_UNKNOWN_PT = PromptType.UNKNOWN.value


def _resolve(
    *,
    state: ConversationState = ConversationState.IDLE,
    last_prompt_type: str | None = _ANYTHING_ELSE,
    cart_has_items: bool = True,
    text: str = "",
    intent: Intent = Intent.UNKNOWN,
):
    return resolve_contextual_control(
        state=state,
        last_prompt_type=last_prompt_type,
        cart_has_items=cart_has_items,
        normalized_text=text,
        intent=intent,
    )


# ---------------------------------------------------------------------------
# Contract A: IDLE + ANYTHING_ELSE → FINISH_ADDING
# ---------------------------------------------------------------------------

class TestFinishAddingExactPhrases:
    """Exact-match phrases in _FINISH_ADDING_EXACT must coerce to CHECKOUT."""

    @pytest.mark.parametrize("text,intent", [
        ("no thats it for now", Intent.UNKNOWN),
        ("no that is it for now", Intent.UNKNOWN),
        ("no thats all for now", Intent.UNKNOWN),
        ("no i dont want anything else", Intent.DENY),
        ("no i do not want anything else", Intent.UNKNOWN),
        ("i dont want anything else", Intent.CANCEL_ORDER),
        ("i do not want anything else", Intent.CANCEL_ORDER),
        ("thats it for now", Intent.UNKNOWN),
        ("that is it for now", Intent.UNKNOWN),
        ("thats all for now", Intent.UNKNOWN),
        ("i think thats all", Intent.UNKNOWN),
        ("i think that is all", Intent.UNKNOWN),
        ("i think thats it", Intent.UNKNOWN),
        ("no thats all", Intent.DENY),
        ("no more items", Intent.UNKNOWN),
    ])
    def test_finish_adding_exact(self, text: str, intent: Intent) -> None:
        decision = _resolve(text=text, intent=intent)
        assert decision.kind == ContextualControlKind.FINISH_ADDING, (
            f"Expected FINISH_ADDING for {text!r} / {intent}, got {decision.kind}"
        )
        assert decision.confidence >= 0.8

    def test_source_is_set(self) -> None:
        decision = _resolve(text="no thats it for now", intent=Intent.UNKNOWN)
        assert decision.source != "none"
        assert decision.reason is not None


class TestCancelMisfireGuard:
    """NLU fires CANCEL_ORDER on 'done adding' text — coerce to FINISH_ADDING."""

    @pytest.mark.parametrize("text", [
        "no i dont want anything else",
        "i dont want more",
        "no more",
        "nothing else",
    ])
    def test_cancel_order_coerced_when_done_like(self, text: str) -> None:
        decision = _resolve(text=text, intent=Intent.CANCEL_ORDER)
        assert decision.kind == ContextualControlKind.FINISH_ADDING, (
            f"Expected FINISH_ADDING for CANCEL_ORDER + {text!r}, got {decision.kind}"
        )

    @pytest.mark.parametrize("text", [
        "cancel my order",
        "cancel everything",
        "cancel the order",
        "cancel the entire order",
        "start over",
    ])
    def test_explicit_cancel_not_coerced(self, text: str) -> None:
        decision = _resolve(text=text, intent=Intent.CANCEL_ORDER)
        # Explicit cancel requests must NOT be coerced to FINISH_ADDING.
        assert decision.kind != ContextualControlKind.FINISH_ADDING, (
            f"FINISH_ADDING should not fire for explicit cancel: {text!r}"
        )

    def test_deny_on_done_text_coerced(self) -> None:
        decision = _resolve(text="no nothing else", intent=Intent.DENY)
        assert decision.kind == ContextualControlKind.FINISH_ADDING


class TestIdleGuards:
    """FINISH_ADDING must not fire outside IDLE or without ANYTHING_ELSE context."""

    def test_no_context_does_not_coerce(self) -> None:
        decision = _resolve(
            text="no thats it for now",
            intent=Intent.CANCEL_ORDER,
            last_prompt_type=_UNKNOWN_PT,
        )
        assert decision.kind == ContextualControlKind.NONE

    def test_none_context_does_not_coerce(self) -> None:
        decision = _resolve(
            text="no thats it for now",
            intent=Intent.CANCEL_ORDER,
            last_prompt_type=None,
        )
        assert decision.kind == ContextualControlKind.NONE

    def test_empty_cart_does_not_coerce(self) -> None:
        decision = _resolve(
            text="no thats it for now",
            intent=Intent.CANCEL_ORDER,
            cart_has_items=False,
        )
        assert decision.kind == ContextualControlKind.NONE

    def test_wrong_state_does_not_coerce(self) -> None:
        decision = _resolve(
            state=ConversationState.CONFIRMING_ORDER,
            text="no thats it for now",
            intent=Intent.CANCEL_ORDER,
        )
        assert decision.kind == ContextualControlKind.NONE

    def test_confirm_order_context_does_not_coerce_to_finish_adding(self) -> None:
        decision = _resolve(
            last_prompt_type=_CONFIRM_ORDER,
            text="no thats it for now",
            intent=Intent.DENY,
        )
        assert decision.kind == ContextualControlKind.NONE


# ---------------------------------------------------------------------------
# Contract B: CONFIRMING_ORDER + payment-status phrase → PAYMENT_STATUS_QUERY
# ---------------------------------------------------------------------------

class TestPaymentStatusInConfirmingOrder:
    """Payment-status phrases in CONFIRMING_ORDER must coerce to PAYMENT_STATUS."""

    @pytest.mark.parametrize("text", [
        "you got your card",
        "you got the card",
        "got your card",
        "got the card",
        "got your payment",
        "i paid",
        "i have paid",
        "already paid",
        "payment received",
        "the code",
        "qr code",
        "qr",
        "the payment code",
        "i have the code",
        "i got the code",
    ])
    def test_payment_status_exact(self, text: str) -> None:
        decision = _resolve(
            state=ConversationState.CONFIRMING_ORDER,
            last_prompt_type=_CONFIRM_ORDER,
            text=text,
            intent=Intent.UNKNOWN,
        )
        assert decision.kind == ContextualControlKind.PAYMENT_STATUS_QUERY, (
            f"Expected PAYMENT_STATUS_QUERY for {text!r}, got {decision.kind}"
        )
        assert decision.confidence >= 0.9

    def test_payment_status_with_payment_status_intent(self) -> None:
        decision = _resolve(
            state=ConversationState.CONFIRMING_ORDER,
            last_prompt_type=_CONFIRM_ORDER,
            text="you got your card",
            intent=Intent.PAYMENT_STATUS,
        )
        assert decision.kind == ContextualControlKind.PAYMENT_STATUS_QUERY

    def test_checkout_phrase_not_coerced_to_payment_status(self) -> None:
        decision = _resolve(
            state=ConversationState.CONFIRMING_ORDER,
            last_prompt_type=_CONFIRM_ORDER,
            text="checkout",
            intent=Intent.CHECKOUT,
        )
        assert decision.kind == ContextualControlKind.NONE

    def test_normal_phrase_in_confirming_order_returns_none(self) -> None:
        decision = _resolve(
            state=ConversationState.CONFIRMING_ORDER,
            last_prompt_type=_CONFIRM_ORDER,
            text="yes that is correct",
            intent=Intent.CONFIRM,
        )
        assert decision.kind == ContextualControlKind.NONE


# ---------------------------------------------------------------------------
# Contract C: Required item step states — resolver must return NONE
# ---------------------------------------------------------------------------

class TestRequiredItemStepGuard:
    """Resolver must never fire in mid-item required states."""

    @pytest.mark.parametrize("state", [
        ConversationState.WAITING_FOR_SIDE,
        ConversationState.WAITING_FOR_MODIFIER,
        ConversationState.WAITING_FOR_SIZE,
        ConversationState.WAITING_FOR_SIDE_SIZE,
        ConversationState.WAITING_FOR_QUANTITY,
        ConversationState.CONFIRMING_ITEM,
    ])
    def test_required_state_returns_none(self, state: ConversationState) -> None:
        decision = _resolve(
            state=state,
            last_prompt_type=_ANYTHING_ELSE,
            text="no thats it for now",
            intent=Intent.CANCEL_ORDER,
        )
        assert decision.kind == ContextualControlKind.NONE, (
            f"Resolver should return NONE in state {state.value}"
        )


# ---------------------------------------------------------------------------
# Contract D: Edge cases / empty input
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_text_returns_none(self) -> None:
        decision = _resolve(text="", intent=Intent.CANCEL_ORDER)
        assert decision.kind == ContextualControlKind.NONE

    def test_whitespace_text_returns_none(self) -> None:
        decision = _resolve(text="   ", intent=Intent.CANCEL_ORDER)
        assert decision.kind == ContextualControlKind.NONE

    def test_add_item_intent_not_coerced(self) -> None:
        decision = _resolve(
            text="chicken burger",
            intent=Intent.ADD_ITEM,
        )
        assert decision.kind == ContextualControlKind.NONE

    def test_payment_status_outside_confirming_order_returns_none(self) -> None:
        decision = _resolve(
            state=ConversationState.IDLE,
            last_prompt_type=_ANYTHING_ELSE,
            text="you got your card",
            intent=Intent.UNKNOWN,
        )
        # In IDLE, "you got your card" is not a finish-adding phrase
        # and the payment-status rule only fires in CONFIRMING_ORDER.
        assert decision.kind == ContextualControlKind.NONE
