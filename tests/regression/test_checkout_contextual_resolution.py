# tests/regression/test_checkout_contextual_resolution.py
"""Multi-turn regression tests for the contextual control resolver.

These tests cover the four production failures that the resolver fixes:

  1. IDLE + "No. That's it for now."  → should be confirm_order_summary
  2. IDLE + "No. I don't want anything else." → should be confirm_order_summary
  3. CONFIRMING_ORDER + "You got your card." → should be payment_not_started
  4. CONFIRMING_ORDER + "The code." → should be payment_not_started

All scenarios use a real TurnEngine + real MenuRepository with scripted NLU.
"""
from __future__ import annotations

import pytest

from app.nlu.intent_resolution.intent import Intent
from app.state_machine.models.conversation_state import ConversationState
from tests.support.voice_test_harness import (
    ScriptedTurn,
    StubCheckoutService,
    build_engine,
    build_menu_repo,
    make_slot,
    new_session,
    simulate_turn,
)


def _pickup_engine_session(*, sms_configured: bool = False):
    """Return an engine+session with pickup order type already selected."""
    from tests.support.voice_test_harness import StubSmsService
    sms = StubSmsService(configured=sms_configured)
    engine = build_engine(
        menu_repo=build_menu_repo(),
        sms_service=sms,
    )
    session = new_session()
    simulate_turn(engine, session, ScriptedTurn("pickup"))
    return engine, session


def _add_carrot_cake(engine, session) -> None:
    """Add a Carrot Cake (no modifiers/sides required) and confirm it is added."""
    # Carrot Cake has no required modifier/side — defaults to quantity 1.
    out = simulate_turn(
        engine, session,
        ScriptedTurn(
            "carrot cake",
            intent=Intent.ADD_ITEM,
            slots=(make_slot("ITEM", "Carrot Cake"),),
        ),
    )
    assert out.response_key == "item_added_successfully", (
        f"Expected item_added_successfully, got {out.response_key}"
    )
    assert session.last_prompt_type == "anything_else", (
        "last_prompt_type should be 'anything_else' after item_added_successfully"
    )


# ---------------------------------------------------------------------------
# Failure 1: "No. That's it for now." after item add
# ---------------------------------------------------------------------------

class TestNoThatIsItForNow:
    def test_thats_it_for_now_routes_to_confirm_order_summary(self) -> None:
        """IDLE + 'thats it for now' (UNKNOWN) → confirm_order_summary."""
        engine, session = _pickup_engine_session()
        _add_carrot_cake(engine, session)

        out = simulate_turn(
            engine, session,
            ScriptedTurn("No. That's it for now.", intent=Intent.UNKNOWN),
        )
        assert out.response_key == "confirm_order_summary", (
            f"Expected confirm_order_summary, got {out.response_key!r}"
        )
        assert out.state_after == ConversationState.CONFIRMING_ORDER

    def test_no_thats_it_routes_to_confirm_order_summary(self) -> None:
        """IDLE + 'no thats it' (DENY) → confirm_order_summary."""
        engine, session = _pickup_engine_session()
        _add_carrot_cake(engine, session)

        out = simulate_turn(
            engine, session,
            ScriptedTurn("no that's it", intent=Intent.DENY),
        )
        assert out.response_key == "confirm_order_summary"
        assert out.state_after == ConversationState.CONFIRMING_ORDER

    def test_thats_all_for_now_routes_to_confirm_order_summary(self) -> None:
        engine, session = _pickup_engine_session()
        _add_carrot_cake(engine, session)

        out = simulate_turn(
            engine, session,
            ScriptedTurn("that's all for now", intent=Intent.UNKNOWN),
        )
        assert out.response_key == "confirm_order_summary"

    def test_i_think_thats_all_routes_to_confirm_order_summary(self) -> None:
        engine, session = _pickup_engine_session()
        _add_carrot_cake(engine, session)

        out = simulate_turn(
            engine, session,
            ScriptedTurn("i think that's all", intent=Intent.UNKNOWN),
        )
        assert out.response_key == "confirm_order_summary"


# ---------------------------------------------------------------------------
# Failure 2: "No. I don't want anything else." — NLU fires CANCEL_ORDER
# ---------------------------------------------------------------------------

class TestNoDontWantAnythingElse:
    def test_cancel_order_on_finish_phrase_routes_to_confirm_order_summary(self) -> None:
        """IDLE + 'i dont want anything else' (CANCEL_ORDER) → confirm_order_summary."""
        engine, session = _pickup_engine_session()
        _add_carrot_cake(engine, session)

        out = simulate_turn(
            engine, session,
            ScriptedTurn(
                "No. I don't want anything else.",
                intent=Intent.CANCEL_ORDER,
            ),
        )
        assert out.response_key == "confirm_order_summary", (
            f"CANCEL_ORDER on 'no I dont want anything else' must not reach "
            f"intent_not_allowed. Got: {out.response_key!r}"
        )
        assert out.state_after == ConversationState.CONFIRMING_ORDER

    def test_dont_want_anything_else_routes_to_confirm_order_summary(self) -> None:
        engine, session = _pickup_engine_session()
        _add_carrot_cake(engine, session)

        out = simulate_turn(
            engine, session,
            ScriptedTurn("I don't want anything else", intent=Intent.CANCEL_ORDER),
        )
        assert out.response_key == "confirm_order_summary"

    def test_explicit_cancel_in_idle_not_coerced(self) -> None:
        """'Cancel my order' in IDLE must NOT be coerced to checkout."""
        engine, session = _pickup_engine_session()
        _add_carrot_cake(engine, session)

        out = simulate_turn(
            engine, session,
            ScriptedTurn("cancel my order", intent=Intent.CANCEL_ORDER),
        )
        # Explicit cancel must NOT result in confirm_order_summary.
        assert out.response_key != "confirm_order_summary"

    def test_no_without_cart_does_not_coerce(self) -> None:
        """CANCEL_ORDER in IDLE with no prior item must not coerce."""
        engine, session = _pickup_engine_session()
        # No items in cart — coercion must not fire.
        out = simulate_turn(
            engine, session,
            ScriptedTurn("no I don't want anything", intent=Intent.CANCEL_ORDER),
        )
        assert out.response_key != "confirm_order_summary"


# ---------------------------------------------------------------------------
# Failure 3: "You got your card." in CONFIRMING_ORDER
# ---------------------------------------------------------------------------

class TestPaymentStatusInConfirmingOrder:
    def _session_at_confirming_order(self):
        """Set up a session in CONFIRMING_ORDER state."""
        engine, session = _pickup_engine_session()
        _add_carrot_cake(engine, session)
        # Get to CONFIRMING_ORDER via "that's all"
        out = simulate_turn(
            engine, session,
            ScriptedTurn("that's all", intent=Intent.UNKNOWN),
        )
        assert out.state_after == ConversationState.CONFIRMING_ORDER, (
            f"Setup failed — expected CONFIRMING_ORDER, got {out.state_after}"
        )
        return engine, session

    def test_you_got_your_card_returns_payment_not_started(self) -> None:
        """'You got your card' in CONFIRMING_ORDER → payment_not_started."""
        engine, session = self._session_at_confirming_order()

        out = simulate_turn(
            engine, session,
            ScriptedTurn("You got your card.", intent=Intent.UNKNOWN),
        )
        assert out.response_key == "payment_not_started", (
            f"Expected payment_not_started, got {out.response_key!r}"
        )
        # Must stay in CONFIRMING_ORDER — order not yet confirmed.
        assert out.state_after == ConversationState.CONFIRMING_ORDER

    def test_already_paid_returns_payment_not_started(self) -> None:
        engine, session = self._session_at_confirming_order()

        out = simulate_turn(
            engine, session,
            ScriptedTurn("I already paid", intent=Intent.UNKNOWN),
        )
        assert out.response_key == "payment_not_started"

    def test_i_paid_returns_payment_not_started(self) -> None:
        engine, session = self._session_at_confirming_order()

        out = simulate_turn(
            engine, session,
            ScriptedTurn("I paid", intent=Intent.UNKNOWN),
        )
        assert out.response_key == "payment_not_started"

    def test_got_your_payment_returns_payment_not_started(self) -> None:
        engine, session = self._session_at_confirming_order()

        out = simulate_turn(
            engine, session,
            ScriptedTurn("got your payment", intent=Intent.UNKNOWN),
        )
        assert out.response_key == "payment_not_started"


# ---------------------------------------------------------------------------
# Failure 4: "The code." in CONFIRMING_ORDER
# ---------------------------------------------------------------------------

class TestTheCodeInConfirmingOrder:
    def _session_at_confirming_order(self):
        engine, session = _pickup_engine_session()
        _add_carrot_cake(engine, session)
        out = simulate_turn(
            engine, session,
            ScriptedTurn("that's all", intent=Intent.UNKNOWN),
        )
        assert out.state_after == ConversationState.CONFIRMING_ORDER
        return engine, session

    def test_the_code_returns_payment_not_started(self) -> None:
        """'The code' in CONFIRMING_ORDER → payment_not_started."""
        engine, session = self._session_at_confirming_order()

        out = simulate_turn(
            engine, session,
            ScriptedTurn("The code.", intent=Intent.UNKNOWN),
        )
        assert out.response_key == "payment_not_started", (
            f"Expected payment_not_started, got {out.response_key!r}"
        )

    def test_qr_code_returns_payment_not_started(self) -> None:
        engine, session = self._session_at_confirming_order()

        out = simulate_turn(
            engine, session,
            ScriptedTurn("qr code", intent=Intent.UNKNOWN),
        )
        assert out.response_key == "payment_not_started"

    def test_i_have_the_code_returns_payment_not_started(self) -> None:
        engine, session = self._session_at_confirming_order()

        out = simulate_turn(
            engine, session,
            ScriptedTurn("i have the code", intent=Intent.UNKNOWN),
        )
        assert out.response_key == "payment_not_started"


# ---------------------------------------------------------------------------
# Regression: normal paths are not affected
# ---------------------------------------------------------------------------

class TestNoRegressions:
    def test_yes_after_confirm_order_summary_proceeds(self) -> None:
        """CONFIRMING_ORDER + 'yes' (AFFIRM) → normal checkout, not payment_not_started."""
        from tests.support.voice_test_harness import StubSmsService
        checkout = StubCheckoutService()
        engine = build_engine(
            menu_repo=build_menu_repo(),
            checkout_service=checkout,
            sms_service=StubSmsService(configured=False),
        )
        session = new_session()
        simulate_turn(engine, session, ScriptedTurn("pickup"))

        _add_carrot_cake(engine, session)

        simulate_turn(engine, session, ScriptedTurn("that's all", intent=Intent.UNKNOWN))
        assert session.conversation_state == ConversationState.CONFIRMING_ORDER

        out = simulate_turn(
            engine, session,
            ScriptedTurn("yes", intent=Intent.AFFIRM),
        )
        # Should proceed to checkout flow, NOT payment_not_started.
        assert out.response_key != "payment_not_started"
        assert out.response_key != "confirm_order_summary_unclear"

    def test_cancel_order_summary_review_stays_in_confirming_order(self) -> None:
        """CONFIRMING_ORDER + 'no cancel' → not coerced to payment_not_started."""
        engine, session = _pickup_engine_session()
        _add_carrot_cake(engine, session)
        simulate_turn(engine, session, ScriptedTurn("that's all", intent=Intent.UNKNOWN))

        out = simulate_turn(
            engine, session,
            ScriptedTurn("no cancel", intent=Intent.CANCEL_ORDER),
        )
        # Should handle as cancel (stay in confirming order with clarification)
        # NOT as payment_not_started.
        assert out.response_key != "payment_not_started"

    def test_last_prompt_type_set_after_item_added(self) -> None:
        """last_prompt_type must be 'anything_else' after item_added_successfully."""
        engine, session = _pickup_engine_session()
        _add_carrot_cake(engine, session)
        assert session.last_prompt_type == "anything_else"

    def test_last_prompt_type_set_after_confirm_order_summary(self) -> None:
        """last_prompt_type must be 'confirm_order' after confirm_order_summary."""
        engine, session = _pickup_engine_session()
        _add_carrot_cake(engine, session)
        out = simulate_turn(engine, session, ScriptedTurn("that's all", intent=Intent.UNKNOWN))
        assert out.response_key == "confirm_order_summary"
        assert session.last_prompt_type == "confirm_order"

    def test_no_coercion_without_preceding_item_add(self) -> None:
        """'That's it for now' without any item in cart must NOT go to confirm_order_summary."""
        engine, session = _pickup_engine_session()
        # Empty cart — coercion must not fire.
        out = simulate_turn(
            engine, session,
            ScriptedTurn("that's it for now", intent=Intent.UNKNOWN),
        )
        assert out.response_key != "confirm_order_summary"
