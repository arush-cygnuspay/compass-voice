# tests/payment/test_pickup_checkout_flow.py
"""
Regression tests for the pickup-specific checkout flow.

Spec:
1.  Pickup order confirmed → order is placed (payment link created).
2.  System offers to text a payment link OR let customer pay on arrival
    (pickup_ask_sms_permission).
3.  User accepts SMS ("yes", "send it", "text me the link") →
    SMS/payment link command is emitted → call ends (COMPLETED).
4.  User declines SMS ("no", "no thanks") or says "pay on pickup" / "pay at the counter" →
    no SMS → call ends with pay-at-pickup message (COMPLETED).
5.  No phone number → skip SMS question, end call directly (COMPLETED).
6.  Pickup flow does NOT enter WAITING_FOR_PAYMENT.
7.  Pickup flow does NOT start payment auto-check (no __auto_payment_check__ loop).
8.  Delivery checkout flow is unchanged (still uses WAITING_FOR_CHECKOUT_COMPLETION).
"""
from __future__ import annotations

from app.nlu.intent_resolution.intent import Intent
from app.state_machine.models.conversation_state import ConversationState
from tests.support.voice_test_harness import (
    ScriptedTurn,
    StubCheckoutService,
    StubSmsService,
    build_engine,
    build_menu_repo,
    new_session,
    seed_cart_item,
    simulate_turn,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pickup_confirm_session(*, has_phone: bool = True):
    session = new_session(
        state=ConversationState.CONFIRMING_ORDER,
        order_type="pickup",
    )
    if not has_phone:
        session.conversation_context.delivery_address.customer_phone_number = None
    seed_cart_item(session, item_id="burger")
    return session


def _delivery_confirm_session():
    session = new_session(
        state=ConversationState.CONFIRMING_ORDER,
        caller_device_type="chat",
        order_type="delivery",
    )
    session.conversation_context.delivery_address.area = "Downtown"
    session.conversation_context.delivery_address.postal_code = "12345"
    seed_cart_item(session, item_id="burger")
    return session


# ---------------------------------------------------------------------------
# 1. Order placed → SMS permission asked
# ---------------------------------------------------------------------------

def test_pickup_confirm_asks_sms_permission() -> None:
    engine = build_engine(
        menu_repo=build_menu_repo(),
        sms_service=StubSmsService(configured=True),
        checkout_service=StubCheckoutService(),
    )
    session = _pickup_confirm_session()

    result = simulate_turn(engine, session, ScriptedTurn("yes please", intent=Intent.CONFIRM))

    assert result.response_key == "pickup_ask_sms_permission"
    assert session.conversation_state == ConversationState.WAITING_FOR_PICKUP_SMS_PERMISSION


def test_pickup_confirm_does_not_enter_waiting_for_payment() -> None:
    engine = build_engine(
        menu_repo=build_menu_repo(),
        sms_service=StubSmsService(configured=True),
        checkout_service=StubCheckoutService(),
    )
    session = _pickup_confirm_session()

    simulate_turn(engine, session, ScriptedTurn("yes", intent=Intent.CONFIRM))

    assert session.conversation_state != ConversationState.WAITING_FOR_PAYMENT


# ---------------------------------------------------------------------------
# 2 & 3. User accepts SMS → link sent → call ends
# ---------------------------------------------------------------------------

def test_pickup_sms_accepted_emits_send_sms_command_and_ends_call() -> None:
    sms = StubSmsService(configured=True)
    engine = build_engine(
        menu_repo=build_menu_repo(),
        sms_service=sms,
        checkout_service=StubCheckoutService(),
    )
    session = _pickup_confirm_session()
    simulate_turn(engine, session, ScriptedTurn("yes please", intent=Intent.CONFIRM))

    result = simulate_turn(engine, session, ScriptedTurn("yes", intent=Intent.CONFIRM))

    assert result.response_key == "pickup_sms_sent_end_call"
    # COMPLETED state causes end_call_after_playback=True in TurnEngine automatically.
    assert session.conversation_state == ConversationState.COMPLETED


def test_pickup_sms_accepted_actually_sends_sms() -> None:
    sms = StubSmsService(configured=True)
    engine = build_engine(
        menu_repo=build_menu_repo(),
        sms_service=sms,
        checkout_service=StubCheckoutService(),
    )
    session = _pickup_confirm_session()
    simulate_turn(engine, session, ScriptedTurn("yes", intent=Intent.CONFIRM))
    simulate_turn(engine, session, ScriptedTurn("yes", intent=Intent.CONFIRM))

    assert len(sms.requests) == 1


# ---------------------------------------------------------------------------
# 4. User declines SMS → no SMS → call ends
# ---------------------------------------------------------------------------

def test_pickup_sms_declined_ends_call_without_sms() -> None:
    sms = StubSmsService(configured=True)
    engine = build_engine(
        menu_repo=build_menu_repo(),
        sms_service=sms,
        checkout_service=StubCheckoutService(),
    )
    session = _pickup_confirm_session()
    simulate_turn(engine, session, ScriptedTurn("yes", intent=Intent.CONFIRM))

    result = simulate_turn(engine, session, ScriptedTurn("no thanks", intent=Intent.UNKNOWN))

    assert result.response_key == "pickup_no_sms_end_call"
    # COMPLETED state causes end_call_after_playback=True in TurnEngine automatically.
    assert session.conversation_state == ConversationState.COMPLETED
    assert len(sms.requests) == 0


# ---------------------------------------------------------------------------
# 5. No phone number → skip SMS question, end call directly
# ---------------------------------------------------------------------------

def test_pickup_no_phone_skips_sms_question_and_ends_call() -> None:
    engine = build_engine(
        menu_repo=build_menu_repo(),
        sms_service=StubSmsService(configured=True),
        checkout_service=StubCheckoutService(),
    )
    session = _pickup_confirm_session(has_phone=False)

    result = simulate_turn(engine, session, ScriptedTurn("yes", intent=Intent.CONFIRM))

    assert result.response_key == "pickup_end_call"
    # COMPLETED state causes end_call_after_playback=True in TurnEngine automatically.
    assert session.conversation_state == ConversationState.COMPLETED


# ---------------------------------------------------------------------------
# 6. Re-prompt on unclear input in SMS-permission state
# ---------------------------------------------------------------------------

def test_pickup_unclear_input_re_asks_sms_permission() -> None:
    engine = build_engine(
        menu_repo=build_menu_repo(),
        sms_service=StubSmsService(configured=True),
        checkout_service=StubCheckoutService(),
    )
    session = _pickup_confirm_session()
    simulate_turn(engine, session, ScriptedTurn("yes", intent=Intent.CONFIRM))

    result = simulate_turn(engine, session, ScriptedTurn("hmm not sure", intent=Intent.UNKNOWN))

    assert result.response_key == "pickup_repeat_sms_permission"
    assert session.conversation_state == ConversationState.WAITING_FOR_PICKUP_SMS_PERMISSION


# ---------------------------------------------------------------------------
# 7. No payment auto-check loop for pickup
# ---------------------------------------------------------------------------

def test_pickup_auto_payment_check_is_silent_noop_after_sms_sent() -> None:
    engine = build_engine(
        menu_repo=build_menu_repo(),
        sms_service=StubSmsService(configured=True),
        checkout_service=StubCheckoutService(),
    )
    session = _pickup_confirm_session()
    simulate_turn(engine, session, ScriptedTurn("yes", intent=Intent.CONFIRM))
    simulate_turn(engine, session, ScriptedTurn("yes", intent=Intent.CONFIRM))

    # Session is COMPLETED — auto-check sentinel must not trigger a live
    # payment reminder or loop back into any waiting state.
    output = engine.process_turn(session=session, user_text="__auto_payment_check__")

    assert session.conversation_state == ConversationState.COMPLETED
    assert output.response_key != "waiting_for_payment"
    assert output.response_key != "payment_not_confirmed_yet"


def test_pickup_in_sms_permission_state_never_receives_payment_reminder() -> None:
    engine = build_engine(
        menu_repo=build_menu_repo(),
        sms_service=StubSmsService(configured=True),
        checkout_service=StubCheckoutService(),
    )
    session = _pickup_confirm_session()
    simulate_turn(engine, session, ScriptedTurn("yes", intent=Intent.CONFIRM))

    # Fire the auto-check sentinel while still in SMS-permission state.
    output = engine.process_turn(session=session, user_text="__auto_payment_check__")

    assert output.response_key not in {"waiting_for_payment", "payment_not_confirmed_yet"}
    assert session.conversation_state == ConversationState.WAITING_FOR_PICKUP_SMS_PERMISSION


# ---------------------------------------------------------------------------
# 8. Delivery flow unchanged
# ---------------------------------------------------------------------------

def test_delivery_confirm_still_uses_checkout_completion_state() -> None:
    engine = build_engine(
        menu_repo=build_menu_repo(),
        sms_service=StubSmsService(configured=True),
        checkout_service=StubCheckoutService(),
    )
    session = _delivery_confirm_session()

    result = simulate_turn(engine, session, ScriptedTurn("go ahead", intent=Intent.CONFIRM))

    assert result.response_key == "checkout_link_sent"
    assert session.conversation_state == ConversationState.WAITING_FOR_CHECKOUT_COMPLETION


def test_delivery_confirm_does_not_ask_sms_permission() -> None:
    engine = build_engine(
        menu_repo=build_menu_repo(),
        sms_service=StubSmsService(configured=True),
        checkout_service=StubCheckoutService(),
    )
    session = _delivery_confirm_session()

    result = simulate_turn(engine, session, ScriptedTurn("confirm", intent=Intent.CONFIRM))

    assert result.response_key != "pickup_ask_sms_permission"
    assert session.conversation_state != ConversationState.WAITING_FOR_PICKUP_SMS_PERMISSION


# ---------------------------------------------------------------------------
# 9. Natural "pay on pickup" phrases → no SMS (phrase detection)
# ---------------------------------------------------------------------------

def test_pickup_pay_at_counter_phrase_ends_call_without_sms() -> None:
    sms = StubSmsService(configured=True)
    engine = build_engine(
        menu_repo=build_menu_repo(),
        sms_service=sms,
        checkout_service=StubCheckoutService(),
    )
    session = _pickup_confirm_session()
    simulate_turn(engine, session, ScriptedTurn("yes", intent=Intent.CONFIRM))

    result = simulate_turn(
        engine, session, ScriptedTurn("I'll pay at the counter", intent=Intent.UNKNOWN)
    )

    assert result.response_key == "pickup_no_sms_end_call"
    assert session.conversation_state == ConversationState.COMPLETED
    assert len(sms.requests) == 0


def test_pickup_pay_when_i_arrive_phrase_ends_call_without_sms() -> None:
    sms = StubSmsService(configured=True)
    engine = build_engine(
        menu_repo=build_menu_repo(),
        sms_service=sms,
        checkout_service=StubCheckoutService(),
    )
    session = _pickup_confirm_session()
    simulate_turn(engine, session, ScriptedTurn("yes", intent=Intent.CONFIRM))

    result = simulate_turn(
        engine, session, ScriptedTurn("I'll pay when I arrive", intent=Intent.UNKNOWN)
    )

    assert result.response_key == "pickup_no_sms_end_call"
    assert session.conversation_state == ConversationState.COMPLETED
    assert len(sms.requests) == 0


def test_pickup_pay_on_pickup_phrase_beats_ambiguous_affirm() -> None:
    """'Yes I'll pay on pickup' contains a pay-on-pickup signal — SMS must NOT be sent."""
    sms = StubSmsService(configured=True)
    engine = build_engine(
        menu_repo=build_menu_repo(),
        sms_service=sms,
        checkout_service=StubCheckoutService(),
    )
    session = _pickup_confirm_session()
    simulate_turn(engine, session, ScriptedTurn("yes", intent=Intent.CONFIRM))

    result = simulate_turn(
        engine, session, ScriptedTurn("yes I'll pay on pickup", intent=Intent.CONFIRM)
    )

    assert result.response_key == "pickup_no_sms_end_call"
    assert len(sms.requests) == 0


# ---------------------------------------------------------------------------
# 10. Natural "send it / text me the link" phrases → SMS sent
# ---------------------------------------------------------------------------

def test_pickup_send_it_phrase_sends_sms() -> None:
    sms = StubSmsService(configured=True)
    engine = build_engine(
        menu_repo=build_menu_repo(),
        sms_service=sms,
        checkout_service=StubCheckoutService(),
    )
    session = _pickup_confirm_session()
    simulate_turn(engine, session, ScriptedTurn("yes", intent=Intent.CONFIRM))

    result = simulate_turn(
        engine, session, ScriptedTurn("send it", intent=Intent.UNKNOWN)
    )

    assert result.response_key == "pickup_sms_sent_end_call"
    assert session.conversation_state == ConversationState.COMPLETED
    assert len(sms.requests) == 1


def test_pickup_text_me_the_link_phrase_sends_sms() -> None:
    sms = StubSmsService(configured=True)
    engine = build_engine(
        menu_repo=build_menu_repo(),
        sms_service=sms,
        checkout_service=StubCheckoutService(),
    )
    session = _pickup_confirm_session()
    simulate_turn(engine, session, ScriptedTurn("yes", intent=Intent.CONFIRM))

    result = simulate_turn(
        engine, session, ScriptedTurn("text me the link", intent=Intent.UNKNOWN)
    )

    assert result.response_key == "pickup_sms_sent_end_call"
    assert session.conversation_state == ConversationState.COMPLETED
    assert len(sms.requests) == 1
