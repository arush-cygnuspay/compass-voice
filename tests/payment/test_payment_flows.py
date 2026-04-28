# D:/Working/Cygnus/compass-voice/tests/payment/test_payment_flows.py
from __future__ import annotations

import time

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


def _delivery_confirmation_session():
    session = new_session(
        state=ConversationState.CONFIRMING_ORDER,
        caller_device_type="chat",
        order_type="delivery",
    )
    session.conversation_context.delivery_address.area = "Downtown"
    session.conversation_context.delivery_address.postal_code = "12345"
    seed_cart_item(session, item_id="burger")
    return session


def test_checkout_link_sent_transitions_to_checkout_wait_state() -> None:
    engine = build_engine(
        menu_repo=build_menu_repo(),
        sms_service=StubSmsService(configured=True),
        checkout_service=StubCheckoutService(),
    )
    session = _delivery_confirmation_session()

    result = simulate_turn(engine, session, ScriptedTurn("go ahead", intent=Intent.CONFIRM))

    assert result.response_key == "checkout_link_sent"
    assert session.conversation_state == ConversationState.WAITING_FOR_CHECKOUT_COMPLETION


def test_stay_on_call_mode_keeps_wait_state_and_allows_polling() -> None:
    checkout = StubCheckoutService()
    engine = build_engine(
        menu_repo=build_menu_repo(),
        sms_service=StubSmsService(configured=True),
        checkout_service=checkout,
    )
    session = _delivery_confirmation_session()
    simulate_turn(engine, session, ScriptedTurn("go ahead", intent=Intent.CONFIRM))

    result = simulate_turn(engine, session, ScriptedTurn("stay on the line", intent=Intent.UNKNOWN))

    assert result.response_key == "checkout_wait_stay_on_call"
    assert session.conversation_state == ConversationState.WAITING_FOR_CHECKOUT_COMPLETION


def test_after_call_mode_suppresses_repeat_reminders() -> None:
    checkout = StubCheckoutService()
    engine = build_engine(
        menu_repo=build_menu_repo(),
        sms_service=StubSmsService(configured=True),
        checkout_service=checkout,
    )
    session = _delivery_confirmation_session()
    simulate_turn(engine, session, ScriptedTurn("go ahead", intent=Intent.CONFIRM))
    simulate_turn(engine, session, ScriptedTurn("I'll complete it after the call", intent=Intent.UNKNOWN))

    result = simulate_turn(engine, session, ScriptedTurn("not yet", intent=Intent.UNKNOWN))

    assert result.response_key == "checkout_after_call_selected"
    assert session.conversation_state == ConversationState.WAITING_FOR_CHECKOUT_COMPLETION


def test_resend_link_respects_cooldown() -> None:
    checkout = StubCheckoutService()
    engine = build_engine(
        menu_repo=build_menu_repo(),
        sms_service=StubSmsService(configured=True),
        checkout_service=checkout,
    )
    session = _delivery_confirmation_session()
    simulate_turn(engine, session, ScriptedTurn("go ahead", intent=Intent.CONFIRM))

    first = simulate_turn(engine, session, ScriptedTurn("resend the link", intent=Intent.PAYMENT_REQUEST))
    second = simulate_turn(engine, session, ScriptedTurn("resend the link", intent=Intent.PAYMENT_REQUEST))

    assert first.response_key == "checkout_link_resent"
    assert second.response_key == "checkout_link_resend_cooldown"


def test_payment_confirmation_completes_once_and_stops_future_prompts() -> None:
    checkout = StubCheckoutService()
    checkout.queue_verify_results(
        "1234567",
        {
            "ok": True,
            "paid": True,
            "payment_completed": True,
            "status": "completed",
            "reference": "ref-1",
            "session": None,
            "error": None,
        },
    )
    engine = build_engine(
        menu_repo=build_menu_repo(),
        sms_service=StubSmsService(configured=True),
        checkout_service=checkout,
    )
    session = new_session(state=ConversationState.WAITING_FOR_PAYMENT, order_type="pickup")
    session.conversation_context.delivery_address.order_number = "1234567"

    confirmed = simulate_turn(engine, session, ScriptedTurn("i paid", intent=Intent.PAYMENT_DONE))
    follow_up = engine.process_turn(session=session, user_text="__auto_payment_check__")

    assert confirmed.response_key == "order_completed"
    assert session.conversation_state == ConversationState.COMPLETED
    assert follow_up.response_key == "order_completed"


def test_payment_reminder_interval_avoids_spam_loops() -> None:
    checkout = StubCheckoutService()
    engine = build_engine(
        menu_repo=build_menu_repo(),
        sms_service=StubSmsService(configured=True),
        checkout_service=checkout,
    )
    session = new_session(state=ConversationState.WAITING_FOR_CHECKOUT_COMPLETION, order_type="delivery")
    session.conversation_context.delivery_address.order_number = "1234567"
    session.last_response_key = "waiting_for_checkout_completion"
    session.last_response_at_epoch = time.time()
    session.conversation_context.delivery_address.payment_status_last_prompt_at_epoch = session.last_response_at_epoch
    session.conversation_context.delivery_address.payment_status_last_response_key = "waiting_for_checkout_completion"

    output = engine.process_turn(session=session, user_text="__auto_payment_check__")

    assert output.spoken_response_text == ""
