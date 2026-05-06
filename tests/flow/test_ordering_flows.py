# D:/Working/Cygnus/compass-voice/tests/flow/test_ordering_flows.py
from __future__ import annotations

from app.nlu.intent_resolution.intent import Intent
from app.state_machine.models.conversation_state import ConversationState
from tests.support.voice_test_harness import (
    ScriptedTurn,
    StubCheckoutService,
    build_engine,
    build_menu_repo,
    make_slot,
    new_session,
    seed_cart_item,
    simulate_conversation,
    simulate_turn,
)


def test_full_happy_path_reaches_checkout_without_unnecessary_reprompts() -> None:
    engine = build_engine(
        menu_repo=build_menu_repo(),
        checkout_service=StubCheckoutService(),
    )
    session = new_session()

    turns = [
        ScriptedTurn("pickup"),
        ScriptedTurn(
            "burger with lettuce tomato no onions extra cheese",
            intent=Intent.ADD_ITEM,
            slots=(make_slot("ITEM", "Burger"),),
        ),
        ScriptedTurn("1", intent=Intent.UNKNOWN, slots=(make_slot("QUANTITY", "1"),)),
        ScriptedTurn(
            "chocolate cake",
            intent=Intent.ADD_ITEM,
            slots=(make_slot("ITEM", "Chocolate Cake"),),
        ),
        ScriptedTurn("1", intent=Intent.UNKNOWN, slots=(make_slot("QUANTITY", "1"),)),
        ScriptedTurn("checkout", intent=Intent.CHECKOUT),
        ScriptedTurn("go ahead", intent=Intent.CONFIRM),
    ]

    results = simulate_conversation(engine, session, turns)

    # Pickup flow: after order confirmation, asks SMS permission (not live payment wait).
    assert [turn.response_key for turn in results][-2:] == ["confirm_order_summary", "pickup_ask_sms_permission"]
    assert session.conversation_state == ConversationState.WAITING_FOR_PICKUP_SMS_PERMISSION
    assert session.reprompt_count_by_field.get("quantity", 0) == 0


def test_partial_input_only_prompts_for_missing_fields() -> None:
    engine = build_engine(menu_repo=build_menu_repo())
    session = new_session()

    simulate_turn(engine, session, ScriptedTurn("pickup"))
    result = simulate_turn(
        engine,
        session,
        ScriptedTurn(
            "burger with lettuce",
            intent=Intent.ADD_ITEM,
            slots=(make_slot("ITEM", "Burger"),),
        ),
    )

    assert result.response_key == "item_added_successfully"
    assert session.conversation_state == ConversationState.IDLE


def test_remove_item_correction_updates_cart_state() -> None:
    engine = build_engine(menu_repo=build_menu_repo())
    session = new_session(state=ConversationState.IDLE, order_type="pickup")
    seed_cart_item(session, item_id="burger")
    seed_cart_item(session, item_id="Chocolate Cake")

    first = simulate_turn(
        engine,
        session,
        ScriptedTurn(
            "remove the cake",
            intent=Intent.REMOVE_ITEM,
            slots=(make_slot("ITEM", "Chocolate Cake"),),
        ),
    )
    second = simulate_turn(engine, session, ScriptedTurn("yes", intent=Intent.CONFIRM))

    assert first.response_key == "confirm_remove_item"
    assert second.response_key == "item_removed_successfully"
    assert len(session.cart.get_items()) == 1


def test_modify_item_request_routes_into_modify_flow() -> None:
    engine = build_engine(menu_repo=build_menu_repo())
    session = new_session(state=ConversationState.IDLE, order_type="pickup")
    seed_cart_item(session, item_id="burger")

    result = simulate_turn(
        engine,
        session,
        ScriptedTurn(
            "change bun to sesame",
            intent=Intent.MODIFY_ITEM,
            slots=(make_slot("ITEM", "Burger"),),
        ),
    )

    assert result.response_key == "confirm_modify_item"
    assert session.conversation_state == ConversationState.MODIFYING_ITEM


def test_invalid_side_input_retries_without_corrupting_state() -> None:
    engine = build_engine(menu_repo=build_menu_repo())
    session = new_session()

    simulate_turn(engine, session, ScriptedTurn("pickup"))
    simulate_turn(
        engine,
        session,
        ScriptedTurn(
            "chicken taco",
            intent=Intent.ADD_ITEM,
            slots=(make_slot("ITEM", "Chicken Taco"),),
        ),
    )
    retry = simulate_turn(engine, session, ScriptedTurn("pineapple", intent=Intent.UNKNOWN))

    assert retry.response_key in {"repeat_side_options", "list_side_options"}
    assert session.conversation_state == ConversationState.WAITING_FOR_SIDE


def test_interruption_total_question_preserves_modifier_flow_context() -> None:
    engine = build_engine(menu_repo=build_menu_repo())
    session = new_session()

    simulate_turn(engine, session, ScriptedTurn("pickup"))
    simulate_turn(
        engine,
        session,
        ScriptedTurn(
            "bourbon chicken",
            intent=Intent.ADD_ITEM,
            slots=(make_slot("ITEM", "Bourbon Chicken"),),
        ),
    )
    result = simulate_turn(
        engine,
        session,
        ScriptedTurn("what's my total", intent=Intent.SHOW_TOTAL),
    )

    assert result.response_key == "readonly_interrupt_with_resume"
    assert result.response_payload["interrupt_response_key"] == "show_total"
    assert session.conversation_state == ConversationState.WAITING_FOR_MODIFIER
