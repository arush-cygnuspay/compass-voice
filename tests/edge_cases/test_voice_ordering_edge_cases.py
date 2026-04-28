# D:/Working/Cygnus/compass-voice/tests/edge_cases/test_voice_ordering_edge_cases.py
from __future__ import annotations

from app.nlu.intent_resolution.intent import Intent
from app.state_machine.models.conversation_state import ConversationState
from tests.support.voice_test_harness import (
    ScriptedTurn,
    build_engine,
    build_menu_repo,
    make_slot,
    new_session,
    simulate_turn,
)


def test_repeated_invalid_inputs_trigger_fallback_logic_without_stuck_state() -> None:
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

    simulate_turn(engine, session, ScriptedTurn("huh", intent=Intent.UNKNOWN))
    simulate_turn(engine, session, ScriptedTurn("whatever", intent=Intent.UNKNOWN))
    result = simulate_turn(engine, session, ScriptedTurn("still not sure", intent=Intent.UNKNOWN))

    assert result.response_key in {"list_side_options", "fallback_escalation_offer"}
    assert session.conversation_state == ConversationState.WAITING_FOR_SIDE
    assert session.fallback_count >= 3


def test_unclear_input_keeps_flow_recoverable() -> None:
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
    result = simulate_turn(engine, session, ScriptedTurn("...", intent=Intent.UNKNOWN))

    assert result.response_key in {"repeat_modifier_options", "list_modifier_options"}
    assert session.conversation_state == ConversationState.WAITING_FOR_MODIFIER


def test_agent_handoff_can_be_requested_from_active_flow() -> None:
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
    result = simulate_turn(engine, session, ScriptedTurn("connect me to a person", intent=Intent.UNKNOWN))

    assert result.response_key == "transferring_to_human_agent"


def test_mixed_intent_sentence_prefers_cart_edit_routing() -> None:
    engine = build_engine(menu_repo=build_menu_repo())
    session = new_session(state=ConversationState.IDLE, order_type="pickup")
    result = simulate_turn(
        engine,
        session,
        ScriptedTurn(
            "remove the cake and add fries",
            intent=Intent.REMOVE_ITEM,
            slots=(make_slot("ITEM", "Chocolate Cake"),),
        ),
    )

    assert result.response_key in {"cart_is_empty", "item_not_found_in_cart", "confirm_remove_item"}


def test_start_over_interrupt_keeps_system_recoverable() -> None:
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
    result = simulate_turn(engine, session, ScriptedTurn("start over", intent=Intent.UNKNOWN))

    assert result.response_key in {"current_item_restarted", "flow_guard_confirm_cancel", "confirm_cancel_current_item"}
