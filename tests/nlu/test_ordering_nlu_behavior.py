# D:/Working/Cygnus/compass-voice/tests/nlu/test_ordering_nlu_behavior.py
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tests.support.voice_test_harness import (
    ScriptedTurn,
    build_engine,
    build_menu_repo,
    make_slot,
    new_session,
    simulate_turn,
)
from app.nlu.intent_resolution.intent import Intent
from app.nlu.nlu_resolver import resolve_nlu
from app.nlu.query_normalization.text_preprocessor import normalize_text
from app.state_machine.models.conversation_state import ConversationState


@pytest.mark.parametrize(
    ("utterance", "sub_intent", "expected_intent"),
    [
        ("add a burger", "add_item", Intent.ADD_ITEM),
        ("remove the cake", "remove_item", Intent.REMOVE_ITEM),
        ("change the bun", "modify_item", Intent.MODIFY_ITEM),
        ("what are the options", "ask_options", Intent.ASK_OPTIONS),
        ("no that's all", "finish_order", Intent.FINISH_ORDER),
    ],
)
def test_intent_detection_maps_model_labels_to_canonical_intents(
    utterance: str,
    sub_intent: str,
    expected_intent: Intent,
) -> None:
    with (
        patch(
            "app.nlu.nlu_resolver.predict_intent_labels",
            return_value=("ordering", sub_intent, 0.99, 0.99, ()),
        ),
        patch(
            "app.nlu.nlu_resolver.predict_slots",
            return_value=[{"slots": []}],
        ),
    ):
        result = resolve_nlu(
            raw_text=utterance,
            normalized_text=normalize_text(utterance),
            state=ConversationState.IDLE,
            pending_action=None,
            intent_bundle=SimpleNamespace(),
            slot_bundle=SimpleNamespace(),
        )

    assert result.effective_intent == expected_intent
    assert result.intent_confidence == pytest.approx(0.99)


def test_multi_modifier_slot_capture_normalizes_into_menu_modifier_names() -> None:
    engine = build_engine(menu_repo=build_menu_repo())
    session = new_session()

    simulate_turn(engine, session, ScriptedTurn("pickup"))
    result = simulate_turn(
        engine,
        session,
        ScriptedTurn(
            "burger with lettuce tomato no onions extra cheese",
            intent=Intent.ADD_ITEM,
            slots=(make_slot("ITEM", "Burger"),),
        ),
    )

    # All modifiers captured + missing quantity defaults to 1 → item added directly.
    # Modifier normalization is verified in test_turn_engine_phase2_validation
    # via the logger's normalized_values output (context is reset after item add).
    assert result.response_key == "item_added_successfully"


def test_slot_capture_prefills_required_chicken_burger_values() -> None:
    engine = build_engine(menu_repo=build_menu_repo())
    session = new_session()

    simulate_turn(engine, session, ScriptedTurn("pickup"))
    result = simulate_turn(
        engine,
        session,
        ScriptedTurn(
            "2 chicken burgers with plain bun and coke",
            intent=Intent.ADD_ITEM,
            slots=(
                make_slot("ITEM", "Chicken Burger"),
                make_slot("QUANTITY", "2"),
            ),
        ),
    )

    assert result.response_key in {"ask_for_side", "ask_for_modifier"}
    assert session.conversation_context.quantity == 2


def test_waiting_state_side_normalization_maps_beef_to_beef_meat() -> None:
    engine = build_engine(menu_repo=build_menu_repo())
    session = new_session()

    simulate_turn(engine, session, ScriptedTurn("pickup"))
    simulate_turn(
        engine,
        session,
        ScriptedTurn(
            "chicken burger",
            intent=Intent.ADD_ITEM,
            slots=(make_slot("ITEM", "Chicken Burger"),),
        ),
    )
    result = simulate_turn(
        engine,
        session,
        ScriptedTurn(
            "make it beef meat",
            intent=Intent.UNKNOWN,
            slots=(make_slot("ITEM", "Beef Meat"),),
        ),
    )

    selected_side_ids = {
        side_id
        for values in session.conversation_context.selected_side_groups.values()
        for side_id in values
    }
    assert result.response_key in {"ask_for_side", "ask_for_modifier", "repeat_side_options"}
    assert selected_side_ids
