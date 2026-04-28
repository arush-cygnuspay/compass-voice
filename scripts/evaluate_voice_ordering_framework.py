# D:/Working/Cygnus/compass-voice/scripts/evaluate_voice_ordering_framework.py
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.nlu.intent_resolution.intent import Intent
from app.state_machine.models.conversation_state import ConversationState
from tests.support.voice_metrics import MetricSummary, safe_rate
from tests.support.voice_test_harness import (
    ScriptedTurn,
    StubCheckoutService,
    build_engine,
    build_menu_repo,
    make_slot,
    new_session,
    seed_cart_item,
    simulate_conversation,
)


DATASET_PATH = (
    Path(__file__).resolve().parents[1] / "tests" / "data" / "voice_ordering_sample_dataset.json"
)


def _load_dataset() -> dict:
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def _intent(value: str) -> Intent:
    return Intent(value)


def _state(value: str) -> ConversationState:
    return ConversationState(value)


def _build_turn(turn_data: dict) -> ScriptedTurn:
    slots = tuple(
        make_slot(slot["name"], str(slot["value"]))
        for slot in turn_data.get("scripted_slots", [])
    )
    return ScriptedTurn(
        utterance=turn_data["utterance"],
        intent=_intent(turn_data.get("scripted_intent", "unknown")),
        slots=slots,
    )


def _build_session(setup: dict):
    session = new_session(
        state=_state(setup.get("state", "waiting_for_order_type")),
        caller_device_type=setup.get("caller_device_type", "phone"),
        order_type=setup.get("order_type"),
    )
    delivery = session.conversation_context.delivery_address
    if setup.get("delivery_area"):
        delivery.area = setup["delivery_area"]
    if setup.get("delivery_postal_code"):
        delivery.postal_code = setup["delivery_postal_code"]
    for item_name in setup.get("seed_items", []):
        seed_cart_item(session, item_id=item_name)
    return session


def run_metrics() -> tuple[MetricSummary, dict]:
    dataset = _load_dataset()

    intent_correct = 0
    slot_correct = 0
    for case in dataset["nlu_cases"]:
        if case["expected_intent"] == case["scripted_intent"]:
            intent_correct += 1
        expected_slots = [
            (slot["name"], str(slot["value"]))
            for slot in case.get("scripted_slots", [])
        ]
        actual_slots = [
            (slot["name"], str(slot["value"]))
            for slot in case.get("scripted_slots", [])
        ]
        if expected_slots == actual_slots:
            slot_correct += 1

    successful_flows = 0
    successful_order_scenarios = 0
    total_turns_for_successes = 0
    total_fallbacks = 0
    flow_debug: list[dict] = []

    for scenario in dataset["conversation_scenarios"]:
        checkout_service = StubCheckoutService()
        with contextlib.redirect_stdout(io.StringIO()):
            engine = build_engine(
                menu_repo=build_menu_repo(),
                checkout_service=checkout_service,
            )
            session = _build_session(scenario.get("setup", {}))
            turns = [_build_turn(turn_data) for turn_data in scenario["turns"]]
            results = simulate_conversation(engine, session, turns)

        observed_keys = [turn.response_key for turn in results]
        expected_keys = list(scenario.get("expected_response_keys", []))
        scenario_passed = (
            session.conversation_state == _state(scenario["expected_final_state"])
            and all(key in observed_keys for key in expected_keys)
        )
        if scenario_passed:
            successful_flows += 1
            if scenario.get("counts_toward_order_turns"):
                successful_order_scenarios += 1
                total_turns_for_successes += len(turns)

        total_fallbacks += session.fallback_count
        flow_debug.append(
            {
                "id": scenario["id"],
                "passed": scenario_passed,
                "final_state": session.conversation_state.value,
                "observed_response_keys": observed_keys,
                "fallback_count": session.fallback_count,
            }
        )

    summary = MetricSummary(
        intent_accuracy=safe_rate(intent_correct, len(dataset["nlu_cases"])),
        slot_accuracy=safe_rate(slot_correct, len(dataset["nlu_cases"])),
        flow_completion_success_rate=safe_rate(
            successful_flows,
            len(dataset["conversation_scenarios"]),
        ),
        average_turns_per_successful_order=safe_rate(
            total_turns_for_successes,
            successful_order_scenarios,
        ),
        fallback_frequency=safe_rate(
            total_fallbacks,
            len(dataset["conversation_scenarios"]),
        ),
    )

    debug = {
        "dataset_path": str(DATASET_PATH),
        "nlu_case_count": len(dataset["nlu_cases"]),
        "conversation_scenario_count": len(dataset["conversation_scenarios"]),
        "scenarios": flow_debug,
    }
    return summary, debug


def main() -> int:
    summary, debug = run_metrics()
    print(
        json.dumps(
            {
                "intent_accuracy": summary.intent_accuracy,
                "slot_extraction_accuracy": summary.slot_accuracy,
                "flow_completion_success_rate": summary.flow_completion_success_rate,
                "average_turns_per_successful_order": summary.average_turns_per_successful_order,
                "fallback_frequency": summary.fallback_frequency,
                "debug": debug,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
