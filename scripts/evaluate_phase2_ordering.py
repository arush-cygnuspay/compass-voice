from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from types import SimpleNamespace
from unittest.mock import patch
import sys
import types

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _install_runtime_stubs() -> None:
    twilio_module = types.ModuleType("twilio")
    twilio_base_module = types.ModuleType("twilio.base")
    twilio_base_exceptions_module = types.ModuleType("twilio.base.exceptions")
    twilio_rest_module = types.ModuleType("twilio.rest")

    class _TwilioRestException(Exception):
        pass

    class _TwilioClient:
        def __init__(self, *args, **kwargs):
            pass

    twilio_base_exceptions_module.TwilioRestException = _TwilioRestException
    twilio_rest_module.Client = _TwilioClient

    sys.modules.setdefault("twilio", twilio_module)
    sys.modules.setdefault("twilio.base", twilio_base_module)
    sys.modules.setdefault("twilio.base.exceptions", twilio_base_exceptions_module)
    sys.modules.setdefault("twilio.rest", twilio_rest_module)

    redis_module = types.ModuleType("redis")

    class _RedisClient:
        def __init__(self, *args, **kwargs):
            pass

    redis_module.Redis = _RedisClient
    sys.modules.setdefault("redis", redis_module)

    intent_inference_module = types.ModuleType("app.ml.intent.inference_intent")
    slot_inference_module = types.ModuleType("app.ml.slot.inference_slot")

    class _IntentBundle:
        pass

    class _SlotBundle:
        pass

    intent_inference_module.IntentBundle = _IntentBundle
    intent_inference_module.predict_intent = lambda *args, **kwargs: []
    slot_inference_module.SlotBundle = _SlotBundle
    slot_inference_module.predict_slots = lambda *args, **kwargs: []
    sys.modules.setdefault("app.ml.intent.inference_intent", intent_inference_module)
    sys.modules.setdefault("app.ml.slot.inference_slot", slot_inference_module)


_install_runtime_stubs()

from app.core.response_builder import ResponseBuilder
from app.core.turn_engine import TurnEngine
from app.menu.repository import MenuRepository
from app.menu.store import MenuStore
from app.nlu.intent_resolution.intent import Intent
from app.nlu.nlu_result import SlotValue
from app.nlu.query_normalization.text_preprocessor import normalize_text
from app.session.session import Session
from app.state_machine.models.conversation_state import ConversationState
from app.state_machine.state_router import StateRouter


class StubSmsService:
    def is_configured(self):
        return False

    def send(self, request):
        return SimpleNamespace(ok=False, sid=None, error_code="not_configured", error_message="not configured")


class CapturingLogger:
    def __init__(self) -> None:
        self.enabled = True
        self.rows: list[dict] = []

    def log_turn(self, **kwargs) -> None:
        self.rows.append(kwargs)


def _build_menu_repo(repo_root: Path) -> MenuRepository:
    data_root = repo_root / "app" / "data" / "restaurants" / "demo"
    store = MenuStore(
        menu_path=data_root / "menu.json",
        entity_index_path=data_root / "entity_index.json",
    )
    return MenuRepository(store)


def _build_engine(menu_repo: MenuRepository, logger: CapturingLogger) -> TurnEngine:
    return TurnEngine(
        router=StateRouter(),
        menu_repo=menu_repo,
        intent_bundle=None,
        slot_bundle=None,
        responder=ResponseBuilder(menu_repo),
        sms_service=StubSmsService(),
        nlu_logger=logger,
    )


def _make_nlu(text: str, intent: Intent, slots: tuple[SlotValue, ...]) -> SimpleNamespace:
    return SimpleNamespace(
        effective_intent=intent,
        intent_confidence=0.99,
        raw_text=text,
        normalized_text=normalize_text(text),
        slots=slots,
        model_main_intent=intent.value,
        model_sub_intent=intent.value,
        slot_model_ran=bool(slots),
        intent_model_ms=None,
        slot_model_ms=None,
    )


def _slot_from_dict(raw: dict) -> SlotValue:
    return SlotValue(
        name=str(raw.get("name", "")),
        value=raw.get("value"),
        raw=raw.get("raw"),
        start=raw.get("start"),
        end=raw.get("end"),
        confidence=raw.get("confidence"),
    )


def _intent_from_name(name: str | None) -> Intent:
    if not name:
        return Intent.UNKNOWN
    return Intent(name)


def _new_session(*, caller_device_type: str) -> Session:
    session = Session(session_id="phase2-eval", restaurant_id="demo")
    session.conversation_state = ConversationState.WAITING_FOR_ORDER_TYPE
    session.conversation_context.caller_device_type = caller_device_type
    return session


def _run_injected_turn(
    engine: TurnEngine,
    session: Session,
    *,
    text: str,
    intent: Intent,
    slots: tuple[SlotValue, ...],
):
    fake_nlu = _make_nlu(text, intent, slots)
    with patch("app.core.turn_engine.resolve_nlu", return_value=fake_nlu):
        return engine.process_turn(session=session, user_text=text)


def _slot_signature(slot: dict | SlotValue) -> tuple[str, str]:
    if isinstance(slot, dict):
        return (
            str(slot.get("name", "")).upper(),
            normalize_text(str(slot.get("value", "") or "")),
        )
    return (
        str(getattr(slot, "name", "")).upper(),
        normalize_text(str(getattr(slot, "value", "") or "")),
    )


def _collect_latency_values(rows: list[dict], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return values


def _evaluate_scenario(menu_repo: MenuRepository, scenario: dict) -> dict:
    logger = CapturingLogger()
    engine = _build_engine(menu_repo, logger)
    session = _new_session(caller_device_type=str(scenario.get("caller_device_type") or "phone"))

    steps: list[dict] = []
    passed = True
    intent_total = 0
    intent_correct = 0
    slot_turn_total = 0
    slot_turn_correct = 0
    slot_pair_expected = 0
    slot_pair_matched = 0

    for turn in scenario.get("turns", []):
        text = str(turn["text"])
        expected_response_key = str(turn.get("expected_response_key") or "")
        expected_intent_name = turn.get("intent")
        expected_slots_raw = turn.get("slots", [])
        expected_slots = tuple(_slot_from_dict(slot) for slot in expected_slots_raw)

        if expected_intent_name is None:
            output = engine.process_turn(session=session, user_text=text)
        else:
            output = _run_injected_turn(
                engine,
                session,
                text=text,
                intent=_intent_from_name(expected_intent_name),
                slots=expected_slots,
            )

        row = logger.rows[-1] if logger.rows else {}
        matched = output.response_key == expected_response_key
        if not matched:
            passed = False

        if expected_intent_name is not None:
            intent_total += 1
            if row.get("pred_intent") == expected_intent_name:
                intent_correct += 1

            expected_slot_signatures = {_slot_signature(slot) for slot in expected_slots_raw}
            actual_slot_signatures = {_slot_signature(slot) for slot in row.get("slots", [])}
            slot_turn_total += 1
            if actual_slot_signatures == expected_slot_signatures:
                slot_turn_correct += 1
            slot_pair_expected += len(expected_slot_signatures)
            slot_pair_matched += len(expected_slot_signatures & actual_slot_signatures)

        steps.append(
            {
                "text": text,
                "expected_response_key": expected_response_key,
                "actual_response_key": output.response_key,
                "expected_intent": expected_intent_name,
                "actual_intent": row.get("pred_intent"),
                "expected_slots": expected_slots_raw,
                "actual_slots": [
                    {
                        "name": getattr(slot, "name", ""),
                        "value": getattr(slot, "value", None),
                    }
                    for slot in row.get("slots", [])
                ],
                "state_after": session.conversation_state.value,
                "passed": matched,
            }
        )

    completable = bool(scenario.get("expect_completed"))
    completed = completable and session.last_response_key == "item_added_successfully"

    return {
        "id": scenario.get("id"),
        "description": scenario.get("description"),
        "passed": passed,
        "completed": completed,
        "completable": completable,
        "turn_count": len(steps),
        "intent_total": intent_total,
        "intent_correct": intent_correct,
        "slot_turn_total": slot_turn_total,
        "slot_turn_correct": slot_turn_correct,
        "slot_pair_expected": slot_pair_expected,
        "slot_pair_matched": slot_pair_matched,
        "session_metrics": {
            "fallback_count": session.fallback_count,
            "reprompt_count_by_field": dict(session.reprompt_count_by_field),
            "reprompt_escalation_count": session.reprompt_escalation_count,
            "slot_extraction_failure_count": session.slot_extraction_failure_count,
            "invalid_modifier_count": session.invalid_modifier_count,
            "repeated_user_turn_count": session.repeated_user_turn_count,
        },
        "latency_ms": {
            "average_preprocess_ms": round(mean(_collect_latency_values(logger.rows, "preprocess_ms")), 3)
            if _collect_latency_values(logger.rows, "preprocess_ms")
            else None,
            "average_nlu_ms": round(mean(_collect_latency_values(logger.rows, "nlu_ms")), 3)
            if _collect_latency_values(logger.rows, "nlu_ms")
            else None,
            "average_flow_ms": round(mean(_collect_latency_values(logger.rows, "flow_ms")), 3)
            if _collect_latency_values(logger.rows, "flow_ms")
            else None,
            "average_route_ms": round(mean(_collect_latency_values(logger.rows, "route_ms")), 3)
            if _collect_latency_values(logger.rows, "route_ms")
            else None,
            "average_handler_ms": round(mean(_collect_latency_values(logger.rows, "handler_ms")), 3)
            if _collect_latency_values(logger.rows, "handler_ms")
            else None,
            "average_total_ms": round(mean(_collect_latency_values(logger.rows, "total_ms")), 3)
            if _collect_latency_values(logger.rows, "total_ms")
            else None,
        },
        "steps": steps,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay the Phase 2 ordering validation dataset.")
    parser.add_argument(
        "--dataset",
        default="app/data/evals/phase2_validation_dataset.json",
        help="Path to the replay dataset JSON file.",
    )
    args = parser.parse_args()

    dataset_path = (REPO_ROOT / args.dataset).resolve() if not Path(args.dataset).is_absolute() else Path(args.dataset)
    scenarios = json.loads(dataset_path.read_text(encoding="utf-8"))

    menu_repo = _build_menu_repo(REPO_ROOT)
    results = [_evaluate_scenario(menu_repo, scenario) for scenario in scenarios]

    scenario_count = len(results)
    passed = [result for result in results if result["passed"]]
    completable = [result for result in results if result["completable"]]
    completed = [result for result in completable if result["completed"]]

    intent_total = sum(result["intent_total"] for result in results)
    intent_correct = sum(result["intent_correct"] for result in results)
    slot_turn_total = sum(result["slot_turn_total"] for result in results)
    slot_turn_correct = sum(result["slot_turn_correct"] for result in results)
    slot_pair_expected = sum(result["slot_pair_expected"] for result in results)
    slot_pair_matched = sum(result["slot_pair_matched"] for result in results)

    summary = {
        "dataset_path": str(dataset_path),
        "evaluation_mode": "replay_expected_nlu",
        "scenario_count": scenario_count,
        "passed_scenarios": len(passed),
        "scenario_pass_rate": round(len(passed) / scenario_count, 4) if scenario_count else 0.0,
        "completed_order_scenarios": len(completed),
        "completion_success_rate": round(len(completed) / len(completable), 4) if completable else 0.0,
        "average_turns_per_order": round(sum(result["turn_count"] for result in results) / scenario_count, 2)
        if scenario_count
        else 0.0,
        "intent_accuracy": round(intent_correct / intent_total, 4) if intent_total else None,
        "slot_extraction_accuracy": round(slot_turn_correct / slot_turn_total, 4) if slot_turn_total else None,
        "slot_pair_accuracy": round(slot_pair_matched / slot_pair_expected, 4) if slot_pair_expected else None,
        "latency_ms": {
            "average_preprocess_ms": round(
                mean([result["latency_ms"]["average_preprocess_ms"] for result in results if result["latency_ms"]["average_preprocess_ms"] is not None]),
                3,
            )
            if any(result["latency_ms"]["average_preprocess_ms"] is not None for result in results)
            else None,
            "average_nlu_ms": round(
                mean([result["latency_ms"]["average_nlu_ms"] for result in results if result["latency_ms"]["average_nlu_ms"] is not None]),
                3,
            )
            if any(result["latency_ms"]["average_nlu_ms"] is not None for result in results)
            else None,
            "average_flow_ms": round(
                mean([result["latency_ms"]["average_flow_ms"] for result in results if result["latency_ms"]["average_flow_ms"] is not None]),
                3,
            )
            if any(result["latency_ms"]["average_flow_ms"] is not None for result in results)
            else None,
            "average_route_ms": round(
                mean([result["latency_ms"]["average_route_ms"] for result in results if result["latency_ms"]["average_route_ms"] is not None]),
                3,
            )
            if any(result["latency_ms"]["average_route_ms"] is not None for result in results)
            else None,
            "average_handler_ms": round(
                mean([result["latency_ms"]["average_handler_ms"] for result in results if result["latency_ms"]["average_handler_ms"] is not None]),
                3,
            )
            if any(result["latency_ms"]["average_handler_ms"] is not None for result in results)
            else None,
            "average_total_ms": round(
                mean([result["latency_ms"]["average_total_ms"] for result in results if result["latency_ms"]["average_total_ms"] is not None]),
                3,
            )
            if any(result["latency_ms"]["average_total_ms"] is not None for result in results)
            else None,
        },
        "results": results,
    }
    print(json.dumps(summary, indent=2))
    return 0 if len(passed) == scenario_count else 1


if __name__ == "__main__":
    raise SystemExit(main())
