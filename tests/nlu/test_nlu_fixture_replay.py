"""Parameterized replay tests over `tests/fixtures/nlu/*.json`.

Each fixture pins a downstream contract for a representative
utterance + state combination. Adding a JSON file under the fixture
directory automatically adds a new test case.

Failure modes (per fixture):
- Expected intent is no longer produced by the NLU layer.
- Expected slot is no longer extracted.
- Expected control-intent kind no longer resolves via the resolver.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from app.nlu.intent_resolution.intent import Intent
from app.nlu.nlu_result import NLUResult, SlotValue
from app.state_machine.control_intent_resolver import (
    ControlIntentKind,
    resolve_control_intent,
)
from app.state_machine.models.conversation_state import ConversationState


FIXTURE_DIR = pathlib.Path(__file__).parent.parent / "fixtures" / "nlu"


def _load_intent(label: str | None) -> Intent | str:
    """Map a fixture intent label to an Intent enum, or fall through
    to the string when the label is not in the enum (so the resolver's
    canonical-label path can still register it)."""
    if not label:
        return Intent.UNKNOWN
    try:
        return Intent(label)
    except ValueError:
        return label


def _build_nlu_result(fixture: dict) -> NLUResult:
    nlu_block = fixture["nlu"]
    slot_dicts = nlu_block.get("slots") or []
    slots = tuple(
        SlotValue(
            name=str(s["name"]),
            value=s["value"],
            raw=s.get("raw"),
            start=s.get("start"),
            end=s.get("end"),
            confidence=s.get("confidence"),
        )
        for s in slot_dicts
    )
    intent_or_label = _load_intent(nlu_block.get("effective_intent"))
    effective_intent = (
        intent_or_label if isinstance(intent_or_label, Intent) else Intent.UNKNOWN
    )
    return NLUResult(
        effective_intent=effective_intent,
        intent_confidence=float(nlu_block.get("intent_confidence", 0.0) or 0.0),
        raw_text=fixture.get("utterance", ""),
        normalized_text=fixture.get("utterance", "").lower(),
        model_main_intent=nlu_block.get("model_main_intent"),
        model_sub_intent=nlu_block.get("model_sub_intent"),
        slots=slots,
        slot_model_ran=bool(nlu_block.get("slot_model_ran", True)),
    )


FIXTURE_PATHS = sorted(FIXTURE_DIR.glob("*.json"))


@pytest.mark.parametrize(
    "fixture_path",
    FIXTURE_PATHS,
    ids=lambda p: p.stem,
)
def test_nlu_fixture_replay(fixture_path: pathlib.Path) -> None:
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    expected = fixture.get("expected") or {}

    state_value = fixture["state"]
    state = ConversationState(state_value)
    nlu = _build_nlu_result(fixture)
    utterance = fixture["utterance"]

    # 1. Effective intent assertion (sanity for the fixture itself).
    if "effective_intent" in expected:
        # When the label exists in the Intent enum we compare the enum.
        # When it doesn't (e.g. an arbitrary classifier label), fall
        # back to comparing the raw label string.
        try:
            expected_intent = Intent(expected["effective_intent"])
            assert nlu.effective_intent == expected_intent, (
                f"{fixture_path.name}: effective_intent "
                f"{nlu.effective_intent} != {expected_intent}"
            )
        except ValueError:
            assert (
                fixture["nlu"].get("effective_intent")
                == expected["effective_intent"]
            ), f"{fixture_path.name}: NLU effective_intent label mismatch"

    # 2. Slot presence assertion.
    if "slots_present" in expected:
        names = {str(s.name).upper() for s in nlu.slots}
        for required in expected["slots_present"]:
            assert required.upper() in names, (
                f"{fixture_path.name}: expected slot {required} missing"
            )

    # 3. Slot value assertion.
    if "slot_value" in expected:
        for slot_name, expected_value in expected["slot_value"].items():
            match = next(
                (
                    s
                    for s in nlu.slots
                    if str(s.name).upper() == slot_name.upper()
                ),
                None,
            )
            assert match is not None, (
                f"{fixture_path.name}: slot {slot_name} not present"
            )
            assert str(match.value) == str(expected_value), (
                f"{fixture_path.name}: slot {slot_name} value "
                f"{match.value!r} != {expected_value!r}"
            )

    # 4. Control-intent resolver assertion.
    if "control_intent_kind" in expected:
        resolved = resolve_control_intent(
            utterance,
            nlu.effective_intent,
            nlu.model_sub_intent,
            state,
            None,
            nlu_result=nlu,
            intent_confidence=nlu.intent_confidence,
        )
        expected_kind = expected["control_intent_kind"]
        if expected_kind is None:
            assert resolved is None, (
                f"{fixture_path.name}: expected resolver=None, got {resolved}"
            )
        else:
            assert resolved is not None, (
                f"{fixture_path.name}: resolver returned None, "
                f"expected kind={expected_kind}"
            )
            assert resolved.kind == ControlIntentKind(expected_kind), (
                f"{fixture_path.name}: resolver kind {resolved.kind} "
                f"!= expected {expected_kind}"
            )


def test_fixture_directory_is_not_empty() -> None:
    """Guard against accidental fixture purge."""
    assert FIXTURE_PATHS, "no NLU fixtures found in tests/fixtures/nlu/"
    assert len(FIXTURE_PATHS) >= 8, (
        f"fixture set shrunk to {len(FIXTURE_PATHS)}; expected >= 8"
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
