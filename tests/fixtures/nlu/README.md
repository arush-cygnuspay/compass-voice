# NLU replay fixtures

These JSON files are version-controlled snapshots of the NLU layer's
output for high-traffic utterance + state combinations. Their purpose
is to catch slot-extraction and intent-classification regressions in
CI before they reach production traffic.

The replay test in
`tests/nlu/test_nlu_fixture_replay.py` parameterizes over every
`*.json` file in this directory — drop a new fixture and a new test
case is automatically added.

## Fixture shape

```json
{
  "utterance": "two burgers please",
  "state": "WAITING_FOR_QUANTITY",
  "nlu": {
    "effective_intent": "UNKNOWN",
    "intent_confidence": 0.0,
    "model_main_intent": null,
    "model_sub_intent": null,
    "slots": [
      {"name": "QUANTITY", "value": "2"},
      {"name": "ITEM", "value": "burger"}
    ]
  },
  "expected": {
    "control_intent_kind": null,
    "slots_present": ["QUANTITY", "ITEM"],
    "slot_value": {"QUANTITY": "2"}
  }
}
```

### Required top-level fields

- `utterance` — verbatim transcript the user spoke.
- `state` — the `ConversationState` enum *value* (not name) at the
  start of the turn. Use the string form, e.g. `"waiting_for_payment"`.
- `nlu` — the NLU output for the utterance (see shape above). The
  `slots` array uses `SlotValue` field names.
- `expected` — at least one of the following keys:
  - `effective_intent` — the Intent enum value the NLU should emit.
  - `slots_present` — list of slot names that must appear in
    `nlu.slots` (verifies fixture self-consistency).
  - `slot_value` — `{"<NAME>": "<expected stringified value>"}`.
  - `control_intent_kind` — the `ControlIntentKind` value
    `resolve_control_intent` must produce when fed this NLU result.
    Use `null` to assert the resolver returns `None`.

## Capturing a new fixture

1. Reproduce the utterance against the live NLU layer (or capture
   from production logs). Note the state.
2. Copy the NLU layer's `NLUResult` fields into the `nlu` block.
3. Decide what downstream behavior you want pinned: a control-intent
   kind, a slot extraction, or both. Fill in `expected`.
4. Save as `tests/fixtures/nlu/<short_descriptor>.json`.
5. Run `pytest tests/nlu/test_nlu_fixture_replay.py` and confirm the
   new case passes.

## When a fixture starts failing

A failing fixture means one of:

- The NLU layer changed and no longer emits the same output for that
  utterance — update the fixture only after confirming the new output
  is intentional.
- The control-intent registry / phrase fallback changed — verify the
  change is intentional and update `expected.control_intent_kind`.
- A slot consumer dropped support for a slot name — restore parity or
  update the fixture to reflect the deliberate change.

Do not silence a failing fixture by deleting it. Capture the regression
and either fix the code or update the fixture deliberately.
