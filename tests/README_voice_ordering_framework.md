# Voice Ordering Test Framework

## Layout

- `tests/nlu/`: intent and slot behavior checks
- `tests/flow/`: end-to-end FSM conversation scenarios
- `tests/payment/`: checkout and payment wait behavior
- `tests/edge_cases/`: repeated failures, unclear input, escalation, mixed-intent coverage
- `tests/support/voice_test_harness.py`: reusable conversation simulator
- `tests/data/voice_ordering_sample_dataset.json`: sample evaluation dataset
- `scripts/evaluate_voice_ordering_framework.py`: aggregate metrics runner

## Run the suites

```powershell
python -m pytest tests\nlu tests\flow tests\payment tests\edge_cases -q
```

Run the evaluation metrics script:

```powershell
python scripts\evaluate_voice_ordering_framework.py
```

## Coverage model

These tests intentionally avoid exact full-text response matching.

They validate:

- canonical intent routing
- slot capture and downstream normalization
- FSM state transitions
- correction and interruption behavior
- payment wait state behavior
- fallback and escalation counters

Response validation should prefer:

- `response_key`
- `conversation_state`
- session counters
- selective phrase checks only when the response contract matters
