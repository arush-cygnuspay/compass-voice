# Phase 3.5 — Offline Replay + Evaluation Harness

**Date**: 2026-05-18  
**Branch**: `feature/app-flow-v2`  
**Scope**: Phase 3.5 — Offline Replay Harness for Phase 3 Option Resolver  
**Status**: Complete — ready for merge review

---

## Overview

The Phase 3.5 Replay Harness lets you replay captured turn logs or built-in fixture turns through
the Phase 3 GPT option resolver decision path without touching any production state.

**What it does:**
- Replays `gpt_repair_turns.jsonl` logs (or fixture turns) through the Phase 3 resolver pipeline
- Compares behavior across GPT modes (`disabled` / `shadow` / `inline`)
- Outputs a JSONL report per turn and a Markdown summary
- Collects training-candidate data from real production logs

**What it never does:**
- Mutates cart, session, FSM state, order, or payment
- Calls OpenAI unless `--use-live-gpt` is explicitly passed
- Logs API keys or PII
- Changes production FSM behavior

---

## Files

| File | Purpose |
|------|---------|
| `app/nlu/semantic_repair/option_resolver_replay.py` | Core library (importable by tests and CLI) |
| `tools/replay_phase3_option_resolver.py` | CLI entry point |
| `tests/tools/test_replay_phase3_option_resolver.py` | 61 automated tests |
| `tests/tools/__init__.py` | Package init |

---

## CLI Usage

```bash
# Replay JSONL log in shadow mode (no live GPT):
python tools/replay_phase3_option_resolver.py \
    --input app/logs/gpt_repair_turns.jsonl \
    --output reports/phase3_replay_report.jsonl \
    --mode shadow \
    --max-turns 500 \
    --filter-state WAITING_FOR_MODIFIER

# Replay built-in fixtures only (no log file needed):
python tools/replay_phase3_option_resolver.py \
    --fixtures-only \
    --mode inline \
    --output reports/fixture_replay.jsonl

# Dry run — no file written, summary to stdout:
python tools/replay_phase3_option_resolver.py \
    --fixtures-only --mode disabled --dry-run

# Live GPT replay (requires OPENAI_API_KEY):
python tools/replay_phase3_option_resolver.py \
    --input app/logs/gpt_repair_turns.jsonl \
    --mode inline \
    --use-live-gpt \
    --max-turns 50 \
    --output reports/live_replay.jsonl

# Verbose per-turn output:
python tools/replay_phase3_option_resolver.py \
    --fixtures-only --mode shadow --verbose
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--input PATH` | `app/logs/gpt_repair_turns.jsonl` | JSONL log to replay |
| `--output PATH` | `reports/phase3_replay_<mode>.jsonl` | Output JSONL report |
| `--mode` | `shadow` | GPT mode: `disabled` \| `shadow` \| `inline` |
| `--max-turns N` | `0` (no limit) | Stop after N turns |
| `--filter-state STATE` | `waiting_for_modifier` | Only replay turns with this `state_before` value. Pass `""` for all. |
| `--filter-response-key KEY` | None | Only replay turns with this `response_key_before` value |
| `--fixtures-only` | False | Run the 7 built-in fixtures, skip JSONL input |
| `--dry-run` | False | Run replay but write no files |
| `--use-live-gpt` | False | Allow real OpenAI calls. Requires `OPENAI_API_KEY`. |
| `--verbose` | False | Print per-turn results to stdout |

---

## Output Format

### Per-Turn JSONL (`<output>.jsonl`)

One `ReplayResult` JSON object per line:

```json
{
  "replay_id": "fixture:1-shadow-a3f2",
  "source_turn_id": "fixture:1",
  "user_text": "macarola cheese",
  "state_before": "WAITING_FOR_MODIFIER",
  "response_key_before": "ask_for_modifier",
  "mode": "shadow",
  "local_intent": null,
  "local_confidence": null,
  "local_slots": [],
  "route_mode": "shadow_gpt",
  "route_reason": "shadow_gpt",
  "gpt_called": true,
  "gpt_decision": "select_option",
  "gpt_selected_names": ["Mozzarella Cheese"],
  "gpt_confidence": 0.91,
  "validator_passed": false,
  "validator_reject_reason": "shadow_mode_never_safe",
  "safe_to_apply": false,
  "would_apply": false,
  "actual_applied": false,
  "error": null,
  "latency_ms": 0.42
}
```

**Key fields:**
- `actual_applied` — Always `false`. Replay never mutates production state.
- `would_apply` — `true` only when `mode=inline` AND `safe_to_apply=true` AND `decision=select_option`. Indicates what *would* have happened in production.
- `validator_passed` — Whether all 5 validator rules passed.
- `validator_reject_reason` — If not passed, which rule failed.

### Summary Markdown (`<output>.summary.md`)

```markdown
# Phase 3 Option Resolver Replay Summary

**Mode**: shadow  
**Total turns replayed**: 7  
**GPT called**: 4  
**Inline candidates**: 0  
**Validator passed**: 0  
**Would apply (inline only)**: 0  
**Errors**: 0  

## Decision Distribution

| Decision | Count |
|----------|-------|
| select_option | 3 |
| no_match | 1 |
| skipped | 3 |
```

---

## Built-In Fixtures

Seven canonical test cases, runnable without any log file (`--fixtures-only`):

| ID | User Text | Group | Expected Route | Would Apply (inline) |
|----|-----------|-------|---------------|---------------------|
| `fixture:1` | `"macarola cheese"` | Cheese | `INLINE_GPT` (>=3 chars, no local) | Yes (phonetic match) |
| `fixture:2` | `"mozarella"` | Cheese | `INLINE_GPT` (>=3 chars, no local) | Yes (fuzzy match) |
| `fixture:3` | `"cheddar"` | Cheese | `INLINE_GPT` (>=3 chars, no local) | Yes (partial match) |
| `fixture:4` | `"no, mozzarella"` | Cheese | `INLINE_GPT` (correction signal) | Yes (correction) |
| `fixture:5` | `"zibblequark snicklefritz supreme"` | Sauce | `INLINE_GPT` (>=3 chars, no local) | No (no_match) |
| `fixture:6` | `"that's all"` | Cheese | `INLINE_GPT` (>=3 chars, no local) | No (done intent) |
| `fixture:7` | `"um"` (repeat_count=3) | Cheese | `INLINE_GPT` (repeat-loop, >=threshold) | No (too short, noise) |

Fixture 4 has `has_correction_signal=True` pre-set (correction signal already detected upstream).  
Fixture 7 has `repeat_count=3` to trigger repeat-loop recovery path.

---

## Library API

```python
from app.nlu.semantic_repair.option_resolver_replay import (
    BUILT_IN_FIXTURES,
    ReplayInputTurn,
    ReplayResult,
    Phase3OptionResolverReplayHarness,
    ReplaySummaryBuilder,
    parse_jsonl_row,
)
from app.config.semantic_repair import SemanticRepairConfig

# Build config (mode: "disabled" | "shadow" | "inline")
cfg = SemanticRepairConfig(
    phase=3,
    model="gpt-4o-mini",
    timeout_seconds=2.0,
    option_resolver_mode="shadow",
    option_resolver_timeout_ms=2000,
    option_resolver_min_confidence=0.75,
    option_resolver_repeat_threshold=2,
)

# Instantiate harness (use_live_gpt=False is the safe default)
harness = Phase3OptionResolverReplayHarness(config=cfg, use_live_gpt=False)
summary = ReplaySummaryBuilder(mode="shadow")

# Replay built-in fixtures
for result in harness.replay_fixtures(BUILT_IN_FIXTURES, mode="shadow"):
    summary.add(result)
    print(result.to_dict())

# Replay from JSONL log
for result in harness.replay_jsonl(
    "app/logs/gpt_repair_turns.jsonl",
    mode="shadow",
    max_turns=100,
    filter_state="waiting_for_modifier",
):
    summary.add(result)

print(summary.to_markdown())
```

### `ReplayInputTurn` Fields

```python
@dataclass(frozen=True, slots=True)
class ReplayInputTurn:
    user_text: str
    state_before: str
    source_turn_id: str | None = None
    response_key_before: str | None = None
    local_intent: str | None = None
    local_confidence: float | None = None
    local_slots: tuple[dict, ...] = ()
    top_intents: tuple[dict, ...] = ()
    choice_names: tuple[str, ...] = ()   # modifier group choices
    group_name: str = "Options"
    item_name: str = "Item"
    repeat_count: int = 0
    has_correction_signal: bool | None = None  # None = auto-detect from user_text
    session_id: str | None = None
    turn_index: int | None = None
```

### `ReplayResult` Fields

```python
@dataclass(frozen=True, slots=True)
class ReplayResult:
    replay_id: str             # "<source_turn_id>-<mode>-<hash>"
    source_turn_id: str | None
    user_text: str
    state_before: str
    mode: str                  # "disabled" | "shadow" | "inline"
    response_key_before: str | None
    local_intent: str | None
    local_confidence: float | None
    local_slots: list[dict]
    route_mode: str            # "no_gpt" | "shadow_gpt" | "inline_gpt"
    route_reason: str
    gpt_called: bool
    gpt_decision: str | None   # "select_option" | "no_match" | "error" | None
    gpt_selected_names: list[str]
    gpt_confidence: float | None
    validator_passed: bool
    validator_reject_reason: str | None
    safe_to_apply: bool
    would_apply: bool          # True only when mode=inline AND safe_to_apply AND select_option
    actual_applied: bool       # ALWAYS FALSE — replay never mutates state
    error: str | None
    latency_ms: float | None
```

---

## JSONL Input Format

The harness reads from `gpt_repair_turns.jsonl`, which uses this structure:

```json
{
  "timestamp_utc": "2026-05-18T10:30:00Z",
  "session_id": "sess_abc123",
  "turn_index": 4,
  "state_before": "WAITING_FOR_MODIFIER",
  "normalized_text": "macarola cheese",
  "response_key": "ask_for_modifier",
  "local": {
    "intent": "add_item",
    "confidence": 0.62,
    "slots": [{"n": "MODIFIER", "v": "macarola"}],
    "top_intents": [{"i": "add_item", "c": 0.62}]
  },
  "allowed": {
    "choices": ["American Cheese", "Mozzarella Cheese", "Cheddar Cheese"]
  }
}
```

The harness normalizes `state_before` to lowercase for filter comparisons.  
Missing `local` or `allowed` blocks default to empty (no local intent, no choices).

### Synthetic Modifier Group

Since JSONL logs store choice *names* only (no modifier IDs), the harness builds a
`PendingModifierGroup` with synthetic IDs (`synthetic_0`, `synthetic_1`, ...).
The Phase 3 validator uses name-based matching (case-insensitive), so synthetic IDs
work correctly for validation. This is replay-only behavior — production always uses
real modifier IDs from the menu.

---

## Safety Guarantees

| Guarantee | Mechanism |
|-----------|-----------|
| No cart / state mutation | `replay_turn()` builds synthetic context; never calls `_apply_modifier_selection()` |
| No live GPT by default | `use_live_gpt=False` in constructor; service uses `mock_client` or returns `NOT_CALLED` sentinel |
| `actual_applied` always False | Hardcoded in `ReplayResult` constructor |
| No API key in output | `ReplayResult.to_dict()` never includes any env vars or secrets |
| No full menu/cart in output | Context builder enforces its existing safety caps (choices=20, history=3) |
| Errors never crash replay | `replay_turn()` catches all exceptions and writes `error` field |

---

## Test Coverage (61 tests)

| Class | Tests | What It Covers |
|-------|-------|----------------|
| `TestParseJsonlRow` | 10 | Nested field extraction, missing blocks, filter normalization, malformed rows |
| `TestReplayHarnessDisabled` | 4 | Disabled mode: `gpt_called=False`, no `would_apply`, serializable, safe |
| `TestReplayHarnessShadow` | 4 | Shadow mode: GPT called for phonetic mismatch, never `would_apply`, fixture batch |
| `TestReplayHarnessInline` | 5 | Inline mode: `would_apply=True` on validator pass, `False` on hallucinated name |
| `TestReplayRobustness` | 7 | Never raises on empty/noise text, all fixtures produce valid JSON, empty choices |
| `TestJsonlReplay` | 6 | JSONL file replay: filter_state, max_turns, filter_response_key, empty file |
| `TestReplaySummaryBuilder` | 8 | Aggregation: totals, would_apply count, decision distribution, zero-turn safe |
| `TestBuiltInFixtures` | 7 | All 7 fixtures: not empty, valid structure, unique IDs, repeat-loop fixture |
| `TestBuildSyntheticGroup` | 4 | Synthetic group builder: correct IDs, choice count, name round-trip |
| `TestCliDryRun` | 6 | CLI: `--fixtures-only --dry-run`, `--use-live-gpt` without key returns exit 1 |

---

## Running Tests

```bash
# Phase 3.5 harness tests only:
pytest tests/tools/test_replay_phase3_option_resolver.py -q

# Phase 3 unit tests (91 tests):
pytest tests/nlu/semantic_repair/test_phase3_option_resolver.py -q

# Full NLU + core + logging + diagnostics:
pytest tests/nlu/semantic_repair tests/core tests/logging tests/diagnostics -q

# Full suite including state_machine:
pytest tests/nlu/semantic_repair tests/core tests/state_machine -q
```

**Expected results** (as of 2026-05-18):

| Suite | Passed | Failed | Notes |
|-------|--------|--------|-------|
| `test_replay_phase3_option_resolver.py` | **61** | 0 | All new Phase 3.5 tests |
| `test_phase3_option_resolver.py` | **91** | 0 | All Phase 3 unit tests |
| `nlu/semantic_repair + core + logging + diagnostics` | **1247** | 11 | 11 pre-existing failures |
| `nlu/semantic_repair + core + state_machine` | **2397** | 49 | 49 pre-existing failures |

All failures are pre-existing (confirmed by stash test). Zero new regressions.

---

## Workflow: Collecting Training Data

### Week 1: Shadow mode — collect data, no behavior change

```bash
# 1. Enable shadow mode in production (env var, no deploy needed):
export COMPASS_GPT_OPTION_RESOLVER_MODE=shadow

# 2. Let the system run for a day or two, then replay the logs:
python tools/replay_phase3_option_resolver.py \
    --input app/logs/gpt_repair_turns.jsonl \
    --output reports/shadow_replay.jsonl \
    --mode shadow \
    --filter-state waiting_for_modifier

# 3. Review reports/shadow_replay.summary.md
# Target: >60% select_option decisions with confidence >= 0.75
```

### Week 2+: Inline mode — live fuzzy recovery

```bash
# 1. Enable inline mode:
export COMPASS_GPT_OPTION_RESOLVER_MODE=inline

# 2. Replay recent logs in inline mode to project impact:
python tools/replay_phase3_option_resolver.py \
    --input app/logs/gpt_repair_turns.jsonl \
    --output reports/inline_projection.jsonl \
    --mode inline

# 3. Check would_apply rate in reports/inline_projection.summary.md
```

### Instant rollback

```bash
export COMPASS_GPT_OPTION_RESOLVER_MODE=disabled
# No restart needed — config reads env var at runtime.
```

---

## Known Limitations

1. **Synthetic modifier IDs**: The harness builds modifier groups from choice names only
   (no real modifier IDs from the menu). This is correct for validation purposes
   (validator is name-based) but means `gpt_selected_names` contains names, not IDs.
   In production, `build_modifier_selections_from_names()` maps names back to real IDs.

2. **`local_resolved=False` always**: The harness conservatively sets `local_resolved=False`
   on every turn, so routing policy may call GPT on turns where local NLU actually succeeded.
   This is the correct behavior for replay — we want to see what GPT *would* have said even
   when local resolved, to collect training signal.

3. **No repeat_count in JSONL**: The `gpt_repair_turns.jsonl` format does not currently
   include `repeat_count`. All replayed-from-log turns have `repeat_count=0`.
   Repeat-loop recovery tests work only in the built-in fixtures (where `repeat_count` is
   set manually).

4. **No group_name / item_name in JSONL**: Log format does not include the specific modifier
   group name or item name. Replayed turns use `group_name="Options"` and `item_name="Item"`.
   This does not affect validation (choices are correct) but affects GPT payload readability.

5. **No persistent state**: The harness does not track cross-turn session state. Each turn is
   independent. This means multi-turn context (history window) in the GPT payload uses only
   the `previous_turns` field if provided, not live session history.

---

## Commit Recommendation

**YES — ready to commit.**

Phase 3.5 safety invariants hold:
- `actual_applied` is always `False` — replay can never change production behavior
- `use_live_gpt=False` default — no OpenAI calls without explicit opt-in
- No API key, PII, full menu, or full cart in any output file
- `would_apply` is read-only signal, not an action
- All Phase 3 production services reused as-is (no new GPT behavior added)
- 61 new Phase 3.5 tests pass, 91 Phase 3 unit tests pass
- Zero new regressions (all pre-existing failures confirmed by stash test)

---

*Generated by Compass Voice Engineering Agent — Phase 3.5 Replay Harness*
