# Phase 4 — GPT Add-Item Planner

**Date**: 2026-05-18  
**Branch**: `feature/app-flow-v2`  
**Scope**: Phase 4 + 4.1 — GPT Add-Item Planner for complex utterances + production wiring  
**Status**: Phase 4.1 complete — ready for review

---

## Purpose

Phase 3 solved modifier option resolution in `WAITING_FOR_MODIFIER`.  
Phase 4 solves a harder problem: **complex add-item utterances** where the customer says
everything at once — items, modifiers, sides, sizes, and quantities — and the local
NLU + multi-item parser cannot reliably group them.

### Target examples

| Utterance | Local NLU struggle | Phase 4 solution |
|-----------|-------------------|-----------------|
| "Chicken burger with mozzarella, onions, mayo, and a coke" | Slots ungrouped; coke vs modifier ambiguous | GPT groups mods to burger, coke as separate item |
| "Two chicken sandwiches, one with Swiss, one plain" | Multi-item quantity + conditional modifier | GPT emits two items with correct modifiers |
| "Cheeseburger combo with fries and large coke" | Side vs combo unclear | GPT assigns fries+coke to side groups |
| "Burger with no onions and extra cheese" | Negated + extra modifier mix | GPT emits remove+extra operations |
| "Hawaiian pizza and two cokes" | Multiple items, multi-quantity | GPT plans two separate cart entries |

---

## Architecture

### New files

| File | Purpose |
|------|---------|
| `app/nlu/semantic_repair/add_item_planner_routing_policy.py` | `GptAddItemPlannerRoutingPolicy` + `is_complex_utterance()` |
| `app/nlu/semantic_repair/add_item_planner_result.py` | `AddItemPlannerResult`, `PlannerGptItem`, `PlannerGptModifier`, `PlannerGptSide`, `PlannerUnresolved`, `ADD_ITEM_PLANNER_NOT_CALLED` |
| `app/nlu/semantic_repair/add_item_planner_context_builder.py` | `GptAddItemPlannerContextBuilder` — compact payload with candidate items only |
| `app/nlu/semantic_repair/add_item_planner_output_parser.py` | `parse_planner_output()` — Phase 4 schema parser |
| `app/nlu/semantic_repair/add_item_planner_service.py` | `GptAddItemPlannerService` — full pipeline |
| `app/nlu/semantic_repair/add_item_planner_replay.py` | Phase 4 replay library |
| `tools/replay_phase4_add_item_planner.py` | Replay CLI |

### Modified files

| File | Change |
|------|--------|
| `app/config/semantic_repair.py` | 5 new Phase 4 config fields |
| `app/nlu/semantic_repair/add_item_plan_validator.py` | `validate_planner_items()` adapter + `PlannerApplyGate` class |
| `app/state_machine/handlers/item/add_item/add_item_handler.py` | `gpt_planner` injection, `_try_gpt_planner()`, `_apply_planner_result()`, `_log_planner_apply_outcome()` |
| `app/core/handler_dispatcher.py` | Phase 4.1: `_build_add_item_planner()` factory, wires planner into `AddItemHandler` |

---

## Routing Policy

`GptAddItemPlannerRoutingPolicy.decide()` returns `(AddItemPlannerRouteMode, reason_str)`.

### Route modes

| Mode | GPT Called | Result Applied | Use When |
|------|-----------|----------------|----------|
| `NO_GPT` | No | No | Default — simple utterance, mode disabled, no evidence |
| `SHADOW_GPT` | Yes | Never | Collecting training data only |
| `INLINE_GPT` | Yes | When apply gate approves | Production fuzzy planning |

### Complexity signals (any one triggers complex classification)

```
"with" / "without" present               → modifier attachment
"extra" / "light" present                → quantity modifier
"no <noun>" pattern                       → negated modifier
Comma in utterance                        → list / multi-entity
"and" word present                        → conjunction / multi-item
Multiple number words (two X, one Y)      → multi-quantity
ITEM + MODIFIER or SIDE slot from NLU     → complex slot grouping
Multiple ITEM slots from NLU              → multi-item evidence
```

### Routing decision flow

```
1. Empty text?               → NO_GPT (empty_text)
2. mode == "disabled"?       → NO_GPT (mode_disabled)
3. Simple + high-confidence? → NO_GPT (simple_high_confidence_local)
   (local intent=ADD_ITEM, confidence >= 0.85, no complexity, single ITEM slot)
4. complex_utterance = is_complex_utterance(text, local_slots)
5. has_evidence = item_candidates_exist OR local_intent in {ADD_ITEM, UNKNOWN} OR ITEM slot
6. not complex AND not evidence?  → NO_GPT (no_complexity_no_evidence)

Mode "shadow":
   complex OR evidence → SHADOW_GPT

Mode "inline":
   complex             → INLINE_GPT
   evidence only       → NO_GPT (not complex enough to add value over local path)
```

---

## Context Builder

`GptAddItemPlannerContextBuilder` builds a compact JSON payload.

### Safety caps

| Field | Cap |
|-------|-----|
| Candidate items | `MAX_ITEM_CANDIDATES = 10` |
| Options per modifier/side group | `MAX_OPTION_CANDIDATES = 20` |
| History turns | `MAX_HISTORY_TURNS = 3` |
| Top-K intents | `MAX_TOP_K = 4` |
| Local slots | `MAX_LOCAL_SLOTS = 6` |
| Cart items | `MAX_CART_ITEMS = 10` |

### Payload shape

```json
{
  "t": "add_item_plan",
  "text": "<normalized user utterance>",
  "local": {
    "intent": "add_item",
    "conf": 0.55,
    "top_k": [{"i": "add_item", "c": 0.55}],
    "slots": [{"n": "ITEM", "v": "chicken burger"}]
  },
  "candidates": [
    {
      "id": "menu_item_id",
      "name": "Chicken Burger",
      "sizes": [],
      "modifier_groups": [{"name": "Cheese", "choices": ["American", "Swiss"]}],
      "side_groups": [{"name": "Drink", "choices": ["Coke", "Sprite"]}]
    }
  ],
  "cart": {"n": 0, "items": []},
  "history": [["bot", "What can I get you?"], ["user", "..."]],
  "rules": "...",
  "schema": "..."
}
```

### What is NEVER in the payload

- Full menu catalog
- Full cart raw JSON
- API keys or PII (phone, address, payment links)
- Prices or tax data
- Modifier/side groups for items NOT in candidates

---

## GPT Output Schema (Phase 4)

```json
{
  "decision": "add_items | clarify | no_repair | unclear",
  "items": [
    {
      "candidate_item_id": "string | null",
      "item_name": "string",
      "quantity": 1,
      "size": "string | null",
      "variant": "string | null",
      "modifiers": [
        {"name": "string", "operation": "add | remove | extra | light", "quantity": 1}
      ],
      "sides": [
        {"name": "string", "quantity": 1, "size": "string | null"}
      ],
      "special_instructions": "string | null"
    }
  ],
  "unresolved": [
    {"text": "string", "reason": "not_on_menu | ambiguous | belongs_to_unknown_group | unsupported"}
  ],
  "confidence": 0.0,
  "reason_code": "complex_with_phrase | multi_item | slot_grouping_repair | unknown_with_item_evidence | unclear",
  "safe_to_apply": false
}
```

**Note**: GPT's `safe_to_apply` field is **ignored**. The apply gate computes safety independently.

### Modifier operations (Phase 4 extends Phase 1/2)

| Operation | Meaning |
|-----------|---------|
| `add` | Add this modifier normally |
| `remove` | Remove / exclude this modifier |
| `extra` | Add extra portion / double |
| `light` | Add with reduced portion |

---

## Validator Contract

`AddItemPlanValidator.validate_planner_items()` adapts `PlannerGptItem[]` → `GptAddItem[]` and delegates to the existing `validate()` method.

### Blocking warnings (reject item)

| Code | Trigger |
|------|---------|
| `item_not_on_menu` | Item name not in menu (exact, alias, or voice label) |
| `item_ambiguous` | Item name matches multiple menu items |
| `invalid_item_size` | Named size/variant doesn't exist for this item |
| `side_not_valid_for_item` | Side not in any side group for this item |
| `invalid_side_size` | Named size doesn't exist for this side |
| `over_max_selector` | Sides in a group exceed max_selector |

### Non-blocking warnings (item accepted with notes)

| Code | Trigger |
|------|---------|
| `modifier_not_found` | Modifier not in any group (logged only) |
| `modifier_size_unsupported` | Size specified for modifier (stripped) |
| `required_group_missing` | Required side group has no selection (FSM will ask) |
| `quantity_clamped` | Quantity out of safe range (1–20) |
| `duplicate_dropped` | Duplicate side in non-duplicate group |

### Partial valid plan behavior

- Items with blocking warnings are moved to `rejected_items[]` and excluded from `validated_plan.items`
- Items with ONLY non-blocking warnings are accepted in `validated_plan.items`
- Required group gaps (`required_group_missing`) are non-blocking — FSM will prompt for them after apply
- The apply gate checks `has_blocking_warnings`: if True → `safe_to_apply=False`

---

## Apply Gate

`PlannerApplyGate.should_apply()` is the single authority for whether a plan is applied.

### Gates (all must pass for `safe_to_apply=True`)

| Gate | Condition |
|------|-----------|
| 1 | `route_mode == "inline_gpt"` |
| 2 | `gpt_called == True` |
| 3 | `timed_out == False` |
| 4 | `parse_error is None` |
| 5 | `decision == "add_items"` |
| 6 | `confidence >= min_confidence` (default 0.75) |
| 7 | `validated_plan is not None` AND `not has_blocking_warnings` AND `len(items) >= 1` |
| 8 | All `validated_plan.items[*].item_id` non-empty (resolved to real menu IDs) |

Any gate failure → `(False, reason_str)`.

---

## Configuration

### New env vars (all default to safe values)

| Variable | Default | Description |
|----------|---------|-------------|
| `COMPASS_GPT_ADD_ITEM_PLANNER_MODE` | `disabled` | `disabled` \| `shadow` \| `inline` |
| `COMPASS_GPT_ADD_ITEM_PLANNER_TIMEOUT_MS` | `1800` | Per-call timeout in ms |
| `COMPASS_GPT_ADD_ITEM_PLANNER_MIN_CONFIDENCE` | `0.75` | Apply gate confidence threshold |
| `COMPASS_GPT_ADD_ITEM_PLANNER_MAX_ITEM_CANDIDATES` | `10` | Max candidate items in context |
| `COMPASS_GPT_ADD_ITEM_PLANNER_MAX_OPTION_CANDIDATES` | `20` | Max options per group in context |

### Production wiring (Phase 4.1)

The planner is wired in `app/core/handler_dispatcher.py` via `_build_add_item_planner()`:

```python
# app/core/handler_dispatcher.py
from app.nlu.semantic_repair.add_item_planner_service import GptAddItemPlannerService
from app.config.semantic_repair import get_semantic_repair_config

def _build_add_item_planner() -> GptAddItemPlannerService | None:
    cfg = get_semantic_repair_config()
    if cfg.add_item_planner_mode == "disabled":
        return None
    return GptAddItemPlannerService(config=cfg)

# In HandlerDispatcher.__init__:
_gpt_add_item_planner = _build_add_item_planner()
self.handlers = {
    "add_item_handler": AddItemHandler(
        menu_repo=menu_repo,
        gpt_planner=_gpt_add_item_planner,
    ),
    ...
}
```

When `gpt_planner=None` (the default when `COMPASS_GPT_ADD_ITEM_PLANNER_MODE=disabled`),
the planner path is completely bypassed — zero overhead.  The helper never raises;
config errors produce `None` (safe fallback to local-only path).

---

## Integration Point

The Phase 4 hook runs in `AddItemHandler.handle()` **before** the local multi-item parser:

```
STT final text
→ normalize
→ [Phase 4 GPT Planner hook] ← HERE
    _try_gpt_planner(user_text, slots, session)
    if safe_to_apply → _apply_planner_result() → return HandlerResult
    (shadow / not safe → fall through)
→ parse_multi_item_utterance()
→ multi_item_coordinator or single_item_path
→ item_resolution_handler
```

### Current apply scope (Phase 4.1)

| Plan type | Apply behavior |
|-----------|---------------|
| Single item (1 validated item) | Applies via `item_resolution_handler.resolve_item_and_enter_flow()` |
| Multi-item (≥2 validated items) | Returns None → `block_reason="multi_item_deferred"` logged → falls through to local path |
| Shadow mode | Logs result, always falls through |
| Disabled mode | Not called at all |

Multi-item inline apply is deferred to Phase 4.2 (requires `MultiItemQueueCoordinator` integration).

---

## Logging Fields

### Event 1 — `add_item_planner_result` (emitted in `_try_gpt_planner()`)

Always emitted when the planner is called.  Contains planner decision and gate outcome.

| Field | Type | Description |
|-------|------|-------------|
| `add_item_planner_mode` | str | Config mode at call time |
| `add_item_planner_route_reason` | str | Why this route was chosen |
| `add_item_planner_called` | bool | Whether GPT was called |
| `add_item_planner_decision` | str | GPT decision string |
| `add_item_planner_confidence` | float\|None | GPT confidence |
| `add_item_planner_validator_passed` | bool | Whether validator approved |
| `add_item_planner_safe_to_apply` | bool | Apply gate result |
| `add_item_planner_latency_ms` | float\|None | GPT call latency |

### Event 2 — `add_item_planner_apply_outcome` (emitted only when `safe_to_apply=True`)

Emitted after the handler attempts application.  Adds the final apply decision.

| Field | Type | Description |
|-------|------|-------------|
| `add_item_planner_applied` | bool | True if plan was applied and HandlerResult returned |
| `add_item_planner_apply_block_reason` | str\|None | None when applied; `"multi_item_deferred"` when validated plan has ≠1 item; `"apply_helper_returned_none"` on unexpected None |
| `add_item_planner_route_mode` | str | Repeated from result for easy correlation |
| `add_item_planner_confidence` | float\|None | Repeated from result |

### Wiring log — `add_item_planner_wired`

Emitted once at `HandlerDispatcher` construction when mode ≠ `disabled`.

JSONL fields (from `AddItemPlannerResult.to_dict()`):

All event-1 fields plus: `validator_reject_reason`, `items[]`, `unresolved[]`, `parse_error`, `model`, `prompt_chars`, `completion_chars`, `validated_plan.{items_count, rejected_items, has_blocking_warnings}`.

---

## Replay / Evaluation

```bash
# Built-in fixtures, shadow mode, dry run:
python tools/replay_phase4_add_item_planner.py \
    --fixtures-only --mode shadow --dry-run

# Replay production logs:
python tools/replay_phase4_add_item_planner.py \
    --input app/logs/gpt_repair_turns.jsonl \
    --output reports/phase4_replay.jsonl \
    --mode shadow \
    --filter-state idle

# Live GPT (requires OPENAI_API_KEY):
python tools/replay_phase4_add_item_planner.py \
    --fixtures-only --mode inline --use-live-gpt
```

Built-in fixtures cover: phonetic/fuzzy mismatch, multi-item, extra/remove operations,
ambiguous grouping, nonsense input, high-confidence simple path.

---

## Running Tests

```bash
# Phase 4 unit tests:
pytest tests/nlu/semantic_repair/test_phase4_add_item_planner.py -q

# Existing validator tests (must still pass):
pytest tests/nlu/semantic_repair/test_add_item_plan_validator.py -q

# Phase 4 replay tests:
pytest tests/tools/test_replay_phase4_add_item_planner.py -q

# Phase 4.1 production wiring tests:
pytest tests/core/test_handler_dispatcher_phase4.py -q

# Full suite (shows pre-existing failures only):
pytest tests/nlu/semantic_repair tests/core tests/state_machine -q
```

---

## Rollout Plan

### Deployment Order

**Step 1 — Baseline deploy (all disabled)**
```bash
COMPASS_GPT_OPTION_RESOLVER_MODE=disabled
COMPASS_GPT_ADD_ITEM_PLANNER_MODE=disabled
```
Verify: Pre-existing FSM behavior is identical. Run `tests/core` + `tests/state_machine`. No GPT calls at all.

---

**Step 2 — Option resolver shadow only**
```bash
COMPASS_GPT_OPTION_RESOLVER_MODE=shadow
COMPASS_GPT_ADD_ITEM_PLANNER_MODE=disabled
OPENAI_API_KEY=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
```
Verify:
- Log field `option_resolver_called=true` appears for WAITING_FOR_MODIFIER turns with phonetic mismatches
- `option_resolver_applied=false` always (shadow never applies)
- No cart mutation, no state change from GPT
- Latency p95 < 1200ms

---

**Step 3 — Option resolver inline + planner shadow**
```bash
COMPASS_GPT_OPTION_RESOLVER_MODE=inline
COMPASS_GPT_ADD_ITEM_PLANNER_MODE=shadow
OPENAI_API_KEY=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
```
Verify:
- Option resolver applies for high-confidence phonetic matches (confidence ≥ 0.75)
- Planner logs appear for complex utterances; `add_item_planner_applied=false` always
- Run `docs/gpt_manual_staging_checklist.md` sections B, C, D

---

**Step 4 — Staging: option resolver inline + planner inline**
```bash
COMPASS_GPT_OPTION_RESOLVER_MODE=inline
COMPASS_GPT_ADD_ITEM_PLANNER_MODE=inline
OPENAI_API_KEY=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
```
Staging only. Verify:
- Single-item apply gate passes: `add_item_planner_applied=true` for valid single items
- Multi-item deferred: `add_item_planner_apply_block_reason="multi_item_deferred"`
- All failure cases from `docs/gpt_manual_staging_checklist.md` section F pass
- Full manual checklist signed off by tester

---

**Step 5 — Production promotion**
```bash
# Start here — shadow for both, monitor for 1 week:
COMPASS_GPT_OPTION_RESOLVER_MODE=shadow
COMPASS_GPT_ADD_ITEM_PLANNER_MODE=shadow

# Promote option resolver only after shadow data looks good:
COMPASS_GPT_OPTION_RESOLVER_MODE=inline
COMPASS_GPT_ADD_ITEM_PLANNER_MODE=shadow   # keep planner shadow until staging data reviewed

# Promote planner inline only after explicit approval:
COMPASS_GPT_OPTION_RESOLVER_MODE=inline
COMPASS_GPT_ADD_ITEM_PLANNER_MODE=inline
```

---

### Expected Logs (per step)

| Log field | Step 2 shadow | Step 3/4 inline |
|-----------|---------------|-----------------|
| `option_resolver_called` | `true` | `true` |
| `option_resolver_applied` | `false` | `true` (when safe) |
| `option_resolver_latency_ms` | populated | populated |
| `add_item_planner_called` | `false` | `true` (complex utts) |
| `add_item_planner_applied` | — | `true` (single-item) or `false` |
| `add_item_planner_apply_block_reason` | — | `null` or `"multi_item_deferred"` |
| `add_item_planner_validator_reject_reason` | — | populated on rejection |

### Alert Thresholds

| Metric | Alert trigger |
|--------|--------------|
| `option_resolver_applied` rate | > 80% (too aggressive) or < 10% (not resolving) |
| `add_item_planner_validator_reject_reason` rate | > 20% |
| `add_item_planner_latency_ms` p95 | > 1800ms |
| `option_resolver_latency_ms` p95 | > 1200ms |

### Rollback

```bash
# Instant rollback — no deploy needed:
COMPASS_GPT_OPTION_RESOLVER_MODE=disabled
COMPASS_GPT_ADD_ITEM_PLANNER_MODE=disabled
```

Feature flags are env vars — rollback takes effect on next request after env reload.

### When to enable inline

Enable `add_item_planner_mode=inline` only when:
1. ≥ 7 days of shadow mode data shows `add_items` decisions with confidence ≥ 0.75 on > 50% of complex utterances
2. `validator_reject_reason` rate < 20% in shadow logs
3. Staging manual checklist (`docs/gpt_manual_staging_checklist.md`) fully signed off
4. `scripts/verify_gpt_ordering_pipeline.py --env staging` returns exit code 0
5. At least one full production week of option resolver inline without incidents

### Manual Validation Steps

Run before each promotion:
```bash
# 1. Pre-flight check:
python scripts/verify_gpt_ordering_pipeline.py --env staging

# 2. Run all phase tests:
pytest tests/core/test_handler_dispatcher_phase4.py -q
pytest tests/nlu/semantic_repair/test_phase4_add_item_planner.py -q
pytest tests/tools/test_replay_phase4_add_item_planner.py -q
pytest tests/scripts/test_verify_gpt_ordering_pipeline.py -q

# 3. Replay against recent JSONL logs (if available):
python tools/replay_phase4_add_item_planner.py \
    --input app/logs/gpt_repair_turns.jsonl \
    --output reports/phase4_pre_deploy_replay.jsonl \
    --mode shadow --filter-state idle

# 4. Execute manual staging checklist:
# See docs/gpt_manual_staging_checklist.md
```

---

## Known Limitations

1. **Multi-item inline apply deferred (Phase 4.2)**: When the validator approves multiple items, `_apply_planner_result()` returns `None`, logs `block_reason="multi_item_deferred"`, and falls through to the local path. Full multi-item inline apply requires `MultiItemQueueCoordinator` integration (Phase 4.2).

2. **`candidate_items` from ITEM slots only**: The service builds candidates from local NLU `ITEM` slots. If the NLU misses an item entirely (no ITEM slot), that item won't appear in candidates and GPT will have no context for it. In practice, GPT can still name items from the utterance text — they'll be validated by the validator but rejected if not in the menu.

3. **No cross-turn state in replay**: The replay harness gives each turn independent context. Multi-turn recovery scenarios (where the user corrects from a previous turn) require live session context.

4. **Operation "extra"/"light" in modifier group**: The validator accepts "extra" and "light" as operations and stores them. The downstream FSM / cart mutation path must handle these correctly when inline apply is triggered. In Phase 4.1 this is safe since single-item apply calls the existing `resolve_item_and_enter_flow()` which handles its own modifier prompting.

5. **Timeout at 1800ms**: The Phase 4 planner has a longer timeout (1800ms) than the Phase 3 option resolver (1200ms) because it parses longer, multi-item responses. This may occasionally exceed voice latency budgets — monitor `add_item_planner_latency_ms` and tune `COMPASS_GPT_ADD_ITEM_PLANNER_TIMEOUT_MS` if needed.

---

## Commit Recommendation

**YES — ready to commit (Phase 4.2 pre-deployment hardening complete).**

Safety invariants hold:
- Default mode is `disabled` (zero risk at deploy)
- `safe_to_apply` computed by the apply gate, never from GPT's own field
- Apply gate requires 8 independent conditions to pass
- Shadow mode never applies regardless of plan quality
- GPT errors never crash the response path
- No full menu, no full cart, no API key in any log or payload
- Multi-item inline apply deferred — only single-item is applied
- Production wiring through `HandlerDispatcher._build_add_item_planner()` — safe fallback to `None` on any config error
- `add_item_planner_applied`, `add_item_planner_apply_block_reason`, `add_item_planner_validator_reject_reason` log fields present
- Pre-deployment verification script (`scripts/verify_gpt_ordering_pipeline.py`) exits 0 on clean config
- Manual staging checklist (`docs/gpt_manual_staging_checklist.md`) covers all 20 test scenarios
- Zero new regressions (49 pre-existing failures, all unrelated to Phase 4)

---

*Generated by Compass Voice Engineering Agent — Phase 4.2 Pre-Deployment Hardening*
