# GPT Ordering Pipeline — Manual Staging Checklist

**Date**: 2026-05-18  
**Branch**: `feature/app-flow-v2`  
**Scope**: Phase 3 (Option Resolver) + Phase 4 (Add-Item Planner)  
**Applies to**: Staging environment before any production promotion

---

## How to use this checklist

1. Deploy the branch to staging with the env vars shown per section.
2. Make test calls using an SIP softphone or Twilio test harness connected to the staging endpoint.
3. After each call, check staging logs (JSONL + structured trace) for the expected fields.
4. Mark each item **PASS / FAIL / SKIP** with a timestamp and tester initials.
5. All items must be **PASS** before promoting to production.

**Log verification** — after each test, confirm with:
```bash
tail -f app/logs/gpt_repair_turns.jsonl | python -m json.tool | grep -E "decision|applied|latency|reject"
```

---

## Env var reference

| Variable | Disabled | Shadow | Inline |
|----------|----------|--------|--------|
| `COMPASS_GPT_OPTION_RESOLVER_MODE` | `disabled` | `shadow` | `inline` |
| `COMPASS_GPT_ADD_ITEM_PLANNER_MODE` | `disabled` | `shadow` | `inline` |
| `OPENAI_API_KEY` | not needed | required | required |

---

## A. Disabled Mode — Baseline Safety

**Env vars:**
```
COMPASS_GPT_OPTION_RESOLVER_MODE=disabled
COMPASS_GPT_ADD_ITEM_PLANNER_MODE=disabled
```

### A1 — Simple add-item utterance

**Call script:** "I'd like a bourbon burger"  
**Expected FSM path:** IDLE → ADD_ITEM handler → item found → WAITING_FOR_MODIFIER or CONFIRMING  
**Expected log fields:**
- `add_item_planner_called`: `false` (no GPT called)
- `option_resolver_called`: `false`  
**Expected response:** "Great, I found Bourbon Burger. [modifier/side prompt or confirmation]"  
**Must NOT occur:** Any `add_item_planner_result` log event  

- [ ] PASS / FAIL — _____________________

### A2 — Simple modifier answer

**Setup:** Session must be in WAITING_FOR_MODIFIER state (follow A1 if it asks for modifier)  
**Call script:** "mozzarella"  
**Expected FSM path:** WAITING_FOR_MODIFIER → exact match → CONFIRMING or WAITING_FOR_SIDE  
**Expected log fields:** `option_resolver_called`: `false`  
**Expected response:** Confirms modifier or asks for next side/size  

- [ ] PASS / FAIL — _____________________

### A3 — Checkout deterministic path

**Setup:** Add item, complete modifier/side selection  
**Call script:** "That's all" → "Yes, confirm my order"  
**Expected FSM path:** IDLE → CONFIRMING_ORDER → WAITING_FOR_PAYMENT  
**Must NOT occur:** Any GPT call in checkout/payment flow  
**Expected response:** Order summary read back, then payment prompt  

- [ ] PASS / FAIL — _____________________

---

## B. Option Resolver — Shadow Mode

**Env vars:**
```
COMPASS_GPT_OPTION_RESOLVER_MODE=shadow
COMPASS_GPT_ADD_ITEM_PLANNER_MODE=disabled
OPENAI_API_KEY=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
```

### B1 — Phonetic mismatch: "macarola cheese"

**Setup:** Session in WAITING_FOR_MODIFIER for a modifier group containing "Mozzarella Cheese"  
**Call script:** "macarola cheese"  
**Expected:**
- Local matcher fails to resolve  
- GPT option resolver called (shadow)  
- Decision: `"select_option"`, `selected_names: ["Mozzarella Cheese"]` (or similar)  
- **Result NOT applied** — bot still reprompts or local path handles  
**Expected log fields:**
- `option_resolver_called`: `true`
- `option_resolver_decision`: `"select_option"`  
- `option_resolver_applied`: `false`  
- `option_resolver_safe_to_apply`: `false` (shadow forces false)  
**Expected response:** Reprompt (local deterministic behavior unchanged)  

- [ ] PASS / FAIL — _____________________

### B2 — Phonetic mismatch: "mozarella" (one 'z')

**Setup:** Same modifier group as B1  
**Call script:** "mozarella"  
**Expected:** Same shadow behavior — GPT logs resolution but does NOT apply  
**Expected log fields:** `option_resolver_called: true`, `option_resolver_applied: false`  

- [ ] PASS / FAIL — _____________________

### B3 — Negation modifier: "no, mozzarella"

**Setup:** Same modifier group  
**Call script:** "no, mozzarella" (user is affirming after hesitation)  
**Expected:** GPT processes utterance, logs intent — not applied  
**Log must show:** `option_resolver_decision` populated, `option_resolver_applied: false`  

- [ ] PASS / FAIL — _____________________

---

## C. Option Resolver — Inline Mode

**Env vars:**
```
COMPASS_GPT_OPTION_RESOLVER_MODE=inline
COMPASS_GPT_ADD_ITEM_PLANNER_MODE=disabled
OPENAI_API_KEY=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
```

> ⚠ **Inline mode applies GPT results** — validate validator behavior before enabling.

### C1 — Phonetic mismatch resolved: "macarola cheese"

**Setup:** Session in WAITING_FOR_MODIFIER for modifier group with "Mozzarella Cheese"  
**Call script:** "macarola cheese"  
**Expected (happy path):**
- GPT resolves to "Mozzarella Cheese"  
- Validator approves (confidence ≥ 0.75, name exists in group)  
- Modifier applied → FSM advances  
**Expected log fields:**
- `option_resolver_called`: `true`  
- `option_resolver_decision`: `"select_option"`  
- `option_resolver_confidence`: ≥ 0.75  
- `option_resolver_safe_to_apply`: `true`  
- `option_resolver_applied`: `true`  
**Expected response:** Confirms modifier selection, asks for next group or confirms item  

- [ ] PASS / FAIL — _____________________

### C2 — Low-confidence response not applied

**Call script:** Mumble or severely mangled option name  
**Expected:**
- GPT called but returns low confidence or `"no_match"`  
- `option_resolver_safe_to_apply`: `false`  
- Bot reprompts (local path)  

- [ ] PASS / FAIL — _____________________

---

## D. Add-Item Planner — Shadow Mode

**Env vars:**
```
COMPASS_GPT_OPTION_RESOLVER_MODE=disabled
COMPASS_GPT_ADD_ITEM_PLANNER_MODE=shadow
OPENAI_API_KEY=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
```

### D1 — Complex multi-entity utterance

**Call script:** "chicken burger with mozzarella, onions, mayo, and a coke"  
**Expected:**
- `is_complex_utterance` → True (comma, "with", "and")  
- Planner called (shadow), GPT plans items + modifiers  
- Result logged, **never applied**  
- Local multi-item parser or single-item path runs normally  
**Expected log fields:**
- `add_item_planner_called`: `true`  
- `add_item_planner_safe_to_apply`: `false` (shadow forces false)  
- `add_item_planner_decision`: `"add_items"` or `"clarify"`  
**Expected response:** Whatever the local deterministic path produces  

- [ ] PASS / FAIL — _____________________

### D2 — Multi-item with conditional modifier

**Call script:** "two chicken sandwiches, one with Swiss, one plain"  
**Expected:**
- Complexity detected (comma, number words, "with")  
- Planner called (shadow), GPT emits two items  
- Result logged, never applied  
**Log must show:** `add_item_planner_called: true`, `add_item_planner_applied` NOT present (shadow event 2 not emitted)  

- [ ] PASS / FAIL — _____________________

### D3 — Simple utterance bypasses planner

**Call script:** "bourbon burger"  
**Expected:**
- `is_complex_utterance` → False  
- `simple_high_confidence_local` bypass if NLU confidence ≥ 0.85  
- Planner NOT called (or called with shadow but route=NO_GPT)  
**Log must show:** No `add_item_planner_result` event, OR event with `add_item_planner_called: false`  

- [ ] PASS / FAIL — _____________________

---

## E. Add-Item Planner — Inline Mode

**Env vars:**
```
COMPASS_GPT_OPTION_RESOLVER_MODE=shadow
COMPASS_GPT_ADD_ITEM_PLANNER_MODE=inline
OPENAI_API_KEY=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
```

> ⚠ **Inline mode applies single-item GPT plans** — validate apply gate behavior.  
> Multi-item is always deferred (`multi_item_deferred`) in Phase 4.1.

### E1 — Single item with modifiers (happy path)

**Call script:** "chicken burger with mozzarella"  
**Expected (happy path):**
- Complexity detected ("with")  
- Planner called (inline), GPT returns single item + modifier  
- Validator approves (item exists in menu, modifier in group, confidence ≥ 0.75)  
- Apply gate passes all 8 conditions  
- Plan applied: `_apply_planner_result()` returns HandlerResult  
**Expected log fields:**
- `add_item_planner_called`: `true`  
- `add_item_planner_decision`: `"add_items"`  
- `add_item_planner_safe_to_apply`: `true`  
- `add_item_planner_applied`: `true` (Event 2)  
- `add_item_planner_apply_block_reason`: `null`  
**Expected response:** Confirms item + modifier, asks for next step (side/size/confirm)  

- [ ] PASS / FAIL — _____________________

### E2 — Multi-item utterance (deferred)

**Call script:** "Hawaiian pizza and two cokes"  
**Expected:**
- Complexity detected ("and", number word)  
- Planner called (inline), GPT returns 2 items  
- Validator approves both items  
- `_apply_planner_result()` returns None (multi-item deferred)  
- Falls through to local path  
**Expected log fields:**
- `add_item_planner_safe_to_apply`: `true`  
- `add_item_planner_applied`: `false` (Event 2)  
- `add_item_planner_apply_block_reason`: `"multi_item_deferred"`  
**Expected response:** Whatever local multi-item path produces  

- [ ] PASS / FAIL — _____________________

### E3 — Item not on menu (validator blocks)

**Call script:** "unicorn burger supreme"  
**Expected:**
- Planner called, GPT may return item  
- Validator: `item_not_on_menu` blocking warning  
- Apply gate fails (gate 7: blocking warnings)  
- `safe_to_apply`: `false`  
**Log must show:** `add_item_planner_validator_reject_reason` populated  
**Expected response:** Item not found or clarification prompt  

- [ ] PASS / FAIL — _____________________

---

## F. Failure Cases

### F1 — Nonsense / noise utterance

**Call script:** "aaasdfkjhqwerty"  
**Expected:**
- If planner is triggered: GPT returns `"no_repair"` or `"unclear"`  
- `safe_to_apply`: `false`  
- No crash  
- Local fallback path runs  

- [ ] PASS / FAIL — _____________________

### F2 — Unsupported / hallucinated item

**Call script:** "a mars bar with extra syrup"  
**Expected:**
- Planner called, validator rejects (not on menu)  
- Validator block: `item_not_on_menu`  
- Local path returns not-found response  

- [ ] PASS / FAIL — _____________________

### F3 — Invalid modifier for item

**Call script:** "chicken burger with extra guacamole" (guacamole not a valid modifier for this item)  
**Expected:**
- Planner may plan the item  
- Validator: `modifier_not_found` (non-blocking) OR item accepted with note  
- No crash  

- [ ] PASS / FAIL — _____________________

### F4 — API key absent with inline mode

**Setup:** Unset OPENAI_API_KEY while mode=inline  
**Call script:** "chicken burger with mozzarella"  
**Expected:**
- GPT call skipped (`skipped_reason: "missing_api_key"`)  
- Local path handles turn normally  
- No crash, no error response to caller  
**Log must show:** `add_item_planner_called: false`, `skipped_reason: "missing_api_key"`  

- [ ] PASS / FAIL — _____________________

### F5 — GPT timeout

**Setup:** Set `COMPASS_GPT_ADD_ITEM_PLANNER_TIMEOUT_MS=1` (1ms — will always timeout)  
**Call script:** "chicken burger with mozzarella"  
**Expected:**
- GPT call starts but times out  
- Result: `timed_out: true`, apply gate fails gate 3  
- Local path handles turn  
- No crash  

- [ ] PASS / FAIL — _____________________

### F6 — Repeated modifier reprompt (option resolver repeat-loop recovery)

**Setup:** Option resolver mode=inline, session in WAITING_FOR_MODIFIER  
**Script:** Give wrong modifier name 2+ times  
**Expected:**
- After `option_resolver_repeat_threshold` missed reprompts, inline GPT triggered  
- If GPT resolves → applies  
- If GPT fails → continues local reprompt  
- No infinite loop  

- [ ] PASS / FAIL — _____________________

---

## Summary Scorecard

| Section | Tests | Passed | Failed | Skipped |
|---------|-------|--------|--------|---------|
| A. Disabled | 3 | | | |
| B. Option Resolver Shadow | 3 | | | |
| C. Option Resolver Inline | 2 | | | |
| D. Add-Item Planner Shadow | 3 | | | |
| E. Add-Item Planner Inline | 3 | | | |
| F. Failure Cases | 6 | | | |
| **Total** | **20** | | | |

**Tester:** ___________________  
**Date:** ___________________  
**Environment:** ___________________  
**Build/commit:** ___________________  

**Decision:** ☐ APPROVED FOR PRODUCTION  ☐ BLOCKED — see failures above  

---

## Log field verification checklist

After each section, verify structured logs contain:

```
Option resolver:
  option_resolver_mode       ✓/✗
  option_resolver_route_reason ✓/✗
  option_resolver_called     ✓/✗
  option_resolver_decision   ✓/✗
  option_resolver_confidence ✓/✗
  option_resolver_safe_to_apply ✓/✗
  option_resolver_applied    ✓/✗
  option_resolver_latency_ms ✓/✗

Add-item planner (Event 1):
  add_item_planner_mode          ✓/✗
  add_item_planner_route_reason  ✓/✗
  add_item_planner_called        ✓/✗
  add_item_planner_decision      ✓/✗
  add_item_planner_confidence    ✓/✗
  add_item_planner_validator_passed ✓/✗
  add_item_planner_validator_reject_reason ✓/✗
  add_item_planner_safe_to_apply ✓/✗
  add_item_planner_latency_ms    ✓/✗

Add-item planner (Event 2 — only when safe_to_apply=True):
  add_item_planner_applied           ✓/✗
  add_item_planner_apply_block_reason ✓/✗
```
