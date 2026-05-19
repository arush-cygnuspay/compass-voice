# Compass Voice — GPT Semantic Resolver Redesign

**Audience:** Principal architect / backend lead
**Status:** Proposal, ready for phased rollout
**Date:** 2026-05-19

---

# Executive Summary

**Architecture gap.** GPT is wired into the turn pipeline but architecturally inert. `GptRepairService.run()` is hardcoded to `applied=False` in every code path (`repair_service.py:384, 466, 520, 546`). It produces a fully-formed `GptRepairResult` with intent, slots, slot_corrections, items, and confidence — and then the result is logged to `turn_events.jsonl` and discarded. The deterministic path consumes the *local* model output coerced through three sequential rule-based rewriters (`idle_checkout_coercion → intent_coercion → contextual_control_resolver`), and `StateRouter.route()` never sees the GPT decision. Phase 3 (`option_resolver_service.py`) and Phase 4 (`add_item_planner_service.py`) exist as inline-applicable services but are gated `disabled` by default and not wired into the waiting-state handlers.

A second, independent gap: idle-state bare item phrases ("chicken burger with small coke", "tuna melt") rely on a fragile chain — local intent model outputs UNKNOWN → `IntentCoercionPolicy` rewrites to ADD_ITEM only if the slot model emitted an `ITEM` span *and* `MenuMatcher` finds evidence. Any slot-model miss (typo, novel paraphrase, phonetic confusion) silently collapses to UNKNOWN → `intent_not_allowed` reprompt. The user gets "Sorry, I didn't catch that" on phrases an elite counter clerk would handle instantly.

**Recommended GPT policy: hybrid (option C+ with shadow telemetry).**

1. GPT runs in **shadow mode on every meaningful turn** (text length ≥ 3, non-terminal state) — purely for logging/training. Cost is bounded by daily budget; latency is hidden via fire-and-forget background dispatch (already implemented as `all_shadow`).
2. GPT runs in **apply mode synchronously** when the turn is in one of four well-defined "GPT buckets" where deterministic NLU is structurally insufficient. Each bucket has a strict context packet, a strict JSON output schema, and a deterministic validator that gates application.
3. **Cart, lifecycle, and payment are never mutated by GPT output.** GPT produces a *plan*; deterministic handlers execute the plan only after `StateRouter` and `FlowControlPolicy` clear it.

This keeps p50 latency on the dominant happy path (high-confidence local intent) flat, opens GPT only where local NLU is structurally weak, and yields rich training data on every turn for future local-model improvement.

---

# Current Code Path

```
Twilio μ-law audio ──► Deepgram STT (streaming) ──► utterance text
                                                          │
                                                          ▼
                                  app/core/turn_engine.py :: TurnEngine.process_turn()
                                                          │
   ┌──────────────────────────────────────────────────────┴──────────────────────────────────┐
   │ 1. Terminal-state guards (COMPLETED, TRANSFERRING_TO_HUMAN_AGENT)                       │
   │ 2. Auto-payment-check fast path (text == "__auto_payment_check__")                      │
   │ 3. Order-type gate: if state == WAITING_FOR_ORDER_TYPE → mini-NLU + dispatch, return    │
   │ 4. NluOrchestrator.resolve()                                                            │
   │       └► app/nlu/nlu_resolver.py :: resolve_nlu()                                       │
   │             ├─ if state ∈ {CANCELLATION_CONFIRMATION, CONFIRMING_*}: rule-only CONFIRM/DENY│
   │             ├─ if state ∈ WAITING_STATES: rule extraction → slot model → intent=UNKNOWN  │
   │             └─ else: predict_intent_labels() (multihead model) + slot model              │
   │ 5. FlowGate._handle_phase3_control_shortcuts()                                          │
   │ 6. FlowGate._rewrite_confirming_order_to_idle_if_needed()                               │
   │ 7. FlowGate._apply_idle_shortcuts()                                                     │
   │ 8. FlowGate._rewrite_idle_unknown_menu_followup()                                       │
   │ 9. idle_checkout_coercion.coerce_idle_to_checkout()                                     │
   │10. intent_coercion.IntentCoercionPolicy.coerce()       ◄── bare-item → ADD_ITEM         │
   │11. contextual_control_resolver.resolve()  (Phase 3 v2)                                  │
   │12. GptRepairService.run()   ◄── all_shadow OR eligible_only; result.applied = False     │
   │       └► RepairPolicy.check() (triggers: UNKNOWN | conf_gap<0.20 | waiting+UNKNOWN ...) │
   │13. GPT fallback gate (apply_fallbacks=False by default → no-op)                         │
   │14. AddItemExtractorService.run() (shadow only)                                          │
   │15. FlowControlPolicy.evaluate() → BLOCK | CANCEL | HANDLE_READONLY_INTERRUPT | PASS     │
   │16. StateRouter.route(state, intent) ────► (state,intent) → handler_name                 │
   │17. HandlerDispatcher.dispatch(handler, intent, context, text, session)                  │
   │       └► Handler.handle() ──► HandlerResult(next_state, response_key, payload)          │
   │18. Optional: ItemQueueService drains multi-item queue                                   │
   │19. Optional: PaymentFlowOrchestrator processes payment events                           │
   │20. ResponseBuilder hydrates text                                                        │
   │21. TurnEventLogger.log(TurnEvent)                                                       │
   └──────────────────────────────────────────────────────┬──────────────────────────────────┘
                                                          ▼
                                                Deepgram TTS μ-law ─► Twilio
```

**Where local NLU runs:** `app/nlu/nlu_resolver.py :: resolve_nlu()` line ~280+.
**Local intent model:** `app/ml/intent/inference_intent.py :: predict_intent()` (multihead). Threshold `INTENT_MIN_CONF=0.55`. Below it → `Intent.UNKNOWN`.
**Local slot model:** `app/ml/slot/inference_slot.py :: predict_slots()` (spaCy NER, labels: ITEM, SIZE, SIDE, MODIFIER, QUANTITY, VARIANT). No per-slot confidence.
**Where GPT runs:** Step 12 above — *after* coercion, *before* `FlowControlPolicy` and `StateRouter`.
**Where final intent is chosen:** Implicit. The "final" intent is whatever survives steps 9–13. GPT's `repaired_intent` is *never* substituted (Phase 2 contract).
**Where cart is mutated:** Inside individual handlers (`AddItemHandler`, `WaitingForSideHandler`, etc.) via direct `session.cart.add_item()` / `remove_item()` calls. No central cart-mutation gate.

---

# Current GPT Usage

| Service                              | File                                           | Trigger                                                                   | Apply behavior            |
| ------------------------------------ | ---------------------------------------------- | ------------------------------------------------------------------------- | ------------------------- |
| `GptRepairService`                   | `app/nlu/semantic_repair/repair_service.py`    | `all_shadow` (every turn) or `eligible_only` (RepairPolicy.check passes)  | **Never applied** (`applied=False` hardcoded) |
| `AddItemExtractorService`            | `app/nlu/semantic_repair/add_item_service.py`  | `add_item_mode=="shadow"`                                                 | **Never applied** (shadow only) |
| `OptionResolverService`              | `app/nlu/semantic_repair/option_resolver_service.py` | `option_resolver_mode != "disabled"` (default disabled)             | Inline-applicable when enabled (not wired into waiting-state handlers yet) |
| `GptAddItemPlannerService`           | `app/nlu/semantic_repair/add_item_planner_service.py` | `add_item_planner_mode != "disabled"` (default disabled)            | Inline-applicable inside `AddItemHandler` when enabled |
| `SmartTurnPlanner`                   | (inside `AddItemHandler` execution pipeline)   | `SMART_TURN_PLANNER_ENABLED=true`                                         | Inline-applicable when enabled |

**Context GPT currently receives** (`repair_service.py` lines 436–452, via `build_messages`):

| Field                  | Source                                              | Quality |
| ---------------------- | --------------------------------------------------- | ------- |
| `utterance`            | `nlu.normalized_text`                               | Good |
| `state_name`           | current `ConversationState`                         | Good |
| `candidates`           | allowed intents in current state                    | Good |
| `current_prompt_field` | what question is open (e.g., "side", "size")        | Partial — not always populated |
| `current_item_name`    | pending_add_item.name                               | Good |
| `intent_candidates`    | top-K local model intents + confidences             | Good |
| `cart_summary`         | `{count, items: [name, qty]}`                       | Partial — no modifiers/sides |
| `slots`                | local slot model output                             | Good |
| `choices`              | available options for waiting state                 | Good (when populated) |
| `required_missing`     | required slots not yet captured                     | Good |
| `previous_turns`       | last 3 (role, text) tuples                          | Good |
| `last_cart_diff`       | **NOT PRESENT**                                     | Missing |
| `pending_action`       | **NOT PRESENT** (enum exists but unused)            | Missing |
| `relevant_menu_subset` | **NOT PRESENT** — GPT does not see menu candidates  | Missing |

**Critical gaps:** (a) GPT has no menu lexicon, so it cannot disambiguate "tuna melt" vs "tuna sandwich"; it can only echo what the slot model already extracted. (b) GPT has no `pending_action` or `last_cart_diff`, so it cannot reason about "remove the burger I just added". (c) GPT output is parsed into a typed result but `applied=False` is hardcoded — the architecture has no live path from `repaired_intent` → `StateRouter`.

---

# Recommended GPT Turn Resolver Architecture

```
                       ┌─────────────────────────────────────────┐
                       │            STT cleaned_text             │
                       └──────────────────┬──────────────────────┘
                                          ▼
                       ┌─────────────────────────────────────────┐
                       │   Local NLU Evidence Layer (existing)   │
                       │   - intent model: intent + top-K + conf │
                       │   - slot model: ITEM/SIZE/SIDE/MOD/QTY  │
                       │   - menu matcher: candidate items + conf│
                       └──────────────────┬──────────────────────┘
                                          ▼
                       ┌─────────────────────────────────────────┐
                       │      Turn Resolver Policy (NEW)         │
                       │   Decides bucket: 0..7  +  shadow_only? │
                       └──────────────────┬──────────────────────┘
                                          ▼
                       ┌─────────────────────────────────────────┐
                       │     GPT Semantic Interpreter (NEW)      │
                       │   - bucketed prompts (one per task)     │
                       │   - strict JSON output schema           │
                       │   - timeout-bounded, budget-bounded     │
                       └──────────────────┬──────────────────────┘
                                          ▼
                       ┌─────────────────────────────────────────┐
                       │     Deterministic Resolver (NEW)        │
                       │   1. is intent in state's allow-list?   │
                       │   2. do slots resolve to live menu?     │
                       │   3. is action legal vs OrderLifecycle? │
                       │   → choose final_intent / final_slots   │
                       │   → choose final_source (local|gpt|fb)  │
                       └──────────────────┬──────────────────────┘
                                          ▼
                       ┌─────────────────────────────────────────┐
                       │  FlowControlPolicy + StateRouter        │
                       │  Handler.handle() — cart mutations      │
                       └──────────────────┬──────────────────────┘
                                          ▼
                       ┌─────────────────────────────────────────┐
                       │       TurnEventLogger (extended)        │
                       └─────────────────────────────────────────┘
```

**Invariant:** GPT *interprets*; the deterministic resolver *decides*; handlers *execute*. GPT output that does not pass the resolver is logged as `final_source=local`, `repair_type=gpt_rejected`, never reaches the handler.

---

# GPT Buckets / Task Modes

Each bucket is a distinct prompt template and a distinct output schema. The resolver policy picks at most one bucket per turn.

## Bucket 0: `idle_menu_item_resolution`

**Trigger:** `state == IDLE` AND `last_response_key == "what_would_you_like"` (or first user turn after greeting) AND local intent ∈ {UNKNOWN, ADD_ITEM with conf<0.85, MENU_QUERY} AND text length ≥ 3.

**Context packet:**
```json
{
  "task": "idle_menu_item_resolution",
  "user_text": "...",
  "normalized_text": "...",
  "previous_assistant_prompt": "What would you like to order?",
  "previous_turns": [...],
  "local_intent": "UNKNOWN",
  "local_intent_confidence": 0.42,
  "local_intent_candidates": [{"intent":"ADD_ITEM","conf":0.41}, ...],
  "local_slots": [{"name":"ITEM","value":"chicken burger"}, ...],
  "cart_summary": {"count":0,"items":[]},
  "order_type": null,
  "menu_candidates": [
    {"item_id":"chicken_burger_1","name":"Chicken Burger","aliases":[...],"sides":[...],"modifiers":[...]},
    ...
  ]
}
```
`menu_candidates` is the *top-N (≤ 8)* of `MenuMatcher.candidates(normalized_text)` — not the full menu.

**Allowed intents (response):** `ADD_ITEM`, `BROWSE_MENU`, `ASK_PRICE`, `ASK_ITEM_INFO`, `GREETING`, `UNKNOWN`.

**Output schema:**
```json
{
  "task": "idle_menu_item_resolution",
  "intent": "ADD_ITEM",
  "item_plan": [
    {
      "item_id": "chicken_burger_1",
      "quantity": 1,
      "sides":     [{"group_id":"drinks","item_id":"coke_1","size":"small"}],
      "modifiers": [],
      "size": null,
      "variant": null,
      "source_span": "chicken burger with small coke"
    }
  ],
  "unresolved_spans": [],
  "confidence": 0.88,
  "reason": "Two menu items: Chicken Burger (chicken_burger_1) + Coke (coke_1) as side with size=small"
}
```

**Validation rules:**
1. Every `item_id` must exist in current menu.
2. Every `sides[*].item_id` must exist in the *picked item's* side group.
3. Every `modifiers[*].id` must exist in the *picked item's* modifier group.
4. `size` must be in the picked item's allowed sizes.
5. `confidence ≥ 0.70` to apply; below that, fall back to local + clarification reprompt.

**Fallback:** Emit `unknown_item_clarification` response, log `repair_type=gpt_unresolved`.

---

## Bucket 1: `state_intent_resolution`

**Trigger:** `state != IDLE` AND local intent confidence in [0.20, 0.85] AND state-allowed intents include local intent OR repair candidates.

**Context packet:** same as Bucket 0 + `current_state`, `allowed_intents_in_state`, `pending_action`, `pending_item_summary`.

**Allowed intents:** intersection of `INTENT_REGISTRY` and current state's allow-list.

**Output schema:**
```json
{ "task": "state_intent_resolution",
  "intent": "CONFIRM",
  "slots": {},
  "confidence": 0.91,
  "reason": "User said 'yeah do it' after confirm_order_summary prompt" }
```

**Validation:** `intent` must be in `allowed_intents_in_state`. Otherwise reject.

**Fallback:** Use local intent if confidence ≥ 0.55; else reprompt.

---

## Bucket 2: `option_resolution`

**Trigger:** `state ∈ {WAITING_FOR_SIZE, WAITING_FOR_SIDE, WAITING_FOR_MODIFIER, WAITING_FOR_SIDE_SIZE, WAITING_FOR_QUANTITY}` AND the deterministic option matcher returned no match OR multiple ambiguous matches OR user said an ordinal/positional phrase.

**Context packet:**
```json
{
  "task": "option_resolution",
  "user_text": "the second one",
  "previous_assistant_prompt": "Would you like fries, onion rings, or a salad?",
  "pending_item": {"name":"Chicken Burger","item_id":"..."},
  "current_group": {"group_id":"sides","name":"Side","min":1,"max":1},
  "available_choices": [
    {"item_id":"fries_1","name":"Fries","aliases":["french fries"],"size_required":true},
    {"item_id":"onion_rings_1","name":"Onion Rings","aliases":[]},
    {"item_id":"salad_1","name":"Salad","aliases":[]}
  ],
  "already_selected": [],
  "allowed_actions": ["select", "skip", "repeat", "cancel", "ask_options", "negate"]
}
```

**Output schema:**
```json
{ "task": "option_resolution",
  "action": "select",
  "selected_item_ids": ["onion_rings_1"],
  "selected_size": null,
  "negated_item_ids": [],
  "confidence": 0.93,
  "reason": "'the second one' refers to onion_rings_1 in current prompt order" }
```

**Validation:**
1. `action` in `allowed_actions`.
2. `selected_item_ids` all present in `available_choices`.
3. Selection count respects `[min, max]` bounds.
4. `negated_item_ids` must be in `already_selected`.

**Fallback:** `repeat_side_options`.

---

## Bucket 3: `multi_item_add_planning`

**Trigger:** `state == IDLE` (or `state == WAITING_FOR_*` with mid-item interrupt that survives `FlowControlPolicy`) AND local NLU extracted ≥ 2 ITEM spans OR text contains conjunction patterns ("and", "plus", "with") AND local heuristic planner returned ambiguous output.

**This replaces** the three-way overlap of `multi_item_order_planner.py`, `multi_item_parser.py`, and `SmartTurnPlanner` for the ambiguous case. For unambiguous cases, the local heuristic still wins (faster, free).

**Context packet:** same as Bucket 0 + full multi-span hint list.

**Output schema:** same as Bucket 0 but `item_plan` may have ≥ 2 items, and each item may carry a `dependent_items` array for explicitly-paired sides (resolves "large fries small onion rings tuna melt" → fries.size=large, onion_rings.size=small, tuna_melt standalone).

**Validation rules:** same as Bucket 0 + cross-item assignment must be consistent: a SIZE adjacent to a SIDE attaches to that SIDE, not the next ITEM.

**Fallback:** `compound_fallback_clarify_one_item`.

---

## Bucket 4: `correction_resolution`

**Trigger:** Local intent ∈ {REMOVE_ITEM, MODIFY_ITEM, REPLACE_ITEM, UNDO_LAST, CANCEL} OR text matches negation patterns ("not that", "no coke", "remove", "change", "actually").

**Context packet:** + `last_cart_diff`, full `cart_items_with_index` (item index, name, qty, modifiers, sides).

**Output schema:**
```json
{ "task": "correction_resolution",
  "action": "remove",
  "target_cart_index": 2,
  "target_item_id": "coke_1",
  "replacement": null,
  "confidence": 0.94,
  "reason": "'no coke' after adding Coke as side → remove coke from cart item idx=2" }
```

**Validation:**
1. `target_cart_index` must exist.
2. `target_item_id` must match cart item at that index (or be `null` for whole-item removal).
3. `replacement` (for MODIFY) must be a valid menu item.

**Fallback:** Ask "Which item would you like to remove?"

---

## Bucket 5: `checkout_resolution`

**Trigger:** Local intent ∈ {CHECKOUT, CONFIRM, DENY} AND state ∈ {IDLE-with-non-empty-cart, CONFIRMING_ORDER, WAITING_FOR_PAYMENT, WAITING_FOR_CHECKOUT_COMPLETION} AND text ambiguous (e.g., "yeah do it", "alright let's go").

**Context packet:** + `order_lifecycle_state` (cart count, order_type set, address set if delivery, payment_link_pending).

**Output schema:**
```json
{ "task": "checkout_resolution",
  "intent": "CONFIRM",
  "scope": "send_payment_link",
  "confidence": 0.96,
  "reason": "'yeah do it' after 'confirm_order_summary' prompt → confirm checkout" }
```

**Validation:** `intent` must be allowed by `OrderLifecycleGuard` (which we will *introduce* — see §"Files To Modify"). E.g., `CONFIRM` in `WAITING_FOR_PAYMENT` only valid if `payment_link_already_sent == True`.

**Fallback:** Reprompt with explicit yes/no.

---

## Bucket 6: `order_type_change`

**Trigger:** Text matches order-type lexicon ("delivery", "deliver", "pickup", "pick up", "I'll come get it", "make it delivery", "actually pickup") OR local intent == `CHANGE_ORDER_TYPE`.

**Context packet:** + `current_order_type`, `delivery_eligible`, `address_collected`.

**Output schema:**
```json
{ "task": "order_type_change",
  "new_order_type": "delivery",
  "confidence": 0.97,
  "reason": "'make it delivery' explicit" }
```

**Validation:**
1. `new_order_type ∈ {pickup, delivery}`.
2. If delivery, must trigger downstream `WAITING_FOR_DELIVERY_ELIGIBILITY`.

**Fallback:** Ask "Will this be pickup or delivery?"

---

## Bucket 7: `generic_repair`

**Trigger:** Catch-all for low-confidence turns that don't match Buckets 0–6 AND repeat_count ≥ 2 (user said something we didn't understand twice) AND state is not terminal.

**Context packet:** wide — everything available.

**Output schema:**
```json
{ "task": "generic_repair",
  "intent": "TRANSFER_TO_HUMAN",
  "slots": {},
  "confidence": 0.55,
  "reason": "User repeated unclear phrase 3 times; escalate" }
```

**Validation:** standard intent allow-list check.

**Fallback:** `transfer_to_human_agent`.

---

# Idle Natural Item Handling

**Current behavior.** In IDLE, local model emits ADD_ITEM only when training data covered the surface phrasing. For bare phrases ("tuna melt", "large fries", "6 piece wings"), the model frequently outputs UNKNOWN with confidence ~0.3–0.5. `IntentCoercionPolicy.coerce()` then rewrites UNKNOWN → ADD_ITEM only if **both** the slot model emitted an ITEM span **and** `MenuMatcher` found a menu match. Either failure mode (slot miss, menu miss) → reprompt.

**Recommended flow:**

1. After local NLU completes, run a **MenuMatcher pre-pass** on `normalized_text` regardless of intent. Output: ranked candidate items with confidence (fuzzy + alias + phonetic).
2. Compute `idle_item_evidence_score = max(slot_item_present, menu_top1_confidence)`.
3. If `state == IDLE` AND `idle_item_evidence_score ≥ 0.85` → final_intent = ADD_ITEM, final_source=local, skip GPT (fast path).
4. If `state == IDLE` AND `idle_item_evidence_score ∈ [0.40, 0.85)` → trigger **Bucket 0** (idle_menu_item_resolution). GPT receives top-N menu candidates and decides ADD_ITEM vs BROWSE_MENU vs ASK_PRICE.
5. If `state == IDLE` AND `idle_item_evidence_score < 0.40` → trigger Bucket 0 anyway but with broader candidate pool; if GPT also returns confidence < 0.70, emit `unknown_item_clarification`.

**Key principle:** The local model is fallback evidence, not the gate. The menu lexicon (deterministic) and GPT (semantic) are the gates. This removes the "I didn't catch that" failure mode on novel phrasings.

**Reuse:** `MenuMatcher` already exists at `app/menu/matcher.py`. Promote its candidate output into the turn context and into `intent_coercion.IntentCoercionPolicy`.

---

# Waiting State Handling

**Current behavior.** Waiting-state handlers (`WaitingForSideHandler`, `WaitingForModifierHandler`, `WaitingForSizeHandler`) own the turn unconditionally. They run a control-phrase classifier ("repeat", "skip", "done") then attempt exact/fuzzy/phonetic match of the user text against `pending.<group>.choices`. GPT is **not** consulted. `OptionResolverService` exists but is gated `disabled`.

**Recommended flow inside each waiting-state handler:**

```
1. Control-phrase classification (existing, fast, free)
   ├─ repeat / skip / done / cancel / negate → existing rule path
   └─ select → continue to 2

2. Deterministic option matcher (existing exact + fuzzy + phonetic)
   ├─ unique match with score ≥ 0.85 → APPLY, fast path
   ├─ multiple matches above 0.70 → ambiguity, trigger Bucket 2
   ├─ ordinal phrase ("the second one", "the first") → trigger Bucket 2
   └─ no match → trigger Bucket 2

3. Bucket 2: option_resolution GPT call (synchronous, 350ms timeout)
   ├─ GPT returns selected_item_ids → run resolver validation
   │     ├─ pass → APPLY
   │     └─ fail → log gpt_rejected, repeat_side_options
   ├─ GPT timeout → repeat_side_options (existing fallback)
   └─ GPT parse_error → repeat_side_options
```

**Cancel/checkout/order-type interrupts inside waiting states are handled at FlowControlPolicy** *before* the waiting-state handler runs (existing behavior — keep). The waiting-state handler only sees turns that survived FlowControlPolicy.

**Example mapping:**

| User text                    | Bucket 2 output                                    |
| ---------------------------- | -------------------------------------------------- |
| "the second one"             | `action=select, selected_item_ids=[<2nd in prompt>]` |
| "plain bun"                  | `action=select, selected_item_ids=["plain_bun"]` (or `selected_modifiers=[...]` for modifier group) |
| "no coke"                    | `action=negate, negated_item_ids=["coke_1"]` (if Coke already selected) OR `action=skip` (if Coke is the only side and user is refusing) |
| "yeah that"                  | `action=select` referring to last suggested option |
| "what do you have?"          | `action=ask_options` → reprompt with choices       |
| "skip it"                    | `action=skip` → skip-group policy applies          |

---

# Interrupt Handling

User interrupts must be routed by state, not by intent alone. Routing matrix:

| User says         | State                          | Route                                                                                                  |
| ----------------- | ------------------------------ | ------------------------------------------------------------------------------------------------------ |
| "cancel that"     | any active task                | `FlowControlPolicy.CANCEL` → `CANCELLATION_CONFIRMATION`                                              |
| "remove coke"     | IDLE (non-empty cart)          | Bucket 4 → `RemoveItemHandler`                                                                         |
| "remove coke"     | waiting_for_modifier           | If "coke" matches a *pending* side already selected → Bucket 2 negate; else `FlowControlPolicy.CANCEL` |
| "checkout"        | waiting_for_*                  | `FlowControlPolicy.BLOCK` → `checkout_blocked_finish_current_item`                                    |
| "checkout"        | IDLE (non-empty cart)          | `StateRouter` → `StartOrderHandler` → CONFIRMING_ORDER                                                 |
| "that's it"       | IDLE (non-empty cart)          | Bucket 5 → CONFIRM checkout                                                                            |
| "that's it"       | waiting_for_side (min met)     | Treated as "done" by control-phrase classifier (existing)                                              |
| "make it delivery"| any non-terminal               | Bucket 6 → `OrderTypeChangeHandler` (introduce; see Files To Modify)                                  |
| "actually pickup" | any non-terminal               | Bucket 6                                                                                               |
| "talk to someone" | any                            | `TRANSFER_TO_HUMAN_AGENT` (existing)                                                                   |

**Rule:** GPT does not invent new state transitions. It only chooses among the routes a given state legally permits. `OrderLifecycleGuard` (new) is the single source of truth for what's legal.

---

# Final Decision Policy

Pseudo-code for the new `TurnResolver.decide()`:

```python
def decide(
    *,
    state: ConversationState,
    local: LocalNluResult,
    menu: MenuCandidateResult,
    gpt: GptInterpretationResult | None,
    pending_action: PendingAction | None,
    cart_summary: CartSummary,
    order_lifecycle: OrderLifecycleState,
) -> FinalDecision:

    allowed = ALLOWED_INTENTS_BY_STATE[state]

    # 1. Fast path: high-confidence local intent that's allowed
    if local.intent in allowed and local.confidence >= HIGH_CONF and gpt is None:
        return FinalDecision(
            intent=local.intent,
            slots=local.slots,
            source="local",
            repair_type="no_repair",
            validation="pass",
        )

    # 2. GPT was called — validate its output
    if gpt is not None and gpt.parse_ok and not gpt.timeout:
        if gpt.intent in allowed and validate_slots(gpt.slots, menu, state):
            # 2a. GPT agrees with local → no_repair (logging fidelity)
            if gpt.intent == local.intent and slots_equal(gpt.slots, local.slots):
                return FinalDecision(gpt.intent, gpt.slots, "gpt", "no_repair", "pass")
            # 2b. GPT changed intent
            if gpt.intent != local.intent:
                return FinalDecision(gpt.intent, gpt.slots, "gpt", "intent_repair", "pass")
            # 2c. GPT changed slots
            return FinalDecision(gpt.intent, gpt.slots, "gpt", "slot_repair", "pass")
        # 2d. GPT output failed validation
        return _fallback(local, menu, state, reason="gpt_validation_failed")

    # 3. GPT timeout / disabled / not called
    if local.intent in allowed and local.confidence >= LOCAL_MIN_CONF:
        return FinalDecision(local.intent, local.slots, "local", "no_repair", "pass")

    # 4. No safe decision
    return _fallback(local, menu, state, reason="no_confident_intent")


def _fallback(local, menu, state, reason):
    return FinalDecision(
        intent=Intent.UNKNOWN,
        slots=[],
        source="fallback",
        repair_type="fallback",
        validation="fail",
        fallback_reason=reason,
    )
```

**Properties:**
- Deterministic given identical inputs.
- GPT can override local intent only if (a) GPT's intent is in the state's allow-list, and (b) slots resolve against the live menu.
- Cart, payment, order-lifecycle correctness is never delegated to GPT — `OrderLifecycleGuard` runs after `TurnResolver.decide()` and can still BLOCK or DOWNGRADE.

---

# Logging Schema

Extend `app/logging/turn_event_schema.py` and `turn_event_logger.py`. Single record per turn, JSONL, one source of truth. CSV exports regenerate from JSONL.

```jsonc
{
  "schema_version": "2.0",
  "timestamp_utc": "...",
  "ids": { "session_id": "...", "call_sid": "...", "stream_sid": "...", "store_id": "...", "company_id": "..." },
  "source": "call | chat | test | replay",

  "turn": { "turn_index": 12, "state_before": "WAITING_FOR_SIDE", "state_after": "WAITING_FOR_SIDE" },
  "previous_assistant_prompt": { "response_key": "ask_for_side", "text": "What side would you like?" },

  "asr": { "raw_text": "...", "cleaned_text": "...", "normalized_text": "..." },

  "local_nlu": {
    "model_main_intent": "ITEM",
    "model_sub_intent": "ADD_ITEM_SUB",
    "intent": "ADD_ITEM",
    "intent_confidence": 0.42,
    "intent_candidates": [{ "intent": "ADD_ITEM", "conf": 0.41 }, ...],
    "slots": [{ "name": "ITEM", "value": "chicken burger", "start": 0, "end": 14 }, ...]
  },

  "menu": {
    "candidates": [{ "item_id": "...", "name": "...", "confidence": 0.91 }, ...],
    "top1_confidence": 0.91
  },

  "coercion": {
    "idle_checkout_applied": false,
    "intent_coercion_applied": true,
    "intent_coercion_from": "UNKNOWN",
    "intent_coercion_to": "ADD_ITEM",
    "contextual_control_applied": false
  },

  "gpt": {
    "bucket": "idle_menu_item_resolution | state_intent_resolution | option_resolution | multi_item_add_planning | correction_resolution | checkout_resolution | order_type_change | generic_repair | none",
    "mode": "shadow | apply | not_called",
    "called": true,
    "model": "gpt-4o-mini-2024-07-18",
    "prompt_chars": 1834,
    "completion_chars": 217,
    "latency_ms": 312,
    "timeout": false,
    "parse_error": null,
    "intent": "ADD_ITEM",
    "slots": [...],
    "item_plan": [...],
    "confidence": 0.88,
    "reason": "..."
  },

  "final_decision": {
    "intent": "ADD_ITEM",
    "slots": [...],
    "source": "local | gpt | fallback",
    "repair_type": "no_repair | intent_repair | slot_repair | plan_repair | fallback | gpt_rejected",
    "gpt_changed_intent": false,
    "gpt_changed_slots": false,
    "validation": "pass | fail",
    "validation_reason": null,
    "fallback_reason": null
  },

  "flow_control": { "action": "PASS | BLOCK | CANCEL | HANDLE_READONLY_INTERRUPT", "reason": "..." },

  "lifecycle_guard": { "blocked": false, "reason": null },

  "handler": { "path": "add_item_handler", "result": "ok | reprompt | error", "next_state": "WAITING_FOR_SIDE" },

  "cart": {
    "before_hash": "sha1:...",
    "after_hash": "sha1:...",
    "diff": { "added": [...], "removed": [...], "modified": [...] }
  },

  "response": { "response_key": "ask_for_side", "spoken_text": "...", "internal_text": "..." },

  "latency": {
    "preprocess_ms": 8,
    "local_nlu_ms": 22,
    "menu_match_ms": 4,
    "coercion_ms": 1,
    "gpt_ms": 312,
    "resolver_ms": 2,
    "flow_ms": 1,
    "route_ms": 1,
    "handler_ms": 14,
    "total_ms": 365
  },

  "training": {
    "candidate_flag": true,
    "candidate_reason": "gpt_changed_intent | gpt_changed_slots | fallback_emitted | low_local_confidence | user_corrected_next_turn",
    "labels_for_supervision": { "true_intent": null, "true_slots": null }
  },

  "errors": []
}
```

**Gaps vs current schema:** Add `menu.*`, `coercion.*`, `gpt.bucket`, `gpt.mode`, `final_decision.source`, `final_decision.repair_type`, `lifecycle_guard.*`, `training.candidate_flag`.

**Redaction:** keep existing PII redaction (phone, email, payment links). Hash cart contents, do not store raw menu prices.

---

# Files To Modify

| File                                                                                  | Change                                                                                                                  |
| ------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| **NEW** `app/nlu/turn_resolver/turn_resolver.py`                                      | `TurnResolver.decide()` — single decision authority. Imports the buckets, runs validation, picks final intent/slots.    |
| **NEW** `app/nlu/turn_resolver/bucket_policy.py`                                      | `BucketPolicy.pick(state, local, menu, ...) -> Bucket | None`. Deterministic dispatcher to 0..7.                       |
| **NEW** `app/nlu/turn_resolver/validators.py`                                         | `validate_intent_allowed`, `validate_slots_against_menu`, `validate_option_selection`, `validate_lifecycle`.            |
| **NEW** `app/state_machine/policy/order_lifecycle_guard.py`                           | `OrderLifecycleGuard.check(intent, state, cart, payment_state) -> GuardDecision`. Centralizes checkout/payment legality. |
| `app/core/turn_engine.py` :: `process_turn`                                           | Replace steps 9–13 with: `menu_match → BucketPolicy.pick → GPT (if bucket) → TurnResolver.decide → OrderLifecycleGuard`. |
| `app/nlu/semantic_repair/repair_service.py`                                           | Generalize into a thin GPT client used by all buckets. Remove the `applied=False` hardcoding — `applied` is set by `TurnResolver`. Add per-bucket prompt routing. |
| `app/nlu/semantic_repair/build_messages.py` (or wherever `build_messages` lives)      | Add per-bucket prompt templates. Add `menu_candidates`, `pending_action`, `last_cart_diff` to context. |
| `app/state_machine/policy/intent_coercion.py`                                         | Delete or reduce to a deterministic-only safety net. Move idle-item logic into Bucket 0 + MenuMatcher pre-pass. |
| `app/state_machine/policy/idle_checkout_coercion.py`                                  | Keep as fast-path *before* GPT for high-confidence checkout phrases; otherwise let Bucket 5 decide. |
| `app/state_machine/policy/contextual_control_resolver.py`                             | Subsume into Bucket 1 (`state_intent_resolution`) prompts. Keep as fallback when GPT disabled.                          |
| `app/state_machine/handlers/item/add_item/waiting_for_side_handler.py`                | After local match fails, call `TurnResolver` (which calls Bucket 2) before reprompting.                                |
| `app/state_machine/handlers/item/add_item/waiting_for_modifier_handler.py`            | Same as above.                                                                                                          |
| `app/state_machine/handlers/item/add_item/waiting_for_size_handler.py`                | Same as above.                                                                                                          |
| `app/state_machine/handlers/item/add_item/add_item_handler.py`                        | Replace the 3-way planner cascade with: local heuristic → if ambiguous, Bucket 3 via `TurnResolver`. Remove `SmartTurnPlanner` and `GptAddItemPlannerService` direct call sites; both flow through the resolver now. |
| `app/state_machine/handlers/item/remove_item_handler.py`                              | Accept correction plan from Bucket 4 (target cart index already resolved).                                              |
| `app/state_machine/handlers/order/start_order_handler.py`                             | Consult `OrderLifecycleGuard` before transitioning to CONFIRMING_ORDER.                                                 |
| `app/state_machine/handlers/payment/waiting_for_payment_handler.py`                   | Same.                                                                                                                   |
| **NEW** `app/state_machine/handlers/order/order_type_change_handler.py`               | Handler for Bucket 6 outputs. Wire into `StateRouter` for `CHANGE_ORDER_TYPE` intent in non-terminal states.            |
| `app/state_machine/state_router.py`                                                   | Add `CHANGE_ORDER_TYPE` routing across IDLE and active task states (after FlowControlPolicy clears it).                 |
| `app/logging/turn_event_schema.py`                                                    | Bump `schema_version` to "2.0". Add `menu`, `coercion`, `gpt.bucket`, `gpt.mode`, `final_decision`, `lifecycle_guard`, `training.candidate_flag` fields. |
| `app/logging/turn_event_logger.py`                                                    | Populate new fields. Preserve existing PII redaction.                                                                   |
| `app/config/semantic_repair.py`                                                       | Add per-bucket apply toggles: `COMPASS_GPT_BUCKET_0_MODE`, `..._1_MODE`, etc. Each: `disabled | shadow | apply`. Per-bucket timeout. |
| `app/menu/matcher.py`                                                                 | Expose `candidates(text, top_n=8) -> list[(item_id, confidence, snippet)]` if not already present.                      |

---

# Tests To Add

Place under `tests/state_machine/turn_engine/` (new directory) and `tests/nlu/turn_resolver/`. Use the existing harness pattern (FakeMenuRepo, ConversationContext, direct `TurnEngine.process_turn`).

| # | Test                                                                                          | Bucket / path                |
| - | --------------------------------------------------------------------------------------------- | ---------------------------- |
| 1 | idle: "chicken burger with small coke" → ADD_ITEM with item_plan attaching coke as size=small | Bucket 0                     |
| 2 | idle: "double bacon burger combo with large fries" → ADD_ITEM, combo + fries.size=large       | Bucket 0 or 3                |
| 3 | idle: "tuna melt" → ADD_ITEM (single item, no slots besides ITEM)                             | Bucket 0                     |
| 4 | idle: "large fries" → ADD_ITEM, fries.size=large                                              | Bucket 0                     |
| 5 | idle: unknown text "asdkfj" with no menu match → Bucket 0 returns conf<0.70 → clarification   | Bucket 0 fallback            |
| 6 | waiting_for_modifier: "plain bun" → modifier=plain_bun selected                               | Bucket 2 select              |
| 7 | waiting_for_modifier: after prompting 3 options, "the second one" → 2nd in prompt order selected | Bucket 2 ordinal          |
| 8 | waiting_for_side: "small coke" → side=coke_1, size=small                                      | Bucket 2 select+size         |
| 9 | waiting_for_side: "no coke" when Coke already selected as side → negate (remove from selection); otherwise → skip group | Bucket 2 negate / skip |
| 10| pending item in waiting_for_modifier, user says "checkout" → FlowControlPolicy.BLOCK → `checkout_blocked_finish_current_item` | FlowControlPolicy |
| 11| CONFIRMING_ORDER state, "yeah do it" → Bucket 5 → CONFIRM → payment link sent                 | Bucket 5                     |
| 12| IDLE state, "yeah do it" with non-empty cart → Bucket 5 → CONFIRM checkout. IDLE empty cart → reprompt | Bucket 5            |
| 13| any non-terminal, "make it delivery" → Bucket 6 → order_type=delivery                         | Bucket 6                     |
| 14| any non-terminal, "I'll come get it" → Bucket 6 → order_type=pickup                           | Bucket 6                     |
| 15| local ADD_ITEM at conf=0.45 with wrong item_id, GPT at conf=0.92 with correct item_id → final_decision uses GPT, repair_type=slot_repair | Bucket 0 |
| 16| GPT returns same intent + same slots as local → final_decision.source=local OR gpt, repair_type=no_repair | TurnResolver       |
| 17| GPT changes intent (UNKNOWN→ADD_ITEM) → repair_type=intent_repair                             | TurnResolver                 |
| 18| GPT keeps intent but adds size slot → repair_type=slot_repair                                 | TurnResolver                 |
| 19| GPT timeout → final_source=local if local valid, else fallback                                | TurnResolver fallback        |
| 20| GPT returns intent not in state's allow-list (e.g., DELETE_ORDER in IDLE) → validation=fail, gpt_rejected, fall back to local | TurnResolver validation |
| 21| GPT returns item_id that does not exist in current menu → validation=fail, gpt_rejected       | TurnResolver validation      |
| 22| Shadow-only bucket: GPT runs, applied=False, logged with `gpt.mode=shadow`                    | Logging                      |
| 23| Logs contain `final_decision.source`, `repair_type`, `gpt.bucket`, `cart.diff` for every turn | Logging schema               |
| 24| Two-item turn: "chicken burger and a coke" → Bucket 3 (or local heuristic if confident) → 2 items added | Bucket 3           |
| 25| "large fries small onion rings tuna melt" → Bucket 3, sizes correctly attached to the right sides/items | Bucket 3 cross-attach |

---

# Minimal Safe Implementation Plan

Five-phase rollout, each phase independently shippable and individually rollback-able via env flag.

**Phase 1 — Shadow telemetry baseline (1 week, zero risk).**
- Set `COMPASS_GPT_CALL_MODE=all_shadow` in staging.
- Extend `gpt_log_record_builder.py` to populate the new `gpt.bucket` and `final_decision.*` shadow fields.
- No behavior change. Goal: collect data to size buckets and tune confidence thresholds.

**Phase 2 — Bucket 0 apply: idle natural items (1 week).**
- Add `MenuMatcher.candidates()` pre-pass to `TurnEngine.process_turn`.
- Wire Bucket 0 with `COMPASS_GPT_BUCKET_0_MODE=apply` behind a per-store rollout flag.
- Validator: every GPT-proposed item_id must be in menu.
- Rollback: set mode back to `shadow`.

**Phase 3 — Bucket 2 apply: waiting-state option resolution (1 week).**
- Wire `OptionResolverService` (already exists, gated) into `WaitingForSideHandler`, `WaitingForModifierHandler`, `WaitingForSizeHandler` after deterministic match fails.
- Synchronous, 350ms timeout.
- Rollback: per-state mode flag back to `disabled`.

**Phase 4 — Bucket 3 apply: multi-item planning (1 week).**
- Wire `GptAddItemPlannerService` (already exists, gated) into `AddItemHandler` only for the ambiguous-multi-item branch. High-confidence single-item path stays on local heuristic.
- Rollback: per-handler flag.

**Phase 5 — Buckets 4, 5, 6, 1, 7 (2 weeks).**
- Roll out remaining buckets one at a time.
- Each release must include the corresponding tests from §Tests To Add.

**Phase 6 — Migrate logging schema to v2 + ingest training pipeline (parallel).**
- Bump `schema_version` to "2.0" on the same release as Phase 2.
- Add `training.candidate_flag` to flag turns where GPT diverged from local — use these for re-training the local intent/slot models.

---

# Recommended Claude Code Prompt

> Implement the GPT semantic resolver architecture defined in `docs/architecture/gpt_semantic_resolver_redesign.md`. Start with Phase 2: idle natural item handling.
>
> Specifically:
> 1. Create `app/nlu/turn_resolver/{turn_resolver.py,bucket_policy.py,validators.py}` and `app/state_machine/policy/order_lifecycle_guard.py` per the file specs in the redesign doc.
> 2. Modify `app/core/turn_engine.py :: process_turn` to replace lines 1203–1397 (the coercion + GPT-shadow block) with `menu_match → BucketPolicy.pick → optional GPT → TurnResolver.decide → OrderLifecycleGuard`.
> 3. Implement Bucket 0 (`idle_menu_item_resolution`) end-to-end: prompt template in `build_messages.py`, output schema in `gpt_repair_result.py`, validator in `validators.py`, apply path through `TurnResolver`.
> 4. Add `MenuMatcher.candidates(text, top_n=8)` if not present at `app/menu/matcher.py`.
> 5. Bump `app/logging/turn_event_schema.py` to schema_version "2.0" with the fields listed in the redesign doc; update `turn_event_logger.py` to populate them.
> 6. Add env flags in `app/config/semantic_repair.py`: `COMPASS_GPT_BUCKET_0_MODE` (`disabled | shadow | apply`), `COMPASS_GPT_BUCKET_0_TIMEOUT_MS` (default 350).
> 7. Add tests #1–#5 and #15–#23 from the redesign doc under `tests/nlu/turn_resolver/`.
>
> Constraints: do not touch transport layer (`app/realtime/`). All cart mutations remain in handlers. GPT must never set `applied=True` directly — `TurnResolver` owns that decision. Backwards-compat: when `COMPASS_GPT_BUCKET_0_MODE=disabled`, behavior must be byte-identical to current main.

---

# Definition of Done

- Natural item phrases in idle work — Bucket 0 routes "chicken burger with small coke", "tuna melt", "large fries", "6 piece wings" to ADD_ITEM without an explicit "add" verb and with a validated item_plan.
- GPT interpretation is logged on every meaningful turn (text length ≥ 3, non-terminal state) via `all_shadow` mode, even when not applied.
- GPT can correct local intent and slots; corrections are visible in `final_decision.repair_type ∈ {intent_repair, slot_repair, plan_repair}`.
- Final action is state-validated: every applied GPT output passed `TurnResolver` validators (intent ∈ state allow-list, slots resolve to menu, lifecycle legal).
- Cart, order, and payment lifecycle remain deterministic — no code path calls `session.cart.*` based on GPT output without going through a handler.
- Existing `tests/state_machine/handlers/item/add_item/*` suite continues to pass (regression bar).
- New tests #1–#25 pass.
- p50 turn latency unchanged on the fast path (high-confidence local intent skips GPT). p95 latency increase ≤ 400ms on bucket-triggering turns.

---

*End of document.*
