# Compass Voice — Real-World Conversation Audit
**Date:** 2026-05-06 · **Auditor:** AI Engineering Audit (read-only, no code changes)
**Method:** Static deep-audit of FSM, transport, NLU, handlers, payment/delivery flows + reconciliation against the prior `Compass_Voice_NLU_Audit_Report.md` (2026-05-07) and `Compass_Voice_Response_Review.md` (2026-05-06).
**Scope blocked:** Live test execution — `pytest` could not be installed in the audit sandbox (proxy 403). Findings are static + log-evidence based.

---

## 1. Executive Summary

**Overall quality score: 6.5 / 10**
The architecture is **strong** — turn-locks, playback-generation cancellation, deterministic FSM, separation of transport and business logic. The conversational surface is **rough** — robotic phrasing, unguarded reprompt loops in payment/delivery, and a hardcoded "ready in 25 minutes" line that is unsafe for a live demo.

**Demo readiness: NOT READY** — three P0 blockers ship with the current build.

### Top 5 Blockers (must fix before demo)
1. **Hardcoded ETA "Will be ready in 25 minutes."** at `app/core/response_builder.py:466` — bot lies on every order completion.
2. **Checkout summary double-prompt + missing punctuation** — `cart_responses.py::render_checkout_review_summary` ends with `"Should I place the order"` (no `?`) and `_confirm_order_summary` then appends `"Would you like to checkout?"` → two questions back-to-back.
3. **Pickup SMS permission has no reprompt counter / no implicit-affirm fallback** (`waiting_for_pickup_sms_permission_handler.py:90-91`). Confirmed loop in production logs (2026-05-01: 3 consecutive re-prompts despite explicit "Yes. Send it."). Also confirms existing audit §7.1.
4. **"I couldn't find <command verb>" false positives** still in code (`prefill_orchestrator._collapse_unresolved_for_feedback` denylist approach). Reproducible: "Can you please add beef tacos" → "I couldn't find can you please add. Beef Tacos added." Existing audit §4.
5. **Side/modifier prompts mis-grammar** — `"Any cheese would you like, like American Cheese …"` (verb embedded in NP) and **wrong noun** `"Any burger would you like, like Mayo?"` (parent-item word leaking into prompt). Existing audit §2.

### Top 5 Non-Blocking Improvements
1. Add reprompt-escalation counters to `waiting_for_payment`, `waiting_for_pickup_sms_permission`, `waiting_for_order_type`, `waiting_for_delivery_address_collection` (4 states currently lack any escalation policy).
2. Switch unresolved-phrase filtering from denylist to **menu-vocab allowlist** (existing audit §4 — confirmed unfixed).
3. Strip every "Yes or no" / "Just say yes or no" tail from confirmation prompts (existing audit §1 — confirmed unfixed; 7 sites).
4. Add a **rejected-candidate denylist** on `ConversationContext` so the same "Did you mean X?" suggestion never repeats after rejection.
5. Add a **2 s watchdog on Twilio `mark` ACK** in `voice_stream_server.py` — currently waits indefinitely; if Twilio drops the mark, the pending-utterance queue never drains.

---

## 2. Coverage Map

| Area | Scenarios audited | Pass / Fail | Evidence | Risk |
|------|------|---|---|----|
| Startup / session | landline-vs-mobile, pickup/delivery prompt, resume | PARTIAL PASS | twilio_server.py:76, voice_stream_server.py, resume_prompt_builder.py | LOW |
| Turn-taking / barge-in | turn-lock, playback-generation, premature input, dedup | PASS | conversation_session.py:152/211/281, voice_stream_server.py:762/879/917 | LOW |
| Single-item order | default qty=1, ambiguity | PASS | confirming_handler.py, ordering_decision_engine.py | LOW |
| Quantity handling | "two", "I said two", QUANTITY-vs-AFFIRM priority | PASS | waiting_for_quantity_handler.py:84-112 | LOW |
| Multi-item | "burger and fries and coke" — queue drain | PASS w/ caveat | item_queue_service.py:96-169 | MED (no slot-segmentation telemetry) |
| Item + side + modifier ("with") | "chicken taco with Coke and American cheese" | FAIL on edge cases | multi_item_parser.py:32-70 | HIGH |
| Required side/modifier prompts | grammar, noun selection | FAIL (P0 phrasing) | sides.py / modifiers.py `ask_for_side` | HIGH |
| Required min_selector enforcement | "skip" on required group | PASS | group_skip_policy.py:27-51 | LOW |
| Not-on-menu / fallback | category fallback, near-miss | PARTIAL PASS | ordering_decision_engine.py:218 (max=3) | MED (no rejected-candidate tracking) |
| Pickup checkout / SMS | "send link" vs "pay there" classification | PASS for clean inputs | pickup_sms_resolver.py:84-159 | MED (no reprompt counter — loop possible) |
| Delivery flow | area→ZIP→eligibility→checkout | PASS | waiting_for_delivery_eligibility_handler.py, confirm_order_handler.py:369-410 | LOW |
| Order modification | corrections, cancel | PASS | waiting_for_quantity_handler, cart_handlers, prepayment_correction_support | LOW |
| Noise / "uh / hmm / wait" | filler-only filter | PASS | utterance_filter.py:199-237 | LOW |
| Hardcoded ETA / forbidden phrases | "ready in 25 minutes" | **FAIL P0** | response_builder.py:466 | **CRITICAL** |
| Confirmation phrasing | "Yes or no" tails, summary `?` missing | **FAIL P0** | response_builder.py:228-435; cart_responses.py:62-64 | **HIGH** |

---

## 3. Conversation Test Results (key scenarios)

> Each row: scenario → expected → observed (from code path tracing or production NLU log evidence in `Compass_Voice_NLU_Audit_Report.md`).

### S1 — "Can you please add beef tacos?"
- **Expected:** "Beef Tacos added. Anything else?"
- **Actual:** *"I couldn't find can you please add. Beef Tacos added. Would you like anything else?"* (NLU log row 10, 2026-05-07).
- **Status:** FAIL.
- **Root cause:** Denylist filter in `prefill_orchestrator._collapse_unresolved_for_feedback` leaves "you", "add" tokens after stripping; phrase echoes as unmatched.
- **Files:** `app/state_machine/handlers/item/add_item/prefill_orchestrator.py:882-938`. Same bug class in `side_group_resolver.py:218-248` and `modifier_group_resolver.py:240-266`.
- **Priority:** P0.

### S2 — "Yes. Send it." (in waiting_for_pickup_sms_permission)
- **Expected:** SMS sent → `COMPLETED`.
- **Actual:** Bot re-prompts the SMS permission question 3 turns in a row (rows 908-910, 2026-05-01). `pred_sub_intent=checkout` mis-fired and the resolver returned UNKNOWN.
- **Status:** FAIL (loop).
- **Root cause:** `waiting_for_pickup_sms_permission_handler.py:47-91` has **no reprompt counter** and **no implicit-affirm fallback** when the literal token "yes" is in `normalized_text`.
- **Priority:** P0.

### S3 — "Korean tacos" → bot disambiguates → "Korean taco spicy chicken"
- **Expected:** Add Korean Tacos – Spicy Chicken. (Disambiguating slot: MODIFIER=spicy chicken.)
- **Actual:** *"Korean Tacos - Spicy Chicken, right? Yes or no."* — secondary confirmation despite the disambiguating slot.
- **Status:** FAIL (wasted turn).
- **Root cause:** `confirming_handler._resolve_candidate_item_from_confirmation` only resolves within the offered shortlist; it does not skip secondary confirmation when the user re-mentions the candidate with at least one disambiguating slot.
- **Priority:** P1.

### S4 — "Chicken taco with Coke and American cheese"
- **Expected:** 1 item: Chicken Taco; side = Coke; modifier = American Cheese.
- **Actual:** Coke is parsed as ITEM by the slot model → `_ITEM_SEPARATORS` (multi_item_parser.py:32-70) splits "with Coke" into a new segment → 2 items added to queue.
- **Status:** FAIL on this phrasing class.
- **Root cause:** Multi-item parser has no semantic side-vs-modifier scope detection. "with" is treated as an item separator if the slot model emits an ITEM after it. The 60-char lookback (`_slot_looks_attached`, line 162-220) does not factor in side/modifier-group membership of the parent item.
- **Priority:** P1 (failure depends on slot-model output for the specific menu).

### S5 — "I want a chicken taco" (single item, no qty)
- **Expected:** "Chicken Taco added. Anything else?" (default qty=1, no "How many?" prompt).
- **Actual:** PASS — confirmed via `quantity_resolver.py` and `waiting_for_quantity_handler` short-circuit when intent is unambiguous and qty omitted.
- **Status:** PASS.

### S6 — "Two." while in waiting_for_quantity (intent often classified as AFFIRM)
- **Expected:** Quantity=2 wins.
- **Actual:** PASS. `waiting_for_quantity_handler.py:84-112` explicitly states: *"QUANTITY slot / text takes priority over intent labels … must win before any intent-based re-ask logic fires."*
- **Status:** PASS.

### S7 — "I want some burgers" (ambiguous quantity word)
- **Expected:** Quantity clarification.
- **Actual:** Likely PASS — `quantity_parser.NUMBER_WORDS` does not map "some" / "a few" → handler enters `waiting_for_quantity` and re-asks. Not directly evidenced; recommend explicit test.
- **Status:** PASS (inferred).

### S8 — "Add 2 burgers, 3 fries, and a large Coke"
- **Expected:** 3 items with bound quantities; queue drains.
- **Actual:** PASS for 60-char-windowed cases. `multi_item_parser._slot_looks_attached` line 179 uses a fixed 60-character lookback — fragile on long utterances.
- **Status:** PASS / FRAGILE.

### S9 — "Will be ready in 25 minutes" promised at order completion
- **Expected:** No false ETA.
- **Actual:** FAIL — `app/core/response_builder.py:466` hardcodes `"Your order has been placed successfully. Will be ready in 25 minutes. Thank you!"`. Same string also referenced in `live_call_service.py`.
- **Priority:** **P0** demo blocker.

### S10 — "Any cheese would you like, like American Cheese, Cheddar Cheese, or Mozzarella Cheese?"
- **Expected:** "Which cheese would you like — American Cheese, Cheddar, or Mozzarella?"
- **Actual:** verb embedded in NP; sounds robotic. Existing audit §2 — confirmed unfixed.
- **Files:** `app/responses/item/sides.py::ask_for_side` lines 53-63; `app/responses/item/modifiers.py::ask_for_modifier` lines 53-63.
- **Priority:** P0 (UX-critical).

### S11 — Empty / silent / "uh" / "hmm"
- **Expected:** Neutral re-prompt; no phantom item.
- **Actual:** PASS. `utterance_filter.is_filler_only()` (lines 199-237) drops filler-only utterances; handler re-prompts.
- **Status:** PASS.

### S12 — "Well Done" (a real menu item) said as a steak doneness preference
- **Expected:** Item match wins.
- **Actual:** Possible FAIL — `control_phrase_classifier` intercepts "done" as the DONE control intent. Menu has an item literally named "Well Done" (`menu.json:5231`). Order of resolution puts control phrase before menu lookup.
- **Status:** SUSPECTED FAIL.
- **Files:** `app/nlu/control_phrase_classifier.py:68-100`, `app/nlu/control_phrases.yaml:114`.
- **Priority:** P2 (depends on whether "Well Done" reachable from current FSM state — IDLE only).

---

## 4. Infinite Loop / Reprompt Audit

| State | Loop guard? | Max reprompts | Behavior at max | Risk |
|---|---|---|---|---|
| `WAITING_FOR_SIDE` | YES — `reprompt_count_by_field` | 3 → escalation prompt variant | escalation only; no hard ceiling, can loop after | LOW |
| `WAITING_FOR_MODIFIER` | YES (same) | 3 → variant | same | LOW |
| `WAITING_FOR_QUANTITY` | YES | 3 → `invalid_quantity_option` | same | LOW |
| `WAITING_FOR_SIZE` | YES | 3 | same | LOW |
| `WAITING_FOR_SIDE_SIZE` | YES | 3 | same | LOW |
| `WAITING_FOR_PICKUP_SMS_PERMISSION` | **NO** | unbounded | re-prompts on every UNKNOWN | **HIGH — confirmed in production** |
| `WAITING_FOR_PAYMENT` | **NO** | unbounded (cooldown only on resend) | indefinite wait + re-prompt | MED |
| `WAITING_FOR_ORDER_TYPE` | **NO** | unbounded | re-prompts identical line ("Is this for pickup or delivery?") on retry — feels like a loop | MED |
| `WAITING_FOR_DELIVERY_ADDRESS_COLLECTION` | YES — `ADDRESS_FIELD_MAX_REPROMPTS` (env, default 3) | 3 | escalation | LOW |
| `WAITING_FOR_CHECKOUT_COMPLETION` | NO | unbounded | re-prompts | MED |

**Confirmed loop:** Pickup SMS permission. Production NLU log rows 908-910 (2026-05-01) show 3 consecutive identical re-prompts despite `text=Yes. Send it.`

**Recommendation:** Add a generic `bump_state_reprompt(session, state)` helper used by every `waiting_for_*` handler. Default escalation at tier 3: (a) suppress the ambiguous prompt, (b) for SMS permission, fall through to implicit-affirm if `"yes"` token present, (c) for payment, offer agent handoff.

---

## 5. Turn-Taking / Barge-In Audit

| Question | Answer | Evidence |
|---|---|---|
| Per-session turn lock? | YES — `asyncio.Lock` per `ConversationSession` | `app/realtime/conversation_session.py:152` |
| Listen while speaking? | YES — STT continues; barge-in evaluated under lock | `voice_stream_server.py:1449` → `conversation_session.evaluate_barge_in_candidate` |
| Playback generation / cancel token? | YES — `playback_generation` counter incremented on interrupt/start; checked on every TTS buffer flush | `voice_stream_server.py:762, 879, 917, 1053, 1162` |
| Cancels TTS on barge-in? | YES — sends Twilio `clear`, calls `dg_tts_client.clear()`, sets phase=LISTENING, clears `active_mark_name` | `voice_stream_server.py:748-785` |
| Premature input gate? | YES — phase=PROCESSING → buffer in bounded deque (`maxlen=2`); phase=SPEAKING → barge-in policy | `conversation_session.py:256-272, 285` |
| Same utterance double-processed? | NO — `TurnCommitController.last_committed_text` dedup + `turn_id` stale-turn guard | `turn_commit_controller.py:81-90`; `conversation_session.py:265-273` |
| Endpointing timeout cancelable on barge-in? | YES — Deepgram endpointing (default 300 ms) + 700 ms commit debounce; debounce timer canceled on barge-in commit | `voice_stream_server.py:1308, 1603` |
| Forgets prior turn if user speaks early? | NO — buffered FIFO into `_pending_queue` (maxlen=2) | `conversation_session.py:256-272, 303-308` |

**Confirmed strengths:** Re-entrancy guard, generation-counter cancellation, dedup at commit time. The transport layer is the **best-engineered subsystem** in the codebase.

**Failure cases / missing guards:**
- `_pending_queue maxlen=2` — utterance #3 dropped silently if user speaks rapidly while handler is slow. Recommendation: log at `WARN` and consider `maxlen=4`.
- **No timeout on Twilio `mark` ACK** (`voice_stream_server.py:1703`) — if the mark event is dropped, `on_playback_completed()` never fires, the pending queue never drains, the call appears frozen. Recommendation: add a 2 s watchdog timer.
- TTS error path may leave `active_mark_name` and `bot_playback_started_at` set after exhausted retries (`voice_stream_server.py:930`). Subsequent barge-in policy checks read stale `bot_playback_started_at`. Low impact, easy fix.
- **Payment auto-check probe** scheduled at `conversation_session.py:113` runs **outside** the turn lock. Acquires the lock before processing, so eventual serialization is fine, but worth documenting that probe and user turn can interleave on FIFO acquisition.

---

## 6. Multi-Item / Modifier / Side Audit

| Capability | Status | Evidence |
|---|---|---|
| Multi-item supported | YES | `multi_item_parser.py:262-403`; `item_queue_service.py:96-169` |
| Quantity binds to correct item | PASS for normal phrasing, FRAGILE on long utterances | 60-char lookback `_slot_looks_attached`:179 |
| `with` parsing — side/modifier attached to parent item | **PARTIAL FAIL** | `_ITEM_SEPARATORS` (line 32-36) treats "with X" as a new segment if X is an ITEM-typed slot; no side-group scope check |
| Item + side + modifier in one utterance | PARTIAL | Works when slot model emits SIDE/MODIFIER labels; fails when sub-item is also a top-level ITEM (e.g., "Coke") |
| Multi-item queue drains without recursion / loop | PASS | `item_queue_service.py:96-169` synchronous popleft drain inside same turn |
| Queue overflow guard | YES — `MAX_QUEUE_DEPTH` enforced in `item_queue_service.py:63-82` (rejected from tail) | underlying `deque` is unbounded but populator caps it |
| Required group enforces min | PASS | `group_skip_policy.py:27-51` |
| Optional group allows skip | PASS | same |
| Too-many-selections | PASS | resolvers handle, but message phrasing ("That is too many extras. You can choose up to 3.") is dense (existing Response Review §51-52) |
| Already-selected groups not re-asked | PASS | next-step resolver iterates only over groups with no selection |
| Removed/default modifiers handled | PASS | `prefill_orchestrator` and modifier resolvers track |
| Acknowledge prior selection on next-group prompt | **FAIL** — `ask_for_side`/`ask_for_modifier` do not consume `matched_names` (existing audit §6) | sides.py / modifiers.py |
| Hard cap on items per utterance | NOT EXPLICIT — `parse_multi_item_utterance` has no length limit; relies on slot model | architectural risk MED |

**Recommendation:** treat the side-vs-modifier-vs-new-item discrimination as a **scoping problem**, not a parsing problem. Before splitting on "with", check whether the candidate token is a member of the parent item's side or modifier groups in the menu. If yes, attach; if no, split.

---

## 7. Checkout / Payment Audit

| Question | Answer / Evidence |
|---|---|
| Pickup checkout — state sequence | `CONFIRMING_ORDER → WAITING_FOR_PICKUP_SMS_PERMISSION → COMPLETED` (`confirm_order_handler.py:245-256, 412`) |
| If no phone | direct `COMPLETED` with `pickup_end_call` (lines 259-263) |
| SMS-permission classifier | regex + control intent + NLU intent (`pickup_sms_resolver.py:84-159`); robust on clean inputs |
| Reprompt counter on SMS permission | **MISSING** (`waiting_for_pickup_sms_permission_handler.py:90-91`) — confirmed loop in production |
| Decline path → ends call cleanly? | YES — `pickup_no_sms_end_call` + `CLEAR_CART` (`waiting_for_pickup_sms_permission_handler.py:83-87`) |
| Delivery checkout | `confirm_order_handler.py:369-410` — branches on phone/SMS configured → `WAITING_FOR_CHECKOUT_COMPLETION` (link via SMS) or `WAITING_FOR_DELIVERY_ADDRESS_COLLECTION` (voice) |
| Pickup vs delivery isolation | YES — single conditional, no shared state mutation | regression risk LOW |
| Payment-link cooldown | 60 s (`PAYMENT_LINK_RESEND_COOLDOWN_SECONDS` in `payment_flow_support.py:17`) — only blocks resend, not state |
| Payment timeout | **NONE** — `WAITING_FOR_PAYMENT` is indefinite |
| Repeat reminders for same order | YES, gated by 60 s cooldown only — no max-resends |
| Forbidden ETA phrase | **CONFIRMED P0** — `"Will be ready in 25 minutes"` hardcoded at `response_builder.py:466` |
| Checkout summary phrasing | **CONFIRMED bug** — `cart_responses.py:62-64` ends `"Should I place the order"` (no `?`); then `_confirm_order_summary` appends `"Would you like to checkout?"` |
| Delivery affirm/deny | intent-based (control intent + `resolve_confirmation_decision`); not literal yes/no |
| ZIP capture | local regex first (`waiting_for_delivery_eligibility_handler.py:226-275`); eligibility is local-only flag, no remote lookup. Correction path via `_handle_confirmation_zip_correction` (line 403-419) |
| Reconnect during payment | no explicit recovery — relies on session persistence + `verify_payment_for_order` re-poll on next turn |
| Post-completion guard | **MISSING** — handlers do not check `if state == COMPLETED: reject` |

---

## 8. NLU / Matching / Fallback Audit

> Most findings here **confirm or supplement** the existing `Compass_Voice_NLU_Audit_Report.md`.

| Issue | Status | Evidence |
|---|---|---|
| `utterance_filter.py` truncation | **REFUTED** — file is intact (255 lines). Earlier truncation was a workspace-mount artefact only. | direct read |
| `menu.json` parse error | REFUTED — parses cleanly | direct read |
| Affirm/deny is keyword-based, not ML | CONFIRMED — `confirmation_resolver._STRONG_AFFIRM_PHRASES` (27 phrases), `_DENY_PHRASES` (25), `_CANCEL_PHRASES` (9). Phrase sets duplicated between this file, `control_phrases.yaml`, and `linguistic_rules.py` with no sync mechanism | `confirmation_resolver.py:45-105` |
| 60-char lookback for quantity binding | NEW finding — magic constant, fragile on long utterances | `multi_item_parser.py:179` |
| Unbounded `pending_item_queue` (`deque` without `maxlen`) | PARTIAL — populator does cap via `MAX_QUEUE_DEPTH` in `item_queue_service.py:63-82`; raw queue itself is unbounded so future code paths could over-append | `conversation_context.py:73-75` + `item_queue_service.py:63-82` |
| Rejected candidate denylist | **MISSING** — only `item_not_found_attempts: Dict[str, int]` exists; tracks query strings, not rejected items. Same suggestion can repeat. | `conversation_context.py:96-99`; rotation at `ordering_decision_engine.py:245-248` is shuffle, not exclusion |
| Category fallback | EXISTS but undocumented; no confidence threshold; always returns top-4 | `ordering_decision_engine.py:227-248, 275-293` |
| Decimal qty leak | NEW — `quantity_parser.extract_weight_quantity` returns floats (0.5, 0.25); no central `quantity_formatter.py` enforcing int coercion before TTS | `quantity_parser.py:296-334` |
| Linguistic-rule negation gaps | NEW — `_LEADING_NEGATION_PHRASES = ("no ", "nope ", "nah ")` only; missing `"don't "`, `"not "`, `"don't want "` | `linguistic_rules.py:23-27` |
| Control phrases YAML / classifier drift | NEW — `_SKIP_EXACT` hardcoded in classifier, `skip` also in YAML; no validation | `control_phrase_classifier.py:68-100` vs `control_phrases.yaml:176-190` |
| Same-utterance dedup | MISSING at NLU layer — relies entirely on transport-layer `TurnCommitController.last_committed_text` | OK in practice |
| Menu ↔ control-phrase collision | NEW — menu item `"Well Done"` (`menu.json:5231`) collides with DONE control phrase. Reachable only from IDLE; low risk in current FSM | `control_phrases.yaml:114` |
| Max clarification attempts | CONFIRMED 3 (not 2 as existing audit phrased it) | `ordering_decision_engine.py:218` |
| Confidence thresholds | CONFIRMED hardcoded — `DEFAULT_CONFIRMATION_CONFIDENCE_THRESHOLD = 0.55`; `INTENT_MIN_CONF` from `nlu_config` | `confirmation_resolver.py:13-15`, `nlu_orchestrator.py:33` |

---

## 9. Test Gaps (suggested for Claude Code automation)

Existing harness (`tests/support/voice_test_harness.py`) is solid. Add the following:

1. **Pickup SMS permission loop test** — utter "yes. send it." 3× with NLU mock returning UNKNOWN; assert max 2 reprompts and implicit-affirm fallback on tier 3.
2. **Quantity-with-affirm priority** — already covered, but extend to cases where intent confidence ≥ 0.9 and slot still wins.
3. **"with" parsing edge case** — `"chicken taco with Coke"`; assert single item, side=Coke (or escalate as item if menu has no Coke as side).
4. **Hardcoded ETA absence** — string-search regression test that asserts `"Will be ready"` does NOT appear in any rendered response from any state.
5. **Confirmation summary punctuation** — render `_confirm_order_summary` and assert text matches `r".*[?]\s*$"`.
6. **Rejected-candidate not repeated** — feed user "no" to "Did you mean Cheeseburger?", then trigger same query; assert Cheeseburger is not the top suggestion.
7. **`Well Done` item resolution** — utter "I'd like a steak well done"; assert intent path resolves to item, not DONE control.
8. **Multi-item queue overflow** — synthesize 50-item utterance; assert queue truncated at `MAX_QUEUE_DEPTH` and tail items reported.
9. **Reprompt escalation on payment / order_type** — currently NO test exists; add tests that fail on indefinite loop.
10. **Mark-ACK watchdog** — simulate Twilio dropping the mark event; assert pending queue drains within 2 s.
11. **Float quantity leak** — assert `quantity_parser.extract_weight_quantity({"value": 0.5, "unit": "lb"})` does not propagate raw floats into any spoken response.
12. **Side+modifier prompt grammar** — assert no rendered side/modifier prompt contains the substring `" would you like, like "` (verb-in-NP smell).

---

## 10. Claude Code Fix Prompts

> One prompt per confirmed bug. Each is self-contained, ready to paste into Claude Code.

### CC-01 — Remove hardcoded ETA "Will be ready in 25 minutes"

**Objective:** Eliminate any string promising a fixed ready-time; substitute a neutral closer or surface the actual ETA from the order if available.

**Files to inspect:** `app/core/response_builder.py:466`, `app/services/live_call_service.py` (search for `"Will be ready"`), every caller of `_order_completed`.

**Required fix:** Replace `"Your order has been placed successfully. Will be ready in 25 minutes. Thank you!"` with `"Your order has been placed successfully. Thank you!"`. If the menu/order model exposes a `prep_time_minutes` or `eta` field, format it conditionally; otherwise omit. Do not hardcode any minute count.

**Tests:** Add a regression test that searches every `response_key` rendered from `WAITING_FOR_*` and `COMPLETED` states for the substring `"Will be ready"` and asserts zero matches.

**Definition of done:** Grep for `"Will be ready"` in `app/` returns zero hits; new test green.

---

### CC-02 — Fix checkout review summary punctuation and remove duplicate question

**Objective:** Render a single, well-punctuated checkout-confirmation prompt.

**Files to inspect:** `app/responses/cart_responses.py:31-68` (`render_checkout_review_summary`), `app/core/response_builder.py:437-448` (`_confirm_order_summary`).

**Required fix:** Drop the trailing `"Should I place the order"` from `render_checkout_review_summary` (or terminate it with `?`). Let `_confirm_order_summary` provide a single follow-up question. Final spoken form: `"…Your total is $14.50. Should I send the payment link?"`

**Tests:** Render `_confirm_order_summary` for sample carts; assert text ends with `?` and contains exactly one question mark.

**Definition of done:** Output for delivery and pickup carts each contains exactly one trailing `?`.

---

### CC-03 — Add reprompt counter + implicit-affirm fallback to pickup SMS permission

**Objective:** Eliminate the "send it / no, send it / send it" loop reproduced in production logs.

**Files to inspect:** `app/state_machine/handlers/payment/waiting_for_pickup_sms_permission_handler.py:47-91`, `app/core/handler_dispatcher.py:146-200` (current per-field reprompt counter pattern).

**Required fix:** Reuse the existing `reprompt_count_by_field` mechanism. Increment on every `PickupSmsDecision.UNKNOWN`. At tier ≥ 2: if `normalized_text` contains a strong-affirm token (`"yes"`, `"yeah"`, `"send"`, `"text"`), collapse to `SEND_SMS`; else if it contains a strong-deny token, collapse to `PAY_ON_PICKUP`; else escalate to a more explicit prompt (`"Just yes for a text link, or no to pay at pickup."`). Reset counter on success/cancel.

**Tests:** Three consecutive UNKNOWN turns with text `"yes. send it."` → asserts SMS is sent on or before turn 3.

**Definition of done:** Production replay of rows 908-910 (2026-05-01) results in SMS sent by turn 2.

---

### CC-04 — Switch unresolved-phrase filter from denylist to allowlist (NLU echo bug)

**Objective:** Stop echoing command verbs ("can you please add") as missing menu items.

**Files to inspect:** `app/state_machine/handlers/item/add_item/prefill_orchestrator.py:882-938` (`_collapse_unresolved_for_feedback`); same bug class in `side_group_resolver.py:218-248` and `modifier_group_resolver.py:240-266`.

**Required fix:** Replace the `ignored_tokens` set with menu-vocab allowlist: keep an unresolved phrase only if it has at least one token in the active item/group vocabulary OR matches a known menu entity via `menu_store.find_entity`. Plumb `menu_store` through the resolver. Existing audit §4 contains a ready code snippet.

**Tests:** Utterances `"can you please add beef tacos"`, `"can you tell the options"`, `"i want a burger"` → assert echoed unmatched list is empty for the first two and contains the failed token (if any) for the third.

**Definition of done:** Greps for `"I couldn't find can"` in test outputs return zero matches.

---

### CC-05 — Drop "Yes or no" / "Just say yes or no" tails from confirmation prompts

**Objective:** Voice-natural binary questions; no tail.

**Files to inspect:** `app/core/response_builder.py:228-234, 435`; `app/responses/flow_control_responses.py:23, 25`.

**Required fix:** Strip every literal `"Yes or no"` / `"Just say yes or no"` from confirmation lambdas. Existing audit §1 has the exact replacements. Centralize via a `app/responses/confirmation.py::yes_no_question(stem)` helper.

**Tests:** Render every `confirm_*` `response_key`; assert no rendered text matches `r"yes\s+or\s+no"` or `r"just\s+say\s+yes\s+or\s+no"`.

**Definition of done:** Test green; no occurrences of those substrings in rendered responses.

---

### CC-06 — Fix side/modifier prompt grammar and noun selection

**Objective:** Replace `"Any cheese would you like, like American Cheese …"` with `"Which cheese would you like — American Cheese, Cheddar, or Mozzarella?"` for required groups; `"Want any cheese? American Cheese, Cheddar, or Mozzarella, or none."` for optional. Reject `prompt_noun` that equals the parent item word (`"Any burger would you like, like Mayo?"`).

**Files to inspect:** `app/responses/item/sides.py::ask_for_side` (lines 53-63), `app/responses/item/modifiers.py::ask_for_modifier` (lines 53-63).

**Required fix:** As specified in existing audit §2 (full snippet provided). Required vs optional templates split; verb out of NP; noun sanitization.

**Tests:** Render `ask_for_side` and `ask_for_modifier` against menu groups with `min_selector=0` and `min_selector>=1`; assert no occurrence of `" would you like, like "`. Required prompt starts with `"Which "`. Optional with `"Want any "`.

**Definition of done:** Tests green; manual demo plays back as natural prompts.

---

### CC-07 — Add rejected-candidate denylist on ConversationContext

**Objective:** Never re-suggest an item the user just rejected.

**Files to inspect:** `app/state_machine/models/conversation_context.py:96-99` (current `item_not_found_attempts`), `app/core/ordering_decision_engine.py:218-248`.

**Required fix:** Add `rejected_item_ids: set[str] = field(default_factory=set)` to context. On user DENY of a "Did you mean X?" prompt, add X's `item_id` to the set. Filter the set out of suggestion candidates in `_find_category_suggestions` and the rotation logic.

**Tests:** User rejects "Cheeseburger" suggestion, then re-utters an ambiguous query → assert "Cheeseburger" no longer appears in the next "Did you mean" list.

**Definition of done:** Test green; no repeated suggestions across rejections.

---

### CC-08 — Add reprompt escalation to 4 unguarded states

**Objective:** Ensure every `waiting_for_*` state has a max-attempts ceiling.

**Files to inspect:** `app/state_machine/handlers/payment/waiting_for_payment_handler.py`, `app/state_machine/handlers/order/waiting_for_order_type_handler.py`, `app/state_machine/handlers/payment/waiting_for_checkout_completion_handler.py`, `app/state_machine/handlers/payment/waiting_for_pickup_sms_permission_handler.py`, `app/core/handler_dispatcher.py:146-200`.

**Required fix:** Generalize the per-field reprompt counter into a per-state escalation policy applied in `handler_dispatcher`. At tier 3, route to a state-specific escalation: agent handoff for payment, "I'll connect you" for order_type repeat, etc.

**Tests:** Per state, simulate 4 UNKNOWN-classification turns; assert tier-3 escalation triggered.

**Definition of done:** No state is reachable that re-prompts the same response_key 4 times in a row.

---

### CC-09 — Skip secondary confirmation when re-mention has disambiguating slot

**Objective:** Avoid wasted "X, right?" turn after the user re-mentions an ambiguous candidate with at least one disambiguating slot.

**Files to inspect:** `app/state_machine/handlers/item/confirming_handler.py:170-200` (`_resolve_candidate_item_from_confirmation`).

**Required fix:** When the user's re-mention contains a slot that uniquely identifies one candidate from the offered shortlist, accept directly and proceed to the next add-flow step (quantity / sides) without re-confirming.

**Tests:** User says "Korean tacos" → bot offers shortlist → user says "Korean taco spicy chicken" → assert next response_key is `ask_for_quantity` (or next), not `confirm_item`.

**Definition of done:** Test green; no second `confirm_item` after disambiguating slot.

---

### CC-10 — Quantity binding lookback constant + bounded queue invariant

**Objective:** Replace magic 60-char lookback with a documented constant; assert queue cap.

**Files to inspect:** `app/nlu/multi_item_parser.py:162-220, 179`; `app/state_machine/models/conversation_context.py:73-75`; `app/core/item_queue_service.py:63-82`.

**Required fix:** Extract `_QUANTITY_LOOKBACK_CHARS = 60` with a docstring explaining the heuristic. Cap `pending_item_queue` at `MAX_QUEUE_DEPTH` at the dataclass level (use `deque(maxlen=...)`); make `MAX_QUEUE_DEPTH` configurable.

**Tests:** 50-item utterance; assert queue length ≤ MAX_QUEUE_DEPTH and tail items reported as "I added X. The rest didn't fit, want to add the others?".

**Definition of done:** Test green; magic constants documented.

---

### CC-11 — Mark-ACK watchdog in voice_stream_server

**Objective:** Recover gracefully if Twilio drops the playback mark event.

**Files to inspect:** `app/api/voice_stream_server.py:1665-1703`, `app/realtime/conversation_session.py:303-308`.

**Required fix:** When sending a mark, schedule a 2 s watchdog timer. If the mark ACK arrives, cancel the timer. If it expires, log `[mark_ack_watchdog_fired]` and force-call `on_playback_completed()` to drain the pending queue.

**Tests:** Patch the WS so `mark` events are never echoed back; assert pending queue drains within ≤ 2.5 s.

**Definition of done:** Test green; no indefinite hangs on dropped mark.

---

### CC-12 — Centralize affirm/deny phrase sets + sync with control_phrases.yaml

**Objective:** Remove drift across `confirmation_resolver.py`, `linguistic_rules.py`, `control_phrase_classifier.py`, and `control_phrases.yaml`.

**Files to inspect:** all four.

**Required fix:** Single source of truth — load all categories from `control_phrases.yaml` at import time into `control_phrase_lexicon.py`. Existing in-file frozensets become thin facades returning lexicon-sourced sets. Add a CI assertion that these sets cover all phrases tested in the unit suite.

**Tests:** Add a test that imports each module and asserts `frozenset(_STRONG_AFFIRM_PHRASES) <= lexicon.affirm_phrases()`.

**Definition of done:** Phrase drift impossible without updating YAML.

---

## Notes

- The audit mounted from `D:\Working\Cygnus\compass-voice` (Windows) / `/sessions/.../mnt/compass-voice/` (Linux sandbox). Do not consider the file-truncation observation in the prior NLU audit as still active — verified intact at audit time.
- Live test execution was blocked because `pip install pytest` failed with HTTP 403 (audit sandbox firewall). The existing test harness (`tests/support/voice_test_harness.py`) is robust; recommend running `python -m pytest tests\nlu tests\flow tests\payment tests\edge_cases tests\regression -q` on the dev machine and capturing the baseline pass/fail rate alongside this report.
- The two prior reports (`Compass_Voice_NLU_Audit_Report.md`, `Compass_Voice_Response_Review.md`) remain authoritative for response-layer wording fixes; this audit cross-validated their findings against current code state and added FSM, transport, and payment-flow coverage they did not include.
