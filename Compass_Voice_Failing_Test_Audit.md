# Compass Voice — Failing-Test Audit (2026-05-07)

> Senior backend QA + FSM review. No code modified. Source-and-cache static analysis only.

---

## ⚠️ Methodology disclosure

The sandbox running this audit could not install `pytest` (PyPI / apt blocked). I therefore did **not** re-execute the suite live. Instead, I anchored the inventory on the canonical failure ledger pytest itself maintains:

```
.pytest_cache/v/cache/lastfailed   (75 live test entries + 6 stale app/tests/* entries)
.pytest_cache/v/cache/nodeids      (collected node ids)
```

Every node id below was then verified against:

- the **current** test source under `tests/` (renamed/removed test ids → flagged STALE),
- the **current** implementation under `app/`,
- the **demo menu fixture** at `app/data/restaurants/demo/menu.json` vs. its backups (`menu.20260506_153616.bak`, `menu.20260506_154748.bak`).

Where I could not deterministically prove the failure mode without executing, I marked the entry **PROBABLE** and stated the evidence relied on. Failures called "real bugs" below are backed by either a reproduced contract violation in the source, or a fixture/menu mismatch that is mechanically certain to fail.

---

## A. Executive summary

| Metric | Value |
|---|---|
| Test files in suite (estimate from collected nodeids) | ~150 |
| `.pytest_cache.lastfailed` entries (raw) | 81 |
| Stale `app/tests/*`, `app/api/*` entries (path no longer used) | 6 |
| Live `tests/*` lastfailed entries | **75** |
| Of those, **stale node ids** (test renamed/removed in current source) | **13** |
| Of those, **file-level collection failure** | 1 (`tests/api/test_chat_demo.py`) |
| **Effective failing tests** that match a current node id | **61** |
| Failure groups (root-cause clusters) | **9** |
| Failures driven by recent demo-menu data changes (45 items removed) | ~33 |
| Failures driven by real production code/contract regressions | ~8 |
| Failures driven by stale test fixtures / payload-shape drift | ~13 |
| Failures driven by intentional behavior changes the tests have not absorbed | ~4 |
| Environment / collection failures | 1 (file-level) + 13 stale ids |

### Highest-risk broken area

**Real production bugs (P0)** are concentrated in two areas:

1. **TTS failure → call termination phase invariant**: `ConversationSession` resets phase to `LISTENING` immediately after a `TTSFailureError`, then ends the call. The session is briefly in a state that says "we are listening" while no audio reached the caller. Tests catch this (`TestPhaseInvariant`, `TestTTSFailureNoAction`). Production impact: in a TTS outage, the FSM thinks the user can speak, but the channel is being torn down — race conditions with deferred barge-in / pending interrupt handling.
2. **`_confirm_item` UX still uses "Yes or no"** — directly violates the project's stated rule (`Do NOT rely on exact keywords like yes/no`). Test enforces the natural-confirmation prompt; code regressed.

The single largest blast radius is **menu-fixture drift** (group D below): the production demo menu was edited on 2026-05-06 and ~45 items were removed (Iced Mocha, 49. Seafood Combo Platter, 61. 50 Wings, etc.) and several remaining items lost required side/modifier groups (Chicken Taco lost both modifier groups; Chicken Burger lost its `Choose Meat` side group and `Bun Modification` modifier group). Tests that script multi-turn flows against these specific items now fail at the first divergent prompt.

### Estimated real bugs vs. test issues

| Class | Count |
|---|---|
| 1 — Real production bug | **~8** |
| 2 — Stale assertion after intended architecture change | ~4 |
| 3 — Test fixture / mock incomplete | ~7 |
| 4 — Menu fixture data drift | **~33** |
| 5 — NLU/model nondeterminism | 0 (all current tests stub NLU) |
| 6 — Environment / dependency issue | 1 file-level + (suite cannot run without `fastapi`, `pytest-asyncio`, etc.) |
| 7 — Intentional behavior change needing test update | ~4 |
| 8 — Stale `lastfailed` ids (test was renamed/removed) | **13 — not real failures** |

---

## B. Failure inventory (grouped, ordered by priority)

| Group | P | Count | Class | Tests / pattern | Production impact |
|---|---|---|---|---|---|
| **A** TTS failure phase invariant | **P0** | 3 | 1 | `test_phase_not_listening_immediately_after_tts_failure`, `test_phase_was_processing_before_speak`, `test_phase_is_not_listening_after_failure` | Real bug — call-control + phase transitions during TTS outage |
| **B** Confirm-item UX violates "no yes/no keywords" rule | **P0** | 1 | 1 | `test_response_builder_confirm_item_uses_natural_confirmation_prompt` | Real bug — UX/architecture violation |
| **C** Cart summary duplicate-side label format | **P1** | 3 | 1 / 7 | `TestDuplicateSideDisplayLabels::test_*` | Real bug if `Coke x2` is the canonical UX (project recently merged duplicate-side support); else stale assertion |
| **D** Demo menu data drift (45 items removed; Chicken Taco / Chicken Burger / Burger groups changed) | **P1** | ~33 | 4 | `test_real_menu_pickup_add_item_flows[*]`, `test_real_menu_delivery_add_item_flows[*]`, `test_real_menu_pickup_add_item_prefills_sparse_side_and_modifiers`, `test_numbered_platter_phrase_matches_numbered_menu_item`, `test_parse_multi_item_utterance_uses_menu_truth_for_boundary_items`, `test_multi_modifier_slot_capture_normalizes_into_menu_modifier_names`, `test_waiting_state_side_normalization_maps_beef_to_beef_meat`, `test_scenario_11_checkout_during_pending_modifier_must_not_short_circuit`, `TestMultiGroupPrefillEngine::*`, `TestAddItemHandlerMultiItemPrefill::*`, `test_required_multi_slot_*`, `test_partial_input_only_asks_for_missing_field`, `test_reprompt_guardrail_lists_options_after_third_invalid_side_attempt`, `test_quantity_reprompt_guardrail_changes_guidance_after_third_invalid_attempt`, `test_turn_logger_records_structured_fields_for_add_item_prompt`, `test_add_item_handler_uses_real_menu_truth_for_multi_item_boundaries`, `test_add_item_handler_prefills_*` (3) | Test-only (production menu is the source of truth); but until reconciled, the suite cannot defend the ordering pipeline |
| **E** GroupResolutionHandler test fixture missing `modifier_groups` / `selected_modifier_groups` (regression after `_spoken_modifiers_for` was added) | **P1** | 7 | 3 | `ReadyToFinalizeTests::test_*` (basic, matched, unmatched, debug, no_matched_key, qty_zero, qty_two) | Test fixture issue. Production code is correct. |
| **F** `verify_payment_for_order` payload now carries `_payment_events` (test asserts strict equality) | **P2** | 1 | 7 | `PaymentFlowSupportTests::test_failed_payment_returns_draft_retry_response` | Stale assertion after observability addition |
| **G** `VoiceSessionSynchronizer` calls `context.reset_session_scope()` but `_FakeContext` only mocks `reset()` | **P2** | 3 | 3 | `TestVoiceSessionSynchronizer::test_payment_completed_sets_delivery_status`, `test_mark_completed_resets_context_and_sets_state`, `CheckoutServicePaymentSyncTests::test_handle_payment_completed_syncs_the_voice_session` (real Session may also lose `delivery.order_number` post-reset) | Likely stale fake; one may be a real ordering bug |
| **H** `WaitingForSideHandler._choice_payload` now keeps already-selected items (duplicate-side support); test still expects exclusion | **P2** | 1 | 7 | `test_side_choice_payload_excludes_already_selected_options` | Intentional behavior change |
| **I** Stale `lastfailed` cache entries (test renamed/removed) | **P3** | 13 | 8 | `test_float_2_5_rounds_to_3`, `test_module_does_not_import_normalize_food_quantity[app.cart.cart_summary_builder]`, `test_each_failure_ends_call`, `test_success_after_prior_turn_works_normally`, `test_payment_auto_check_dispatches_auto_text_then_cancels`, `test_response_builder_ask_for_modifier_supports_multi_select_guidance`, `test_response_builder_ask_for_modifier_uses_open_ended_prompt`, `test_always_asks_for_quantity_even_when_present_in_first_request`, `test_single_meaningful_token_is_echable`, `test_single_meaningful_token_with_stop_words_is_echable`, `test_control_intent_resolver_has_no_inline_frozenset_literals`, `test_control_intent_resolver_has_no_inline_phrase_frozensets`, `test_resolve_item_normalized_below_threshold_returns_none`, `test_normalize_quantity_both_of_them`, `test_no_random_module_imported_in_generator`, `test_strips_filler_prefixes[i would like to order two tacos-two tacos]`, `test_invalid_raw_returns_none[-1]` | Not real failures — first green run will drop them |
| **J** `tests/api/test_chat_demo.py` collection failure | **P3** | 1 | 6 | file-level | Likely missing `fastapi` in current sandbox or import side-effect |

Group counts add to ~62; the remaining 13 are Group I (stale ids) — the 75 live `lastfailed` entries minus 13 stale id entries = 62 effective failures. Some entries fall into more than one bucket (e.g. real-menu tests in Group D also depend on fixture order assumptions).

---

## C. Deep dive per group

### Group A — TTS failure phase invariant (**P0 / real bug**)

**Failing tests**

```
tests/realtime/test_tts_failure_handling.py::TestTTSFailureNoAction::test_phase_is_not_listening_after_failure
tests/realtime/test_tts_failure_handling.py::TestPhaseInvariant::test_phase_not_listening_immediately_after_tts_failure
tests/realtime/test_tts_failure_handling.py::TestPhaseInvariant::test_phase_was_processing_before_speak
```

**Source under test**

`app/realtime/conversation_session.py`, the TTS-failure `except` arm of `process_committed_turn`:

```python
# line 466–491
try:
    await self.transport.speak_response(
        spoken_response_text, trace=trace,
        end_call_after_playback=self.should_end_call_after_playback,
    )
except TTSFailureError as exc:
    print("[TTS_FAILURE_HANDLED]", {...})
    # Reset phase so the session is not stuck in PROCESSING after
    # the failure path completes.
    self.set_phase_listening()              # ← BUG
    handled = await self.handle_playback_failure(
        end_call_after_playback=self.should_end_call_after_playback,
    )
    if not handled:
        await self.transport.end_call()
    return
```

**Root cause**

1. After `TTSFailureError`, `set_phase_listening()` runs *before* the call is ended. The contract the tests defend is: *if no audio was delivered, do not return to LISTENING* — a LISTENING phase signals to the rest of the system that the user can speak now. Setting LISTENING right before `end_call()` creates a tiny window where any deferred barge-in / pending-interrupt handler could try to re-enter `process_turn`.
2. `set_phase_speaking()` is set unconditionally on line 455 *before* delegating to `transport.speak_response`. The test `test_phase_was_processing_before_speak` captures the phase exactly as `transport.speak_response` is entered and asserts `RealtimePhase.PROCESSING in phases_seen` — the contract is that the **transport** owns the SPEAKING transition (so a TTS that fails before producing audio never falsely reports SPEAKING). The current code violates that ownership by promoting the phase from PROCESSING → SPEAKING outside the transport.

**Why this fails now**: both transitions are a recent addition (the docstring on line 451–454 explicitly justifies setting SPEAKING for "test environments where the transport stub is passive" — i.e. the change was made to make a different test pass). It moved the SPEAKING transition out of the transport.

**Fix**

- Remove the `self.set_phase_speaking()` on line 455. Let `transport.speak_response` (or `voice_stream_server`) own the transition. For the "passive test stub" concern, make the test stubs explicitly mark SPEAKING when entered, OR rely on a "minimal_speak_response" base class.
- Replace `self.set_phase_listening()` in the except arm with either (a) leaving the phase unchanged and relying on the call-control side effect to terminate, or (b) introducing a new `RealtimePhase.TERMINATING` value and setting that. Option (a) is simpler and lower-risk.

**Risk / blast radius**

Phase invariants ripple through `is_speaking()`, `is_listening()`, barge-in policy, and pending queue drain. Removing premature SPEAKING is very safe (other paths set SPEAKING in the transport). Removing premature LISTENING in the failure path is safe so long as `end_call()` / `transfer_call()` always follow.

**Test plan**

- Re-run `tests/realtime/test_tts_failure_handling.py` (entire file).
- Re-run `tests/realtime/test_conversation_session.py` to confirm SPEAKING transitions still work via the transport.
- Add a guard test: after `end_call_after_playback=True` + TTSFailure, phase ∈ `{PROCESSING, SPEAKING, TERMINATING}` — never LISTENING.

---

### Group B — `confirm_item` UX violates project rule (**P0 / real bug**)

**Failing test**

```
tests/core/test_response_builder_add_item.py::test_response_builder_confirm_item_uses_natural_confirmation_prompt
```

**Source under test**

`app/core/response_builder.py:429`:

```python
def _confirm_item(self, context, menu_repo, payload):
    item_name = payload.get("item_name") or context.current_item_name
    ...
    return f"{item_name}, right? Yes or no."
```

**Root cause**

The project instructions explicitly prohibit yes/no keyword prompts. The test expects:

```python
assert "is that right" in text.lower()
assert "yes or no" not in text.lower()
```

Code returns `"Zinger Burger, right? Yes or no."`. Mismatched on both clauses.

**Fix**

Change to a natural confirmation prompt — e.g.

```python
return f"{item_name}, is that right?"
```

**Risk**

None — string-only change, downstream NLU already handles affirm/deny via `ControlPhraseLexicon`.

**Test plan**

`tests/core/test_response_builder_add_item.py` (full file). Also re-run `tests/core/test_turn_engine_real_menu_add_item_flows.py` to make sure no flow asserts the literal "yes or no" string.

---

### Group C — Cart summary duplicate-side label format (**P1**)

**Failing tests**

```
tests/cart/test_cart_summary_builder_duplicate_sides.py::TestDuplicateSideDisplayLabels::test_two_same_sides_shows_x2
tests/cart/test_cart_summary_builder_duplicate_sides.py::TestDuplicateSideDisplayLabels::test_three_same_sides_shows_x3
tests/cart/test_cart_summary_builder_duplicate_sides.py::TestDuplicateSideDisplayLabels::test_mixed_sides_correct_labels
```

**Source under test**

`app/cart/read_models/cart_summary_builder.py:140-141`:

```python
if count > 1:
    label = f"{count} {label}"      # produces "2 Coke"
```

**Root cause**

Test asserts `["Coke x2"]` / `["Coke x3"]`. Code produces `["2 Coke"]`. Pricing tests pass (counted correctly) — only the **display format** differs.

**Decision required**

Which label format is canonical?

- `"Coke x2"` — proposed by tests; matches POS-style output.
- `"2 Coke"` — current; matches plain-English speech.

For a **voice** ordering system, "two cokes" is the spoken form; for the cart summary string used in TTS, you probably want plain English. For checkout-screen rendering you may want `Coke x2`.

**Recommended fix**

Adopt the test's `"Coke xN"` rendering for cart-summary display path (it's the structured cart line that gets read out via `render_cart_summary` — speech rendering already inflects "two" vs "2" elsewhere). One-line change at `cart_summary_builder.py:140-141`:

```python
if count > 1:
    label = f"{label} x{count}"
```

**Risk**

`render_cart_summary` and `render_checkout_review_summary` may concatenate with the side label and produce TTS text containing "x2" — which TTS will pronounce as "x two". Mitigation: keep `"{count} {label}"` for the speech path and use `"{label} x{count}"` only for the structured `sides` list in the summary dict. Two render paths, two formats.

**Test plan**

`tests/cart/test_cart_summary_builder_duplicate_sides.py` full file + `tests/core/test_quantity_formatter.py::TestRenderCartSummaryQuantity` to confirm voice rendering unchanged.

---

### Group D — Demo menu data drift (**P1 — biggest blast radius**)

**Affected tests (representative subset; ~33 tests in total)**

```
tests/menu/test_numbered_item_matching.py::NumberedItemMatchingTests::test_numbered_platter_phrase_matches_numbered_menu_item
tests/nlu/test_multi_item_parser.py::test_parse_multi_item_utterance_uses_menu_truth_for_boundary_items
tests/nlu/test_ordering_nlu_behavior.py::test_multi_modifier_slot_capture_normalizes_into_menu_modifier_names
tests/nlu/test_ordering_nlu_behavior.py::test_waiting_state_side_normalization_maps_beef_to_beef_meat
tests/regression/test_required_ordering_scenarios.py::test_scenario_11_checkout_during_pending_modifier_must_not_short_circuit
tests/state_machine/handlers/item/add_item/test_multi_item_prefill.py::TestMultiGroupPrefillEngine::test_coke_attaches_to_can_drinks_even_when_slot_is_item
tests/state_machine/handlers/item/add_item/test_multi_item_prefill.py::TestMultiGroupPrefillEngine::test_chicken_burger_segment_prefills_required_groups
tests/state_machine/handlers/item/add_item/test_multi_item_prefill.py::TestAddItemHandlerMultiItemPrefill::test_chicken_taco_with_coke_steak_chicken_does_not_reask_can_drinks
tests/state_machine/handlers/item/add_item/test_multi_item_prefill.py::TestAddItemHandlerMultiItemPrefill::test_chicken_taco_with_jelly_then_bare_a_chicken_burger_splits
tests/state_machine/handlers/item/add_item/test_prefill_feedback.py::test_add_item_handler_prefills_*    (7)
tests/core/test_turn_engine_phase2_validation.py::test_required_multi_slot_*    (2)
tests/core/test_turn_engine_phase2_validation.py::test_partial_input_only_asks_for_missing_field
tests/core/test_turn_engine_phase2_validation.py::test_reprompt_guardrail_lists_options_after_third_invalid_side_attempt
tests/core/test_turn_engine_phase2_validation.py::test_quantity_reprompt_guardrail_changes_guidance_after_third_invalid_attempt
tests/core/test_turn_engine_phase2_validation.py::test_turn_logger_records_structured_fields_for_add_item_prompt
tests/core/test_turn_engine_real_menu_add_item_flows.py::test_real_menu_pickup_add_item_flows[*]    (4)
tests/core/test_turn_engine_real_menu_add_item_flows.py::test_real_menu_delivery_add_item_flows[*]    (2)
tests/core/test_turn_engine_real_menu_add_item_flows.py::test_real_menu_pickup_add_item_prefills_sparse_side_and_modifiers
```

**Root cause**

The demo menu was edited on 2026-05-06 (commit `cc7fd6f` "fix(voice-ordering): normalize food quantities…"). Compared to the immediately-preceding backup `menu.20260506_153616.bak`:

| Item | Old menu | Current menu |
|---|---|---|
| Bourbon Chicken | mods: `Bourbon Chicken Toppings` | unchanged |
| Chicken Taco | sides: `Can Drinks`, mods: `Additional Meat for Plates`, `Additional Extras For Biscuits` | sides: `Can Drinks`, **mods: []** |
| Iced Mocha | mods: `Coffee Condiments` | **REMOVED** |
| 49. Seafood Combo Platter | sides: `Platter Sides` | **REMOVED** |
| Crabcake Combo | sides: `Choose Drink`, mods: `Sandwich Condiments` | sides: `Choose Drink`, mods: `Sandwich Condiments - Select to Add or Remove` *(renamed)* |
| 61. 50 Wings | sides: `Wing Flavors`, `Baked or Fried Wing Choice` | **REMOVED** |
| Chicken Burger | sides: `Choose Cheese`, `Choose Meat`, `Choose Bun`, mods: `Burger Modification`, `Bun Modification` | sides: `Choose Cheese`, `Choose Bun`, mods: `Burger Modification` |
| Double Bacon Burger | present | **REMOVED** |

In total **45 items were removed** and several remaining items lost groups or had groups renamed. None of the dependent tests were updated.

**Why this fails now**: the failing tests script multi-turn flows that drive specific FSM transitions (`ask_for_side` → `ask_for_modifier` → …). The first turn whose expected response_key depends on a missing group/item silently emits `item_not_found` or skips a state, and every subsequent assertion cascades.

**Two fix paths**

**Path 1 — Restore the menu fixture used by tests (recommended).**
The test suite needs a *frozen* menu fixture, decoupled from the production demo menu. Candidates:

- Move the test menu to `tests/fixtures/menu/demo.json` and load it via the existing `build_menu_repo()` helper, parametrised by `menu_path`.
- Or copy `menu.20260506_153616.bak` → `tests/fixtures/menu/demo_for_phase2.json` and have these tests opt in.

This decouples test contracts from any future menu edit.

**Path 2 — Update each scripted flow to match the current menu.**
Cheaper short-term, but every future menu edit re-breaks these tests. NOT recommended unless ownership of the demo menu transfers to the test author.

**Risk**

Menu data is not "the system under test" — the FSM/handler logic is. So pinning a fixture is correct. Only `tests/core/test_turn_engine_real_menu_add_item_flows.py` may legitimately need to test against the live demo menu (because that's its purpose). For that file: either restore the items it scripts (Iced Mocha, 49. Seafood Combo Platter, 61. 50 Wings, Crabcake Combo with old `Sandwich Condiments` name) or rewrite to use items that still exist with currently-correct group names.

**Test plan**

After restoring/pinning the menu, re-run all of the listed tests in one batch.

---

### Group E — `GroupResolutionHandler` test fixture missing modifier_groups (**P1 / fixture defect**)

**Failing tests**

```
tests/state_machine/handlers/item/add_item/test_group_resolution_handler.py::ReadyToFinalizeTests::test_basic_finalization
                                                                       ::test_matched_names_injected
                                                                       ::test_unmatched_names_injected
                                                                       ::test_match_debug_merged
                                                                       ::test_no_matched_names_key_when_empty
                                                                       ::test_quantity_zero_defaults_to_one
                                                                       ::test_quantity_two_preserved
```

**Source under test**

`app/state_machine/handlers/item/add_item/group_resolution_handler.py:22-44`:

```python
def _spoken_modifiers_for(context):
    pending = context.pending_add_item
    if pending is None:
        return []
    spoken = []
    for group in pending.modifier_groups:           # ← AttributeError here
        for selection in context.selected_modifier_groups.get(group.group_id, []):
            ...
```

This helper was added so `item_added_successfully` payloads carry `spoken_modifiers` (e.g. "...with no onions and extra cheese"). It's called inside `_step_to_result` for the `ReadyToFinalize` branch.

**Test fixture** (`test_group_resolution_handler.py:36-43`):

```python
def _make_context(*, has_pending=True):
    pending = None
    if has_pending:
        pending = types.SimpleNamespace(item_name="Zinger Burger")   # ← no modifier_groups
    return types.SimpleNamespace(
        pending_add_item=pending,
        quantity=1,
    )
```

The fixture stub `pending_add_item` does not provide `modifier_groups`, and the context stub does not provide `selected_modifier_groups`. So the helper raises `AttributeError` for every `ReadyToFinalize` test.

**Fix (test-side)**

```python
def _make_context(*, has_pending=True):
    pending = None
    if has_pending:
        pending = types.SimpleNamespace(
            item_name="Zinger Burger",
            modifier_groups=[],
        )
    return types.SimpleNamespace(
        pending_add_item=pending,
        quantity=1,
        selected_modifier_groups={},
    )
```

**Risk**

None — pure test fixture fix.

**Test plan**

`tests/state_machine/handlers/item/add_item/test_group_resolution_handler.py` full file. Then re-run `tests/state_machine/handlers/item/add_item/test_extracted_components.py` for any additional test that mocks the same context shape.

---

### Group F — `verify_payment_for_order` payload shape changed (**P2 / stale assertion**)

**Failing test**

```
tests/state_machine/handlers/payment/test_payment_flow_support.py::PaymentFlowSupportTests::test_failed_payment_returns_draft_retry_response
```

**Source under test**

`app/state_machine/handlers/payment/payment_flow_support.py:266-275`:

```python
if status_lower in PAYMENT_FAILURE_STATUSES:
    return HandlerResult(
        next_state=ConversationState.CONFIRMING_ORDER,
        response_key="payment_draft_saved_retry_later",
        response_payload=append_payment_event(
            {"order_number": order_number},
            event_name="payment_failed",
            metadata={"status": status_lower},
        ),
    )
```

`append_payment_event` adds a `_payment_events` list:

```python
{"order_number": "1234567",
 "_payment_events": [{"event_name": "payment_failed", "metadata": {"status": "failed"}}]}
```

Test expects strict equality with `{"order_number": "1234567"}`. Test was written before the observability event channel was added.

**Fix (test-side)**

Change the assertion to use subset match:

```python
self.assertEqual(result.response_payload["order_number"], "1234567")
self.assertEqual(
    result.response_payload["_payment_events"][-1]["event_name"],
    "payment_failed",
)
```

**Risk**

None.

**Test plan**

`tests/state_machine/handlers/payment/test_payment_flow_support.py` full file.

---

### Group G — `_FakeContext` stale after `reset()` → `reset_session_scope()` rename (**P2 / fixture + possible real bug**)

**Failing tests**

```
tests/services/test_checkout_extracted_components.py::TestVoiceSessionSynchronizer::test_payment_completed_sets_delivery_status
                                                  ::test_mark_completed_resets_context_and_sets_state
tests/services/test_checkout_service_payment_sync.py::CheckoutServicePaymentSyncTests::test_handle_payment_completed_syncs_the_voice_session
```

**Source under test**

`app/services/voice_session_synchronizer.py:142-150`:

```python
if mark_completed or checkout_session.payment_completed:
    context.reset_session_scope()
    voice_session.cart.clear()
    voice_session.conversation_state = ConversationState.COMPLETED
    voice_session.last_response_key = "order_completed"
    voice_session.last_response_payload = {
        "order_number": checkout_session.order_number,
        "payment_reference": checkout_session.payment_reference,
    }
```

**Test fake** (`test_checkout_extracted_components.py:287-294`):

```python
class _FakeContext:
    def __init__(self):
        self.delivery_address = _FakeDelivery()
        self.delivery_address_confirmed = False
        self._reset_called = False
    def reset(self):                  # ← stale name
        self._reset_called = True
```

**Root cause**

- The fake provides `reset()` but the production code calls `reset_session_scope()`. Result: `AttributeError` when the failure path is taken. This is fixture rot from a method rename.
- `test_payment_completed_sets_delivery_status` *also* fails for a second reason: it sets `delivery.payment_status` BEFORE calling `reset_session_scope()`. If the real `reset_session_scope()` (or this fake's substitute) wipes the delivery_address fields, the assertion `delivery.payment_status == "payment_confirmed"` may also fail on the real Session. **Verify whether `reset_session_scope` zeroes `delivery_address.payment_status` / `payment_reference`** — if it does, the production code orders side effects incorrectly: it should set the final delivery state AFTER reset, not before. That is a real bug, separate from the fixture rot.

**Fix**

1. Update `_FakeContext.reset()` → `reset_session_scope()` (additionally, keep `reset()` if any other production caller still uses it).
2. Inspect `reset_session_scope` (in `ConversationContext`); if it nulls payment_status/payment_reference/order_number, refactor the synchronizer so payment finalization fields are written *after* reset (or `reset_session_scope` should preserve `delivery_address.order_number` and `payment_reference` — they belong to the just-completed order, not the next conversation scope).
3. For `test_handle_payment_completed_syncs_the_voice_session`, the real `Session` is used; if it loses `delivery.order_number` post-reset, the assertion `saved.conversation_context.delivery_address.order_number == "1234567"` fails. Same fix as (2).

**Risk**

If reset_session_scope clears too much, this is a P0 production bug — payment confirmation events would lose their reference. Worth re-prioritising once verified.

**Test plan**

Inspect `app/state_machine/models/conversation_context.py::reset_session_scope`. Re-run all three failing tests + `tests/session/test_conversation_context.py`.

---

### Group H — duplicate-side default keeps already-selected items in choice payload (**P2 / intentional behavior**)

**Failing test**

```
tests/state_machine/handlers/item/add_item/test_group_option_payloads.py::test_side_choice_payload_excludes_already_selected_options
```

**Source under test**

`app/state_machine/handlers/item/add_item/waiting_for_side_handler.py:612-648`:

```python
def _choice_payload(self, context, group):
    allow_dupes = getattr(group, "allow_duplicate_selections", True)   # ← default True
    ...
    if allow_dupes:
        remaining_choice_names = [choice.name for choice in group.choices]
    else:
        ...
```

**Root cause**

Recent duplicate-side support sets the default to `True`. The test fixture is a `SimpleNamespace` without `allow_duplicate_selections`, so `getattr` returns `True` → all sides remain in the prompt. Test expects exclusion.

Per project rules: "duplicate sides allowed only through supported payload semantics if implemented". Defaulting `allow_dupes=True` for any group that lacks the attribute means:

- A group fixture from the menu loader that doesn't set the field → duplicates allowed silently. Risky.

**Fix (recommended)**

Default to `False` and require menu loader to opt-in by setting `allow_duplicate_selections=True` on groups that genuinely want duplicates:

```python
allow_dupes = bool(getattr(group, "allow_duplicate_selections", False))
```

Then update the test that does want duplicate behaviour to set it explicitly. Alternative: keep the `True` default and update this test.

The architectural-safer choice is to default `False`. The duplicate-side display tests (Group C) work either way (they're not testing this branch).

**Risk**

If `MenuStore`/`MenuRepository` doesn't propagate this attribute onto `PendingSideGroup`, switching the default flips behaviour for all production sides. Need to confirm `pending_add_item_factory` reads the field from the `SideGroup` model.

**Test plan**

`tests/state_machine/handlers/item/add_item/test_group_option_payloads.py` + cart duplicate-side tests + a new test that asserts duplicate-allowed groups still keep their choices visible.

---

### Group I — Stale `lastfailed` cache entries (**P3 / not real failures**)

**Affected entries (13)**

```
tests/core/test_quantity_formatter.py::TestParseItemQuantity::test_float_2_5_rounds_to_3
tests/cart/test_cart_summary_builder_money_safety.py::TestMoneyModulesDoNotImportNormalizeFoodQuantity::test_module_does_not_import_normalize_food_quantity[app.cart.cart_summary_builder]
tests/realtime/test_tts_failure_handling.py::TestMultipleTtsFailures::test_each_failure_ends_call
tests/realtime/test_tts_failure_handling.py::TestMultipleTtsFailures::test_success_after_prior_turn_works_normally
tests/realtime/test_conversation_session.py::test_payment_auto_check_dispatches_auto_text_then_cancels
tests/core/test_response_builder_add_item.py::test_response_builder_ask_for_modifier_supports_multi_select_guidance
tests/core/test_response_builder_add_item.py::test_response_builder_ask_for_modifier_uses_open_ended_prompt
tests/state_machine/handlers/item/add_item/test_add_item_quantity_prompt.py::AddItemQuantityPromptTests::test_always_asks_for_quantity_even_when_present_in_first_request
tests/responses/test_side_response_ux.py::TestHasEchableContent::test_single_meaningful_token_is_echable
tests/responses/test_side_response_ux.py::TestHasEchableContent::test_single_meaningful_token_with_stop_words_is_echable
tests/nlu/test_control_phrase_lexicon.py::TestNoInlinePhrasesets::test_control_intent_resolver_has_no_inline_frozenset_literals
tests/nlu/test_control_phrase_lexicon.py::TestNoInlinePhrasesets::test_control_intent_resolver_has_no_inline_phrase_frozensets
tests/menu/test_extracted_menu_components.py::TestMenuQueryService::test_resolve_item_normalized_below_threshold_returns_none
tests/nlu/test_quantity_parser_extras.py::TestSpecialQuantitiesAdditions::test_normalize_quantity_both_of_them
tests/services/test_order_number_generator.py::OrderNumberFormatTests::test_no_random_module_imported_in_generator
tests/state_machine/handlers/item/add_item/test_extracted_components.py::TestNormalizeItemRequestText::test_strips_filler_prefixes[i would like to order two tacos-two tacos]
tests/state_machine/handlers/item/add_item/test_prefill_quantity_decimal.py::TestParseQuantityValue::test_invalid_raw_returns_none[-1]
```

**Root cause**

Each id either:

- references a test name that no longer exists in the source file (renamed/removed), or
- references a parametrize id that no longer exists (parametrize values were edited).

Verified by `grep -rn "<test_name>" tests/` — no matches in non-pycache files.

**Fix**

`rm .pytest_cache/v/cache/lastfailed` after the next clean green run, or wait — pytest will repopulate naturally. Nothing to fix in code.

**Risk**

None.

---

### Group J — `tests/api/test_chat_demo.py` collection failure (**P3**)

**Failing entry**: `tests/api/test_chat_demo.py` (no `::test_name`).

**Root cause (probable)**

The bare-file entry in `lastfailed` indicates pytest could not collect the file. The file imports `from fastapi import FastAPI` at module top. If the test environment lacks `fastapi`, collection raises `ImportError`. Other failure-handling files in the suite (`test_turn_engine_phase2_validation.py`, `test_turn_engine_real_menu_add_item_flows.py`, `test_payment_flow_support.py`) install **stub** modules for `twilio` / `redis` / `app.ml.intent.inference_intent` to allow import without those packages. `test_chat_demo.py` does not stub `fastapi` because it actually depends on the real ASGI app.

**Fix**

Either:

- Install `fastapi` and `httpx` (for `TestClient`) in the CI/test image, or
- Mark the file with `pytest.importorskip("fastapi")` at module top so it is skipped (not failed) when fastapi is absent, or
- Move the test to a separate `tests/integration/` collection that's only enabled when fastapi is installed.

**Test plan**

Re-run with `pip install fastapi httpx` in the test environment.

---

## D. 0-failure plan (ordered)

The order below is chosen so each step lands a runnable green increment without being blocked by the next.

### Step 1 — Stabilise the test environment (P3, 30 min)

- Add `pytest`, `pytest-asyncio`, `fastapi`, `httpx` to `requirements-dev.txt` (or whichever dev requirements the project uses). 
- Wrap `tests/api/test_chat_demo.py` with `pytest.importorskip("fastapi")` as a belt-and-suspenders measure.
- Delete `.pytest_cache/v/cache/lastfailed` after Step 5 to clear Group I.

**Files**: `requirements*.txt`, `tests/api/test_chat_demo.py`, possibly `pytest.ini`. **Re-run**: full collection only (`pytest --collect-only`) — must report 0 collection errors.

### Step 2 — Restore / pin the menu fixture used by deterministic tests (P1, ~2h, ~33 failures)

The single highest-leverage step.

- Decision: keep `app/data/restaurants/demo/menu.json` as the **production** demo menu (free to evolve), and introduce `tests/fixtures/menu/demo_test.json` (or similar) used by every deterministic test that scripts a multi-turn flow. 
- Source it from `menu.20260506_153616.bak` (the menu version under which most of these tests were written).
- Update test helpers (`tests/support/voice_test_harness.py::build_menu_repo`, `tests/state_machine/handlers/item/add_item/test_multi_item_prefill.py::_demo_repo`, `tests/state_machine/handlers/item/add_item/test_prefill_feedback.py::_build_demo_menu_repo`, `tests/core/test_turn_engine_real_menu_add_item_flows.py::_build_menu_repo`, `tests/menu/test_numbered_item_matching.py::_build_repo`, `tests/nlu/test_multi_item_parser.py::_build_demo_store`) to point at the new fixture path.

**Files**: 1 new fixture json + ~6 test helper changes. **Re-run**: all of Group D.

### Step 3 — Fix real production code regressions (P0, ~2h, 4 failures)

- **Group A**: edit `app/realtime/conversation_session.py` to remove the premature `set_phase_speaking()` (line 455) and the post-failure `set_phase_listening()` (line 485). 
- **Group B**: edit `app/core/response_builder.py::_confirm_item` to natural prompt.
- **Group C**: edit `app/cart/read_models/cart_summary_builder.py:140-141` to `f"{label} x{count}"`.
- **Group H** (decision required): change default `allow_duplicate_selections` to `False` in `_choice_payload`, and confirm the menu loader sets it explicitly on groups that need duplicates.

**Re-run**: 
- `tests/realtime/test_tts_failure_handling.py`
- `tests/core/test_response_builder_add_item.py`
- `tests/cart/test_cart_summary_builder_duplicate_sides.py`
- `tests/state_machine/handlers/item/add_item/test_group_option_payloads.py`

### Step 4 — Update stale fixtures and assertions (P1/P2, ~2h, ~11 failures)

- **Group E**: extend `_make_context` in `test_group_resolution_handler.py` to set `modifier_groups=[]` on the pending stub and `selected_modifier_groups={}` on the context.
- **Group G**: rename `_FakeContext.reset()` → `reset_session_scope()` (or add `reset_session_scope` as an alias). Verify `ConversationContext.reset_session_scope` does not wipe `delivery_address.order_number` / `payment_reference`. If it does, fix the synchronizer ordering (set fields *after* reset).
- **Group F**: relax the strict equality on payment_draft_saved payloads to subset checks.

**Re-run**: 
- `tests/state_machine/handlers/item/add_item/test_group_resolution_handler.py`
- `tests/services/test_checkout_extracted_components.py`
- `tests/services/test_checkout_service_payment_sync.py`
- `tests/state_machine/handlers/payment/test_payment_flow_support.py`

### Step 5 — Drop stale `lastfailed` (Group I, P3, 1 min)

- Run the full suite. Pytest will rebuild `lastfailed`. Stale ids drop automatically.

### Step 6 — Stabilise NLU / model tests (no failing entries currently)

This system already isolates the ML model from tests via stub modules (`app.ml.intent.inference_intent`, `app.ml.slot.inference_slot`) — there is no NLU model nondeterminism in the failing set. Document this in `tests/README_voice_ordering_framework.md` so future contributors maintain the discipline.

### Step 7 — Full suite green run

After Steps 1–5 the suite should be green. Re-run with:

```
pytest -ra --strict-markers --strict-config
```

and a targeted re-run of the four highest-impact directories:

```
pytest tests/realtime tests/core tests/state_machine/handlers/item/add_item tests/cart -q
```

---

## E. Claude Code prompt outlines

### Prompt A — Fixture & schema repair (Step 2 + Step 4)

```
Goal: stabilise the test fixtures so the FSM/handler tests pass deterministically.

Tasks:
1. Create tests/fixtures/menu/demo_test.json by copying
   app/data/restaurants/demo/menu.20260506_153616.bak.
2. Update build_menu_repo() in tests/support/voice_test_harness.py to load
   from the new fixture path (settable via env var TEST_MENU_PATH for callers
   that need a different menu).
3. Update the inline menu loaders in the tests below to use the same fixture:
   - tests/core/test_turn_engine_real_menu_add_item_flows.py::_build_menu_repo
   - tests/state_machine/handlers/item/add_item/test_multi_item_prefill.py::_demo_repo
   - tests/state_machine/handlers/item/add_item/test_prefill_feedback.py::_build_demo_menu_repo
   - tests/menu/test_numbered_item_matching.py::_build_repo
   - tests/nlu/test_multi_item_parser.py::_build_demo_store
   - tests/regression/test_required_ordering_scenarios.py (all helpers loading the demo menu)
   - tests/core/test_turn_engine_phase2_validation.py::_build_menu_repo
   - tests/nlu/test_ordering_nlu_behavior.py (via voice_test_harness)
4. In tests/state_machine/handlers/item/add_item/test_group_resolution_handler.py,
   extend _make_context to also expose modifier_groups=[] on the pending stub
   and selected_modifier_groups={} on the context.
5. Rename _FakeContext.reset() to reset_session_scope() in
   tests/services/test_checkout_extracted_components.py (and add a `reset` alias
   so any other consumer keeps working).

DO NOT modify app/.
After each step, run the affected test directory in isolation and report
pass/fail count.
```

### Prompt B — Real production bug fixes (Step 3, Groups A/B/C)

```
Goal: fix production-code regressions detected by the test suite.

CONTEXT: the system is a voice ordering FSM where transport owns SPEAKING,
and the FSM rule set forbids "yes/no" keyword prompts.

Tasks (each independent):

1. app/realtime/conversation_session.py
   - Remove the unconditional self.set_phase_speaking() at line 455.
     Let voice_stream_server.speak_response own the SPEAKING transition.
   - In the TTSFailureError except arm (around line 472–491), remove
     self.set_phase_listening() before handle_playback_failure / end_call.
     Phase should remain at PROCESSING (or be promoted to a new TERMINATING
     phase if you choose to add one) until end_call/transfer_call resolves.

2. app/core/response_builder.py::_confirm_item
   - Replace the trailing "right? Yes or no." with a natural prompt:
     `return f"{item_name}, is that right?"`.

3. app/cart/read_models/cart_summary_builder.py
   - Lines 140–141: change the duplicate-side label from
     `f"{count} {label}"` to `f"{label} x{count}"`.
   - Keep speech-path renderers in app/responses/cart_responses.py untouched
     so TTS still says "two cokes".

Re-run after each step:
- (1) tests/realtime/test_tts_failure_handling.py
- (2) tests/core/test_response_builder_add_item.py
- (3) tests/cart/test_cart_summary_builder_duplicate_sides.py
```

### Prompt C — Response / FSM expectation updates (Step 4 partial, Groups F/H)

```
Goal: align stale test assertions and behaviour defaults with the current
intended architecture.

Tasks:

1. tests/state_machine/handlers/payment/test_payment_flow_support.py
   - PaymentFlowSupportTests::test_failed_payment_returns_draft_retry_response
     asserts strict equality of response_payload. The production code now
     records observability events under `_payment_events`. Relax the assertion
     to:
       * order_number == "1234567"
       * `_payment_events`[-1]["event_name"] == "payment_failed"
       * `_payment_events`[-1]["metadata"]["status"] == "failed"

2. app/state_machine/handlers/item/add_item/waiting_for_side_handler.py
   ::_choice_payload
   - Change the default of `allow_dupes` from True to False:
       `allow_dupes = bool(getattr(group, "allow_duplicate_selections", False))`
   - Then in app/state_machine/handlers/item/add_item/pending_add_item_factory.py
     (or wherever PendingSideGroup is built from SideGroup), propagate
     `allow_duplicate_selections` from SideGroup so groups that genuinely
     allow duplicates keep their behaviour.
   - If your menu loader does not surface this field yet, add it as an
     opt-in attribute defaulting to False on SideGroup.

Re-run:
- tests/state_machine/handlers/payment/test_payment_flow_support.py
- tests/state_machine/handlers/item/add_item/test_group_option_payloads.py
- tests/cart/test_cart_summary_builder_duplicate_sides.py    (regression check)
```

### Prompt D — Dependency / test environment cleanup (Step 1, Group J)

```
Goal: eliminate file-level collection failures and the suite's ability to
run in a minimal environment.

Tasks:

1. requirements (or requirements-dev.txt):
   - Pin pytest, pytest-asyncio, fastapi, httpx as test-only deps if not
     already present. Verify the docker test image installs them.

2. tests/api/test_chat_demo.py
   - At module top, add:
       import pytest
       pytest.importorskip("fastapi")
       pytest.importorskip("httpx")
     so the file is skipped (not failed) on minimal envs.

3. After the suite runs cleanly, delete .pytest_cache/v/cache/lastfailed.
```

### Prompt E — NLU deterministic test stabilisation

```
The current failing set has no NLU nondeterminism — the suite already stubs
app.ml.intent.inference_intent and app.ml.slot.inference_slot. To prevent
future regressions:

1. Move the stub-module installation block currently duplicated across
   tests/core/test_turn_engine_*.py and
   tests/state_machine/handlers/payment/test_payment_flow_support.py
   into a shared fixture in tests/conftest.py:
       @pytest.fixture(autouse=True, scope="session")
       def _install_ml_stubs():
           from tests.support.voice_test_harness import install_test_stubs
           install_test_stubs()
2. Document the stubbing contract in
   tests/README_voice_ordering_framework.md.
3. Add a pytest mark `requires_model` and skip-by-default any test that
   imports the real model bundles, so model-dependent tests are explicit.

Do NOT change production code in this step.
```

---

## F. Final recommendation

### Should we fix tests first or code first?

**Fixtures first, then code, then assertions.**

Justification: the menu drift (Group D) hides whether the underlying handlers and FSM work correctly. Until Group D is resolved, you cannot tell if a `test_real_menu_*` failure is "menu fixture lost an item" or "AddItemHandler regressed". Fix the deterministic fixture, then patch the four real production bugs (A/B/C/H), then update the stale assertions (E/F/G).

### Which batch should go first?

Order:

1. **Prompt D** — env cleanup (cheapest, unblocks the rest).
2. **Prompt A** — fixture/schema repair (largest count, decouples test contracts from production menu edits).
3. **Prompt B** — real production bug fixes (unblocks payment + TTS + UX correctness).
4. **Prompt C** — assertion / default updates.
5. **Prompt E** — NLU stub centralisation (preventive).

### Which failures must not be ignored?

- **Group A** (TTS phase invariant) — production race-condition risk during TTS outages.
- **Group B** (`Yes or no.` confirmation) — direct violation of a stated architecture rule; if left in place it normalises keyword-driven NLU.
- **Group G** (`reset_session_scope` ordering) — possibly a real bug that wipes payment confirmation fields. Verify `ConversationContext.reset_session_scope` before declaring this a fixture issue.
- **Scenario 11** (`test_scenario_11_checkout_during_pending_modifier_must_not_short_circuit`) — payment/checkout safety regression test. Currently failing because of menu drift, but the assertion itself defends a P0 invariant ("checkout while pending modifier must not short-circuit"). After fixing the menu fixture, ensure this passes.

### Which failures can be safely deferred?

- **Group I** (stale `lastfailed` ids) — they self-clear after the next green run.
- **Group H** if the team explicitly wants the "duplicates allowed by default" behaviour: leave the production default as `True` and update the test fixture to mark the group `allow_duplicate_selections=False`. Document the decision either way.

### Sanity check before merging fixes

After all five prompts:

- Re-run the full suite under the strict pyproject options (`-ra --strict-markers --strict-config`).
- Re-run `tests/realtime`, `tests/core`, `tests/state_machine/handlers/item/add_item`, `tests/cart`, `tests/services`, `tests/regression` separately to ensure isolation.
- Confirm `git grep -n "Yes or no" app/` returns nothing (Group B regression check).
- Confirm `git grep -n "set_phase_listening()" app/realtime/` shows only intentional sites (post-playback, not post-failure).

---

## Appendix — failing-test inventory (raw)

For traceability, the canonical 75 entries from `.pytest_cache/v/cache/lastfailed` (live `tests/*` only) grouped by file:

```
 1  tests/api/test_chat_demo.py                                                               [Group J]
 3  tests/cart/test_cart_summary_builder_duplicate_sides.py                                   [Group C]
 1  tests/cart/test_cart_summary_builder_money_safety.py                                      [Group I — stale param]
 1  tests/core/test_quantity_formatter.py                                                     [Group I — stale name]
 3  tests/core/test_response_builder_add_item.py                                              [Group B (1) + Group I (2)]
 6  tests/core/test_turn_engine_phase2_validation.py                                          [Group D]
 7  tests/core/test_turn_engine_real_menu_add_item_flows.py                                   [Group D]
 1  tests/menu/test_extracted_menu_components.py                                              [Group I — stale name]
 1  tests/menu/test_numbered_item_matching.py                                                 [Group D]
 2  tests/nlu/test_control_phrase_lexicon.py                                                  [Group I — stale name]
 1  tests/nlu/test_multi_item_parser.py                                                       [Group D]
 2  tests/nlu/test_ordering_nlu_behavior.py                                                   [Group D]
 1  tests/nlu/test_quantity_parser_extras.py                                                  [Group I — stale name]
 1  tests/realtime/test_conversation_session.py                                               [Group I — stale name]
 5  tests/realtime/test_tts_failure_handling.py                                               [Group A (3) + Group I (2)]
 1  tests/regression/test_required_ordering_scenarios.py                                      [Group D]
 4  tests/responses/test_address_response_variation.py                                        [feature gap — see note below]
 2  tests/responses/test_side_response_ux.py                                                  [Group I — stale names]
 2  tests/services/test_checkout_extracted_components.py                                      [Group G]
 1  tests/services/test_checkout_service_payment_sync.py                                      [Group G]
 1  tests/services/test_order_number_generator.py                                             [Group I — stale name]
 4  tests/state_machine/handlers/item/add_item/test_add_item_handler.py                       [Group D + payload schema]
 1  tests/state_machine/handlers/item/add_item/test_add_item_quantity_prompt.py               [Group I — stale name]
 2  tests/state_machine/handlers/item/add_item/test_extracted_components.py                   [Group I — stale param/name]
 1  tests/state_machine/handlers/item/add_item/test_group_option_payloads.py                  [Group H]
 7  tests/state_machine/handlers/item/add_item/test_group_resolution_handler.py               [Group E]
 4  tests/state_machine/handlers/item/add_item/test_multi_item_prefill.py                     [Group D]
 7  tests/state_machine/handlers/item/add_item/test_prefill_feedback.py                       [Group D]
 1  tests/state_machine/handlers/item/add_item/test_prefill_quantity_decimal.py               [Group I — stale param]
 1  tests/state_machine/handlers/payment/test_payment_flow_support.py                         [Group F]
```

### Note on `tests/responses/test_address_response_variation.py` (4 tests)

The four failing tests assert that `repeat_delivery_house_number`, `repeat_delivery_street`, `repeat_delivery_secondary_address` produce **three different strings** for `attempt_count` 1/2/3, and that `address_collection_giving_up` exists as a key.

`app/core/response_builder.py:312-314` defines those keys as constants — same string regardless of `attempt_count`. `repeat_delivery_secondary_address` and `address_collection_giving_up` are **not present** in the builder at all.

This is a **feature gap** (escalating attempt-count copy variation, plus a graceful-handoff key) rather than a regression. Treat as P2 product work, not a P0 bug:

- Implement variation by mapping `attempt_count` → distinct copy in those keys.
- Add `address_collection_giving_up` returning a sentence containing "trouble" and "team member".

I did not include these in any of the production-bug groups because the tests are aspirational — the implementation never had them.

---

*Audit produced 2026-05-07 by static analysis only. No code modified.*
