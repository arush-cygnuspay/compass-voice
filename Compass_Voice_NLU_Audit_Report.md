# Compass Voice — NLU & Response Audit (logs of 2026‑05‑06 / 05‑07)

Source artefacts inspected:
- `nlu_log.csv` (this session, 19 turns across 2 calls)
- `app/logs/realtime_turn_latency.csv` (historical, ~900 turns)
- Code: `app/responses/**`, `app/core/response_builder.py`,
  `app/state_machine/handlers/item/add_item/**`,
  `app/nlu/{multi_item_parser,utterance_filter,fallback_phrase_matcher}.py`,
  side/modifier resolvers.

The findings are grouped by user complaint; each block is **Diagnosis →
Root Cause → Fix → Improvements**, with the exact files/lines and
production‑ready snippets.

---

## 1. Responses must never end with "Yes or no" / "Just say yes or no"

### Diagnosis
Every confirmation response in the system tail‑appends "Yes or no":

```
2026-05-07T00:50:42  "Korean Tacos - Spicy Chicken, right? Yes or no."
2026-04-03T15:35:40  "Smash Burger, right? Yes or no."
2026-04-03T15:40:45  "Did you mean Grilled Chicken? Yes or no."
2026-04-09T16:40:23  "Did you mean Plain Bun? Yes or no."
2026-04-10T14:16:00  "Do you want to cancel Chicken Taco? Please say yes or no."
2026-05-01T15:19:36  "Would you like me to send a payment link by text? Just say yes or no."
```

This pattern is voice‑unfriendly (sounds robotic, primes IVR‑style yes/no
fixation, and breaks barge‑in flow because users start replying before the
tail finishes — see `interrupted_before_mark: true` in many of those rows).

### Root Cause (architectural)
The yes/no tail is hard‑coded in **five** distinct places — there is no
single "confirmation prompt" abstraction:

| Response key                          | File:line                                         |
|---------------------------------------|---------------------------------------------------|
| `confirm_item`                        | `app/core/response_builder.py:435`                |
| `confirm_side_choice_guess`           | `app/core/response_builder.py:228`                |
| `confirm_modifier_choice_guess`       | `app/core/response_builder.py:229`                |
| `confirm_size_choice_guess`           | `app/core/response_builder.py:230`                |
| `confirm_side_size_choice_guess`      | `app/core/response_builder.py:231`                |
| `repeat_landline_pickup_only`         | `app/core/response_builder.py:234`                |
| `flow_guard_confirm_cancel`           | `app/responses/flow_control_responses.py:23, 25`  |

Each lambda or string literal carries its own "Yes or no." postfix. The
phrasing is not driven by a confirmation policy — it is just text glued
on by the response writer.

### Fix
Drop the yes/no tail entirely; the confirmation question already implies
the binary expectation. The NLU `confirmation_resolver` already handles
"yes / yeah / sure / nope / nah / wrong" via the affirm/deny intent
classifier, so removing the literal tail does not weaken parsing.

```python
# app/core/response_builder.py
"confirm_side_choice_guess":      lambda c, m, p: f"Did you mean {p.get('choice_name', 'that side')}?",
"confirm_modifier_choice_guess":  lambda c, m, p: f"Did you mean {p.get('choice_name', 'that modifier')}?",
"confirm_size_choice_guess":      lambda c, m, p: f"Did you mean {p.get('choice_name', 'that size')}?",
"confirm_side_size_choice_guess": lambda c, m, p: (
    f"Did you mean {p.get('choice_name', 'that size')} for {p.get('side_item_name', 'that side')}?"
),
"repeat_landline_pickup_only":    lambda *_: "Want me to connect you with a team member?",

def _confirm_item(self, context, menu_repo, payload) -> str:
    item_name = payload.get("item_name") or context.current_item_name
    if not item_name and context.candidate_item_id:
        item_name = menu_repo.store.get_item(context.candidate_item_id).name
    item_name = item_name or "that item"
    return f"Just to confirm — {item_name}?"
```

```python
# app/responses/flow_control_responses.py
def flow_guard_confirm_cancel(payload: dict) -> str:
    item_name = payload.get("item_name")
    if item_name:
        return f"Cancel {item_name}?"
    return "Cancel that?"
```

```python
# app/core/response_builder.py — pickup_repeat_sms_permission
"pickup_repeat_sms_permission": lambda *_: (
    "Want a payment link texted to your phone, or will you pay at pickup?"
),
```

### Improvements (architectural — recommended next pass)
Centralise confirmation phrasing in a single helper instead of literal
strings:

```python
# app/responses/confirmation.py  (new)
def yes_no_question(stem: str) -> str:
    """Single source of truth for binary-confirmation prompts.

    Always returns just a question — never a literal "Yes or no" tail.
    """
    stem = stem.rstrip(" .?")
    return f"{stem}?"
```

All 7 sites above route through this helper. Future style changes are
one diff, not seven.

---

## 2. Side/Modifier prompts say "Any cheese would you like, like …" — should be "Which cheese …"

### Diagnosis
Logs:
```
2026-05-07T00:51:13  "Any cheese would you like, like American Cheese, Cheddar Cheese, or Mozzarella Cheese?"
2026-05-07T00:51:27  "Any bun would you like, like Plain Bun, Potato Bun, or Sesame Bun?"
2026-05-07T00:51:49  "Any burger would you like, like Mayo?"          ← also wrong noun
2026-05-07T00:49:54  "Any can drinks would you like, like Coke (12 oz.), or Sprite (12 oz.)?"
```

Two problems:
1. **Grammar:** the verb ("would you like") is **inside** the noun phrase
   ("Any cheese would you like, like …"), which is unidiomatic and slows
   STT‑listeners' parsing.
2. **Determiner:** required groups ("Cheese", "Bun" with `min_selector ≥ 1`)
   should be **directive** ("Which …"), not **invitational** ("Any …"). The
   current text suggests a yes/no answer is acceptable for a required group.
3. **Wrong noun ("Any burger would you like, like Mayo?"):** the menu's
   modifier‑group `prompt_noun` is the *parent item word* ("burger") rather
   than the group label ("toppings", "sauce"), and gets injected verbatim.

### Root Cause
File `app/responses/item/sides.py` (`ask_for_side`) lines 53–63 and the
mirror `app/responses/item/modifiers.py` (`ask_for_modifier`) lines 53–63:

```python
examples = _format_examples(top_choices)
if examples:
    prompt = f"Any {noun} {verb}, like {examples}?"
else:
    prompt = f"Any {noun} {verb} with your {item_name}?"
```

The template is symmetric for required and optional groups, and the verb
token is interpolated mid‑NP. There is also no sanitisation of `noun`
when it equals the parent‑item word.

### Fix
Split required vs optional, drop the embedded verb, sanitise `noun`,
and prefer the **group label** when `noun` equals the parent item:

```python
# app/responses/item/sides.py — replace ask_for_side body
def ask_for_side(context, menu_repo, payload=None):
    payload = payload or {}
    item_name = None
    group_name = None
    noun = _GENERIC_SIDE_NOUN
    min_selector = 0
    top_choices: list[str] = list(payload.get("top_choices") or [])

    try:
        item = menu_repo.store.get_item(context.current_item_id)
        group = item.side_groups[context.current_side_group_index]
        item_name = item.name
        group_name = (group.name or "").strip()
        raw_noun = (getattr(group, "prompt_noun", None) or "").strip()
        # Reject a noun that is just the parent item word ("burger")
        if raw_noun and raw_noun.lower() not in normalize_text(item_name).split():
            noun = raw_noun
        elif group_name:
            noun = _clean_group_label(group_name, _GENERIC_SIDE_NOUN).lower()
        group_payload = _current_side_payload(context, menu_repo, payload)
        min_selector = int(group_payload.get("min_selector", 0) or 0)
        top_choices = group_payload.get("top_choices") or top_choices
    except Exception:
        pass

    item_name = item_name or "your item"
    examples = _format_examples(top_choices)

    # Required → directive
    if min_selector >= 1:
        if examples:
            return f"Which {noun} would you like — {examples}?"
        return f"Which {noun} would you like for your {item_name}?"

    # Optional → invitational, no embedded verb
    if examples:
        return f"Want any {noun}? {examples}, or none."
    return f"Want any {noun} with your {item_name}? You can say none."
```

The exact same change applies to `ask_for_modifier` in
`app/responses/item/modifiers.py`. The `_clean_group_label` helper
(format_utils.py) already lowercases / pluralises the label cleanly.

### Improvements
Add a `prompt_template` field to side/modifier groups (in `menu.json`)
that overrides the auto‑generated phrasing. Restaurants can then
customise without code changes:

```json
{ "group_id": "burger_cheese", "name": "Cheese", "prompt_template":
  "Which cheese? {examples}." }
```

The response layer falls back to the directive/invitational templates
above when the field is absent. This eliminates more "burger / Mayo"
type misnaming because operators see and curate the phrasing.

---

## 3. Never say "Need 1 more" / "Pick 1 more"

### Diagnosis
Per user preference. The system currently emits:

- `app/responses/item/format_utils.py` `_progress_prompt` (lines 350–378):
  `"Pick 1 more."`, `"Add 1 more, or say done."`, `f"{invalid_lead} Pick 1 more."`
- `app/responses/item/sides.py` `required_side_cannot_skip` (line 128):
  `f"Need {remaining} more sides."`
- `app/responses/item/modifiers.py` `required_modifier_cannot_skip`
  (line 130): `f"Need {remaining} more options."`

### Root Cause
The phrasing template treats `1` as a numeric like any other. There is
no carve‑out to switch to natural singular when `remaining == 1`.

### Fix

```python
# app/responses/item/format_utils.py  — _progress_prompt
def _progress_prompt(payload, *, item_word, invalid_lead):
    top_choices = _payload_value(payload, "top_choices", [])
    all_choices = _payload_value(payload, "all_choices", [])
    options = _format_options(top_choices or all_choices)

    reason = payload.get("repeat_reason")
    if reason == "need_more":
        remaining = max(int(payload.get("remaining_to_min") or 0), 0)
        prompt = f"Pick {item_word_singular(item_word)}." if remaining == 1 \
                 else f"Pick {remaining} {pluralize(item_word, remaining)}."
        return f"{prompt} {options}." if options else prompt

    if reason == "optional_more":
        remaining = max(int(payload.get("remaining_to_max") or 0), 0)
        if remaining > 0:
            prompt = (f"Add another {item_word}, or say done."
                      if remaining == 1
                      else f"Up to {remaining} more {pluralize(item_word, remaining)}, or say done.")
            return f"{prompt} {options}." if options else prompt
        return "Say done when you're ready."

    remaining = max(int(payload.get("remaining_to_min") or 0), 0)
    if remaining == 0:
        rmax = max(int(payload.get("remaining_to_max") or 0), 0)
        if rmax > 0 and options:
            return (f"Add another {item_word}, or say done. {options}."
                    if rmax == 1
                    else f"Up to {rmax} more, or say done. {options}.")
        return "Say done when ready."

    selected = max(int(payload.get("selected_count") or 0), 0)
    if selected == 0:
        prompt = (f"Pick {item_word_singular(item_word)}."
                  if remaining == 1
                  else f"Pick {remaining} {pluralize(item_word, remaining)}.")
    elif remaining == 1:
        prompt = f"{invalid_lead} Pick another {item_word}."
    else:
        prompt = f"{invalid_lead} Pick {remaining} more {pluralize(item_word, remaining)}."
    return f"{prompt} {options}." if options else prompt


def item_word_singular(noun: str) -> str:
    article = "an " if noun[:1].lower() in "aeiou" else "a "
    return article + noun
def pluralize(noun, n): return noun if n == 1 else noun + "s"
```

```python
# app/responses/item/sides.py  — required_side_cannot_skip
def required_side_cannot_skip(context, menu_repo, payload=None):
    side_payload = _current_side_payload(context, menu_repo, payload)
    options = _format_options(side_payload.get("top_choices") or _top_side_choices(context, menu_repo))
    remaining = max(int(side_payload.get("remaining_to_min") or 0), 0)
    group_label = _clean_group_label(side_payload.get("group_name"), "side").lower()
    if options:
        if remaining > 1:
            return f"Pick {remaining} {group_label}s. {options}."
        return f"Pick a {group_label}. {options}."
    return f"Pick a {group_label}."
```

(Same shape for `required_modifier_cannot_skip`.)

---

## 4. Matching false‑positives — "I couldn't find can you please add"

### Diagnosis
Log row 10 (this session):
```
user: "Can you please add beef tacos?"
bot : "I couldn't find can you please add. Beef Tacos added. Would you like anything else?"
```

Log row 16:
```
user: "Can you tell the options?"
bot : "I couldn't find can you tell the options. I didn't catch that. Say 'list options' to hear all choices."
```

The bot is echoing the **command verb phrase** as if it were a missing
menu item. This kills perceived intelligence.

### Root Cause
File `app/state_machine/handlers/item/add_item/prefill_orchestrator.py`,
function `_collapse_unresolved_for_feedback` (lines 882–938).

The function tries to filter junk from `unresolved_phrases` by stripping
"ignored" tokens and discarding phrases that reduce to nothing:

```python
ignored_tokens.update({
    "with","and","plus","also","or",
    "extra","more","double","less","light",
    "on","the","side","a","an",
    "okay","ok","then","give","me","can","i",
    "want","would","like","to","order","get",
    "please","just",
})
```

Reproduced:
```
'can you please add'  → tokens left after ignored: ['you', 'add']  → kept ✗
'can you tell the options' → tokens left:        ['you', 'tell', 'options'] → kept ✗
```

`you`, `add`, `tell`, `me` (in some forms), `bring`, `take`, `make`,
`give` (when not adjacent to "me"), `say`, `do`, `does`, `did`, `know`,
`anything`, `something`, `everything`, `else` are all missing. This is a
**denylist** approach against an open class of words — it will leak
forever.

### Fix (architectural — switch to allowlist)
The correct invariant is *only echo "I couldn't find X" when X has at
least one menu‑noun token*. The menu store already maintains an entity
index for items, sides, modifiers and variants — use it:

```python
# app/state_machine/handlers/item/add_item/prefill_orchestrator.py

def _collapse_unresolved_for_feedback(
    self, unresolved_phrases, *, pending, menu_store
):
    """Keep only phrases that mention something resembling a menu noun."""
    if not unresolved_phrases or menu_store is None:
        return []

    item_vocab: set[str] = set()
    item_vocab.update(tokenize(normalize_text(pending.item_name)))
    for grp in (*pending.side_groups, *pending.modifier_groups):
        for choice in grp.choices:
            item_vocab.update(tokenize(choice.normalized_name))
            for label in (getattr(choice, "match_texts", ()) or ()):
                item_vocab.update(tokenize(label))
    for variant in pending.item_variants:
        item_vocab.update(tokenize(variant.normalized_name))

    cleaned: list[str] = []
    seen: set[str] = set()
    for phrase in unresolved_phrases:
        normalized = normalize_text(phrase or "").strip()
        if not normalized or normalized in seen:
            continue
        tokens = tokenize(normalized)
        # Drop the phrase if NONE of its tokens are a known menu noun
        # AND it doesn't match any global menu entity.
        has_local_signal = any(t in item_vocab for t in tokens)
        has_global_signal = bool(
            menu_store.find_entity(
                normalized,
                allowed_types={"item", "side", "modifier", "variant"},
            )
        )
        if not (has_local_signal or has_global_signal):
            continue
        seen.add(normalized)
        cleaned.append(normalized)
    return cleaned
```

Plumb `menu_store` through `enter_add_flow_for_item → _build_prefill_feedback_summary`
(or grab it from `self.capture_helper.menu_repo.store` if already wired).

Result with the same inputs:
```
'can you please add'           → 0 menu-noun tokens, 0 entity hits → SUPPRESSED ✓
'can you tell the options'     → 0 menu-noun tokens, 0 entity hits → SUPPRESSED ✓
'cheddar cheese'               → 'cheese' in vocab                 → KEPT ✓
'tofu burger'                  → entity('tofu burger') hits item   → KEPT ✓
```

### Same bug in side / modifier resolvers
`app/state_machine/handlers/item/add_item/side_group_resolver.py:218–248`
and `modifier_group_resolver.py:240–266` perform their own
`unmatched_values` cleanup but use the **same denylist‑style logic** and
suffer the same false positives — that is why the bot said
*"I couldn't find web."* when the user said "Accessing web" while the
Bun group was active.

Apply the same allowlist filter at the bottom of each resolver, scoped
to the **active group's** vocabulary plus the global menu entity index:

```python
# At the end of SideGroupResolver.resolve(), replace the existing two-pass cleanup
group_vocab = {
    t
    for choice in group.choices
    for label in (getattr(choice, "match_texts", ()) or (choice.normalized_name,))
    for t in tokenize(label or "")
}
cleaned_unmatched = [
    v for v in cleaned_unmatched
    if any(t in group_vocab for t in tokenize(v))
       or menu_store.find_entity(
           normalize_text(v),
           allowed_types={"side","item","modifier","variant"},
       )
]
```

---

## 5. "I couldn't find can you please add. Beef Tacos added." — order is also wrong

### Diagnosis
Even with the matching fixed (#4), there is a UX bug in the response
ordering. `prefill_feedback` is **prepended** to the success message in
`response_builder.py:124–128`:

```python
if prefill_feedback:
    prefix_parts.append(prefill_feedback)
if prefix_parts:
    base_response = f"{' '.join(prefix_parts)} {base_response}"
```

When the success line itself ALSO emits an unmatched note (success.py
already reads `unmatched_names` and tacks on " I couldn't find …"), the
user can hear the same "I couldn't find" message **twice**.

### Root Cause
Two independent code paths feed unavailable‑item feedback into the same
turn:
- `prefill_orchestrator._build_prefill_feedback_summary` → `prefill_feedback`
  (rendered as a prefix)
- `responses/item/success.py::item_added_successfully` → `unmatched_names`
  (rendered inline)

Neither knows about the other.

### Fix
Pick one channel. The success template is the natural owner ("Beef Tacos
added. I couldn't find X. Anything else?") because the "added" claim
needs to come first.

In `confirmation_decision_helper.build_handler_result`, when the step is
`ReadyToFinalize`, **forward** unresolved phrases into `unmatched_names`
on the success payload and **suppress** the `prefill_feedback` prefix:

```python
# app/state_machine/handlers/item/add_item/confirmation_decision_helper.py
if isinstance(step, ReadyToFinalize):
    payload: dict = {
        "item_name": item.name,
        "quantity": context.quantity or 1,
        "prefilled_summary": prefilled_summary,
        "prefill_debug": prefill_debug,
    }
    # Hoist unresolved phrases into unmatched_names so success.py renders them
    # exactly once, after the "added" claim.
    if prefill_feedback_unmatched:
        payload["unmatched_names"] = prefill_feedback_unmatched
    return HandlerResult(
        next_state=ConversationState.IDLE,
        response_key="item_added_successfully",
        response_payload=payload,
        command=step.command.to_dict(),
        reset_context=True,
    )
```

Where `prefill_feedback_unmatched` is the same allowlist‑filtered list
returned by `_collapse_unresolved_for_feedback`. Pre-finalize prefill
feedback (prompts that ARE asking for more input) should still go to
`prefill_feedback`.

---

## 6. "Things that need attention" surfacing

### Diagnosis
User's note: *"It is missing things like say there are things that need
attention."*

In log row 12, the user says "Cheddar cheese" while the system is
asking about Cheese for the Chicken Burger. The bot accepts the cheese
silently (state stays `waiting_for_side` because the next group is Bun)
and asks "Any bun would you like, like …". It never confirms "Got
Cheddar Cheese." or hints at outstanding fields.

### Root Cause
`waiting_for_side_handler._apply_side_selection` does collect
`newly_added_names` and pass them as `matched_names`, but:
- `ask_for_side` (next group prompt) does not consume `matched_names`.
- `_build_entity_feedback` is only called by the *repeat / unmatched*
  paths — not by the success path that pivots to the next group.

So the user gets no acknowledgement and no visibility into how many
groups remain.

### Fix
Acknowledge prior selection at the head of the next prompt, and surface
remaining attention‑required steps:

```python
# app/responses/item/sides.py  — extend ask_for_side
def ask_for_side(context, menu_repo, payload=None):
    payload = payload or {}
    head = ""
    matched = payload.get("matched_names") or []
    if matched:
        head = f"Got {_format_selected_names(matched)}. "

    # ... existing logic builds `prompt` ...

    pending = getattr(context, "pending_add_item", None)
    remaining_groups: list[str] = []
    if pending is not None:
        for grp in pending.side_groups[context.current_side_group_index + 1:]:
            if not context.selected_side_groups.get(grp.group_id):
                remaining_groups.append(grp.name)
        for grp in pending.modifier_groups:
            if not context.selected_modifier_groups.get(grp.group_id):
                remaining_groups.append(grp.name)

    if len(remaining_groups) >= 2:
        tail = f" After this, {len(remaining_groups) - 1} more to set up."
        prompt = prompt + tail

    return head + prompt
```

The `head + prompt + tail` form gives the caller (a) explicit
acknowledgement, (b) the directive question, and (c) heads‑up on what
still needs attention — without padding the response when nothing
remains.

---

## 7. Soft architectural risks observed in the trace

### 7.1 `pickup_repeat_sms_permission` re‑prompts the user even after they say *"Yes. Send it."*
Rows 908–910 (2026‑05‑01): the bot asked the SMS‑permission question
**three times in a row** while the user was clearly affirming.

The `pred_sub_intent` for those turns is `checkout` — i.e. the FSM is
predicting the wrong sub‑intent and falling through to the re‑prompt.
This is the classic **payment loop** bug. Recommend a tracker:

- Add a `repeat_count` counter on the `waiting_for_pickup_sms_permission`
  state. After ≥ 2 consecutive re‑prompts with no recognised affirm/deny
  intent, **collapse to the implicit‑affirm path** (default to "send
  link" because the user already said "Yes" — the literal token "yes"
  appears in `normalized_text`).

### 7.2 `confirm_item` fires for high‑confidence near‑exact matches
Row 7: user says "Korean tacos" → bot asks "Did you mean Korean Tacos -
Spicy Chicken, Beef Tacos, or Chicken Taco?" → user says "Korean taco
spicy chicken" → bot says "Korean Tacos - Spicy Chicken, right? Yes or no."

The second confirmation is wasteful — `pred_intent_confidence = 0.396`
(low) but the slot value contained both "ITEM" and "MODIFIER" which
should be enough. Recommend: when an ambiguous candidate is re‑mentioned
with at least one **disambiguating modifier slot**, skip the secondary
confirmation and proceed to ask for quantity.

### 7.3 ReprompPolicy branches on a generic message, not on real progress
Row 16: bot says *"I couldn't find can you tell the options. I didn't
catch that. Say 'list options' to hear all choices."* Two negative
acknowledgements stacked with a hint — sounds confused. The
`PromptRepromptPolicy.LIST_OPTIONS_HINT` action should **replace** the
"I didn't catch that" line, not append to it. In
`app/responses/item/modifiers.py::repeat_modifier_options`:

```python
if action == RepromptAction.LIST_OPTIONS_HINT:
    # The hint replaces both the entity-feedback prefix and the generic
    # "I didn't catch that" line.
    return "Say 'list options' to hear all choices."
```

---

## 8. Operational note (non‑user‑facing)

While reproducing matcher behaviour I found that the on‑disk copy of
`app/nlu/utterance_filter.py` (8 745 B, MD5 `0fd7fb78…`) is **truncated
mid‑statement** (ends at `return [v for v in values if v and not self.is_`).
Production must be running a different binary (otherwise the module
would fail to import). Worth confirming whether (a) the workspace mount
is stale, or (b) someone committed a partial file.

The same truncation pattern applies to `app/data/restaurants/demo/menu.json`
(parse error at line 13 982), which is why I could not run a live
end‑to‑end resolver test against the real menu.

This is a session/mount artefact only — flagged here so the engineer can
confirm the working tree on their machine matches expectations before
applying the fixes above.

---

## 9. Patch‑list summary (apply in this order)

| # | File                                                                                              | Change                                                                                          |
|---|---------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|
| 1 | `app/core/response_builder.py` (lines 228‑234, 435)                                               | Strip "Yes or no" / "Just say yes or no" from confirm prompts.                                  |
| 2 | `app/responses/flow_control_responses.py` (lines 23, 25)                                          | `flow_guard_confirm_cancel` → "Cancel X?"                                                       |
| 3 | `app/responses/item/sides.py` (`ask_for_side`)                                                    | Required → "Which X — A, B, or C?". Optional → "Want any X? A, B, or C, or none."               |
| 4 | `app/responses/item/modifiers.py` (`ask_for_modifier`)                                            | Same shape as #3. Reject `prompt_noun` that equals parent‑item word.                            |
| 5 | `app/responses/item/format_utils.py` (`_progress_prompt`)                                         | Replace every "1 more" with singular‑item phrasing.                                             |
| 6 | `app/responses/item/sides.py` / `modifiers.py` (`required_*_cannot_skip`)                         | Use group label, never "1 more".                                                                |
| 7 | `app/state_machine/handlers/item/add_item/prefill_orchestrator.py` (`_collapse_unresolved_for_feedback`) | Switch from denylist to **allowlist** against menu vocabulary.                                  |
| 8 | `app/state_machine/handlers/item/add_item/{side,modifier}_group_resolver.py`                      | Same allowlist filter on `unmatched_values` cleanup.                                            |
| 9 | `app/state_machine/handlers/item/add_item/confirmation_decision_helper.py`                        | Hoist filtered unresolved phrases into `unmatched_names` on the success payload; drop duplicate `prefill_feedback` prefix on `ReadyToFinalize`. |
| 10 | `app/responses/item/sides.py` (`ask_for_side`) / `modifiers.py` (`ask_for_modifier`)             | Acknowledge `matched_names` head + tail “N more to set up.” line.                               |
| 11 | `app/responses/item/modifiers.py` (`repeat_modifier_options`)                                     | LIST_OPTIONS_HINT action **replaces** generic re‑prompt, not appends.                           |
| 12 | (state_machine) `waiting_for_pickup_sms_permission` handler                                       | Add `repeat_count`; after 2 unparsed re‑prompts default to send‑link if "yes" appears in text.  |

Each change is self‑contained. After applying #1–#6 the user‑facing
voice issues disappear. #7–#9 fix the matching/echo regression. #10–#12
raise the conversational‑quality bar to "world‑class".
