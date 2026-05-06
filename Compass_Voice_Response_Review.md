# Compass Voice — Customer-Facing Response Audit & Review

**Audit date:** 2026-05-06
**Reviewer:** Aroosh Ahmad (engineering) → Steve / Ali / team
**Scope:** Every string the IVR speaks back to a caller on a US restaurant phone-ordering call.
**Method:** Static audit of `app/responses/`, `app/core/response_builder.py`, `app/state_machine/handlers/`, `app/api/twilio_server.py`, `app/api/voice_stream_server.py`, `app/state_machine/handlers/payment/payment_flow_support.py`. No code was modified.
**Review goal:** Forward this document to non-engineers. Each row has a *Reviewer Comment* column to capture feedback, an *Approved/Rejected* status, and a suggested rewrite. Section 5 (Review Sheet) is CSV-paste-ready for Excel / Google Sheets / Notion.

---

## 1. Executive Diagnosis

**Where copy lives.** Customer-facing copy is **mostly centralized** in two layers:

1. `app/responses/*` — pure renderer functions (item, sides, modifiers, sizes, quantity, cart, menu, flow control, intent-not-allowed, side-size).
2. `app/core/response_builder.py` — a single `_build_registry()` dict mapping ~85 `response_key` values to either lambdas (with inline literals) or imported renderers.

This is architecturally clean: handlers return `response_key`, never raw text. **However**, response_builder.py itself contains ~50+ inline string literals inside lambdas — those should arguably move into `app/responses/` for full separation. (Architectural note in §6.)

**A few strays exist outside the response layer:**

- `app/api/twilio_server.py` — the *very first* greeting (`Welcome to Compass. Is this for pickup or delivery?`) and the no-speech fallback (`Sorry, I didn't catch that. Could you repeat?`) are hardcoded into the Twilio voice route. **Architectural violation** — copy in transport.
- `app/api/voice_stream_server.py` — the human-agent transfer line (`Okay. Connecting you to a team member now. One moment please.`) duplicates `transferring_to_human_agent` from the registry. **Drift risk.**
- `app/state_machine/handlers/payment/payment_flow_support.py` — SMS body fallback `View full order details in the checkout link.` Lives in a handler, but it's an SMS string, not a TTS line — keep an eye on it.

**Top quality issues found:**

1. **Robotic / terse phrasings** — `"Pickup. What would you like to order?"`, `"Cancel {item}?"`, `"Need 2 more sides. {options}."`, `"Up to 3. {options}."` These read as machine output, not a server taking an order.
2. **Confusing "yes or no" tags** appended to ambiguous questions — `"Did you mean cheeseburger? Yes or no."` sounds like a quiz; native phrasing is just `"Did you mean a cheeseburger?"`
3. **Item Not Found does not always acknowledge** what the caller asked for, and the suggestion list can echo unrelated categories (e.g., suggesting `American cheese` when the caller asked for *hamburger*).
4. **Repetitive "What would you like next/else?" tail** on nearly every confirmation creates a pushy, robotic cadence.
5. **Order completion is overly prescriptive** — `"Will be ready in 25 minutes."` — hardcoded promise that may not match reality. **P0**.
6. **Two separate fallbacks for "I didn't understand"** (`"Sorry, I didn't understand that."` in registry default, `"I didn't catch that. Please say it again."` in `intent_not_allowed`) — inconsistent voice.
7. **Multi-modifier limit messaging is dense** (`"That is too many extras. You can choose up to 3. Please pick again from..."`) — too many clauses for a phone call.
8. **"Say done when you're ready"** appears in modifier and side flows — implies caller knows they need to issue the literal command "done". Not natural.
9. **"You can say none"** is a leak of the system's vocabulary — slightly better than nothing but still off.
10. **Mojibake risk** — multiple files use curly quotes (`I'll`, `don't`, `I didn't`). Most TTS engines handle them, but they will break exact-match unit tests and any keyword logging. **P2 polish.**

**Verdict.** The system speaks in short, technically-correct sentences but lacks the warmth and natural cadence of a human order-taker. Most copy is **understandable but not delightful**; a focused pass on ~40 of the ~85 keys would dramatically lift perceived quality without any FSM rework.

---

## 2. Response Inventory By Category

> Examples below substitute the dynamic placeholders with these realistic items: **Hamburger, Cheeseburger, Patty Burger, Fries, Onion Rings, Diet Coke, Chicken Sandwich, Pizza**.

| # | Category | Flow / State | File Path | Function / Class | Current Response | Current Example | Issue | Suggested Response | Suggested Example | Priority | Reviewer Comment |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Greeting / Call Start | First TwiML voice() | `app/api/twilio_server.py` | `voice()` (line 76) | `Welcome to Compass. Is this for pickup or delivery?` | Welcome to Compass. Is this for pickup or delivery? | Asks order type before brand reassurance; double-asks on landline path because device-type prompt also greets. Also lives in transport layer (architectural violation). | `Thanks for calling Compass! Is this order for pickup or delivery?` | Thanks for calling Compass! Is this order for pickup or delivery? | P1 | |
| 2 | Greeting / Call Start | WAITING_FOR_CALLER_DEVICE_TYPE | `app/core/response_builder.py` | registry `ask_for_caller_device_type` (line 233) | `Welcome to Compass. Are you calling from a landline or a mobile phone?` | Welcome to Compass. Are you calling from a landline or a mobile phone? | "Calling from a landline or mobile phone" is awkward; callers don't think in those terms. | `Hi, thanks for calling Compass. Quick question — are you on a cell phone or a home phone?` | Hi, thanks for calling Compass. Quick question — are you on a cell phone or a home phone? | P1 | |
| 3 | Fallback / Unknown Intent | Empty STT | `app/api/twilio_server.py` | `process_speech()` (line 104) | `Sorry, I didn't catch that. Could you repeat?` | Sorry, I didn't catch that. Could you repeat? | Fine. Slightly robotic. Inconsistent with the other "I didn't catch that" copy in `intent_not_allowed`. | `Sorry, I didn't quite catch that — could you say it again?` | Sorry, I didn't quite catch that — could you say it again? | P2 | |
| 4 | Fallback / Unknown Intent | Default missing renderer | `app/core/response_builder.py` | `build()` (line 90) | `Sorry, I didn't understand that.` | Sorry, I didn't understand that. | Generic. Caller has no path forward. | `I didn't catch that. You can add an item, hear the menu, or say checkout.` | I didn't catch that. You can add an item, hear the menu, or say checkout. | P0 | |
| 5 | Fallback / Unknown Intent | intent_not_allowed default | `app/responses/intent_not_allowed.py` | `handle_intent_not_allowed()` (line 33) | `I didn't catch that. Please say it again.` | I didn't catch that. Please say it again. | Same intent as #4 above — duplicate fallback voice. | Unify with #4. | I didn't catch that. You can add an item, hear the menu, or say checkout. | P0 | |
| 6 | Fallback / Unknown Intent | handler_not_implemented | `app/core/response_builder.py` | registry (line 140) | `That feature isn't ready yet.` | That feature isn't ready yet. | Reveals system internals to the caller; sounds broken. | `I'm not able to do that right now. Anything else I can help with?` | I'm not able to do that right now. Anything else I can help with? | P0 | |
| 7 | Error Recovery | confirmation_state_error | `app/core/response_builder.py` | registry (line 141) | `Something went wrong. Start again.` | Something went wrong. Start again. | Tells the customer to "start again" with no context — abrupt. | `Sorry, something went off track. Let me know what you'd like to add and we'll keep going.` | Sorry, something went off track. Let me know what you'd like to add and we'll keep going. | P0 | |
| 8 | Error Recovery | item_context_missing | `app/responses/item/success.py` | `item_context_missing()` | `Something went wrong with that item. Let's start again.` | Something went wrong with that item. Let's start again. | Same issue — no specifics, asks caller to restart silently. | `I lost track of that item. Could you tell me which one you wanted again?` | I lost track of that item. Could you tell me which one you wanted again? | P0 | |
| 9 | Error Recovery | readonly_interrupt fallback | `app/core/response_builder.py` | `_readonly_interrupt_with_resume()` (line 498) | `Sorry, I couldn't process that.` | Sorry, I couldn't process that. | Vague. | `Sorry, I missed that one — let's keep going.` | Sorry, I missed that one — let's keep going. | P1 | |
| 10 | Greeting → Pickup/Delivery | WAITING_FOR_LANDLINE_PICKUP_CONFIRMATION | `app/core/response_builder.py` | `confirm_landline_pickup_only` (line 239) | `I'll connect you with a team member to place your order. Would you like to proceed?` | I'll connect you with a team member to place your order. Would you like to proceed? | "Place your order" is fine, but caller doesn't know *why* they need a human — should explain. | `For landline orders, I'll hand you off to one of our team members. Want me to connect you?` | For landline orders, I'll hand you off to one of our team members. Want me to connect you? | P1 | |
| 11 | Greeting → Pickup/Delivery | repeat_landline_pickup_only | `app/core/response_builder.py` | registry (line 242) | `Would you like to connect with a team member? Yes or no.` | Would you like to connect with a team member? Yes or no. | Tail "Yes or no" sounds like a quiz prompt. | `Should I connect you with a team member?` | Should I connect you with a team member? | P1 | |
| 12 | End Call | landline_pickup_declined | `app/core/response_builder.py` | registry (line 248) | `No problem. Call us anytime. Goodbye.` | No problem. Call us anytime. Goodbye. | Acceptable; "Goodbye" is stiff for US phone-ordering. | `No problem — give us a call anytime. Take care!` | No problem — give us a call anytime. Take care! | P2 | |
| 13 | End Call | transferring_to_human_agent | `app/core/response_builder.py` | registry (line 245) | `Connecting you now. One moment.` | Connecting you now. One moment. | Fine, but duplicate copy lives in `voice_stream_server.py`. | `One moment — connecting you now.` | One moment — connecting you now. | P2 | |
| 14 | End Call | TwiML stream-ended transfer | `app/api/voice_stream_server.py` | line 482 | `Okay. Connecting you to a team member now. One moment please.` | Okay. Connecting you to a team member now. One moment please. | Duplicate of registry `transferring_to_human_agent`; drift risk. | Pull from registry. Same copy as #13. | One moment — connecting you now. | P1 | |
| 15 | Order Type | ask_for_order_type | `app/core/response_builder.py` | registry (line 252) | `Is this for pickup or delivery?` | Is this for pickup or delivery? | OK, but appears 3+ times verbatim across paths — consider variant copy. | `Is this for pickup or delivery?` (keep) | Is this for pickup or delivery? | P2 | |
| 16 | Order Type | repeat_order_type | `app/core/response_builder.py` | registry (line 253) | `Is this for pickup or delivery?` | Is this for pickup or delivery? | **Identical to #15** — caller hears the exact same line on retry. Feels like a loop. | `Sorry — pickup or delivery?` | Sorry — pickup or delivery? | P0 | |
| 17 | Order Type | order_type_captured_pickup | `app/core/response_builder.py` | registry (line 254) | `Pickup. What would you like to order?` | Pickup. What would you like to order? | Single-word ack ("Pickup.") sounds robotic. | `Got it — pickup. What can I get started for you?` | Got it — pickup. What can I get started for you? | P1 | |
| 18 | Order Type | order_type_captured_delivery | `app/core/response_builder.py` | registry (line 255) | `Delivery. What would you like to order?` | Delivery. What would you like to order? | Same as #17. | `Got it — delivery. What can I get started for you?` | Got it — delivery. What can I get started for you? | P1 | |
| 19 | Order Type | ordering_blocked_need_order_type | `app/core/response_builder.py` | registry (line 258) | `I'll get to your order right away. First, is this for pickup or delivery?` | I'll get to your order right away. First, is this for pickup or delivery? | OK. Could be slightly warmer. | `Happy to grab that for you — first, is this pickup or delivery?` | Happy to grab that for you — first, is this pickup or delivery? | P2 | |
| 20 | Item Add Confirmation | _confirm_item | `app/core/response_builder.py` | `_confirm_item()` (line 444) | `{item_name}, right? Yes or no.` | Cheeseburger, right? Yes or no. | "Yes or no" tag is robotic; native phrasing drops it. | `Just to confirm — {item_name}?` | Just to confirm — Cheeseburger? | P0 | |
| 21 | Item Add Confirmation | item_added_successfully (single) | `app/responses/item/success.py` | `item_added_successfully()` | `{item_name} added. Would you like anything else?` | Cheeseburger added. Would you like anything else? | "Added" sounds like a database log entry; "anything else" is fine but always asked. | `Added one {item_name}. Anything else?` | Added one Cheeseburger. Anything else? | P1 | |
| 22 | Item Add Confirmation | item_added_successfully (qty>1) | `app/responses/item/success.py` | `item_added_successfully()` | `Added {quantity} {item_name}. Would you like anything else?` | Added 2 Cheeseburgers. Would you like anything else? | OK; pluralization handled by item_name string. Edge: "Added 2 Fries" reads weird if the name is already plural. | `Added {quantity} {item_name}. Anything else?` | Added 2 Cheeseburgers. Anything else? | P2 | |
| 23 | Item Add Confirmation | item_added_successfully (queue) | `app/responses/item/success.py` | `item_added_successfully()` | `{prev added}. {this added}. {N} more to go. Would you like anything else?` | Cheeseburger added. Fries added. 1 more to go. Would you like anything else? | "1 more to go" implies the caller is being timed; awkward. | `Got the {prev}. And the {this}. Still {N} more on your order — anything else first?` | Got the Cheeseburger. And the Fries. Still 1 more on your order — anything else first? | P1 | |
| 24 | Item Add Confirmation | item_added_successfully (queue end) | `app/responses/item/success.py` | `item_added_successfully()` | `{prev}. {this}. That's everything. Would you like anything else?` | Cheeseburger added. Fries added. That's everything. Would you like anything else? | "That's everything" then "would you like anything else" contradicts itself. | `{prev added}. And the {this}. Anything else, or should I read your order back?` | Cheeseburger added. And the Fries. Anything else, or should I read your order back? | P0 | |
| 25 | Item Add Confirmation | _confirm_order_summary tail | `app/core/response_builder.py` | `_confirm_order_summary()` (line 456) | `{summary} Would you like to checkout?` | …Your total is $14.50. Should I place the order Would you like to checkout? | **Bug**: `render_checkout_review_summary` ends with `"Should I place the order"` (no question mark), then this lambda appends "Would you like to checkout?" — caller hears two questions back-to-back, no punctuation between. | Drop one of the two prompts; keep `Should I place the order?` only OR `Want me to send the payment link?` only. | …Your total is $14.50. Should I send the payment link? | P0 | |
| 26 | Item Add Confirmation | render_checkout_review_summary | `app/responses/cart_responses.py` | `render_checkout_review_summary()` | `{intro}Please review your order: {items_text}. Your total is {total}. Should I place the order` | This is a delivery order. Please review your order: 1 Cheeseburger, with Fries, add Cheese. 1 Diet Coke. Your total is $14.50. Should I place the order | Sentence ends without `?`. Item-line construction reads strangely on voice — "1 Cheeseburger, with Fries, add Cheese" — "add" is jarring. | `That's a delivery order. Let me read it back: {items}. Your total is {total}. Should I send the payment link?` | That's a delivery order. Let me read it back: one Cheeseburger with Fries, plus extra Cheese. One Diet Coke. Your total is $14.50. Should I send the payment link? | P0 | |
| 27 | Item Disambiguation | confirm_item_ambiguous | `app/responses/item/confirmation.py` | `confirm_item_ambiguous()` | `Did you mean {options}?` | Did you mean Cheeseburger, Patty Burger, or Hamburger? | OK. Could be more helpful. | `I'm not sure which one — did you mean {options}?` | I'm not sure which one — did you mean the Cheeseburger, the Patty Burger, or the Hamburger? | P1 | |
| 28 | Item Disambiguation | confirm_item_ambiguous (no options) | `app/responses/item/confirmation.py` | `confirm_item_ambiguous()` | `I found a few matches. Which one did you mean?` | I found a few matches. Which one did you mean? | If we don't have options to read, why are we calling this disambiguation handler? Empty-options branch is a smell. | `I caught a few options — could you say the item again?` | I caught a few options — could you say the item again? | P2 | |
| 29 | Item Disambiguation | confirm_item_from_category | `app/responses/item/confirmation.py` | `confirm_item_from_category()` | `In {category_name}, I found {options}. Which one would you like?` | In Burgers, I found Cheeseburger, Patty Burger, or Hamburger. Which one would you like? | Reads slightly stilted. "I found … or" is odd; should use "and". | `Under {category}, we have {options}. Which one would you like?` | Under Burgers, we have the Cheeseburger, the Patty Burger, and the Hamburger. Which one? | P1 | |
| 30 | Item Disambiguation | menu_ambiguity_response | `app/responses/menu_responses.py` | `menu_ambiguity_response()` | `I found {options}. Which one did you mean?` | I found Cheeseburger, Patty Burger, or Hamburger. Which one did you mean? | Same "found … or" smell as #29. | `Couple of matches — was that the {opt1}, the {opt2}, or the {opt3}?` | Couple of matches — was that the Cheeseburger, the Patty Burger, or the Hamburger? | P1 | |
| 31 | Item Not Found | item_not_found (with query, with suggestions) | `app/responses/item/not_found.py` | `item_not_found()` | `I don't see {query} on the menu, but we have {options}. Which one would you like?` | I don't see hamburger on the menu, but we have Cheeseburger or Patty Burger. Which one would you like? | This is **good**. Acknowledges + offers + asks one question. | Keep. | I don't see hamburger on the menu, but we have Cheeseburger or Patty Burger. Which one would you like? | P2 | |
| 32 | Item Not Found | item_not_found (no query) | `app/responses/item/not_found.py` | `item_not_found()` | `I don't see that on the menu. What else can I get you?` | I don't see that on the menu. What else can I get you? | OK, but lost the original word — should encourage caller to repeat. | `I don't see that one. What else were you in the mood for?` | I don't see that one. What else were you in the mood for? | P2 | |
| 33 | Item Not Found | item_not_found (customizable suggestion) | `app/responses/item/not_found.py` | `item_not_found()` | `{unavail} {customizable_item} can be customized to your liking. Want that?` | I don't see hamburger on the menu. Build Your Own Burger can be customized to your liking. Want that? | "Customized to your liking" is corporate. "Want that?" is curt. | `{unavail} We do have a {customizable} you can build however you'd like — want me to set one up?` | I don't see hamburger on the menu. We do have a Build Your Own Burger you can build however you'd like — want me to set one up? | P1 | |
| 34 | Item Not Found | item_not_found_near_miss | `app/responses/item/not_found.py` | `item_not_found_near_miss()` | `Did you mean {item_name}?` | Did you mean Cheeseburger? | Fine. Doesn't acknowledge what the caller said. | `Did you mean a {item_name}?` | Did you mean a Cheeseburger? | P2 | |
| 35 | Item Not Found | item_not_found_escalation | `app/responses/item/not_found.py` | `item_not_found_escalation()` | `I don't have that item. You can choose another item, or I can connect you to the restaurant.` | I don't have that item. You can choose another item, or I can connect you to the restaurant. | "Connect you to the restaurant" — caller is *already* calling the restaurant. Confusing. | `I'm having trouble finding that one. Want to try a different item, or I can hand you off to a team member?` | I'm having trouble finding that one. Want to try a different item, or I can hand you off to a team member? | P0 | |
| 36 | Item Not Found | item_clarification_limit_reached | `app/responses/item/not_found.py` | `item_clarification_limit_reached()` | `I'm having trouble finding that item. What else can I get for you?` | I'm having trouble finding that item. What else can I get for you? | Acceptable. | Keep. | I'm having trouble finding that item. What else can I get for you? | P2 | |
| 37 | Item Not Found | repeat_item_request | `app/responses/item/not_found.py` | `repeat_item_request()` | `Which item would you like?` | Which item would you like? | Cold restart, no acknowledgment. | `What can I get for you?` | What can I get for you? | P1 | |
| 38 | Item Not Found | menu_not_found_response | `app/responses/menu_responses.py` | `menu_not_found_response()` | `I could not find that on the menu.` | I could not find that on the menu. | "Could not" is formal; missing follow-up. | `I don't see that on the menu. Want me to suggest something close?` | I don't see that on the menu. Want me to suggest something close? | P1 | |
| 39 | Item Not Found | price_not_found | `app/core/response_builder.py` | registry (line 208) | `I couldn't find that item. Say the item name again.` | I couldn't find that item. Say the item name again. | Imperative "Say…" sounds robotic. | `I couldn't find that one — could you say the name again?` | I couldn't find that one — could you say the name again? | P1 | |
| 40 | Modifier Question | ask_for_modifier (with examples) | `app/responses/item/modifiers.py` | `ask_for_modifier()` | `Any {noun} {verb}, like {examples}? You can say none.` | Any toppings would you like, like Cheese, Bacon, or Lettuce? You can say none. | "Any toppings would you like" is **grammatically broken** when prompt_verb is "would you like". | `Any {noun}? We've got {examples} — or just say no.` | Any toppings? We've got Cheese, Bacon, or Lettuce — or just say no. | P0 | |
| 41 | Modifier Question | ask_for_modifier (no examples) | `app/responses/item/modifiers.py` | `ask_for_modifier()` | `Any {noun} {verb} on your {item_name}? You can say none.` | Any toppings would you like on your Cheeseburger? You can say none. | Same grammar bug as #40. | `Any toppings on your {item_name}? You can skip if you don't want any.` | Any toppings on your Cheeseburger? You can skip if you don't want any. | P0 | |
| 42 | Modifier Question | repeat_modifier_options (concise) | `app/responses/item/modifiers.py` | `repeat_modifier_options()` | `Which option?` | Which option? | Two words; sounds curt and robotic, especially after "got cheese." | `Which one would you like?` | Which one would you like? | P1 | |
| 43 | Modifier Question | repeat_modifier_options (list hint) | `app/responses/item/modifiers.py` | `repeat_modifier_options()` | `I didn't catch that. Say 'list options' to hear all choices.` | I didn't catch that. Say 'list options' to hear all choices. | Telling caller to use a magic phrase ("list options") is unnatural. | `I didn't catch that. Want me to read the full list?` | I didn't catch that. Want me to read the full list? | P0 | |
| 44 | Modifier Question | repeat_modifier_options (fallback) | `app/responses/item/modifiers.py` | `repeat_modifier_options()` | `Please choose one of the available options.` | Please choose one of the available options. | "Available options" is corporate. | `Just pick whichever one you'd like.` | Just pick whichever one you'd like. | P1 | |
| 45 | Modifier Question | list_modifier_options (limit>1) | `app/responses/item/modifiers.py` | `list_modifier_options()` | `Up to {max_selector}. {options}.` | Up to 3. Cheese, Bacon, or Lettuce. | "Up to 3." reads like a math expression. | `You can pick up to {max}. We've got {options}.` | You can pick up to 3. We've got Cheese, Bacon, or Lettuce. | P0 | |
| 46 | Modifier Question | list_modifier_options (limit=1) | `app/responses/item/modifiers.py` | `list_modifier_options()` | `Your options are {options}.` | Your options are Cheese, Bacon, or Lettuce. | Acceptable. Slightly stiff. | `Here are your choices: {options}.` | Here are your choices: Cheese, Bacon, or Lettuce. | P2 | |
| 47 | Modifier Question | list_modifier_options (no options) | `app/responses/item/modifiers.py` | `list_modifier_options()` | `Let me list the options.` | Let me list the options. | Useless without options. Dead-end response. | `I don't have a list to read here — what would you like?` | I don't have a list to read here — what would you like? | P1 | |
| 48 | Modifier Question | clarify_modifier_choice | `app/responses/item/modifiers.py` | `clarify_modifier_choice()` | `Did you mean {options}?` | Did you mean Cheese, Bacon, or Lettuce? | Acceptable; matches item disambiguation. | `Did you mean {options}?` (keep) | Did you mean Cheese, Bacon, or Lettuce? | P2 | |
| 49 | Modifier Question | required_modifier_cannot_skip | `app/responses/item/modifiers.py` | `required_modifier_cannot_skip()` | `Need {N} more options. {options}.` | Need 2 more options. Cheese, Bacon, or Lettuce. | Telegraphic. "Need 2 more options" sounds like a system error. | `You'll need to pick {N} more — choose from {options}.` | You'll need to pick 2 more — choose from Cheese, Bacon, or Lettuce. | P0 | |
| 50 | Modifier Question | required_modifier_cannot_skip (1 needed) | `app/responses/item/modifiers.py` | `required_modifier_cannot_skip()` | `An option is required. {options}.` | An option is required. Cheese, Bacon, or Lettuce. | "An option is required" is corporate / form-validation style. | `Pick one to continue — {options}.` | Pick one to continue — Cheese, Bacon, or Lettuce. | P0 | |
| 51 | Modifier Question | too_many_modifier_choices (with limit) | `app/responses/item/modifiers.py` | `too_many_modifier_choices()` | `That is too many extras. You can choose up to {max}. Please pick again from {options}.` | That is too many extras. You can choose up to 3. Please pick again from Cheese, Bacon, or Lettuce. | Three sentences = too dense for voice. | `Only {max} extras allowed — which {max} from {options}?` | Only 3 extras allowed — which 3 from Cheese, Bacon, or Lettuce? | P0 | |
| 52 | Modifier Question | too_many_modifier_choices (limit=1) | `app/responses/item/modifiers.py` | `too_many_modifier_choices()` | `That is too many extras. Please choose from {options}.` | That is too many extras. Please choose from Cheese, Bacon, or Lettuce. | "Too many extras" — no, the caller named *one* too many. | `Just one of those — {options}?` | Just one of those — Cheese, Bacon, or Lettuce? | P0 | |
| 53 | Modifier Question | too_many_modifier_choices (mixed accepted/dropped) | `app/responses/item/modifiers.py` | `too_many_modifier_choices()` | `I added {accepted}. You can only pick {max}, so I couldn't add {dropped}. I couldn't find {unmatched}. Say done when you're ready.` | I added Cheese and Bacon. You can only pick 2, so I couldn't add Lettuce. I couldn't find pickle. Say done when you're ready. | Four-clause sentence is too long for voice. "Say done when you're ready" exposes system grammar. | `Got {accepted}. {dropped} put me over the limit and I couldn't find {unmatched}. Want anything else, or move on?` | Got Cheese and Bacon. Lettuce put me over the limit and I couldn't find pickle. Want anything else, or should we move on? | P0 | |
| 54 | Modifier Question | confirm_modifier_choice_guess | `app/core/response_builder.py` | registry (line 229) | `Did you mean {choice_name}? Yes or no.` | Did you mean Cheese? Yes or no. | "Yes or no" tag is unnatural. | `Did you mean Cheese?` | Did you mean Cheese? | P0 | |
| 55 | Side Question | ask_for_side (with examples) | `app/responses/item/sides.py` | `ask_for_side()` | `Any {noun} {verb}, like {examples}? You can say none.` | Any side would you like, like Fries, Onion Rings, or Salad? You can say none. | Same grammar bug as #40. | `Want a side? We've got {examples} — or no thanks.` | Want a side? We've got Fries, Onion Rings, or Salad — or no thanks. | P0 | |
| 56 | Side Question | ask_for_side (no examples) | `app/responses/item/sides.py` | `ask_for_side()` | `Any {noun} {verb} with your {item_name}? You can say none.` | Any side would you like with your Cheeseburger? You can say none. | Same as #55. | `Any sides with your {item_name}?` | Any sides with your Cheeseburger? | P0 | |
| 57 | Side Question | repeat_side_options (no options) | `app/responses/item/sides.py` | `repeat_side_options()` | `Please choose one of the available sides.` | Please choose one of the available sides. | Stiff. | `Just pick a side and we'll roll.` | Just pick a side and we'll roll. | P1 | |
| 58 | Side Question | required_side_cannot_skip (>1) | `app/responses/item/sides.py` | `required_side_cannot_skip()` | `Need {N} more sides. {options}.` | Need 2 more sides. Fries, Onion Rings, or Salad. | Telegraphic. | `Two more sides to pick — choose from {options}.` | Two more sides to pick — choose from Fries, Onion Rings, or Salad. | P0 | |
| 59 | Side Question | required_side_cannot_skip (=1) | `app/responses/item/sides.py` | `required_side_cannot_skip()` | `A side is required. {options}.` | A side is required. Fries, Onion Rings, or Salad. | "Required" is corporate. | `Pick one side to continue — {options}.` | Pick one side to continue — Fries, Onion Rings, or Salad. | P0 | |
| 60 | Side Question | list_side_options (no options) | `app/responses/item/sides.py` | `list_side_options()` | `Let me list the side options.` | Let me list the side options. | Dead-end. | `I don't have any sides for that one — anything else?` | I don't have any sides for that one — anything else? | P1 | |
| 61 | Side Question | clarify_side_choice | `app/responses/item/sides.py` | `clarify_side_choice()` | `Did you mean {options}?` | Did you mean Fries, Onion Rings, or Salad? | OK. | Keep. | Did you mean Fries, Onion Rings, or Salad? | P2 | |
| 62 | Side Question | clarify_side_choice (no options) | `app/responses/item/sides.py` | `clarify_side_choice()` | `Which side did you want?` | Which side did you want? | OK. | Keep. | Which side did you want? | P2 | |
| 63 | Side Question | too_many_side_choices (limit=1) | `app/responses/item/sides.py` | `too_many_side_choices()` | `That is too many sides. Please choose from {options}.` | That is too many sides. Please choose from Fries, Onion Rings, or Salad. | Same as #52 for modifiers. | `Just one side — {options}?` | Just one side — Fries, Onion Rings, or Salad? | P0 | |
| 64 | Side Question | too_many_side_choices (limit>1) | `app/responses/item/sides.py` | `too_many_side_choices()` | `That is too many sides. You can choose up to {max}. Please pick again from {options}.` | That is too many sides. You can choose up to 2. Please pick again from Fries or Onion Rings. | Same as #51. | `Only {max} sides allowed — which {max} from {options}?` | Only 2 sides allowed — which 2 from Fries or Onion Rings? | P0 | |
| 65 | Side Question | confirm_side_choice_guess | `app/core/response_builder.py` | registry (line 228) | `Did you mean {choice_name}? Yes or no.` | Did you mean Fries? Yes or no. | "Yes or no" tag is unnatural. | `Did you mean Fries?` | Did you mean Fries? | P0 | |
| 66 | Size Question | ask_for_size | `app/responses/item/sizes.py` | `ask_for_size()` | `What size would you like for {item_name}?` | What size would you like for Pizza? | OK. | Keep. | What size would you like for Pizza? | P2 | |
| 67 | Size Question | repeat_size_options (full) | `app/responses/item/sizes.py` | `repeat_size_options()` | `Available sizes for {item_name} are {options}.` | Available sizes for Pizza are Small, Medium, or Large. | "Available sizes are" is corporate. | `For Pizza we have {options} — which one?` | For Pizza we have Small, Medium, or Large — which one? | P1 | |
| 68 | Size Question | repeat_size_options (concise) | `app/responses/item/sizes.py` | `repeat_size_options()` | `What size for {item_name}?` | What size for Pizza? | Acceptable. | Keep. | What size for Pizza? | P2 | |
| 69 | Size Question | repeat_size_options (list hint) | `app/responses/item/sizes.py` | `repeat_size_options()` | `I didn't catch that. Say 'list options' to hear all sizes for {item_name}.` | I didn't catch that. Say 'list options' to hear all sizes for Pizza. | Magic-phrase leak. | `I didn't catch that. Want me to read out the sizes?` | I didn't catch that. Want me to read out the sizes? | P0 | |
| 70 | Size Question | required_size_cannot_skip | `app/responses/item/sizes.py` | `required_size_cannot_skip()` | `Please choose a size for {item_name}: {options}.` | Please choose a size for Pizza: Small, Medium, or Large. | Acceptable. | Keep. | Please choose a size for Pizza: Small, Medium, or Large. | P2 | |
| 71 | Size Question | invalid_size_option | `app/responses/item/sizes.py` | `invalid_size_option()` | `That size is not available for {item_name}. Please choose {options}.` | That size is not available for Pizza. Please choose Small, Medium, or Large. | "Not available" is corporate. | `We don't have that size for {item_name} — try {options}.` | We don't have that size for Pizza — try Small, Medium, or Large. | P1 | |
| 72 | Size Question | invalid_size_option (escalation) | `app/responses/item/sizes.py` | `invalid_size_option()` | `Let's make this easy. Say {options} for {item_name}.` | Let's make this easy. Say Small, Medium, or Large for Pizza. | "Let's make this easy" sounds condescending after a few attempts. | `Let me list it out: for {item_name} it's {options}.` | Let me list it out: for Pizza it's Small, Medium, or Large. | P1 | |
| 73 | Size Question | size_not_applicable | `app/responses/item/success.py` | `size_not_applicable()` | `{item_name} does not need a size. Let's continue.` | Cheeseburger does not need a size. Let's continue. | OK. Slightly mechanical. | `{item_name} only comes one way — moving on.` | Cheeseburger only comes one way — moving on. | P2 | |
| 74 | Size Question | confirm_size_choice_guess | `app/core/response_builder.py` | registry (line 230) | `Did you mean {choice_name}? Yes or no.` | Did you mean Large? Yes or no. | "Yes or no" tag. | `Did you mean Large?` | Did you mean Large? | P0 | |
| 75 | Side Size (size of a side) | ask_for_side_size | `app/responses/side_size_responses.py` | `ask_for_side_size()` | `Size for {side_item_name}? {sizes}.` | Size for Fries? Small, Medium, or Large. | Telegraphic. | `What size {side_item_name} — {sizes}?` | What size Fries — Small, Medium, or Large? | P1 | |
| 76 | Side Size | repeat_side_size_options | `app/responses/side_size_responses.py` | `repeat_side_size_options()` | `Choose {sizes}.` | Choose Small, Medium, or Large. | Curt. | `Pick one — {sizes}.` | Pick one — Small, Medium, or Large. | P1 | |
| 77 | Side Size | confirm_side_size_choice_guess | `app/core/response_builder.py` | registry (line 231) | `Did you mean {choice_name} for {side_item_name}? Yes or no.` | Did you mean Large for Fries? Yes or no. | "Yes or no" tag. | `Did you mean Large {side_item_name}?` | Did you mean Large Fries? | P0 | |
| 78 | Quantity Question | ask_item_quantity | `app/responses/item/quantity.py` | `ask_item_quantity()` | `How many {item_name} would you like?` | How many Cheeseburgers would you like? | OK, but **forces** a quantity question for every item. Default to 1 unless ambiguous. | Skip if FSM gate detects no ambiguity. Otherwise: `How many {item_name}?` | How many Cheeseburgers? | P0 | |
| 79 | Quantity Question | invalid_quantity_option | `app/responses/item/quantity.py` | `invalid_quantity_option()` | `Please give a valid quantity for {item_name}.` | Please give a valid quantity for Cheeseburger. | "Valid quantity" is corporate. | `How many {item_name} — like 1 or 2?` | How many Cheeseburgers — like 1 or 2? | P1 | |
| 80 | Quantity Question | invalid_quantity_option (escalation) | `app/responses/item/quantity.py` | `invalid_quantity_option()` | `Please say a number for {item_name}, like 1 or 2.` | Please say a number for Cheeseburger, like 1 or 2. | OK. | Keep. | Please say a number for Cheeseburger, like 1 or 2. | P2 | |
| 81 | Cart Summary | render_cart_summary (empty) | `app/responses/cart_responses.py` | `render_cart_summary()` | `Your cart is empty.` | Your cart is empty. | Acceptable. | Keep. | Your cart is empty. | P2 | |
| 82 | Cart Summary | render_cart_summary (1 item, total) | `app/responses/cart_responses.py` | `render_cart_summary()` | `You have {qty} {name}. Total {total}. Would you like to add more or check out?` | You have 1 Cheeseburger. Total $5.99. Would you like to add more or check out? | "Total $5.99" reads like a receipt label. "Check out" is two words inconsistent with "checkout" elsewhere. | `That's {qty} {name} — {total} so far. Add anything else, or ready to check out?` | That's 1 Cheeseburger — $5.99 so far. Add anything else, or ready to check out? | P1 | |
| 83 | Cart Summary | render_cart_summary (multi-item) | `app/responses/cart_responses.py` | `render_cart_summary()` | `You have {N} items. Total {total}. Would you like to add more or check out?` | You have 3 items. Total $14.50. Would you like to add more or check out? | Doesn't list items — caller can't verify. | Read items back if ≤3, otherwise summarize: `You've got {N} items adding up to {total}. Want to hear them, add more, or check out?` | You've got 3 items adding up to $14.50. Want to hear them, add more, or check out? | P0 | |
| 84 | Cart Summary | cart_empty | `app/core/response_builder.py` | registry (line 215) | `Your cart is empty.` | Your cart is empty. | Duplicate of #81. Consolidate. | Keep. | Your cart is empty. | P2 | |
| 85 | Cart Summary | show_total | `app/core/response_builder.py` | registry (line 214) | `Your total is {total}.` | Your total is $14.50. | Acceptable. | Keep. | Your total is $14.50. | P2 | |
| 86 | Cart Summary | idle_nothing_to_checkout | `app/core/response_builder.py` | registry (line 134) | `Your cart is empty. Add something first.` | Your cart is empty. Add something first. | "Add something first" is curt / parental. | `Your cart is empty — what would you like to start with?` | Your cart is empty — what would you like to start with? | P1 | |
| 87 | Cart Summary | resume_shopping | `app/core/response_builder.py` | registry (line 135) | `Okay. Add more, remove items, or check your cart.` | Okay. Add more, remove items, or check your cart. | Reads like a menu of commands. | `Okay — add anything else, or want to hear what you've got?` | Okay — add anything else, or want to hear what you've got? | P1 | |
| 88 | Cart Summary | confirm_clear_cart | `app/responses/cart_responses.py` | `confirm_clear_cart_response()` | `Should I clear the cart?` | Should I clear the cart? | OK. | Keep. | Should I clear the cart? | P2 | |
| 89 | Cart Summary | cart_cleared | `app/responses/cart_responses.py` | `cart_cleared_response()` | `Okay, your cart is cleared.` | Okay, your cart is cleared. | OK. | `Okay, cart cleared. What would you like to add?` | Okay, cart cleared. What would you like to add? | P1 | |
| 90 | Cart Summary | clear_cart_cancelled | `app/responses/cart_responses.py` | `clear_cart_cancelled_response()` | `Okay, I kept your cart.` | Okay, I kept your cart. | OK. Slightly stiff. | `Okay, kept it as is.` | Okay, kept it as is. | P2 | |
| 91 | Remove Item | confirm_remove_item | `app/core/response_builder.py` | registry (line 175) | `Remove {item_name}?` | Remove Cheeseburger? | Two-word prompt. Caller may not know they need to say yes/no. | `Want me to remove the {item_name}?` | Want me to remove the Cheeseburger? | P1 | |
| 92 | Remove Item | item_removed_successfully | `app/core/response_builder.py` | registry (line 187) | `Removed {item_name}. Anything else?` | Removed Cheeseburger. Anything else? | OK. | Keep. | Removed Cheeseburger. Anything else? | P2 | |
| 93 | Remove Item | item_removal_cancelled | `app/core/response_builder.py` | registry (line 190) | `Okay, keeping it.` | Okay, keeping it. | OK. | Keep. | Okay, keeping it. | P2 | |
| 94 | Replace Item | confirm_replace_item | `app/core/response_builder.py` | registry (line 181) | `Replace {item_name} with {replacement_item_name}?` | Replace Cheeseburger with Patty Burger? | OK. | Keep. | Replace Cheeseburger with Patty Burger? | P2 | |
| 95 | Replace Item | ask_replacement_item | `app/core/response_builder.py` | registry (line 184) | `What would you like instead of {item_name}?` | What would you like instead of Cheeseburger? | OK. | Keep. | What would you like instead of Cheeseburger? | P2 | |
| 96 | Replace Item | item_replacement_cancelled | `app/core/response_builder.py` | registry (line 191) | `Okay, no changes.` | Okay, no changes. | OK. | Keep. | Okay, no changes. | P2 | |
| 97 | Modify Item | confirm_modify_item | `app/core/response_builder.py` | registry (line 178) | `Update {item_name}? I'll swap it with the new version.` | Update Cheeseburger? I'll swap it with the new version. | "Swap with the new version" leaks system internals. Caller doesn't know there are versions. | `Got it — update your {item_name} with those changes?` | Got it — update your Cheeseburger with those changes? | P0 | |
| 98 | Modify Item | item_modification_cancelled | `app/core/response_builder.py` | registry (line 192) | `Okay, leaving it as is.` | Okay, leaving it as is. | OK. | Keep. | Okay, leaving it as is. | P2 | |
| 99 | Modify Item | action_cancelled | `app/core/response_builder.py` | registry (line 193) | `Okay, cancelled.` | Okay, cancelled. | OK but bare; should bridge to next step. | `Okay, cancelled — what's next?` | Okay, cancelled — what's next? | P1 | |
| 100 | Cancel current item flow | confirm_cancel_current_item | `app/responses/item/confirmation.py` | `confirm_cancel_current_item()` | `Cancel {item_name}?` | Cancel Cheeseburger? | Two-word prompt. | `Want to cancel the {item_name}?` | Want to cancel the Cheeseburger? | P1 | |
| 101 | Cancel current item flow | confirm_cancel_current_item_for_new_request | `app/responses/item/confirmation.py` | `confirm_cancel_current_item_for_new_request()` | `Still adding {item_name}. Cancel it and move on?` | Still adding Cheeseburger. Cancel it and move on? | "Cancel it and move on" is OK but blunt. | `We're still on the {item_name} — drop it and move on?` | We're still on the Cheeseburger — drop it and move on? | P1 | |
| 102 | Cancel current item flow | continue_current_item_after_cancel_denied | `app/responses/item/confirmation.py` | `continue_current_item_after_cancel_denied()` | `Okay, continuing. {options}.` | Okay, continuing. Cheese, Bacon, or Lettuce. | "Continuing" then a list — abrupt. | `Okay, sticking with it. Now — {options}?` | Okay, sticking with it. Now — Cheese, Bacon, or Lettuce? | P1 | |
| 103 | Cancel current item flow | item_cancelled_successfully | `app/responses/item/success.py` | `item_cancelled_successfully()` | `Okay, cancelled. What would you like next?` | Okay, cancelled. What would you like next? | OK. | Keep. | Okay, cancelled. What would you like next? | P2 | |
| 104 | Flow Guard | flow_guard_finish_current_step | `app/responses/flow_control_responses.py` | `flow_guard_finish_current_step()` | `Please finish {item_name}, or say cancel.` | Please finish Cheeseburger, or say cancel. | OK; "or say cancel" exposes vocab but is acceptable. | `Let's wrap up the {item_name} first — or say cancel.` | Let's wrap up the Cheeseburger first — or say cancel. | P1 | |
| 105 | Flow Guard | flow_guard_confirm_cancel | `app/responses/flow_control_responses.py` | `flow_guard_confirm_cancel()` | `Do you want to cancel {item_name}? Please say yes or no.` | Do you want to cancel Cheeseburger? Please say yes or no. | "Please say yes or no" is unnatural. | `Cancel the {item_name}?` | Cancel the Cheeseburger? | P0 | |
| 106 | Flow Guard | flow_guard_cancelled | `app/responses/flow_control_responses.py` | `flow_guard_cancelled()` | `Okay, cancelled. What would you like next?` | Okay, cancelled. What would you like next? | OK. | Keep. | Okay, cancelled. What would you like next? | P2 | |
| 107 | Intent Not Allowed | WAITING_FOR_CALLER_DEVICE_TYPE | `app/responses/intent_not_allowed.py` | `handle_intent_not_allowed()` | `Please say landline or mobile phone.` | Please say landline or mobile phone. | Same wording-issue as #2; consistency. | `Are you on a cell phone or a home phone?` | Are you on a cell phone or a home phone? | P1 | |
| 108 | Intent Not Allowed | WAITING_FOR_LANDLINE_PICKUP_CONFIRMATION | `app/responses/intent_not_allowed.py` | `handle_intent_not_allowed()` | `Pickup is available for landline callers only. Would you like to proceed?` | Pickup is available for landline callers only. Would you like to proceed? | "Available for landline callers only" is system-jargon. | `For landline callers, I'll connect you with a team member — want to proceed?` | For landline callers, I'll connect you with a team member — want to proceed? | P1 | |
| 109 | Intent Not Allowed | WAITING_FOR_ORDER_TYPE | `app/responses/intent_not_allowed.py` | `handle_intent_not_allowed()` | `Please say pickup or delivery.` | Please say pickup or delivery. | Curt. | `Pickup or delivery?` | Pickup or delivery? | P2 | |
| 110 | Intent Not Allowed | ADD_ITEM_FLOW + SHOW_CART | `app/responses/intent_not_allowed.py` | `handle_intent_not_allowed()` | `Finish this item first, or say cancel.` | Finish this item first, or say cancel. | OK. | Keep. | Finish this item first, or say cancel. | P2 | |
| 111 | Intent Not Allowed | ADD_ITEM_FLOW (default) | `app/responses/intent_not_allowed.py` | `handle_intent_not_allowed()` | `Please finish this item, or say cancel.` | Please finish this item, or say cancel. | Almost identical to #110 — consolidate. | Keep one. | Please finish this item, or say cancel. | P2 | |
| 112 | Intent Not Allowed | CONFIRMING_ORDER | `app/responses/intent_not_allowed.py` | `handle_intent_not_allowed()` | `Please confirm the order, or say cancel.` | Please confirm the order, or say cancel. | OK. | Keep. | Please confirm the order, or say cancel. | P2 | |
| 113 | Intent Not Allowed | WAITING_FOR_PAYMENT | `app/responses/intent_not_allowed.py` | `handle_intent_not_allowed()` | `Please complete payment, or say cancel.` | Please complete payment, or say cancel. | OK. | Keep. | Please complete payment, or say cancel. | P2 | |
| 114 | Checkout | checkout_blocked_finish_current_item | `app/core/response_builder.py` | registry (line 268) | `Please finish this item first, or say cancel.` | Please finish this item first, or say cancel. | Duplicate of #110/#111. Consolidate copy. | Keep. | Please finish this item first, or say cancel. | P2 | |
| 115 | Pickup | pickup_ask_sms_permission | `app/core/response_builder.py` | registry (line 330) | `Your order is in! Would you like me to text you a payment link, or would you prefer to pay when you arrive?` | Your order is in! Would you like me to text you a payment link, or would you prefer to pay when you arrive? | Slightly long but acceptable. | `Order's in! Want me to text you a payment link, or pay when you get here?` | Order's in! Want me to text you a payment link, or pay when you get here? | P2 | |
| 116 | Pickup | pickup_repeat_sms_permission | `app/core/response_builder.py` | registry (line 334) | `Would you like a payment link sent to your phone, or will you pay when you pick up?` | Would you like a payment link sent to your phone, or will you pay when you pick up? | OK. | Keep. | Would you like a payment link sent to your phone, or will you pay when you pick up? | P2 | |
| 117 | Pickup / End Call | pickup_sms_sent_end_call | `app/core/response_builder.py` | registry (line 337) | `Done! The payment link is on its way to your phone. See you soon!` | Done! The payment link is on its way to your phone. See you soon! | Good. | Keep. | Done! The payment link is on its way to your phone. See you soon! | P2 | |
| 118 | Pickup / End Call | pickup_no_sms_end_call | `app/core/response_builder.py` | registry (line 340) | `No problem! We'll see you when you get here. You can pay at the counter.` | No problem! We'll see you when you get here. You can pay at the counter. | Good. | Keep. | No problem! We'll see you when you get here. You can pay at the counter. | P2 | |
| 119 | Pickup / End Call | pickup_end_call | `app/core/response_builder.py` | registry (line 343) | `All set! We'll see you soon. You can pay when you arrive.` | All set! We'll see you soon. You can pay when you arrive. | Good. | Keep. | All set! We'll see you soon. You can pay when you arrive. | P2 | |
| 120 | Delivery | ask_for_delivery_area | `app/core/response_builder.py` | registry (line 277) | `Got it. Delivery. Please say your delivery area.` | Got it. Delivery. Please say your delivery area. | "Delivery area" is system jargon — caller thinks "neighborhood" or "zip". | `Got it — delivery. What neighborhood or area are we delivering to?` | Got it — delivery. What neighborhood or area are we delivering to? | P1 | |
| 121 | Delivery | repeat_delivery_area | `app/core/response_builder.py` | registry (line 278) | `Please say your delivery area.` | Please say your delivery area. | Same. | `Sorry — what neighborhood or area?` | Sorry — what neighborhood or area? | P1 | |
| 122 | Delivery | ask_for_delivery_zip | `app/core/response_builder.py` | registry (line 279) | `Now please say your ZIP code.` | Now please say your ZIP code. | "Now please" is awkward. | `And what's your ZIP code?` | And what's your ZIP code? | P2 | |
| 123 | Delivery | repeat_delivery_zip | `app/core/response_builder.py` | registry (line 280) | `Please say the ZIP code.` | Please say the ZIP code. | Imperative. | `Sorry — what was that ZIP?` | Sorry — what was that ZIP? | P2 | |
| 124 | Delivery | confirm_delivery_area_zip | `app/core/response_builder.py` | registry (line 281) | `Just to confirm, that is {area}, ZIP code {postal_code}. Is that correct?` | Just to confirm, that is Brooklyn, ZIP code 11201. Is that correct? | OK. | Keep. | Just to confirm — Brooklyn, ZIP 11201, right? | P2 | |
| 125 | Delivery | repeat_delivery_area_zip_confirmation | `app/core/response_builder.py` | registry (line 284) | `I have {area}, ZIP code {postal_code}. Is that correct?` | I have Brooklyn, ZIP code 11201. Is that correct? | OK. | Keep. | I have Brooklyn, ZIP 11201. Is that correct? | P2 | |
| 126 | Delivery | delivery_area_confirmed | `app/core/response_builder.py` | registry (line 287) | `Great, we deliver there. What would you like to order?` | Great, we deliver there. What would you like to order? | Good. | Keep. | Great, we deliver there. What would you like to order? | P2 | |
| 127 | Delivery | ask_delivery_address_method | `app/core/response_builder.py` | registry (line 289) | `I've sent you a checkout link. Fill in your address and pay there. I'll confirm once it goes through.` | I've sent you a checkout link. Fill in your address and pay there. I'll confirm once it goes through. | Three-clause sentence — long for voice. | `Sent you a checkout link — finish your address there and pay. I'll confirm once it's in.` | Sent you a checkout link — finish your address there and pay. I'll confirm once it's in. | P1 | |
| 128 | Delivery | waiting_for_checkout_completion | `app/core/response_builder.py` | registry (line 292) | `Still waiting on checkout. I'll confirm as soon as payment goes through.` | Still waiting on checkout. I'll confirm as soon as payment goes through. | OK. | Keep. | Still waiting on checkout. I'll confirm as soon as payment goes through. | P2 | |
| 129 | Delivery | confirm_delivery_house_number | `app/core/response_builder.py` | registry (line 296) | `I heard {house_number}. Is that correct?` | I heard 245. Is that correct? | OK. | Keep. | I heard 245. Is that correct? | P2 | |
| 130 | Delivery | confirm_delivery_street | `app/core/response_builder.py` | registry (line 299) | `I heard {street}. Is that correct?` | I heard Main Street. Is that correct? | OK. | Keep. | I heard Main Street. Is that correct? | P2 | |
| 131 | Delivery | confirm_delivery_secondary_address | `app/core/response_builder.py` | registry (line 303) | `I heard {secondary_address}. Is that correct?` | I heard Apartment 4B. Is that correct? | OK. | Keep. | I heard Apartment 4B. Is that correct? | P2 | |
| 132 | Delivery | ask_for_delivery_house_number | `app/core/response_builder.py` | registry (line 320) | `Please say your house number.` | Please say your house number. | Curt. | `What's your house number?` | What's your house number? | P2 | |
| 133 | Delivery | ask_for_delivery_street | `app/core/response_builder.py` | registry (line 322) | `Please say your street name or street number.` | Please say your street name or street number. | "Street name or street number" is confusing — caller doesn't know which is meant. | `What's the street name?` | What's the street name? | P1 | |
| 134 | Delivery | ask_for_delivery_secondary_address | `app/core/response_builder.py` | registry (line 324) | `Apartment or suite number? Say none if there isn't one.` | Apartment or suite number? Say none if there isn't one. | Good — explicitly handles the empty case. | Keep. | Apartment or suite number? Say none if there isn't one. | P2 | |
| 135 | Delivery | delivery_address_captured_resume_checkout | `app/core/response_builder.py` | registry (line 327) | `Got your address. Sending payment link now.` | Got your address. Sending payment link now. | Good. | Keep. | Got your address. Sending payment link now. | P2 | |
| 136 | Delivery | checkout_link_sent | `app/core/response_builder.py` | registry (line 311) | `Checkout link sent. Enter your address and pay there. I'll confirm once it goes through.` | Checkout link sent. Enter your address and pay there. I'll confirm once it goes through. | Slightly long. | `Checkout link sent — finish up there and I'll confirm as soon as it's paid.` | Checkout link sent — finish up there and I'll confirm as soon as it's paid. | P1 | |
| 137 | Delivery | checkout_link_unavailable_fallback_voice | `app/core/response_builder.py` | registry (line 314) | `I'll take your address here instead. What's your house number?` | I'll take your address here instead. What's your house number? | Good. | Keep. | I'll take your address here instead. What's your house number? | P2 | |
| 138 | Delivery | checkout_link_failed_fallback_voice | `app/core/response_builder.py` | registry (line 317) | `I'll take your address here instead. What's your house number?` | I'll take your address here instead. What's your house number? | **Duplicate** of #137. Identical text under different key. | Same copy intentionally; consider consolidation. | I'll take your address here instead. What's your house number? | P2 | |
| 139 | Delivery | ordering_blocked_need_delivery_info (zip) | `app/core/response_builder.py` | `_ordering_blocked_delivery_info()` (line 409) | `I'll get your order started right after. What's your ZIP code?` | I'll get your order started right after. What's your ZIP code? | Good. | Keep. | I'll get your order started right after. What's your ZIP code? | P2 | |
| 140 | Delivery | ordering_blocked_need_delivery_info (eligibility) | `app/core/response_builder.py` | `_ordering_blocked_delivery_info()` (line 411) | `Almost there. Please confirm your delivery area first.` | Almost there. Please confirm your delivery area first. | Good. | Keep. | Almost there. Please confirm your delivery area first. | P2 | |
| 141 | Delivery | ordering_blocked_need_delivery_info (default) | `app/core/response_builder.py` | `_ordering_blocked_delivery_info()` (line 412) | `I'll get your order started right after. What's your delivery area?` | I'll get your order started right after. What's your delivery area? | "Delivery area" jargon. | `I'll get your order started right after — what neighborhood are we delivering to?` | I'll get your order started right after — what neighborhood are we delivering to? | P1 | |
| 142 | Delivery | ordering_blocked_need_delivery_address (street) | `app/core/response_builder.py` | `_ordering_blocked_delivery_address()` (line 418) | `I'll get your order started right after. What's your street name?` | I'll get your order started right after. What's your street name? | Good. | Keep. | I'll get your order started right after. What's your street name? | P2 | |
| 143 | Delivery | ordering_blocked_need_delivery_address (secondary) | `app/core/response_builder.py` | `_ordering_blocked_delivery_address()` (line 420) | `Almost done with your address. Apartment or suite number?` | Almost done with your address. Apartment or suite number? | Good. | Keep. | Almost done with your address. Apartment or suite number? | P2 | |
| 144 | Delivery | ordering_blocked_need_delivery_address (confirmation) | `app/core/response_builder.py` | `_ordering_blocked_delivery_address()` (line 423) | `Let me finish confirming your address first. Is that correct?` | Let me finish confirming your address first. Is that correct? | "Is that correct?" without context — confusing because the prior fragment doesn't echo what we have. | `Just need to confirm your address first — does {echoed} sound right?` | Just need to confirm your address first — does 245 Main Street sound right? | P1 | |
| 145 | Delivery | ordering_blocked_need_delivery_address (default) | `app/core/response_builder.py` | `_ordering_blocked_delivery_address()` (line 424) | `I'll get your order started right after. What's your house number?` | I'll get your order started right after. What's your house number? | Good. | Keep. | I'll get your order started right after. What's your house number? | P2 | |
| 146 | Payment Link / SMS | waiting_for_payment | `app/core/response_builder.py` | registry (line 220) | `Waiting for payment. I'll confirm it as soon as it goes through.` | Waiting for payment. I'll confirm it as soon as it goes through. | Good. | Keep. | Waiting for payment. I'll confirm it as soon as it goes through. | P2 | |
| 147 | Payment Link / SMS | payment_link_sent | `app/core/response_builder.py` | registry (line 347) | `Payment link sent. I'll confirm once payment goes through.` | Payment link sent. I'll confirm once payment goes through. | Good. | Keep. | Payment link sent. I'll confirm once payment goes through. | P2 | |
| 148 | Payment Link / SMS | payment_link_send_failed | `app/core/response_builder.py` | registry (line 270) | `I couldn't send the payment link. Your order is saved. Please try again shortly.` | I couldn't send the payment link. Your order is saved. Please try again shortly. | Acceptable, but caller is left without a clear next step. | `Couldn't send the link — your order's saved. Want me to try again, or pay when you arrive?` | Couldn't send the link — your order's saved. Want me to try again, or pay when you arrive? | P0 | |
| 149 | Payment Link / SMS | checkout_link_send_failed | `app/core/response_builder.py` | registry (line 273) | `I couldn't send the checkout link. Your order is saved. Please try again shortly.` | I couldn't send the checkout link. Your order is saved. Please try again shortly. | Same as #148. | Same fix path. | Couldn't send the link — your order's saved. Want me to try again? | P0 | |
| 150 | Payment Link / SMS | payment_link_unavailable_now | `app/core/response_builder.py` | registry (line 307) | `I couldn't send the payment link. Your order is saved. Please try again shortly.` | I couldn't send the payment link. Your order is saved. Please try again shortly. | **Triplicate** of #148/#149. | Consolidate to one key. | Couldn't send the link — your order's saved. Want me to try again? | P1 | |
| 151 | Payment Link / SMS | payment_draft_saved_retry_later | `app/core/response_builder.py` | registry (line 350) | `Payment didn't go through. Your order is saved. Try again shortly.` | Payment didn't go through. Your order is saved. Try again shortly. | "Try again shortly" doesn't tell caller what to do. | `Payment didn't go through — your order's saved. Want to try again, or call us back?` | Payment didn't go through — your order's saved. Want to try again, or call us back? | P0 | |
| 152 | Payment Link / SMS | payment_not_started | `app/core/response_builder.py` | registry (line 226) | `Payment has not started. Say checkout when ready.` | Payment has not started. Say checkout when ready. | Magic-phrase leak ("say checkout"). | `We haven't started checkout yet — let me know when you're ready.` | We haven't started checkout yet — let me know when you're ready. | P1 | |
| 153 | Payment Link / SMS | payment_not_confirmed_yet | `app/core/response_builder.py` | registry (line 356) | `I haven't received confirmation yet. Please complete payment on the link. I'll confirm as soon as it goes through.` | I haven't received confirmation yet. Please complete payment on the link. I'll confirm as soon as it goes through. | OK but stiff. **Avoids the payment-loop bug** — good guard. | `I don't see the payment confirmed yet — give the link a moment, I'll let you know once it's in.` | I don't see the payment confirmed yet — give the link a moment, I'll let you know once it's in. | P1 | |
| 154 | Payment Link / SMS | payment_verification_error | `app/core/response_builder.py` | registry (line 361) | `Having trouble checking payment. Give it a moment, I'm still checking.` | Having trouble checking payment. Give it a moment, I'm still checking. | OK. Slightly redundant ("trouble checking" + "still checking"). | `Having trouble verifying — give me a moment.` | Having trouble verifying — give me a moment. | P1 | |
| 155 | Payment Link / SMS | no_active_payment | `app/core/response_builder.py` | registry (line 225) | `There's no payment in progress.` | There's no payment in progress. | Cold. | `No payment is in progress right now.` | No payment is in progress right now. | P2 | |
| 156 | Payment Link / SMS | no_active_order_to_cancel | `app/core/response_builder.py` | registry (line 224) | `There's no active order to cancel.` | There's no active order to cancel. | OK. | Keep. | There's no active order to cancel. | P2 | |
| 157 | Payment Link / SMS | order_cancelled | `app/core/response_builder.py` | registry (line 227) | `Okay, checkout cancelled. Your cart is still here.` | Okay, checkout cancelled. Your cart is still here. | Good. | Keep. | Okay, checkout cancelled. Your cart is still here. | P2 | |
| 158 | Payment Link / SMS | SMS body fallback | `app/state_machine/handlers/payment/payment_flow_support.py` | `format_order_summary_sms()` (line 72) | `View full order details in the checkout link.` | View full order details in the checkout link. | OK for SMS. Not TTS. | Keep. | View full order details in the checkout link. | P2 | |
| 159 | Order Confirmation | order_completed | `app/core/response_builder.py` | `_order_completed()` (line 473) | `Payment confirmed.{order_sentence} Your order has been placed successfully. Will be ready in 25 minutes. Thank you!` | Payment confirmed. Your order number is 1 2 3 4. Your order has been placed successfully. Will be ready in 25 minutes. Thank you! | **Hardcoded "25 minutes"** — risky promise; varies by item, kitchen load, time of day. **P0**. Also "your order has been placed successfully" is corporate. | `Payment confirmed!{order_sentence} Your order's all set — we'll have it ready as soon as we can. Thanks for calling!` (Drop the hardcoded ETA; or pull from a per-restaurant config) | Payment confirmed! Your order number is 1 2 3 4. Your order's all set — we'll have it ready as soon as we can. Thanks for calling! | P0 | |
| 160 | Order Confirmation | order number digit-spell | `app/core/response_builder.py` | `_spoken_order_number()` (line 478) | `Your order number is {digit-spaced}.` | Your order number is 1 2 3 4. | Reading "1 2 3 4" digit-by-digit is correct for IVR — keep. | Keep. | Your order number is 1 2 3 4. | P2 | |
| 161 | Show Menu | show_menu_categories (with categories) | `app/responses/menu_responses.py` | `show_menu_categories_response()` | `Our categories are {categories}. Which one would you like?` | Our categories are Burgers, Sandwiches, Sides, or Drinks. Which one would you like? | "Categories" is restaurant jargon — say "sections" or just "we have". | `We've got {categories}. Which sounds good?` | We've got Burgers, Sandwiches, Sides, or Drinks. Which sounds good? | P1 | |
| 162 | Show Menu | show_menu_categories (none) | `app/responses/menu_responses.py` | `show_menu_categories_response()` | `Which category would you like?` | Which category would you like? | Cold; "category" is jargon. | `What kind of food are you in the mood for?` | What kind of food are you in the mood for? | P1 | |
| 163 | Show Menu | show_category (with items) | `app/responses/menu_responses.py` | `show_category_response()` | `In {category}, we have {items}. What would you like?` | In Burgers, we have Cheeseburger, Patty Burger, or Hamburger. What would you like? | OK. | Keep. | In Burgers, we have Cheeseburger, Patty Burger, or Hamburger. What would you like? | P2 | |
| 164 | Show Menu | show_category (empty) | `app/responses/menu_responses.py` | `show_category_response()` | `There is nothing available in {category} right now.` | There is nothing available in Pizza right now. | "Nothing available" is corporate. | `We don't have any {category} on the menu today — anything else?` | We don't have any Pizza on the menu today — anything else? | P1 | |
| 165 | Show Menu | show_category (no items) | `app/responses/menu_responses.py` | `show_category_response()` | `What would you like from {category}?` | What would you like from Pizza? | OK. | Keep. | What would you like from Pizza? | P2 | |
| 166 | Show Menu | show_item_info (with description) | `app/responses/menu_responses.py` | `show_item_info_response()` | `{item_name}. {description}` | Cheeseburger. Quarter-pound beef patty with American cheese. | Period after item name reads as a hard stop. | `The {item_name} — {description}` | The Cheeseburger — quarter-pound beef patty with American cheese. | P1 | |
| 167 | Show Menu | show_item_info (no description) | `app/responses/menu_responses.py` | `show_item_info_response()` | `{item_name} is on the menu.` | Cheeseburger is on the menu. | Stiff; tells caller nothing useful. | `Yes, we have the {item_name}.` | Yes, we have the Cheeseburger. | P1 | |
| 168 | Show Menu | show_item_price (variant) | `app/responses/menu_responses.py` | `show_item_price_response()` | `{variant_label} {name} is {variant_price}.` | Large Pizza is $14.99. | OK. | Keep. | Large Pizza is $14.99. | P2 | |
| 169 | Show Menu | show_item_price (fixed) | `app/responses/menu_responses.py` | `show_item_price_response()` | `{name} is {price}.` | Cheeseburger is $5.99. | OK. | Keep. | Cheeseburger is $5.99. | P2 | |
| 170 | Show Menu | show_item_price (unit) | `app/responses/menu_responses.py` | `show_item_price_response()` | `{name} is {price} each.` | Wings are $1.50 each. | OK. | Keep. | Wings are $1.50 each. | P2 | |
| 171 | Show Menu | show_item_price (variant list) | `app/responses/menu_responses.py` | `show_item_price_response()` | `{name} comes in {variants}.` | Pizza comes in Small $9.99, Medium $11.99, or Large $14.99. | Reading prices inline is dense. | `{name} is {smallest_price}-{largest_price} depending on size.` | Pizza is $9.99 to $14.99 depending on size. | P1 | |
| 172 | Show Menu | show_item_price (fallback) | `app/responses/menu_responses.py` | `show_item_price_response()` | `Price information is not available right now.` | Price information is not available right now. | Corporate. | `I don't have the price on hand — sorry about that.` | I don't have the price on hand — sorry about that. | P1 | |
| 173 | Show Menu | show_item_availability (with variants) | `app/responses/menu_responses.py` | `show_item_availability_response()` | `Yes, {item} is available. {desc} It comes in {variants}.` | Yes, Pizza is available. Wood-fired pizza. It comes in Small, Medium, or Large. | OK. Three sentences = a lot. | `Yep, we've got {item} — comes in {variants}.` | Yep, we've got Pizza — comes in Small, Medium, or Large. | P1 | |
| 174 | Show Menu | show_item_availability (no variant) | `app/responses/menu_responses.py` | `show_item_availability_response()` | `Yes, {item} is available.` | Yes, Cheeseburger is available. | OK. | Keep. | Yes, Cheeseburger is available. | P2 | |
| 175 | Show Menu | show_modifier_availability (modifier) | `app/responses/menu_responses.py` | `show_modifier_availability_response()` | `Yes, {modifier} is available for {price}.` | Yes, Bacon is available for $1.50. | OK. | Keep. | Yes, Bacon is available for $1.50. | P2 | |
| 176 | Show Menu | show_modifier_availability (no price) | `app/responses/menu_responses.py` | `show_modifier_availability_response()` | `Yes, {name} is available.` | Yes, Bacon is available. | OK. | Keep. | Yes, Bacon is available. | P2 | |
| 177 | Show Menu | show_modifier_availability (default) | `app/responses/menu_responses.py` | `show_modifier_availability_response()` | `Yes, that option is available.` | Yes, that option is available. | "That option" — caller doesn't know what we matched. | Avoid this branch; if we matched, echo the name. | (skip) | P1 | |
| 178 | Show Menu | modifier_available_with_item_context | `app/responses/menu_responses.py` | `modifier_available_with_item_context_response()` | `{name} is an add-on. Tell me the item, and I'll check it.` | Bacon is an add-on. Tell me the item, and I'll check it. | OK. | Keep. | Bacon is an add-on. Tell me the item, and I'll check it. | P2 | |
| 179 | Show Menu | modifier_requires_item_context | `app/core/response_builder.py` | registry (line 209) | `That goes with a specific item. Which item would you like it on?` | That goes with a specific item. Which item would you like it on? | OK. | Keep. | That goes with a specific item. Which item would you like it on? | P2 | |
| 180 | Show Menu | show_modifier_price | `app/core/response_builder.py` | registry (line 205) | `{modifier} on {item} costs {price}.` | Bacon on Cheeseburger costs $1.50. | OK. | Keep. | Bacon on Cheeseburger costs $1.50. | P2 | |
| 181 | Multi-item Ack Prefix | _build_multi_item_ack_prefix | `app/core/response_builder.py` | `_build_multi_item_ack_prefix()` (line 376) | `Got it, {items_text}. Starting with the {current}.` | Got it, Cheeseburger and Fries. Starting with the Cheeseburger. | "Starting with the X" exposes queue mechanics. | `Got it — {items_text}. Let's start with the {current}.` | Got it — Cheeseburger and Fries. Let's start with the Cheeseburger. | P1 | |
| 182 | Multi-item Ack Prefix | _build_multi_item_ack_prefix (no current) | `app/core/response_builder.py` | `_build_multi_item_ack_prefix()` (line 377) | `Got it, {items_text}.` | Got it, Cheeseburger and Fries. | OK. | Keep. | Got it, Cheeseburger and Fries. | P2 | |
| 183 | Queue Transition Prefix | _build_queue_transition_prefix (with next) | `app/core/response_builder.py` | `_build_queue_transition_prefix()` (line 390) | `{added}. Now for the {next}.` | Cheeseburger added. Now for the Fries. | OK. | Keep. | Cheeseburger added. Now for the Fries. | P2 | |
| 184 | Queue Transition Prefix | _build_queue_transition_prefix (no next) | `app/core/response_builder.py` | `_build_queue_transition_prefix()` (line 391) | `{added}. Next item.` | Cheeseburger added. Next item. | "Next item" exposes queue mechanics. | `{added}. What's next?` | Cheeseburger added. What's next? | P1 | |
| 185 | Prefilled Confirmation Prefix | _build_prefilled_confirmation (with name) | `app/core/response_builder.py` | `_build_prefilled_confirmation()` (line 400) | `{item_name} {summary} — got it.` | Cheeseburger with Cheese, no onions — got it. | Em-dash followed by "got it" reads odd. | `Got the {item_name} {summary}.` | Got the Cheeseburger with Cheese, no onions. | P1 | |
| 186 | Prefilled Confirmation Prefix | _build_prefilled_confirmation (no name) | `app/core/response_builder.py` | `_build_prefilled_confirmation()` (line 401) | `Got it, {summary}.` | Got it, with Cheese, no onions. | OK. | Keep. | Got it, with Cheese, no onions. | P2 | |
| 187 | Entity Feedback (matched/unmatched) | _build_entity_feedback | `app/responses/item/format_utils.py` | `_build_entity_feedback()` (line 179, 183) | `Got {names}. I couldn't find {names}.` | Got Cheese and Bacon. I couldn't find pickle. | Good. Two-clause but useful. | Keep. | Got Cheese and Bacon. I couldn't find pickle. | P2 | |

---

## 3. Category Sections

> Each section captures the *current behavior pattern*, the *problems* surfaced in the inventory, and the *recommended replacement style*. Reviewers can comment in-line.

### 3.1 Greeting / Call Start

**Current behavior summary.** Two distinct greetings exist: one in `twilio_server.py` (Twilio voice route) and one in `response_builder.py` for the device-type prompt. Both reveal the brand and immediately ask a yes/no-style routing question.

**Current responses found.**

- `Welcome to Compass. Is this for pickup or delivery?` (transport)
- `Welcome to Compass. Are you calling from a landline or a mobile phone?` (FSM)
- `Sorry, are you on a landline or mobile phone?` (retry)

**Problems.** "Calling from a landline or a mobile phone" is a system question, not a customer-friendly one. Caller may not know which they're on. The same brand-greet repeats across paths, creating a cold "kiosk" feeling.

**Suggested response style.** One warm greeting line, then *one* short question. Avoid "calling from".

**Recommended.**
- `Thanks for calling Compass! Is this for pickup or delivery?`
- `Hi, thanks for calling Compass — are you on a cell phone or a home phone?`
- `Sorry, was that a cell or a home phone?`

### 3.2 Item Add Confirmation

**Current behavior summary.** Three distinct ack styles exist: terse `"Cheeseburger added"`, queue-aware `"... N more to go"`, and the order summary tail.

**Current responses found.**

- `{item_name} added. Would you like anything else?`
- `Added {qty} {item_name}.`
- `{added}. {this_added}. {N} more to go.`
- `{added}. {this_added}. That's everything. Would you like anything else?` (logical contradiction)
- `{item_name}, right? Yes or no.` (#20 — pre-add)
- `…Should I place the order Would you like to checkout?` (#25 — bug, double prompt)

**Problems.** Pre-add `"Yes or no"` tag and post-add `"Anything else?"` cadence both feel pushy. The "That's everything" + "anything else" contradiction is confusing. The unpunctuated `"Should I place the order"` tail is a likely bug.

**Suggested response style.** Short ack + soft handoff, never two questions in one breath.

**Recommended.**
- `Just to confirm — {item_name}?`
- `Added one {item_name}. Anything else?`
- `Got the {prev}. And the {this}. Anything else, or should I read your order back?`
- `Let me read it back: {items}. Total {total}. Should I send the payment link?`

### 3.3 Item Not Found

**Current behavior summary.** Five distinct branches: query+suggestions, query+no-suggestions, customizable-only, near-miss, and escalation. The "with suggestions" branch is the strongest. The escalation branch refers callers to "the restaurant" — confusing because they *are* calling the restaurant.

**Current responses found.**

- `I don't see {query} on the menu, but we have {options}. Which one would you like?` (good)
- `I don't see that on the menu. What else can I get you?`
- `{unavail} {customizable} can be customized to your liking. Want that?`
- `Did you mean {item}?`
- `I don't have that item. You can choose another item, or I can connect you to the restaurant.`
- `I'm having trouble finding that item. What else can I get for you?`
- `I could not find that on the menu.`
- `I couldn't find that item. Say the item name again.`

**Problems.**

- "Connect you to the restaurant" makes no sense from inside the restaurant's IVR.
- Multiple slightly-different fallbacks for "not found" — voice drift.
- Escalation never offers a way out.

**Suggested response style.** Always (a) acknowledge what was missing by name, (b) offer 2–3 close suggestions, (c) ask one question. No magic phrases.

**Recommended.**

- `I don't see {requested} on the menu. We do have {opt1}, {opt2}, or {opt3}. Which one would you like?`
- `I don't see hamburger on the menu. We do have Cheeseburger, Patty Burger, or Chicken Sandwich. Which one would you like?`
- `I'm having trouble finding that one — want to try a different item, or I can hand you off to a team member?`

### 3.4 Item Disambiguation

**Current behavior summary.** Triggered when STT/NLU yields multiple candidates. Three renderers: `confirm_item_ambiguous`, `confirm_item_from_category`, `menu_ambiguity_response`.

**Current responses found.**

- `Did you mean {options}?`
- `In {category}, I found {options}. Which one would you like?`
- `I found {options}. Which one did you mean?`
- `I found a few matches. Which one did you mean?`

**Problems.** "I found … or" reads stilted; should use commas and "or" only at the last item. The empty-options branch falls through to a generic fallback that gives the caller no path forward.

**Suggested response style.** Lead with a brief acknowledgment, list 2–3 options, ask one question. Use natural conjunctions (`X, Y, or Z`).

**Recommended.**

- `I'm not sure which one — did you mean {opt1}, {opt2}, or {opt3}?`
- `Under {category}, we have {options}. Which one?`

### 3.5 Modifier Question

**Current behavior summary.** Multi-step subdialog: ask, repeat, list, clarify, required, too-many, choice-guess. Logic accommodates `prompt_noun`/`prompt_verb` from menu config (e.g., `noun="topping"`, `verb="would you like"`).

**Current responses found.**

- `Any {noun} {verb}, like {examples}? You can say none.` — *grammar bug* when verb is "would you like"
- `Which option?`
- `I didn't catch that. Say 'list options' to hear all choices.`
- `Up to {N}. {options}.`
- `Need {N} more options. {options}.`
- `That is too many extras. You can choose up to {max}. Please pick again from {options}.`
- `Did you mean {modifier}? Yes or no.`

**Problems.**

1. **Grammar bug.** `"Any toppings would you like, like Cheese, Bacon..."` — `verb` is interpolated mid-sentence and creates a malformed sentence. Either the noun phrase or the verb phrase belongs at the end, not both. This is a P0.
2. Magic phrase `"say 'list options'"` exposes vocabulary.
3. `"Need 2 more options"` and `"Up to 3."` read like compiler errors.
4. `"Yes or no"` suffix on choice-guess.
5. Multi-clause too-many message is too dense.

**Suggested response style.** Reword so noun and verb fit grammatically. Drop magic phrases. Use natural connectors. Cap density to two clauses.

**Recommended.**

- `Any toppings? We've got Cheese, Bacon, or Lettuce — or just say no.`
- `Want me to read the full list?`
- `You'll need to pick 2 more — choose from Cheese, Bacon, or Lettuce.`
- `Only 3 extras allowed — which 3 from Cheese, Bacon, or Lettuce?`
- `Did you mean Cheese?` (no `Yes or no` tag)
- `Got Cheese and Bacon. Lettuce put me over the limit, and I couldn't find pickle. Want anything else, or move on?`

### 3.6 Side Question

**Current behavior summary.** Mirrors the Modifier subdialog. Supports min/max selectors, group prompts, escalation hints.

**Current responses found.**

- `Any {noun} {verb}, like {examples}? You can say none.` — same grammar bug
- `Need {N} more sides. {options}.`
- `A side is required. {options}.`
- `That is too many sides. You can choose up to {max}. Please pick again from {options}.`
- `Did you mean {side}? Yes or no.`

**Problems.** Identical to modifier flow — same grammar bug, same telegraphic structure, same yes/no suffix.

**Suggested response style.** Same rules as modifiers. Specifically reframe `"required"` and `"need N more"` into something a server would actually say.

**Recommended.**

- `Want a side? We've got Fries, Onion Rings, or Salad — or no thanks.`
- `Two more sides to pick — choose from Fries, Onion Rings, or Salad.`
- `Pick one side to continue — Fries, Onion Rings, or Salad.`
- `Only 2 sides allowed — which 2 from Fries or Onion Rings?`
- `Did you mean Fries?`

### 3.7 Quantity Question

**Current behavior summary.** Asks `How many {item}?` whenever the FSM enters the quantity step.

**Current responses found.**

- `How many {item_name} would you like?`
- `Please give a valid quantity for {item_name}.`
- `Please say a number for {item_name}, like 1 or 2.` (escalation)

**Problems.** The system **always** asks for quantity, even when there's no ambiguity. Per the project's core principle ("default to 1 unless ambiguous"), this is the wrong default and creates extra turns. **P0** — but this is a flow gate, not pure copy.

**Suggested response style.** When asked, keep concise.

**Recommended.**

- `How many {item_name}?`
- `How many — like 1 or 2?`

### 3.8 Combo / Drink Upsell

**Current behavior summary.** No upsell copy is wired in. Multi-item ack and queue transitions are the closest thing.

**Current responses found.**

- (None — system does not actively upsell.)

**Problems.** Missed revenue. Phone ordering benefits significantly from a single, polite upsell at the right moment.

**Suggested response style.** Single optional line, never repeated, only at appropriate FSM transitions (after first item added, before checkout).

**Recommended.**

- `Want to add a drink or a side to that?`
- `A Diet Coke or some Fries to go with it?`
- (FSM/handler change required — out of scope for copy review.)

### 3.9 Cart Summary

**Current behavior summary.** Three branches: empty, single item, multiple items. Multi-item branch does **not** read items back.

**Current responses found.**

- `Your cart is empty.`
- `You have {qty} {name}. Total {total}. Would you like to add more or check out?`
- `You have {N} items. Total {total}. Would you like to add more or check out?`
- `Your cart is empty. Add something first.`
- `Okay. Add more, remove items, or check your cart.`

**Problems.** Multi-item summary doesn't list contents — caller can't verify. "Add more, remove items, or check your cart" reads as a command menu.

**Suggested response style.** Read items back when feasible (≤3); summarize otherwise. Avoid command-menu phrasing.

**Recommended.**

- `Your cart is empty — what would you like to start with?`
- `That's 1 Cheeseburger — $5.99 so far. Add anything else, or ready to check out?`
- `You've got 3 items adding up to $14.50. Want me to read them, add more, or check out?`

### 3.10 Remove / Replace / Modify Item

**Current behavior summary.** Each operation has confirm + cancel + success copy. `confirm_modify_item` leaks system internals ("swap with the new version").

**Current responses found.**

- `Remove {item_name}?`
- `Removed {item_name}. Anything else?`
- `Replace {item_name} with {replacement}?`
- `What would you like instead of {item_name}?`
- `Update {item_name}? I'll swap it with the new version.`
- `Okay, keeping it.` / `Okay, no changes.` / `Okay, leaving it as is.` / `Okay, cancelled.`

**Problems.** Two-word prompts (`Remove Cheeseburger?`) lack the "I'll do X for you" tone of a server. The "swap with the new version" wording reveals internal state.

**Suggested response style.** First-person, friendly: "Want me to …".

**Recommended.**

- `Want me to remove the Cheeseburger?`
- `Removed Cheeseburger. Anything else?`
- `Got it — update your Cheeseburger with those changes?`
- `Okay, kept it as is.`

### 3.11 Checkout

**Current behavior summary.** Pre-checkout review summary + tail prompt. Has the double-question bug (`"Should I place the order Would you like to checkout?"`).

**Current responses found.**

- `{intro}Please review your order: {items_text}. Your total is {total}. Should I place the order` (no `?`)
- `{summary} Would you like to checkout?` (appended)
- `Your cart is empty. Add something first.`
- `Please finish this item first, or say cancel.`

**Problems.** The two questions back-to-back are a clear P0 bug. Item-line construction reads strangely (`"1 Cheeseburger, with Fries, add Cheese"` — `add` jars).

**Suggested response style.** Read items in natural prose, *one* question at the end.

**Recommended.**

- `That's a delivery order. Let me read it back: one Cheeseburger with Fries, plus extra Cheese. One Diet Coke. Your total is $14.50. Should I send the payment link?`

### 3.12 Pickup

**Current behavior summary.** Three end-of-call branches (SMS sent, no-SMS, generic) — all warm and friendly. The strongest copy in the system.

**Current responses found.**

- `Your order is in! Would you like me to text you a payment link, or would you prefer to pay when you arrive?`
- `Done! The payment link is on its way to your phone. See you soon!`
- `No problem! We'll see you when you get here. You can pay at the counter.`
- `All set! We'll see you soon. You can pay when you arrive.`

**Problems.** None substantive. P2 polish only.

**Suggested response style.** Maintain this tone everywhere else.

**Recommended.** Keep as-is.

### 3.13 Delivery

**Current behavior summary.** Multi-step address capture: area → ZIP → confirmation → fall through to checkout link, with a voice fallback for SMS-link failures. Heavy use of "delivery area" jargon.

**Current responses found.**

- `Got it. Delivery. Please say your delivery area.`
- `Please say your delivery area.` / `Please say the ZIP code.`
- `Please say your house number.` / `Please say your street name or street number.`
- `Apartment or suite number? Say none if there isn't one.` (good)
- `Just to confirm, that is {area}, ZIP code {postal_code}. Is that correct?`
- `Great, we deliver there. What would you like to order?` (good)
- `I've sent you a checkout link. Fill in your address and pay there. I'll confirm once it goes through.`
- `I'll take your address here instead. What's your house number?` (fallback, good)

**Problems.**

- "Delivery area" is jargon; callers say "neighborhood" or "zip" or "side of town".
- "Street name or street number" confuses — caller doesn't know which is asked.
- Some prompts lead with "Please say…" which reads imperative.

**Suggested response style.** Concrete neighborhood-friendly language. One ask per prompt. Lead with the question, not "Please say".

**Recommended.**

- `Got it — delivery. What neighborhood or area are we delivering to?`
- `What's your house number?` / `What's the street name?`
- `Apartment or suite number? Say none if there isn't one.`
- `Sent you a checkout link — finish your address there and pay. I'll confirm once it's in.`

### 3.14 Payment Link / SMS

**Current behavior summary.** Heavy duplication — three near-identical copies for "couldn't send the link" (`payment_link_send_failed`, `checkout_link_send_failed`, `payment_link_unavailable_now`). Strong loop guards via `payment_not_confirmed_yet` and `payment_verification_error` — these prevent the historical payment-loop bug.

**Current responses found.**

- `Payment link sent. I'll confirm once payment goes through.`
- `Waiting for payment. I'll confirm it as soon as it goes through.`
- `I haven't received confirmation yet. Please complete payment on the link. I'll confirm as soon as it goes through.`
- `I couldn't send the payment link. Your order is saved. Please try again shortly.` (×3 keys)
- `Payment didn't go through. Your order is saved. Try again shortly.`
- `Having trouble checking payment. Give it a moment, I'm still checking.`
- `There's no payment in progress.`
- `Payment has not started. Say checkout when ready.` (magic phrase)
- `Okay, checkout cancelled. Your cart is still here.` (good)

**Problems.**

- Triplicate "couldn't send the link" — consolidate.
- "Try again shortly" leaves the caller without an actual choice.
- Magic phrase "say checkout".

**Suggested response style.** Always offer a fallback path — "want me to retry?" or "pay when you arrive?"

**Recommended.**

- `Couldn't send the link — your order's saved. Want me to try again, or pay when you arrive?`
- `Payment didn't go through — your order's saved. Want to try again, or call us back?`
- `We haven't started checkout yet — let me know when you're ready.`

### 3.15 Order Confirmation

**Current behavior summary.** Single end-call line, with optional order number digit-spelled.

**Current responses found.**

- `Payment confirmed.{order_sentence} Your order has been placed successfully. Will be ready in 25 minutes. Thank you!`

**Problems.**

- **Hardcoded "25 minutes"** — false promise risk if kitchen is busy. **P0**.
- "Has been placed successfully" is corporate.

**Suggested response style.** Confirm. Read order number. Drop the time guarantee unless fed from per-restaurant config.

**Recommended.**

- `Payment confirmed!{order_sentence} Your order's all set — we'll have it ready as soon as we can. Thanks for calling Compass!`
- (If a per-restaurant ETA is configured: `… ready in about {N} minutes. Thanks for calling!`)

### 3.16 Error Recovery

**Current behavior summary.** Three distinct error fallbacks with similar intent, scattered across files.

**Current responses found.**

- `Sorry, I didn't understand that.` (registry default)
- `I didn't catch that. Please say it again.` (intent_not_allowed)
- `Sorry, I couldn't process that.` (readonly_interrupt)
- `Something went wrong. Start again.` (confirmation_state_error)
- `Something went wrong with that item. Let's start again.` (item_context_missing)

**Problems.** Three different "I didn't understand" lines; voice drift. None give the caller a path forward.

**Suggested response style.** Unified fallback that always offers next steps. Avoid "start again" — always re-prompt with context.

**Recommended.**

- `I didn't catch that. You can add an item, hear the menu, or say checkout.`
- `Sorry, something went off track. Let me know what you'd like to add and we'll keep going.`
- `I lost track of that item — could you tell me which one you wanted again?`

### 3.17 Fallback / Unknown Intent

**Current behavior summary.** Per-state guards in `intent_not_allowed.py`. Mostly "Please say X or Y" patterns.

**Current responses found.**

- `Please say landline or mobile phone.`
- `Please say pickup or delivery.`
- `Please finish this item, or say cancel.`
- `Please confirm the order, or say cancel.`
- `Please complete payment, or say cancel.`
- `That feature isn't ready yet.`

**Problems.** "Please say X or Y" reads as a chatbot. "That feature isn't ready yet" is a rough internal-tools phrase that should never reach a customer. **P0**.

**Suggested response style.** Reframe as a question, not an instruction.

**Recommended.**

- `Are you on a cell phone or a home phone?`
- `Pickup or delivery?`
- `Let's wrap up the {item} first — or say cancel.`
- `Just need a yes or no on the order.`
- `I'm not able to do that right now. Anything else I can help with?`

### 3.18 End Call

**Current behavior summary.** Multiple end-call paths: pickup SMS sent / no-SMS, landline declined, transfer-to-agent.

**Current responses found.**

- `Done! The payment link is on its way to your phone. See you soon!`
- `No problem! We'll see you when you get here. You can pay at the counter.`
- `All set! We'll see you soon. You can pay when you arrive.`
- `No problem. Call us anytime. Goodbye.`
- `Connecting you now. One moment.`
- `Okay. Connecting you to a team member now. One moment please.` (transport, drift)

**Problems.** Two slightly different transfer lines (one in registry, one in stream-ended TwiML). "Goodbye" reads stiff for a US restaurant context.

**Suggested response style.** Match the warm pickup-end tone. Always thank or wish well.

**Recommended.**

- `No problem — give us a call anytime. Take care!`
- `One moment — connecting you now.`

### 3.19 Other (Show Menu / Info / Price)

Covered in inventory rows #161–180. Categorically: prefer "we've got" over "categories include", drop period-pauses inside item blurbs, never read variant prices inline as a long list.

---

## 4. Suggested Response Style Guide — Compass Voice

These rules apply to every customer-facing string the IVR speaks. Reviewers can amend.

1. **Telephony first.** Optimize for spoken English on a phone, not a chat UI. No bullet points, no parenthetical asides, no "Please" front-loading.
2. **Short sentences.** Two clauses max. If a thought needs three clauses, split it across two prompts.
3. **One question at a time.** Never end with two question marks (e.g., the Checkout summary bug).
4. **Acknowledge, then ask.** `"Got it — delivery. What's your ZIP?"` — not just `"Please say your ZIP."`
5. **Echo names, not pronouns.** Say `"the Cheeseburger"`, not `"that"`. Voice context is fragile.
6. **No magic phrases.** Never tell the caller to say `"list options"`, `"checkout"`, or `"done"` as if they were commands. Reframe as a question.
7. **No yes/no tags.** Drop `"…Yes or no."` from the end of guesses. The question already implies a yes/no answer.
8. **Default quantity = 1.** Don't ask `"How many?"` unless the caller said something quantitative or ambiguous.
9. **Mention sides/drinks only when relevant** — at first item add or pre-checkout. Never repeat the same upsell.
10. **Avoid technical words.** Never say `intent`, `entity`, `slot`, `modifier`, `selector`, `available`, `option`, `category`, `valid`, `version`, `feature`. These leak system internals. Use `topping`, `extra`, `add-on`, `side`, `kind`, `section`.
11. **No false promises.** Don't hardcode `"25 minutes"`, `"on its way"`, or any time/state guarantee that isn't verified.
12. **Consistent voice.** Pick one phrasing for `cart_empty`, one for `couldn't_understand`, one for `connecting_you`. Don't have three slightly-different versions.
13. **Friendly but efficient.** Warm phrasing (`"Got it"`, `"All set"`, `"No problem"`) on success and end-call paths. Tight phrasing (`"Pickup or delivery?"`) on retries.
14. **Use natural conjunctions.** `"X, Y, or Z"` not `"X, Y, and Z"` for choices. Reserve `"and"` for confirmations of what's already added.
15. **Limit options spoken aloud to 3** — even if 8 are valid. Always allow `"want me to read the full list?"`.
16. **Never reveal queue/state internals.** No `"Starting with the X"`, `"Next item"`, `"swap with the new version"`, or `"That feature isn't ready yet."`
17. **Always offer a path forward.** Every error/recovery line ends with a hint — `"want to try again?"`, `"or pay when you arrive?"`, `"or hand you off to a team member?"`
18. **Curly-quote hygiene.** All copy uses straight ASCII quotes (`'`, `"`) so unit tests, TTS, and logs match. Curly characters in source files (`’`, `"`) must be normalized.

---

## 5. Review Sheet (CSV-paste-ready)

> Copy the block below into Excel / Google Sheets / Notion. The header row has 8 columns. Filter by `Priority` to triage P0 items first, by `Category` for owner-by-domain review.

```csv
ID,Category,Current Response,Suggested Response,Priority,Status,Reviewed By,Reviewer Comment
1,Greeting / Call Start,"Welcome to Compass. Is this for pickup or delivery?","Thanks for calling Compass! Is this order for pickup or delivery?",P1,Pending,,
2,Greeting / Call Start,"Welcome to Compass. Are you calling from a landline or a mobile phone?","Hi, thanks for calling Compass. Quick question — are you on a cell phone or a home phone?",P1,Pending,,
3,Fallback / Unknown Intent,"Sorry, I didn't catch that. Could you repeat?","Sorry, I didn't quite catch that — could you say it again?",P2,Pending,,
4,Fallback / Unknown Intent,"Sorry, I didn't understand that.","I didn't catch that. You can add an item, hear the menu, or say checkout.",P0,Pending,,
5,Fallback / Unknown Intent,"I didn't catch that. Please say it again.","I didn't catch that. You can add an item, hear the menu, or say checkout.",P0,Pending,,
6,Fallback / Unknown Intent,"That feature isn't ready yet.","I'm not able to do that right now. Anything else I can help with?",P0,Pending,,
7,Error Recovery,"Something went wrong. Start again.","Sorry, something went off track. Let me know what you'd like to add and we'll keep going.",P0,Pending,,
8,Error Recovery,"Something went wrong with that item. Let's start again.","I lost track of that item. Could you tell me which one you wanted again?",P0,Pending,,
9,Error Recovery,"Sorry, I couldn't process that.","Sorry, I missed that one — let's keep going.",P1,Pending,,
10,Greeting → Pickup/Delivery,"I'll connect you with a team member to place your order. Would you like to proceed?","For landline orders, I'll hand you off to one of our team members. Want me to connect you?",P1,Pending,,
11,Greeting → Pickup/Delivery,"Would you like to connect with a team member? Yes or no.","Should I connect you with a team member?",P1,Pending,,
12,End Call,"No problem. Call us anytime. Goodbye.","No problem — give us a call anytime. Take care!",P2,Pending,,
13,End Call,"Connecting you now. One moment.","One moment — connecting you now.",P2,Pending,,
14,End Call,"Okay. Connecting you to a team member now. One moment please.","One moment — connecting you now.",P1,Pending,,
15,Order Type,"Is this for pickup or delivery?","Is this for pickup or delivery?",P2,Pending,,
16,Order Type,"Is this for pickup or delivery? (repeat)","Sorry — pickup or delivery?",P0,Pending,,
17,Order Type,"Pickup. What would you like to order?","Got it — pickup. What can I get started for you?",P1,Pending,,
18,Order Type,"Delivery. What would you like to order?","Got it — delivery. What can I get started for you?",P1,Pending,,
19,Order Type,"I'll get to your order right away. First, is this for pickup or delivery?","Happy to grab that for you — first, is this pickup or delivery?",P2,Pending,,
20,Item Add Confirmation,"{item_name}, right? Yes or no.","Just to confirm — {item_name}?",P0,Pending,,
21,Item Add Confirmation,"{item_name} added. Would you like anything else?","Added one {item_name}. Anything else?",P1,Pending,,
22,Item Add Confirmation,"Added {qty} {item_name}. Would you like anything else?","Added {qty} {item_name}. Anything else?",P2,Pending,,
23,Item Add Confirmation,"{prev added}. {this added}. {N} more to go. Would you like anything else?","Got the {prev}. And the {this}. Still {N} more on your order — anything else first?",P1,Pending,,
24,Item Add Confirmation,"{prev}. {this}. That's everything. Would you like anything else?","{prev added}. And the {this}. Anything else, or should I read your order back?",P0,Pending,,
25,Checkout,"…Should I place the order Would you like to checkout? (double-prompt bug)","…Your total is $14.50. Should I send the payment link?",P0,Pending,,
26,Checkout,"{intro}Please review your order: {items}. Your total is {total}. Should I place the order","That's a {order_type} order. Let me read it back: {items}. Your total is {total}. Should I send the payment link?",P0,Pending,,
27,Item Disambiguation,"Did you mean {options}?","I'm not sure which one — did you mean {options}?",P1,Pending,,
28,Item Disambiguation,"I found a few matches. Which one did you mean?","I caught a few options — could you say the item again?",P2,Pending,,
29,Item Disambiguation,"In {category}, I found {options}. Which one would you like?","Under {category}, we have {options}. Which one would you like?",P1,Pending,,
30,Item Disambiguation,"I found {options}. Which one did you mean?","Couple of matches — was that the {opt1}, the {opt2}, or the {opt3}?",P1,Pending,,
31,Item Not Found,"I don't see {query} on the menu, but we have {options}. Which one would you like?","I don't see {query} on the menu, but we have {options}. Which one would you like?",P2,Pending,,
32,Item Not Found,"I don't see that on the menu. What else can I get you?","I don't see that one. What else were you in the mood for?",P2,Pending,,
33,Item Not Found,"{unavail} {customizable} can be customized to your liking. Want that?","{unavail} We do have a {customizable} you can build however you'd like — want me to set one up?",P1,Pending,,
34,Item Not Found,"Did you mean {item_name}?","Did you mean a {item_name}?",P2,Pending,,
35,Item Not Found,"I don't have that item. You can choose another item, or I can connect you to the restaurant.","I'm having trouble finding that one. Want to try a different item, or I can hand you off to a team member?",P0,Pending,,
36,Item Not Found,"I'm having trouble finding that item. What else can I get for you?","I'm having trouble finding that item. What else can I get for you?",P2,Pending,,
37,Item Not Found,"Which item would you like?","What can I get for you?",P1,Pending,,
38,Item Not Found,"I could not find that on the menu.","I don't see that on the menu. Want me to suggest something close?",P1,Pending,,
39,Item Not Found,"I couldn't find that item. Say the item name again.","I couldn't find that one — could you say the name again?",P1,Pending,,
40,Modifier Question,"Any {noun} {verb}, like {examples}? You can say none. (grammar bug)","Any {noun}? We've got {examples} — or just say no.",P0,Pending,,
41,Modifier Question,"Any {noun} {verb} on your {item_name}? You can say none. (grammar bug)","Any {noun} on your {item_name}? You can skip if you don't want any.",P0,Pending,,
42,Modifier Question,"Which option?","Which one would you like?",P1,Pending,,
43,Modifier Question,"I didn't catch that. Say 'list options' to hear all choices.","I didn't catch that. Want me to read the full list?",P0,Pending,,
44,Modifier Question,"Please choose one of the available options.","Just pick whichever one you'd like.",P1,Pending,,
45,Modifier Question,"Up to {max_selector}. {options}.","You can pick up to {max}. We've got {options}.",P0,Pending,,
46,Modifier Question,"Your options are {options}.","Here are your choices: {options}.",P2,Pending,,
47,Modifier Question,"Let me list the options.","I don't have a list to read here — what would you like?",P1,Pending,,
48,Modifier Question,"Did you mean {options}?","Did you mean {options}?",P2,Pending,,
49,Modifier Question,"Need {N} more options. {options}.","You'll need to pick {N} more — choose from {options}.",P0,Pending,,
50,Modifier Question,"An option is required. {options}.","Pick one to continue — {options}.",P0,Pending,,
51,Modifier Question,"That is too many extras. You can choose up to {max}. Please pick again from {options}.","Only {max} extras allowed — which {max} from {options}?",P0,Pending,,
52,Modifier Question,"That is too many extras. Please choose from {options}.","Just one of those — {options}?",P0,Pending,,
53,Modifier Question,"I added {accepted}. You can only pick {max}, so I couldn't add {dropped}. I couldn't find {unmatched}. Say done when you're ready.","Got {accepted}. {dropped} put me over the limit and I couldn't find {unmatched}. Want anything else, or move on?",P0,Pending,,
54,Modifier Question,"Did you mean {choice_name}? Yes or no.","Did you mean {choice_name}?",P0,Pending,,
55,Side Question,"Any {noun} {verb}, like {examples}? You can say none.","Want a side? We've got {examples} — or no thanks.",P0,Pending,,
56,Side Question,"Any {noun} {verb} with your {item_name}? You can say none.","Any sides with your {item_name}?",P0,Pending,,
57,Side Question,"Please choose one of the available sides.","Just pick a side and we'll roll.",P1,Pending,,
58,Side Question,"Need {N} more sides. {options}.","Two more sides to pick — choose from {options}.",P0,Pending,,
59,Side Question,"A side is required. {options}.","Pick one side to continue — {options}.",P0,Pending,,
60,Side Question,"Let me list the side options.","I don't have any sides for that one — anything else?",P1,Pending,,
61,Side Question,"Did you mean {options}? (sides)","Did you mean {options}?",P2,Pending,,
62,Side Question,"Which side did you want?","Which side did you want?",P2,Pending,,
63,Side Question,"That is too many sides. Please choose from {options}.","Just one side — {options}?",P0,Pending,,
64,Side Question,"That is too many sides. You can choose up to {max}. Please pick again from {options}.","Only {max} sides allowed — which {max} from {options}?",P0,Pending,,
65,Side Question,"Did you mean {choice_name}? Yes or no. (side)","Did you mean {choice_name}?",P0,Pending,,
66,Size Question,"What size would you like for {item_name}?","What size would you like for {item_name}?",P2,Pending,,
67,Size Question,"Available sizes for {item_name} are {options}.","For {item_name} we have {options} — which one?",P1,Pending,,
68,Size Question,"What size for {item_name}?","What size for {item_name}?",P2,Pending,,
69,Size Question,"I didn't catch that. Say 'list options' to hear all sizes for {item_name}.","I didn't catch that. Want me to read out the sizes?",P0,Pending,,
70,Size Question,"Please choose a size for {item_name}: {options}.","Please choose a size for {item_name}: {options}.",P2,Pending,,
71,Size Question,"That size is not available for {item_name}. Please choose {options}.","We don't have that size for {item_name} — try {options}.",P1,Pending,,
72,Size Question,"Let's make this easy. Say {options} for {item_name}.","Let me list it out: for {item_name} it's {options}.",P1,Pending,,
73,Size Question,"{item_name} does not need a size. Let's continue.","{item_name} only comes one way — moving on.",P2,Pending,,
74,Size Question,"Did you mean {choice_name}? Yes or no. (size)","Did you mean {choice_name}?",P0,Pending,,
75,Side Size,"Size for {side_item_name}? {sizes}.","What size {side_item_name} — {sizes}?",P1,Pending,,
76,Side Size,"Choose {sizes}.","Pick one — {sizes}.",P1,Pending,,
77,Side Size,"Did you mean {choice_name} for {side_item_name}? Yes or no.","Did you mean {choice_name} {side_item_name}?",P0,Pending,,
78,Quantity Question,"How many {item_name} would you like?","How many {item_name}? (FSM should default to 1 unless caller said a number)",P0,Pending,,
79,Quantity Question,"Please give a valid quantity for {item_name}.","How many {item_name} — like 1 or 2?",P1,Pending,,
80,Quantity Question,"Please say a number for {item_name}, like 1 or 2.","Please say a number for {item_name}, like 1 or 2.",P2,Pending,,
81,Cart Summary,"Your cart is empty.","Your cart is empty.",P2,Pending,,
82,Cart Summary,"You have {qty} {name}. Total {total}. Would you like to add more or check out?","That's {qty} {name} — {total} so far. Add anything else, or ready to check out?",P1,Pending,,
83,Cart Summary,"You have {N} items. Total {total}. Would you like to add more or check out?","You've got {N} items adding up to {total}. Want to hear them, add more, or check out?",P0,Pending,,
84,Cart Summary,"Your cart is empty. (registry)","Your cart is empty.",P2,Pending,,
85,Cart Summary,"Your total is {total}.","Your total is {total}.",P2,Pending,,
86,Cart Summary,"Your cart is empty. Add something first.","Your cart is empty — what would you like to start with?",P1,Pending,,
87,Cart Summary,"Okay. Add more, remove items, or check your cart.","Okay — add anything else, or want to hear what you've got?",P1,Pending,,
88,Cart Summary,"Should I clear the cart?","Should I clear the cart?",P2,Pending,,
89,Cart Summary,"Okay, your cart is cleared.","Okay, cart cleared. What would you like to add?",P1,Pending,,
90,Cart Summary,"Okay, I kept your cart.","Okay, kept it as is.",P2,Pending,,
91,Remove Item,"Remove {item_name}?","Want me to remove the {item_name}?",P1,Pending,,
92,Remove Item,"Removed {item_name}. Anything else?","Removed {item_name}. Anything else?",P2,Pending,,
93,Remove Item,"Okay, keeping it.","Okay, keeping it.",P2,Pending,,
94,Replace Item,"Replace {item_name} with {replacement}?","Replace {item_name} with {replacement}?",P2,Pending,,
95,Replace Item,"What would you like instead of {item_name}?","What would you like instead of {item_name}?",P2,Pending,,
96,Replace Item,"Okay, no changes.","Okay, no changes.",P2,Pending,,
97,Modify Item,"Update {item_name}? I'll swap it with the new version.","Got it — update your {item_name} with those changes?",P0,Pending,,
98,Modify Item,"Okay, leaving it as is.","Okay, leaving it as is.",P2,Pending,,
99,Modify Item,"Okay, cancelled.","Okay, cancelled — what's next?",P1,Pending,,
100,Cancel Current Item,"Cancel {item_name}?","Want to cancel the {item_name}?",P1,Pending,,
101,Cancel Current Item,"Still adding {item_name}. Cancel it and move on?","We're still on the {item_name} — drop it and move on?",P1,Pending,,
102,Cancel Current Item,"Okay, continuing. {options}.","Okay, sticking with it. Now — {options}?",P1,Pending,,
103,Cancel Current Item,"Okay, cancelled. What would you like next?","Okay, cancelled. What would you like next?",P2,Pending,,
104,Flow Guard,"Please finish {item_name}, or say cancel.","Let's wrap up the {item_name} first — or say cancel.",P1,Pending,,
105,Flow Guard,"Do you want to cancel {item_name}? Please say yes or no.","Cancel the {item_name}?",P0,Pending,,
106,Flow Guard,"Okay, cancelled. What would you like next?","Okay, cancelled. What would you like next?",P2,Pending,,
107,Intent Not Allowed,"Please say landline or mobile phone.","Are you on a cell phone or a home phone?",P1,Pending,,
108,Intent Not Allowed,"Pickup is available for landline callers only. Would you like to proceed?","For landline callers, I'll connect you with a team member — want to proceed?",P1,Pending,,
109,Intent Not Allowed,"Please say pickup or delivery.","Pickup or delivery?",P2,Pending,,
110,Intent Not Allowed,"Finish this item first, or say cancel.","Finish this item first, or say cancel.",P2,Pending,,
111,Intent Not Allowed,"Please finish this item, or say cancel.","Please finish this item, or say cancel.",P2,Pending,,
112,Intent Not Allowed,"Please confirm the order, or say cancel.","Please confirm the order, or say cancel.",P2,Pending,,
113,Intent Not Allowed,"Please complete payment, or say cancel.","Please complete payment, or say cancel.",P2,Pending,,
114,Checkout,"Please finish this item first, or say cancel. (checkout-blocked)","Please finish this item first, or say cancel.",P2,Pending,,
115,Pickup,"Your order is in! Would you like me to text you a payment link, or would you prefer to pay when you arrive?","Order's in! Want me to text you a payment link, or pay when you get here?",P2,Pending,,
116,Pickup,"Would you like a payment link sent to your phone, or will you pay when you pick up?","Would you like a payment link sent to your phone, or will you pay when you pick up?",P2,Pending,,
117,Pickup,"Done! The payment link is on its way to your phone. See you soon!","Done! The payment link is on its way to your phone. See you soon!",P2,Pending,,
118,Pickup,"No problem! We'll see you when you get here. You can pay at the counter.","No problem! We'll see you when you get here. You can pay at the counter.",P2,Pending,,
119,Pickup,"All set! We'll see you soon. You can pay when you arrive.","All set! We'll see you soon. You can pay when you arrive.",P2,Pending,,
120,Delivery,"Got it. Delivery. Please say your delivery area.","Got it — delivery. What neighborhood or area are we delivering to?",P1,Pending,,
121,Delivery,"Please say your delivery area.","Sorry — what neighborhood or area?",P1,Pending,,
122,Delivery,"Now please say your ZIP code.","And what's your ZIP code?",P2,Pending,,
123,Delivery,"Please say the ZIP code.","Sorry — what was that ZIP?",P2,Pending,,
124,Delivery,"Just to confirm, that is {area}, ZIP code {postal_code}. Is that correct?","Just to confirm — {area}, ZIP {postal_code}, right?",P2,Pending,,
125,Delivery,"I have {area}, ZIP code {postal_code}. Is that correct?","I have {area}, ZIP {postal_code}. Is that correct?",P2,Pending,,
126,Delivery,"Great, we deliver there. What would you like to order?","Great, we deliver there. What would you like to order?",P2,Pending,,
127,Delivery,"I've sent you a checkout link. Fill in your address and pay there. I'll confirm once it goes through.","Sent you a checkout link — finish your address there and pay. I'll confirm once it's in.",P1,Pending,,
128,Delivery,"Still waiting on checkout. I'll confirm as soon as payment goes through.","Still waiting on checkout. I'll confirm as soon as payment goes through.",P2,Pending,,
129,Delivery,"I heard {house_number}. Is that correct?","I heard {house_number}. Is that correct?",P2,Pending,,
130,Delivery,"I heard {street}. Is that correct?","I heard {street}. Is that correct?",P2,Pending,,
131,Delivery,"I heard {secondary_address}. Is that correct?","I heard {secondary_address}. Is that correct?",P2,Pending,,
132,Delivery,"Please say your house number.","What's your house number?",P2,Pending,,
133,Delivery,"Please say your street name or street number.","What's the street name?",P1,Pending,,
134,Delivery,"Apartment or suite number? Say none if there isn't one.","Apartment or suite number? Say none if there isn't one.",P2,Pending,,
135,Delivery,"Got your address. Sending payment link now.","Got your address. Sending payment link now.",P2,Pending,,
136,Delivery,"Checkout link sent. Enter your address and pay there. I'll confirm once it goes through.","Checkout link sent — finish up there and I'll confirm as soon as it's paid.",P1,Pending,,
137,Delivery,"I'll take your address here instead. What's your house number?","I'll take your address here instead. What's your house number?",P2,Pending,,
138,Delivery,"I'll take your address here instead. What's your house number? (failed fallback)","I'll take your address here instead. What's your house number?",P2,Pending,,
139,Delivery,"I'll get your order started right after. What's your ZIP code?","I'll get your order started right after. What's your ZIP code?",P2,Pending,,
140,Delivery,"Almost there. Please confirm your delivery area first.","Almost there. Please confirm your delivery area first.",P2,Pending,,
141,Delivery,"I'll get your order started right after. What's your delivery area?","I'll get your order started right after — what neighborhood are we delivering to?",P1,Pending,,
142,Delivery,"I'll get your order started right after. What's your street name?","I'll get your order started right after. What's your street name?",P2,Pending,,
143,Delivery,"Almost done with your address. Apartment or suite number?","Almost done with your address. Apartment or suite number?",P2,Pending,,
144,Delivery,"Let me finish confirming your address first. Is that correct?","Just need to confirm your address first — does {echoed} sound right?",P1,Pending,,
145,Delivery,"I'll get your order started right after. What's your house number?","I'll get your order started right after. What's your house number?",P2,Pending,,
146,Payment Link / SMS,"Waiting for payment. I'll confirm it as soon as it goes through.","Waiting for payment. I'll confirm it as soon as it goes through.",P2,Pending,,
147,Payment Link / SMS,"Payment link sent. I'll confirm once payment goes through.","Payment link sent. I'll confirm once payment goes through.",P2,Pending,,
148,Payment Link / SMS,"I couldn't send the payment link. Your order is saved. Please try again shortly.","Couldn't send the link — your order's saved. Want me to try again, or pay when you arrive?",P0,Pending,,
149,Payment Link / SMS,"I couldn't send the checkout link. Your order is saved. Please try again shortly.","Couldn't send the link — your order's saved. Want me to try again?",P0,Pending,,
150,Payment Link / SMS,"I couldn't send the payment link. Your order is saved. Please try again shortly. (3rd duplicate)","Couldn't send the link — your order's saved. Want me to try again?",P1,Pending,,
151,Payment Link / SMS,"Payment didn't go through. Your order is saved. Try again shortly.","Payment didn't go through — your order's saved. Want to try again, or call us back?",P0,Pending,,
152,Payment Link / SMS,"Payment has not started. Say checkout when ready.","We haven't started checkout yet — let me know when you're ready.",P1,Pending,,
153,Payment Link / SMS,"I haven't received confirmation yet. Please complete payment on the link. I'll confirm as soon as it goes through.","I don't see the payment confirmed yet — give the link a moment, I'll let you know once it's in.",P1,Pending,,
154,Payment Link / SMS,"Having trouble checking payment. Give it a moment, I'm still checking.","Having trouble verifying — give me a moment.",P1,Pending,,
155,Payment Link / SMS,"There's no payment in progress.","No payment is in progress right now.",P2,Pending,,
156,Payment Link / SMS,"There's no active order to cancel.","There's no active order to cancel.",P2,Pending,,
157,Payment Link / SMS,"Okay, checkout cancelled. Your cart is still here.","Okay, checkout cancelled. Your cart is still here.",P2,Pending,,
158,Payment Link / SMS,"View full order details in the checkout link.","View full order details in the checkout link.",P2,Pending,,
159,Order Confirmation,"Payment confirmed.{order_sentence} Your order has been placed successfully. Will be ready in 25 minutes. Thank you!","Payment confirmed!{order_sentence} Your order's all set — we'll have it ready as soon as we can. Thanks for calling!",P0,Pending,,
160,Order Confirmation,"Your order number is {digit-spaced}.","Your order number is {digit-spaced}.",P2,Pending,,
161,Show Menu,"Our categories are {categories}. Which one would you like?","We've got {categories}. Which sounds good?",P1,Pending,,
162,Show Menu,"Which category would you like?","What kind of food are you in the mood for?",P1,Pending,,
163,Show Menu,"In {category}, we have {items}. What would you like?","In {category}, we have {items}. What would you like?",P2,Pending,,
164,Show Menu,"There is nothing available in {category} right now.","We don't have any {category} on the menu today — anything else?",P1,Pending,,
165,Show Menu,"What would you like from {category}?","What would you like from {category}?",P2,Pending,,
166,Show Menu,"{item_name}. {description}","The {item_name} — {description}",P1,Pending,,
167,Show Menu,"{item_name} is on the menu.","Yes, we have the {item_name}.",P1,Pending,,
168,Show Menu,"{variant_label} {name} is {variant_price}.","{variant_label} {name} is {variant_price}.",P2,Pending,,
169,Show Menu,"{name} is {price}.","{name} is {price}.",P2,Pending,,
170,Show Menu,"{name} is {price} each.","{name} is {price} each.",P2,Pending,,
171,Show Menu,"{name} comes in {variants}.","{name} is {smallest_price} to {largest_price} depending on size.",P1,Pending,,
172,Show Menu,"Price information is not available right now.","I don't have the price on hand — sorry about that.",P1,Pending,,
173,Show Menu,"Yes, {item} is available. {desc} It comes in {variants}.","Yep, we've got {item} — comes in {variants}.",P1,Pending,,
174,Show Menu,"Yes, {item} is available.","Yes, {item} is available.",P2,Pending,,
175,Show Menu,"Yes, {modifier} is available for {price}.","Yes, {modifier} is available for {price}.",P2,Pending,,
176,Show Menu,"Yes, {name} is available.","Yes, {name} is available.",P2,Pending,,
177,Show Menu,"Yes, that option is available.","(skip — echo the matched name instead)",P1,Pending,,
178,Show Menu,"{name} is an add-on. Tell me the item, and I'll check it.","{name} is an add-on. Tell me the item, and I'll check it.",P2,Pending,,
179,Show Menu,"That goes with a specific item. Which item would you like it on?","That goes with a specific item. Which item would you like it on?",P2,Pending,,
180,Show Menu,"{modifier} on {item} costs {price}.","{modifier} on {item} costs {price}.",P2,Pending,,
181,Multi-item Ack,"Got it, {items_text}. Starting with the {current}.","Got it — {items_text}. Let's start with the {current}.",P1,Pending,,
182,Multi-item Ack,"Got it, {items_text}.","Got it, {items_text}.",P2,Pending,,
183,Queue Transition,"{added}. Now for the {next}.","{added}. Now for the {next}.",P2,Pending,,
184,Queue Transition,"{added}. Next item.","{added}. What's next?",P1,Pending,,
185,Prefilled Confirmation,"{item_name} {summary} — got it.","Got the {item_name} {summary}.",P1,Pending,,
186,Prefilled Confirmation,"Got it, {summary}.","Got it, {summary}.",P2,Pending,,
187,Entity Feedback,"Got {names}. I couldn't find {names}.","Got {names}. I couldn't find {names}.",P2,Pending,,
```

**Allowed `Status` values:** `Pending`, `Approved`, `Rejected`, `Needs More Info`.

---

## 6. Architecture Observation (informational — do not implement now)

**Centralization status.**

| Layer | Owns Copy? | Notes |
|---|---|---|
| `app/responses/*` | ✅ Yes — primary owner | All item, side, modifier, size, quantity, cart, menu, flow-guard, intent-not-allowed renderers live here. Pure functions of `(context, menu_repo, payload)`. **Strongest part of the architecture.** |
| `app/core/response_builder.py` | ⚠️ Partial — registry-only, but with ~50 inline lambda literals | The `_build_registry()` dict imports renderers from `app/responses/`, but it also inlines short literals (`"Your cart is empty."`, `"Pickup. What would you like to order?"`, etc.). These should arguably move into per-domain modules in `app/responses/` (e.g., `app/responses/order_type_responses.py`, `app/responses/payment_responses.py`, `app/responses/delivery_responses.py`). Today they violate the "all copy lives in `app/responses/`" rule. |
| `app/api/twilio_server.py` | ❌ Owns 2 strings — **architectural violation** | The first-call greeting and the empty-STT fallback are hardcoded in the transport layer. Per project STRICT RULES (no business logic in streaming layer), these should move to `app/responses/greeting_responses.py` (or similar) and be invoked via a `response_key`. |
| `app/api/voice_stream_server.py` | ❌ Owns 1 string — **drift risk** | The transfer-to-agent line duplicates the registry's `transferring_to_human_agent`. Single source of truth is broken. |
| `app/state_machine/handlers/*` | ✅ None found | Handlers correctly return `response_key` strings only. **Clean.** |
| `app/state_machine/handlers/payment/payment_flow_support.py` | ⚠️ One SMS string | Acceptable — it's an SMS body fallback, not TTS. Still, would be cleaner under `app/responses/sms_responses.py`. |

**Recommended future state (do not implement now).**

1. Create per-domain response modules: `payment_responses.py`, `delivery_responses.py`, `pickup_responses.py`, `order_type_responses.py`, `device_type_responses.py`, `greeting_responses.py`.
2. Migrate the inline lambda literals out of `response_builder.py` into those modules — `response_builder.py` becomes a pure registry.
3. Move the two strings in `twilio_server.py` into `greeting_responses.py` and invoke them via `ResponseBuilder.build("greeting_call_start")` and `ResponseBuilder.build("twilio_no_speech_fallback")`. Remove duplicate transfer string from `voice_stream_server.py`.
4. Add a unit test that asserts no string literal containing more than 5 alphabetic characters appears outside `app/responses/` (an architectural fitness function).
5. **Do not change copy directly inside handlers** — that's already not happening, which is good. The cleanup is purely about consolidating the registry.

**Risks of changing wording inline (today, before consolidation).**

- Some keys are referenced from multiple FSM states (e.g., `payment_link_send_failed`, `checkout_link_send_failed`, `payment_link_unavailable_now` — all near-identical). Editing one without auditing the others creates voice drift.
- Tests in `tests/responses/` and `tests/state_machine/handlers/` likely assert exact strings. Each copy change needs its test updated. (Out of scope for this audit; flagged for the implementation pass.)
- The `_build_entity_feedback` helper composes prefixes; changing its output cascades into several flows.

---

## How to use this document

1. **Triage P0 first.** Filter the table in §2 or the CSV in §5 by `Priority = P0`. Review wording, mark `Status` as `Approved`, `Rejected`, or `Needs More Info`, and add a comment.
2. **Forward to non-engineers.** This document is self-contained — file paths and function names exist for traceability but the copy review can happen entirely from the table and the per-category sections.
3. **Implementation pass (separate ticket).** Once approved copy is finalized in the `Suggested Response` column, an engineer can execute the changes file-by-file with the file paths preserved here. Tests in `tests/responses/` will need string updates.
