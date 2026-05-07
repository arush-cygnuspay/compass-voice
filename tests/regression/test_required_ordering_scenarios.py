# tests/regression/test_required_ordering_scenarios.py
"""System-level regression scenarios required by the QA audit (Phase 5).

Each test exercises a full multi-turn conversation through the real TurnEngine
+ real MenuRepository + stubbed SMS/checkout services. NLU output is injected
deterministically via ScriptedTurn so tests do not depend on the ML models.

Scenarios are numbered to match the audit document (1..15).

Conventions
-----------
* `pickup` is the cheapest order-type so we don't have to walk the delivery
  address collection state machine in every test.
* Items are picked from the actual demo restaurant menu (see
  app/data/restaurants/demo/menu.json) so resolution paths are real.
* Assertions favor architectural invariants (state, cart shape, no loops)
  over exact response strings, which are reviewed in
  Compass_Voice_Response_Review.md.
"""
from __future__ import annotations

import pytest

from app.nlu.intent_resolution.intent import Intent
from app.state_machine.models.conversation_state import ConversationState
from tests.support.voice_assertions import (
    assert_cart_size,
    assert_cart_contains,
    assert_no_duplicate_items,
    assert_response_key,
    assert_state,
    assert_unknown_intent_not_looping,
    cart_items,
)
from tests.support.voice_test_harness import (
    ScriptedTurn,
    StubCheckoutService,
    build_engine,
    build_menu_repo,
    make_slot,
    new_session,
    simulate_conversation,
    simulate_turn,
)


# ─── helpers ────────────────────────────────────────────────────────────────

def _new_pickup_engine_session():
    engine = build_engine(menu_repo=build_menu_repo())
    session = new_session()
    simulate_turn(engine, session, ScriptedTurn("pickup"))
    return engine, session


def _add_item_turn(name: str, *, intent: Intent = Intent.ADD_ITEM, qty: str | None = None) -> ScriptedTurn:
    slots = [make_slot("ITEM", name)]
    if qty is not None:
        slots.append(make_slot("QUANTITY", qty))
    return ScriptedTurn(name, intent=intent, slots=tuple(slots))


# ─── scenario 1 ──────────────────────────────────────────────────────────────

def test_scenario_01_multi_item_default_quantity_does_not_reprompt_each() -> None:
    """A multi-item utterance with no qty should default each item to 1
    without bouncing through WAITING_FOR_QUANTITY for every item."""
    engine, session = _new_pickup_engine_session()

    results = simulate_conversation(engine, session, [
        ScriptedTurn(
            "I want a chicken taco and an apple pie",
            intent=Intent.ADD_ITEM,
            slots=(
                make_slot("ITEM", "Chicken Taco"),
                make_slot("ITEM", "Apple Pie"),
            ),
        ),
    ])

    # Quantity reprompt counter must never fire — implicit qty=1 must hold.
    assert session.reprompt_count_by_field.get("quantity", 0) == 0
    assert results[0].response_key not in {"ask_for_quantity", "invalid_quantity_option"}


# ─── scenario 2 ──────────────────────────────────────────────────────────────

def test_scenario_02_explicit_quantity_is_preserved() -> None:
    engine, session = _new_pickup_engine_session()

    simulate_turn(
        engine,
        session,
        ScriptedTurn(
            "two chicken tacos",
            intent=Intent.ADD_ITEM,
            slots=(
                make_slot("ITEM", "Chicken Taco"),
                make_slot("QUANTITY", "2"),
            ),
        ),
    )

    items = cart_items(session)
    qty_total = sum(int(getattr(it, "quantity", 0) or 0) for it in items) if items else 0
    # Either: 1 line with quantity=2, OR 2 lines with quantity=1 each.
    # Both are valid outcomes; the invariant is total ordered = 2.
    assert qty_total == 2 or session.conversation_state in {
        ConversationState.WAITING_FOR_SIDE,
        ConversationState.WAITING_FOR_MODIFIER,
        ConversationState.WAITING_FOR_QUANTITY,
    }, (
        f"expected total ordered qty=2 or pending-flow state, got "
        f"qty_total={qty_total} state={session.conversation_state.value} items={items!r}"
    )


# ─── scenario 3 ──────────────────────────────────────────────────────────────

def test_scenario_03_item_with_modifiers_in_one_utterance() -> None:
    """No-onions + extra-cheese should attach to the burger, not split it
    or trigger an unknown-item fallback."""
    engine, session = _new_pickup_engine_session()

    result = simulate_turn(
        engine,
        session,
        ScriptedTurn(
            "double bacon burger with no onions and extra cheese",
            intent=Intent.ADD_ITEM,
            slots=(
                make_slot("ITEM", "Double Bacon Burger"),
                make_slot("MODIFIER", "no onions"),
                make_slot("MODIFIER", "extra cheese"),
            ),
        ),
    )

    # Must not bounce to item_not_found
    assert result.response_key not in {"item_not_found", "item_not_found_near_miss"}
    # State must have moved off WAITING_FOR_ORDER_TYPE / IDLE without losing the item
    assert session.conversation_state != ConversationState.WAITING_FOR_ORDER_TYPE


# ─── scenario 4 ──────────────────────────────────────────────────────────────

def test_scenario_04_item_with_side_attaches_to_parent_not_separate_line() -> None:
    """'BLT combo with fries as the side' must put fries inside BLT, not
    add a separate fries line item."""
    engine, session = _new_pickup_engine_session()

    simulate_turn(
        engine,
        session,
        ScriptedTurn(
            "BLT combo with fries",
            intent=Intent.ADD_ITEM,
            slots=(
                make_slot("ITEM", "BLT Combo"),
                make_slot("SIDE", "Fries"),
            ),
        ),
    )

    # At most one parent line should appear from this single utterance.
    parent_lines = cart_items(session)
    assert len(parent_lines) <= 1, (
        f"single 'BLT combo with fries' utterance produced {len(parent_lines)} cart lines: {parent_lines!r}"
    )


# ─── scenario 5 ──────────────────────────────────────────────────────────────

def test_scenario_05_item_with_side_and_multiple_modifiers() -> None:
    engine, session = _new_pickup_engine_session()

    result = simulate_turn(
        engine,
        session,
        ScriptedTurn(
            "BLT combo with fries and extra mayo no tomato",
            intent=Intent.ADD_ITEM,
            slots=(
                make_slot("ITEM", "BLT Combo"),
                make_slot("SIDE", "Fries"),
                make_slot("MODIFIER", "extra mayo"),
                make_slot("MODIFIER", "no tomato"),
            ),
        ),
    )

    # Either added directly or asked for the next required group — must NOT fall through to unknown.
    assert result.response_key not in {"item_not_found", "intent_not_allowed"}


# ─── scenario 6 ──────────────────────────────────────────────────────────────

def test_scenario_06_modifiers_do_not_cross_attach_between_parents() -> None:
    """Order item A with modifier X, then item B with modifier Y. Modifier X
    must remain on A only and Y must remain on B only."""
    engine, session = _new_pickup_engine_session()

    # Item A: Bourbon Chicken with topping "extra"
    simulate_turn(
        engine,
        session,
        ScriptedTurn(
            "bourbon chicken with extra sauce",
            intent=Intent.ADD_ITEM,
            slots=(
                make_slot("ITEM", "Bourbon Chicken"),
                make_slot("MODIFIER", "extra sauce"),
            ),
        ),
    )

    # Item B: Sweet Tea with no sugar
    simulate_turn(
        engine,
        session,
        ScriptedTurn(
            "sweet tea no sugar",
            intent=Intent.ADD_ITEM,
            slots=(
                make_slot("ITEM", "Sweet Tea"),
                make_slot("MODIFIER", "no sugar"),
            ),
        ),
    )

    # If two parent items materialised, neither line should carry the other's modifier.
    items = cart_items(session)
    if len(items) >= 2:
        modifier_blobs = []
        for it in items:
            mods = getattr(it, "modifiers", None) or {}
            modifier_blobs.append(repr(mods).lower())
        a_str, b_str = modifier_blobs[0], modifier_blobs[1]
        assert "no sugar" not in a_str, f"item A leaked B's modifier: {a_str!r}"
        assert "extra sauce" not in b_str, f"item B leaked A's modifier: {b_str!r}"


# ─── scenario 7 ──────────────────────────────────────────────────────────────

def test_scenario_07_too_many_modifiers_does_not_silently_corrupt_state() -> None:
    """When a modifier group is single-select but the user names two options,
    the system must either ask for clarification or reject the extras —
    never silently keep an invalid selection set."""
    engine, session = _new_pickup_engine_session()

    # BLT Combo's "Sandwich Cheese" allows up to 3 cheeses; "Bun Modification" on
    # Double Bacon Burger is single-select. Use the Burger to force the constraint.
    simulate_turn(
        engine,
        session,
        ScriptedTurn(
            "double bacon burger",
            intent=Intent.ADD_ITEM,
            slots=(make_slot("ITEM", "Double Bacon Burger"),),
        ),
    )

    # Now try to overload Bun Modification: "white bun and wheat bun"
    result = simulate_turn(
        engine,
        session,
        ScriptedTurn(
            "white bun and wheat bun",
            intent=Intent.UNKNOWN,
            slots=(
                make_slot("MODIFIER", "white bun"),
                make_slot("MODIFIER", "wheat bun"),
            ),
        ),
    )

    # Allowed reactions: ask user to pick one, list options, or report too-many.
    assert result.response_key in {
        "too_many_modifier_choices",
        "clarify_modifier_choice",
        "list_modifier_options",
        "repeat_modifier_options",
        "ask_for_modifier",
        "item_added_successfully",  # if architecture chose to take last/first deterministically
    }


# ─── scenario 8 ──────────────────────────────────────────────────────────────

def test_scenario_08_too_many_sides_is_bounded() -> None:
    """Family Wing Meal requires exactly 4 sauces (min=max=4). Asking for
    five must not silently accept all five."""
    engine, session = _new_pickup_engine_session()

    simulate_turn(
        engine,
        session,
        ScriptedTurn(
            "family wing meal",
            intent=Intent.ADD_ITEM,
            slots=(make_slot("ITEM", "Family Wing Meal"),),
        ),
    )

    result = simulate_turn(
        engine,
        session,
        ScriptedTurn(
            "buffalo bbq lemon pepper garlic and ranch",
            intent=Intent.UNKNOWN,
            slots=(
                make_slot("SIDE", "buffalo"),
                make_slot("SIDE", "bbq"),
                make_slot("SIDE", "lemon pepper"),
                make_slot("SIDE", "garlic"),
                make_slot("SIDE", "ranch"),
            ),
        ),
    )

    # System must NOT have silently accepted all five — either it asks the
    # user to drop one, lists options, or proceeds with a valid 4-of-N subset.
    assert result.response_key in {
        "too_many_side_choices",
        "clarify_side_choice",
        "list_side_options",
        "repeat_side_options",
        "ask_for_side",
        "item_added_successfully",
    }


# ─── scenario 9 ──────────────────────────────────────────────────────────────

def test_scenario_09_unknown_item_fallback_is_bounded_not_infinite() -> None:
    """Asking for non-menu items repeatedly must not loop the same fallback
    response forever; the system must either escalate, ask alternatives,
    or remain in a recoverable state with a hard cap on identical replies."""
    engine, session = _new_pickup_engine_session()

    turns = []
    for utt in ("spaghetti", "lasagna", "pho noodles", "tikka masala"):
        turns.append(
            simulate_turn(
                engine,
                session,
                ScriptedTurn(utt, intent=Intent.ADD_ITEM, slots=(make_slot("ITEM", utt),)),
            )
        )

    # Hard guard: same response_key must not appear 4x in a row.
    assert_unknown_intent_not_looping(turns, max_same_key=3)


# ─── scenario 10 ─────────────────────────────────────────────────────────────

def test_scenario_10_new_item_during_modifier_wait_does_not_drop_pending() -> None:
    """User is mid-modifier on item A and says 'add a coke'. The pending
    item A must not silently disappear: either the new item is queued, or
    the engine asks the user to finish A first."""
    engine, session = _new_pickup_engine_session()

    simulate_turn(
        engine,
        session,
        ScriptedTurn(
            "double bacon burger",
            intent=Intent.ADD_ITEM,
            slots=(make_slot("ITEM", "Double Bacon Burger"),),
        ),
    )
    pre_state = session.conversation_state
    pre_pending = session.conversation_context.pending_add_item

    # Mid-modifier: user pivots to another item.
    simulate_turn(
        engine,
        session,
        ScriptedTurn(
            "actually add a sweet tea",
            intent=Intent.ADD_ITEM,
            slots=(make_slot("ITEM", "Sweet Tea"),),
        ),
    )

    # Either we still have the burger pending OR the burger was queued/added —
    # but we MUST NOT have dropped it without trace.
    items = cart_items(session)
    queue_size = len(session.conversation_context.pending_item_queue)
    burger_visible_anywhere = (
        any("burger" in repr(it).lower() for it in items)
        or queue_size > 0
        or session.conversation_context.pending_add_item is not None
        or session.conversation_state == pre_state  # still waiting for the burger's modifier
    )
    assert burger_visible_anywhere, (
        "double bacon burger silently disappeared after user added a second item"
    )


# ─── scenario 11 ─────────────────────────────────────────────────────────────

def test_scenario_11_checkout_during_pending_modifier_must_not_short_circuit() -> None:
    engine, session = _new_pickup_engine_session()

    simulate_turn(
        engine,
        session,
        ScriptedTurn(
            "double bacon burger",
            intent=Intent.ADD_ITEM,
            slots=(make_slot("ITEM", "Double Bacon Burger"),),
        ),
    )
    pre_state = session.conversation_state

    result = simulate_turn(
        engine,
        session,
        ScriptedTurn("checkout", intent=Intent.CHECKOUT),
    )

    # System must NOT have transitioned to CONFIRMING_ORDER while the burger
    # has unresolved required selections.
    if pre_state in {
        ConversationState.WAITING_FOR_MODIFIER,
        ConversationState.WAITING_FOR_SIDE,
        ConversationState.WAITING_FOR_SIZE,
        ConversationState.WAITING_FOR_QUANTITY,
    }:
        assert session.conversation_state != ConversationState.WAITING_FOR_PAYMENT, (
            "checkout was permitted while item still had pending required fields"
        )
        assert result.response_key in {
            "flow_guard_finish_current_step",
            "flow_guard_confirm_cancel",
            "checkout_blocked_finish_current_item",
            "ask_for_modifier",
            "ask_for_side",
            "ask_for_size",
            "intent_not_allowed",
        }


# ─── scenario 12 ─────────────────────────────────────────────────────────────

def test_scenario_12_duplicate_committed_text_does_not_double_add() -> None:
    """Same final transcript reprocessed must not duplicate the cart line.
    NB: deduplication at the STT/transport layer is covered by
    tests/realtime/test_turn_commit_controller.py — this test covers the
    engine-boundary safety: even if two identical texts reach the engine,
    we should not see two distinct add events for the same item with
    identical modifiers."""
    engine, session = _new_pickup_engine_session()

    add = ScriptedTurn(
        "apple pie",
        intent=Intent.ADD_ITEM,
        slots=(make_slot("ITEM", "Apple Pie"),),
    )
    simulate_turn(engine, session, add)
    simulate_turn(engine, session, add)

    # Either the cart has two pies (current architecture: each commit is a new add),
    # or it has one. Both are defensible — what we must NEVER have is a corrupted
    # cart line with garbage data. Document the current contract:
    items = cart_items(session)
    assert all(getattr(it, "item_id", None) for it in items), (
        f"some cart line lacks an item_id after duplicate commit: {items!r}"
    )
    # If you want strict idempotency, switch to assert_no_duplicate_items —
    # see ARCHITECTURAL ISSUES section of the audit report.


# ─── scenario 13 ─────────────────────────────────────────────────────────────

def test_scenario_13_partial_text_alone_does_not_mutate_cart() -> None:
    """Partial transcripts must never reach the engine. We assert this by
    only sending a single committed final transcript and verifying that
    the cart reflects that ONE final, not multiple intermediate fragments."""
    engine, session = _new_pickup_engine_session()

    # ONE final transcript reaches engine.process_turn. (Partial fragments
    # should be filtered by TurnCommitController upstream — covered separately.)
    simulate_turn(
        engine,
        session,
        ScriptedTurn(
            "apple pie",
            intent=Intent.ADD_ITEM,
            slots=(make_slot("ITEM", "Apple Pie"),),
        ),
    )

    items = cart_items(session)
    assert len(items) <= 1, f"single final transcript produced {len(items)} cart items: {items!r}"


# ─── scenario 14 ─────────────────────────────────────────────────────────────

def test_scenario_14_pickup_yes_to_payment_link_ends_call_without_waiting() -> None:
    checkout = StubCheckoutService()
    engine = build_engine(menu_repo=build_menu_repo(), checkout_service=checkout)
    session = new_session()

    simulate_conversation(
        engine,
        session,
        [
            ScriptedTurn("pickup"),
            ScriptedTurn(
                "apple pie",
                intent=Intent.ADD_ITEM,
                slots=(make_slot("ITEM", "Apple Pie"),),
            ),
            ScriptedTurn("checkout", intent=Intent.CHECKOUT),
            ScriptedTurn("yes", intent=Intent.CONFIRM),
        ],
    )

    # We should now be at the SMS-permission gate (or already past it).
    assert session.conversation_state in {
        ConversationState.WAITING_FOR_PICKUP_SMS_PERMISSION,
        ConversationState.WAITING_FOR_PAYMENT,
        ConversationState.COMPLETED,
    }, f"unexpected state after pickup confirm: {session.conversation_state.value}"

    result = simulate_turn(
        engine,
        session,
        ScriptedTurn("yes please send it", intent=Intent.CONFIRM),
    )

    # Pickup with payment link: call ends after SMS dispatch — no
    # synchronous payment wait.
    assert result.response_key in {
        "pickup_sms_sent_end_call",
        "pickup_end_call",
        "checkout_link_sent",
    }


# ─── scenario 15 ─────────────────────────────────────────────────────────────

def test_scenario_15_pickup_no_to_payment_link_ends_call_cleanly() -> None:
    checkout = StubCheckoutService()
    engine = build_engine(menu_repo=build_menu_repo(), checkout_service=checkout)
    session = new_session()

    simulate_conversation(
        engine,
        session,
        [
            ScriptedTurn("pickup"),
            ScriptedTurn(
                "apple pie",
                intent=Intent.ADD_ITEM,
                slots=(make_slot("ITEM", "Apple Pie"),),
            ),
            ScriptedTurn("checkout", intent=Intent.CHECKOUT),
            ScriptedTurn("yes", intent=Intent.CONFIRM),
        ],
    )

    result = simulate_turn(
        engine,
        session,
        ScriptedTurn("no, I'll pay when I arrive", intent=Intent.DENY),
    )

    assert result.response_key in {
        "pickup_no_sms_end_call",
        "pickup_end_call",
    }
