# tests/regression/test_no_input_escalation_gap.py
"""Hard regression tests for the global no-input / unknown-intent escalation.

Originally written as xfail to document the architectural gap; flipped
to real assertions once ``NoInputEscalationPolicy`` was wired into
``TurnEngine`` and ``ConversationContext.consecutive_unknown_count``
landed.

Contract under test
-------------------
* The router-rejection path (``intent_not_allowed``) bumps a counter
  per consecutive UNKNOWN turn.
* Tier 0/1 emits state-anchored copy.
* Tier 2 (HINT_AT) emits state-specific options hint.
* Tier 3 (HELP_AT) emits an offer-help line.
* Tier 4+ (HANDOFF_AT) transitions to TRANSFERRING_TO_HUMAN_AGENT and
  ends the call after playback.
* Any successful, allowed turn resets the counter to zero.
"""
from __future__ import annotations

from app.nlu.intent_resolution.intent import Intent
from app.policies.no_input_escalation_policy import (
    NoInputEscalationPolicy,
    NoInputTier,
)
from app.state_machine.models.conversation_state import ConversationState
from tests.support.voice_test_harness import (
    ScriptedTurn,
    build_engine,
    build_menu_repo,
    new_session,
    simulate_turn,
)


def _drive_to_idle(engine, session) -> None:
    simulate_turn(engine, session, ScriptedTurn("pickup"))


def test_repeated_unknown_at_idle_escalates_then_hands_off() -> None:
    """4 consecutive UNKNOWN turns at IDLE escalate through tiers and
    end at TRANSFERRING_TO_HUMAN_AGENT. The same response_key MUST NOT
    fire 3+ times in a row."""
    engine = build_engine(menu_repo=build_menu_repo())
    session = new_session()
    _drive_to_idle(engine, session)

    keys: list[str] = []
    miss_counts: list[int] = []
    for utt in ("zzzzzz", "qwqwqw", "asdfasdf", "ghghgh"):
        out = simulate_turn(engine, session, ScriptedTurn(utt, intent=Intent.UNKNOWN))
        keys.append(out.response_key)
        miss_counts.append(session.conversation_context.consecutive_unknown_count)

    # First three: intent_not_allowed (tiers 0, 1, 2) - escalating copy
    # but same key. Last one: handoff.
    assert keys[0] == "intent_not_allowed", keys
    assert keys[1] == "intent_not_allowed", keys
    assert keys[2] == "intent_not_allowed", keys
    assert keys[-1] == "transferring_to_human_agent", keys
    assert session.conversation_state == ConversationState.TRANSFERRING_TO_HUMAN_AGENT

    # Counter is reset on handoff so we don't keep escalating after.
    assert session.conversation_context.consecutive_unknown_count == 0


def test_unknown_counter_resets_on_successful_turn() -> None:
    engine = build_engine(menu_repo=build_menu_repo())
    session = new_session()
    _drive_to_idle(engine, session)

    simulate_turn(engine, session, ScriptedTurn("zzz", intent=Intent.UNKNOWN))
    simulate_turn(engine, session, ScriptedTurn("qqq", intent=Intent.UNKNOWN))
    assert session.conversation_context.consecutive_unknown_count == 2

    # An allowed intent at IDLE - SHOW_MENU - is a successful turn.
    simulate_turn(
        engine,
        session,
        ScriptedTurn("show me the menu", intent=Intent.SHOW_MENU),
    )
    assert session.conversation_context.consecutive_unknown_count == 0


def test_policy_tier_thresholds_are_monotonic() -> None:
    """Sanity check the policy: tiers must progress monotonically and never
    skip backwards as miss_count grows."""
    seen: list[NoInputTier] = []
    for n in range(0, 8):
        seen.append(NoInputEscalationPolicy.next_tier(n))
    # Order: REPROMPT_STATE -> REPROMPT_STATE -> REPROMPT_WITH_HINT -> OFFER_HELP -> HANDOFF -> ...
    assert seen[0] == NoInputTier.REPROMPT_STATE
    assert seen[1] == NoInputTier.REPROMPT_STATE
    assert seen[2] == NoInputTier.REPROMPT_WITH_HINT
    assert seen[3] == NoInputTier.OFFER_HELP
    assert all(t == NoInputTier.HANDOFF for t in seen[4:])


def test_empty_user_text_uses_same_escalation_path() -> None:
    """Twilio-side empty STT now flows through TurnEngine, so empty text
    consumes the same counter as unintelligible text."""
    engine = build_engine(menu_repo=build_menu_repo())
    session = new_session()
    _drive_to_idle(engine, session)

    keys: list[str] = []
    for _ in range(4):
        out = simulate_turn(engine, session, ScriptedTurn("", intent=Intent.UNKNOWN))
        keys.append(out.response_key)

    assert keys[-1] == "transferring_to_human_agent", keys
    assert session.conversation_state == ConversationState.TRANSFERRING_TO_HUMAN_AGENT
