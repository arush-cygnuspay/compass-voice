# tests/payment/test_pickup_sms_resolver.py
"""
Unit tests for pickup_sms_resolver.resolve_pickup_sms_decision().

Tests the resolver in isolation — no TurnEngine or session setup needed.
Each test maps directly to one of the 12 required spec scenarios plus
regression cases.
"""
from __future__ import annotations

import pytest

from app.nlu.intent_resolution.intent import Intent
from app.state_machine.control_intent_resolver import (
    ControlIntentKind,
    ResolvedControlIntent,
)
from app.state_machine.handlers.payment.pickup_sms_resolver import (
    PickupSmsDecision,
    resolve_pickup_sms_decision,
)

# Convenience factories -------------------------------------------------------

def _affirm(source: str = "intent_registry") -> ResolvedControlIntent:
    return ResolvedControlIntent(
        kind=ControlIntentKind.AFFIRM,
        source=source,
        normalized_text="",
    )


def _deny(source: str = "intent_registry") -> ResolvedControlIntent:
    return ResolvedControlIntent(
        kind=ControlIntentKind.DENY,
        source=source,
        normalized_text="",
    )


def _cancel(source: str = "phrase_fallback") -> ResolvedControlIntent:
    return ResolvedControlIntent(
        kind=ControlIntentKind.CANCEL,
        source=source,
        normalized_text="",
    )


def _resolve(text: str, intent: Intent = Intent.UNKNOWN, control=None) -> PickupSmsDecision:
    return resolve_pickup_sms_decision(text, intent, control)


# ── Spec scenario 1: "yes" → SEND_SMS ────────────────────────────────────────

def test_yes_with_affirm_control_intent_returns_send_sms():
    assert _resolve("yes", Intent.CONFIRM, _affirm()) == PickupSmsDecision.SEND_SMS


def test_yes_with_no_nlu_but_affirm_phrase_fallback_returns_send_sms():
    # Simulates: NLU gives UNKNOWN, phrase fallback fires AFFIRM
    assert _resolve("yes", Intent.UNKNOWN, _affirm("phrase_fallback")) == PickupSmsDecision.SEND_SMS


# ── Spec scenario 2: "yeah send it" → SEND_SMS ───────────────────────────────

def test_yeah_send_it_returns_send_sms():
    # NLU UNKNOWN, control_intent None — resolved by exact-phrase candidate "send it"
    assert _resolve("yeah send it", Intent.UNKNOWN, None) == PickupSmsDecision.SEND_SMS


def test_go_ahead_and_send_it_returns_send_sms():
    # " and " split produces "send it" as candidate
    assert _resolve("go ahead and send it", Intent.UNKNOWN, None) == PickupSmsDecision.SEND_SMS


# ── Spec scenario 3: "text me the link" → SEND_SMS ───────────────────────────

def test_text_me_the_link_returns_send_sms():
    assert _resolve("text me the link", Intent.UNKNOWN, None) == PickupSmsDecision.SEND_SMS


def test_send_me_the_link_returns_send_sms():
    assert _resolve("send me the link", Intent.UNKNOWN, None) == PickupSmsDecision.SEND_SMS


# ── Spec scenario 4: "send payment link" → SEND_SMS ──────────────────────────

def test_send_payment_link_returns_send_sms():
    assert _resolve("send payment link", Intent.UNKNOWN, None) == PickupSmsDecision.SEND_SMS


def test_bare_payment_link_returns_send_sms():
    assert _resolve("payment link", Intent.UNKNOWN, None) == PickupSmsDecision.SEND_SMS


def test_sms_me_returns_send_sms():
    assert _resolve("sms me", Intent.UNKNOWN, None) == PickupSmsDecision.SEND_SMS


def test_text_me_the_payment_link_returns_send_sms():
    assert _resolve("text me the payment link", Intent.UNKNOWN, None) == PickupSmsDecision.SEND_SMS


# ── Spec scenario 5: "no" → PAY_ON_PICKUP ────────────────────────────────────

def test_no_with_deny_control_intent_returns_pay_on_pickup():
    assert _resolve("no", Intent.UNKNOWN, _deny()) == PickupSmsDecision.PAY_ON_PICKUP


# ── Spec scenario 6: "no thanks" → PAY_ON_PICKUP ─────────────────────────────

def test_no_thanks_with_deny_fallback_returns_pay_on_pickup():
    assert _resolve("no thanks", Intent.UNKNOWN, _deny("phrase_fallback")) == PickupSmsDecision.PAY_ON_PICKUP


# ── Spec scenario 7: "I'll pay when I arrive" → PAY_ON_PICKUP ────────────────

def test_ill_pay_when_i_arrive_returns_pay_on_pickup():
    assert _resolve("I'll pay when I arrive", Intent.UNKNOWN, None) == PickupSmsDecision.PAY_ON_PICKUP


def test_ill_pay_when_i_get_there_returns_pay_on_pickup():
    assert _resolve("I'll pay when I get there", Intent.UNKNOWN, None) == PickupSmsDecision.PAY_ON_PICKUP


# ── Spec scenario 8: "pay at the counter" → PAY_ON_PICKUP ────────────────────

def test_pay_at_the_counter_returns_pay_on_pickup():
    assert _resolve("pay at the counter", Intent.UNKNOWN, None) == PickupSmsDecision.PAY_ON_PICKUP


def test_pay_on_pickup_phrase_returns_pay_on_pickup():
    assert _resolve("pay on pickup", Intent.UNKNOWN, None) == PickupSmsDecision.PAY_ON_PICKUP


def test_pay_in_person_returns_pay_on_pickup():
    assert _resolve("pay in person", Intent.UNKNOWN, None) == PickupSmsDecision.PAY_ON_PICKUP


def test_ill_pay_there_returns_pay_on_pickup():
    assert _resolve("I'll pay there", Intent.UNKNOWN, None) == PickupSmsDecision.PAY_ON_PICKUP


def test_ill_pay_later_returns_pay_on_pickup():
    assert _resolve("I'll pay later", Intent.UNKNOWN, None) == PickupSmsDecision.PAY_ON_PICKUP


# ── Spec scenario 9: "don't send it" → PAY_ON_PICKUP ─────────────────────────

def test_dont_send_it_returns_pay_on_pickup():
    assert _resolve("don't send it", Intent.UNKNOWN, None) == PickupSmsDecision.PAY_ON_PICKUP


def test_do_not_send_it_returns_pay_on_pickup():
    assert _resolve("do not send it", Intent.UNKNOWN, None) == PickupSmsDecision.PAY_ON_PICKUP


def test_no_link_returns_pay_on_pickup():
    assert _resolve("no link", Intent.UNKNOWN, None) == PickupSmsDecision.PAY_ON_PICKUP


def test_no_sms_returns_pay_on_pickup():
    assert _resolve("no SMS", Intent.UNKNOWN, None) == PickupSmsDecision.PAY_ON_PICKUP


def test_no_text_message_returns_pay_on_pickup():
    assert _resolve("no text", Intent.UNKNOWN, None) == PickupSmsDecision.PAY_ON_PICKUP


def test_dont_text_me_returns_pay_on_pickup():
    assert _resolve("don't text me", Intent.UNKNOWN, None) == PickupSmsDecision.PAY_ON_PICKUP


# ── Spec scenario 10: "yes I'll pay at pickup" → PAY_ON_PICKUP ───────────────
# PAY_ON_PICKUP regex must win over AFFIRM control intent.

def test_yes_ill_pay_at_pickup_returns_pay_on_pickup_not_send_sms():
    # NLU CONFIRM → control_intent AFFIRM, but pay-there regex fires first.
    assert _resolve("yes I'll pay at pickup", Intent.CONFIRM, _affirm()) == PickupSmsDecision.PAY_ON_PICKUP


def test_sure_ill_pay_there_returns_pay_on_pickup():
    assert _resolve("sure I'll pay there", Intent.CONFIRM, _affirm()) == PickupSmsDecision.PAY_ON_PICKUP


def test_yes_ill_pay_when_i_arrive_returns_pay_on_pickup():
    assert _resolve("yes I'll pay when I arrive", Intent.CONFIRM, _affirm()) == PickupSmsDecision.PAY_ON_PICKUP


# ── Spec scenario 11: unknown/ambiguous → UNKNOWN (re-prompt) ────────────────

def test_hmm_not_sure_returns_unknown():
    assert _resolve("hmm not sure", Intent.UNKNOWN, None) == PickupSmsDecision.UNKNOWN


def test_let_me_think_returns_unknown():
    assert _resolve("let me think", Intent.UNKNOWN, None) == PickupSmsDecision.UNKNOWN


def test_empty_text_returns_unknown():
    assert _resolve("", Intent.UNKNOWN, None) == PickupSmsDecision.UNKNOWN


# ── Spec scenario 12: no "yes or no" wording in prompts ──────────────────────
# These tests reach into response_builder to confirm no robotic prompts remain.

def test_pickup_sms_prompts_contain_no_say_yes_or_no_wording():
    """None of the pickup SMS prompts should contain 'say yes or no' language."""
    from tests.support.voice_test_harness import install_test_stubs
    install_test_stubs()
    from app.core.response_builder import ResponseBuilder
    from app.menu.repository import MenuRepository
    from app.menu.store import MenuStore
    from pathlib import Path

    data_root = Path(__file__).resolve().parents[2] / "app" / "data" / "restaurants" / "steves_grill"
    store = MenuStore(
        menu_path=data_root / "menu.json",
        entity_index_path=data_root / "entity_index.json",
    )
    rb = ResponseBuilder(MenuRepository(store))

    robotic_fragments = [
        "please say yes or no",
        "just say yes or no",
        "say yes or no",
        "yes or no",
    ]

    pickup_sms_keys = [
        "pickup_ask_sms_permission",
        "pickup_repeat_sms_permission",
        "pickup_sms_sent_end_call",
        "pickup_no_sms_end_call",
        "pickup_end_call",
    ]

    for key in pickup_sms_keys:
        text = rb.build(key, None, {})
        text_lower = (text or "").lower()
        for fragment in robotic_fragments:
            assert fragment not in text_lower, (
                f"Response '{key}' contains robotic phrasing '{fragment}': {text!r}"
            )


# ── Extra coverage: payment_request / checkout intents ───────────────────────

def test_payment_request_intent_returns_send_sms():
    # payment_request is scoped to CONFIRMING_ORDER in the global registry;
    # the resolver handles it explicitly for this state.
    assert _resolve("send it", Intent.PAYMENT_REQUEST, None) == PickupSmsDecision.SEND_SMS


def test_checkout_intent_returns_send_sms():
    assert _resolve("checkout", Intent.CHECKOUT, None) == PickupSmsDecision.SEND_SMS


def test_finish_order_intent_returns_send_sms():
    assert _resolve("finish", Intent.FINISH_ORDER, None) == PickupSmsDecision.SEND_SMS


# ── cancel control_intent → PAY_ON_PICKUP ────────────────────────────────────

def test_cancel_control_intent_returns_pay_on_pickup():
    assert _resolve("cancel", Intent.CANCEL_ORDER, _cancel()) == PickupSmsDecision.PAY_ON_PICKUP


# ── Phrase variants from the spec ────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "yes",
    "yeah",
    "yep",
    "sure",
    "okay",
    "ok",
    "go ahead",
])
def test_affirm_phrases_return_send_sms_via_control_intent(text):
    # These are all in the global affirm phrase set → control_intent.AFFIRM.
    # We simulate that by passing _affirm() directly.
    assert _resolve(text, Intent.UNKNOWN, _affirm("phrase_fallback")) == PickupSmsDecision.SEND_SMS


@pytest.mark.parametrize("text", [
    "no",
    "nope",
    "nah",
    "no thanks",
    "I'll pass",
])
def test_deny_phrases_return_pay_on_pickup_via_control_intent(text):
    assert _resolve(text, Intent.UNKNOWN, _deny("phrase_fallback")) == PickupSmsDecision.PAY_ON_PICKUP


@pytest.mark.parametrize("text", [
    "send it",
    "text it",
    "send me",
])
def test_exact_send_phrases_return_send_sms(text):
    assert _resolve(text, Intent.UNKNOWN, None) == PickupSmsDecision.SEND_SMS


@pytest.mark.parametrize("text", [
    "I'll pay when I arrive",
    "I'll pay when I get there",
    "pay at the counter",
    "pay on pickup",
    "pay when I pick up",
    "pay at the store",
    "pay in person",
    "I'll pay there",
    "I'll pay later",
    "don't send it",
    "do not send it",
    "no link",
    "no SMS",
])
def test_pay_on_pickup_phrases_return_pay_on_pickup(text):
    assert _resolve(text, Intent.UNKNOWN, None) == PickupSmsDecision.PAY_ON_PICKUP
