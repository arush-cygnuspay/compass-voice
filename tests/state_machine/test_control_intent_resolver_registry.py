"""Registry-driven tests for ``control_intent_resolver``.

These tests exercise the post-refactor resolution order:

1. Primary NLU intent → ``_INTENT_RULES`` lookup (source ``intent_registry``).
2. Sub-intent → same registry (source ``sub_intent_registry``).
3. State phrase rule (``continue``/``next`` in selection states).
4. Phrase fallback (deprecated safety net, source ``phrase_fallback`` /
   ``recognizer_extended``).
"""
from __future__ import annotations

import logging

import pytest

from app.nlu.intent_resolution.intent import Intent
from app.state_machine.control_intent_resolver import (
    ControlIntentKind,
    IntentRule,
    _INTENT_RULES,
    _SELECTION_STATES,
    _ORDER_CONFIRM_STATES,
    log_control_intent_event,
    resolve_control_intent,
)
from app.state_machine.models.conversation_state import ConversationState


SELECTION_STATE_LIST = tuple(_SELECTION_STATES)


OPTIONS_REQUEST_LABELS = (
    "ask_options",
    "options_request",
    "list_options",
    "browse_menu",
    "browse_category",
    "availability_query",
    "recommendation_query",
    "ask_item_info",
)


def _pick_in_scope_state(rule: IntentRule) -> ConversationState:
    if rule.states is None:
        return ConversationState.WAITING_FOR_MODIFIER
    return next(iter(rule.states))


def _pick_out_of_scope_state(rule: IntentRule) -> ConversationState | None:
    if rule.states is None:
        return None
    for state in ConversationState:
        if state not in rule.states:
            return state
    return None


@pytest.mark.parametrize("label,rules", sorted(_INTENT_RULES.items()))
def test_registry_label_resolves_in_scope(label: str, rules: tuple[IntentRule, ...]):
    rule = rules[0]
    state = _pick_in_scope_state(rule)

    resolved = resolve_control_intent(
        "anything",
        label,
        None,
        state,
        intent_confidence=0.9,
    )

    assert resolved is not None, f"expected registry hit for label={label!r} in {state}"
    assert resolved.kind == rule.target_kind
    assert resolved.source == "intent_registry"
    assert resolved.detected_intent == label


@pytest.mark.parametrize("label,rules", sorted(_INTENT_RULES.items()))
def test_registry_label_does_not_resolve_out_of_scope(
    label: str, rules: tuple[IntentRule, ...]
):
    rule = rules[0]
    out_of_scope = _pick_out_of_scope_state(rule)
    if out_of_scope is None:
        pytest.skip(f"label {label!r} is unscoped (no out-of-scope state)")

    resolved = resolve_control_intent(
        "anything",
        label,
        None,
        out_of_scope,
        intent_confidence=0.9,
    )

    if resolved is not None:
        assert resolved.source != "intent_registry", (
            f"unexpected registry hit for {label!r} in out-of-scope {out_of_scope}"
        )


@pytest.mark.parametrize("state", SELECTION_STATE_LIST)
@pytest.mark.parametrize("label", OPTIONS_REQUEST_LABELS)
def test_options_labels_resolve_in_every_selection_state(state: ConversationState, label: str):
    resolved = resolve_control_intent(
        "anything",
        label,
        None,
        state,
        intent_confidence=0.9,
    )

    assert resolved is not None
    assert resolved.kind == ControlIntentKind.OPTIONS_REQUEST
    assert resolved.source == "intent_registry"


@pytest.mark.parametrize("label", [
    "list_options",
    "browse_menu",
    "browse_category",
    "availability_query",
    "recommendation_query",
    "ask_item_info",
])
def test_state_gated_options_labels_do_not_fire_at_idle(label: str):
    resolved = resolve_control_intent(
        "anything",
        label,
        None,
        ConversationState.IDLE,
        intent_confidence=0.9,
    )

    if resolved is not None:
        assert resolved.kind != ControlIntentKind.OPTIONS_REQUEST


def test_unscoped_options_labels_still_resolve_at_idle():
    for label in ("ask_options", "options_request"):
        resolved = resolve_control_intent(
            "anything",
            label,
            None,
            ConversationState.IDLE,
            intent_confidence=0.9,
        )
        assert resolved is not None, f"{label!r} should resolve in any state"
        assert resolved.kind == ControlIntentKind.OPTIONS_REQUEST


def test_sub_intent_registry_resolves_when_primary_misses():
    resolved = resolve_control_intent(
        "tell me what you got",
        Intent.UNKNOWN,
        "browse_menu",
        ConversationState.WAITING_FOR_SIDE,
        intent_confidence=0.9,
    )

    assert resolved is not None
    assert resolved.kind == ControlIntentKind.OPTIONS_REQUEST
    assert resolved.source == "sub_intent_registry"
    assert resolved.detected_sub_intent == "browse_menu"


def test_state_phrase_continue_in_selection_state():
    resolved = resolve_control_intent(
        "continue",
        Intent.UNKNOWN,
        None,
        ConversationState.WAITING_FOR_MODIFIER,
    )
    assert resolved is not None
    assert resolved.kind == ControlIntentKind.DONE
    assert resolved.source == "state_phrase"


def test_phrase_fallback_emits_log_event(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.INFO, logger="app.state_machine.control_intent_resolver")

    resolved = resolve_control_intent(
        "yeah",
        Intent.UNKNOWN,
        None,
        ConversationState.WAITING_FOR_MODIFIER,
    )

    assert resolved is not None
    assert resolved.kind == ControlIntentKind.AFFIRM
    assert resolved.source == "phrase_fallback"

    fallback_events = [
        record for record in caplog.records
        if getattr(record, "event_name", None) == "phrase_fallback_used"
    ]
    assert fallback_events, "expected at least one phrase_fallback_used event"
    event = fallback_events[-1]
    assert event.kind == ControlIntentKind.AFFIRM.value
    assert event.normalized_text == "yeah"
    assert event.state == ConversationState.WAITING_FOR_MODIFIER.value


def test_intent_registry_emits_no_phrase_fallback_log(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.INFO, logger="app.state_machine.control_intent_resolver")

    resolved = resolve_control_intent(
        "yes please",
        Intent.AFFIRM,
        None,
        ConversationState.CONFIRMING_ORDER,
        intent_confidence=0.9,
    )

    assert resolved is not None
    assert resolved.source == "intent_registry"

    fallback_events = [
        record for record in caplog.records
        if getattr(record, "event_name", None) == "phrase_fallback_used"
    ]
    assert not fallback_events, "registry path must not emit phrase fallback events"

    registry_events = [
        record for record in caplog.records
        if getattr(record, "event_name", None) == "intent_registry_resolved"
    ]
    assert registry_events, "expected an intent_registry_resolved log event"
    event = registry_events[-1]
    assert event.label == "affirm"
    assert event.target_kind == ControlIntentKind.AFFIRM.value
    assert event.confidence == pytest.approx(0.9)


def test_low_confidence_blocks_registry_but_phrase_fallback_can_fire():
    resolved = resolve_control_intent(
        "yes",
        Intent.AFFIRM,
        None,
        ConversationState.CONFIRMING_ORDER,
        intent_confidence=0.30,
    )

    assert resolved is not None
    assert resolved.kind == ControlIntentKind.AFFIRM
    assert resolved.source == "phrase_fallback"


def test_low_confidence_no_fallback_returns_none():
    resolved = resolve_control_intent(
        "totally unrelated text",
        Intent.AFFIRM,
        None,
        ConversationState.CONFIRMING_ORDER,
        intent_confidence=0.30,
    )

    assert resolved is None


def test_payment_request_only_resolves_in_confirm_states():
    state_in = next(iter(_ORDER_CONFIRM_STATES))
    resolved_in = resolve_control_intent(
        "ready to pay",
        "payment_request",
        None,
        state_in,
        intent_confidence=0.9,
    )
    assert resolved_in is not None
    assert resolved_in.kind == ControlIntentKind.AFFIRM

    resolved_out = resolve_control_intent(
        "ready to pay",
        "payment_request",
        None,
        ConversationState.WAITING_FOR_MODIFIER,
        intent_confidence=0.9,
    )
    if resolved_out is not None:
        assert resolved_out.source != "intent_registry"


def test_negative_slot_guard_preserved_against_deny_intent():
    """Existing behavior: ``no <slot>`` in modifier/side states must not be
    routed as DENY even when NLU misclassifies the utterance."""
    resolved = resolve_control_intent(
        "no onions",
        Intent.DENY,
        None,
        ConversationState.WAITING_FOR_MODIFIER,
        intent_confidence=0.9,
    )
    assert resolved is None

    # Pure deny utterances still resolve.
    resolved_clean = resolve_control_intent(
        "no thanks",
        Intent.DENY,
        None,
        ConversationState.WAITING_FOR_MODIFIER,
        intent_confidence=0.9,
    )
    assert resolved_clean is not None
    assert resolved_clean.kind == ControlIntentKind.DENY


PHRASE_FALLBACK_REGRESSIONS = (
    ("yeah", ControlIntentKind.AFFIRM),
    ("nope", ControlIntentKind.DENY),
    ("done", ControlIntentKind.DONE),
    ("cancel", ControlIntentKind.CANCEL),
    ("help", ControlIntentKind.META_CLARIFY),
    ("what are the options", ControlIntentKind.OPTIONS_REQUEST),
    ("options", ControlIntentKind.OPTIONS_REQUEST),
)


@pytest.mark.parametrize("utterance,expected_kind", PHRASE_FALLBACK_REGRESSIONS)
def test_phrase_fallback_regression(utterance: str, expected_kind: ControlIntentKind):
    resolved = resolve_control_intent(
        utterance,
        Intent.UNKNOWN,
        None,
        ConversationState.WAITING_FOR_MODIFIER,
    )
    assert resolved is not None
    assert resolved.kind == expected_kind


def test_log_control_intent_event_smoke():
    # Importable and callable; the parametrized tests above check field shape.
    log_control_intent_event("test_event", state="idle", kind="affirm")
