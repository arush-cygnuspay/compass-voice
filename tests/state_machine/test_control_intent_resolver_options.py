import pytest

from app.nlu.intent_resolution.intent import Intent
from app.state_machine.control_intent_resolver import (
    ControlIntentKind,
    resolve_control_intent,
)
from app.state_machine.models.conversation_state import ConversationState


SELECTION_STATES = (
    ConversationState.WAITING_FOR_MODIFIER,
    ConversationState.WAITING_FOR_SIDE,
    ConversationState.WAITING_FOR_SIDE_SIZE,
    ConversationState.WAITING_FOR_SIZE,
    ConversationState.WAITING_FOR_QUANTITY,
)


RECOGNIZER_UTTERANCES = (
    "what's available",
    "whats available",
    "what is available",
    "what can i get",
    "what can i choose",
    "list them",
    "show me the options",
    "show me the choices",
    "what choices do i have",
    "menu",
    "what do you offer",
    "give me options",
    "read the options",
    "options",
    "choices",
    "available options",
)


STATE_GATED_LABELS = (
    "browse_menu",
    "browse_category",
    "availability_query",
    "recommendation_query",
    "ask_item_info",
)


@pytest.mark.parametrize("state", SELECTION_STATES)
@pytest.mark.parametrize("utterance", RECOGNIZER_UTTERANCES)
def test_options_recognizer_in_selection_states(state, utterance):
    resolved = resolve_control_intent(utterance, Intent.UNKNOWN, None, state)
    assert resolved is not None, f"expected OPTIONS_REQUEST for {utterance!r} in {state}"
    assert resolved.kind == ControlIntentKind.OPTIONS_REQUEST


@pytest.mark.parametrize("state", SELECTION_STATES)
@pytest.mark.parametrize("label", STATE_GATED_LABELS)
def test_state_gated_classifier_labels_in_selection_states(state, label):
    resolved = resolve_control_intent(
        "anything",
        label,
        None,
        state,
        intent_confidence=0.9,
    )
    assert resolved is not None, f"expected OPTIONS_REQUEST for label={label!r} in {state}"
    assert resolved.kind == ControlIntentKind.OPTIONS_REQUEST
    assert resolved.source == "intent_registry"


@pytest.mark.parametrize("utterance", [
    "whats available",
    "what can i get",
    "list them",
    "show me the options",
    "what choices do i have",
    "menu",
    "what do you offer",
    "give me options",
    "read the options",
    "choices",
])
def test_options_recognizer_does_not_intercept_when_idle(utterance):
    resolved = resolve_control_intent(utterance, Intent.UNKNOWN, None, ConversationState.IDLE)
    if resolved is not None:
        assert resolved.kind != ControlIntentKind.OPTIONS_REQUEST


@pytest.mark.parametrize("label", STATE_GATED_LABELS)
def test_state_gated_classifier_labels_do_not_fire_when_idle(label):
    resolved = resolve_control_intent(
        "anything",
        label,
        None,
        ConversationState.IDLE,
        intent_confidence=0.9,
    )
    assert resolved is None or resolved.kind != ControlIntentKind.OPTIONS_REQUEST


@pytest.mark.parametrize("utterance", [
    "options",
    "what are the options",
    "what are my options",
    "what can i choose",
    "what cheese do you have",
    "what sides do you have",
    "what sizes do you have",
    "available toppings",
    "tell me the options",
    "tell me the choices",
    "list the choices",
    "list the options",
])
def test_existing_options_whitelist_still_resolves(utterance):
    resolved = resolve_control_intent(
        utterance,
        Intent.UNKNOWN,
        None,
        ConversationState.WAITING_FOR_MODIFIER,
    )
    assert resolved is not None
    assert resolved.kind == ControlIntentKind.OPTIONS_REQUEST


@pytest.mark.parametrize("utterance,expected_kind", [
    ("help", ControlIntentKind.META_CLARIFY),
    ("what do you mean", ControlIntentKind.META_CLARIFY),
    ("repeat that", ControlIntentKind.META_CLARIFY),
    ("yes", ControlIntentKind.AFFIRM),
    ("no", ControlIntentKind.DENY),
    ("done", ControlIntentKind.DONE),
    ("cancel", ControlIntentKind.CANCEL),
])
def test_other_control_intents_still_resolve(utterance, expected_kind):
    resolved = resolve_control_intent(
        utterance,
        Intent.UNKNOWN,
        None,
        ConversationState.WAITING_FOR_MODIFIER,
    )
    assert resolved is not None
    assert resolved.kind == expected_kind


@pytest.mark.parametrize("utterance", [
    "cheese",
    "extra bacon",
    "no onions",
    "fries please",
])
def test_slot_like_utterances_pass_through(utterance):
    resolved = resolve_control_intent(
        utterance,
        Intent.UNKNOWN,
        None,
        ConversationState.WAITING_FOR_MODIFIER,
    )
    assert resolved is None or resolved.kind not in {
        ControlIntentKind.OPTIONS_REQUEST,
    }
