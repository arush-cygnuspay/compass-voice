from app.realtime.barge_in_policy import is_actionable_barge_in
from app.session.session import Session
from app.state_machine.models.conversation_state import ConversationState


def _build_session(state: ConversationState) -> Session:
    session = Session(session_id="barge-in-test", restaurant_id="demo")
    session.conversation_state = state
    return session


def test_modifier_done_like_phrase_is_actionable() -> None:
    session = _build_session(ConversationState.WAITING_FOR_MODIFIER)
    session.conversation_context.available_choices_values = ("Cheese", "Bacon")

    assert is_actionable_barge_in(session, "yeah thats good thanks")


def test_size_choice_is_actionable_during_playback() -> None:
    session = _build_session(ConversationState.WAITING_FOR_SIZE)
    session.conversation_context.available_choices_values = ("Small", "Medium", "Large")

    assert is_actionable_barge_in(session, "make it large")


def test_preorder_ordering_request_is_actionable() -> None:
    session = _build_session(ConversationState.WAITING_FOR_ORDER_TYPE)

    assert is_actionable_barge_in(session, "add a coke")


def test_non_contextual_filler_is_not_actionable() -> None:
    session = _build_session(ConversationState.WAITING_FOR_MODIFIER)
    session.conversation_context.available_choices_values = ("Cheese", "Bacon")

    assert not is_actionable_barge_in(session, "uh huh")


def test_delivery_zip_phrase_is_actionable_during_playback() -> None:
    session = _build_session(ConversationState.WAITING_FOR_DELIVERY_ELIGIBILITY)
    session.conversation_context.current_prompt_field = "delivery_postal_code"

    assert is_actionable_barge_in(session, "my zip code is 30000.")


def test_spoken_delivery_zip_phrase_is_actionable_during_playback() -> None:
    session = _build_session(ConversationState.WAITING_FOR_DELIVERY_ELIGIBILITY)
    session.conversation_context.current_prompt_field = "delivery_postal_code"

    assert is_actionable_barge_in(session, "it's twenty one thousand")
