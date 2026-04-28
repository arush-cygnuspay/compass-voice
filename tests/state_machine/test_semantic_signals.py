# tests/state_machine/test_semantic_signals.py
from app.nlu.intent_resolution.intent import Intent
from app.state_machine.semantic_signals import (
    is_confirmation_accept_response,
    is_confirmation_reject_response,
    is_options_like_response,
)


def test_confirmation_accept_response_supports_natural_affirm_phrases() -> None:
    assert is_confirmation_accept_response(Intent.UNKNOWN, "sounds good")
    assert is_confirmation_accept_response(Intent.UNKNOWN, "place it")
    assert is_confirmation_accept_response(Intent.UNKNOWN, "confirm it")


def test_confirmation_reject_response_supports_natural_modify_phrases() -> None:
    assert is_confirmation_reject_response(Intent.UNKNOWN, "not correct")
    assert is_confirmation_reject_response(Intent.UNKNOWN, "change it")
    assert is_confirmation_reject_response(Intent.UNKNOWN, "go back")
    assert is_confirmation_reject_response(Intent.UNKNOWN, "wait hold on")


def test_options_like_response_supports_natural_option_requests() -> None:
    assert is_options_like_response(Intent.UNKNOWN, "what are the options")
    assert is_options_like_response(Intent.UNKNOWN, "what can i choose")
    assert is_options_like_response(Intent.UNKNOWN, "tell me the choices")
    assert is_options_like_response(Intent.UNKNOWN, "available toppings")
