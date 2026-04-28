"""Tests for affirm/deny detection in the NLU linguistic_rules module."""
from __future__ import annotations

import pytest

from app.nlu.intent_resolution.intent import Intent
from app.nlu.linguistic_rules import is_affirm_like_response, is_deny_like_response


@pytest.mark.parametrize(
    "text",
    [
        "yes",
        "yeah",
        "yep",
        "yup",
        "correct",
        "ok",
        "okay",
        "sure",
        "sounds good",
        "go ahead",
        "confirm",
        "proceed",
        "continue",
        "place it",
        "checkout",
    ],
)
def test_is_affirm_like_response_core_words(text: str) -> None:
    assert is_affirm_like_response(None, text)


@pytest.mark.parametrize(
    "text",
    [
        "well yes",
        "uh yeah please",
        "yes thanks",
        "um okay thank you",
        "so sure",
    ],
)
def test_is_affirm_like_response_with_fillers(text: str) -> None:
    assert is_affirm_like_response(None, text)


def test_is_affirm_like_response_via_intent() -> None:
    assert is_affirm_like_response(Intent.CONFIRM, "")
    assert is_affirm_like_response(Intent.AFFIRM, "")


@pytest.mark.parametrize(
    "text",
    [
        "no",
        "nope",
        "nah",
        "wrong",
        "incorrect",
        "not correct",
        "stop",
        "cancel",
        "go back",
        "change it",
    ],
)
def test_is_deny_like_response_core_words(text: str) -> None:
    assert is_deny_like_response(None, text)


@pytest.mark.parametrize(
    "text",
    [
        "well no",
        "uh nope please",
        "no thanks",
    ],
)
def test_is_deny_like_response_with_fillers(text: str) -> None:
    assert is_deny_like_response(None, text)


def test_is_deny_like_response_via_intent() -> None:
    assert is_deny_like_response(Intent.DENY, "")


def test_affirm_and_deny_are_mutually_exclusive_for_clear_inputs() -> None:
    assert is_affirm_like_response(None, "yes")
    assert not is_deny_like_response(None, "yes")

    assert is_deny_like_response(None, "no")
    assert not is_affirm_like_response(None, "no")


def test_unknown_utterance_returns_false_for_both() -> None:
    assert not is_affirm_like_response(None, "a large coffee please")
    assert not is_deny_like_response(None, "a large coffee please")
