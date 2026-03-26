import pytest


class DummyContext:
    def __init__(self):
        self.waiting_for = "size"


def handle_unknown(ctx, user_text):
    if user_text not in ["small", "medium", "large"]:
        return "reprompt"
    return "accepted"


def test_unknown_response():
    ctx = DummyContext()

    result = handle_unknown(ctx, "what?")

    assert result == "reprompt"


def test_valid_response():
    ctx = DummyContext()

    result = handle_unknown(ctx, "medium")

    assert result == "accepted"