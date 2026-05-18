"""Unit tests for OrderTypeResolver — the single source of truth for
pickup / delivery lexical matching."""
import pytest

from app.state_machine.common.order_type_resolver import OrderTypeMatch, OrderTypeResolver


# ---------------------------------------------------------------------------
# Pickup phrases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "pickup",
    "pick up",
    "for pickup please",
    "take out",
    "takeout",
    "carry out",
    "carryout",
    "I'll pick it up",
    "ill pick it up",
    "I'll grab it",
    "ill grab it",
    "come get it",
    "  Pickup  ",
    "PICKUP",
    "Pick Up",
])
def test_pickup_phrases(text):
    result = OrderTypeResolver.resolve(text)
    assert result is not None
    assert result.order_type == "pickup"
    assert result.source == "lexical"


# ---------------------------------------------------------------------------
# Delivery phrases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "delivery",
    "for delivery",
    "delivery please",
    "deliver it",
    "drop it off",
    "drop off",
    "send it",
    "bring it",
    "DELIVERY",
    "Deliver It",
])
def test_delivery_phrases(text):
    result = OrderTypeResolver.resolve(text)
    assert result is not None
    assert result.order_type == "delivery"
    assert result.source == "lexical"


# ---------------------------------------------------------------------------
# No-match cases
# ---------------------------------------------------------------------------

def test_empty_text_returns_none():
    assert OrderTypeResolver.resolve("") is None


def test_whitespace_only_returns_none():
    assert OrderTypeResolver.resolve("   ") is None


def test_unrelated_text_returns_none():
    assert OrderTypeResolver.resolve("I want a burger") is None


def test_gibberish_returns_none():
    assert OrderTypeResolver.resolve("asdfgh qwerty") is None


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def test_extra_spaces_normalized():
    result = OrderTypeResolver.resolve("  pick    up  ")
    assert result is not None
    assert result.order_type == "pickup"


def test_mixed_case_normalized():
    result = OrderTypeResolver.resolve("PICK UP")
    assert result is not None
    assert result.order_type == "pickup"


def test_punctuation_stripped():
    result = OrderTypeResolver.resolve("pickup, please!")
    assert result is not None
    assert result.order_type == "pickup"


# ---------------------------------------------------------------------------
# OrderTypeMatch fields
# ---------------------------------------------------------------------------

def test_match_carries_matched_phrase():
    result = OrderTypeResolver.resolve("for delivery")
    assert result is not None
    assert result.matched_phrase == "for delivery"


def test_match_remainder_text_is_stripped():
    result = OrderTypeResolver.resolve("yeah for delivery thanks")
    assert result is not None
    assert result.order_type == "delivery"
    # remainder should contain the surrounding words, not the matched phrase
    assert "for delivery" not in result.remainder_text


def test_source_is_lexical():
    result = OrderTypeResolver.resolve("pickup")
    assert result is not None
    assert result.source == "lexical"


# ---------------------------------------------------------------------------
# Longest-match-first: "for pickup" wins over bare "pickup"
# ---------------------------------------------------------------------------

def test_longest_phrase_matched_for_pickup():
    result = OrderTypeResolver.resolve("for pickup")
    assert result is not None
    assert result.matched_phrase == "for pickup"


def test_longest_phrase_matched_for_delivery():
    result = OrderTypeResolver.resolve("for delivery")
    assert result is not None
    assert result.matched_phrase == "for delivery"


# ---------------------------------------------------------------------------
# Apostrophe normalization (STT transcribes "I'll" as "I'll" or "ill")
# ---------------------------------------------------------------------------

def test_ill_pick_it_up_with_apostrophe():
    result = OrderTypeResolver.resolve("I'll pick it up")
    assert result is not None
    assert result.order_type == "pickup"


def test_ill_grab_it_with_apostrophe():
    result = OrderTypeResolver.resolve("I'll grab it")
    assert result is not None
    assert result.order_type == "pickup"
