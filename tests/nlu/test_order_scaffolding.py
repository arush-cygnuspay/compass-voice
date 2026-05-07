# tests/nlu/test_order_scaffolding.py
"""Phase C regression tests — filler/scaffolding phrases must not leak into feedback."""
import pytest

from app.nlu.order_scaffolding import ORDER_FILLER_PREFIXES, ORDER_FILLER_TOKENS
from app.state_machine.handlers.item.add_item.prefill_orchestrator import (
    normalize_item_request_text,
    PrefillOrchestrator,
)


class TestFillerPrefixStripping:
    """normalize_item_request_text must strip all scaffolding prefixes."""

    @pytest.mark.parametrize("utterance,expected", [
        ("i wanted chicken taco", "chicken taco"),
        ("i need a diet coke", "diet coke"),
        ("i would like to add all meat pizza", "all meat pizza"),
        ("i would like to have a lobster tail platter", "lobster tail platter"),
        ("i said i want a double bacon burger", "double bacon burger"),
        ("i will take double bacon burger", "double bacon burger"),
        ("can i get a sprite", "sprite"),
        ("give me a cheeseburger", "cheeseburger"),
        ("add a chicken taco", "chicken taco"),
        ("get me a burger", "burger"),
        ("let me get a coke", "coke"),
        ("i need a burger", "burger"),
        ("i needed a burger", "burger"),
        ("to add a pizza", "pizza"),
        ("to have a salad", "salad"),
        # Ensure real item names are preserved.
        ("chicken taco", "chicken taco"),
        ("double bacon burger", "double bacon burger"),
    ])
    def test_strips_prefix(self, utterance, expected):
        result = normalize_item_request_text(utterance)
        assert result == expected, f"Input: {utterance!r} → got {result!r}, expected {expected!r}"


class TestCollapseUnresolvedForFeedback:
    """Filler tokens must be removed so they are never surfaced as 'couldn't find ...'."""

    def _collapse(self, phrases: list[str]) -> list[str]:
        class _FakePending:
            item_name = "Chicken Taco"
            side_groups = []
            modifier_groups = []
            item_variants = []

        return PrefillOrchestrator._collapse_unresolved_for_feedback(phrases, pending=_FakePending())

    @pytest.mark.parametrize("phrase", [
        "wanted",
        "i wanted",
        "needed",
        "i needed",
        "need",
        "i need a",
        "said",
        "i said",
        "will",
        "i will",
        "take",
        "i will take",
        "to add",
        "to have",
        "to have a",
        "okay then give me a",
        "can i get",
        "give me",
    ])
    def test_pure_filler_is_suppressed(self, phrase):
        result = self._collapse([phrase])
        assert result == [], f"Expected {phrase!r} to be suppressed, got {result!r}"

    def test_real_content_passes_through(self):
        result = self._collapse(["pineapple"])
        assert result == ["pineapple"]

    def test_real_content_with_filler_prefix(self):
        # "pineapple" is real content even after stripping filler tokens.
        result = self._collapse(["i wanted pineapple"])
        # pineapple should survive since it's not a filler token.
        assert "pineapple" in " ".join(result)

    def test_duplicate_suppression(self):
        result = self._collapse(["wanted", "wanted", "needed"])
        assert result == []


class TestOrderFillerTokens:
    """Smoke tests on the shared token set."""

    @pytest.mark.parametrize("token", [
        "i", "want", "wanted", "need", "needed", "would", "like",
        "will", "take", "get", "give", "said", "let", "can", "add",
        "have", "having", "bring",
    ])
    def test_token_present(self, token):
        assert token in ORDER_FILLER_TOKENS, f"{token!r} should be in ORDER_FILLER_TOKENS"

    def test_real_menu_tokens_not_present(self):
        for token in ("pizza", "burger", "coke", "taco", "rice", "pineapple"):
            assert token not in ORDER_FILLER_TOKENS
