# tests/nlu/test_order_scaffolding_expansion.py
"""Tests for Phase 4: expanded ORDER_FILLER_PREFIXES and ORDER_FILLER_TOKENS.

Validates:
- New prefixes are present in ORDER_FILLER_PREFIXES.
- New filler tokens are in ORDER_FILLER_TOKENS.
- Existing prefixes/tokens not accidentally removed.
- ORDER_FILLER_PREFIXES_SET is consistent with ORDER_FILLER_PREFIXES.
"""
from __future__ import annotations

import pytest

from app.nlu.order_scaffolding import (
    ORDER_FILLER_PREFIXES,
    ORDER_FILLER_PREFIXES_SET,
    ORDER_FILLER_TOKENS,
)


class TestNewPrefixesPresent:
    @pytest.mark.parametrize("prefix", [
        "i would like to order an ",
        "i would like to order a ",
        "i would like to order ",
        "i want to order an ",
        "i want to order a ",
        "i want to order ",
        "i want to add an ",
        "i want to add a ",
        "i want to add ",
        "can i please get an ",
        "can i please get a ",
        "can i please get ",
        "can i please have an ",
        "can i please have a ",
        "can i please have ",
        "can you add an ",
        "can you add a ",
        "can you add ",
        "can you please add an ",
        "can you please add a ",
        "can you please add ",
        "could i get an ",
        "could i get a ",
        "could i get ",
        "could i have an ",
        "could i have a ",
        "could i have ",
        "i also want an ",
        "i also want a ",
        "i also want ",
        "please add an ",
        "please add a ",
        "please add ",
        "also add an ",
        "also add a ",
        "also add ",
        "order an ",
        "order a ",
        "order ",
        "to order an ",
        "to order a ",
        "to order ",
    ])
    def test_new_prefix_in_list(self, prefix):
        assert prefix in ORDER_FILLER_PREFIXES, (
            f"Expected '{prefix}' in ORDER_FILLER_PREFIXES"
        )


class TestExistingPrefixesPreserved:
    @pytest.mark.parametrize("prefix", [
        "i want a ",
        "i want an ",
        "i want ",
        "i would like a ",
        "can i get a ",
        "can i have a ",
        "give me a ",
        "add a ",
        "add ",
        "get a ",
        "have a ",
        "a ",
        "an ",
        "the ",
    ])
    def test_existing_prefix_still_present(self, prefix):
        assert prefix in ORDER_FILLER_PREFIXES, (
            f"Existing prefix '{prefix}' was accidentally removed"
        )


class TestNewFillerTokens:
    @pytest.mark.parametrize("token", [
        "could",
        "wanna",
        "gonna",
        "lemme",
        "actually",
        "so",
        "yeah",
    ])
    def test_new_token_in_set(self, token):
        assert token in ORDER_FILLER_TOKENS, (
            f"Expected new filler token '{token}' in ORDER_FILLER_TOKENS"
        )


class TestExistingTokensPreserved:
    @pytest.mark.parametrize("token", [
        "i", "me", "a", "an", "the",
        "want", "would", "like", "need",
        "get", "have", "add", "order",
        "please", "and", "or", "to",
    ])
    def test_existing_token_still_present(self, token):
        assert token in ORDER_FILLER_TOKENS, (
            f"Existing filler token '{token}' was accidentally removed"
        )


class TestPrefixesSetConsistency:
    def test_set_matches_tuple(self):
        """ORDER_FILLER_PREFIXES_SET must equal frozenset(ORDER_FILLER_PREFIXES)."""
        assert ORDER_FILLER_PREFIXES_SET == frozenset(ORDER_FILLER_PREFIXES)

    def test_no_duplicates_in_prefixes(self):
        assert len(ORDER_FILLER_PREFIXES) == len(set(ORDER_FILLER_PREFIXES)), (
            "ORDER_FILLER_PREFIXES contains duplicates"
        )
