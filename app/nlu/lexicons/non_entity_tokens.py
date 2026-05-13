# app/nlu/lexicons/non_entity_tokens.py
"""Shared lexicon of tokens that are never standalone menu-item entities.

These tokens appear in user utterances as quantity expressions, structural
filler, or connector words.  They carry no menu-item signal and must never be
surfaced in "I couldn't find X." feedback.

Design constraints
------------------
BINDING WINS OVER FILTERING.  This lexicon is consulted ONLY after the
option-binding phase finishes.  If the menu has a legitimate option named
"Number Two Sauce" or "Two Piece Combo", it will bind before these filters
are consulted.  Only unbound, unresolved phrases are filtered here.
"""
from __future__ import annotations

from typing import Iterable

from app.nlu.matching.quantity_parser import NUMBER_WORDS, SPECIAL_QUANTITIES, UNIT_WORDS
from app.nlu.order_scaffolding import ORDER_FILLER_TOKENS
from app.nlu.query_normalization.text_preprocessor import normalize_text


# ---------------------------------------------------------------------------
# Sub-lexicons (each is a frozenset for O(1) membership tests)
# ---------------------------------------------------------------------------

# Digit-string tokens "0" … "99" — covers ASR numeric transcriptions.
_DIGIT_TOKENS: frozenset[str] = frozenset(str(i) for i in range(100))

# Connector / bridge tokens used during phrase construction.
_CONNECTOR_TOKENS: frozenset[str] = frozenset({
    "with", "and", "plus", "also", "or",
    "extra", "more", "double", "less", "light",
    "on", "the", "side",
    "a", "an",
    "no", "without", "hold", "remove",
})

# ---------------------------------------------------------------------------
# Master non-entity token set
# ---------------------------------------------------------------------------
NON_ENTITY_TOKENS: frozenset[str] = (
    frozenset(NUMBER_WORDS.keys())   # zero … ninety, hundred
    | frozenset(SPECIAL_QUANTITIES.keys())  # a, an, single, couple, pair …
    | frozenset(UNIT_WORDS)          # dozen, dozens, piece, pieces, pcs …
    | _DIGIT_TOKENS                  # "0" … "99"
    | ORDER_FILLER_TOKENS            # i, me, want, add, get, please …
    | _CONNECTOR_TOKENS              # with, and, no, extra …
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def normalize_token(text: str) -> str:
    """Lowercase + normalize whitespace using the shared text preprocessor."""
    return normalize_text(text or "").strip()


def tokenize_for_non_entity_filter(text: str) -> set[str]:
    """Return the set of individual normalized tokens in *text*."""
    normalized = normalize_token(text)
    if not normalized:
        return set()
    return set(normalized.split())


def is_non_entity_phrase(phrase: str) -> bool:
    """Return True when every token in *phrase* is a non-entity token.

    Examples::

        is_non_entity_phrase("one")         → True
        is_non_entity_phrase("two")         → True
        is_non_entity_phrase("a dozen")     → True
        is_non_entity_phrase("3")           → True
        is_non_entity_phrase("unicorn sauce") → False
        is_non_entity_phrase("mayo")        → False
    """
    tokens = tokenize_for_non_entity_filter(phrase)
    if not tokens:
        return True  # empty phrase has no entity content
    return tokens.issubset(NON_ENTITY_TOKENS)


def filter_unmatched_for_speech(
    values: Iterable[str],
    consumed_tokens: Iterable[str] | None = None,
) -> list[str]:
    """Filter *values* to only those that deserve "I couldn't find X." speech.

    Drops phrases where:
    - every token is a non-entity token (quantity / filler / connector), OR
    - every token is already covered by *consumed_tokens* (tokens attributed
      to successfully resolved slots / bindings).

    Returns only phrases that represent genuine unresolved menu entities.

    IMPORTANT: call this ONLY after option binding — never before.  Binding
    must win over filtering so that menu options named "Two Piece Combo" or
    "Number One Sauce" are not silently suppressed.
    """
    consumed: frozenset[str] = frozenset(consumed_tokens or ())
    result: list[str] = []
    seen: set[str] = set()

    for value in values:
        normalized = normalize_token(value)
        if not normalized or normalized in seen:
            continue
        tokens = set(normalized.split())
        # Drop pure non-entity phrases (quantity words, filler, connectors).
        if tokens.issubset(NON_ENTITY_TOKENS):
            continue
        # Drop phrases fully covered by already-consumed tokens.
        if consumed and tokens.issubset(consumed):
            continue
        seen.add(normalized)
        result.append(normalized)

    return result
