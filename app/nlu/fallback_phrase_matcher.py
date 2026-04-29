# app/nlu/fallback_phrase_matcher.py
"""Phrase-only fallback matcher for control intents.

Fires exclusively when NLU confidence is below threshold. Never import
or call directly from flow modules — use ControlDecisionService instead.
"""
from __future__ import annotations

from app.nlu.query_normalization.text_preprocessor import normalize_text

# Agent-request detection — mirrors legacy phase3_controls._AGENT_REQUEST_PHRASES
# and _STUCK_PHRASES. These are substring-scanned so single-word triggers like
# "agent" or "human" light up inside longer utterances.
_AGENT_REQUEST_SUBSTRINGS: tuple[str, ...] = (
    "agent",
    "operator",
    "person",
    "human",
    "team member",
    "representative",
    "customer service",
    "connect me to a person",
    "connect me to someone",
    "let me talk to someone",
    "i need a person",
    "i need an agent",
)

_STUCK_SUBSTRINGS: tuple[str, ...] = (
    "im stuck",
    "i am stuck",
    "this isnt working",
    "this is not working",
    "not working",
    "having trouble",
    "not helping",
)

# Quantity-correction detection — mirrors legacy flow_gate inline checks.
_QUANTITY_CORRECTION_SUBSTRINGS: tuple[str, ...] = ("instead of",)
_QUANTITY_CORRECTION_PREFIXES: tuple[str, ...] = (
    "make it ",
    "change it to ",
    "set it to ",
)


def _normalize(text: str) -> str:
    return normalize_text(text or "")


def _contains_any(normalized: str, phrases: tuple[str, ...]) -> bool:
    return any(p in normalized for p in phrases)


class FallbackPhraseMatcher:
    """Stateless phrase matcher for control-intent fallback.

    All methods accept raw or pre-normalized text and return bool.
    """

    def match_agent_request(self, text: str) -> bool:
        """True when *text* contains an agent-request or stuck phrase."""
        normalized = _normalize(text)
        if not normalized:
            return False
        return _contains_any(normalized, _AGENT_REQUEST_SUBSTRINGS) or _contains_any(
            normalized, _STUCK_SUBSTRINGS
        )

    def match_quantity_correction(self, text: str) -> bool:
        """True when *text* signals a quantity-correction utterance."""
        normalized = _normalize(text)
        if not normalized:
            return False
        return _contains_any(
            normalized, _QUANTITY_CORRECTION_SUBSTRINGS
        ) or any(normalized.startswith(p) for p in _QUANTITY_CORRECTION_PREFIXES)


# Module-level singleton — import this rather than constructing per call.
DEFAULT_MATCHER: FallbackPhraseMatcher = FallbackPhraseMatcher()
