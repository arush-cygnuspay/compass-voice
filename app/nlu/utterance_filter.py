# app/nlu/utterance_filter.py
"""Ordering filler removal and unmatched-feedback sanitization.

FillerFilter centralizes two related concerns:
1. Stripping structural ordering filler ("i would like to order a") from
   raw utterances before menu matching.
2. Preventing filler-only or control-phrase-only text from being echoed
   back to the caller as "I couldn't find X".

IMPORTANT: The filler denylist contains only structural/conversational
scaffold words.  It must never contain menu nouns (burger, cheese, bun,
chicken, fries, etc.) — those must remain echable so callers hear feedback
on genuinely unavailable menu items.
"""
from __future__ import annotations

from typing import Iterable

from app.nlu.query_normalization.text_preprocessor import normalize_text


# ---------------------------------------------------------------------------
# Structural ordering filler phrases
#
# These are conversational scaffold phrases stripped before menu matching.
# Listed longest-first so the greedy prefix loop removes the longest match.
#
# RULE: Add ONLY functional/structural words here.
#       Never add food nouns, adjectives, or portion sizes.
# ---------------------------------------------------------------------------

_FILLER_PHRASES: tuple[str, ...] = (
    # Multi-word phrases — longest first
    "i would like to order",
    "i would like to have",
    "i would like to get",
    "i would like",
    "i want to order",
    "i want to have",
    "i want to get",
    "i want",
    "i will have",
    "ill have",
    "i would have",
    "could i get",
    "could i have",
    "may i have",
    "may i get",
    "can i get",
    "can i have",
    "let me get",
    "let me have",
    "give me",
    "then give me",
    "get me",
    "to order a",
    "to order an",
    "to order the",
    "to order",
    "to get a",
    "to get an",
    "to get",
    "to have",
    "order a",
    "order an",
    "order the",
    "okay then",
    "okay so",
    "okay",
    "ok then",
    "ok so",
    "ok",
    "alright",
    "all right",
    "please",
    "with a",
    "with an",
    "with the",
    "with some",
    "just a",
    "just an",
    "just the",
    "just",
    "and a",
    "and an",
    "and the",
    "and",
)

# Single stop-word tokens — used for the is_filler_only token check.
# These are structural/functional words only; no food nouns.
_FILLER_STOP_TOKENS: frozenset[str] = frozenset({
    "i", "me", "my", "am", "is", "are", "was", "were",
    "do", "did", "does", "be", "been", "have", "has",
    "okay", "ok", "alright", "then", "give", "want", "would", "like",
    "can", "get", "please", "just", "to", "order",
    "a", "an", "the", "and", "or", "for", "in", "at", "on", "it",
    "this", "that", "will", "shall", "may", "could", "should",
    "let", "so", "up", "with", "some",
    # Negation/quantifier function words — not menu-item candidates.
    # Note: "no" is intentionally excluded so "no cheese please" is still echable.
    "not", "any", "none",
})

# Known control / meta phrases that must never be echoed as unavailable-item
# feedback.  Primary defence is the ControlPhraseClassifier intercepting
# these at handler level; this set is the format-level backstop.
_NON_ECHABLE_CONTROL_PHRASES: frozenset[str] = frozenset({
    # Skip variants
    "skip", "skip that", "skip it",
    "no skip", "no skip that", "no skip it",
    "nah skip", "nah skip that",
    "leave it", "leave that", "leave it off",
    "dont add that", "dont add it", "dont want that",
    # Done variants
    "done", "add done", "no done",
    "thats all", "that is all", "thats it", "that is it",
    "all good", "all good then",
    "nothing else", "no more", "no nothing else",
    "im good", "i am good", "im done", "i am done",
    "finished", "im finished", "i am finished",
    # Repeat variants
    "repeat", "repeat that",
    "can you repeat", "can you repeat that",
    "say that again",
    "what are the options", "list options",
    "what options do you have", "what are my options",
    "what are the choices",
    # Checkout variants (belt-and-suspenders)
    "checkout", "check out", "place the order", "place my order",
    "confirm order",
    # Generic conversational residue
    "okay then",
    "okay then give me",
    "okay then give me a",
    "okay so",
    "ok then",
    "ok so",
})

# Pre-sort filler phrases longest-first for greedy stripping.
_SORTED_FILLERS: tuple[str, ...] = tuple(
    sorted(_FILLER_PHRASES, key=len, reverse=True)
)


# ---------------------------------------------------------------------------
# FillerFilter
# ---------------------------------------------------------------------------

class FillerFilter:
    """Centralizes ordering filler removal and prevents filler-only text
    from being echoed as unavailable-item feedback.

    Stateless — use the module-level ``DEFAULT_FILTER`` singleton.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def strip_ordering_filler(self, text: str) -> str:
        """Strip structural ordering filler prefix phrases from *text*.

        Only removes phrases from the structural filler denylist.
        Does NOT remove menu nouns or modifiers.

        Example::

            filter.strip_ordering_filler("i would like to order a chicken burger")
            # → "chicken burger"
        """
        normalized = normalize_text(text or "")
        if not normalized:
            return ""

        result = normalized
        changed = True
        while changed and result:
            changed = False
            for phrase in _SORTED_FILLERS:
                if result == phrase:
                    return ""
                if result.startswith(phrase + " "):
                    result = result[len(phrase):].strip()
                    changed = True
                    break

        # Strip a leading bare article left over after filler removal.
        # e.g. "i want a chicken burger" → strip "i want" → "a chicken burger"
        #      → strip leading "a " → "chicken burger"
        for article in ("a ", "an ", "the "):
            if result.startswith(article):
                result = result[len(article):]
                break

        return result

    def is_filler_only(self, text: str) -> bool:
        """Return True when *text* contains no genuine menu-item content.

        A value is filler-only when:
        - It normalizes to an empty string, OR
        - After stripping known structural phrases, nothing remains, OR
        - The remaining tokens are all functional stop-words (< 2 content
          tokens survive), OR
        - The normalized text is a known non-echable control phrase.

        Used to replace the old ``_has_echable_content`` token heuristic
        in ``format_utils._build_entity_feedback``.
        """
        normalized = normalize_text(text or "")
        if not normalized:
            return True

        # Known control / meta phrase — never echo these.
        if normalized in _NON_ECHABLE_CONTROL_PHRASES:
            return True

        # Try stripping multi-word filler phrases.
        remainder = self.strip_ordering_filler(normalized)
        if not remainder:
            return True

        # Token-level check.
        # Fast path: a single non-stop token is always a genuine menu candidate
        # ("rice", "avocado").  A lone stop-word ("just", "please") falls through
        # to the count check below.
        token_list = remainder.lower().split()
        if len(token_list) == 1 and token_list[0] not in _FILLER_STOP_TOKENS:
            return False

        # Multi-token check: fewer than 2 content tokens → filler.
        # This catches short STT fragments like "am i confused" or "my country is"
        # (only 1 content word each) while allowing "dragon burger" (2 content words).
        content_tokens = set(token_list) - _FILLER_STOP_TOKENS
        return len(content_tokens) < 2

    def strip_unmatched(self, values: Iterable[str]) -> list[str]:
        """Return only values with genuine menu-item content.

        Filters out filler-only strings from a resolver's ``unmatched_values``
        list.  Real unavailable candidates ("dragon burger", "bacon cheeseburger")
        are preserved; conversational residue ("to order a", "okay then") is
        dropped.
        """
        return [v for v in values if v and not self.is_filler_only(v)]


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

DEFAULT_FILTER: FillerFilter = FillerFilter()
