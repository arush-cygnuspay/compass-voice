# app/nlu/order_scaffolding.py
"""Centralised lexicon of ordering scaffolding / filler phrases.

These are the structural phrases callers say when ordering ("I want", "can I
get", "give me", etc.) that carry no menu-item signal and must be stripped
before feedback is generated or option phrases are mined.

Three public surfaces are exported:

ORDER_FILLER_PREFIXES
    Ordered tuple of full prefix strings to strip from the *start* of a
    normalised utterance (longest-first recommended — callers should iterate
    in order and strip the first match).  Used by normalize_item_request_text.

ORDER_FILLER_TOKENS
    Frozenset of individual tokens that appear inside scaffolding phrases.
    Used for per-token suppression in phrase miners and unresolved-feedback
    collapse logic.

ORDER_FILLER_PREFIXES_SET
    Frozenset of the same prefix strings (for O(1) membership tests when
    order doesn't matter).
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Full-phrase prefixes (strip in order from the start of an utterance)
# ---------------------------------------------------------------------------

ORDER_FILLER_PREFIXES: tuple[str, ...] = (
    # Longest multi-word forms first so they shadow shorter overlaps.
    "i said i want a ",
    "i said i want an ",
    "i said i want ",
    "i would like to have an ",
    "i would like to have a ",
    "i would like to have ",
    "i would like to add an ",
    "i would like to add a ",
    "i would like to add ",
    "i would like to order an ",
    "i would like to order a ",
    "i would like to get an ",
    "i would like to get a ",
    "i would like to get ",
    "i would like an ",
    "i would like a ",
    "i would like ",
    "i will take an ",
    "i will take a ",
    "i will take ",
    "i wanted to have an ",
    "i wanted to have a ",
    "i wanted to have ",
    "i wanted an ",
    "i wanted a ",
    "i wanted ",
    "i want to have an ",
    "i want to have a ",
    "i want to have ",
    "i want to order an ",
    "i want to order a ",
    "i want to get an ",
    "i want to get a ",
    "i want to get ",
    "i want an ",
    "i want a ",
    "i want ",
    "i need to have an ",
    "i need to have a ",
    "i need to have ",
    "i need an ",
    "i need a ",
    "i need ",
    "i needed an ",
    "i needed a ",
    "i needed ",
    "i said ",
    "can i get an ",
    "can i get a ",
    "can i get ",
    "can i have an ",
    "can i have a ",
    "can i have ",
    "let me get an ",
    "let me get a ",
    "let me get ",
    "let me have an ",
    "let me have a ",
    "let me have ",
    "give me an ",
    "give me a ",
    "give me ",
    "get me an ",
    "get me a ",
    "get me ",
    "add me an ",
    "add me a ",
    "add me ",
    "to add an ",
    "to add a ",
    "to add ",
    "to have an ",
    "to have a ",
    "to have ",
    "to order an ",
    "to order a ",
    "to order ",
    "i also want an ",
    "i also want a ",
    "i also want ",
    "i want to add an ",
    "i want to add a ",
    "i want to add ",
    "i want to order ",
    "i would like to order ",
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
    "please add an ",
    "please add a ",
    "please add ",
    "also add an ",
    "also add a ",
    "also add ",
    "order an ",
    "order a ",
    "order ",
    "add an ",
    "add a ",
    "add ",
    "get an ",
    "get a ",
    "get ",
    "have an ",
    "have a ",
    "have ",
    "having an ",
    "having a ",
    "having ",
    "bring me an ",
    "bring me a ",
    "bring me ",
    "bring an ",
    "bring a ",
    "bring ",
    "make it an ",
    "make it a ",
    "make it ",
    "wanted an ",
    "wanted a ",
    "wanted ",
    "needed an ",
    "needed a ",
    "needed ",
    "need an ",
    "need a ",
    "need ",
    "an ",
    "a ",
    "the ",
)

ORDER_FILLER_PREFIXES_SET: frozenset[str] = frozenset(ORDER_FILLER_PREFIXES)

# ---------------------------------------------------------------------------
# Individual filler tokens (for per-token suppression)
# ---------------------------------------------------------------------------

ORDER_FILLER_TOKENS: frozenset[str] = frozenset({
    # Pronouns / articles
    "i", "me", "my", "a", "an", "the",
    # Common verbs
    "want", "wanted", "would", "like",
    "need", "needed", "will", "ill",
    "take", "took",
    "get", "give",
    "have", "having", "had",
    "bring",
    "add",
    "order",
    "said",
    "let",
    "can",
    "make",
    # Connectors / particles
    "to", "for", "and", "or",
    "please", "just", "also",
    "okay", "ok", "um", "uh",
    "then",
    # Informal / colloquial
    "could",
    "wanna",
    "gonna",
    "lemme",
    "actually",
    "so",
    "yeah",
})
