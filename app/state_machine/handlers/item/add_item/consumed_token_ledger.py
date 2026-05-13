# app/state_machine/handlers/item/add_item/consumed_token_ledger.py
"""Turn-scoped ledger of tokens consumed by successful slot resolution.

When a slot (e.g. QUANTITY="one", ITEM="Spicy Tuna Roll") is resolved and
attributed to a structured field, its surface tokens must not later be reported
as unresolved entity feedback ("I couldn't find one.").

Usage
-----
Create one instance per prefill call, accumulate resolved surface forms, then
pass ``ledger.tokens()`` and ``ledger.consumed_phrases()`` to
``MultiGroupPrefillEngine.prefill()`` as the ``consumed_tokens`` and
``consumed_phrases`` arguments.

The ledger is intentionally thin — no business logic, just accumulation and
tokenization.
"""
from __future__ import annotations

from app.nlu.nlu_result import SlotValue
from app.nlu.query_normalization.text_preprocessor import normalize_text
from app.utils.token_matcher import tokenize


class ConsumedTokenLedger:
    """Accumulates tokens and phrases consumed during successful slot resolution."""

    __slots__ = ("_tokens", "_phrases")

    def __init__(self) -> None:
        self._tokens: set[str] = set()
        # Whole normalized phrases consumed during item resolution.  Used for
        # exact-phrase suppression so that ASR variants like "port stickers"
        # (resolved to "Pot Stickers") are not reported as unresolved entities.
        self._phrases: set[str] = set()

    # ------------------------------------------------------------------
    # Accumulators
    # ------------------------------------------------------------------

    def add_text(self, text: str, source: str = "") -> None:  # noqa: ARG002
        """Add all individual tokens from *text* as consumed."""
        normalized = normalize_text(text or "")
        if normalized:
            self._tokens.update(tokenize(normalized))

    def add_phrase(self, text: str, source: str = "") -> None:  # noqa: ARG002
        """Record a complete normalized phrase as consumed.

        Unlike ``add_text()``, this registers the whole phrase for exact-phrase
        lookup (``consumed_phrases()``) in addition to adding its individual
        tokens.  Use this when the surface form of an accepted match differs
        from the canonical item name (e.g. raw query "port stickers" accepted
        as "Pot Stickers").
        """
        normalized = normalize_text(text or "")
        if normalized:
            self._phrases.add(normalized)
            self._tokens.update(tokenize(normalized))

    def consume_match(
        self,
        *,
        raw_query: str,
        canonical_name: str,
        aliases: list[str] | tuple | None = None,
        voice_labels: list[str] | tuple | None = None,
        source: str = "item_match",
        score: float | None = None,  # noqa: ARG002  (stored for future logging)
        tier: str | None = None,  # noqa: ARG002    (stored for future logging)
    ) -> None:
        """Record a successful item-match acceptance.

        Consumes both the raw query phrase (e.g. "port stickers") and the
        canonical item name tokens (e.g. "pot stickers"), plus any alias/
        voice-label forms.  This prevents ASR-variant raw queries from later
        appearing in unresolved-entity feedback.
        """
        self.add_phrase(raw_query, source=source)
        self.add_text(canonical_name, source=f"{source}_canonical")
        for label in aliases or ():
            self.add_text(label, source=f"{source}_alias")
        for label in voice_labels or ():
            self.add_text(label, source=f"{source}_voice_label")

    def add_slot(self, slot: SlotValue, source: str = "") -> None:
        """Add the value of a SlotValue as consumed."""
        value = getattr(slot, "value", None)
        if isinstance(value, str):
            self.add_text(value, source=source)

    def add_quantity_value(self, value: object, source: str = "") -> None:
        """Add a resolved quantity in numeric string form as consumed.

        Call this with the *surface text* that produced the quantity (e.g.
        ``"one"``, ``"two dozen"``) so the word tokens are recorded.  Also
        call ``add_text(str(int_value))`` if you want the digit string ("1",
        "2") to be consumed as well.
        """
        if value is not None:
            self.add_text(str(value), source=source)

    def add_item_name(self, item_name: str) -> None:
        """Add item-name tokens as consumed."""
        self.add_text(item_name, source="item_name")

    def add_item_labels(
        self,
        aliases: list[str] | tuple,
        voice_labels: list[str] | tuple,
    ) -> None:
        """Add all alias and voice-label tokens as consumed."""
        for label in aliases or ():
            self.add_text(label, source="item_alias")
        for label in voice_labels or ():
            self.add_text(label, source="item_voice_label")

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def tokens(self) -> frozenset[str]:
        """Return the full set of consumed tokens as an immutable frozenset."""
        return frozenset(self._tokens)

    def consumed_phrases(self) -> frozenset[str]:
        """Return the set of whole normalized phrases consumed via add_phrase / consume_match."""
        return frozenset(self._phrases)

    def is_consumed_phrase(self, phrase: str) -> bool:
        """Return True if *phrase* is an exact consumed phrase OR every token was consumed."""
        normalized = normalize_text(phrase or "")
        if not normalized:
            return True
        if normalized in self._phrases:
            return True
        phrase_tokens = set(tokenize(normalized))
        return bool(phrase_tokens) and phrase_tokens.issubset(self._tokens)
