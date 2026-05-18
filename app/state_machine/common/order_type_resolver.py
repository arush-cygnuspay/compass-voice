# app/state_machine/common/order_type_resolver.py
from __future__ import annotations

from dataclasses import dataclass

from app.nlu.query_normalization.text_preprocessor import normalize_text


@dataclass(frozen=True, slots=True)
class OrderTypeMatch:
    order_type: str
    matched_phrase: str
    remainder_text: str
    source: str = "lexical"


class OrderTypeResolver:
    """
    Deterministic lexical resolver for pickup / delivery.

    Design goals:
    - O(k) phrase checks over a very small constant set
    - longest-match-first to avoid partial collisions
    - usable both before and during normal NLU flow
    """

    _DELIVERY_PHRASES: tuple[str, ...] = (
        "for delivery",
        "delivery please",
        "it is delivery",
        "its delivery",
        "deliver it",
        "drop it off",
        "drop off",
        "dropoff",
        "delivery",
        "deliver",
        "send it",
        "bring it",
    )

    _PICKUP_PHRASES: tuple[str, ...] = (
        "ill pick it up",
        "for pickup",
        "pickup please",
        "it is pickup",
        "its pickup",
        "ill grab it",
        "come get it",
        "pick up",
        "carry out",
        "take out",
        "pickup",
        "carryout",
        "takeout",
        "collection",
    )

    _ORDERED_PHRASES: tuple[tuple[str, str], ...] = tuple(
        sorted(
            tuple(("delivery", phrase) for phrase in _DELIVERY_PHRASES)
            + tuple(("pickup", phrase) for phrase in _PICKUP_PHRASES),
            key=lambda x: len(x[1]),
            reverse=True,
        )
    )

    @classmethod
    def resolve(cls, text: str) -> OrderTypeMatch | None:
        normalized = normalize_text(text or "").strip()
        if not normalized:
            return None

        for order_type, phrase in cls._ORDERED_PHRASES:
            idx = normalized.find(phrase)
            if idx < 0:
                continue

            before = normalized[:idx].strip(" ,.-")
            after = normalized[idx + len(phrase):].strip(" ,.-")
            remainder = f"{before} {after}".strip()

            return OrderTypeMatch(
                order_type=order_type,
                matched_phrase=phrase,
                remainder_text=remainder,
            )

        return None