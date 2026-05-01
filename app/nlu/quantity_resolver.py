# app/nlu/quantity_resolver.py
"""Centralized quantity decision engine for the add-item flow.

Resolution order:
  1. Explicit extracted quantity (slot or regex) → use it.
  2. Leading vague expression ("some", "a few", "several") → ask clarification.
  3. No quantity information → default to 1.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

# Anchored to the start of the (item-name-stripped) text so that
# "burger with some modifications" does NOT trigger vague detection.
_LEADING_VAGUE_PATTERNS = (
    re.compile(r"^some\b", re.IGNORECASE),
    re.compile(r"^a few\b", re.IGNORECASE),
    re.compile(r"^several\b", re.IGNORECASE),
)


@dataclass(frozen=True, slots=True)
class QuantityResolution:
    quantity: int | None
    source: Literal["explicit", "implicit_default", "ambiguous"]
    needs_clarification: bool
    clarification_reason: str | None = None


class QuantityResolver:
    """Decide whether quantity needs clarification or can be defaulted."""

    def resolve(
        self,
        *,
        extracted: int | None,
        user_text: str,
    ) -> QuantityResolution:
        """
        Args:
            extracted: Quantity already parsed from slots/regex (None if absent).
            user_text: Item-name-stripped utterance fragment used to detect
                       leading vague expressions.
        """
        if isinstance(extracted, int) and extracted > 0:
            return QuantityResolution(
                quantity=extracted,
                source="explicit",
                needs_clarification=False,
            )

        normalized = (user_text or "").lower().strip()
        if any(p.search(normalized) for p in _LEADING_VAGUE_PATTERNS):
            return QuantityResolution(
                quantity=None,
                source="ambiguous",
                needs_clarification=True,
                clarification_reason="vague_quantity",
            )

        return QuantityResolution(
            quantity=1,
            source="implicit_default",
            needs_clarification=False,
        )
