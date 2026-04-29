# app/nlu/control_decision_service.py
"""NLU-first control-intent resolution with confidence-gated phrase fallback.

ControlDecisionService is the single entry point for resolving agent-request
and quantity-correction control decisions. Resolution order:

1. NLU effective intent — accepted when confidence >= threshold.
2. Phrase fallback via FallbackPhraseMatcher — fires only when NLU is absent
   or below threshold.  Every fallback hit is logged as
   ``fallback_hit{source, text}`` for offline retraining.

Flow modules must consume ControlDecision objects; they must not call
FallbackPhraseMatcher or check raw text patterns directly.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from app.nlu.fallback_phrase_matcher import DEFAULT_MATCHER, FallbackPhraseMatcher
from app.nlu.intent_resolution.intent import Intent
from app.nlu.nlu_result import NLUResult
from app.nlu.query_normalization.text_preprocessor import normalize_text

logger = logging.getLogger(__name__)

DEFAULT_CONTROL_DECISION_CONFIDENCE_THRESHOLD: float = float(
    os.getenv("COMPASS_CONTROL_INTENT_CONFIDENCE_THRESHOLD", "0.55")
)


@dataclass(frozen=True, slots=True)
class ControlDecision:
    """Result of a single control-intent resolution attempt."""

    intent: Intent | None
    """Resolved control intent, or None when no signal was found."""

    confidence: float
    """NLU confidence for the resolved intent (0.0 when no NLU result available)."""

    used_fallback: bool
    """True when the decision came from phrase fallback, not NLU."""

    fallback_source: str | None = None
    """Identifies which phrase set triggered the fallback (for retraining logs)."""


class ControlDecisionService:
    """Resolves agent-request and quantity-correction control intents.

    Stateless after construction — safe to use as a module-level singleton.
    """

    def __init__(
        self,
        *,
        fallback_matcher: FallbackPhraseMatcher | None = None,
        confidence_threshold: float = DEFAULT_CONTROL_DECISION_CONFIDENCE_THRESHOLD,
    ) -> None:
        self._matcher = fallback_matcher or DEFAULT_MATCHER
        self._threshold = confidence_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve_agent_request(
        self,
        text: str,
        nlu_result: NLUResult | None = None,
    ) -> ControlDecision:
        """Resolve whether *text* is an agent-transfer request.

        Returns a ControlDecision with intent=Intent.REQUEST_AGENT on a positive
        signal, intent=None otherwise.
        """
        intent, confidence = self._intent_and_confidence(nlu_result)
        normalized = normalize_text(text or "")

        if intent == Intent.REQUEST_AGENT and confidence >= self._threshold:
            return ControlDecision(
                intent=Intent.REQUEST_AGENT,
                confidence=confidence,
                used_fallback=False,
            )

        if self._matcher.match_agent_request(normalized):
            self._emit_fallback("agent_request_phrase", normalized)
            return ControlDecision(
                intent=Intent.REQUEST_AGENT,
                confidence=confidence,
                used_fallback=True,
                fallback_source="agent_request_phrase",
            )

        return ControlDecision(intent=None, confidence=confidence, used_fallback=False)

    def resolve_quantity_correction(
        self,
        text: str,
        nlu_result: NLUResult | None = None,
    ) -> ControlDecision:
        """Resolve whether *text* is a prepayment quantity-correction request.

        Returns a ControlDecision with intent=Intent.CHANGE_QUANTITY on a
        positive signal, intent=None otherwise.
        """
        intent, confidence = self._intent_and_confidence(nlu_result)
        normalized = normalize_text(text or "")

        if intent == Intent.CHANGE_QUANTITY and confidence >= self._threshold:
            return ControlDecision(
                intent=Intent.CHANGE_QUANTITY,
                confidence=confidence,
                used_fallback=False,
            )

        if self._matcher.match_quantity_correction(normalized):
            self._emit_fallback("quantity_correction_phrase", normalized)
            return ControlDecision(
                intent=Intent.CHANGE_QUANTITY,
                confidence=confidence,
                used_fallback=True,
                fallback_source="quantity_correction_phrase",
            )

        return ControlDecision(intent=None, confidence=confidence, used_fallback=False)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _intent_and_confidence(
        nlu_result: NLUResult | None,
    ) -> tuple[Intent | None, float]:
        if nlu_result is None:
            return None, 0.0
        return nlu_result.effective_intent, float(nlu_result.intent_confidence)

    @staticmethod
    def _emit_fallback(source: str, text: str) -> None:
        logger.info(
            "fallback_hit",
            extra={"event_name": "fallback_hit", "source": source, "text": text},
        )


# Module-level singleton — import this; do not construct per-call.
DEFAULT_SERVICE: ControlDecisionService = ControlDecisionService()
