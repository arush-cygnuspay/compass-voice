# app/nlu/intent_resolution/confirmation_resolver.py
from __future__ import annotations

import os
from typing import Literal

from app.nlu.intent_resolution.intent import Intent
from app.nlu.nlu_result import NLUResult
from app.nlu.query_normalization.text_preprocessor import normalize_text

ConfirmationDecision = Literal["affirm", "deny", "cancel", "unknown"]

DEFAULT_CONFIRMATION_CONFIDENCE_THRESHOLD = float(
    os.getenv("COMPASS_CONFIRMATION_INTENT_CONFIDENCE_THRESHOLD", "0.55")
)

_LEADING_FILLERS: tuple[str, ...] = (
    "well",
    "so",
    "just",
    "please",
    "uh",
    "um",
    "hmm",
    # Align with control_intent_resolver so candidates like "yeah go ahead"
    # strip to "go ahead" before set-membership matching.
    "okay",
    "ok",
    "yeah",
    "yep",
    "yup",
    "yes",
)
_TRAILING_FILLERS: tuple[str, ...] = (
    "please",
    "thanks",
    "thank you",
)
_LEADING_NEGATION_PHRASES: tuple[str, ...] = (
    "no ",
    "nope ",
    "nah ",
)

_STRONG_AFFIRM_PHRASES: frozenset[str] = frozenset(
    {
        "yes",
        "yeah",
        "yep",
        "yup",
        "correct",
        "that is right",
        "thats right",
        "that is correct",
        "thats correct",
        "sounds good",
        "go ahead",
        "please do",
        "do it",
        "confirm",
        "confirm it",
        "place it",
        "proceed",
        "continue",
        "continue with it",
        "continue with that",
        "continue with checkout",
        "continue to checkout",
        "continue to payment",
        "please continue",
    }
)
_WEAK_AFFIRM_PHRASES: frozenset[str] = frozenset(
    {
        "ok",
        "okay",
        "sure",
        "all good",
        "that works",
    }
)
_DENY_PHRASES: frozenset[str] = frozenset(
    {
        "no",
        "nope",
        "nah",
        "not really",
        "wrong",
        "that is wrong",
        "thats wrong",
        "incorrect",
        "not correct",
        "that is not right",
        "thats not right",
        "not that",
        "no not that",
        "change it",
        "go back",
        "leave it",
        "keep it",
        "do not do it",
        "dont do it",
        "no thanks",
    }
)
_CANCEL_PHRASES: frozenset[str] = frozenset(
    {
        "cancel",
        "cancel that",
        "cancel it",
        "cancel order",
        "cancel the order",
        "stop",
        "hold on",
        "wait",
        "wait hold on",
        "not yet",
    }
)


def _signal_candidates(text: str) -> set[str]:
    normalized = normalize_text(text)
    if not normalized:
        return set()

    candidates: set[str] = {normalized}
    queue: list[str] = [normalized]
    seen: set[str] = set()

    while queue:
        value = queue.pop()
        if not value or value in seen:
            continue
        seen.add(value)
        candidates.add(value)

        for phrase in _LEADING_FILLERS:
            prefix = f"{phrase} "
            if value.startswith(prefix):
                queue.append(value[len(prefix):].strip())

        for prefix in _LEADING_NEGATION_PHRASES:
            if value.startswith(prefix):
                queue.append(value[len(prefix):].strip())

        for phrase in _TRAILING_FILLERS:
            suffix = f" {phrase}"
            if value.endswith(suffix):
                queue.append(value[: -len(suffix)].strip())

        if " and " in value:
            parts = [part.strip() for part in value.split(" and ") if part.strip()]
            queue.extend(parts)
            if len(parts) >= 2 and len(set(parts)) == 1:
                queue.append(parts[0])

    return candidates


def _extract_resolved_intent(
    nlu_result: NLUResult | None,
    resolved_intent: Intent | None,
    resolved_intent_confidence: float | None,
) -> tuple[Intent, float]:
    if nlu_result is not None:
        return nlu_result.effective_intent, float(nlu_result.intent_confidence)

    if resolved_intent is None:
        return Intent.UNKNOWN, 0.0

    if resolved_intent_confidence is not None:
        return resolved_intent, float(resolved_intent_confidence)

    if resolved_intent == Intent.UNKNOWN:
        return resolved_intent, 0.0

    return resolved_intent, 1.0


def _decision_from_classifier_label(
    label: str | None,
    *,
    expect_confirmation: bool,
) -> ConfirmationDecision:
    if not label:
        return "unknown"

    # Normalise to space-separated lowercase so both "cancel_order" and
    # "cancel order" reach the same comparison bucket.  Do NOT use
    # normalize_text here because that strips underscores via the punctuation
    # translation table, turning "cancel_order" into "cancelorder".
    normalized = label.lower().strip().replace("_", " ").replace("-", " ")
    normalized = " ".join(normalized.split())
    if not normalized:
        return "unknown"

    if normalized in {"affirm", "confirm"}:
        return "affirm"

    if expect_confirmation and normalized in {"confirm order"}:
        return "affirm"

    if normalized in {"deny"}:
        return "deny"

    if normalized in {"cancel", "cancel order"}:
        return "cancel"

    return "unknown"


def _fallback_phrase_decision(
    text: str,
    *,
    expect_confirmation: bool,
) -> ConfirmationDecision:
    candidates = _signal_candidates(text)
    if not candidates:
        return "unknown"

    if any(candidate in _CANCEL_PHRASES for candidate in candidates):
        return "cancel"

    if any(candidate in _DENY_PHRASES for candidate in candidates):
        return "deny"

    if any(candidate in _STRONG_AFFIRM_PHRASES for candidate in candidates):
        return "affirm"

    if expect_confirmation and any(
        candidate in _WEAK_AFFIRM_PHRASES for candidate in candidates
    ):
        return "affirm"

    return "unknown"


def resolve_confirmation_decision(
    nlu_result: NLUResult | None,
    text: str,
    *,
    resolved_intent: Intent | None = None,
    resolved_intent_confidence: float | None = None,
    expect_confirmation: bool = False,
    confidence_threshold: float = DEFAULT_CONFIRMATION_CONFIDENCE_THRESHOLD,
) -> ConfirmationDecision:
    intent, confidence = _extract_resolved_intent(
        nlu_result=nlu_result,
        resolved_intent=resolved_intent,
        resolved_intent_confidence=resolved_intent_confidence,
    )

    if confidence >= confidence_threshold:
        classifier_labels: tuple[str | None, ...] = (
            intent.value if isinstance(intent, Intent) else None,
            getattr(nlu_result, "model_sub_intent", None),
        )

        for label in classifier_labels:
            decision = _decision_from_classifier_label(
                label,
                expect_confirmation=expect_confirmation,
            )
            if decision != "unknown":
                return decision

        if intent not in {Intent.UNKNOWN, Intent.META_CLARIFY}:
            return "unknown"

    return _fallback_phrase_decision(
        text,
        expect_confirmation=expect_confirmation,
    )


def is_affirmation(
    nlu_result: NLUResult | None,
    text: str,
    *,
    resolved_intent: Intent | None = None,
    resolved_intent_confidence: float | None = None,
    expect_confirmation: bool = False,
    confidence_threshold: float = DEFAULT_CONFIRMATION_CONFIDENCE_THRESHOLD,
) -> bool:
    return (
        resolve_confirmation_decision(
            nlu_result,
            text,
            resolved_intent=resolved_intent,
            resolved_intent_confidence=resolved_intent_confidence,
            expect_confirmation=expect_confirmation,
            confidence_threshold=confidence_threshold,
        )
        == "affirm"
    )


def is_denial(
    nlu_result: NLUResult | None,
    text: str,
    *,
    resolved_intent: Intent | None = None,
    resolved_intent_confidence: float | None = None,
    expect_confirmation: bool = False,
    confidence_threshold: float = DEFAULT_CONFIRMATION_CONFIDENCE_THRESHOLD,
) -> bool:
    return (
        resolve_confirmation_decision(
            nlu_result,
            text,
            resolved_intent=resolved_intent,
            resolved_intent_confidence=resolved_intent_confidence,
            expect_confirmation=expect_confirmation,
            confidence_threshold=confidence_threshold,
        )
        == "deny"
    )
