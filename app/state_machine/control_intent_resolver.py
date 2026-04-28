"""
Control intent resolution.

Resolution order is intent-first, state-aware, phrase-fallback:

1. NLU intent label  → ``_INTENT_RULES`` registry lookup.
2. NLU sub-intent label → same registry.
3. Inline state phrase rule (``continue``/``next`` in selection states).
4. Phrase fallback (``_AFFIRM_PHRASES`` / ``_DENY_PHRASES`` / etc.).

The registry (``_INTENT_RULES``) is the source of truth — adding a label or
scoping it to a state set is a one-line edit there. The phrase frozensets
below are kept as a safety net only; they fire just for utterances that the
NLU layer did not recognize. Every fallback hit emits ``phrase_fallback_used``
so we can monitor and drive fallback rate to zero over time.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from app.nlu.intent_resolution.intent import Intent
from app.nlu.nlu_result import NLUResult
from app.nlu.query_normalization.text_preprocessor import normalize_text
from app.state_machine.models.conversation_state import ConversationState

logger = logging.getLogger(__name__)

DEFAULT_CONTROL_INTENT_CONFIDENCE_THRESHOLD = float(
    os.getenv("COMPASS_CONTROL_INTENT_CONFIDENCE_THRESHOLD", "0.55")
)


class ControlIntentKind(str, Enum):
    AFFIRM = "affirm"
    DENY = "deny"
    OPTIONS_REQUEST = "options_request"
    CANCEL = "cancel"
    META_CLARIFY = "meta_clarify"
    DONE = "done"
    PAYMENT_STAY_ON_CALL = "payment_stay_on_call"
    PAYMENT_AFTER_CALL = "payment_after_call"
    PAYMENT_CANNOT_OPEN_LINK = "payment_cannot_open_link"
    SKIP = "skip"


@dataclass(frozen=True, slots=True)
class ResolvedControlIntent:
    kind: ControlIntentKind
    source: str
    normalized_text: str
    detected_intent: str | None = None
    detected_sub_intent: str | None = None
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class IntentRule:
    target_kind: ControlIntentKind
    states: frozenset[ConversationState] | None = None
    min_confidence: float = DEFAULT_CONTROL_INTENT_CONFIDENCE_THRESHOLD


_SELECTION_STATES: frozenset[ConversationState] = frozenset(
    {
        ConversationState.WAITING_FOR_MODIFIER,
        ConversationState.WAITING_FOR_SIDE,
        ConversationState.WAITING_FOR_SIDE_SIZE,
        ConversationState.WAITING_FOR_SIZE,
        ConversationState.WAITING_FOR_QUANTITY,
    }
)
_GROUP_SELECTION_STATES: frozenset[ConversationState] = frozenset(
    {
        ConversationState.WAITING_FOR_MODIFIER,
        ConversationState.WAITING_FOR_SIDE,
        ConversationState.WAITING_FOR_QUANTITY,
    }
)
_NEGATIVE_SLOT_STATES: frozenset[ConversationState] = frozenset(
    {
        ConversationState.WAITING_FOR_MODIFIER,
        ConversationState.WAITING_FOR_SIDE,
    }
)
_ORDER_CONFIRM_STATES: frozenset[ConversationState] = frozenset(
    {ConversationState.CONFIRMING_ORDER}
)
_PAYMENT_STATES: frozenset[ConversationState] = frozenset(
    {
        ConversationState.WAITING_FOR_PAYMENT,
        ConversationState.WAITING_FOR_CHECKOUT_COMPLETION,
    }
)
_OPTIONAL_FIELD_STATES: frozenset[ConversationState] = frozenset(
    {ConversationState.WAITING_FOR_DELIVERY_ADDRESS_COLLECTION}
)

def _canonical_label(value: Intent | str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, Intent):
        return value.value
    text = str(value).strip().lower()
    if not text:
        return None
    return text.replace("-", "_").replace(" ", "_")


def _build_intent_rules() -> dict[str, tuple[IntentRule, ...]]:
    raw: dict[str, tuple[IntentRule, ...]] = {
        "affirm": (IntentRule(ControlIntentKind.AFFIRM),),
        "confirm": (IntentRule(ControlIntentKind.AFFIRM),),
        "confirm_order": (IntentRule(ControlIntentKind.AFFIRM),),
        "deny": (IntentRule(ControlIntentKind.DENY),),
        "cancel": (IntentRule(ControlIntentKind.CANCEL),),
        "cancel_order": (IntentRule(ControlIntentKind.CANCEL),),
        "meta_clarify": (IntentRule(ControlIntentKind.META_CLARIFY),),
        "ask_options": (IntentRule(ControlIntentKind.OPTIONS_REQUEST),),
        "options_request": (IntentRule(ControlIntentKind.OPTIONS_REQUEST),),
        "end_adding": (IntentRule(ControlIntentKind.DONE),),
        "finish_order": (IntentRule(ControlIntentKind.DONE),),
        "list_options": (IntentRule(ControlIntentKind.OPTIONS_REQUEST, _SELECTION_STATES),),
        "list_items": (IntentRule(ControlIntentKind.OPTIONS_REQUEST, _SELECTION_STATES),),
        "show_options": (IntentRule(ControlIntentKind.OPTIONS_REQUEST, _SELECTION_STATES),),
        "show_menu": (IntentRule(ControlIntentKind.OPTIONS_REQUEST, _SELECTION_STATES),),
        "ask_menu_info": (IntentRule(ControlIntentKind.OPTIONS_REQUEST, _SELECTION_STATES),),
        "what_can_i_get": (IntentRule(ControlIntentKind.OPTIONS_REQUEST, _SELECTION_STATES),),
        "browse_menu": (IntentRule(ControlIntentKind.OPTIONS_REQUEST, _SELECTION_STATES),),
        "browse_category": (IntentRule(ControlIntentKind.OPTIONS_REQUEST, _SELECTION_STATES),),
        "availability_query": (IntentRule(ControlIntentKind.OPTIONS_REQUEST, _SELECTION_STATES),),
        "recommendation_query": (IntentRule(ControlIntentKind.OPTIONS_REQUEST, _SELECTION_STATES),),
        "ask_item_info": (IntentRule(ControlIntentKind.OPTIONS_REQUEST, _SELECTION_STATES),),
        "checkout": (IntentRule(ControlIntentKind.DONE, _SELECTION_STATES),),
        "review_order": (IntentRule(ControlIntentKind.DONE, _SELECTION_STATES),),
        "payment_request": (IntentRule(ControlIntentKind.AFFIRM, _ORDER_CONFIRM_STATES),),
        "stay_on_call": (IntentRule(ControlIntentKind.PAYMENT_STAY_ON_CALL, _PAYMENT_STATES),),
        "after_call": (IntentRule(ControlIntentKind.PAYMENT_AFTER_CALL, _PAYMENT_STATES),),
        "cannot_open_link": (IntentRule(ControlIntentKind.PAYMENT_CANNOT_OPEN_LINK, _PAYMENT_STATES),),
        "skip": (IntentRule(ControlIntentKind.SKIP, _OPTIONAL_FIELD_STATES),),
    }
    canonical: dict[str, tuple[IntentRule, ...]] = {}
    for label, rules in raw.items():
        key = _canonical_label(label)
        if key is None:
            continue
        canonical[key] = rules
    return canonical


_INTENT_RULES: dict[str, tuple[IntentRule, ...]] = _build_intent_rules()


# ---------------------------------------------------------------------------
# Phrase frozensets — deprecated safety net.
#
# The intent registry above is the source of truth for control intent
# resolution. The frozensets below remain only to catch utterances the NLU
# layer did not recognize. Do NOT add new control vocabulary here — register
# the corresponding NLU label in ``_INTENT_RULES`` instead. Every match emits
# ``phrase_fallback_used`` so fallback usage is visible in production.
# ---------------------------------------------------------------------------
_AFFIRM_PHRASES: frozenset[str] = frozenset(
    {
        "yes",
        "yeah",
        "yep",
        "yup",
        "correct",
        "thats right",
        "that is right",
        "thats correct",
        "that is correct",
        "sounds good",
        "that works",
        "thatll work",
        "that will work",
        "that is fine",
        "thats fine",
        "confirm",
        "confirm it",
        "i confirm",
        "proceed",
        "continue",
        "go ahead",
        "please continue",
        "please proceed",
        "all good",
        "looks good",
        "works for me",
    }
)
_DENY_PHRASES: frozenset[str] = frozenset(
    {
        "no",
        "nope",
        "nah",
        "no thanks",
        "no thank you",
        "not that",
        "not that one",
        "not correct",
        "incorrect",
        "thats not right",
        "that is not right",
        "thats wrong",
        "that is wrong",
        "change it",
        "i dont want that",
        "i do not want that",
        "i dont want any",
        "i do not want any",
        "no i dont want any",
        "no i do not want any",
        "ill pass",
        "leave it",
        "skip it",
        "none",
        "nothing",
    }
)
_DONE_PHRASES: frozenset[str] = frozenset(
    {
        "done",
        "im done",
        "i am done",
        "thats it",
        "that is it",
        "thats all",
        "that is all",
        "no more",
        "nothing else",
        "all set",
        "good",
        "thats good",
        "that is good",
    }
)
_OPTIONS_PHRASES: frozenset[str] = frozenset(
    {
        "options",
        "what are the options",
        "what are my options",
        "what can i choose",
        "what do you have",
        "tell me the options",
        "tell me the choices",
        "list the choices",
        "list the options",
        "available toppings",
        "what cheese do you have",
        "what sides do you have",
        "what sizes do you have",
    }
)
_CANCEL_PHRASES: frozenset[str] = frozenset(
    {
        "cancel",
        "cancel that",
        "cancel it",
        "never mind",
        "start over",
        "restart",
        "restart this item",
        "restart the item",
        "forget it",
        "scratch that",
        "remove that",
    }
)
_META_CLARIFY_PHRASES: frozenset[str] = frozenset(
    {
        "what do you mean",
        "i dont understand",
        "i do not understand",
        "repeat that",
        "say that again",
        "what was that",
        "can you explain",
        "help",
    }
)

# ---------------------------------------------------------------------------
# State-gated phrase tuples for payment-mode utterances. Substring-matched
# (to preserve parity with the legacy phase3_controls._contains_any path)
# and only consulted when current_state ∈ _PAYMENT_STATES. These remain
# deprecated safety nets — register the corresponding NLU label in
# ``_INTENT_RULES`` instead.
# ---------------------------------------------------------------------------
_STAY_ON_CALL_PHRASES: tuple[str, ...] = (
    "stay on the line",
    "stay on line",
    "stay with me",
    "stay here",
    "stay on the call",
    "wait on the line",
    "hold on the line",
)
_AFTER_CALL_PHRASES: tuple[str, ...] = (
    "after the call",
    "after this call",
    "i will do it after the call",
    "ill do it after the call",
    "i will complete it after the call",
    "ill complete it after the call",
    "i will do it later",
    "ill do it later",
    "i'll do it later",
    "later",
    "not now",
)
_CANNOT_OPEN_LINK_PHRASES: tuple[str, ...] = (
    "cant open the link",
    "cannot open the link",
    "cant open the message",
    "cannot open the message",
    "cant open the sms",
    "cannot open the sms",
    "i cant open the link",
    "i cannot open the link",
    "i cant open the message",
    "i cannot open the message",
    "phone is to my ear",
    "while im on the phone",
    "while i am on the phone",
)

# Skip phrases for optional fields (e.g. apt/suite). Candidate-matched
# for parity with the legacy semantic_signals.OPTIONAL_SKIP_WORDS path
# in ``is_optional_skip_response``. Only consulted when current_state ∈
# _OPTIONAL_FIELD_STATES. Handler-specific phrasing like "no apartment"
# / "no suite" is intentionally NOT included here — those are checked
# at the handler call site against its own ``OPTIONAL_NONE_WORDS`` set
# to preserve the exact-match behavior of the legacy code path.
_SKIP_PHRASES: frozenset[str] = frozenset(
    {
        "none",
        "nothing",
        "skip",
        "skip it",
        "no thanks",
        "nothing else",
        "no more",
        "that is all",
        "thats all",
        "im good",
        "i am good",
    }
)

_OPTION_HELP_LABELS_SELECTION_GATED: frozenset[str] = frozenset(
    {
        "browse_menu",
        "browse_category",
        "availability_query",
        "recommendation_query",
        "ask_item_info",
        "show_menu",
        "ask_menu_info",
        "list_options",
        "list_items",
        "show_options",
        "what_can_i_get",
    }
)

_OPTION_HELP_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^(options|choices|menu|available)$"),
    re.compile(r"^available\s+(options|choices|toppings|sides|sizes|drinks|items|cheeses|extras)\b"),
    re.compile(r"^any\s+options(\s+available)?$"),
    re.compile(
        r"^what(s|\s+is|\s+are)?\s+("
        r"available"
        r"|(the|my)\s+(options|choices)"
        r"|i\s+can\s+(get|choose|order|have|pick)"
        r"|(else\s+)?(do|did|can)\s+(you|i)\s+(have|offer|got|get|choose|pick)"
        r"|choices\s+do\s+i\s+have"
        r"|sizes|sides|toppings|drinks|cheeses|extras"
        r")\b"
    ),
    re.compile(
        r"^(list|show(\s+me)?|read|give\s+me|tell\s+me)\s+(the\s+|me\s+the\s+)?("
        r"options|choices|them|available|menu|sizes|sides|toppings|drinks|cheeses|extras"
        r")\b"
    ),
    re.compile(r"^which\s+(options|choices|ones|can|sizes|sides|toppings|drinks|cheeses|extras)\b"),
)


def _matches_option_help_pattern(normalized_text: str) -> bool:
    if not normalized_text:
        return False
    for pattern in _OPTION_HELP_PATTERNS:
        if pattern.match(normalized_text):
            return True
    return False


_LEADING_FILLERS: tuple[str, ...] = (
    "well",
    "so",
    "just",
    "please",
    "uh",
    "um",
    "hmm",
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


def resolve_control_intent(
    transcript: str,
    detected_intent: Intent | str | None,
    detected_sub_intent: str | None,
    current_state: ConversationState | str,
    pending_context: Any | None = None,
    *,
    nlu_result: NLUResult | None = None,
    intent_confidence: float | None = None,
    confidence_threshold: float = DEFAULT_CONTROL_INTENT_CONFIDENCE_THRESHOLD,
) -> ResolvedControlIntent | None:
    del pending_context

    state = _normalize_state(current_state)
    if state is None:
        return None

    # Preserve pre-registry guard: in modifier/side states, ``no <slot>`` and
    # ``without <slot>`` must reach the slot extractor, even when NLU emits a
    # DENY-shaped intent. Only states in ``_NEGATIVE_SLOT_STATES`` pay the
    # normalization cost here.
    if state in _NEGATIVE_SLOT_STATES:
        early_normalized = normalize_text(transcript or "")
        if early_normalized and _looks_like_specific_negative_slot_instruction(
            early_normalized, state
        ):
            return None

    # Preserve legacy precedence: in payment states the substring scan
    # beats NLU classification (legacy ``detect_payment_wait_mode_choice``
    # ran before any control-intent resolution). NLU does not currently
    # emit ``stay_on_call`` / ``after_call`` / ``cannot_open_link``
    # labels, so the registry path for these never fires today; this
    # early check is the authoritative source. The registry entries
    # added under ``_INTENT_RULES`` future-proof the path for when the
    # NLU layer learns these labels.
    if state in _PAYMENT_STATES:
        early_normalized = normalize_text(transcript or "")
        payment_kind = _early_payment_phrase_kind(early_normalized)
        if payment_kind is not None:
            log_control_intent_event(
                "phrase_fallback_used",
                state=state.value,
                kind=payment_kind.value,
                source="payment_state_phrase",
                normalized_text=early_normalized,
            )
            return ResolvedControlIntent(
                kind=payment_kind,
                source="payment_state_phrase",
                normalized_text=early_normalized,
            )

    effective_intent: Intent | str | None = detected_intent
    sub_intent: str | None = detected_sub_intent
    confidence: float | None = intent_confidence
    if nlu_result is not None:
        effective_intent = nlu_result.effective_intent
        confidence = float(nlu_result.intent_confidence)
        sub_intent = sub_intent or nlu_result.model_sub_intent

    if confidence is None:
        if isinstance(effective_intent, Intent) and effective_intent != Intent.UNKNOWN:
            confidence = 1.0
        else:
            confidence = 0.0

    intent_label = _canonical_label(effective_intent)
    sub_intent_label = _canonical_label(sub_intent)

    # ── Step 1: primary NLU intent path (registry lookup) ──────────────────
    primary_kind = _resolve_from_registry(
        intent_label, state, confidence, confidence_threshold
    )
    if primary_kind is not None:
        log_control_intent_event(
            "intent_registry_resolved",
            state=state.value,
            label=intent_label,
            source="intent_registry",
            confidence=confidence,
            target_kind=primary_kind.value,
        )
        return ResolvedControlIntent(
            kind=primary_kind,
            source="intent_registry",
            normalized_text=normalize_text(transcript or ""),
            detected_intent=intent_label,
            detected_sub_intent=sub_intent_label,
            confidence=confidence,
        )

    # ── Step 2: sub-intent registry lookup ─────────────────────────────────
    sub_kind = _resolve_from_registry(
        sub_intent_label, state, confidence, confidence_threshold
    )
    if sub_kind is not None:
        log_control_intent_event(
            "intent_registry_resolved",
            state=state.value,
            label=sub_intent_label,
            source="sub_intent_registry",
            confidence=confidence,
            target_kind=sub_kind.value,
        )
        return ResolvedControlIntent(
            kind=sub_kind,
            source="sub_intent_registry",
            normalized_text=normalize_text(transcript or ""),
            detected_intent=intent_label,
            detected_sub_intent=sub_intent_label,
            confidence=confidence,
        )

    # ── From here on we need the normalized utterance for safety nets ─────
    normalized_text = normalize_text(transcript or "")
    if not normalized_text:
        return None

    if _looks_like_specific_negative_slot_instruction(normalized_text, state):
        return None

    # ── Step 3: inline state phrase rule ───────────────────────────────────
    if state in _GROUP_SELECTION_STATES and normalized_text in {"continue", "next"}:
        log_control_intent_event(
            "control_intent_detected",
            state=state.value,
            kind=ControlIntentKind.DONE.value,
            source="state_phrase",
            normalized_text=normalized_text,
        )
        return ResolvedControlIntent(
            kind=ControlIntentKind.DONE,
            source="state_phrase",
            normalized_text=normalized_text,
        )

    # ── Step 4: phrase fallback (deprecated safety net) ────────────────────
    fallback_match = _resolve_from_phrase_fallback(
        normalized_text=normalized_text,
        current_state=state,
    )
    if fallback_match is not None:
        log_control_intent_event(
            "phrase_fallback_used",
            state=state.value,
            kind=fallback_match.kind.value,
            source=fallback_match.source,
            normalized_text=normalized_text,
        )
        return fallback_match

    return None


def control_intent_to_confirmation_decision(
    control_intent: ResolvedControlIntent | None,
) -> str:
    if control_intent is None:
        return "unknown"
    if control_intent.kind == ControlIntentKind.AFFIRM:
        return "affirm"
    if control_intent.kind == ControlIntentKind.DENY:
        return "deny"
    if control_intent.kind == ControlIntentKind.CANCEL:
        return "cancel"
    return "unknown"


def log_control_intent_event(event_name: str, **data: Any) -> None:
    logger.info(event_name, extra={"event_name": event_name, **data})


def _resolve_from_registry(
    label: str | None,
    current_state: ConversationState,
    confidence: float | None,
    confidence_threshold: float = DEFAULT_CONTROL_INTENT_CONFIDENCE_THRESHOLD,
) -> ControlIntentKind | None:
    if not label:
        return None
    rules = _INTENT_RULES.get(label)
    if not rules:
        return None
    conf = float(confidence) if confidence is not None else 1.0
    for rule in rules:
        if rule.states is not None and current_state not in rule.states:
            continue
        threshold = rule.min_confidence
        if confidence_threshold > threshold:
            threshold = confidence_threshold
        if conf < threshold:
            continue
        return rule.target_kind
    return None


def _resolve_from_phrase_fallback(
    *,
    normalized_text: str,
    current_state: ConversationState,
) -> ResolvedControlIntent | None:
    candidates = _signal_candidates(normalized_text)
    if not candidates:
        return None

    # ── State-gated payment-mode phrases (substring match, parity with
    # the legacy phase3_controls._contains_any). Must run before the
    # generic CANCEL/DENY/AFFIRM matchers because some phrases overlap
    # ("not now" → DENY in selection states, but PAYMENT_AFTER_CALL in
    # payment states). Substring-matched against normalized_text only,
    # not the candidate set, since the legacy matcher did a substring
    # scan on the whole utterance.
    if current_state in _PAYMENT_STATES:
        if _contains_any_phrase(normalized_text, _STAY_ON_CALL_PHRASES):
            return ResolvedControlIntent(
                kind=ControlIntentKind.PAYMENT_STAY_ON_CALL,
                source="phrase_fallback",
                normalized_text=normalized_text,
            )
        if _contains_any_phrase(normalized_text, _AFTER_CALL_PHRASES):
            return ResolvedControlIntent(
                kind=ControlIntentKind.PAYMENT_AFTER_CALL,
                source="phrase_fallback",
                normalized_text=normalized_text,
            )
        if _contains_any_phrase(normalized_text, _CANNOT_OPEN_LINK_PHRASES):
            return ResolvedControlIntent(
                kind=ControlIntentKind.PAYMENT_CANNOT_OPEN_LINK,
                source="phrase_fallback",
                normalized_text=normalized_text,
            )

    # ── State-gated SKIP fallback for optional fields. Candidate-matched
    # for parity with the legacy semantic_signals.is_optional_skip_response.
    if current_state in _OPTIONAL_FIELD_STATES and _candidate_matches(
        candidates, _SKIP_PHRASES
    ):
        return ResolvedControlIntent(
            kind=ControlIntentKind.SKIP,
            source="phrase_fallback",
            normalized_text=normalized_text,
        )

    if _candidate_matches(candidates, _META_CLARIFY_PHRASES):
        return ResolvedControlIntent(
            kind=ControlIntentKind.META_CLARIFY,
            source="phrase_fallback",
            normalized_text=normalized_text,
        )

    if _candidate_matches(candidates, _OPTIONS_PHRASES):
        return ResolvedControlIntent(
            kind=ControlIntentKind.OPTIONS_REQUEST,
            source="phrase_fallback",
            normalized_text=normalized_text,
        )

    if current_state in _SELECTION_STATES and any(
        _matches_option_help_pattern(candidate) for candidate in candidates if candidate
    ):
        return ResolvedControlIntent(
            kind=ControlIntentKind.OPTIONS_REQUEST,
            source="recognizer_extended",
            normalized_text=normalized_text,
        )

    if _candidate_matches(candidates, _CANCEL_PHRASES):
        return ResolvedControlIntent(
            kind=ControlIntentKind.CANCEL,
            source="phrase_fallback",
            normalized_text=normalized_text,
        )

    if _candidate_matches(candidates, _DONE_PHRASES):
        return ResolvedControlIntent(
            kind=ControlIntentKind.DONE,
            source="phrase_fallback",
            normalized_text=normalized_text,
        )

    if _candidate_matches(candidates, _DENY_PHRASES):
        return ResolvedControlIntent(
            kind=ControlIntentKind.DENY,
            source="phrase_fallback",
            normalized_text=normalized_text,
        )

    if _candidate_matches(candidates, _AFFIRM_PHRASES):
        return ResolvedControlIntent(
            kind=ControlIntentKind.AFFIRM,
            source="phrase_fallback",
            normalized_text=normalized_text,
        )

    return None


def _map_classifier_label_to_kind(
    label: str,
    current_state: ConversationState,
) -> ControlIntentKind | None:
    """Backward-compat shim. Prefer ``_resolve_from_registry``.

    Confidence is unknown here, so the registry is queried as if the label
    had max confidence — preserving the pre-registry behavior of this helper.
    """
    return _resolve_from_registry(_canonical_label(label), current_state, None)


def _normalize_state(value: ConversationState | str) -> ConversationState | None:
    if isinstance(value, ConversationState):
        return value
    try:
        return ConversationState(str(value))
    except ValueError:
        return None


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

        for filler in _LEADING_FILLERS:
            prefix = f"{filler} "
            if value.startswith(prefix):
                queue.append(value[len(prefix):].strip())

        for filler in _TRAILING_FILLERS:
            suffix = f" {filler}"
            if value.endswith(suffix):
                queue.append(value[: -len(suffix)].strip())

        if " and " in value:
            queue.extend(part.strip() for part in value.split(" and ") if part.strip())

    return candidates


def _candidate_matches(candidates: set[str], phrases: frozenset[str]) -> bool:
    return any(candidate in phrases for candidate in candidates if candidate)


def _contains_any_phrase(normalized_text: str, phrases: tuple[str, ...]) -> bool:
    """Substring scan over the full normalized utterance.

    Used for state-gated payment-mode phrase fallback to preserve parity
    with the legacy ``phase3_controls._contains_any`` matcher.
    """
    if not normalized_text:
        return False
    return any(phrase in normalized_text for phrase in phrases)


def _early_payment_phrase_kind(
    normalized_text: str,
) -> ControlIntentKind | None:
    """Map a normalized payment-state utterance to a payment-mode kind.

    Mirrors legacy ``phase3_controls.detect_payment_wait_mode_choice``
    precedence (stay → after → cannot_open_link). Returns ``None`` when
    no payment-mode phrase matches.
    """
    if not normalized_text:
        return None
    if _contains_any_phrase(normalized_text, _STAY_ON_CALL_PHRASES):
        return ControlIntentKind.PAYMENT_STAY_ON_CALL
    if _contains_any_phrase(normalized_text, _AFTER_CALL_PHRASES):
        return ControlIntentKind.PAYMENT_AFTER_CALL
    if _contains_any_phrase(normalized_text, _CANNOT_OPEN_LINK_PHRASES):
        return ControlIntentKind.PAYMENT_CANNOT_OPEN_LINK
    return None


def _looks_like_specific_negative_slot_instruction(
    normalized_text: str,
    current_state: ConversationState,
) -> bool:
    if current_state not in _NEGATIVE_SLOT_STATES:
        return False

    if normalized_text.startswith("without "):
        return True

    if not normalized_text.startswith("no "):
        return False

    if normalized_text in _DENY_PHRASES or normalized_text in _DONE_PHRASES:
        return False

    remainder = normalized_text[3:].strip()
    if not remainder:
        return False

    if remainder.startswith(
        (
            "thank",
            "thats all",
            "that is all",
            "thats it",
            "that is it",
            "more",
            "not that",
            "thats not",
            "that is not",
        )
    ):
        return False

    return True


