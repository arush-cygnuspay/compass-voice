# app/realtime/barge_in_policy.py
from __future__ import annotations

import re
import time
from dataclasses import dataclass

from app.nlu.query_normalization.text_preprocessor import normalize_text, normalize_waiting_value
from app.session.session import Session
from app.state_machine.flow_sets import (
    looks_like_done_answer,
    looks_like_more_options_answer,
    looks_like_skip_answer,
)
from app.state_machine.handlers.common.preorder_redirect_utils import (
    looks_like_ordering_request,
)
from app.state_machine.common.order_type_resolver import OrderTypeResolver
from app.state_machine.models.conversation_state import ConversationState
from app.utils.quantity_detection import normalize_quantity

_YES_LIKE = (
    "yes",
    "yeah",
    "yep",
    "yup",
    "correct",
    "right",
    "thats right",
    "yes it is",
    "that is correct",
    "thats correct",
    "go ahead",
    "proceed",
    "continue",
)

_NO_LIKE = (
    "no",
    "nope",
    "nah",
    "wrong",
    "incorrect",
    "cancel",
    "stop",
    "not now",
)

_CONTROL_PHRASES = (
    "cancel",
    "cancel order",
    "show cart",
    "cart",
    "show total",
    "total",
    "checkout",
    "check out",
    "pay",
    "payment",
    "show menu",
    "menu",
    "price",
    "how much",
)

# Single filler/noise tokens that carry no informational content.
_FILLER_WORDS: frozenset[str] = frozenset({
    "uh", "um", "hmm", "uhh", "mm", "mhm", "huh", "ah", "oh",
    "uhm", "umm", "err", "er", "mmm",
})

# Known command/correction phrases — must never be classified as filler even
# though they can be short.
_COMMAND_OVERRIDES: frozenset[str] = frozenset({
    "no", "stop", "cancel", "wait", "hold on", "change that",
    "agent", "repeat", "go back", "undo",
})


# ---------------------------------------------------------------------------
# Guard-window helper
# ---------------------------------------------------------------------------

def is_within_barge_in_guard_window(
    playback_started_at: float | None,
    guard_seconds: float,
    now_monotonic: float | None = None,
) -> bool:
    """Return True if we are still inside the post-playback guard window.

    During the guard window (first N ms of TTS playback), bot-tail audio and
    echo artefacts should not be treated as user speech.

    Args:
        playback_started_at: monotonic timestamp when TTS started, or None.
        guard_seconds:        Guard window duration in seconds (0 = disabled).
        now_monotonic:        Inject a fake clock for testing; defaults to
                              ``time.monotonic()``.
    """
    if playback_started_at is None:
        return False
    if guard_seconds <= 0:
        return False
    now = now_monotonic if now_monotonic is not None else time.monotonic()
    return (now - playback_started_at) < guard_seconds


# ---------------------------------------------------------------------------
# Filler detection
# ---------------------------------------------------------------------------

def is_filler_only(text: str) -> bool:
    """Return True when the transcript consists entirely of noise/filler tokens.

    Real command words (stop, cancel, no, …) always return False even when
    they are one-word utterances.  Empty transcripts return True.

    Examples:
        is_filler_only("")          → True
        is_filler_only("uh")        → True
        is_filler_only("um hmm")    → True
        is_filler_only("no")        → False  (command override)
        is_filler_only("change that") → False
        is_filler_only("coke")      → False
    """
    if not text:
        return True

    # Strip punctuation, lowercase
    normalized = re.sub(r"[^\w\s]", "", text.lower()).strip()
    if not normalized:
        return True

    # Command phrases take priority — never treat as filler
    if any(cmd in normalized for cmd in _COMMAND_OVERRIDES):
        return False

    tokens = normalized.split()
    return all(token in _FILLER_WORDS for token in tokens)


# ---------------------------------------------------------------------------
# Barge-in evaluation
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class BargeInDecision:
    """Result of evaluate_barge_in_candidate."""
    accepted: bool
    reason: str


def evaluate_barge_in_candidate(
    *,
    session: Session | None,
    text: str,
    audio_duration_ms: float | None,
    confidence: float | None,
    playback_started_at: float | None,
    config,  # RealtimeTurnConfig — avoid circular import at module level
) -> BargeInDecision:
    """Run the full acceptance pipeline for a barge-in candidate.

    Returns BargeInDecision(accepted=True, reason="accepted") or a rejection
    with a specific human-readable reason that gets logged.

    Acceptance requires ALL of:
    - barge_in_enabled flag is True
    - not inside post-playback guard window
    - transcript is not filler/noise
    - audio_duration_ms (if known) >= min_barge_in_audio_ms
    - confidence (if known) >= min_barge_in_confidence
    - word count >= min_barge_in_words  OR  transcript is a known command
    - is_actionable_barge_in() returns True (semantic/state gate)
    """
    if not config.barge_in_enabled:
        return BargeInDecision(accepted=False, reason="barge_in_disabled")

    guard_seconds = config.post_playback_guard_ms / 1000.0
    if is_within_barge_in_guard_window(playback_started_at, guard_seconds):
        return BargeInDecision(accepted=False, reason="inside_playback_guard_window")

    if is_filler_only(text):
        return BargeInDecision(accepted=False, reason="filler_only")

    if audio_duration_ms is not None and audio_duration_ms < config.min_barge_in_audio_ms:
        return BargeInDecision(
            accepted=False,
            reason=f"audio_too_short_ms={audio_duration_ms:.0f}",
        )

    if confidence is not None and confidence < config.min_barge_in_confidence:
        return BargeInDecision(
            accepted=False,
            reason=f"confidence_too_low={confidence:.2f}",
        )

    normalized = normalize_text(text) or ""
    word_count = len(normalized.split()) if normalized else 0
    known_commands = {"stop", "cancel", "no", "hold on", "wait", "change that", "agent", "repeat"}
    is_known_command = any(cmd in normalized for cmd in known_commands)

    # Semantic gate: if actionable in the current state, accept regardless of
    # word count — the state machine already confirmed this is meaningful.
    is_action = is_actionable_barge_in(session, text)

    if not is_action:
        # Not state-actionable. Check if it's a known global command that
        # bypasses state-specific gating.
        if is_known_command:
            return BargeInDecision(accepted=True, reason="accepted")
        # Apply word count as a noise pre-filter for non-actionable text.
        if word_count < config.min_barge_in_words:
            return BargeInDecision(
                accepted=False,
                reason=f"too_few_words={word_count}",
            )
        return BargeInDecision(accepted=False, reason="not_actionable")

    return BargeInDecision(accepted=True, reason="accepted")


# ---------------------------------------------------------------------------
# Semantic / state-aware barge-in gate
# ---------------------------------------------------------------------------

def is_actionable_barge_in(
    session: Session | None,
    user_text: str,
) -> bool:
    if session is None:
        return False

    normalized = normalize_text(user_text)
    if not normalized:
        return False

    context = session.conversation_context
    state = session.conversation_state

    if _looks_like_global_control(context=context, normalized_text=normalized):
        return True

    if state == ConversationState.WAITING_FOR_ORDER_TYPE:
        return OrderTypeResolver.resolve(normalized) is not None

    if state == ConversationState.WAITING_FOR_DELIVERY_ELIGIBILITY:
        return _is_delivery_eligibility_reply(context.current_prompt_field or "delivery_area", normalized)

    if state == ConversationState.WAITING_FOR_DELIVERY_ADDRESS_COLLECTION:
        return _is_delivery_address_reply(context.current_prompt_field or "delivery_seed_confirmation", normalized)

    if state in {
        ConversationState.WAITING_FOR_SIDE,
        ConversationState.WAITING_FOR_MODIFIER,
    }:
        return (
            looks_like_done_answer(normalized)
            or looks_like_skip_answer(normalized)
            or looks_like_more_options_answer(normalized)
            or _matches_available_choice(normalized, context.available_choices_values)
        )

    if state in {
        ConversationState.WAITING_FOR_SIZE,
        ConversationState.WAITING_FOR_SIDE_SIZE,
    }:
        return _matches_available_choice(normalized, context.available_choices_values)

    if state == ConversationState.WAITING_FOR_QUANTITY:
        return normalize_quantity(normalized) is not None

    if state in {
        ConversationState.CONFIRMING_ITEM,
        ConversationState.CONFIRMING_ORDER,
        ConversationState.WAITING_FOR_PAYMENT,
        ConversationState.WAITING_FOR_CHECKOUT_COMPLETION,
    }:
        return _is_yes_like(normalized) or _is_no_like(normalized)

    return False


def _looks_like_global_control(*, context, normalized_text: str) -> bool:
    if _contains_phrase(normalized_text, _CONTROL_PHRASES):
        return True

    if looks_like_ordering_request(context, normalized_text, include_slots=False):
        return True

    return False


def _contains_phrase(text: str, phrases: tuple[str, ...] | set[str]) -> bool:
    return any(phrase in text for phrase in phrases)


def _is_yes_like(text: str) -> bool:
    return _contains_phrase(text, _YES_LIKE)


def _is_no_like(text: str) -> bool:
    return _contains_phrase(text, _NO_LIKE)


def _matches_available_choice(text: str, choices: tuple[str, ...] | list[str]) -> bool:
    if not text or not choices:
        return False

    candidates = {
        normalize_text(text),
        normalize_waiting_value(text),
    }
    candidates = {candidate for candidate in candidates if candidate}
    if not candidates:
        return False

    normalized_choices = [normalize_text(choice) for choice in choices if normalize_text(choice)]
    if not normalized_choices:
        return False

    for candidate in candidates:
        for choice in normalized_choices:
            if candidate == choice:
                return True
            if len(choice) >= 3 and choice in candidate:
                return True

    return False


def _extract_zip(text: str) -> str | None:
    if not text:
        return None

    normalized = re.sub(r"[^a-z0-9\s-]", " ", text.lower())

    match = re.search(r"\b(\d{5})(?:-\d{4})?\b", normalized)
    if match:
        return match.group(1)

    spaced_digits_match = re.search(r"(?<!\d)((?:\d[\s-]*){5,9})(?!\d)", normalized)
    if spaced_digits_match:
        digits_only = re.sub(r"\D", "", spaced_digits_match.group(1))
        if len(digits_only) >= 5:
            return digits_only[:5]

    tokens = normalized.replace("-", " ").split()
    word_to_digit = {
        "zero": "0",
        "oh": "0",
        "o": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
    }
    number_words = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
        "thirteen": 13,
        "fourteen": 14,
        "fifteen": 15,
        "sixteen": 16,
        "seventeen": 17,
        "eighteen": 18,
        "nineteen": 19,
        "twenty": 20,
        "thirty": 30,
        "forty": 40,
        "fifty": 50,
        "sixty": 60,
        "seventy": 70,
        "eighty": 80,
        "ninety": 90,
    }

    def parse_number_phrase(candidate_tokens: list[str]) -> int | None:
        total = 0
        current = 0
        used = False

        for candidate in candidate_tokens:
            if candidate in number_words:
                current += number_words[candidate]
                used = True
                continue

            if candidate == "hundred":
                if current == 0:
                    current = 1
                current *= 100
                used = True
                continue

            if candidate == "thousand":
                if current == 0:
                    current = 1
                total += current * 1000
                current = 0
                used = True
                continue

            return None

        if not used:
            return None

        return total + current

    digits: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]

        if token in {"double", "triple"} and index + 1 < len(tokens):
            digit = word_to_digit.get(tokens[index + 1])
            if digit:
                digits.extend([digit] * (2 if token == "double" else 3))
                index += 2
                continue

        if token.isdigit():
            digits.extend(list(token))
        elif token in word_to_digit:
            digits.append(word_to_digit[token])

        index += 1

    joined = "".join(digits)
    if len(joined) >= 5:
        return joined[:5]

    phrase_tokens: list[str] = []
    for token in tokens + [""]:
        if token in number_words or token in {"hundred", "thousand"}:
            phrase_tokens.append(token)
            continue

        if phrase_tokens:
            parsed_value = parse_number_phrase(phrase_tokens)
            if parsed_value is not None and 10000 <= parsed_value <= 99999:
                return str(parsed_value)
            phrase_tokens = []

    return None


def _extract_house_number(text: str) -> str | None:
    match = re.search(r"\bhouse number\s+([a-z0-9\-]+)\b", text)
    if match:
        return match.group(1)

    candidate = text.strip(" ,.")
    if re.fullmatch(r"[a-z0-9\-]+", candidate or ""):
        return candidate

    return None


def _is_delivery_eligibility_reply(step: str, text: str) -> bool:
    if step == "delivery_area":
        return len(text) >= 2 and _extract_zip(text) is None

    if step == "delivery_postal_code":
        return _extract_zip(text) is not None

    if step == "delivery_eligibility_confirmation":
        return _is_yes_like(text) or _is_no_like(text) or _extract_zip(text) is not None

    return False


def _is_delivery_address_reply(step: str, text: str) -> bool:
    if step == "delivery_seed_confirmation":
        return _is_yes_like(text) or _is_no_like(text) or _extract_zip(text) is not None

    if step == "delivery_area":
        return len(text) >= 2 and _extract_zip(text) is None

    if step == "delivery_postal_code":
        return _extract_zip(text) is not None

    if step == "delivery_house_number":
        return _extract_house_number(text) is not None

    if step == "delivery_house_number_confirmation":
        return _is_yes_like(text) or _is_no_like(text)

    if step == "delivery_street":
        return len(text.strip(" ,.")) >= 2

    if step == "delivery_street_confirmation":
        return _is_yes_like(text) or _is_no_like(text)

    if step == "delivery_secondary_address":
        return bool(text)

    if step == "delivery_secondary_address_confirmation":
        return _is_yes_like(text) or _is_no_like(text)

    return False
