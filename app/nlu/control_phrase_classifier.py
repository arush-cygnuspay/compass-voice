# app/nlu/control_phrase_classifier.py
"""State-aware control phrase classifier.

Classifies normalized user text into control actions BEFORE menu/option
resolution runs.  This prevents control phrases ("no skip that", "can you
repeat", "add done") from leaking into the resolver and being echoed back
as "I couldn't find X".

Design constraints
------------------
- Stateless and deterministic — safe as a module-level singleton.
- No menu/option knowledge — handlers decide what to do with the result.
- No shared mutable state.
- Precedence: repeat > state-specific (checkout/confirm/deny) > skip/done
  > negated_option > none.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Optional

from app.nlu.query_normalization.text_preprocessor import normalize_text


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ControlPhraseResult:
    action: Literal[
        "skip",
        "done",
        "repeat",
        "checkout",
        "confirm",
        "deny",
        "negated_option",
        "none",
    ]
    confidence: float
    normalized_target: Optional[str] = None
    reason: Optional[str] = None


_NONE_RESULT = ControlPhraseResult(action="none", confidence=0.0)


# ---------------------------------------------------------------------------
# State groupings (compare against .value to stay string-based)
# ---------------------------------------------------------------------------

_IDLE_STATES: frozenset[str] = frozenset({
    "idle",
})

_SIDE_MODIFIER_STATES: frozenset[str] = frozenset({
    "waiting_for_side",
    "waiting_for_modifier",
})

_CONFIRM_ORDER_STATES: frozenset[str] = frozenset({
    "confirming_order",
})


# ---------------------------------------------------------------------------
# Phrase sets — side / modifier waiting states
# ---------------------------------------------------------------------------

# Exact-match → SKIP
_SKIP_EXACT: frozenset[str] = frozenset({
    "skip",
    "skip that",
    "skip it",
    "no skip",
    "no skip that",
    "no skip it",
    "leave it",
    "leave that",
    "leave it off",
    "dont add that",
    "dont add it",
    "dont want that",
    "nah skip",
    "nah skip that",
})

# Exact-match → DONE (advance / close group)
_DONE_EXACT: frozenset[str] = frozenset({
    "done",
    "add done",
    "no done",
    "thats all",
    "that is all",
    "thats it",
    "that is it",
    "all good",
    "all good then",
    "nothing else",
    "no more",
    "no nothing else",
    "im good",
    "i am good",
    "im done",
    "i am done",
    "finished",
    "im finished",
    "i am finished",
    "nothing more",
    "were good",
    "we are good",
    "thats enough",
    "that is enough",
    "no more please",
})

# Exact-match → REPEAT (meta-clarify: repeat the last prompt)
# Only phrases that the existing OPTIONS_REQUEST / META_CLARIFY resolver
# cannot handle on its own (e.g. "can you repeat" leaks to unmatched
# feedback without this interception).
# "What are the options", "list options", etc. are intentionally left OUT so
# they fall through to resolve_control_intent → OPTIONS_REQUEST →
# "list_modifier_options" / "list_side_options" as before.
_REPEAT_EXACT: frozenset[str] = frozenset({
    "repeat",
    "repeat that",
    "can you repeat",
    "can you repeat that",
    "say that again",
})


# ---------------------------------------------------------------------------
# Negation-prefix pattern
# Captures: "no X", "without X", "don't want X", "dont want X",
#           "remove X", "hold X"
# ---------------------------------------------------------------------------

_NEGATION_PREFIX_RE: re.Pattern[str] = re.compile(
    r"^(?:no|without|dont\s+want|do\s+not\s+want|remove|hold)\s+(.+)$"
)

# Words that indicate a control phrase — NOT a menu item.
# If the negation target contains any of these, the utterance is NOT a
# "negated option" referring to a menu item; it is a control phrase.
_CONTROL_WORDS: frozenset[str] = frozenset({
    "skip",
    "done",
    "repeat",
    "more",
    "nothing",
    "thanks",
    "thank",
    "that",
    "it",
    "this",
    "those",
    "these",
    "anything",
    "else",
})

# Additional functional/negation words that carry no menu-item meaning.
# Used to detect targets like "i do not want any" that have zero real content.
_NEGATION_FUNCTION_WORDS: frozenset[str] = frozenset({
    "i", "do", "dont", "not", "want", "any", "none", "there", "just",
    "really", "at", "all", "have", "me", "my", "we", "us", "you",
})


# ---------------------------------------------------------------------------
# Confirm-order state phrase sets
# ---------------------------------------------------------------------------

_CHECKOUT_EXACT: frozenset[str] = frozenset({
    "checkout",
    "check out",
    "place the order",
    "place my order",
    "confirm order",
    "finalize order",
    "go ahead",
    "proceed",
})

# Optional leading phrases before a checkout keyword
_CHECKOUT_LEADING: tuple[str, ...] = (
    "i said ",
    "oh yeah ",
    "actually ",
    "well actually ",
    "go ahead and ",
    "please ",
    "yeah ",
    "yes ",
    "ok ",
    "okay ",
    "alright ",
    "all right ",
    "uh ",
    "um ",
)

_CONFIRM_EXACT: frozenset[str] = frozenset({
    "yes",
    "yeah",
    "yep",
    "yup",
    "confirm",
    "thats correct",
    "that is correct",
    "correct",
    "right",
    "sounds good",
    "looks good",
    "absolutely",
    "sure",
    "definitely",
})

_DENY_EXACT: frozenset[str] = frozenset({
    "no",
    "nope",
    "nah",
    "cancel",
    "dont place it",
    "do not place it",
    "dont confirm",
    "do not confirm",
    "stop",
    "dont do it",
})

# ---------------------------------------------------------------------------
# IDLE-state checkout phrase sets
# Phrases that unambiguously mean "I'm done adding items, proceed to checkout"
# when spoken in IDLE with a non-empty cart.
# All entries must be in normalize_text() form (lowercase, no punctuation).
# Bare "yes"/"no" are deliberately excluded — they are confirm/deny responses.
# ---------------------------------------------------------------------------

_IDLE_CHECKOUT_EXACT: frozenset[str] = frozenset({
    # Direct checkout / wrap-up
    "checkout",
    "check out",
    "thats it",
    "that is it",
    "thats all",
    "that is all",
    "nothing else",
    "no more",
    "im done",
    "i am done",
    "done",
    "finished",
    "im finished",
    "i am finished",
    "all good",
    "were good",
    "we are good",
    "im good",
    "i am good",
    # Order finalization
    "complete my order",
    "complete the order",
    "finish my order",
    "finish the order",
    "finalize",
    "finalize my order",
    "finalize the order",
    "place the order",
    "place my order",
    "lets checkout",
    "lets check out",
    "go ahead and checkout",
    "go ahead and check out",
    "ready to checkout",
    "ready to check out",
    # "for now" variants — NLU fires UNKNOWN for these; coerce_idle_to_checkout
    # will convert them to CHECKOUT via exact match.
    "thats it for now",
    "that is it for now",
    "thats all for now",
    "that is all for now",
    "i think thats all",
    "i think that is all",
    "i think thats it",
    "i think that is it",
    "i think im good",
    "i think i am good",
    "i think were good",
    "i think we are good",
    # Payment-specific — these must route to confirm_order_summary first,
    # never directly to WaitingForPayment.
    "continue to payment",
    "proceed to payment",
    "payment",
    "pay",
    "pay now",
    "send payment link",
    "send the payment link",
    "send me the payment link",
    "text me payment link",
    "text me the payment link",
    "text me the link",
    "text me a payment link",
})


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class ControlPhraseClassifier:
    """Stateless, deterministic control-phrase classifier.

    Call ``classify()`` before any menu/option resolver to intercept
    control phrases that would otherwise produce bad "I couldn't find X"
    responses.

    The classifier is state-aware: it applies different rules for
    side/modifier waiting states vs. the confirm-order state, and returns
    ``action="none"`` for all other states (no-op, preserve existing flow).
    """

    def classify(
        self,
        normalized_text: str,
        state: str | None,
        current_prompt_field: str | None = None,  # reserved for future scoping
    ) -> ControlPhraseResult:
        """Classify *normalized_text* for *state*.

        Returns a :class:`ControlPhraseResult` with ``action="none"`` when
        no control phrase is detected.  The caller is responsible for acting
        on the result.
        """
        text = normalize_text(normalized_text or "").strip()
        if not text:
            return _NONE_RESULT

        state_val = (state or "").lower().strip()

        if state_val in _IDLE_STATES:
            return self._classify_idle(text)

        if state_val in _CONFIRM_ORDER_STATES:
            return self._classify_confirm_order(text)

        if state_val in _SIDE_MODIFIER_STATES:
            return self._classify_side_modifier(text)

        return _NONE_RESULT

    # ------------------------------------------------------------------
    # Private — IDLE state
    # ------------------------------------------------------------------

    def _classify_idle(self, text: str) -> ControlPhraseResult:
        """Classify for IDLE state.

        Detects checkout / done / payment phrases that signal the caller
        wants to finalize their order.  Bare affirmations ("yes", "yeah")
        and bare denials ("no", "nope") are intentionally excluded — those
        are not checkout commands.
        """
        # Direct match
        if text in _IDLE_CHECKOUT_EXACT:
            return ControlPhraseResult(
                action="checkout",
                confidence=1.0,
                reason="exact_idle_checkout_phrase",
            )

        # Prefixed match: "please checkout", "okay payment", etc.
        for prefix in _CHECKOUT_LEADING:
            if text.startswith(prefix):
                remainder = text[len(prefix):].strip()
                if remainder in _IDLE_CHECKOUT_EXACT:
                    return ControlPhraseResult(
                        action="checkout",
                        confidence=0.95,
                        reason="prefixed_idle_checkout_phrase",
                    )

        return _NONE_RESULT

    # ------------------------------------------------------------------
    # Private — side / modifier states
    # ------------------------------------------------------------------

    def _classify_side_modifier(self, text: str) -> ControlPhraseResult:
        """Classify for WAITING_FOR_SIDE / WAITING_FOR_MODIFIER.

        Precedence: repeat > done > skip > negated_option > none.
        """
        # 1. REPEAT — highest priority (must come before done/skip so
        #    "what are the options" doesn't get swallowed by anything else)
        if text in _REPEAT_EXACT:
            return ControlPhraseResult(
                action="repeat",
                confidence=1.0,
                reason="exact_repeat_phrase",
            )

        # 2. DONE
        if text in _DONE_EXACT:
            return ControlPhraseResult(
                action="done",
                confidence=1.0,
                reason="exact_done_phrase",
            )

        # 3. SKIP
        if text in _SKIP_EXACT:
            return ControlPhraseResult(
                action="skip",
                confidence=1.0,
                reason="exact_skip_phrase",
            )

        # 4. Negation prefix — "no X", "without X", etc.
        negation_match = _NEGATION_PREFIX_RE.match(text)
        if negation_match:
            target = negation_match.group(1).strip()
            target_words = set(target.lower().split())
            control_hit = target_words & _CONTROL_WORDS

            if control_hit:
                # Target refers to a control concept, not a menu item.
                if "skip" in control_hit:
                    return ControlPhraseResult(
                        action="skip",
                        confidence=0.92,
                        reason="negation_control_skip",
                    )
                if "done" in control_hit or "finish" in control_hit:
                    return ControlPhraseResult(
                        action="done",
                        confidence=0.92,
                        reason="negation_control_done",
                    )
                if "repeat" in control_hit:
                    return ControlPhraseResult(
                        action="repeat",
                        confidence=0.92,
                        reason="negation_control_repeat",
                    )
                if "more" in control_hit or "nothing" in control_hit:
                    return ControlPhraseResult(
                        action="done",
                        confidence=0.92,
                        reason="negation_control_done_more",
                    )
                # Other control word (that, it, thanks, etc.) → treat as skip
                return ControlPhraseResult(
                    action="skip",
                    confidence=0.80,
                    reason="negation_control_generic_skip",
                )

            # Check if the target has any real content beyond functional words.
            # "no i do not want any" → all functional → treat as skip/done.
            target_content = target_words - _CONTROL_WORDS - _NEGATION_FUNCTION_WORDS
            if not target_content:
                return ControlPhraseResult(
                    action="skip",
                    confidence=0.75,
                    reason="negation_no_content_target",
                )

            # Target is a real content word — this is a negated option
            # instruction ("no onions", "without bun", etc.).
            return ControlPhraseResult(
                action="negated_option",
                confidence=0.90,
                normalized_target=target,
                reason="negation_prefix",
            )

        return _NONE_RESULT

    # ------------------------------------------------------------------
    # Private — confirm-order state
    # ------------------------------------------------------------------

    def _classify_confirm_order(self, text: str) -> ControlPhraseResult:
        """Classify for CONFIRMING_ORDER.

        Only handles checkout detection — confirm/deny fall through to
        the existing ControlIntentResolver which already handles them.
        """
        # Direct checkout phrase
        if text in _CHECKOUT_EXACT:
            return ControlPhraseResult(
                action="checkout",
                confidence=1.0,
                reason="exact_checkout_phrase",
            )

        # Prefixed checkout phrase: "i said checkout", "oh yeah check out"
        for prefix in _CHECKOUT_LEADING:
            if text.startswith(prefix):
                remainder = text[len(prefix):].strip()
                if remainder in _CHECKOUT_EXACT or remainder in {"checkout", "check out"}:
                    return ControlPhraseResult(
                        action="checkout",
                        confidence=0.95,
                        reason="prefixed_checkout_phrase",
                    )

        return _NONE_RESULT


# ---------------------------------------------------------------------------
# Module-level singleton — import this; do not construct per call.
# ---------------------------------------------------------------------------

DEFAULT_CLASSIFIER: ControlPhraseClassifier = ControlPhraseClassifier()
