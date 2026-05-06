# app/state_machine/handlers/payment/pickup_sms_resolver.py
"""State-scoped resolver for WAITING_FOR_PICKUP_SMS_PERMISSION.

Resolution contract
-------------------
  PickupSmsDecision.SEND_SMS     — customer wants the payment link texted.
  PickupSmsDecision.PAY_ON_PICKUP — customer will pay at the counter/on arrival.
  PickupSmsDecision.UNKNOWN      — ambiguous; re-prompt with the two-option prompt.

Resolution order (intent-first, phrase/regex fallback):
  1.  PAY_ON_PICKUP regex runs first — explicit pay-there language wins even
      when NLU also emits AFFIRM (e.g. "yes I'll pay at pickup").
  2.  control_intent AFFIRM  → SEND_SMS.
  3.  control_intent DENY / CANCEL → PAY_ON_PICKUP.
  4.  NLU intent: payment_request / checkout / finish_order → SEND_SMS.
      (These are scoped away from this state in the global registry — handled
      here to keep the override tight.)
  5.  SEND_SMS regex scan.
  6.  SEND_SMS exact-phrase candidates (post-filler stripping).
  7.  → UNKNOWN.

Every phrase/regex fallback hit is logged as ``phrase_fallback_used`` with
source, text, state, and decision for offline retraining.
"""
from __future__ import annotations

import re
from enum import Enum

from app.nlu.intent_resolution.intent import Intent
from app.nlu.query_normalization.text_preprocessor import normalize_text
from app.state_machine.control_intent_resolver import (
    ControlIntentKind,
    ResolvedControlIntent,
    log_control_intent_event,
)

_STATE_LABEL = "waiting_for_pickup_sms_permission"

# ── NLU intents that imply SEND_SMS in this state ───────────────────────────
# payment_request → AFFIRM is scoped to CONFIRMING_ORDER in the global
# registry. checkout / finish_order → DONE only in selection states.
# Both are handled here explicitly to avoid a global-registry change for a
# single state.
_SEND_SMS_INTENTS: frozenset[Intent] = frozenset({
    Intent.CONFIRM,
    Intent.PAYMENT_REQUEST,
    Intent.CHECKOUT,
    Intent.CONFIRM_ORDER,
    Intent.FINISH_ORDER,
})

# ── Exact-phrase candidates (matched post-filler-stripping) ─────────────────
# Short, unambiguous phrases not in the global affirm set.
_SEND_SMS_EXACT: frozenset[str] = frozenset({
    "send it",
    "text it",
    "send me",
})

# ── Regex: compositional "send me the link"-type utterances ─────────────────
# Matches e.g. "text me the payment link", "send it to my phone", "SMS me",
# "message me the link", "please send the payment link".
_RE_SEND_LINK = re.compile(
    r"\b(send|text|sms|message)\b.{0,50}\b(link|payment|me|phone|number)\b"
    r"|\bpayment link\b",
)

# ── Regex: pay-on-pickup utterances ─────────────────────────────────────────
# Runs BEFORE control_intent checks so that "yes I'll pay at pickup" resolves
# as PAY_ON_PICKUP even when NLU emits a high-confidence AFFIRM intent.
#
# Covers:
#   pay-at-location  — "I'll pay there / at the counter / when I arrive / ..."
#   negation-of-send — "don't send it / no link / no SMS / do not text me"
_RE_PAY_ON_PICKUP = re.compile(
    # pay-at-location: "ill pay at the counter", "pay when i get there", etc.
    r"\b(ill pay|i will pay|pay)\b.{0,60}"
    r"\b(pick\s*up|there|counter|arrival|in person|store"
    r"|when i arrive|when i get there|when i pick up|when i come|later)\b"
    r"|"
    # negation of any send/link/sms action: "dont send it", "no link", "no sms", etc.
    r"\b(dont|do not|no)\b.{0,40}\b(send|text|sms|message|link|payment)\b",
)


class PickupSmsDecision(str, Enum):
    SEND_SMS = "send_sms"
    PAY_ON_PICKUP = "pay_on_pickup"
    UNKNOWN = "unknown"


def resolve_pickup_sms_decision(
    user_text: str,
    intent: Intent,
    control_intent: ResolvedControlIntent | None,
) -> PickupSmsDecision:
    """Return the SEND_SMS / PAY_ON_PICKUP / UNKNOWN decision for this turn.

    Args:
        user_text:      Raw or pre-normalized utterance from this turn.
        intent:         NLU intent (may have been gated to Intent.UNKNOWN).
        control_intent: Result of resolve_control_intent() for this state.
    """
    normalized = normalize_text(user_text or "")

    # ── 1. PAY_ON_PICKUP regex — highest priority ────────────────────────────
    # Must run before control_intent checks so that utterances containing both
    # an affirmative signal and a pay-there signal (e.g. "yes I'll pay at
    # pickup") are resolved as PAY_ON_PICKUP rather than SEND_SMS.
    if normalized and _RE_PAY_ON_PICKUP.search(normalized):
        _log_fallback("pay_on_pickup_regex", normalized, PickupSmsDecision.PAY_ON_PICKUP)
        return PickupSmsDecision.PAY_ON_PICKUP

    # ── 2. control_intent AFFIRM → SEND_SMS ─────────────────────────────────
    if control_intent is not None and control_intent.kind == ControlIntentKind.AFFIRM:
        return PickupSmsDecision.SEND_SMS

    # ── 3. control_intent DENY / CANCEL → PAY_ON_PICKUP ─────────────────────
    if control_intent is not None and control_intent.kind in {
        ControlIntentKind.DENY,
        ControlIntentKind.CANCEL,
    }:
        return PickupSmsDecision.PAY_ON_PICKUP

    # ── 4. NLU intent override (state-scoped) ───────────────────────────────
    if intent in _SEND_SMS_INTENTS:
        return PickupSmsDecision.SEND_SMS

    if not normalized:
        return PickupSmsDecision.UNKNOWN

    # ── 5. SEND_SMS regex ────────────────────────────────────────────────────
    if _RE_SEND_LINK.search(normalized):
        _log_fallback("send_link_regex", normalized, PickupSmsDecision.SEND_SMS)
        return PickupSmsDecision.SEND_SMS

    # ── 6. SEND_SMS exact-phrase candidates ──────────────────────────────────
    # Uses the same filler-stripping semantics as control_intent_resolver so
    # that "yeah send it" → candidate "send it" is correctly resolved.
    candidates = _signal_candidates(normalized)
    if any(c in _SEND_SMS_EXACT for c in candidates if c):
        _log_fallback("send_link_phrase", normalized, PickupSmsDecision.SEND_SMS)
        return PickupSmsDecision.SEND_SMS

    return PickupSmsDecision.UNKNOWN


# ── Private helpers ──────────────────────────────────────────────────────────

# Leading fillers to strip — mirrors control_intent_resolver._LEADING_FILLERS,
# which includes the broader set ("okay", "yeah", "yes", etc.) that
# linguistic_rules._signal_candidates intentionally omits. We need the wider
# set here so "yeah send it" → candidate "send it" is correctly generated.
_LEADING_FILLERS: tuple[str, ...] = (
    "well", "so", "just", "please", "uh", "um", "hmm",
    "okay", "ok", "yeah", "yep", "yup", "yes",
)
_TRAILING_FILLERS: tuple[str, ...] = ("please", "thanks", "thank you")


def _signal_candidates(normalized: str) -> set[str]:
    """Generate phrase candidates by iteratively stripping leading/trailing fillers.

    Splitting on ' and ' is also applied, matching the semantics of
    control_intent_resolver._signal_candidates so that "go ahead and send it"
    produces "send it" as a candidate.
    """
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
            queue.extend(p.strip() for p in value.split(" and ") if p.strip())

    return candidates


def _log_fallback(source: str, text: str, decision: PickupSmsDecision) -> None:
    log_control_intent_event(
        "phrase_fallback_used",
        state=_STATE_LABEL,
        source=source,
        normalized_text=text,
        decision=decision.value,
    )
