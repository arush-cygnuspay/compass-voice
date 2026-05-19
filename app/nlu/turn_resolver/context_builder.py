# app/nlu/turn_resolver/context_builder.py
"""Compact GPT context packet builder for the turn-resolution layer.

Safety contract
---------------
* NEVER includes: API key, phone number, email, payment links,
  full menu JSON, or full cart raw JSON.
* Cart is represented as item count + item names only.
* Previous turns are capped at 3 pairs maximum.
* Choices (option names) are capped at 20 entries.
* All string fields are stripped of leading/trailing whitespace.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from app.nlu.nlu_result import IntentCandidate, NLUResult, SlotValue
    from app.state_machine.models.conversation_context import ConversationContext
    from app.state_machine.models.conversation_state import ConversationState

_MAX_PREVIOUS_TURNS = 3
_MAX_CHOICES = 20
_MAX_CART_NAMES = 15


@dataclass(frozen=True, slots=True)
class GptTurnContextPacket:
    """Compact, PII-free snapshot of turn context for a GPT resolution call.

    Fields
    ------
    bucket:
        Which resolution bucket triggered this call.
    user_text:
        Normalized user utterance (no raw STT noise).
    state:
        Current conversation state value string.
    prompt_field:
        Active slot-filling field name (e.g. "modifier", "size") if any.
    local_intent:
        Effective intent string from local NLU (Intent enum value).
    local_confidence:
        Intent confidence from local NLU (0.0–1.0).
    top_intents:
        Top-K intent candidates as (intent_str, confidence) tuples.
    local_slots:
        Compact slot list — each entry is {"n": name, "v": value}.
        Raw offsets and confidence are excluded to keep payload small.
    allowed_intents:
        Intent strings that are valid for the current state (for Bucket 0).
    choices:
        Option names available in the current group (Bucket 2 only).
        Never exceeds _MAX_CHOICES entries.
    cart_item_count:
        Number of distinct items currently in the cart.
    cart_item_names:
        Item names from the cart (names only — no prices, IDs, or quantities).
    previous_turns:
        Last up to 3 (role, text) pairs from turn memory.
        role is "user" or "bot".
    """

    bucket: str
    user_text: str
    state: str
    prompt_field: str | None
    local_intent: str
    local_confidence: float
    top_intents: tuple[tuple[str, float], ...]
    local_slots: tuple[dict, ...]
    allowed_intents: tuple[str, ...]
    choices: tuple[str, ...]
    cart_item_count: int
    cart_item_names: tuple[str, ...]
    previous_turns: tuple[tuple[str, str], ...]

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict for prompt building or logging."""
        return {
            "bucket": self.bucket,
            "user_text": self.user_text,
            "state": self.state,
            "prompt_field": self.prompt_field,
            "local_intent": self.local_intent,
            "local_confidence": round(self.local_confidence, 4),
            "top_intents": [
                {"intent": i, "confidence": round(c, 4)}
                for i, c in self.top_intents
            ],
            "local_slots": list(self.local_slots),
            "allowed_intents": list(self.allowed_intents),
            "choices": list(self.choices),
            "cart_item_count": self.cart_item_count,
            "cart_item_names": list(self.cart_item_names),
            "previous_turns": [
                {"role": r, "text": t} for r, t in self.previous_turns
            ],
        }


def _compact_slots(slots: "tuple[SlotValue, ...]") -> tuple[dict, ...]:
    """Convert SlotValue tuples to compact {"n", "v"} dicts for the packet."""
    result = []
    for s in slots:
        val = s.value
        if val is None:
            continue
        result.append({"n": str(s.name), "v": str(val)})
    return tuple(result)


def _compact_top_intents(
    candidates: "tuple[IntentCandidate, ...]",
    top_k: int = 4,
) -> tuple[tuple[str, float], ...]:
    """Extract top-K (intent_str, confidence) pairs from NLU candidates."""
    seen: set[str] = set()
    result: list[tuple[str, float]] = []
    for cand in candidates[:top_k]:
        label = cand.canonical_intent
        if label and label not in seen:
            seen.add(label)
            result.append((label, round(float(cand.confidence), 4)))
    return tuple(result)


def build_context_packet(
    bucket: str,
    user_text: str,
    state: "ConversationState",
    context: "ConversationContext",
    local_nlu: "NLUResult",
    *,
    choices: Sequence[str] = (),
    cart_item_names: Sequence[str] = (),
    allowed_intents: Sequence[str] = (),
    previous_turns: Sequence[tuple[str, str]] | None = None,
) -> GptTurnContextPacket:
    """Build a compact, PII-free context packet for a GPT turn-resolution call.

    Parameters
    ----------
    bucket:
        The bucket name returned by ``pick_bucket()``.
    user_text:
        Normalized user utterance for this turn.
    state:
        Current ``ConversationState``.
    context:
        Current ``ConversationContext`` — used for prompt field and turn memory.
    local_nlu:
        NLU result for this turn — slots, confidence, and top-K candidates.
    choices:
        Option names from the current modifier/side group (Bucket 2).
    cart_item_names:
        Item names from the cart (names only, not full cart JSON).
    allowed_intents:
        Canonical intent strings permitted in this state.
    previous_turns:
        Override for previous turns.  If None, read from ``context.turn_memory``.

    Returns
    -------
    ``GptTurnContextPacket`` — frozen, safe to pass across threads.
    """
    # Previous turns: use override or read from context memory
    if previous_turns is None:
        raw_memory = list(context.get_turn_memory(_MAX_PREVIOUS_TURNS))
        prev = tuple(
            (str(role), str(text))
            for role, text in raw_memory
            if role and text and str(text).strip()
        )
    else:
        prev = tuple(
            (str(r), str(t)) for r, t in previous_turns
            if r and t and str(t).strip()
        )[-_MAX_PREVIOUS_TURNS:]

    return GptTurnContextPacket(
        bucket=bucket,
        user_text=(user_text or "").strip(),
        state=state.value if hasattr(state, "value") else str(state),
        prompt_field=context.current_prompt_field,
        local_intent=local_nlu.effective_intent.value,
        local_confidence=round(float(local_nlu.intent_confidence), 4),
        top_intents=_compact_top_intents(local_nlu.intent_candidates),
        local_slots=_compact_slots(local_nlu.slots),
        allowed_intents=tuple(str(i) for i in allowed_intents),
        choices=tuple(str(c) for c in choices[:_MAX_CHOICES]),
        cart_item_count=len(cart_item_names),
        cart_item_names=tuple(str(n) for n in cart_item_names[:_MAX_CART_NAMES]),
        previous_turns=prev,
    )
