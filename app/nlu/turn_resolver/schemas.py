# app/nlu/turn_resolver/schemas.py
"""Immutable output types for the unified GPT turn-resolution layer.

Safety contract
---------------
  * ``safe_to_apply`` is always set by ``validators.py``, never by GPT.
  * None of these objects carry API keys, phone numbers, payment links,
    full menu JSON, or full cart JSON.
  * ``items`` are logged but never applied to the cart in shadow mode.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ResolvedModifierPlan:
    """One modifier from a GPT-proposed item plan."""

    name: str
    operation: str = "add"  # add | remove | extra | light


@dataclass(frozen=True, slots=True)
class ResolvedSidePlan:
    """One side/drink from a GPT-proposed item plan."""

    name: str
    quantity: int = 1
    size: str | None = None


@dataclass(frozen=True, slots=True)
class ResolvedItemPlan:
    """One item entry from a GPT turn-resolution result.

    Parsed and logged; never directly applied to the cart.
    Cart mutations happen through the existing deterministic handlers
    using the item_name and slots as hints.
    """

    item_name: str
    quantity: int = 1
    size: str | None = None
    variant: str | None = None
    sides: tuple[ResolvedSidePlan, ...] = ()
    modifiers: tuple[ResolvedModifierPlan, ...] = ()
    # Required slots GPT says are missing
    missing: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GptTurnResolution:
    """Full result of one GPT turn-resolution attempt.

    ``bucket`` identifies which resolution path was taken:
      "idle_menu_item_resolution" — Bucket 0: idle low-confidence item query
      "option_resolution"         — Bucket 2: waiting-state option matching
      "multi_item_add_planning"   — Bucket 3: compound multi-item utterance
      "none"                      — GPT was not called

    ``decision`` values:
      "add_items"     — GPT identified item(s) to add
      "select_option" — GPT selected option(s) from the current group choices
      "clarify"       — GPT suggests asking the user to clarify
      "no_match"      — GPT could not resolve anything from the allowed set
      "skipped"       — GPT was not called (mode disabled / budget / terminal state)
      "error"         — GPT call failed or response could not be parsed

    ``safe_to_apply`` is set by ``validators.py`` after checking GPT output
    against the current menu/choice state.  GPT never sets this field.
    """

    bucket: str = "none"
    decision: str = "skipped"

    # Suggested canonical intent string (Intent enum value, not enum member)
    intent: str | None = None
    # Detected control intent ("skip" | "done" | "cancel" | None)
    control_intent: str | None = None

    # Items resolved (Bucket 0 and 3)
    items: tuple[ResolvedItemPlan, ...] = ()
    # Option names resolved (Bucket 2)
    selected_option_names: tuple[str, ...] = ()

    confidence: float | None = None
    reason_code: str | None = None

    # Set by validators.py — never from GPT
    safe_to_apply: bool = False

    latency_ms: float | None = None
    gpt_called: bool = False
    skipped_reason: str | None = None
    parse_error: str | None = None

    prompt_chars: int = 0
    completion_chars: int = 0
    model: str | None = None

    def to_log_dict(self) -> dict:
        """Return a JSON-serialisable dict for JSONL logging.

        Excludes fields that are never safe to log (API keys, PII).
        """
        return {
            "bucket": self.bucket,
            "decision": self.decision,
            "intent": self.intent,
            "control_intent": self.control_intent,
            "items": [
                {
                    "item_name": it.item_name,
                    "quantity": it.quantity,
                    "size": it.size,
                    "variant": it.variant,
                    "sides": [
                        {"name": s.name, "quantity": s.quantity, "size": s.size}
                        for s in it.sides
                    ],
                    "modifiers": [
                        {"name": m.name, "operation": m.operation}
                        for m in it.modifiers
                    ],
                    "missing": list(it.missing),
                }
                for it in self.items
            ],
            "selected_option_names": list(self.selected_option_names),
            "confidence": self.confidence,
            "reason_code": self.reason_code,
            "safe_to_apply": self.safe_to_apply,
            "latency_ms": self.latency_ms,
            "gpt_called": self.gpt_called,
            "skipped_reason": self.skipped_reason,
            "parse_error": self.parse_error,
            "prompt_chars": self.prompt_chars,
            "completion_chars": self.completion_chars,
            "model": self.model,
        }


# Sentinel returned when GPT was not called at all.
GPT_TURN_RESOLUTION_SKIPPED = GptTurnResolution(
    bucket="none",
    decision="skipped",
    gpt_called=False,
    skipped_reason="not_called",
)
