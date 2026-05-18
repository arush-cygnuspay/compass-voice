# app/nlu/semantic_repair/add_item_planner_result.py
"""Phase 4 GPT Add-Item Planner result dataclasses.

These are the output types of GptAddItemPlannerService.run().
They are frozen / immutable so they can be safely passed across threads.

Safety contract
---------------
* AddItemPlannerResult.safe_to_apply is set by the apply gate, not by GPT.
* The result never mutates cart, session, state, or FSM.
* The service wrapper always returns a result — never raises.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Unresolved entity
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PlannerUnresolved:
    """An entity from the user utterance that GPT could not resolve to a menu item."""

    text: str
    reason: str  # "not_on_menu" | "ambiguous" | "belongs_to_unknown_group" | "unsupported"

    def to_dict(self) -> dict[str, str]:
        return {"text": self.text, "reason": self.reason}


# ---------------------------------------------------------------------------
# Per-item GPT output
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PlannerGptItem:
    """One structured add-item entry returned by GPT in the Phase 4 planner.

    Mirrors GptAddItem from the Phase 1/2 extractor but adds candidate_item_id
    (the menu item ID hint GPT extracted from the candidate list) and extends
    modifier operations to include "extra" and "light".
    """

    item_name: str
    candidate_item_id: str | None = None
    quantity: int | None = None
    size: str | None = None
    variant: str | None = None
    modifiers: tuple["PlannerGptModifier", ...] = ()
    sides: tuple["PlannerGptSide", ...] = ()
    special_instructions: str | None = None
    parse_notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PlannerGptModifier:
    """A modifier extracted from the Phase 4 GPT planner output."""

    name: str
    operation: str = "add"    # "add" | "remove" | "extra" | "light"
    quantity: int = 1


@dataclass(frozen=True, slots=True)
class PlannerGptSide:
    """A side dish extracted from the Phase 4 GPT planner output."""

    name: str
    quantity: int = 1
    size: str | None = None


# ---------------------------------------------------------------------------
# Full planner result
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AddItemPlannerResult:
    """Full result of one GptAddItemPlannerService.run() call (or a no-op).

    Fields
    ------
    decision:
        "add_items" | "clarify" | "no_repair" | "unclear" | "skipped" | "error"
    gpt_called:
        True when an OpenAI call was made for this turn.
    route_mode:
        "no_gpt" | "shadow_gpt" | "inline_gpt"
    route_reason:
        Short string explaining why this route was chosen.
    items:
        GPT-extracted items (before menu validation).
    unresolved:
        Entities GPT could not resolve to a menu item.
    confidence:
        GPT's self-reported confidence (0.0–1.0). Informational only;
        the apply gate uses its own validated threshold.
    reason_code:
        GPT's reason for the decision.
    validated_plan:
        Result of running AddItemPlanValidator on items (None when not run).
    validator_passed:
        True when validated_plan has no blocking warnings and ≥1 valid item.
    validator_reject_reason:
        Short reason when validator_passed=False.
    safe_to_apply:
        True only when the apply gate approves — NOT from GPT's own field.
    skipped_reason:
        Why GPT was not called (when gpt_called=False).
    parse_error:
        Error string when the GPT response could not be parsed.
    latency_ms:
        Wall-clock time from GPT request to response (ms).
    model:
        OpenAI model used for the call.
    prompt_chars / completion_chars:
        Character counts for budget / cost tracking.
    """

    decision: str = "skipped"
    gpt_called: bool = False
    route_mode: str = "no_gpt"
    route_reason: str = ""
    items: tuple[PlannerGptItem, ...] = ()
    unresolved: tuple[PlannerUnresolved, ...] = ()
    confidence: float | None = None
    reason_code: str | None = None
    validated_plan: object | None = None      # ValidatedAddItemPlan | None
    validator_passed: bool = False
    validator_reject_reason: str | None = None
    safe_to_apply: bool = False               # set by apply gate, never from GPT
    skipped_reason: str | None = None
    parse_error: str | None = None
    latency_ms: float | None = None
    model: str | None = None
    prompt_chars: int = 0
    completion_chars: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict suitable for JSONL logging."""
        validated_dict: dict[str, Any] | None = None
        vp = self.validated_plan
        if vp is not None:
            try:
                validated_dict = {
                    "items_count": len(getattr(vp, "items", ())),
                    "rejected_items": list(getattr(vp, "rejected_items", ())),
                    "has_blocking_warnings": getattr(vp, "has_blocking_warnings", False),
                    "validator_ms": getattr(vp, "validator_ms", None),
                }
            except Exception:
                validated_dict = None

        return {
            "decision": self.decision,
            "gpt_called": self.gpt_called,
            "route_mode": self.route_mode,
            "route_reason": self.route_reason,
            "items": [
                {
                    "item_name": it.item_name,
                    "candidate_item_id": it.candidate_item_id,
                    "quantity": it.quantity,
                    "size": it.size,
                    "variant": it.variant,
                    "modifiers": [
                        {"name": m.name, "operation": m.operation, "quantity": m.quantity}
                        for m in it.modifiers
                    ],
                    "sides": [
                        {"name": s.name, "quantity": s.quantity, "size": s.size}
                        for s in it.sides
                    ],
                    "special_instructions": it.special_instructions,
                    "parse_notes": list(it.parse_notes),
                }
                for it in self.items
            ],
            "unresolved": [u.to_dict() for u in self.unresolved],
            "confidence": self.confidence,
            "reason_code": self.reason_code,
            "validated_plan": validated_dict,
            "validator_passed": self.validator_passed,
            "validator_reject_reason": self.validator_reject_reason,
            "safe_to_apply": self.safe_to_apply,
            "skipped_reason": self.skipped_reason,
            "parse_error": self.parse_error,
            "latency_ms": self.latency_ms,
            "model": self.model,
            "prompt_chars": self.prompt_chars,
            "completion_chars": self.completion_chars,
        }


# Sentinel returned when the planner was not called at all.
ADD_ITEM_PLANNER_NOT_CALLED = AddItemPlannerResult(
    decision="skipped",
    gpt_called=False,
    route_mode="no_gpt",
    route_reason="not_called",
    skipped_reason="not_called",
)
