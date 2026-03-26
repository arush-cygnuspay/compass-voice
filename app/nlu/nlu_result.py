# app/nlu/nlu_result.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from app.nlu.intent_resolution.intent import Intent


@dataclass(frozen=True, slots=True)
class SlotValue:
    """
    A single extracted slot/entity from the user's utterance.

    Offsets are relative to NLUResult.normalized_text.
    """
    name: str
    value: Any
    raw: Optional[str] = None
    start: Optional[int] = None
    end: Optional[int] = None
    confidence: Optional[float] = None


@dataclass(frozen=True, slots=True)
class NLUResult:
    """
    Output of the NLU layer for a single turn.

    This object intentionally separates:
    - effective_intent: the intent that downstream orchestration should use
    - model_main_intent / model_sub_intent: raw intent labels predicted by the model
    - slots: raw slots predicted by the slot model (or deterministic waiting-state rules)

    Notes:
    - raw_text: text passed into resolve_nlu after upstream cleanup.
    - normalized_text: canonical normalized text used for inference/downstream.
    - intent_confidence: confidence of the effective intent selected by NLU layer.
    """
    effective_intent: Intent
    intent_confidence: float

    raw_text: str
    normalized_text: str

    model_main_intent: Optional[str] = None
    model_sub_intent: Optional[str] = None

    slots: tuple[SlotValue, ...] = ()
    slot_model_ran: bool = False