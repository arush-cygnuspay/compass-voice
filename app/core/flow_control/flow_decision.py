# app/core/flow_control/flow_decision.py

from dataclasses import dataclass
from typing import Optional, Dict
from enum import Enum

from app.nlu.intent_resolution.intent import Intent


class FlowAction(Enum):
    PASS = "pass"
    REWRITE = "rewrite"
    BLOCK = "block"
    CANCEL = "cancel"


@dataclass(frozen=True)
class FlowDecision:
    """
    Result of FlowControlPolicy evaluation.
    This object is PURE control logic — no side effects.
    """
    action: FlowAction
    effective_intent: Optional[Intent] = None
    slot_interaction: Optional[str] = None
    response_key: Optional[str] = None
    response_payload: Optional[Dict] = None