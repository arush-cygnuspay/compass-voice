# app/state_machine/policy/flow_decision.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from app.nlu.intent_resolution.intent import Intent


class FlowAction(str, Enum):
    """What FlowControlPolicy wants TurnEngine to do for this turn."""
    PASS = "pass"
    REWRITE = "rewrite"
    BLOCK = "block"
    CANCEL = "cancel"
    HANDLE_READONLY_INTERRUPT = "handle_readonly_interrupt"


@dataclass(frozen=True, slots=True)
class FlowDecision:
    """Pure control decision emitted by FlowControlPolicy."""
    action: FlowAction
    effective_intent: Intent | None = None
    response_key: str | None = None
    response_payload: Mapping[str, Any] | None = None
