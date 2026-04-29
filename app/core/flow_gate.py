# app/core/flow_gate.py
# Compatibility wrapper — logic now lives in app.state_machine.policy.
from app.state_machine.policy.flow_gate import (  # noqa: F401
    CONFIRMING_ORDER_EXIT_TO_IDLE_INTENTS,
    FlowGate,
    FlowGateDecision,
)
