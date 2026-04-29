# app/core/flow_control/flow_decision.py
# Compatibility wrapper — logic now lives in app.state_machine.policy.
from app.state_machine.policy.flow_decision import (  # noqa: F401
    FlowAction,
    FlowDecision,
)
