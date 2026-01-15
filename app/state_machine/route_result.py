# app/state_machine/route_result.py

from dataclasses import dataclass
from typing import Optional


@dataclass
class RouteResult:
    """
    Result of routing a (state, intent) pair.
    """

    allowed: bool
    handler_name: Optional[str] = None
    reason: Optional[str] = None
