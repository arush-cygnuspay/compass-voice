"""Pure decision policy for deny/done in selection groups.

Encapsulates the three-tier skip rule shared by the modifier and side
handlers. The rule is intentionally pure (no I/O, no context access)
so it is trivial to unit-test and reuse.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class GroupSkipDecision(str, Enum):
    SKIP_OPTIONAL = "skip_optional"
    ADVANCE_MIN_MET = "advance_min_met"
    BLOCK_UNDER_MIN = "block_under_min"


@dataclass(frozen=True, slots=True)
class GroupSkipResult:
    decision: GroupSkipDecision
    remaining_to_min: int
    selected_count: int
    min_required: int


def evaluate_group_skip(min_required: int, selected_count: int) -> GroupSkipResult:
    """Pure function. Three-tier rule for deny/done in selection groups."""
    min_required = max(int(min_required or 0), 0)
    selected_count = max(int(selected_count or 0), 0)

    if min_required == 0:
        return GroupSkipResult(
            decision=GroupSkipDecision.SKIP_OPTIONAL,
            remaining_to_min=0,
            selected_count=selected_count,
            min_required=0,
        )
    if selected_count >= min_required:
        return GroupSkipResult(
            decision=GroupSkipDecision.ADVANCE_MIN_MET,
            remaining_to_min=0,
            selected_count=selected_count,
            min_required=min_required,
        )
    return GroupSkipResult(
        decision=GroupSkipDecision.BLOCK_UNDER_MIN,
        remaining_to_min=min_required - selected_count,
        selected_count=selected_count,
        min_required=min_required,
    )
