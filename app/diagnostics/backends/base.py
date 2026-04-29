# app/diagnostics/backends/base.py
"""DiagnosticsBackend Protocol — the interface every sink must satisfy."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.diagnostics.turn_event import TurnEvent


@runtime_checkable
class DiagnosticsBackend(Protocol):
    def record(self, event: TurnEvent) -> None: ...
