# app/contracts/command_result.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Typed, immutable result returned by CommandExecutor.execute().

    Replaces the raw ``dict[str, Any]`` return so callers get attribute
    access and type safety instead of string-keyed dict lookups.
    """

    ok: bool
    sid: Optional[str] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    template: Optional[str] = None
    attempts_made: int = 1
    idempotency_key: Optional[str] = None
    transport_only: bool = False
    transfer_number: Optional[str] = None

    # ── factory ──────────────────────────────────────────────────────────

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CommandResult:
        """Adapt a legacy result dict — used only in migration shims."""
        return cls(
            ok=bool(d.get("ok", False)),
            sid=d.get("sid"),
            error_code=d.get("error_code"),
            error_message=d.get("error_message"),
            template=d.get("template"),
            attempts_made=int(d.get("attempts_made") or 1),
            idempotency_key=d.get("idempotency_key"),
            transport_only=bool(d.get("transport_only", False)),
            transfer_number=d.get("transfer_number"),
        )
