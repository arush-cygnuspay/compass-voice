# tests/guards/test_dead_code_removal.py
"""Regression guards: dead caller-device flow must not re-appear in production source."""
from __future__ import annotations

import re
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[2] / "app"
THIS_FILE = Path(__file__).resolve()

_DEAD_SYMBOLS = [
    "WaitingForCallerDeviceTypeHandler",
    "WAITING_FOR_CALLER_DEVICE_TYPE",
    "WAITING_FOR_LANDLINE_PICKUP_CONFIRMATION",
    "waiting_for_caller_device_type_handler",
    "ask_for_caller_device_type",
    "repeat_caller_device_type",
    "confirm_landline_pickup_only",
]


def _collect_py_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.py"))


def test_no_dead_caller_device_symbols_in_production_source() -> None:
    """No file under app/ may reference any dead caller-device symbol."""
    violations: list[str] = []
    for path in _collect_py_files(APP_ROOT):
        text = path.read_text(encoding="utf-8")
        for symbol in _DEAD_SYMBOLS:
            if re.search(r"\b" + re.escape(symbol) + r"\b", text):
                violations.append(f"{path.relative_to(APP_ROOT.parent)}: {symbol!r}")
    assert not violations, (
        "Dead caller-device symbols found in production source:\n"
        + "\n".join(f"  {v}" for v in violations)
    )
