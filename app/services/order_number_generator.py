# app/services/order_number_generator.py
"""Injectable, deterministic order-number generator.

Wraps the module-level generate_order_number() from the checkout session
model.  Accepting injected clock and randint dependencies makes this
trivially testable without monkey-patching.

Format (preserved from original): 7 digits — 2 random prefix + 5 timestamp
  e.g. 4738291
"""
from __future__ import annotations

import random as _random_module
from datetime import datetime, timezone
from typing import Callable


def _default_clock() -> datetime:
    return datetime.now(timezone.utc)


class OrderNumberGenerator:
    """Generate 7-digit numeric order numbers.

    Parameters
    ----------
    clock:
        Zero-argument callable returning a UTC datetime.  Defaults to
        ``datetime.now(timezone.utc)``.
    randint:
        Two-argument callable ``(a, b) -> int`` mirroring ``random.randint``.
        Defaults to ``random.randint``.
    """

    def __init__(
        self,
        *,
        clock: Callable[[], datetime] | None = None,
        randint: Callable[[int, int], int] | None = None,
    ) -> None:
        self._clock = clock or _default_clock
        self._randint = randint or _random_module.randint

    def generate(self, order_number: str | None = None) -> str:
        """Return *order_number* unchanged if it is already a digit string,
        otherwise mint a fresh 7-digit order number."""
        if order_number and order_number.isdigit():
            return order_number

        ts_part = int(self._clock().timestamp() * 100) % 100_000  # 5 digits
        rnd_part = self._randint(10, 99)                            # 2-digit prefix
        return f"{rnd_part}{ts_part:05d}"
