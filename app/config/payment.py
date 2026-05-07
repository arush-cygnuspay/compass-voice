# app/config/payment.py
"""Typed payment/checkout settings loaded once from environment variables.

Usage::

    from app.config.payment import get_payment_config
    cfg = get_payment_config()
    print(cfg.checkout_data_dir)

``get_payment_config()`` is cached — env vars are read on the first call only.
Swap ``PaymentConfig`` for a pydantic BaseSettings subclass if pydantic-settings
is added to the project dependencies.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


def _load_interval(*env_names: str, default: float) -> float:
    """Load a cooldown interval from the first non-empty env var in *env_names*.

    Enforces a minimum of 20 s for any positive value so the payment poller
    never busy-loops, and returns 0.0 exactly when the env var is set to 0.
    """
    for env_name in env_names:
        raw = os.getenv(env_name)
        if raw is None or not raw.strip():
            continue
        value = float(raw)
        if value <= 0:
            return 0.0
        return max(20.0, value)
    return max(20.0, default)


@dataclass(frozen=True)
class PaymentConfig:
    """Immutable payment/checkout settings snapshot."""

    checkout_data_dir: Path
    payment_link_session_data_dir: Path
    payment_poll_interval_seconds: float
    payment_poll_max_duration_seconds: float
    public_checkout_base_url: str
    reverse_geocode_url: str
    reverse_geocode_user_agent: str
    checkout_pending_reminder_interval_seconds: float
    payment_pending_reminder_interval_seconds: float


@lru_cache(maxsize=1)
def get_payment_config() -> PaymentConfig:
    """Return the singleton PaymentConfig, loading env vars on first call."""
    return PaymentConfig(
        checkout_data_dir=Path(
            os.getenv("COMPASS_CHECKOUT_DATA_DIR", "app/data/checkout_sessions")
        ),
        payment_link_session_data_dir=Path(
            os.getenv(
                "COMPASS_PAYMENT_LINK_SESSION_DATA_DIR",
                "app/data/payment_link_sessions",
            )
        ),
        payment_poll_interval_seconds=float(
            os.getenv("COMPASS_PAYMENT_POLL_INTERVAL", "6")
        ),
        payment_poll_max_duration_seconds=float(
            os.getenv("COMPASS_PAYMENT_POLL_MAX_DURATION", "900")
        ),
        public_checkout_base_url=os.getenv(
            "COMPASS_PUBLIC_CHECKOUT_BASE_URL",
            "https://6184-2407-aa80-116-3caa-20d1-a207-5dbd-6875.ngrok-free.app/checkout",
        ).rstrip("/"),
        reverse_geocode_url=os.getenv(
            "COMPASS_REVERSE_GEOCODE_URL",
            "https://nominatim.openstreetmap.org/reverse",
        ).strip(),
        reverse_geocode_user_agent=os.getenv(
            "COMPASS_REVERSE_GEOCODE_USER_AGENT",
            "CompassCheckout/1.0 (support@cygnuspayments.com)",
        ).strip(),
        checkout_pending_reminder_interval_seconds=_load_interval(
            "CHECKOUT_PENDING_REMINDER_INTERVAL_SECONDS",
            "COMPASS_CHECKOUT_PENDING_REMINDER_INTERVAL_SECONDS",
            "COMPASS_PAYMENT_STATUS_PROMPT_COOLDOWN_SECONDS",
            default=30.0,
        ),
        payment_pending_reminder_interval_seconds=_load_interval(
            "PAYMENT_PENDING_REMINDER_INTERVAL_SECONDS",
            "COMPASS_PAYMENT_PENDING_REMINDER_INTERVAL_SECONDS",
            "COMPASS_PAYMENT_STATUS_PROMPT_COOLDOWN_SECONDS",
            default=30.0,
        ),
    )
