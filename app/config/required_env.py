# app/config/required_env.py
"""Single source of truth for env vars that MUST be set in production.

Why this exists:
    Several runtime errors (Deepgram WS, Datacap auth, Twilio SMS) only
    surface when the first request arrives, because clients are constructed
    lazily inside request handlers. With gunicorn `--preload`, importing
    this module at app load time converts those latent failures into a
    boot-time crash, which is what we want — a broken deploy should fail
    `docker compose up`, not the first customer call.

Usage:
    - For boot-time validation: call ``assert_required_env_or_die()`` from
      the app entrypoint before the first request.
    - For runtime introspection (e.g. /healthz): call
      ``missing_required_env()`` and inspect the returned list.

Adding a new required variable:
    Append it to ``REQUIRED_ENV_VARS`` with a short description. Description
    is surfaced in error messages so on-call engineers know what's missing
    without grepping the codebase.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class RequiredEnv:
    name: str
    description: str


# Order roughly matches the request path so the first missing var hints at
# which subsystem will fail first.
REQUIRED_ENV_VARS: tuple[RequiredEnv, ...] = (
    RequiredEnv(
        name="DEEPGRAM_API_KEY",
        description="Deepgram API key for STT/TTS websocket auth.",
    ),
    RequiredEnv(
        name="TWILIO_ACCOUNT_SID",
        description="Twilio account SID for voice + SMS.",
    ),
    RequiredEnv(
        name="TWILIO_AUTH_TOKEN",
        description="Twilio auth token for voice + SMS.",
    ),
    RequiredEnv(
        name="TWILIO_SMS_FROM_NUMBER",
        description="Twilio number used as the sender for checkout/payment SMS.",
    ),
    RequiredEnv(
        name="DATACAP_BASIC_USERNAME",
        description="Datacap PaymentLinks API basic-auth username.",
    ),
    RequiredEnv(
        name="DATACAP_BASIC_PASSWORD",
        description="Datacap PaymentLinks API basic-auth password.",
    ),
)


def missing_required_env() -> list[RequiredEnv]:
    """Return the list of required env vars that are unset or blank."""
    missing: list[RequiredEnv] = []
    for var in REQUIRED_ENV_VARS:
        value = os.getenv(var.name, "").strip()
        if not value:
            missing.append(var)
    return missing


def assert_required_env_or_die() -> None:
    """Raise RuntimeError listing every missing required env var.

    Call this from the app entrypoint at import time (with gunicorn
    `--preload`, that means before workers fork). A failed deploy will
    crash the master process, so `docker compose up` exits non-zero and
    the deploy job fails — instead of silently producing a container that
    explodes on the first WebSocket connection.
    """
    missing = missing_required_env()
    if not missing:
        return

    lines = ["Refusing to start: required environment variables are not set."]
    for var in missing:
        lines.append(f"  - {var.name}: {var.description}")
    lines.append(
        "Hint: ensure /home/ubuntu/apps/compass-voice/.env exists and contains "
        "all of the above. See .env.example in the repo for the full schema."
    )
    raise RuntimeError("\n".join(lines))
