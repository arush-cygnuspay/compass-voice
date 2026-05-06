# app/api/health_routes.py
"""Health endpoint used by Docker healthcheck and the deploy gate.

Exposes ``GET /healthz`` which returns 200 only when the process is
ready to serve real traffic — i.e. all required env vars are present
and the persistent state directories are writable.

This is intentionally cheap (no external network calls) so the docker
healthcheck can poll it on a 30s cadence without hammering Deepgram
or Datacap. Deeper smoke tests belong in a separate ``/readyz``.
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.config.required_env import missing_required_env

router = APIRouter(tags=["health"])


def _check_writable_dir(env_name: str, default: str) -> dict | None:
    """Return a problem dict if the directory is missing or not writable."""
    raw = os.getenv(env_name, default).strip()
    path = Path(raw)
    if not path.is_dir():
        return {"check": "writable_dir", "env": env_name, "path": str(path), "reason": "not_a_directory"}
    if not os.access(path, os.W_OK):
        return {"check": "writable_dir", "env": env_name, "path": str(path), "reason": "not_writable"}
    return None


@router.get("/healthz")
def healthz() -> JSONResponse:
    problems: list[dict] = []

    for var in missing_required_env():
        problems.append({"check": "required_env", "env": var.name, "reason": "unset_or_blank"})

    for env_name, default in (
        ("COMPASS_CHECKOUT_DATA_DIR", "app/data/checkout_sessions"),
        ("COMPASS_PAYMENT_LINK_SESSION_DATA_DIR", "app/data/payment_link_sessions"),
    ):
        problem = _check_writable_dir(env_name, default)
        if problem:
            problems.append(problem)

    if problems:
        return JSONResponse(status_code=503, content={"ok": False, "problems": problems})
    return JSONResponse(status_code=200, content={"ok": True})
