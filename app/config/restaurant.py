# app/config/restaurant.py
"""Active-restaurant configuration.

Exposes the canonical restaurant identifier used by the runtime, transport
layer, CLI, and API surface. Resolved once from the ``COMPASS_RESTAURANT_ID``
environment variable so a deployment can pin a different tenant without code
changes. The default points at the bundled ``steves_grill`` data root
(``app/data/restaurants/steves_grill/{menu.json,entity_index.json}``).

Usage::

    from app.config.restaurant import DEFAULT_RESTAURANT_ID
    runtime = build_runtime(restaurant_id=DEFAULT_RESTAURANT_ID)

Keep this module dependency-free so it can be imported from any layer
(transport, bootstrap, services) without introducing import cycles.
"""
from __future__ import annotations

import os

# Resolved once at import time. Tests can override by re-importing after
# mutating the env var, but production deployments should set this in the
# process environment before startup.
DEFAULT_RESTAURANT_ID: str = (
    os.getenv("COMPASS_RESTAURANT_ID", "steves_grill").strip() or "steves_grill"
)


def get_default_restaurant_id() -> str:
    """Return the active restaurant id (env override honored at call time).

    Prefer the module-level ``DEFAULT_RESTAURANT_ID`` constant for normal use;
    this helper exists for callers that need late-binding (e.g. test harnesses
    that mutate the env var after import).
    """
    return (os.getenv("COMPASS_RESTAURANT_ID") or DEFAULT_RESTAURANT_ID).strip()
