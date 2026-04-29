# bootstrap/runtime.py
"""Single application bootstrap entry point.

Import this module early in the process lifecycle (e.g. from the ASGI app
factory or the main entry point) to load all typed settings at once rather
than lazily on first use.  Subsequent calls to the individual ``get_*_config``
accessors return the already-cached instances.

Example::

    # In app startup / main.py:
    import bootstrap.runtime  # noqa: F401 — loads all config at startup

    from app.config.payment import get_payment_config
    cfg = get_payment_config()  # returns cached instance, no env reads
"""
from app.config.nlu import get_nlu_config
from app.config.payment import get_payment_config
from app.config.realtime import get_realtime_config

# Eagerly populate all caches so env vars are read exactly once,
# at a known point in time, rather than scattered across module imports.
nlu_config = get_nlu_config()
payment_config = get_payment_config()
realtime_config = get_realtime_config()

__all__ = [
    "nlu_config",
    "payment_config",
    "realtime_config",
]
