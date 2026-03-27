# app/session/repository.py

from __future__ import annotations

import json
import os

import redis

from app.session.session import Session

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
SESSION_TTL_SECONDS = 60 * 60  # 1 hour

_redis = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    decode_responses=True,
)

# =================================================
# Public API
# =================================================

def load_session(session_id: str, restaurant_id: str) -> Session:
    key = _key(session_id)
    raw = _redis.get(key)

    if not raw:
        return Session(session_id=session_id, restaurant_id=restaurant_id)

    session = Session.from_dict(json.loads(raw))

    if session.restaurant_id != restaurant_id:
        return Session(session_id=session_id, restaurant_id=restaurant_id)

    return session


def save_session(session: Session) -> None:
    key = _key(session.session_id)
    _redis.setex(key, SESSION_TTL_SECONDS, json.dumps(session.to_dict()))


def _key(session_id: str) -> str:
    return f"session:{session_id}"
