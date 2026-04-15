# app/session/repository.py

from __future__ import annotations

import json
import os

import redis

from app.session.session import Session
from app.state_machine.models.conversation_state import ConversationState

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

def _new_order_type_session(session_id: str, restaurant_id: str) -> Session:
    session = Session(session_id=session_id, restaurant_id=restaurant_id)
    session.conversation_state = ConversationState.WAITING_FOR_ORDER_TYPE
    session.conversation_context.order_type = None
    session.conversation_context.onboarding_complete = False
    session.conversation_context.delivery_address_required = False
    session.conversation_context.delivery_address_confirmed = False
    return session


def load_session(session_id: str, restaurant_id: str) -> Session:
    key = _key(session_id)
    raw = _redis.get(key)

    if not raw:
        return _new_order_type_session(session_id, restaurant_id)

    session = Session.from_dict(json.loads(raw))

    if session.restaurant_id != restaurant_id:
        return _new_order_type_session(session_id, restaurant_id)

    return session


def load_existing_session(session_id: str, restaurant_id: str) -> Session | None:
    key = _key(session_id)
    raw = _redis.get(key)

    if not raw:
        return None

    session = Session.from_dict(json.loads(raw))
    if session.restaurant_id != restaurant_id:
        return None

    return session


def save_session(session: Session) -> None:
    key = _key(session.session_id)
    _redis.setex(key, SESSION_TTL_SECONDS, json.dumps(session.to_dict()))


def _key(session_id: str) -> str:
    return f"session:{session_id}"
