# app/session/repository.py

from __future__ import annotations

import json
import os
import redis

from app.session.session import Session
from app.state_machine.conversation_state import ConversationState
from app.state_machine.context import ConversationContext
from app.cart.cart import Cart
from app.nlu.intent_resolution.intent import Intent

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
        return Session(
            session_id=session_id,
            restaurant_id=restaurant_id,
        )

    return _deserialize(raw)


def save_session(session: Session) -> None:
    key = _key(session.session_id)

    _redis.setex(
        key,
        SESSION_TTL_SECONDS,
        _serialize(session),
    )


# =================================================
# Serialization
# =================================================

def _serialize(session: Session) -> str:
    return json.dumps({
        "session_id": session.session_id,
        "restaurant_id": session.restaurant_id,

        "conversation_state": session.conversation_state.value,

        "conversation_context": {
            "current_item_id": session.conversation_context.current_item_id,
            "candidate_item_id": session.conversation_context.candidate_item_id,
            "selected_variant_id": session.conversation_context.selected_variant_id,
            "selected_side_groups": session.conversation_context.selected_side_groups,
            "selected_modifier_groups": session.conversation_context.selected_modifier_groups,
            "quantity": session.conversation_context.quantity,
            "pending_action": session.conversation_context.pending_action,
            "awaiting_confirmation_for": session.conversation_context.awaiting_confirmation_for,
            "skipped_modifier_groups": list(
                session.conversation_context.skipped_modifier_groups
            ),
        },

        "cart": session.cart.to_dict(),

        "turn_count": session.turn_count,
        "last_intent": session.last_intent.value if session.last_intent else None,
        "last_response_key": session.last_response_key,
    })


def _deserialize(raw: str) -> Session:
    data = json.loads(raw)

    session = Session(
        session_id=data["session_id"],
        restaurant_id=data["restaurant_id"],
        conversation_state=ConversationState(data["conversation_state"]),
    )

    # Restore context
    ctx = ConversationContext()
    ctx.current_item_id = data["conversation_context"].get("current_item_id")
    ctx.candidate_item_id = data["conversation_context"].get("candidate_item_id")
    ctx.selected_variant_id = data["conversation_context"].get("selected_variant_id")
    ctx.selected_side_groups = data["conversation_context"].get(
        "selected_side_groups", {}
    )
    ctx.selected_modifier_groups = data["conversation_context"].get(
        "selected_modifier_groups", {}
    )
    ctx.quantity = data["conversation_context"].get("quantity")
    ctx.pending_action = data["conversation_context"].get("pending_action")
    ctx.awaiting_confirmation_for = data["conversation_context"].get(
        "awaiting_confirmation_for"
    )
    ctx.skipped_modifier_groups = set(
        data["conversation_context"].get("skipped_modifier_groups", [])
    )

    session.conversation_context = ctx

    # Restore cart
    session.cart = Cart.from_dict(data["cart"])

    session.turn_count = data.get("turn_count", 0)

    if data.get("last_intent"):
        session.last_intent = Intent(data["last_intent"])

    session.last_response_key = data.get("last_response_key")

    return session


def _key(session_id: str) -> str:
    return f"session:{session_id}"

