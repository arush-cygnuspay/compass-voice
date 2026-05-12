# app/nlu/prompt_type.py
"""Semantic prompt-type classification for last_prompt_type tracking.

Maps response_key values to a small set of semantic categories that the
contextual control resolver uses to correctly interpret the *next* user turn.
"""
from __future__ import annotations

from enum import Enum


class PromptType(str, Enum):
    ANYTHING_ELSE = "anything_else"           # bot: "Anything else?" after item added
    CONFIRM_ORDER = "confirm_order"           # bot: "Your order is X, ready to checkout?"
    PICKUP_SMS_PERMISSION = "pickup_sms_permission"  # bot: "Want me to text you a receipt?"
    PAYMENT_LINK_SENT = "payment_link_sent"   # bot: "I've sent you a payment link"
    UNKNOWN = "unknown"


# Maps response_key → PromptType.
# Only keys that meaningfully shape the next user turn are listed.
_RESPONSE_KEY_TO_PROMPT_TYPE: dict[str, PromptType] = {
    "item_added_successfully": PromptType.ANYTHING_ELSE,
    "item_added_with_quantity": PromptType.ANYTHING_ELSE,
    "confirm_order_summary": PromptType.CONFIRM_ORDER,
    "confirm_order_summary_repeat": PromptType.CONFIRM_ORDER,
    "confirm_order_summary_unclear": PromptType.CONFIRM_ORDER,
    "pickup_ask_sms_permission": PromptType.PICKUP_SMS_PERMISSION,
    "payment_link_sent": PromptType.PAYMENT_LINK_SENT,
    "checkout_link_sent": PromptType.PAYMENT_LINK_SENT,
}


def classify_response_key(response_key: str | None) -> PromptType:
    """Return the PromptType for *response_key*, defaulting to UNKNOWN."""
    if response_key is None:
        return PromptType.UNKNOWN
    return _RESPONSE_KEY_TO_PROMPT_TYPE.get(response_key, PromptType.UNKNOWN)
