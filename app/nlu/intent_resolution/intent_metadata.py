# app/nlu/intent_resolution/intent_metadata.py
"""Single source of truth for per-intent behavioral metadata.

Adding a new intent here is the only change required to register its flow
policy (read-only interrupt, mid-item block, escalation) and its default
handler — no separate set or mapping needs updating.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from app.nlu.intent_resolution.intent import Intent


@dataclass(frozen=True, slots=True)
class IntentMeta:
    # Intent can interrupt mid-item flow and return a read-only answer before
    # resuming the active state (e.g. price check, cart summary).
    readonly: bool = False
    # Intent is a checkout-style signal that must be blocked while the user is
    # in the middle of building an item.
    blocks_mid_item: bool = False
    # Intent should immediately escalate the call to a human agent.
    escalates: bool = False
    # Handler key used when dispatching a read-only interrupt. None means the
    # intent is not routed through _handle_readonly_interrupt.
    default_handler: str | None = None


_DEFAULT = IntentMeta()

_ENTRIES: dict[Intent, IntentMeta] = {
    # ── Read-only interrupts ──────────────────────────────────────────────────
    # These four intents are handled inline during any active state without
    # abandoning the current flow.
    Intent.ASK_PRICE: IntentMeta(
        readonly=True,
        default_handler="ask_price_handler",
    ),
    Intent.SHOW_CART: IntentMeta(
        readonly=True,
        default_handler="cart_handler",
    ),
    Intent.SHOW_TOTAL: IntentMeta(
        readonly=True,
        default_handler="cart_handler",
    ),
    Intent.AVAILABILITY_QUERY: IntentMeta(
        readonly=True,
        default_handler="ask_menu_info_handler",
    ),

    # ── Menu / info intents (not read-only interrupts, but have a handler) ───
    # These reach _handle_readonly_interrupt only when FlowControlPolicy
    # returns HANDLE_READONLY_INTERRUPT, which currently only fires for the
    # four readonly=True intents above. The default_handler is kept here so
    # that future policy changes require no further edits.
    Intent.ASK_ITEM_INFO: IntentMeta(default_handler="ask_menu_info_handler"),
    Intent.ASK_MENU_INFO: IntentMeta(default_handler="ask_menu_info_handler"),
    Intent.ASK_OPTIONS: IntentMeta(default_handler="ask_menu_info_handler"),
    Intent.BROWSE_MENU: IntentMeta(default_handler="ask_menu_info_handler"),
    Intent.BROWSE_CATEGORY: IntentMeta(default_handler="ask_menu_info_handler"),
    Intent.RECOMMENDATION_QUERY: IntentMeta(default_handler="ask_menu_info_handler"),
    Intent.SHOW_MENU: IntentMeta(default_handler="ask_menu_info_handler"),

    # ── Mid-item blockers ─────────────────────────────────────────────────────
    # Checkout-style intents that are blocked while the caller is mid-item.
    Intent.START_ORDER: IntentMeta(blocks_mid_item=True),
    Intent.END_ADDING: IntentMeta(blocks_mid_item=True),
    Intent.CHECKOUT: IntentMeta(blocks_mid_item=True),
    Intent.CONFIRM_ORDER: IntentMeta(blocks_mid_item=True),
    Intent.FINISH_ORDER: IntentMeta(blocks_mid_item=True),
    Intent.PAYMENT_REQUEST: IntentMeta(blocks_mid_item=True),
    Intent.REVIEW_ORDER: IntentMeta(blocks_mid_item=True),

    # ── Escalation ────────────────────────────────────────────────────────────
    Intent.REQUEST_AGENT: IntentMeta(escalates=True),
}


class IntentMetadataRegistry:
    def __init__(self, entries: dict[Intent, IntentMeta]) -> None:
        self._entries = entries

    def get(self, intent: Intent) -> IntentMeta:
        return self._entries.get(intent, _DEFAULT)


INTENT_REGISTRY: Final[IntentMetadataRegistry] = IntentMetadataRegistry(_ENTRIES)
