# app/core/command_executor.py
"""
Executes side-effect commands returned by FSM handlers.

Commands are dicts with a "type" key and optional "payload".
Supported types: ADD_ITEM_TO_CART, CLEAR_CART, REMOVE_ITEM_FROM_CART,
SEND_SMS, transfer_call.
"""
from __future__ import annotations

import logging
from typing import Any

from app.session.session import Session
from app.services.sms_service import SmsSendRequest, SmsSendResult, SmsService

logger = logging.getLogger(__name__)

SMS_MAX_RETRIES = 2


class CommandExecutor:
    """Stateless executor — delegates to the appropriate service per command type."""

    def __init__(self, sms_service: SmsService) -> None:
        self.sms_service = sms_service

    def execute(self, session: Session, command: dict[str, Any]) -> dict[str, Any]:
        """
        Execute a handler command and return a result dict.

        Always returns {"ok": bool, ...}. Raises ValueError for unknown types.
        """
        cmd_type = command.get("type")
        payload = command.get("payload") or {}

        if cmd_type == "ADD_ITEM_TO_CART":
            return self._add_item_to_cart(session, payload)

        if cmd_type == "CLEAR_CART":
            session.cart.clear()
            return {"ok": True}

        if cmd_type == "REMOVE_ITEM_FROM_CART":
            session.cart.remove_item(payload["cart_item_id"])
            return {"ok": True}

        if cmd_type == "SEND_SMS":
            return self._send_sms(payload)

        if cmd_type == "transfer_call":
            return {
                "ok": True,
                "transport_only": True,
                "transfer_number": command.get("transfer_number"),
            }

        raise ValueError(f"Unknown command type: {cmd_type}")

    # ── Private helpers ──────────────────────────────────────────────────

    @staticmethod
    def _add_item_to_cart(session: Session, payload: dict[str, Any]) -> dict[str, Any]:
        from app.cart.cart_item import CartItem

        cart_item = CartItem.create(
            item_id=payload["item_id"],
            quantity=payload["quantity"],
            variant_id=payload.get("variant_id"),
            sides=payload.get("sides", {}),
            side_variants=payload.get("side_variants", {}),
            modifiers=payload.get("modifiers", {}),
        )
        session.cart.add_item(cart_item)
        return {"ok": True}

    def _send_sms(self, payload: dict[str, Any]) -> dict[str, Any]:
        template = payload["template"]

        sms_request = SmsSendRequest(
            template=template,
            phone_number=payload["phone_number"],
            order_number=payload.get("order_number", ""),
            link=payload.get("link", ""),
            area=payload.get("area", ""),
            summary_text=payload.get("summary_text", ""),
        )

        sms_result: SmsSendResult | None = None
        attempts_made = 0

        for _ in range(SMS_MAX_RETRIES):
            attempts_made += 1
            try:
                sms_result = self.sms_service.send(sms_request)
                if sms_result.ok:
                    break
            except Exception:
                logger.exception(
                    "SMS send attempt %d failed for template=%s",
                    attempts_made, template,
                )

        return {
            "ok": bool(sms_result and sms_result.ok),
            "sid": sms_result.sid if sms_result else None,
            "error_code": sms_result.error_code if sms_result else "sms_send_failed",
            "error_message": sms_result.error_message if sms_result else "SMS send failed.",
            "template": template,
            "attempts_made": attempts_made,
        }
