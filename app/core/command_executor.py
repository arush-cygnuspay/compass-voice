# app/core/command_executor.py
"""
Executes side-effect commands returned by FSM handlers.

Commands are dicts with a "type" key and optional "payload".
Supported types: ADD_ITEM_TO_CART, CLEAR_CART, REMOVE_ITEM_FROM_CART,
SEND_SMS, transfer_call.
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from app.contracts.command_result import CommandResult
from app.session.session import Session
from app.services.sms_exceptions import PermanentSmsError, TransientSmsError
from app.services.sms_service import SmsSendRequest, SmsSendResult, SmsService

logger = logging.getLogger(__name__)

SMS_MAX_RETRIES = 2


def _derive_idempotency_key(session_id: str, command_id: str) -> str:
    """Return sha256(session_id:command_id) as a hex digest."""
    raw = f"{session_id}:{command_id}".encode()
    return hashlib.sha256(raw).hexdigest()


def _derive_command_id(payload: dict[str, Any]) -> str:
    """Deterministic ID for an SMS command from its payload contents.

    Using a content-hash means the same logical SMS always gets the same
    idempotency key within a session, even across process restarts.
    """
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


class CommandExecutor:
    """Stateless executor — delegates to the appropriate service per command type."""

    def __init__(self, sms_service: SmsService) -> None:
        self.sms_service = sms_service

    def execute(self, session: Session, command: dict[str, Any]) -> CommandResult:
        """Execute a handler command and return a typed CommandResult."""
        cmd_type = command.get("type")
        payload = command.get("payload") or {}

        if cmd_type == "ADD_ITEM_TO_CART":
            return self._add_item_to_cart(session, payload)

        if cmd_type == "CLEAR_CART":
            session.cart.clear()
            return CommandResult(ok=True)

        if cmd_type == "REMOVE_ITEM_FROM_CART":
            session.cart.remove_item(payload["cart_item_id"])
            return CommandResult(ok=True)

        if cmd_type == "SEND_SMS":
            return self._send_sms(session, payload)

        if cmd_type == "transfer_call":
            return CommandResult(
                ok=True,
                transport_only=True,
                transfer_number=command.get("transfer_number"),
            )

        raise ValueError(f"Unknown command type: {cmd_type}")

    # ── Private helpers ──────────────────────────────────────────────────

    @staticmethod
    def _add_item_to_cart(session: Session, payload: dict[str, Any]) -> CommandResult:
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
        return CommandResult(ok=True)

    def _send_sms(self, session: Session, payload: dict[str, Any]) -> CommandResult:
        template = payload["template"]
        session_id = getattr(session, "session_id", "") or ""

        command_id = _derive_command_id(payload)
        idempotency_key = _derive_idempotency_key(session_id, command_id)

        sms_request = SmsSendRequest(
            template=template,
            phone_number=payload["phone_number"],
            order_number=payload.get("order_number", ""),
            link=payload.get("link", ""),
            area=payload.get("area", ""),
            summary_text=payload.get("summary_text", ""),
            idempotency_key=idempotency_key,
        )

        sms_result: SmsSendResult | None = None
        attempts_made = 0
        last_error_code: str = "sms_send_failed"
        last_error_message: str = "SMS send failed."

        for attempt in range(1, SMS_MAX_RETRIES + 1):
            attempts_made = attempt
            try:
                sms_result = self.sms_service.send(sms_request)
                break  # success
            except PermanentSmsError as exc:
                last_error_code = exc.error_code or "sms_permanent_error"
                last_error_message = str(exc)
                logger.error(
                    "SMS permanent failure (no retry) attempt=%d template=%s "
                    "error_code=%s idempotency_key=%s: %s",
                    attempt, template, last_error_code, idempotency_key, exc,
                )
                break  # do not retry permanent failures
            except TransientSmsError as exc:
                last_error_code = exc.error_code or "sms_transient_error"
                last_error_message = str(exc)
                logger.warning(
                    "SMS transient failure attempt=%d/%d template=%s "
                    "error_code=%s idempotency_key=%s: %s",
                    attempt, SMS_MAX_RETRIES, template,
                    last_error_code, idempotency_key, exc,
                )
                # continue to next attempt

        return CommandResult(
            ok=sms_result is not None,
            sid=sms_result.sid if sms_result else None,
            error_code=None if sms_result else last_error_code,
            error_message=None if sms_result else last_error_message,
            template=template,
            attempts_made=attempts_made,
            idempotency_key=idempotency_key,
        )
