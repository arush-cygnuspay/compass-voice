# app/core/sms_command_fallback.py
"""Business logic for SEND_SMS command failures.

When a SEND_SMS command fails, the turn engine needs to decide what response
to give the caller — checkout-link failure falls back to voice collection,
payment-link failure stays in the current state with an apology.

Extracted from TurnEngine.process_turn so that retry/fallback rules live in
one focused place rather than inline in a 1 200-line orchestrator method.
"""
from __future__ import annotations

from app.contracts.command_result import CommandResult
from app.session.session import Session
from app.state_machine.handler_result import HandlerResult
from app.state_machine.models.conversation_state import ConversationState


def resolve_sms_failure(
    session: Session,
    result: HandlerResult,
    command: dict,
    command_result: CommandResult,
) -> HandlerResult | None:
    """Return a replacement HandlerResult when an SMS command fails.

    Returns ``None`` if the failure does not warrant a response override
    (e.g. the command type is not SEND_SMS).  The caller should use the
    returned result instead of the original when the return value is not None.
    """
    command_type = command.get("type")
    if command_type != "SEND_SMS":
        return None

    command_payload = command.get("payload") or {}
    template = command_payload.get("template")
    delivery = session.conversation_context.delivery_address
    attempts_made = command_result.attempts_made

    if template == "checkout_link":
        # _apply_command already retried internally.  If it still failed after
        # those attempts, fall back to voice-collected address immediately.
        if attempts_made >= 2:
            delivery.source = "voice"
            session.conversation_context.current_prompt_field = "delivery_seed_confirmation"
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_DELIVERY_ADDRESS_COLLECTION,
                response_key="checkout_link_failed_fallback_voice",
                response_payload={
                    "area": delivery.area,
                    "postal_code": delivery.postal_code,
                    "order_number": delivery.order_number,
                    "error_code": command_result.error_code,
                    "error_message": command_result.error_message,
                },
            )
        return HandlerResult(
            next_state=ConversationState.CONFIRMING_ORDER,
            response_key="checkout_link_send_failed",
            response_payload={
                "order_number": delivery.order_number,
                "error_code": command_result.error_code,
                "error_message": command_result.error_message,
            },
        )

    if template == "payment_link":
        # Payment-link flow also retried internally.
        # If still failed, apologize and hold current state.
        if attempts_made >= 2:
            return HandlerResult(
                next_state=session.conversation_state,
                response_key="payment_link_unavailable_now",
                response_payload={
                    "order_number": delivery.order_number,
                    "error_code": command_result.error_code,
                    "error_message": command_result.error_message,
                },
            )
        return HandlerResult(
            next_state=session.conversation_state,
            response_key="payment_link_send_failed",
            response_payload={
                "order_number": delivery.order_number,
                "error_code": command_result.error_code,
                "error_message": command_result.error_message,
            },
        )

    return None
