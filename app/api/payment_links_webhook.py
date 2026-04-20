# app/api/payment_links_webhook.py
"""Datacap payment-links webhook handler.

Datacap POSTs to this endpoint after a customer completes (or fails) a payment
on their hosted payment page.  The key event is ``payment.completed``.

Expected headers (Datacap sends these):
    X-Datacap-Event        — e.g. "payment.completed"
    X-Datacap-Webhook-Id   — unique ID for this delivery
    X-Datacap-Timestamp    — ISO-8601 timestamp
    X-Datacap-Signature    — HMAC-SHA256 signature (for future verification)

Expected JSON body (subset we care about):
    {
        "event": "payment.completed",
        "data": {
            "invoiceNo": "<order_number>",   ← this equals our order_number
            "requestId": "<datacap-id>",
            "refNo":     "<payment-ref>"
        }
    }
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Header, HTTPException, Request

from app.services.checkout_service import CheckoutService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/payment-links", tags=["payment-links"])

checkout_service = CheckoutService()


@router.post("/webhook")
async def payment_links_webhook(
    request: Request,
    x_datacap_signature: str | None = Header(default=None),
    x_datacap_timestamp: str | None = Header(default=None),
    x_datacap_event: str | None = Header(default=None),
    x_datacap_webhook_id: str | None = Header(default=None),
):
    """Handle inbound Datacap payment lifecycle events.

    On ``payment.completed``:
      1. Locate the checkout session by ``invoiceNo`` (== our order_number).
      2. Mark the session as payment-completed (idempotent).
      3. Send a Twilio SMS order-confirmation to the customer.
      4. Return 200 OK so Datacap knows the webhook was received.

    All other event types are acknowledged (200) but not acted upon.
    """
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    # ── Extract fields ────────────────────────────────────────────────────────
    # Datacap may send the event name either in a header or in the body.
    event_name = x_datacap_event or payload.get("event")
    data = payload.get("data") or payload

    # invoiceNo is always our order_number (digits-only string we created).
    invoice_no = (
        data.get("invoiceNo")
        or data.get("invoice_no")
        or data.get("InvoiceNo")
    )

    # Payment reference — use the most specific ID available.
    payment_reference = (
        data.get("refNo")
        or data.get("RefNo")
        or data.get("requestId")
        or data.get("RequestId")
        or x_datacap_webhook_id
    )

    logger.info(
        "Datacap webhook received: event=%s invoice_no=%s webhook_id=%s",
        event_name,
        invoice_no,
        x_datacap_webhook_id,
    )

    # ── Handle payment.completed ──────────────────────────────────────────────
    if event_name == "payment.completed":
        if not invoice_no:
            logger.warning(
                "payment.completed webhook missing invoiceNo — cannot identify order"
            )
            return {
                "ok": False,
                "reason": "invoiceNo missing in webhook payload",
                "event": event_name,
                "webhook_id": x_datacap_webhook_id,
            }

        # Delegate to CheckoutService which handles:
        #   • session lookup by order_number
        #   • idempotency check
        #   • marking payment_completed = True
        #   • sending Twilio SMS confirmation
        session = checkout_service.handle_payment_completed(
            order_number=invoice_no,
            payment_reference=payment_reference,
        )

        if session is None:
            # No matching session — log it but still return 200 so Datacap
            # doesn't retry indefinitely (it may belong to a different system).
            logger.warning(
                "payment.completed: no checkout session found for order_number=%s",
                invoice_no,
            )
            return {
                "ok": False,
                "reason": f"No checkout session found for order_number={invoice_no}",
                "event": event_name,
                "webhook_id": x_datacap_webhook_id,
            }

        return {
            "ok": True,
            "event": event_name,
            "order_number": invoice_no,
            "payment_reference": payment_reference,
            "session_token": session.token,
            "webhook_id": x_datacap_webhook_id,
        }

    # ── All other events — acknowledge without side-effects ──────────────────
    logger.info("Unhandled Datacap event '%s' — acknowledged.", event_name)
    return {
        "ok": True,
        "event": event_name,
        "webhook_id": x_datacap_webhook_id,
        "signature_present": bool(x_datacap_signature),
        "timestamp": x_datacap_timestamp,
        "invoice_no": invoice_no,
        "note": "Event type not handled; acknowledged only.",
    }
