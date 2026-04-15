from __future__ import annotations

import logging
import os
from xml.sax.saxutils import escape as xml_escape

from twilio.rest import Client


logger = logging.getLogger(__name__)


class LiveCallService:
    def __init__(self) -> None:
        self._account_sid = os.getenv("TWILIO_ACCOUNT_SID", "").strip()
        self._auth_token = os.getenv("TWILIO_AUTH_TOKEN", "").strip()

        self._client: Client | None = None
        if self._account_sid and self._auth_token:
            self._client = Client(self._account_sid, self._auth_token)

    def is_configured(self) -> bool:
        return self._client is not None

    def announce_order_completed(
        self,
        *,
        call_sid: str | None,
        order_number: str | None,
    ) -> bool:
        normalized_call_sid = (call_sid or "").strip()
        if not normalized_call_sid or not self.is_configured():
            return False

        message = self._build_completion_message(order_number)
        twiml = f"<Response><Say>{xml_escape(message)}</Say><Hangup/></Response>"

        try:
            self._client.calls(normalized_call_sid).update(twiml=twiml)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to announce order completion on live call %s: %s",
                normalized_call_sid,
                exc,
            )
            return False

    def _build_completion_message(self, order_number: str | None) -> str:
        normalized_order_number = (order_number or "").strip()
        order_sentence = ""
        if normalized_order_number:
            order_sentence = (
                f" Your order number is {self._spoken_order_number(normalized_order_number)}."
            )

        return (
            f"Payment confirmed.{order_sentence} "
            "Your order has been placed successfully. Will be ready in 25 minutes. Thank you."
        )

    def _spoken_order_number(self, order_number: str) -> str:
        if order_number.isdigit():
            return " ".join(order_number)
        return order_number
