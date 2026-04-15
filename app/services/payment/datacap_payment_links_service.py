# app/services/payment/datacap_payment_links_service.py
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.models.payment_link_session import PaymentLinkSession


PAYMENT_LINK_SESSION_DIR = Path(
    os.getenv("COMPASS_PAYMENT_LINK_SESSION_DIR", "app/data/payment_link_sessions")
)
PAYMENT_LINK_SESSION_DIR.mkdir(parents=True, exist_ok=True)


class DatacapPaymentLinksService:
    def __init__(self) -> None:
        self.base_url = os.getenv(
            "DATACAP_PAYMENT_LINKS_BASE_URL",
            "https://paylink-cert.dcap.com/api/v1",
        ).rstrip("/")

        # Support either explicit Basic username/password
        # or fallback to MID/API key naming already used in your app.
        self.basic_username = (
            os.getenv("DATACAP_BASIC_USERNAME", "").strip()
            or os.getenv("DATACAP_ECOMMERCE_MID", "").strip()
        )
        self.basic_password = (
            os.getenv("DATACAP_BASIC_PASSWORD", "").strip()
            or os.getenv("DATACAP_API_KEY", "").strip()
        )

        self.merchant_name = os.getenv(
            "DATACAP_MERCHANT_NAME",
            "Cygnus Payments",
        ).strip()

        print(
            "[DATACAP PAYMENT LINKS CONFIG]",
            {
                "base_url": self.base_url,
                "has_basic_username": bool(self.basic_username),
                "has_basic_password": bool(self.basic_password),
                "merchant_name": self.merchant_name,
            },
        )

    def _auth_header(self) -> str:
        if not self.basic_username or not self.basic_password:
            raise ValueError("Datacap Basic auth username/password is missing.")

        raw = f"{self.basic_username}:{self.basic_password}".encode("utf-8")
        return "Basic " + base64.b64encode(raw).decode("ascii")

    def _session_path(self, session_id: str) -> Path:
        return PAYMENT_LINK_SESSION_DIR / f"{session_id}.json"

    def save_session(self, session: PaymentLinkSession) -> None:
        session.touch()
        self._session_path(session.id).write_text(
            json.dumps(session.to_dict(), indent=2),
            encoding="utf-8",
        )

    def create_payment_link(
        self,
        *,
        checkout_token: str,
        invoice_no: str,
        amount: str,
        credit_mid: str,
    ) -> PaymentLinkSession:
        normalized_amount = str(amount).replace("$", "").strip()

        payload: dict[str, Any] = {
            "transactionProperties": {
                "amount": normalized_amount,
                "invoiceNo": invoice_no,
            },
            "displayProperties": {
                "merchantName": self.merchant_name,
            },
            "paymentTypesAllowed": [
                {
                    "merchantId": credit_mid,
                    "paymentType": "CreditDebit",
                }
            ],
        }

        print(
            "[DATACAP PAYMENT LINK REQUEST]",
            {
                "url": f"{self.base_url}/paymentrequest",
                "payload": payload,
                "credit_mid_present": bool(credit_mid),
            },
        )

        request = Request(
            url=f"{self.base_url}/paymentrequest",
            method="POST",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": self._auth_header(),
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )

        try:
            with urlopen(request, timeout=20) as response:
                raw_body = response.read().decode("utf-8")
                body = json.loads(raw_body)
        except HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="ignore")
            raise ValueError(f"Datacap HTTP {exc.code}: {error_body}") from exc
        except URLError as exc:
            raise ValueError(f"Datacap network error: {exc}") from exc
        except Exception as exc:
            raise ValueError(
                f"Datacap payment-link request failed: {type(exc).__name__}: {exc}"
            ) from exc

        data = body[0] if isinstance(body, list) and body else body
        if not isinstance(data, dict):
            raise ValueError(
                f"Datacap returned an unexpected payment-link response: {body}"
            )

        public_link_url = data.get("publicLinkUrl")
        if not public_link_url:
            raise ValueError(
                f"Datacap response missing publicLinkUrl: {json.dumps(data)}"
            )

        session = PaymentLinkSession(
            checkout_token=checkout_token,
            invoice_no=invoice_no,
            amount=normalized_amount,
            request_id=data.get("requestId"),
            public_link_id=data.get("publicLinkId"),
            public_link_url=public_link_url,
            public_link_embedded_url=data.get("publicLinkEmbeddedUrl"),
            public_link_qr_code_url=data.get("publicLinkQRCodeUrl"),
            status=data.get("status", "Open"),
            payment_type_used=data.get("paymentTypeUsed"),
            provider_reference=data.get("requestId"),
            raw_response=data,
        )
        self.save_session(session)
        return session

    # ------------------------------------------------------------------
    # Payment-status verification (pull model)
    # ------------------------------------------------------------------

    # Datacap-reported statuses that mean money has been captured.
    PAID_STATUSES = {
        "paid",
        "complete",
        "completed",
        "succeeded",
        "success",
        "approved",
        "captured",
    }

    def get_payment_status(self, *, request_id: str) -> dict[str, Any]:
        """Query Datacap for the current status of a payment-request.

        Returns a normalized dict:
            {
                "ok":         bool,                # http request succeeded
                "paid":       bool,                # payment has been captured
                "status":     str,                 # raw status string from Datacap
                "reference":  str | None,          # best reference id we could extract
                "raw":        dict,                # full response body
                "error":      str | None,
            }
        """
        if not request_id:
            return {
                "ok": False,
                "paid": False,
                "status": "",
                "reference": None,
                "raw": {},
                "error": "request_id missing",
            }

        url = f"{self.base_url}/paymentrequest/{request_id}"
        request = Request(
            url=url,
            method="GET",
            headers={
                "Authorization": self._auth_header(),
                "Accept": "application/json",
            },
        )

        try:
            with urlopen(request, timeout=15) as response:
                raw_body = response.read().decode("utf-8")
                body = json.loads(raw_body) if raw_body else {}
        except HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="ignore")
            return {
                "ok": False,
                "paid": False,
                "status": "",
                "reference": None,
                "raw": {},
                "error": f"Datacap HTTP {exc.code}: {error_body}",
            }
        except URLError as exc:
            return {
                "ok": False,
                "paid": False,
                "status": "",
                "reference": None,
                "raw": {},
                "error": f"Datacap network error: {exc}",
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "ok": False,
                "paid": False,
                "status": "",
                "reference": None,
                "raw": {},
                "error": f"Datacap status request failed: {type(exc).__name__}: {exc}",
            }

        # Datacap sometimes returns a list-wrapped object — normalize.
        data = body[0] if isinstance(body, list) and body else body
        if not isinstance(data, dict):
            data = {}

        status_raw = str(
            data.get("status")
            or data.get("Status")
            or data.get("paymentStatus")
            or ""
        ).strip()

        # Detect paid state from multiple possible signals Datacap may send.
        paid = status_raw.lower() in self.PAID_STATUSES
        if not paid:
            # Fallback: look into nested paymentResult / transactions
            payment_result = data.get("paymentResult") or data.get("PaymentResult") or {}
            if isinstance(payment_result, dict):
                pr_status = str(
                    payment_result.get("status") or payment_result.get("Status") or ""
                ).lower()
                if pr_status in self.PAID_STATUSES:
                    paid = True
                # Some gateways use responseCode "000" for success
                resp_code = str(
                    payment_result.get("responseCode")
                    or payment_result.get("ResponseCode")
                    or ""
                ).strip()
                if resp_code == "000" or resp_code.lower() == "approved":
                    paid = True

            # Explicit boolean flags
            if data.get("paymentCompleted") is True or data.get("PaymentCompleted") is True:
                paid = True

        reference = (
            data.get("refNo")
            or data.get("RefNo")
            or data.get("referenceNumber")
            or data.get("transactionId")
            or data.get("TransactionId")
            or data.get("requestId")
            or request_id
        )

        return {
            "ok": True,
            "paid": paid,
            "status": status_raw or ("Paid" if paid else "Open"),
            "reference": reference,
            "raw": data,
            "error": None,
        }