# app/repositories/checkout_session_repository.py
"""Checkout session and payment-link session persistence.

Owns all file I/O, path construction, and index management for checkout
sessions and payment-link sessions.  No business logic lives here.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from app.models.checkout_session import CheckoutSession
from app.models.payment_link_session import PaymentLinkSession

logger = logging.getLogger(__name__)

# Sentinel raised by callers that need to distinguish missing vs expired.
class CheckoutNotFoundError(Exception):
    pass


class CheckoutExpiredError(Exception):
    pass


def _lookup_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class CheckoutSessionRepository:
    """File-system backed store for CheckoutSession and PaymentLinkSession objects.

    All path construction is derived from the two root directories supplied at
    construction time, so tests can inject temporary directories without
    patching module-level constants.
    """

    def __init__(
        self,
        data_dir: Path,
        payment_link_session_dir: Path,
    ) -> None:
        self._data_dir = data_dir
        self._payment_link_session_dir = payment_link_session_dir

        # Derived index directories — computed from root dirs so no extra
        # patching is needed in tests.
        self._checkout_index_dir = data_dir / "_indexes"
        self._checkout_order_index_dir = self._checkout_index_dir / "by_order_number"
        self._payment_link_index_dir = payment_link_session_dir / "_indexes"
        self._payment_link_by_checkout_index_dir = (
            self._payment_link_index_dir / "latest_by_checkout_token"
        )
        self._payment_link_by_request_index_dir = (
            self._payment_link_index_dir / "by_request_id"
        )

        for directory in (
            self._checkout_index_dir,
            self._checkout_order_index_dir,
            self._payment_link_index_dir,
            self._payment_link_by_checkout_index_dir,
            self._payment_link_by_request_index_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _path_for_token(self, token: str) -> Path:
        return self._data_dir / f"{token}.json"

    def _payment_link_session_path(self, session_id: str) -> Path:
        return self._payment_link_session_dir / f"{session_id}.json"

    def _order_index_path(self, order_number: str) -> Path:
        return self._checkout_order_index_dir / f"{_lookup_key(order_number)}.json"

    def _payment_link_latest_index_path(self, checkout_token: str) -> Path:
        return (
            self._payment_link_by_checkout_index_dir
            / f"{_lookup_key(checkout_token)}.json"
        )

    def _payment_link_request_index_path(self, request_id: str) -> Path:
        return (
            self._payment_link_by_request_index_dir
            / f"{_lookup_key(request_id)}.json"
        )

    # ------------------------------------------------------------------
    # Raw JSON I/O
    # ------------------------------------------------------------------

    def _write_json(
        self,
        path: Path,
        payload: dict[str, Any],
        *,
        compact: bool = False,
    ) -> None:
        if compact:
            text = json.dumps(payload, separators=(",", ":"))
        else:
            text = json.dumps(payload, indent=2)
        path.write_text(text, encoding="utf-8")

    def _read_json(self, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not parse JSON file %s: %s", path.name, exc)
            return None

    # ------------------------------------------------------------------
    # Deserializers
    # ------------------------------------------------------------------

    def _load_checkout_session_file(self, path: Path) -> CheckoutSession | None:
        data = self._read_json(path)
        if not data:
            return None
        try:
            return CheckoutSession.from_dict(data)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not parse checkout session %s: %s", path.name, exc)
            return None

    def _load_payment_link_session_file(self, path: Path) -> PaymentLinkSession | None:
        data = self._read_json(path)
        if not data:
            return None
        try:
            return PaymentLinkSession.from_dict(data)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not parse payment link session %s: %s", path.name, exc
            )
            return None

    # ------------------------------------------------------------------
    # Index persistence
    # ------------------------------------------------------------------

    def _save_session_indexes(self, session: CheckoutSession) -> None:
        if not session.order_number:
            return
        self._write_json(
            self._order_index_path(session.order_number),
            {
                "order_number": session.order_number,
                "token": session.token,
                "updated_at": session.updated_at.isoformat(),
            },
            compact=True,
        )

    def _save_payment_link_indexes(self, payment_link_session: PaymentLinkSession) -> None:
        if payment_link_session.checkout_token:
            self._write_json(
                self._payment_link_latest_index_path(payment_link_session.checkout_token),
                {
                    "checkout_token": payment_link_session.checkout_token,
                    "session_id": payment_link_session.id,
                    "updated_at": payment_link_session.updated_at.isoformat(),
                },
                compact=True,
            )

        if payment_link_session.request_id:
            self._write_json(
                self._payment_link_request_index_path(payment_link_session.request_id),
                {
                    "request_id": payment_link_session.request_id,
                    "session_id": payment_link_session.id,
                    "updated_at": payment_link_session.updated_at.isoformat(),
                },
                compact=True,
            )

    # ------------------------------------------------------------------
    # CheckoutSession CRUD
    # ------------------------------------------------------------------

    def save_session(self, session: CheckoutSession) -> None:
        session.touch()
        self._write_json(self._path_for_token(session.token), session.to_dict())
        self._save_session_indexes(session)

    def get_session(self, token: str) -> CheckoutSession:
        path = self._path_for_token(token)
        session = self._load_checkout_session_file(path)
        if session is None:
            raise CheckoutNotFoundError(token)
        if session.is_expired():
            raise CheckoutExpiredError(token)
        return session

    def find_session_by_order_number(
        self, order_number: str | None
    ) -> CheckoutSession | None:
        if not order_number:
            return None

        indexed = self._read_json(self._order_index_path(order_number))
        if indexed:
            token = indexed.get("token")
            if token:
                try:
                    session = self.get_session(token)
                except (CheckoutNotFoundError, CheckoutExpiredError):
                    session = None
                if session and session.order_number == order_number:
                    return session

        return self._scan_session_by_order_number(order_number)

    def _scan_session_by_order_number(
        self, order_number: str
    ) -> CheckoutSession | None:
        for path in self._data_dir.glob("*.json"):
            session = self._load_checkout_session_file(path)
            if session is None:
                continue
            if session.order_number == order_number and not session.is_expired():
                self._save_session_indexes(session)
                return session
        return None

    # ------------------------------------------------------------------
    # PaymentLinkSession CRUD
    # ------------------------------------------------------------------

    def save_payment_link_session(
        self, payment_link_session: PaymentLinkSession
    ) -> None:
        payment_link_session.touch()
        self._write_json(
            self._payment_link_session_path(payment_link_session.id),
            payment_link_session.to_dict(),
        )
        self._save_payment_link_indexes(payment_link_session)

    def find_latest_payment_link_session(
        self, checkout_token: str
    ) -> PaymentLinkSession | None:
        indexed = self._read_json(
            self._payment_link_latest_index_path(checkout_token)
        )
        if indexed:
            session_id = indexed.get("session_id")
            if session_id:
                session = self._load_payment_link_session_file(
                    self._payment_link_session_path(session_id)
                )
                if session and session.checkout_token == checkout_token:
                    return session

        return self._scan_latest_payment_link_session(checkout_token)

    def _scan_latest_payment_link_session(
        self, checkout_token: str
    ) -> PaymentLinkSession | None:
        candidates: list[tuple[float, PaymentLinkSession]] = []
        for path in self._payment_link_session_dir.glob("*.json"):
            payment_link_session = self._load_payment_link_session_file(path)
            if payment_link_session is None:
                continue
            if payment_link_session.checkout_token == checkout_token:
                candidates.append(
                    (
                        payment_link_session.created_at.timestamp(),
                        payment_link_session,
                    )
                )

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0], reverse=True)
        latest = candidates[0][1]
        self._save_payment_link_indexes(latest)
        return latest
