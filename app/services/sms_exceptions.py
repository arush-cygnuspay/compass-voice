# app/services/sms_exceptions.py
"""Typed exceptions for SMS send failures.

Callers should catch these in preference to bare ``Exception`` so that
retry policy can distinguish retriable from non-retriable conditions.
"""
from __future__ import annotations


class SmsError(Exception):
    """Base class for all SMS send failures."""

    def __init__(self, message: str, *, error_code: str | None = None) -> None:
        super().__init__(message)
        self.error_code = error_code


class TransientSmsError(SmsError):
    """Retryable failure: rate limit, provider 5xx, network timeout.

    The send *may* have reached the provider. Retrying with the same
    idempotency key is safe because the provider will deduplicate.
    """


class PermanentSmsError(SmsError):
    """Non-retryable failure: invalid phone number, auth error, bad payload.

    Retrying will not help and may trigger additional provider errors.
    """
