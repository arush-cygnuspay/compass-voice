# app/api/checkout_routes.py
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app.config.restaurant import DEFAULT_RESTAURANT_ID
from app.services.checkout_service import (
    CheckoutExpiredError,
    CheckoutNotFoundError,
    CheckoutService,
    ReverseGeocodeError,
)

router = APIRouter(prefix="/api/checkout", tags=["checkout"])
page_router = APIRouter(tags=["checkout-page"])

checkout_service = CheckoutService()
CHECKOUT_HTML_PATH = Path("app/static/checkout/index.html")


class CreateCheckoutSessionPayload(BaseModel):
    restaurant_id: str = DEFAULT_RESTAURANT_ID
    call_sid: str | None = None
    order_number: str | None = None
    customer_phone_number: str | None = None
    address_required: bool = False
    area: str | None = None
    postal_code: str | None = None
    order_summary: dict = Field(default_factory=dict)


class AddressPayload(BaseModel):
    house_number: str = Field(min_length=1, max_length=50)
    street: str = Field(min_length=1, max_length=120)
    secondary_address: str | None = Field(default=None, max_length=120)


class LocationPayload(BaseModel):
    latitude: float | None = None
    longitude: float | None = None
    permission_granted: bool | None = None


class PaymentPayload(BaseModel):
    amount: str


def _load_session(token: str):
    try:
        return checkout_service.get_session(token)
    except CheckoutNotFoundError:
        raise HTTPException(status_code=404, detail="Checkout session not found.")
    except CheckoutExpiredError:
        raise HTTPException(status_code=410, detail="Checkout session expired.")


@router.post("/session")
def create_checkout_session(payload: CreateCheckoutSessionPayload):
    session = checkout_service.create_session(
        restaurant_id=payload.restaurant_id,
        call_sid=payload.call_sid,
        order_number=payload.order_number,
        customer_phone_number=payload.customer_phone_number,
        address_required=payload.address_required,
        area=payload.area,
        postal_code=payload.postal_code,
        order_summary=payload.order_summary,
    )
    return {
        "token": session.token,
        "checkout_url": checkout_service.build_checkout_url(session.token),
        "session": checkout_service.serialize_session(session),
    }


@router.get("/{token}")
def get_checkout_session(token: str):
    session = _load_session(token)
    return checkout_service.serialize_session(session)


@router.post("/{token}/address")
def save_checkout_address(token: str, payload: AddressPayload):
    session = checkout_service.update_address(
        token=token,
        house_number=payload.house_number,
        street=payload.street,
        secondary_address=payload.secondary_address,
    )
    return checkout_service.serialize_session(session)


@router.post("/{token}/location")
def save_checkout_location(token: str, payload: LocationPayload):
    try:
        session = checkout_service.update_location(
            token=token,
            latitude=payload.latitude,
            longitude=payload.longitude,
            permission_granted=payload.permission_granted,
        )
        return checkout_service.serialize_session(session)
    except ReverseGeocodeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@router.post("/{token}/payment")
def start_checkout_payment(token: str, payload: PaymentPayload):
    _ = _load_session(token)
    try:
        return checkout_service.start_payment(
            token=token,
            amount=payload.amount,
        )
    except ValueError as exc:
        print("[CHECKOUT PAYMENT ERROR]", {"token": token, "error": str(exc)})
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        print(
            "[CHECKOUT PAYMENT UNHANDLED ERROR]",
            {"token": token, "error": f"{type(exc).__name__}: {exc}"},
        )
        raise HTTPException(
            status_code=500,
            detail="Internal payment initialization error.",
        )


@router.post("/{token}/complete")
def complete_checkout(token: str):
    _load_session(token)
    result = checkout_service.verify_payment_with_provider(token)
    if not result.get("payment_completed"):
        raise HTTPException(
            status_code=409,
            detail="Payment is not verified yet.",
        )
    return result["session"]


@router.post("/{token}/verify-payment")
def verify_checkout_payment(token: str):
    """Ask Datacap directly whether this payment has been captured.

    The frontend calls this endpoint after the customer has visited the Datacap
    payment page.  We never trust the user's own "I paid" claim — we only mark
    the order complete once Datacap confirms the money is captured.

    Returns the same shape as ``verify_payment_with_provider``:
        {
            "ok":                 bool,
            "paid":               bool,
            "payment_completed":  bool,
            "status":             str,
            "reference":          str | None,
            "session":            dict,   # full session state
            "error":              str | None,
        }
    """
    # Validates the token exists + not expired before we hit Datacap.
    _load_session(token)
    result = checkout_service.verify_payment_with_provider(token)
    return result


@page_router.get("/checkout/{token}", response_class=HTMLResponse)
def checkout_page(token: str):
    if not CHECKOUT_HTML_PATH.exists():
        raise HTTPException(status_code=500, detail="Checkout page not found.")
    html = CHECKOUT_HTML_PATH.read_text(encoding="utf-8")
    return html.replace("__CHECKOUT_TOKEN__", token)
