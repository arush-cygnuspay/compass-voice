import pytest


def get_variant_price(item, size):
    for v in item["pricing"]["variants"]:
        if v["variant_id"] == size:
            return v["price_cents"]
    return None


def test_variant_price_lookup():
    item = {
        "pricing": {
            "variants": [
                {"variant_id": "small", "price_cents": 228},
                {"variant_id": "medium", "price_cents": 248},
                {"variant_id": "large", "price_cents": 277}
            ]
        }
    }

    price = get_variant_price(item, "medium")

    assert price == 248


def test_invalid_variant():
    item = {
        "pricing": {
            "variants": [
                {"variant_id": "small", "price_cents": 228}
            ]
        }
    }

    price = get_variant_price(item, "large")

    assert price is None