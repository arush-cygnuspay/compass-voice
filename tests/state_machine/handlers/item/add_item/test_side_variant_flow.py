import pytest


def select_side_variant(side, size):
    for v in side["pricing"]["variants"]:
        if v["variant_id"] == size:
            return v["price_cents"]
    return None


def test_side_variant_selection():
    side = {
        "name": "Coke",
        "pricing": {
            "variants": [
                {"variant_id": "small", "price_cents": 228},
                {"variant_id": "medium", "price_cents": 248},
                {"variant_id": "large", "price_cents": 277}
            ]
        }
    }

    price = select_side_variant(side, "large")

    assert price == 277