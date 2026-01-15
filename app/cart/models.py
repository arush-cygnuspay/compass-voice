# app/cart/models.py

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class CartItemView:
    """
    Read-only representation of a cart item for presentation layers.
    """

    cart_item_id: str
    item_id: str
    quantity: int
    variant_id: Optional[str]
    sides: Dict[str, List[str]]
    modifiers: Dict[str, List[str]]
