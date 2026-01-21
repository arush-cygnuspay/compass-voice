# app/responses/utils.py

from typing import Iterable


def numbered_list(
    items: Iterable[str],
    *,
    max_items: int | None = None,
) -> list[str]:
    """
    Formats items as a numbered list.

    Example:
    1. Item A
    2. Item B

    Truncation is presentation-only.
    """
    lines = []
    for idx, item in enumerate(items, start=1):
        if max_items and idx > max_items:
            break
        lines.append(f"{idx}. {item}")
    return lines
