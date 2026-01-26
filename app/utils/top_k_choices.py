# app/utils/top_k_choices.py
import random
from typing import Iterable, List, TypeVar

T = TypeVar("T")


def get_top_k_choices(choices: Iterable[T], k: int = 3) -> List[T]:
    """Return up to k choices, randomized when possible, falling back to list order.

    Used for recovery prompts only; keep selection small and conversational.
    """
    choices_list = list(choices)
    if k <= 0 or not choices_list:
        return []

    # If we have more than k, sample; otherwise return all.
    if len(choices_list) > k:
        return random.sample(choices_list, k)

    return choices_list
