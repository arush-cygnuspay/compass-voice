# app/utils/top_k_choices.py
import random
from typing import Iterable, List, TypeVar

T = TypeVar("T")


def get_top_k_choices(choices: Iterable[T], k: int = 4) -> List[T]:
    """
    Return up to k choices.
    - If choices > k → randomly sample k
    - If choices <= k → return all (preserving order)
    - If k <= 0 or choices empty → []

    Used for recovery prompts only; keep selection small and conversational.
    """
    if k <= 0:
        return []

    choices_list = list(choices)
    if not choices_list:
        return []

    k = min(k, len(choices_list))

    if len(choices_list) > k:
        return random.sample(choices_list, k)

    return choices_list
