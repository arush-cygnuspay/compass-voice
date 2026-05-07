# app/state_machine/handlers/item/add_item/group_collection_utils.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from app.nlu.query_normalization.text_preprocessor import normalize_text
from app.utils.candidate_texts import build_candidate_texts_normalized


@dataclass(frozen=True, slots=True)
class MultiValueResolution:
    matched_ids: list[str]
    matched_names: list[str]
    unmatched_values: list[str]


def dedupe_keep_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        value = (value or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def effective_group_selector_bounds(group) -> tuple[int, int]:
    """
    Clamp selector bounds to what the caller can actually choose from the group.
    Optional groups may legitimately have min_selector=0.

    When ``allow_duplicate_selections`` is True the max is NOT clamped to
    option_count because the same choice may be selected multiple times.
    """
    raw_min = int(getattr(group, "min_selector", 0) or 0)
    raw_max = int(getattr(group, "max_selector", 0) or 0)
    option_count = len(getattr(group, "choice_names", ()) or getattr(group, "choices", ()) or ())
    allow_dupes = getattr(group, "allow_duplicate_selections", True)

    min_selector = max(raw_min, 1 if bool(getattr(group, "is_required", False)) else 0)
    if option_count > 0:
        min_selector = min(min_selector, option_count)

    if raw_max > 0:
        max_selector = raw_max
        # Only clamp by option_count when duplicates are disallowed; when
        # duplicates are allowed a single option can fill multiple slots.
        if option_count > 0 and not allow_dupes:
            max_selector = min(max_selector, option_count)
    else:
        max_selector = option_count

    if max_selector and min_selector > max_selector:
        min_selector = max_selector

    return min_selector, max_selector


def normalize_candidate_values(
    *,
    normalized_user_text: str,
    normalized_slot_values: list[str],
    remove_leading_filler_fn,
) -> list[str]:
    full_candidates = dedupe_keep_order(
        [remove_leading_filler_fn(value) for value in normalized_slot_values if value]
    )

    if normalized_user_text:
        cleaned_text = remove_leading_filler_fn(normalized_user_text)
        if cleaned_text and cleaned_text not in full_candidates:
            full_candidates.append(cleaned_text)

    split_candidates = build_candidate_texts_normalized(
        normalized_user_text=normalized_user_text,
        normalized_slot_values=normalized_slot_values,
        allow_split=True,
    )
    split_candidates = dedupe_keep_order(
        [remove_leading_filler_fn(value) for value in split_candidates if value]
    )

    return dedupe_keep_order(full_candidates + split_candidates)


def build_group_progress_payload(
    *,
    group_id: str,
    group_name: str,
    min_selector: int,
    max_selector: int,
    selected_ids: list[str],
    selected_names: list[str],
    option_names: list[str],
    item_name: str | None = None,
) -> dict:
    selected_count = len(selected_ids)
    remaining_to_min = max(min_selector - selected_count, 0)
    remaining_to_max = max(max_selector - selected_count, 0) if max_selector > 0 else 999999

    payload = {
        "group_id": group_id,
        "group_name": group_name,
        "item_name": item_name,
        "selected_count": selected_count,
        "selected_names": selected_names,
        "min_selector": min_selector,
        "max_selector": max_selector,
        "remaining_to_min": remaining_to_min,
        "remaining_to_max": remaining_to_max,
        "top_choices": option_names[:6],
    }
    return payload
