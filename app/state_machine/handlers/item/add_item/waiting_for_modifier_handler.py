from __future__ import annotations

from difflib import SequenceMatcher

from app.menu.repository import MenuRepository
from app.nlu.intent_resolution.intent import Intent
from app.nlu.query_normalization.text_preprocessor import normalize_text
from app.session.session import Session
from app.state_machine.conversation_context import (
    ConversationContext,
    InterruptProposal,
    PendingModifierGroup,
)
from app.state_machine.conversation_state import ConversationState
from app.state_machine.handler_result import HandlerResult
from app.state_machine.handlers.base_handler import BaseHandler
from app.state_machine.handlers.item.add_item.add_item_flow import (
    build_add_item_command,
    determine_next_add_item_step,
)
from app.utils.candidate_texts import build_candidate_texts_normalized
from app.utils.token_matcher import is_controlled_partial_match, is_strong_token_match

SOFT_SWITCH_INTENTS: set[Intent] = {
    Intent.ADD_ITEM,
    Intent.REMOVE_ITEM,
    Intent.MODIFY_ITEM,
    Intent.SHOW_MENU,
    Intent.ASK_MENU_INFO,
    Intent.ASK_PRICE,
    Intent.SHOW_CART,
    Intent.SHOW_TOTAL,
    Intent.START_ORDER,
    Intent.END_ADDING,
    Intent.CHECKOUT,
    Intent.CONFIRM_ORDER,
    Intent.FINISH_ORDER,
    Intent.REVIEW_ORDER,
    Intent.PAYMENT_REQUEST,
    Intent.CANCEL_ORDER,
}


def _extract_modifier_slot_values_normalized(context: ConversationContext) -> list[str]:
    slots = context.last_slots or ()
    values: list[str] = []
    seen: set[str] = set()

    for slot in slots:
        name = str(slot.name).upper()
        if name not in {"MODIFIER", "ITEM", "MENU_ITEM"}:
            continue

        value = slot.value
        if not isinstance(value, str):
            continue

        normalized = normalize_text(value)
        if not normalized or normalized in seen:
            continue

        seen.add(normalized)
        values.append(normalized)

    return values


def _remove_leading_filler(text: str) -> str:
    filler_words = {
        "the",
        "a",
        "an",
        "with",
        "add",
        "please",
        "thanks",
        "thank",
        "you",
        "and",
        "extra",
        "only",
        "just",
        "um",
        "uh",
        "okay",
        "ok",
        "ill",
        "i",
        "want",
        "take",
        "have",
        "get",
        "like",
        "would",
        "id",
        "said",
        "mean",
        "my",
        "will",
    }

    tokens = [token for token in text.split() if token not in filler_words]
    return " ".join(tokens).strip()


def _looks_like_skip_modifier_answer(normalized_user_text: str, group: PendingModifierGroup) -> bool:
    if group.is_required:
        return False

    text = (normalized_user_text or "").strip()
    if not text:
        return False

    direct_skip_phrases = {
        "no",
        "none",
        "nothing",
        "skip",
        "skip it",
        "no thanks",
        "without it",
        "dont add one",
        "do not add one",
    }
    if text in direct_skip_phrases:
        return True

    if text.startswith("without "):
        return True

    if text.startswith("no "):
        remainder = text[3:].strip()
        if not remainder:
            return True

        for choice in group.choices:
            names_to_check = [choice.normalized_name]
            normalized_aliases = getattr(choice, "normalized_aliases", ()) or ()
            names_to_check.extend(normalized_aliases)
            voice_labels = getattr(choice, "voice_labels", ()) or ()
            names_to_check.extend(voice_labels)

            for candidate_name in names_to_check:
                if remainder == candidate_name:
                    return True
                if is_strong_token_match(remainder, candidate_name):
                    return True
                if is_controlled_partial_match(remainder, candidate_name):
                    return True

    return False


def _looks_like_pure_modifier_answer(
    normalized_user_text: str,
    normalized_choice_names: tuple[str, ...],
) -> bool:
    if not normalized_user_text:
        return False

    blocked_phrases = {
        "how much",
        "price",
        "cost",
        "show menu",
        "show me",
        "menu",
        "checkout",
        "check out",
        "cart",
        "total",
        "remove item",
        "change item",
        "modify item",
        "start order",
        "finish order",
        "pay now",
        "payment",
        "add another",
        "add item",
        "remove",
        "change",
        "modify",
        "instead",
    }
    if any(phrase in normalized_user_text for phrase in blocked_phrases):
        return False

    compact = _remove_leading_filler(normalized_user_text)
    if not compact:
        return False

    for choice_name in normalized_choice_names:
        if compact == choice_name:
            return True
        if len(compact) >= 3 and (compact in choice_name or choice_name in compact):
            return True

    return False


class WaitingForModifierHandler(BaseHandler):
    """
    Resolve modifier selections strictly from the active pending item snapshot.

    Important:
    - no broad menu resolution during waiting state
    - only choices from the current active modifier group can match
    - interruption is considered before broader free-text matching
    - supports optional-group skip phrases like 'no sauce' / 'without onions'
    - uses scoped repository resolution first, then local fallback rescue
    """

    def __init__(self, menu_repo: MenuRepository | None = None) -> None:
        self.menu_repo = menu_repo

    def handle(
        self,
        intent: Intent,
        context: ConversationContext,
        user_text: str,
        session: Session | None = None,
    ) -> HandlerResult:
        pending = context.pending_add_item
        if pending is None:
            return HandlerResult(
                next_state=ConversationState.ERROR_RECOVERY,
                response_key="item_context_missing",
            )

        normalized_user_text = user_text or ""
        groups = pending.modifier_groups
        idx = context.current_modifier_group_index

        if idx >= len(groups):
            step = determine_next_add_item_step(context)
            return self._step_to_result(context, step)

        group = groups[idx]

        if intent == Intent.DENY or _looks_like_skip_modifier_answer(normalized_user_text, group):
            if group.is_required:
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_MODIFIER,
                    response_key="required_modifier_cannot_skip",
                    response_payload=self._choice_payload(group),
                )

            context.skipped_modifier_groups.add(group.group_id)
            context.current_modifier_group_index += 1

            step = determine_next_add_item_step(context)
            return self._step_to_result(context, step)

        if intent == Intent.ASK_OPTIONS:
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_MODIFIER,
                response_key="list_modifier_options",
                response_payload=self._choice_payload(group),
            )

        if intent in SOFT_SWITCH_INTENTS:
            context.awaiting_flow_confirmation = True
            context.return_state = ConversationState.WAITING_FOR_MODIFIER
            context.interrupt_proposal = InterruptProposal(
                text=normalized_user_text,
                predicted_main_intent=None,
                predicted_sub_intent=intent.value,
            )

            return HandlerResult(
                next_state=ConversationState.CANCELLATION_CONFIRMATION,
                response_key="confirm_cancel_current_item_for_new_request",
                response_payload={"item_name": pending.item_name},
            )

        matched_ids: list[str] = []

        normalized_slot_values = _extract_modifier_slot_values_normalized(context)
        if normalized_slot_values:
            matched_ids = self._match_modifier_choices_from_values(
                group=group,
                normalized_values=normalized_slot_values,
            )

        if not matched_ids and normalized_user_text:
            matched_ids = self._match_modifier_choices_from_values(
                group=group,
                normalized_values=[normalized_user_text],
            )

        if not matched_ids and _looks_like_pure_modifier_answer(
            normalized_user_text,
            group.normalized_choice_names,
        ):
            matched_ids = self._match_modifier_choices_from_values(
                group=group,
                normalized_values=[_remove_leading_filler(normalized_user_text)],
            )

        if matched_ids:
            unique_ids = list(dict.fromkeys(matched_ids))

            if len(unique_ids) > group.max_selector:
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_MODIFIER,
                    response_key="too_many_modifier_choices",
                    response_payload=self._choice_payload(group),
                )

            return self._apply_modifier_selection(
                context=context,
                group=group,
                matched_ids=unique_ids,
            )

        return HandlerResult(
            next_state=ConversationState.WAITING_FOR_MODIFIER,
            response_key="repeat_modifier_options",
            response_payload={
                **self._choice_payload(group),
                "repeat_reason": "invalid",
            },
        )

    def _apply_modifier_selection(
        self,
        *,
        context: ConversationContext,
        group: PendingModifierGroup,
        matched_ids: list[str],
    ) -> HandlerResult:
        existing_ids = list(context.selected_modifier_groups.get(group.group_id, []))
        proposed_ids = list(existing_ids)

        for modifier_id in matched_ids:
            if modifier_id not in proposed_ids:
                proposed_ids.append(modifier_id)

        max_selector = int(group.max_selector or 1)
        min_selector = int(group.min_selector or 1)

        if max_selector > 0 and len(proposed_ids) > max_selector:
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_MODIFIER,
                response_key="too_many_modifier_choices",
                response_payload=self._choice_payload(group),
            )

        context.selected_modifier_groups[group.group_id] = proposed_ids
        context.skipped_modifier_groups.discard(group.group_id)

        if len(proposed_ids) < min_selector:
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_MODIFIER,
                response_key="repeat_modifier_options",
                response_payload={
                    **self._choice_payload(group),
                    "repeat_reason": "options",
                },
            )

        context.current_modifier_group_index += 1
        step = determine_next_add_item_step(context)
        return self._step_to_result(context, step)

    def _choice_payload(self, group: PendingModifierGroup) -> dict:
        return {
            "group_name": group.name,
            "top_choices": list(group.top_choice_names),
        }

    def _candidate_labels_for_choice(self, choice) -> list[str]:
        labels = [choice.normalized_name]
        normalized_aliases = getattr(choice, "normalized_aliases", ()) or ()
        labels.extend([alias for alias in normalized_aliases if alias])
        voice_labels = getattr(choice, "voice_labels", ()) or ()
        labels.extend([label for label in voice_labels if label])
        return list(dict.fromkeys(labels))

    def _best_fuzzy_match(self, candidate: str, group: PendingModifierGroup) -> list[str]:
        best_modifier_id: str | None = None
        best_score = 0.0
        second_score = 0.0

        for choice in group.choices:
            labels = self._candidate_labels_for_choice(choice)
            local_best = 0.0

            for label in labels:
                if not label:
                    continue
                score = SequenceMatcher(None, candidate, label).ratio()
                if score > local_best:
                    local_best = score

            if local_best > best_score:
                second_score = best_score
                best_score = local_best
                best_modifier_id = choice.modifier_id
            elif local_best > second_score:
                second_score = local_best

        if best_modifier_id is None:
            return []

        if best_score >= 0.84 and (best_score - second_score) >= 0.08:
            return [best_modifier_id]

        return []

    def _match_single_candidate(self, candidate: str, group: PendingModifierGroup) -> list[str]:
        if self.menu_repo is not None:
            label_map = {
                choice.modifier_id: tuple(self._candidate_labels_for_choice(choice))
                for choice in group.choices
            }
            repo_matches = self.menu_repo.resolve_modifier_choice_within_group_normalized(
                normalized_text=candidate,
                group_id=group.group_id,
                candidate_names_by_id=label_map,
            )
            if repo_matches:
                return list(dict.fromkeys(repo_matches))

        exact_choices = group.choices_by_normalized_name.get(candidate, ())
        if exact_choices:
            return [choice.modifier_id for choice in exact_choices]

        token_matches: list[str] = []
        for choice in group.choices:
            for label in self._candidate_labels_for_choice(choice):
                if is_strong_token_match(candidate, label):
                    token_matches.append(choice.modifier_id)
                    break
        if token_matches:
            return list(dict.fromkeys(token_matches))

        partial_matches: list[str] = []
        for choice in group.choices:
            for label in self._candidate_labels_for_choice(choice):
                if is_controlled_partial_match(candidate, label):
                    partial_matches.append(choice.modifier_id)
                    break
        if partial_matches:
            return list(dict.fromkeys(partial_matches))

        return self._best_fuzzy_match(candidate, group)

    def _match_modifier_choices_from_values(
        self,
        *,
        group: PendingModifierGroup,
        normalized_values: list[str],
    ) -> list[str]:
        def _dedupe_keep_order(values: list[str]) -> list[str]:
            seen: set[str] = set()
            result: list[str] = []
            for value in values:
                value = (value or "").strip()
                if not value or value in seen:
                    continue
                seen.add(value)
                result.append(value)
            return result

        full_candidates = _dedupe_keep_order(
            [_remove_leading_filler(value) for value in normalized_values]
        )
        split_candidates = build_candidate_texts_normalized(
            normalized_user_text="",
            normalized_slot_values=normalized_values,
            allow_split=True,
        )
        split_candidates = _dedupe_keep_order(
            [_remove_leading_filler(value) for value in split_candidates]
        )
        split_candidates = [candidate for candidate in split_candidates if candidate not in full_candidates]

        full_matches: list[str] = []
        seen_full: set[str] = set()

        for candidate in full_candidates:
            matched_ids = self._match_single_candidate(candidate, group)

            if len(matched_ids) == 1:
                return matched_ids

            for modifier_id in matched_ids:
                if modifier_id not in seen_full:
                    seen_full.add(modifier_id)
                    full_matches.append(modifier_id)

        if full_matches:
            return full_matches

        split_matches: list[str] = []
        seen_split: set[str] = set()

        for candidate in split_candidates:
            matched_ids = self._match_single_candidate(candidate, group)

            for modifier_id in matched_ids:
                if modifier_id not in seen_split:
                    seen_split.add(modifier_id)
                    split_matches.append(modifier_id)

            if len(split_matches) > 1:
                break

        return split_matches

    def _step_to_result(self, context: ConversationContext, step) -> HandlerResult:
        pending = context.pending_add_item
        if pending is None:
            return HandlerResult(
                next_state=ConversationState.ERROR_RECOVERY,
                response_key="item_context_missing",
            )

        if step.next_state == ConversationState.FINALIZING_ADD_ITEM:
            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="item_added_successfully",
                response_payload={
                    "item_name": pending.item_name,
                    "quantity": context.quantity or 1,
                },
                command=build_add_item_command(context),
                reset_context=True,
            )

        return HandlerResult(
            next_state=step.next_state,
            response_key=step.response_key,
            response_payload=step.response_payload,
        )