from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher

from app.menu.repository import MenuRepository
from app.nlu.intent_resolution.intent import Intent
from app.nlu.query_normalization.text_preprocessor import normalize_text
from app.session.session import Session
from app.state_machine.models.conversation_context import (
    ConversationContext,
    InterruptProposal,
    PendingModifierGroup,
)
from app.state_machine.models.conversation_state import ConversationState
from app.state_machine.handler_result import HandlerResult
from app.state_machine.handlers.base_handler import BaseHandler
from app.state_machine.handlers.item.add_item.add_item_flow import (
    build_add_item_command,
    determine_next_add_item_step,
)
from app.utils.candidate_texts import build_candidate_texts_normalized
from app.utils.token_matcher import is_controlled_partial_match, is_strong_token_match, tokenize

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

AUTO_ACCEPT_THRESHOLD = 0.80
CONFIRM_THRESHOLD = 0.60
MIN_CONFIRM_GAP = 0.08


@dataclass(frozen=True, slots=True)
class _ScoredModifierChoice:
    modifier_id: str
    choice_name: str
    confidence: float


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
    - if best fuzzy match is medium confidence, ask yes/no confirmation
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

        modifier_confirmation = self._get_pending_modifier_confirmation(context, group)
        if modifier_confirmation is not None:
            if intent == Intent.CONFIRM:
                return self._apply_modifier_selection(
                    context=context,
                    group=group,
                    matched_ids=[modifier_confirmation["modifier_id"]],
                    clear_pending_confirmation=True,
                )

            if intent == Intent.DENY:
                self._clear_pending_modifier_confirmation(context)
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_MODIFIER,
                    response_key="repeat_modifier_options",
                    response_payload={
                        **self._choice_payload(group),
                        "repeat_reason": "invalid",
                    },
                )

            if intent == Intent.ASK_OPTIONS:
                self._clear_pending_modifier_confirmation(context)
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_MODIFIER,
                    response_key="list_modifier_options",
                    response_payload=self._choice_payload(group),
                )

            self._clear_pending_modifier_confirmation(context)

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

        scored_match: _ScoredModifierChoice | None = None

        normalized_slot_values = _extract_modifier_slot_values_normalized(context)
        if normalized_slot_values:
            scored_match = self._resolve_best_modifier_choice_from_values(
                group=group,
                normalized_values=normalized_slot_values,
            )

        if scored_match is None and normalized_user_text:
            scored_match = self._resolve_best_modifier_choice_from_values(
                group=group,
                normalized_values=[normalized_user_text],
            )

        if scored_match is None and _looks_like_pure_modifier_answer(
            normalized_user_text,
            group.normalized_choice_names,
        ):
            scored_match = self._resolve_best_modifier_choice_from_values(
                group=group,
                normalized_values=[_remove_leading_filler(normalized_user_text)],
            )

        if scored_match is not None:
            if scored_match.confidence >= AUTO_ACCEPT_THRESHOLD:
                return self._apply_modifier_selection(
                    context=context,
                    group=group,
                    matched_ids=[scored_match.modifier_id],
                    clear_pending_confirmation=True,
                )

            if scored_match.confidence >= CONFIRM_THRESHOLD:
                self._set_pending_modifier_confirmation(
                    context=context,
                    group=group,
                    modifier_id=scored_match.modifier_id,
                    choice_name=scored_match.choice_name,
                    confidence=scored_match.confidence,
                )
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_MODIFIER,
                    response_key="confirm_modifier_choice_guess",
                    response_payload={
                        "group_name": group.name,
                        "choice_name": scored_match.choice_name,
                    },
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
        clear_pending_confirmation: bool = False,
    ) -> HandlerResult:
        if clear_pending_confirmation:
            self._clear_pending_modifier_confirmation(context)

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

    def _similarity_ratio(self, a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()

    def _choice_confidence(self, candidate: str, choice) -> float:
        labels = self._candidate_labels_for_choice(choice)
        if not labels:
            return 0.0

        best = 0.0
        candidate_tokens = set(tokenize(candidate))

        for label in labels:
            if not label:
                continue

            if candidate == label:
                return 1.0

            label_tokens = set(tokenize(label))
            overlap = len(candidate_tokens & label_tokens)
            if candidate_tokens and label_tokens:
                coverage = overlap / len(label_tokens)
                candidate_coverage = overlap / len(candidate_tokens)
                token_score = max(coverage, candidate_coverage)
                best = max(best, token_score)

            if is_strong_token_match(candidate, label):
                best = max(best, 0.92)

            if is_controlled_partial_match(candidate, label):
                best = max(best, 0.82)

            best = max(best, self._similarity_ratio(candidate, label))

        return best

    def _resolve_best_modifier_choice_from_values(
        self,
        *,
        group: PendingModifierGroup,
        normalized_values: list[str],
    ) -> _ScoredModifierChoice | None:
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

        all_candidates = full_candidates + split_candidates
        if not all_candidates:
            return None

        best_choice = None
        best_confidence = 0.0
        second_confidence = 0.0

        for choice in group.choices:
            choice_best = 0.0
            for candidate in all_candidates:
                confidence = self._choice_confidence(candidate, choice)
                if confidence > choice_best:
                    choice_best = confidence

            if choice_best > best_confidence:
                second_confidence = best_confidence
                best_confidence = choice_best
                best_choice = choice
            elif choice_best > second_confidence:
                second_confidence = choice_best

        if best_choice is None:
            return None

        if best_confidence < CONFIRM_THRESHOLD:
            return None

        if (
            best_confidence < AUTO_ACCEPT_THRESHOLD
            and (best_confidence - second_confidence) < MIN_CONFIRM_GAP
        ):
            return None

        return _ScoredModifierChoice(
            modifier_id=best_choice.modifier_id,
            choice_name=best_choice.name,
            confidence=best_confidence,
        )

    def _get_pending_modifier_confirmation(
        self,
        context: ConversationContext,
        group: PendingModifierGroup,
    ) -> dict | None:
        confirmation = getattr(context, "awaiting_confirmation_for", None)
        if not isinstance(confirmation, dict):
            return None

        if confirmation.get("type") != "modifier_choice_guess":
            return None

        if confirmation.get("group_id") != group.group_id:
            return None

        return confirmation

    def _set_pending_modifier_confirmation(
        self,
        *,
        context: ConversationContext,
        group: PendingModifierGroup,
        modifier_id: str,
        choice_name: str,
        confidence: float,
    ) -> None:
        context.awaiting_confirmation_for = {
            "type": "modifier_choice_guess",
            "group_id": group.group_id,
            "modifier_id": modifier_id,
            "choice_name": choice_name,
            "confidence": confidence,
        }

    def _clear_pending_modifier_confirmation(self, context: ConversationContext) -> None:
        confirmation = getattr(context, "awaiting_confirmation_for", None)
        if isinstance(confirmation, dict) and confirmation.get("type") == "modifier_choice_guess":
            context.awaiting_confirmation_for = None

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