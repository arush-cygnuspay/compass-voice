"""Diagnostic helpers extracted from TurnEngine.

Pure plumbing for trace setup/finalize, context snapshotting, response-key
classification, response-text rendering for logs, and NLU CSV emission.
No FSM transitions, no business decisions — moved verbatim from
turn_engine.py.

Module-level constants (``REPROMPT_RESPONSE_KEYS``,
``FALLBACK_RESPONSE_KEYS``, ``TURN_TIMING_ENABLED``) are owned here and
re-exported from ``turn_engine`` for backwards compatibility.
"""
from __future__ import annotations

import os
import time
from typing import Any

from app.menu.repository import MenuRepository
from app.nlu.intent_resolution.intent import Intent
from app.session.session import Session
from app.state_machine.handler_result import HandlerResult
from app.state_machine.handlers.item.add_item.group_collection_utils import (
    effective_group_selector_bounds,
)
from app.state_machine.models.conversation_state import ConversationState


TURN_TIMING_ENABLED = os.getenv("COMPASS_TURN_TIMING_ENABLED", "0") == "1"


FALLBACK_RESPONSE_KEYS: set[str] = {
    "repeat_side_options",
    "repeat_modifier_options",
    "repeat_size_options",
    "repeat_side_size_options",
    "invalid_size_option",
    "invalid_side_size_option",
    "invalid_quantity_option",
    "item_not_found",
    "intent_not_allowed",
    "item_context_missing",
    "confirmation_state_error",
    "payment_not_confirmed_yet",
    "payment_verification_error",
    "checkout_link_send_failed",
    "payment_link_send_failed",
    "payment_link_unavailable_now",
}
REPROMPT_RESPONSE_KEYS: set[str] = {
    "repeat_side_options",
    "repeat_modifier_options",
    "repeat_size_options",
    "repeat_side_size_options",
    "invalid_size_option",
    "invalid_side_size_option",
    "invalid_quantity_option",
    "required_side_cannot_skip",
    "required_modifier_cannot_skip",
    "required_size_cannot_skip",
    "required_side_size_cannot_skip",
}


class TurnDiagnostics:
    """Pure helper class. All methods previously lived on TurnEngine."""

    def __init__(self, menu_repo: MenuRepository, nlu_logger: Any, responder: Any) -> None:
        self.menu_repo = menu_repo
        self.nlu_logger = nlu_logger
        self.responder = responder

    def _log_if_enabled(
        self,
        *,
        session: Session,
        state_before: ConversationState,
        nlu: Any,
        result: HandlerResult | None,
        response_key: str,
        response_payload: dict[str, Any] | None,
        next_state: ConversationState | None = None,
        preprocess_ms: float | None = None,
        nlu_ms: float | None = None,
        flow_ms: float | None = None,
        route_ms: float | None = None,
        handler_ms: float | None = None,
        total_ms: float | None = None,
    ) -> dict[str, Any]:
        diagnostics = self._update_session_diagnostics(
            session=session,
            state_before=state_before,
            nlu=nlu,
            response_key=response_key,
            response_payload=response_payload,
        )
        if not self.nlu_logger.enabled:
            return diagnostics

        context_snapshot = self._snapshot_context_for_logging(session)
        response_text = self._build_response_text_for_logging(
            session=session,
            response_key=response_key,
            response_payload=response_payload,
        )
        self._log_nlu_and_turn(
            session=session,
            state_before=state_before,
            context_snapshot=context_snapshot,
            nlu=nlu,
            result=result,
            response_key=response_key,
            response_payload=response_payload,
            response_text=response_text,
            next_state=(next_state or session.conversation_state).value,
            diagnostics=diagnostics,
            preprocess_ms=preprocess_ms,
            nlu_ms=nlu_ms,
            flow_ms=flow_ms,
            route_ms=route_ms,
            handler_ms=handler_ms,
            total_ms=total_ms,
        )
        return diagnostics

    def _finalize_trace_and_timing(
        self,
        *,
        trace: Any | None,
        session: Session,
        response_key: str,
        total_start_monotonic: float,
        total_ms: float,
        preprocess_ms: float,
        nlu_ms: float,
        flow_ms: float,
        route_ms: float,
        handler_ms: float,
        command: dict[str, Any] | None = None,
    ) -> None:
        self._trace_finalize(
            trace=trace,
            session=session,
            response_key=response_key,
            total_start_monotonic=total_start_monotonic,
            total_ms=total_ms,
            preprocess_ms=preprocess_ms,
            nlu_ms=nlu_ms,
            flow_ms=flow_ms,
            route_ms=route_ms,
            handler_ms=handler_ms,
            command=command,
        )

        self._maybe_print_timing(
            total_ms=total_ms,
            preprocess_ms=preprocess_ms,
            nlu_ms=nlu_ms,
            flow_ms=flow_ms,
            route_ms=route_ms,
            handler_ms=handler_ms,
        )

    def _maybe_print_timing(
        self,
        *,
        total_ms: float,
        preprocess_ms: float,
        nlu_ms: float,
        flow_ms: float,
        route_ms: float,
        handler_ms: float,
    ) -> None:
        if not TURN_TIMING_ENABLED:
            return

        print(
            "[TURN_TIMING]",
            {
                "total_ms": round(total_ms, 3),
                "preprocess_ms": round(preprocess_ms, 3),
                "nlu_ms": round(nlu_ms, 3),
                "flow_ms": round(flow_ms, 3),
                "route_ms": round(route_ms, 3),
                "handler_ms": round(handler_ms, 3),
            },
        )

    def _safe_session_id(self, session: Session) -> str:
        value = getattr(session, "session_id", None)
        if value is not None:
            return str(value)

        value = getattr(session, "id", None)
        if value is not None:
            return str(value)

        return "unknown_session"

    def _safe_pending_action_from_context(self, context: Any) -> str:
        action = getattr(context, "pending_action", None)
        return action.value if action is not None else ""

    def _safe_current_prompt_field_from_context(self, context: Any) -> str:
        return getattr(context, "current_prompt_field", None) or ""

    def _safe_current_item_id_from_context(self, context: Any) -> str:
        return getattr(context, "current_item_id", None) or ""

    def _safe_current_item_name_from_context(self, context: Any) -> str:
        return getattr(context, "current_item_name", None) or ""

    def _snapshot_context_for_logging(self, session: Session) -> dict[str, str]:
        ctx = session.conversation_context
        return {
            "pending_action": self._safe_pending_action_from_context(ctx),
            "current_prompt_field": self._safe_current_prompt_field_from_context(ctx),
            "current_item_id": self._safe_current_item_id_from_context(ctx),
            "current_item_name": self._safe_current_item_name_from_context(ctx),
        }

    def _format_modifier_selection_name(self, selection: Any) -> str:
        name = str(getattr(selection, "name", "") or "").strip()
        action = str(getattr(selection, "action", "add") or "add").strip()
        instruction = getattr(selection, "instruction", None)
        if not name:
            return ""
        if action == "remove":
            return f"no {name}"
        if instruction == "extra":
            return f"extra {name}"
        if instruction == "less":
            return f"less {name}"
        if instruction == "on_side":
            return f"{name} on the side"
        return name

    def _build_normalized_values_snapshot(
        self,
        *,
        session: Session,
        result: HandlerResult | None,
    ) -> dict[str, Any]:
        command = (result.command or {}) if result is not None else {}
        payload = command.get("payload") or {}
        if command.get("type") == "ADD_ITEM_TO_CART" and payload:
            item_id = payload.get("item_id")
            item = self.menu_repo.store.get_item(item_id) if item_id else None
            side_choice_by_id: dict[str, Any] = {}
            side_group_by_id: dict[str, Any] = {}
            modifier_group_by_id: dict[str, Any] = {}
            if item is not None:
                for group in getattr(item, "side_groups", ()) or ():
                    side_group_by_id[group.group_id] = group
                    for choice in getattr(group, "choices", ()) or ():
                        side_choice_by_id[choice.item_id] = choice
                for group in getattr(item, "modifier_groups", ()) or ():
                    modifier_group_by_id[group.group_id] = group

            mapped_sides: dict[str, list[str]] = {}
            for group_id, side_ids in (payload.get("sides") or {}).items():
                group = side_group_by_id.get(group_id)
                group_name = getattr(group, "name", None) or str(group_id)
                mapped_sides[group_name] = [
                    getattr(side_choice_by_id.get(side_id), "name", side_id)
                    for side_id in side_ids
                ]

            mapped_side_variants: dict[str, str] = {}
            for side_id, variant_id in (payload.get("side_variants") or {}).items():
                choice = side_choice_by_id.get(side_id)
                side_name = getattr(choice, "name", None) or str(side_id)
                variant_label = str(variant_id)
                if choice is not None:
                    for variant in getattr(getattr(choice, "pricing", None), "variants", ()) or ():
                        if getattr(variant, "variant_id", None) == variant_id:
                            variant_label = getattr(variant, "label", variant_label)
                            break
                mapped_side_variants[side_name] = variant_label

            mapped_modifiers: dict[str, list[str]] = {}
            for group_id, selections in (payload.get("modifiers") or {}).items():
                group = modifier_group_by_id.get(group_id)
                group_name = getattr(group, "name", None) or str(group_id)
                names: list[str] = []
                for selection in selections or ():
                    if isinstance(selection, dict):
                        names.append(
                            self._format_modifier_selection_name(
                                type("ModifierPayload", (), selection)()
                            )
                        )
                    else:
                        names.append(self._format_modifier_selection_name(selection))
                mapped_modifiers[group_name] = [name for name in names if name]

            mapped_variant = None
            variant_id = payload.get("variant_id")
            if item is not None and variant_id:
                for variant in getattr(getattr(item, "pricing", None), "variants", ()) or ():
                    if getattr(variant, "variant_id", None) == variant_id:
                        mapped_variant = getattr(variant, "label", None) or variant_id
                        break
            elif variant_id:
                mapped_variant = variant_id

            return {
                "item_name": getattr(item, "name", None) or str(item_id or ""),
                "quantity": payload.get("quantity"),
                "variant": mapped_variant,
                "sides": mapped_sides,
                "side_variants": mapped_side_variants,
                "modifiers": mapped_modifiers,
            }

        ctx = session.conversation_context
        pending = ctx.pending_add_item
        if pending is None:
            return {
                "item_name": ctx.current_item_name,
                "quantity": ctx.quantity,
            }

        mapped_sides: dict[str, list[str]] = {}
        for group in pending.side_groups:
            selected_ids = ctx.selected_side_groups.get(group.group_id, ())
            if not selected_ids:
                continue
            mapped_sides[group.name] = [
                group.choices_by_item_id[item_id].name
                for item_id in selected_ids
                if item_id in group.choices_by_item_id
            ]

        mapped_modifiers: dict[str, list[str]] = {}
        for group in pending.modifier_groups:
            selections = ctx.selected_modifier_groups.get(group.group_id, ())
            if not selections:
                continue
            mapped_modifiers[group.name] = [
                self._format_modifier_selection_name(selection)
                for selection in selections
                if self._format_modifier_selection_name(selection)
            ]

        mapped_variant = None
        if ctx.selected_variant_id:
            variant = pending.item_variants_by_id.get(ctx.selected_variant_id)
            mapped_variant = getattr(variant, "name", None) or ctx.selected_variant_id

        mapped_side_variants: dict[str, str] = {}
        for side_item_id, variant_id in ctx.selected_side_variants.items():
            choice = pending.side_choice_by_item_id.get(side_item_id)
            side_name = getattr(choice, "name", None) or str(side_item_id)
            variant_name = variant_id
            if choice is not None:
                variant = choice.variants_by_id.get(variant_id)
                if variant is not None:
                    variant_name = variant.name
            mapped_side_variants[side_name] = variant_name

        return {
            "item_name": pending.item_name,
            "quantity": ctx.quantity,
            "variant": mapped_variant,
            "sides": mapped_sides,
            "side_variants": mapped_side_variants,
            "modifiers": mapped_modifiers,
        }

    def _build_missing_required_fields(self, session: Session) -> list[str]:
        ctx = session.conversation_context
        pending = ctx.pending_add_item
        if pending is None:
            return []

        missing: list[str] = []
        if pending.item_variants and not ctx.selected_variant_id:
            missing.append("size")

        for group in pending.side_groups:
            selected_ids = ctx.selected_side_groups.get(group.group_id, ())
            min_selector, _ = effective_group_selector_bounds(group)
            if bool(getattr(group, "is_required", False)) and len(selected_ids) < min_selector:
                missing.append(group.name)

        for group in pending.modifier_groups:
            selections = ctx.selected_modifier_groups.get(group.group_id, ())
            min_selector, _ = effective_group_selector_bounds(group)
            if bool(getattr(group, "is_required", False)) and len(selections) < min_selector:
                missing.append(group.name)

        if not (isinstance(ctx.quantity, int) and ctx.quantity > 0):
            missing.append("quantity")

        return missing

    def _infer_prompt_field_for_response(
        self,
        *,
        response_key: str,
        session: Session,
    ) -> str:
        key = response_key or ""
        if "modifier" in key:
            return "modifier"
        if "side_size" in key:
            return "side_size"
        if "size" in key:
            return "size"
        if "side" in key:
            return "side"
        if "quantity" in key:
            return "quantity"
        return session.conversation_context.current_prompt_field or ""

    def _update_session_diagnostics(
        self,
        *,
        session: Session,
        state_before: ConversationState,
        nlu: Any,
        response_key: str,
        response_payload: dict[str, Any] | None,
    ) -> dict[str, Any]:
        normalized_text = getattr(nlu, "normalized_text", "") or ""
        user_repeated = bool(
            normalized_text
            and normalized_text == (session.last_normalized_user_text or "")
        )
        if user_repeated:
            session.repeated_user_turn_count += 1
        session.last_normalized_user_text = normalized_text or None

        field = self._infer_prompt_field_for_response(
            response_key=response_key,
            session=session,
        )
        reprompt_escalated = bool((response_payload or {}).get("reprompt_escalation"))
        is_quantity_reprompt = (
            state_before == ConversationState.WAITING_FOR_QUANTITY
            and response_key == "ask_for_quantity"
        )
        reprompt_count = int(session.reprompt_count_by_field.get(field, 0) or 0)
        if field:
            if (
                response_key in REPROMPT_RESPONSE_KEYS
                or response_key in {"list_side_options", "list_modifier_options"}
                or is_quantity_reprompt
            ):
                reprompt_count += 1
                session.reprompt_count_by_field[field] = reprompt_count
            else:
                reprompt_count = 0
                session.reprompt_count_by_field[field] = 0

        fallback_triggered = response_key in FALLBACK_RESPONSE_KEYS or response_key in REPROMPT_RESPONSE_KEYS
        if response_key in {"list_side_options", "list_modifier_options"}:
            fallback_triggered = True
        if fallback_triggered:
            session.fallback_count += 1
        if reprompt_escalated:
            session.reprompt_escalation_count += 1

        slot_extraction_failed = (
            state_before in {
                ConversationState.WAITING_FOR_SIDE,
                ConversationState.WAITING_FOR_MODIFIER,
                ConversationState.WAITING_FOR_SIZE,
                ConversationState.WAITING_FOR_SIDE_SIZE,
            }
            and response_key
            in {
                "repeat_side_options",
                "repeat_modifier_options",
                "invalid_size_option",
                "invalid_side_size_option",
                "list_side_options",
                "list_modifier_options",
            }
            and not (getattr(nlu, "slots", ()) or ())
        )
        if slot_extraction_failed:
            session.slot_extraction_failure_count += 1

        invalid_modifier = (
            state_before == ConversationState.WAITING_FOR_MODIFIER
            and response_key in {"repeat_modifier_options", "too_many_modifier_choices", "list_modifier_options"}
        )
        if invalid_modifier:
            session.invalid_modifier_count += 1

        fallback_reason = ""
        if reprompt_escalated:
            fallback_reason = "reprompt_escalation"
        elif slot_extraction_failed:
            fallback_reason = "slot_extraction_failed"
        elif invalid_modifier:
            fallback_reason = "invalid_modifier"
        elif fallback_triggered:
            fallback_reason = response_key

        return {
            "reprompt_field": field,
            "reprompt_count": reprompt_count,
            "reprompt_escalated": reprompt_escalated,
            "fallback_triggered": fallback_triggered,
            "fallback_reason": fallback_reason,
            "slot_extraction_failed": slot_extraction_failed,
            "invalid_modifier": invalid_modifier,
            "user_repeated": user_repeated,
        }

    def _build_response_text_for_logging(
        self,
        *,
        session: Session,
        response_key: str,
        response_payload: dict[str, Any] | None,
    ) -> str:
        return self._normalize_response_text(
            self.responder.build(
                response_key=response_key,
                context=session.conversation_context,
                payload=response_payload,
            )
        )

    @staticmethod
    def _normalize_response_text(text: str | None) -> str:
        return " ".join((text or "").split()).strip()

    def _log_nlu_and_turn(
        self,
        *,
        session: Session,
        state_before: ConversationState,
        context_snapshot: dict[str, str],
        nlu: Any,
        result: HandlerResult | None,
        response_key: str,
        response_payload: dict[str, Any] | None,
        response_text: str,
        next_state: str,
        diagnostics: dict[str, Any],
        preprocess_ms: float | None = None,
        nlu_ms: float | None = None,
        flow_ms: float | None = None,
        route_ms: float | None = None,
        handler_ms: float | None = None,
        total_ms: float | None = None,
    ) -> None:
        try:
            slot_values = getattr(nlu, "slots", ()) or ()
            normalized_values = self._build_normalized_values_snapshot(
                session=session,
                result=result,
            )
            missing_required_fields = self._build_missing_required_fields(session)

            self.nlu_logger.log_turn(
                session_id=self._safe_session_id(session),
                turn_index=session.turn_count,
                state_before=state_before.value,
                state_after=session.conversation_state.value,
                pending_action=context_snapshot["pending_action"],
                current_prompt_field=context_snapshot["current_prompt_field"],
                current_item_id=context_snapshot["current_item_id"],
                current_item_name=context_snapshot["current_item_name"],
                raw_user_text=getattr(nlu, "raw_text", "") or session.conversation_context.last_user_text or "",
                user_text=session.conversation_context.last_user_text or "",
                normalized_text=getattr(nlu, "normalized_text", "") or "",
                pred_main_intent=getattr(nlu, "model_main_intent", "") or "",
                pred_sub_intent=getattr(nlu, "model_sub_intent", "") or "",
                pred_intent=getattr(getattr(nlu, "effective_intent", None), "value", "") or "",
                pred_intent_confidence=getattr(nlu, "intent_confidence", None),
                slot_model_ran=bool(getattr(nlu, "slot_model_ran", False)),
                response_key=response_key,
                response_text=response_text,
                next_state=next_state,
                command=result.command if result else None,
                slots=slot_values,
                normalized_values=normalized_values,
                missing_required_fields=missing_required_fields,
                fallback_count=session.fallback_count,
                reprompt_field=diagnostics["reprompt_field"],
                reprompt_count=diagnostics["reprompt_count"],
                reprompt_counts=dict(session.reprompt_count_by_field),
                reprompt_escalated=diagnostics["reprompt_escalated"],
                reprompt_escalation_count=session.reprompt_escalation_count,
                fallback_triggered=diagnostics["fallback_triggered"],
                fallback_reason=diagnostics["fallback_reason"],
                slot_extraction_failed=diagnostics["slot_extraction_failed"],
                slot_extraction_failure_count=session.slot_extraction_failure_count,
                invalid_modifier=diagnostics["invalid_modifier"],
                invalid_modifier_count=session.invalid_modifier_count,
                user_repeated=diagnostics["user_repeated"],
                repeated_user_turn_count=session.repeated_user_turn_count,
                preprocess_ms=preprocess_ms,
                nlu_ms=nlu_ms,
                flow_ms=flow_ms,
                route_ms=route_ms,
                handler_ms=handler_ms,
                total_ms=total_ms,
                latency_breakdown={
                    "preprocess_ms": preprocess_ms,
                    "nlu_ms": nlu_ms,
                    "flow_ms": flow_ms,
                    "route_ms": route_ms,
                    "handler_ms": handler_ms,
                    "total_ms": total_ms,
                },
            )
        except Exception as exc:
            print(f"[NLU_CSV_LOGGER_ERROR] {type(exc).__name__}: {exc}")

    def _trace_set_initial_fields(
        self,
        *,
        trace: Any | None,
        session: Session,
        user_text: str,
        state_before: ConversationState,
        engine_start_monotonic: float,
    ) -> None:
        ctx = session.conversation_context
        self._trace_set_attr(trace, "session_id", self._safe_session_id(session))
        self._trace_set_attr(trace, "user_text", user_text)
        self._trace_set_attr(trace, "state_before", state_before.value)
        self._trace_set_attr(trace, "pending_action", self._safe_pending_action_from_context(ctx))
        self._trace_set_attr(trace, "current_prompt_field", self._safe_current_prompt_field_from_context(ctx))
        self._trace_set_attr(trace, "current_item_id", self._safe_current_item_id_from_context(ctx))
        self._trace_set_attr(trace, "current_item_name", self._safe_current_item_name_from_context(ctx))
        self._trace_set_attr(trace, "engine_start_monotonic", engine_start_monotonic)

    def _trace_set_nlu_fields(self, *, trace: Any | None, nlu: Any) -> None:
        normalized_text = getattr(nlu, "normalized_text", "") or ""
        effective_intent = getattr(getattr(nlu, "effective_intent", None), "value", "") or ""

        self._trace_set_attr(trace, "normalized_text", normalized_text)
        self._trace_set_attr(trace, "pred_main_intent", getattr(nlu, "model_main_intent", "") or "")
        self._trace_set_attr(trace, "pred_sub_intent", getattr(nlu, "model_sub_intent", "") or "")
        self._trace_set_attr(trace, "pred_intent", effective_intent)
        self._trace_set_attr(trace, "pred_intent_confidence", getattr(nlu, "intent_confidence", None))
        self._trace_set_attr(trace, "stt_final_text_chars", len(normalized_text))

        slots = getattr(nlu, "slots", ()) or ()
        slot_names: list[str] = []
        slot_values: list[str] = []

        for slot in slots:
            label = getattr(slot, "label", None) or getattr(slot, "name", None)
            value = getattr(slot, "value", None)

            if isinstance(slot, dict):
                label = label or slot.get("label") or slot.get("name")
                value = value if value is not None else slot.get("value")

            if label:
                slot_names.append(str(label))
            if value is not None:
                slot_values.append(str(value))

        self._trace_set_attr(trace, "slot_names", slot_names)
        self._trace_set_attr(trace, "slot_values", slot_values)
        self._trace_set_attr(trace, "intent_model_ms", getattr(nlu, "intent_model_ms", None))
        self._trace_set_attr(trace, "slot_model_ms", getattr(nlu, "slot_model_ms", None))

    def _trace_finalize(
        self,
        *,
        trace: Any | None,
        session: Session,
        response_key: str,
        total_start_monotonic: float,
        total_ms: float,
        preprocess_ms: float,
        nlu_ms: float,
        flow_ms: float,
        route_ms: float,
        handler_ms: float,
        command: dict[str, Any] | None = None,
    ) -> None:
        engine_end = time.perf_counter()
        ctx = session.conversation_context

        self._trace_set_attr(trace, "response_key", response_key)
        self._trace_set_attr(trace, "state_after", session.conversation_state.value)
        self._trace_set_attr(trace, "pending_action", self._safe_pending_action_from_context(ctx))
        self._trace_set_attr(trace, "current_prompt_field", self._safe_current_prompt_field_from_context(ctx))
        self._trace_set_attr(trace, "current_item_id", self._safe_current_item_id_from_context(ctx))
        self._trace_set_attr(trace, "current_item_name", self._safe_current_item_name_from_context(ctx))
        self._trace_set_attr(trace, "engine_end_monotonic", engine_end)
        self._trace_set_attr(trace, "turn_total_ms", round(total_ms, 3))
        self._trace_set_attr(trace, "preprocess_ms", round(preprocess_ms, 3))
        self._trace_set_attr(trace, "nlu_ms", round(nlu_ms, 3))
        self._trace_set_attr(trace, "flow_ms", round(flow_ms, 3))
        self._trace_set_attr(trace, "route_ms", round(route_ms, 3))
        self._trace_set_attr(trace, "handler_ms", round(handler_ms, 3))
        self._trace_set_attr(trace, "engine_total_ms", round((engine_end - total_start_monotonic) * 1000.0, 3))
        if command is not None:
            self._trace_set_attr(trace, "command", command)

    def _trace_set_attr(self, trace: Any | None, attr_name: str, value: Any) -> None:
        if trace is None:
            return
        try:
            setattr(trace, attr_name, value)
        except Exception:
            return
