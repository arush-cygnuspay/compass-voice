# app/nlu/semantic_repair/option_resolver_replay.py
"""Phase 3.5 Offline Replay + Evaluation Harness — core library.

Provides the importable engine for replaying captured turn logs or fixture
turns through the Phase 3 option resolver decision path without mutating any
production state (cart, session, FSM, or order).

Usage (from CLI or tests)
--------------------------
    from app.nlu.semantic_repair.option_resolver_replay import (
        Phase3OptionResolverReplayHarness,
        ReplayInputTurn,
        ReplayResult,
        ReplaySummaryBuilder,
        BUILT_IN_FIXTURES,
        parse_jsonl_row,
    )

    harness = Phase3OptionResolverReplayHarness(config=cfg)
    for result in harness.replay_fixtures(BUILT_IN_FIXTURES):
        print(result)

Safety invariants
------------------
* GPT is NEVER called unless `use_live_gpt=True` is passed at construction.
* `actual_applied` is always False — the harness never applies selections.
* No API key, PII, or cart mutation in output.
* All group/choice objects are synthetic (name-only) — no real menu loaded.
"""
from __future__ import annotations

import json
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Generator, Iterator

_WAITING_FOR_MODIFIER_STATE = "waiting_for_modifier"
_DEFAULT_GROUP_NAME = "Options"

# ---------------------------------------------------------------------------
# Input / output dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReplayInputTurn:
    """A single turn ready for Phase 3 option resolver replay.

    All fields are optional except `user_text` and `state_before`.
    Missing fields degrade gracefully: empty choices → NO_GPT route.
    """

    user_text: str
    state_before: str
    source_turn_id: str | None = None
    response_key_before: str | None = None
    local_intent: str | None = None
    local_confidence: float | None = None
    local_slots: tuple[dict, ...] = ()
    top_intents: tuple[dict, ...] = ()
    choice_names: tuple[str, ...] = ()
    group_name: str = _DEFAULT_GROUP_NAME
    item_name: str = "Item"
    repeat_count: int = 0
    # None = auto-detect from user_text via has_correction_signal()
    has_correction_signal: bool | None = None
    session_id: str | None = None
    turn_index: int | None = None


@dataclass(frozen=True, slots=True)
class ReplayResult:
    """Structured result for a single replayed turn.

    `actual_applied` is always False — the harness never applies results.
    `would_apply` reflects what WOULD happen in inline mode if this were live.
    """

    replay_id: str
    source_turn_id: str | None
    user_text: str
    state_before: str
    mode: str
    response_key_before: str | None = None
    local_intent: str | None = None
    local_confidence: float | None = None
    local_slots: tuple[dict, ...] = ()
    route_mode: str = "no_gpt"
    gpt_called: bool = False
    gpt_decision: str | None = None
    gpt_selected_names: tuple[str, ...] = ()
    gpt_confidence: float | None = None
    validator_passed: bool = False
    validator_reject_reason: str | None = None
    safe_to_apply: bool = False
    would_apply: bool = False
    actual_applied: bool = False  # always False
    error: str | None = None
    latency_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict (no PII, no API keys)."""
        return {
            "replay_id": self.replay_id,
            "source_turn_id": self.source_turn_id,
            "user_text": self.user_text,
            "state_before": self.state_before,
            "mode": self.mode,
            "response_key_before": self.response_key_before,
            "local_intent": self.local_intent,
            "local_confidence": self.local_confidence,
            "local_slots": list(self.local_slots),
            "route_mode": self.route_mode,
            "gpt_called": self.gpt_called,
            "gpt_decision": self.gpt_decision,
            "gpt_selected_names_or_ids": list(self.gpt_selected_names),
            "gpt_confidence": self.gpt_confidence,
            "validator_passed": self.validator_passed,
            "validator_reject_reason": self.validator_reject_reason,
            "safe_to_apply": self.safe_to_apply,
            "would_apply": self.would_apply,
            "actual_applied": False,
            "error": self.error,
            "latency_ms": self.latency_ms,
        }


# ---------------------------------------------------------------------------
# JSONL input parser
# ---------------------------------------------------------------------------


def parse_jsonl_row(
    row: dict[str, Any],
    *,
    filter_state: str | None = None,
) -> ReplayInputTurn | None:
    """Convert a gpt_repair_turns.jsonl record to a ReplayInputTurn.

    Returns None when the row is malformed, missing required fields, or
    filtered out by `filter_state`.

    The nested JSONL structure expected:
        {
          "session_id": "...",
          "turn_index": 1,
          "state_before": "waiting_for_modifier",
          "normalized_text": "macarola cheese",
          "response_key": "ask_for_modifier",
          "local": {"intent": "...", "confidence": 0.72, "slots": [], "top_intents": []},
          "allowed": {"choices": ["..."], ...},
        }
    """
    try:
        state = str(row.get("state_before") or "").strip().lower()
        if filter_state and state != filter_state.lower().strip():
            return None

        user_text = str(
            row.get("normalized_text") or row.get("customer_text") or ""
        ).strip()
        if not user_text:
            return None  # silence turn — skip

        local_block = row.get("local") or {}
        allowed_block = row.get("allowed") or {}

        local_intent = str(local_block.get("intent") or "").strip() or None
        local_confidence_raw = local_block.get("confidence")
        local_confidence: float | None = None
        if local_confidence_raw is not None:
            try:
                local_confidence = float(local_confidence_raw)
            except (TypeError, ValueError):
                pass

        raw_slots = local_block.get("slots") or []
        local_slots: tuple[dict, ...] = ()
        if isinstance(raw_slots, list):
            local_slots = tuple(
                s for s in raw_slots if isinstance(s, dict)
            )

        raw_top = local_block.get("top_intents") or []
        top_intents: tuple[dict, ...] = ()
        if isinstance(raw_top, list):
            top_intents = tuple(t for t in raw_top if isinstance(t, dict))

        raw_choices = allowed_block.get("choices") or []
        choice_names: tuple[str, ...] = ()
        if isinstance(raw_choices, list):
            choice_names = tuple(
                str(c).strip() for c in raw_choices if c and str(c).strip()
            )

        session_id = str(row.get("session_id") or "").strip() or None
        turn_index_raw = row.get("turn_index")
        turn_index: int | None = None
        if turn_index_raw is not None:
            try:
                turn_index = int(turn_index_raw)
            except (TypeError, ValueError):
                pass

        source_id = (
            f"{session_id}:{turn_index}"
            if session_id and turn_index is not None
            else session_id or str(turn_index) if session_id or turn_index else None
        )

        return ReplayInputTurn(
            user_text=user_text,
            state_before=state,
            source_turn_id=source_id,
            response_key_before=str(row.get("response_key") or "").strip() or None,
            local_intent=local_intent,
            local_confidence=local_confidence,
            local_slots=local_slots,
            top_intents=top_intents,
            choice_names=choice_names,
            session_id=session_id,
            turn_index=turn_index,
        )

    except Exception:
        return None


# ---------------------------------------------------------------------------
# Synthetic PendingModifierGroup builder
# ---------------------------------------------------------------------------


def _build_synthetic_group(
    choice_names: tuple[str, ...],
    group_name: str = _DEFAULT_GROUP_NAME,
) -> "Any":
    """Build a PendingModifierGroup from option names for replay.

    modifier_ids are synthetic (index-based) because logs don't store them.
    The Phase 3 validator uses name-based matching, so synthetic IDs are safe
    for routing + validation purposes.
    """
    from app.state_machine.models.pending_item_models import (
        PendingModifierChoice,
        PendingModifierGroup,
    )

    choices = [
        PendingModifierChoice(
            modifier_id=f"synthetic_{i}",
            name=name,
            group_id="replay_grp",
            normalized_name=name.lower(),
        )
        for i, name in enumerate(choice_names)
        if name.strip()
    ]
    return PendingModifierGroup(
        group_id="replay_grp",
        name=group_name,
        is_required=True,
        min_selector=1,
        max_selector=1,
        choices=choices,
    )


# ---------------------------------------------------------------------------
# Core harness
# ---------------------------------------------------------------------------


class Phase3OptionResolverReplayHarness:
    """Replay turns through the Phase 3 option resolver without production side effects.

    Parameters
    ----------
    config:
        SemanticRepairConfig. `option_resolver_mode` is overridden per-call
        by the `mode` argument to `replay_turn()`.
    use_live_gpt:
        If False (default), GPT is never called — the service returns
        `skipped/missing_api_key`. Set to True with a real API key for staging.
    mock_client:
        Inject a test double for the OpenAI client (used in unit tests).
    """

    def __init__(
        self,
        config: "Any | None" = None,
        *,
        use_live_gpt: bool = False,
        mock_client: "Any | None" = None,
    ) -> None:
        from app.config.semantic_repair import SemanticRepairConfig, get_semantic_repair_config

        if config is None:
            config = get_semantic_repair_config()
        self._base_config = config
        self._use_live_gpt = use_live_gpt
        self._mock_client = mock_client

    def _make_service(self, mode: str) -> "Any":
        """Create a GptOptionResolverService with the requested mode."""
        from app.config.semantic_repair import SemanticRepairConfig
        from app.nlu.semantic_repair.option_resolver_service import GptOptionResolverService

        cfg = SemanticRepairConfig(
            phase=self._base_config.phase,
            model=self._base_config.model,
            timeout_seconds=float(
                getattr(self._base_config, "option_resolver_timeout_ms", 1200) / 1000.0
            ),
            option_resolver_mode=mode,
            option_resolver_timeout_ms=getattr(
                self._base_config, "option_resolver_timeout_ms", 1200
            ),
            option_resolver_min_confidence=getattr(
                self._base_config, "option_resolver_min_confidence", 0.75
            ),
            option_resolver_repeat_threshold=getattr(
                self._base_config, "option_resolver_repeat_threshold", 2
            ),
        )
        svc = GptOptionResolverService(config=cfg)
        if self._mock_client is not None:
            svc._client = self._mock_client
        elif not self._use_live_gpt:
            # Ensure no live calls by pre-setting client to a no-op sentinel
            svc._client = None  # will hit missing_api_key path
        return svc

    def replay_turn(self, turn: ReplayInputTurn, *, mode: str) -> ReplayResult:
        """Replay a single turn through the Phase 3 option resolver.

        Parameters
        ----------
        turn : ReplayInputTurn
        mode : "disabled" | "shadow" | "inline"

        Returns
        -------
        ReplayResult — never raises; errors are captured in `result.error`.
        """
        from app.nlu.semantic_repair.option_routing_policy import has_correction_signal

        replay_id = str(uuid.uuid4())[:12]
        t_start = time.perf_counter()

        try:
            svc = self._make_service(mode)
            group = _build_synthetic_group(turn.choice_names, turn.group_name)

            # Auto-detect correction signal if not provided
            correction = (
                turn.has_correction_signal
                if turn.has_correction_signal is not None
                else has_correction_signal(turn.user_text)
            )

            local_slots_list = [dict(s) for s in turn.local_slots]
            top_intents_list = [dict(t) for t in turn.top_intents]

            result = svc.run(
                user_text=turn.user_text,
                item_name=turn.item_name,
                group=group,
                existing_selections=[],
                local_resolved=False,  # conservative: always attempt resolution
                repeat_count=turn.repeat_count,
                previous_turns=None,
                last_response_key=turn.response_key_before,
                local_slots=local_slots_list or None,
                top_intents=top_intents_list or None,
                has_correction_signal=correction,
            )

            latency_ms = (time.perf_counter() - t_start) * 1000.0

            # would_apply: what WOULD happen if this were live in inline mode
            would_apply = (
                mode == "inline"
                and result.safe_to_apply
                and result.decision == "select_option"
                and bool(result.selected_names)
            )

            # Determine validator reject reason
            reject_reason: str | None = None
            if result.gpt_called and not result.safe_to_apply:
                if result.decision == "no_match":
                    reject_reason = "gpt_no_match"
                elif result.decision == "error":
                    reject_reason = f"gpt_error:{result.reason_code}"
                elif result.route_mode == "shadow_gpt":
                    reject_reason = "shadow_mode"
                elif result.confidence is not None and result.confidence < getattr(
                    svc._config, "option_resolver_min_confidence", 0.75
                ):
                    reject_reason = f"low_confidence:{result.confidence:.2f}"
                elif result.selected_names:
                    reject_reason = "invalid_names_or_max_selector"
                else:
                    reject_reason = "unknown"

            return ReplayResult(
                replay_id=replay_id,
                source_turn_id=turn.source_turn_id,
                user_text=turn.user_text,
                state_before=turn.state_before,
                mode=mode,
                response_key_before=turn.response_key_before,
                local_intent=turn.local_intent,
                local_confidence=turn.local_confidence,
                local_slots=turn.local_slots,
                route_mode=result.route_mode,
                gpt_called=result.gpt_called,
                gpt_decision=result.decision if result.gpt_called else None,
                gpt_selected_names=result.selected_names,
                gpt_confidence=result.confidence if result.gpt_called else None,
                validator_passed=result.safe_to_apply,
                validator_reject_reason=reject_reason,
                safe_to_apply=result.safe_to_apply,
                would_apply=would_apply,
                actual_applied=False,
                error=result.parse_error,
                latency_ms=latency_ms,
            )

        except Exception as exc:
            latency_ms = (time.perf_counter() - t_start) * 1000.0
            return ReplayResult(
                replay_id=replay_id,
                source_turn_id=turn.source_turn_id,
                user_text=turn.user_text,
                state_before=turn.state_before,
                mode=mode,
                error=f"{type(exc).__name__}: {exc}"[:200],
                latency_ms=latency_ms,
            )

    def replay_fixtures(
        self,
        fixtures: list[ReplayInputTurn],
        *,
        mode: str,
    ) -> Generator[ReplayResult, None, None]:
        """Replay a list of fixture turns. Yields one ReplayResult per turn."""
        for turn in fixtures:
            yield self.replay_turn(turn, mode=mode)

    def replay_jsonl(
        self,
        path: "str | Any",
        *,
        mode: str,
        max_turns: int = 0,
        filter_state: str | None = _WAITING_FOR_MODIFIER_STATE,
        filter_response_key: str | None = None,
    ) -> Generator[ReplayResult, None, None]:
        """Stream-replay turns from a gpt_repair_turns.jsonl file.

        Parameters
        ----------
        path:
            Path to the JSONL file.
        mode:
            GPT mode for this replay run.
        max_turns:
            Stop after this many successful turns (0 = unlimited).
        filter_state:
            Only replay turns where `state_before` matches this value.
            Defaults to "waiting_for_modifier". Pass None to replay all states.
        filter_response_key:
            Optional additional filter on `response_key`.

        Yields
        ------
        ReplayResult for each qualifying turn.
        """
        from pathlib import Path as _Path

        path = _Path(path)
        processed = 0

        with path.open(encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    yield _error_result(
                        error=f"JSONDecodeError on line {line_no}",
                        user_text="",
                        state_before=filter_state or "",
                        mode=mode,
                    )
                    continue

                if not isinstance(row, dict):
                    continue

                turn = parse_jsonl_row(row, filter_state=filter_state)
                if turn is None:
                    continue  # filtered out or malformed

                if filter_response_key and turn.response_key_before != filter_response_key:
                    continue

                result = self.replay_turn(turn, mode=mode)
                yield result

                processed += 1
                if max_turns and processed >= max_turns:
                    break


def _error_result(
    *,
    error: str,
    user_text: str,
    state_before: str,
    mode: str,
) -> ReplayResult:
    return ReplayResult(
        replay_id=str(uuid.uuid4())[:12],
        source_turn_id=None,
        user_text=user_text,
        state_before=state_before,
        mode=mode,
        error=error,
    )


# ---------------------------------------------------------------------------
# Summary builder
# ---------------------------------------------------------------------------


class ReplaySummaryBuilder:
    """Aggregates ReplayResult objects into a structured summary.

    Usage::

        builder = ReplaySummaryBuilder(mode="shadow")
        for result in harness.replay_fixtures(BUILT_IN_FIXTURES, mode="shadow"):
            builder.add(result)
        summary = builder.to_dict()
        print(builder.to_markdown())
    """

    def __init__(self, mode: str) -> None:
        self._mode = mode
        self._total = 0
        self._by_state: Counter[str] = Counter()
        self._gpt_called = 0
        self._inline_candidate = 0
        self._validator_pass = 0
        self._validator_reject = 0
        self._would_apply = 0
        self._error = 0
        self._timeout = 0
        self._reject_reasons: Counter[str] = Counter()
        self._route_modes: Counter[str] = Counter()
        self._gpt_decisions: Counter[str] = Counter()
        self._would_apply_examples: list[dict] = []
        self._rejected_examples: list[dict] = []
        self._error_examples: list[dict] = []
        self._latencies_ms: list[float] = []

    def add(self, result: ReplayResult) -> None:
        """Register one replay result."""
        self._total += 1
        self._by_state[result.state_before or "unknown"] += 1
        self._route_modes[result.route_mode] += 1

        if result.latency_ms is not None:
            self._latencies_ms.append(result.latency_ms)

        if result.error:
            self._error += 1
            if len(self._error_examples) < 5:
                self._error_examples.append({
                    "user_text": result.user_text[:80],
                    "state": result.state_before,
                    "error": result.error[:120],
                })
            # Count timeout separately if it looks like a timeout
            if "timeout" in (result.error or "").lower():
                self._timeout += 1
            return

        if result.gpt_called:
            self._gpt_called += 1
            self._gpt_decisions[result.gpt_decision or "none"] += 1

        if result.route_mode == "inline_gpt":
            self._inline_candidate += 1

        if result.validator_passed:
            self._validator_pass += 1
        elif result.gpt_called and not result.validator_passed:
            self._validator_reject += 1
            reason = result.validator_reject_reason or "unknown"
            self._reject_reasons[reason] += 1
            if len(self._rejected_examples) < 5:
                self._rejected_examples.append({
                    "user_text": result.user_text[:80],
                    "state": result.state_before,
                    "gpt_decision": result.gpt_decision,
                    "gpt_selected": list(result.gpt_selected_names),
                    "reject_reason": reason,
                })

        if result.would_apply:
            self._would_apply += 1
            if len(self._would_apply_examples) < 5:
                self._would_apply_examples.append({
                    "user_text": result.user_text[:80],
                    "state": result.state_before,
                    "gpt_selected": list(result.gpt_selected_names),
                    "gpt_confidence": result.gpt_confidence,
                })

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable summary dict."""
        p50, p95, p_max = _percentiles(self._latencies_ms)
        return {
            "mode": self._mode,
            "total_turns": self._total,
            "by_state": dict(self._by_state),
            "route_modes": dict(self._route_modes),
            "gpt_called": self._gpt_called,
            "inline_candidate_count": self._inline_candidate,
            "validator_pass_count": self._validator_pass,
            "validator_reject_count": self._validator_reject,
            "would_apply_count": self._would_apply,
            "error_count": self._error,
            "timeout_count": self._timeout,
            "reject_reasons": dict(self._reject_reasons),
            "gpt_decisions": dict(self._gpt_decisions),
            "latency_ms": {"p50": p50, "p95": p95, "max": p_max},
            "would_apply_examples": self._would_apply_examples,
            "rejected_examples": self._rejected_examples,
            "error_examples": self._error_examples,
        }

    def to_markdown(self) -> str:
        """Render a human-readable markdown summary."""
        d = self.to_dict()
        total = max(d["total_turns"], 1)
        gpt = d["gpt_called"]

        lines: list[str] = [
            "# Phase 3.5 Option Resolver Replay Summary",
            "",
            f"**Mode**: `{d['mode']}`  ",
            f"**Total turns replayed**: {d['total_turns']}",
            "",
            "## Routing",
            "",
            "| Route Mode | Count | % |",
            "|------------|-------|---|",
        ]
        for mode, cnt in sorted(d["route_modes"].items(), key=lambda x: -x[1]):
            lines.append(f"| `{mode}` | {cnt} | {100*cnt/total:.1f}% |")

        lines += [
            "",
            "## GPT Calls",
            "",
            f"- **GPT called**: {gpt} ({100*gpt/total:.1f}%)",
            f"- **Inline candidates**: {d['inline_candidate_count']}",
            f"- **Validator pass**: {d['validator_pass_count']}",
            f"- **Validator reject**: {d['validator_reject_count']}",
            f"- **Would apply (offline)**: {d['would_apply_count']}",
            f"- **Errors**: {d['error_count']}",
            f"- **Timeouts**: {d['timeout_count']}",
        ]

        if d["gpt_decisions"]:
            lines += ["", "### GPT Decision Distribution", ""]
            for dec, cnt in sorted(d["gpt_decisions"].items(), key=lambda x: -x[1]):
                lines.append(f"- `{dec}`: {cnt} ({100*cnt/max(gpt,1):.1f}%)")

        if d["reject_reasons"]:
            lines += ["", "### Validator Reject Reasons", ""]
            for reason, cnt in sorted(d["reject_reasons"].items(), key=lambda x: -x[1]):
                lines.append(f"- `{reason}`: {cnt}")

        lat = d["latency_ms"]
        if lat["p50"] or lat["p95"] or lat["max"]:
            lines += [
                "",
                "## Latency (ms)",
                "",
                f"- p50: {lat['p50']:.1f}",
                f"- p95: {lat['p95']:.1f}",
                f"- max: {lat['max']:.1f}",
            ]

        if d["would_apply_examples"]:
            lines += ["", "## Would-Apply Examples (GPT improves local behavior)", ""]
            for ex in d["would_apply_examples"]:
                lines.append(
                    f"- `{ex['user_text']}` → {ex['gpt_selected']} "
                    f"(conf={ex.get('gpt_confidence', '?'):.2f})"
                    if ex.get("gpt_confidence") else
                    f"- `{ex['user_text']}` → {ex['gpt_selected']}"
                )

        if d["rejected_examples"]:
            lines += ["", "## Rejected Examples (needs prompt or routing improvement)", ""]
            for ex in d["rejected_examples"]:
                lines.append(
                    f"- `{ex['user_text']}` → rejected: `{ex['reject_reason']}` "
                    f"(selected={ex['gpt_selected']})"
                )

        if d["error_examples"]:
            lines += ["", "## Errors", ""]
            for ex in d["error_examples"]:
                lines.append(f"- `{ex['user_text']}`: {ex['error']}")

        lines += ["", "---", "", "*Generated by Phase 3.5 Option Resolver Replay Harness*"]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _percentiles(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    s = sorted(values)
    n = len(s)
    p50 = s[int(n * 0.50)]
    p95 = s[min(int(n * 0.95), n - 1)]
    return p50, p95, s[-1]


# ---------------------------------------------------------------------------
# Built-in fixtures
# ---------------------------------------------------------------------------

BUILT_IN_FIXTURES: list[ReplayInputTurn] = [
    # 1. Phonetic mismatch — canonical Phase 3 example
    ReplayInputTurn(
        user_text="macarola cheese",
        state_before="waiting_for_modifier",
        source_turn_id="fixture:1",
        response_key_before="ask_for_modifier",
        local_intent="add_item",
        local_confidence=0.72,
        choice_names=("American Cheese", "Cheddar Cheese", "Mozzarella Cheese", "Swiss Cheese"),
        group_name="Cheese",
        item_name="Burger",
        repeat_count=0,
    ),
    # 2. Fuzzy spelling — single word
    ReplayInputTurn(
        user_text="mozarella",
        state_before="waiting_for_modifier",
        source_turn_id="fixture:2",
        response_key_before="ask_for_modifier",
        local_intent="unknown",
        local_confidence=0.18,
        choice_names=("American Cheese", "Mozzarella Cheese", "Cheddar Cheese"),
        group_name="Cheese",
        item_name="Burger",
        repeat_count=0,
    ),
    # 3. Exact / deterministic match — local should resolve without GPT
    #    In replay, local_resolved is always False (conservative), so GPT
    #    will be attempted but validator should approve quickly.
    ReplayInputTurn(
        user_text="cheddar",
        state_before="waiting_for_modifier",
        source_turn_id="fixture:3",
        response_key_before="ask_for_modifier",
        local_intent="add_item",
        local_confidence=0.88,
        choice_names=("American Cheese", "Mozzarella Cheese", "Cheddar Cheese"),
        group_name="Cheese",
        item_name="Burger",
        repeat_count=0,
    ),
    # 4. Correction signal — "no, mozzarella" should escalate to INLINE_GPT
    ReplayInputTurn(
        user_text="no, mozzarella",
        state_before="waiting_for_modifier",
        source_turn_id="fixture:4",
        response_key_before="ask_for_modifier",
        local_intent="deny",
        local_confidence=0.82,
        choice_names=("American Cheese", "Mozzarella Cheese", "Swiss Cheese"),
        group_name="Cheese",
        item_name="Burger",
        repeat_count=0,
        has_correction_signal=True,
    ),
    # 5. Nonsense text — GPT should return no_match → not safe
    ReplayInputTurn(
        user_text="zibblequark snicklefritz supreme",
        state_before="waiting_for_modifier",
        source_turn_id="fixture:5",
        response_key_before="ask_for_modifier",
        local_intent="unknown",
        local_confidence=0.05,
        choice_names=("American Cheese", "Mozzarella Cheese", "Cheddar Cheese"),
        group_name="Cheese",
        item_name="Burger",
        repeat_count=0,
    ),
    # 6. "That's all" — required modifier should not be skipped
    #    No choices match this → routing should NOT produce a safe apply
    ReplayInputTurn(
        user_text="that's all",
        state_before="waiting_for_modifier",
        source_turn_id="fixture:6",
        response_key_before="ask_for_modifier",
        local_intent="done",
        local_confidence=0.74,
        choice_names=("American Cheese", "Mozzarella Cheese"),
        group_name="Cheese",
        item_name="Burger",
        repeat_count=0,
    ),
    # 7. Repeat-loop: same bad response twice → repeat_count >= threshold escalates to INLINE_GPT
    ReplayInputTurn(
        user_text="um",
        state_before="waiting_for_modifier",
        source_turn_id="fixture:7",
        response_key_before="repeat_modifier_options",
        local_intent="unknown",
        local_confidence=0.10,
        choice_names=("American Cheese", "Mozzarella Cheese", "Cheddar Cheese"),
        group_name="Cheese",
        item_name="Burger",
        repeat_count=3,  # >= default threshold=2 → escalates
    ),
]
