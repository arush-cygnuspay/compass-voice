# app/realtime/turn_commit_controller.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from app.nlu.query_normalization.text_preprocessor import normalize_text
from app.state_machine.flow_sets import looks_like_done_answer


@dataclass(slots=True)
class CommittedTurn:
    text: str
    turn_id: int = 0


@dataclass(slots=True)
class TurnCommitController:
    """
    Aggregates streaming transcript events into a single committed user turn.

    Strategy:
    - speech_final / utterance_end are the primary commit signals
    - early commit is limited to a small whitelist of obviously-complete
      single-utterance replies (yes/no/ok/cancel/checkout)
    - terminal punctuation alone does NOT trigger early commit — that was the
      main cause of split utterances like "chicken burger … with cheese"
    - a debounce timer (USER_TURN_COMMIT_DELAY_MS) is managed externally by
      voice_stream_server.py to handle the gap between consecutive finals

    Important:
    This controller is intentionally conservative for voice ordering flows.
    We do NOT early-commit generic single-word answers like:
      - cheese
      - fries
      - coke
      - large
      - medium
    because those often arrive before the user is actually done speaking.
    """

    final_segments: List[str] = field(default_factory=list)
    latest_interim: str = ""
    speech_started: bool = False
    utterance_active: bool = False
    committed_in_current_utterance: bool = False
    last_committed_text: str = ""
    _turn_counter: int = field(default=0)

    def on_speech_started(self) -> None:
        # Duplicate suppression is only valid within a single utterance.
        # If the caller repeats the same answer on the next utterance
        # ("one", "yes", etc.), we must commit it again.
        self.last_committed_text = ""
        self.speech_started = True
        self.utterance_active = True
        self.committed_in_current_utterance = False

    def on_transcript(self, text: str, is_final: bool) -> Optional[CommittedTurn]:
        cleaned = self._clean(text)
        if not cleaned:
            return None

        if is_final:
            if not self.final_segments or self.final_segments[-1] != cleaned:
                self.final_segments.append(cleaned)

            if self._should_commit_early():
                return self.commit()
            return None

        self.latest_interim = cleaned
        return None

    def on_utterance_end(self) -> Optional[CommittedTurn]:
        return self.commit()

    def on_speech_final(self) -> Optional[CommittedTurn]:
        return self.commit()

    def commit(self) -> Optional[CommittedTurn]:
        text = self._build_text()
        self._clear_buffers()
        self.committed_in_current_utterance = True

        if not text:
            return None

        if text == self.last_committed_text:
            return None

        self.last_committed_text = text
        self._turn_counter += 1
        return CommittedTurn(text=text, turn_id=self._turn_counter)

    def reset(self) -> None:
        self._clear_buffers()
        self.committed_in_current_utterance = False
        self.last_committed_text = ""

    def _clear_buffers(self) -> None:
        self.final_segments.clear()
        self.latest_interim = ""
        self.speech_started = False
        self.utterance_active = False

    def _build_text(self) -> str:
        if self.final_segments:
            return self._merge_segments(self.final_segments)
        return self._clean(self.latest_interim)

    def _merge_segments(self, segments: List[str]) -> str:
        merged = " ".join(self._clean(segment) for segment in segments if self._clean(segment))
        return self._clean(merged)

    def _should_commit_early(self) -> bool:
        """
        Early-commit only for a narrow whitelist of obviously complete replies.

        Terminal punctuation alone is NOT sufficient — it caused premature
        splits on multi-fragment utterances like "chicken burger. with cheese."
        The debounce timer in voice_stream_server.py handles the general case.
        """
        if self.committed_in_current_utterance:
            return False

        built = self._build_text()
        if not built:
            return False

        if built == self.last_committed_text:
            return False

        normalized = normalize_text(built)

        safe_exact_replies = {
            "yes",
            "yeah",
            "yep",
            "yup",
            "correct",
            "right",
            "ok",
            "okay",
            "no",
            "nope",
            "nah",
            "cancel",
            "stop",
            "checkout",
            "check out",
        }

        return normalized in safe_exact_replies or looks_like_done_answer(normalized)

    @staticmethod
    def _clean(text: str) -> str:
        return " ".join((text or "").strip().split())
