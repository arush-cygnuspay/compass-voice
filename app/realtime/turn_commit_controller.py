# app/realtime/turn_commit_controller.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from app.nlu.query_normalization.text_preprocessor import normalize_text
from app.state_machine.flow_sets import looks_like_done_answer


@dataclass(slots=True)
class CommittedTurn:
    text: str


@dataclass(slots=True)
class TurnCommitController:
    """
    Aggregates streaming transcript events into a single committed user turn.

    Strategy:
    - keep speech_final / utterance_end as the primary safe commit path
    - allow early commit only for a very small whitelist of obviously complete replies
    - avoid duplicate commits for the same utterance and repeated text

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
        return CommittedTurn(text=text)

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
        Early-commit only when the current final transcript is extremely likely
        to be a fully complete reply.

        Conservative rules:
        - terminal punctuation => safe
        - exact whitelist of confirmation/control replies => safe
        - otherwise wait for speech_final / utterance_end

        This avoids cutting off users who are still speaking option answers
        inside add-item flows.
        """
        if self.committed_in_current_utterance:
            return False

        built = self._build_text()
        if not built:
            return False

        if built == self.last_committed_text:
            return False

        normalized = normalize_text(built)

        if built[-1:] in {".", "!", "?"}:
            return True

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
