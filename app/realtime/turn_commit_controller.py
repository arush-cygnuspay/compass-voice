# app/realtime/turn_commit_controller.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(slots=True)
class CommittedTurn:
    text: str


@dataclass(slots=True)
class TurnCommitController:
    """
    Aggregates streaming transcript events into a single committed user turn.

    Strategy:
    - keep the existing speech_final / utterance_end commit path as the safe fallback
    - add an early-commit path for short, obviously complete final transcripts
      so replies like "yes", "no", "coke", "sprite", "small", "i'm done",
      "add one coke" do not wait unnecessarily for later end-of-utterance events
    - avoid duplicate commits for the same utterance and for repeated text
    """

    final_segments: List[str] = field(default_factory=list)
    latest_interim: str = ""
    speech_started: bool = False
    utterance_active: bool = False
    committed_in_current_utterance: bool = False
    last_committed_text: str = ""

    def on_speech_started(self) -> None:
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
        Early-commit only when the current final transcript looks safely complete.

        This is intentionally conservative:
        - always commit if the built text ends with terminal punctuation
        - commit short command-like replies without waiting for speech_final
        - do not early-commit obviously incomplete prefixes
        """
        if self.committed_in_current_utterance:
            return False

        built = self._build_text()
        if not built:
            return False

        if built == self.last_committed_text:
            return False

        if built[-1:] in {".", "!", "?"}:
            return True

        words = built.lower().split()
        word_count = len(words)
        if word_count == 0:
            return False

        # Single-word replies are usually complete in this domain.
        if word_count == 1:
            return True

        # Two-word replies like "i'm done", "add coke", "no thanks" are also safe.
        if word_count == 2 and not self._ends_in_incomplete_token(words):
            return True

        # Small command-like replies can be committed early if they don't look truncated.
        if word_count <= 4 and not self._looks_incomplete_prefix(words):
            return True

        return False

    @staticmethod
    def _ends_in_incomplete_token(words: List[str]) -> bool:
        last = words[-1]
        return last in {
            "a",
            "an",
            "the",
            "and",
            "or",
            "to",
            "for",
            "of",
            "with",
            "plus",
        }

    def _looks_incomplete_prefix(self, words: List[str]) -> bool:
        text = " ".join(words)

        if self._ends_in_incomplete_token(words):
            return True

        incomplete_prefixes = {
            "i want",
            "i would",
            "i would like",
            "can i",
            "could i",
            "let me",
            "show me",
            "tell me",
            "how much",
            "what is",
            "what's",
            "which",
        }
        if text in incomplete_prefixes:
            return True

        incomplete_exact = {
            "i want a",
            "i want an",
            "i want the",
            "add a",
            "add an",
            "add the",
            "give me",
            "make it",
            "with a",
            "with an",
            "with the",
        }
        return text in incomplete_exact

    @staticmethod
    def _clean(text: str) -> str:
        return " ".join((text or "").strip().split())