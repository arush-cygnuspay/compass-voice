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
    """

    final_segments: List[str] = field(default_factory=list)
    latest_interim: str = ""
    speech_started: bool = False
    utterance_active: bool = False
    last_committed_text: str = ""

    def on_speech_started(self) -> None:
        self.speech_started = True
        self.utterance_active = True

    def on_transcript(self, text: str, is_final: bool) -> None:
        cleaned = self._clean(text)
        if not cleaned:
            return

        if is_final:
            if not self.final_segments or self.final_segments[-1] != cleaned:
                self.final_segments.append(cleaned)
        else:
            self.latest_interim = cleaned

    def on_utterance_end(self) -> Optional[CommittedTurn]:
        return self.commit()

    def on_speech_final(self) -> Optional[CommittedTurn]:
        return self.commit()

    def commit(self) -> Optional[CommittedTurn]:
        text = self._build_text()
        self.reset()

        if not text:
            return None

        if text == self.last_committed_text:
            return None

        self.last_committed_text = text
        return CommittedTurn(text=text)

    def reset(self) -> None:
        self.final_segments.clear()
        self.latest_interim = ""
        self.speech_started = False
        self.utterance_active = False

    def _build_text(self) -> str:
        if self.final_segments:
            return self._merge_segments(self.final_segments)

        return self._clean(self.latest_interim)

    def _merge_segments(self, segments: List[str]) -> str:
        merged = " ".join(self._clean(s) for s in segments if self._clean(s))
        merged = " ".join(merged.split())
        return merged.strip()

    @staticmethod
    def _clean(text: str) -> str:
        return " ".join((text or "").strip().split())