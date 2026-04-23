from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.realtime.turn_commit_controller import TurnCommitController


def test_allows_same_text_in_new_utterance() -> None:
    controller = TurnCommitController()

    controller.on_speech_started()
    assert controller.on_transcript("one", is_final=True) is None
    first = controller.on_speech_final()
    assert first is not None
    assert first.text == "one"

    controller.on_speech_started()
    assert controller.on_transcript("one", is_final=True) is None
    second = controller.on_speech_final()
    assert second is not None
    assert second.text == "one"


def test_suppresses_duplicate_commit_within_same_utterance() -> None:
    controller = TurnCommitController()

    controller.on_speech_started()
    committed = controller.on_transcript("yes", is_final=True)
    assert committed is not None
    assert committed.text == "yes"

    assert controller.on_speech_final() is None
    assert controller.on_utterance_end() is None


def test_done_like_reply_commits_early_with_wrapped_phrase() -> None:
    controller = TurnCommitController()

    controller.on_speech_started()
    committed = controller.on_transcript("no done", is_final=True)

    assert committed is not None
    assert committed.text == "no done"
