from app.realtime.barge_in_policy import is_within_barge_in_guard_window


def test_returns_false_when_playback_not_started() -> None:
    assert is_within_barge_in_guard_window(None, 0.25, now_monotonic=10.0) is False


def test_returns_false_when_guard_disabled() -> None:
    assert is_within_barge_in_guard_window(10.0, 0.0, now_monotonic=10.05) is False
    assert is_within_barge_in_guard_window(10.0, -0.1, now_monotonic=10.05) is False


def test_returns_true_when_within_guard_window() -> None:
    # Playback started 100 ms ago, guard window is 250 ms → guarded.
    assert is_within_barge_in_guard_window(10.0, 0.25, now_monotonic=10.1) is True


def test_returns_true_at_zero_age() -> None:
    # Identical timestamp — playback "just started", definitely guarded.
    assert is_within_barge_in_guard_window(10.0, 0.25, now_monotonic=10.0) is True


def test_returns_false_at_or_after_guard_boundary() -> None:
    # Boundary is exclusive: age >= guard_seconds → not guarded.
    assert is_within_barge_in_guard_window(10.0, 0.25, now_monotonic=10.25) is False
    assert is_within_barge_in_guard_window(10.0, 0.25, now_monotonic=10.5) is False


def test_uses_real_clock_when_now_omitted(monkeypatch) -> None:
    fake_now = [42.0]

    monkeypatch.setattr(
        "app.realtime.barge_in_policy.time.monotonic",
        lambda: fake_now[0],
    )

    # Playback started 100 ms before "now" → should be guarded.
    assert is_within_barge_in_guard_window(41.9, 0.25) is True

    # Advance the fake clock past the guard boundary.
    fake_now[0] = 42.3
    assert is_within_barge_in_guard_window(41.9, 0.25) is False
