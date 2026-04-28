from app.state_machine.flow_sets import looks_like_done_answer, looks_like_skip_answer


def test_done_signal_accepts_wrapped_real_world_phrases() -> None:
    assert looks_like_done_answer("no done")
    assert looks_like_done_answer("okay thats good thanks")
    assert looks_like_done_answer("yeah were good")
    assert looks_like_done_answer("done and done")
    assert looks_like_done_answer("no no more")


def test_done_signal_does_not_consume_modifier_removals() -> None:
    assert not looks_like_done_answer("no onions")
    assert not looks_like_done_answer("without sauce")
    assert not looks_like_done_answer("cheese and bacon")


def test_skip_signal_accepts_repeated_no() -> None:
    assert looks_like_skip_answer("no")
    assert looks_like_skip_answer("no no")
