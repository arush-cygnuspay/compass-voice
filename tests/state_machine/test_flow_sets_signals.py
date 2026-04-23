from app.state_machine.flow_sets import looks_like_done_answer


def test_done_signal_accepts_wrapped_real_world_phrases() -> None:
    assert looks_like_done_answer("no done")
    assert looks_like_done_answer("okay thats good thanks")
    assert looks_like_done_answer("yeah were good")


def test_done_signal_does_not_consume_modifier_removals() -> None:
    assert not looks_like_done_answer("no onions")
    assert not looks_like_done_answer("without sauce")
