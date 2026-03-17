import pytest


class DummyContext:
    def __init__(self):
        self.pending_item = "Chicken Taco"
        self.waiting_for = "modifier"
        self.cart = []


def cancel_order(ctx):
    ctx.pending_item = None
    ctx.waiting_for = None


def test_cancel_flow():
    ctx = DummyContext()

    cancel_order(ctx)

    assert ctx.pending_item is None
    assert ctx.waiting_for is None


def test_skip_flow():
    ctx = DummyContext()

    ctx.waiting_for = "modifier"
    ctx.waiting_for = None

    assert ctx.waiting_for is None