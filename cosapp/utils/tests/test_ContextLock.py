from cosapp.utils.context import ContextLock


def test_ContextLock___init__():
    context = ContextLock()
    assert not context.is_active


def test_ContextLock_is_active():
    context = ContextLock()
    assert not context.is_active

    with context:
        assert context.is_active

    assert not context.is_active
