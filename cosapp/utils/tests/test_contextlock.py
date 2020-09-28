from cosapp.utils.context import ContextLock


def test_ContextLock___init__():
    c = ContextLock()
    assert c.is_locked() == False


def test_ContextLock_context_manager():
    c = ContextLock()
    assert c.is_locked() == False

    with c:
        assert c.is_locked() == True

    assert c.is_locked() == False
