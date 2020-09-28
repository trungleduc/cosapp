import logging
from unittest import mock

import pytest

from cosapp.core import signal as signal_module
from cosapp.core.signal import Signal, Slot


def test_Signal_is_connected(slot_factory):
    signal_a = Signal(threadsafe=True)
    signal_b = Signal(args=["foo"])
    slot_a = slot_factory('mock')
    slot_b = slot_factory('mock')
    signal_a.connect(Slot(slot_a))

    assert signal_a.is_connected(slot_a)
    assert not signal_a.is_connected(slot_b)
    assert not signal_b.is_connected(slot_a)
    assert not signal_b.is_connected(slot_b)


def test_Signal_emit_one_slot(slot_factory):
    signal_a = Signal(threadsafe=True)
    slot_a = slot_factory('mock')
    slot_b = slot_factory('mock')
    signal_a.connect(Slot(slot_a))

    signal_a.emit()

    slot_a.assert_called_once_with()
    assert slot_b.call_count == 0


def test_Signal_emit_two_slots(slot_factory):
    signal_a = Signal(threadsafe=True)
    slot_a = slot_factory('mock')
    slot_b = slot_factory('mock')
    signal_a.connect(Slot(slot_a))
    signal_a.connect(Slot(slot_b))

    signal_a.emit()

    slot_a.assert_called_once_with()
    slot_b.assert_called_once_with()


@mock.patch("cosapp.core.signal.signal.inspect")
def test_Signal_emit_one_slot_with_arguments(inspect, slot_factory):
    signal_b = Signal(args=["foo"])
    slot_a = slot_factory('mock')
    slot_b = slot_factory('mock')
    signal_b.connect(Slot(slot_a))

    signal_b.emit(foo="bar")

    slot_a.assert_called_once_with(foo="bar")
    assert slot_b.call_count == 0


@mock.patch("cosapp.core.signal.signal.inspect")
def test_Signal_emit_two_slots_with_arguments(inspect, slot_factory):
    signal_b = Signal(args=["foo"])
    slot_a = slot_factory('mock')
    slot_b = slot_factory('mock')

    signal_b.connect(Slot(slot_a))
    signal_b.connect(Slot(slot_b))

    signal_b.emit(foo="bar")

    slot_a.assert_called_once_with(foo="bar")
    slot_b.assert_called_once_with(foo="bar")


def test_Signal_reconnect_does_not_duplicate(slot_factory):
    signal_a = Signal(threadsafe=True)
    slot_a = slot_factory('mock')
    signal_a.connect(Slot(slot_a))
    signal_a.connect(Slot(slot_a))
    signal_a.emit()

    slot_a.assert_called_once_with()


def test_Signal_disconnect_does_not_fail_on_not_connected_slot(slot_factory):
    signal_a = Signal(threadsafe=True)
    slot_b = slot_factory('mock')
    signal_a.disconnect(slot_b)


@pytest.mark.parametrize("kwargs, expected", [
    (dict(), "NO_NAME"),
    (dict(name="foo.bar"), "foo.bar"),
    (dict(name="update_stuff"), "update_stuff"),
])
def test_Signal__repr__(kwargs, expected):
    signal = Signal(**kwargs)
    assert repr(signal) == f"<cosapp.core.signal.Signal: {expected}>"


def test_Signal_connect_not_slot():
    signal = Signal()

    def cb(**kwargs):
        pass
    # Should be accepted (automatically wrap in a Slot)
    signal.connect(cb)

    a = "hello"
    with pytest.raises(TypeError, match="Only callable objects"):
        signal.connect(a)


def test_Signal_connect_with_kwargs():
    def cb(**kwargs):
        pass

    signal = Signal()
    signal2 = Signal(args=["dummy"])

    signal.connect(Slot(cb))
    signal2.connect(Slot(cb))


def test_Signal_connect_without_kwargs():
    def cb():
        pass

    signal = Signal()
    signal2 = Signal(args=["dummy"])

    signal.connect(Slot(cb))

    with pytest.raises(TypeError, match="function must accept keyword arguments"):
        signal2.connect(Slot(cb))


class MyTestError(Exception):
    pass


def test_Signal_emit_exception(caplog):
    signal = Signal(threadsafe=False)

    def failing_slot(**args):
        raise MyTestError("die!")

    signal.connect(Slot(failing_slot, weak=False))

    # Don't raise MyTestError
    caplog.clear()
    with caplog.at_level(logging.ERROR, logger=signal_module.__name__):
        signal.emit()

        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert record.levelno == logging.ERROR
        assert record.message.startswith("MyTestError('die!'")


@pytest.mark.parametrize("weak", [True, False])
@pytest.mark.parametrize("obj, ok", [
    (lambda *args: None, True),
    (lambda *args, **kwargs: 0.12, True),
    (lambda **kwargs: None, True),
    (float, True),
    ("hello", False),
])
def test_Slot__init__(obj, weak, ok):
    if ok:
        slot = Slot(obj, weak)
        assert slot.func is obj

    else:
        pattern = "must be defined from a callable object"
        with pytest.raises(TypeError, match=pattern):
            Slot(obj, weak)


def test_Slot_non_weak_alive():
    def slot(**kwargs):
        pass

    slot = Slot(slot, weak=False)
    assert slot.is_alive


def test_Slot_non_weak_call():
    called = False

    def slot(**kwargs):
        nonlocal called
        called = True

    slot = Slot(slot, weak=False)

    assert not called
    slot(testing=1234)
    assert called


def test_Slot_weak_function_alive(slot_factory):

    slot_ref = slot_factory('function')
    slot = Slot(slot_ref, weak=True)

    assert slot.is_alive
    assert repr(slot) == f"<cosapp.core.signal.Slot: {repr(slot_ref)}>"


def test_Slot_weak_function_call():
    called = False

    def slot_factory():
        def slot(**kwargs):
            nonlocal called
            called = True

        return slot

    slot_ref = slot_factory()
    slot = Slot(slot_ref, weak=True)

    assert not called
    slot(testing=1234)
    assert called


def test_Slot_weak_function_gc(slot_factory):

    slot_ref = slot_factory('function')
    slot = Slot(slot_ref, weak=True)

    slot_ref = None
    assert not slot.is_alive
    assert repr(slot) == "<cosapp.core.signal.Slot: dead>"
    slot(testing=1234)


def test_Slot_weak_method_alive(slot_factory):

    obj_ref = slot_factory('object')
    slot = Slot(obj_ref.slot, weak=True)
    signal = Signal()
    signal.connect(slot)

    assert slot.is_alive


def test_Slot_weak_method_call(slot_factory):

    obj_ref = slot_factory('object')
    slot = Slot(obj_ref.slot, weak=True)
    signal = Signal()
    signal.connect(slot)

    assert not obj_ref.called
    signal.emit(testing=1234)
    assert obj_ref.called


def test_Slot_weak_method_gc(slot_factory):

    obj_ref = slot_factory('object')
    slot = Slot(obj_ref.slot, weak=True)
    signal = Signal()
    signal.connect(slot)

    obj_ref = None
    assert not slot.is_alive
    signal.emit(testing=1234)


@pytest.mark.parametrize("weak_a", [True, False])
@pytest.mark.parametrize("weak_b", [True, False])
def test_Slot_eq_other(weak_a, weak_b):
    def f(**kwargs):
        pass

    slot_a = Slot(f, weak=weak_a)
    slot_b = Slot(f, weak=weak_b)

    assert slot_a is not slot_b
    assert slot_a == slot_b


@pytest.mark.parametrize("weak", [True, False])
def test_Slot_eq_self(weak):
    def f(**kwargs):
        pass

    slot = Slot(f, weak)
    assert slot == slot


@pytest.mark.parametrize("weak", [True, False])
def test_Slot_eq_func(weak):
    def f(**kwargs):
        pass

    slot = Slot(f, weak)
    assert slot is not f
    assert slot == f
