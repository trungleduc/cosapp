import logging
import re
import weakref
from unittest import mock

import pytest

import cosapp.core.module as module_module
from cosapp.core.module import Module
from cosapp.core.signal import Slot
from cosapp.drivers import Driver
from cosapp.utils.logging import LogFormat, LogLevel
from cosapp.utils.orderedset import OrderedSet
from cosapp.utils.testing import no_exception


class MockModule(Module):
    def is_standalone(self) -> bool:
        return False


@pytest.fixture(scope="function")
def fake():
    return MockModule("fake")


@pytest.mark.parametrize("name, error", [
    ("a", None),
    ("A", None),
    ("foobar", None),
    ("foo4bar", None),
    ("loveYou2", None),
    ("CamelBack", None),
    ("foo_bar", None),
    ("foobar_", None),
    ("_foobar", ValueError),
    ("foo bar", ValueError),
    ("foobar?", ValueError),
    ("foo.bar", ValueError),
    ("foo:bar", ValueError),
    ("foo/bar", ValueError),
    ("1foobar", ValueError),
    ("foobar-2", ValueError),
    ("foobar:2", ValueError),
    ("foobar.2", ValueError),
    (23, TypeError),
    (1.0, TypeError),
    (dict(a=True), TypeError),
    (list(), TypeError),
])
def test_Module__init__(name, error):
    if error is None:
        module = MockModule(name)
        assert module.name == name
    else:
        with pytest.raises(error):
            MockModule(name)


def test_Module__weakref__(fake):
    with no_exception():
        proxy = weakref.proxy(fake)


def test_Module_name_setter():
    module = MockModule("foo")
    assert module.name == "foo"
    module.name = "bar"
    assert module.name == "bar"
    with pytest.raises(ValueError):
        module.name = "invalid.name"


def test_Module_contextual_name():
    s = MockModule("s")
    T = MockModule("T")
    u = MockModule("u")
    v = MockModule("v")

    s.add_child(T)
    T.add_child(u)
    u.add_child(v)

    assert v.contextual_name == "T.u.v"
    assert u.contextual_name == "T.u"
    assert s.contextual_name == ""
    assert T.contextual_name == "T"


@pytest.mark.parametrize("skip_root", [True, False])
def test_Module__get_fullname(skip_root):
    s = MockModule("s")
    T = MockModule("T")
    u = MockModule("u")
    v = MockModule("v")

    s.add_child(T)
    T.add_child(u)
    u.add_child(v)

    if skip_root:
        assert v._get_fullname(skip_root) == "T.u.v"
        assert u._get_fullname(skip_root) == "T.u"
        assert s._get_fullname(skip_root) == ""
        assert T._get_fullname(skip_root) == "T"
    else:
        assert v._get_fullname(skip_root) == "s.T.u.v"
        assert u._get_fullname(skip_root) == "s.T.u"
        assert s._get_fullname(skip_root) == "s"
        assert T._get_fullname(skip_root) == "s.T"


def test_Module_compute_calls():
    m = MockModule("m")

    assert m.compute_calls == 0

    m.run_once()
    assert m.compute_calls == 1

    m.call_setup_run()  # Reset counter
    assert m.compute_calls == 0

def test_Module_size():
    s = MockModule("s")
    T = MockModule("T")
    u = MockModule("u")
    v = MockModule("v")

    s.add_child(T)
    T.add_child(u)
    T.add_child(v)
    s.add_child(MockModule("r"))

    assert s.size == 5
    assert u.size == 1
    assert T.size == 3


def test_Module_get_path_to_child():
    s = MockModule("s")
    T = MockModule("T")
    u = MockModule("u")
    v = MockModule("v")

    s.add_child(T)
    T.add_child(u)
    u.add_child(v)

    assert s.get_path_to_child(v) == "T.u.v"
    assert s.get_path_to_child(u) == "T.u"
    assert s.get_path_to_child(s) == ""
    assert T.get_path_to_child(v) == "u.v"


def test_Module_pop_child_driver():
    # From Driver
    d = Driver("dummy")
    assert d.children == {}

    update = d.add_child(Driver("mydriver"))
    assert d.children == {"mydriver": update}

    d.pop_child("mydriver")
    assert d.children == {}

    with pytest.raises(AttributeError):
        d.pop_child("mydriver")


@pytest.mark.parametrize("values, expected", [
    ([], dict(values=[])),
    (None, dict(values=[])),
    (['foo', 'bar'], dict(values=['foo', 'bar'])),
    (['foo', 'bar', 'foo'], dict(values=['foo', 'bar'])),
    (('foo', 'bar', 'foo'), dict(values=['foo', 'bar'])),
    ([1, 'foo'], dict(error=TypeError, match="All elements of .*exec_order must be strings")),
    (2, dict(error=TypeError, match="object is not iterable")),
])
def test_Module_exec_order(fake, values, expected):
    error = expected.get("error", None)
    if error is None:
        fake.exec_order = values
        assert isinstance(fake.exec_order, OrderedSet)
        assert list(fake.exec_order) == expected["values"]
    else:
        pattern = expected.get("match", None)
        with pytest.raises(error, match=pattern):
            fake.exec_order = values


def test_Module_setup_ran(fake):
    with mock.patch("cosapp.core.signal.signal.inspect"):
        fake_callback = mock.Mock(spec=lambda **kwargs: None)
        fake_callback.return_value = None

        fake.setup_ran.connect(Slot(fake_callback))
        fake.call_setup_run()

        fake_callback.assert_called_once_with()


def test_Module_computed(fake):
    with mock.patch("cosapp.core.signal.signal.inspect"):
        fake_callback = mock.Mock(spec=lambda **kwargs: None)
        fake_callback.return_value = None

        fake.computed.connect(Slot(fake_callback))
        fake.run_once()

        fake_callback.assert_called_once_with()


def test_Module_clean_ran(fake):
    with mock.patch("cosapp.core.signal.signal.inspect"):
        fake_callback = mock.Mock(spec=lambda **kwargs: None)
        fake_callback.return_value = None

        fake.clean_ran.connect(Slot(fake_callback))
        fake.call_clean_run()

        fake_callback.assert_called_once_with()


@pytest.mark.parametrize("format", LogFormat)
@pytest.mark.parametrize("msg, kwargs, to_log, emitted", [
    ("zombie call_setup_run", dict(), False, None),
    ("useless start call_clean_run", dict(activate=True), False, None),
    (f"{Module.CONTEXT_EXIT_MESSAGE} call_clean_run", dict(activate=False), False, dict(levelno=LogLevel.DEBUG, pattern=r"Compute calls for [\w\.]+: \d+")),
    ("common message", dict(), True, None),
])
def test_Module_log_debug_message(format, msg, kwargs, to_log, emitted):
    handler = mock.MagicMock(level=LogLevel.DEBUG, log=mock.MagicMock())
    rec = logging.getLogRecordFactory()("log_test", LogLevel.INFO, __file__, 22, msg, (), None)
    for key, value in kwargs.items():
        setattr(rec, key, value)
    
    m = MockModule("dummy")

    assert m.log_debug_message(handler, rec, format) == to_log

    if emitted:
        handler.log.assert_called_once()
        args = handler.log.call_args[0]
        assert args[0] == emitted["levelno"]
        assert re.match(emitted["pattern"], args[1]) is not None
    else:
        handler.log.assert_not_called()
