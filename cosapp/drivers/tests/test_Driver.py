import pytest

from cosapp.drivers import Driver
from cosapp.systems import System
from cosapp.recorders import DataFrameRecorder
from cosapp.utils.options_dictionary import OptionsDictionary
from cosapp.utils.testing import get_args

# TODO add fundamental case with subsystem solving itself - this lacks here so TurboFan
# and PressureLosses case should be kept without those tests.

class DummyDriver(Driver):

    __slots__ = ()

    def compute(self):
        pass

@pytest.fixture(scope="function")
def dummy():
    return DummyDriver("dummy")


@pytest.mark.parametrize("ctor_data, expected", [
    # Test name pattern:
    (get_args("a"), dict()),
    (get_args("A"), dict()),
    (get_args("foobar"), dict()),
    (get_args("foo4bar"), dict()),
    (get_args("loveYou2"), dict()),
    (get_args("CamelBack"), dict()),
    (get_args("foo_bar"), dict()),
    (get_args("foobar_"), dict()),
    # Erroneous cases:
    (get_args("_foobar"), dict(error=ValueError)),
    (get_args("foo bar"), dict(error=None)),
    (get_args("foobar?"), dict(error=ValueError)),
    (get_args("foo.bar"), dict(error=ValueError)),
    (get_args("foo:bar"), dict(error=ValueError)),
    (get_args("foo/bar"), dict(error=ValueError)),
    (get_args("1foobar"), dict(error=ValueError)),
    (get_args("foobar-2"), dict(error=None)),
    (get_args("foobar:2"), dict(error=ValueError)),
    (get_args("foobar.2"), dict(error=ValueError)),
    (get_args(23), dict(error=TypeError)),
    (get_args(1.0), dict(error=TypeError)),
    (get_args(dict(a=True)), dict(error=TypeError)),
    (get_args(list()), dict(error=TypeError)),
    # Tests with specified owner:
    (get_args("foobar", System("boss")), dict()),
    (get_args("foobar", owner=System("boss")), dict()),
    (get_args("foobar", owner=None), dict()),
    (get_args("foobar", owner="boss"), dict(error=TypeError, match="owner"))
])
def test_Driver__init__(ctor_data, expected):
    """Test object instantiation"""
    args, kwargs = ctor_data
    assert len(args) > 0
    try:
        owner = args[1]
    except:
        owner = kwargs.get("owner", None)
    error = expected.get("error", None)

    if error is None:
        d = DummyDriver(*args, **kwargs)
        assert d.name == args[0]
        assert isinstance(d.options, OptionsDictionary)
        assert d.options["verbose"] == 0
        assert d.start_time == 0
        assert d.owner is owner
        assert d.recorder is None
    else:
        pattern = expected.get("match", None)
        with pytest.raises(error, match=pattern):
            d = DummyDriver(*args, **kwargs)


def test_Driver_from_System():
    s = System('boss')
    d = s.add_driver(DummyDriver('a_driver'))
    assert isinstance(d.options, OptionsDictionary)
    assert d.options["verbose"] == 0
    assert d.start_time == 0
    assert d.owner is s
    assert d.recorder is None


def test_Driver_owner():
    d = DummyDriver('the_driver')
    assert d.owner is None

    d = DummyDriver('the_driver')
    s = System('dummy')
    s.add_driver(d)
    assert d.owner is s

    d = DummyDriver('the_driver')
    d.owner = s
    assert d.owner is s

    d = DummyDriver('the_driver')
    d1 = DummyDriver('the_subdriver')
    d.add_child(d1)
    d.owner = s
    assert d1.owner is s

    d = DummyDriver('the_driver')
    d1 = DummyDriver('the_subdriver')
    with pytest.raises(TypeError):
        d.owner = d1


def test_Driver___repr__(dummy):
    assert repr(dummy) == "dummy (alone) - DummyDriver"

    s = System('boss')
    d = s.add_driver(DummyDriver('d'))
    assert repr(d) == "d (on System 'boss') - DummyDriver"


def test_Driver__setattr__(dummy):
    # Error is raised when setting an absent attribute
    with pytest.raises(AttributeError):
        dummy.ftol = 1e-5


@pytest.mark.skip(reason="TODO")
def test_Driver__precompute():
    pytest.fail()


@pytest.mark.skip(reason="TODO")
def test_Driver__postcompute():
    pytest.fail()


def test_Driver_add_child(dummy):
    assert len(dummy.children) == 0

    update = dummy.add_child(Driver('foo'))
    assert set(dummy.children.keys()) == {'foo'}
    assert dummy.children['foo'] is update

    with pytest.raises(ValueError, match="Module already contains an object with the same name"):
        dummy.add_child(Driver('foo'))

    with pytest.raises(TypeError):
        dummy.add_child(Driver)

    with pytest.raises(TypeError):
        dummy.add_child(System('oops'))


def test_Driver_add_driver(dummy):
    assert len(dummy.children) == 0

    update = dummy.add_driver(Driver('foo'))
    assert set(dummy.children.keys()) == {'foo'}
    assert dummy.children['foo'] is update

    with pytest.raises(ValueError, match="Module already contains an object with the same name"):
        dummy.add_driver(Driver('foo'))

    with pytest.raises(TypeError):
        dummy.add_driver(Driver)

    with pytest.raises(TypeError):
        dummy.add_driver(System('oops'))


def test_Driver_add_recorder(dummy):
    rec = DataFrameRecorder()

    dummy.add_recorder(rec)
    assert dummy.recorder is rec

    rec2 = DataFrameRecorder()
    dummy.add_recorder(rec2)
    assert dummy.recorder is rec2
    assert dummy.recorder is not rec

    with pytest.raises(TypeError):
        dummy.add_recorder(System('oops'))


def test_Driver_recorder(dummy):
    assert dummy.recorder is None

    rec = DataFrameRecorder()
    with pytest.raises(AttributeError):
        dummy.recorder = rec

    dummy.add_recorder(rec)
    assert dummy.recorder is rec


@pytest.mark.parametrize("attr, ok", [
    ('K1', True),
    ('K2', True),
    ('Ksum', True),
    ('inwards.K1', True),
    ('outwards.Ksum', True),
    ('banana', False),
    ('p_in.x', True),
    ('p_out.x', True),
    ('p_in.foo', False),
    ('p_out.foo', False),
])
def test_Driver_check_owner_attr(DummyFactory, attr, ok):
    bogus = DummyFactory('bogus')
    driver = Driver('driver', bogus)

    if ok:
        assert attr in driver.owner
        driver.check_owner_attr(attr)
    else:
        with pytest.raises(AttributeError, match="'.*' not found in System 'bogus'"):
            driver.check_owner_attr(attr)
