import pytest
from unittest import mock
from typing import Optional

from cosapp.drivers import Driver
from cosapp.systems import System
from cosapp.recorders import DataFrameRecorder
from cosapp.utils.options_dictionary import OptionsDictionary
from cosapp.utils.testing import get_args


@pytest.fixture(autouse=True)
def PatchDriver():
    """Patch Driver to make it instanciable for tests"""
    patcher = mock.patch.multiple(
        Driver,
        __abstractmethods__ = set(),
        is_standalone = lambda self: False,
    )
    patcher.start()
    yield
    patcher.stop()


@pytest.fixture(scope="function")
def driver():
    """Create dummy, detached driver"""
    return Driver("driver")


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
    (get_args("inwards"), dict(error=ValueError)),
    (get_args("outwards"), dict(error=ValueError)),
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
    
    error = expected.get("error", None)

    if error is None:
        d = Driver(*args, **kwargs)
        try:
            owner = args[1]
        except IndexError:
            owner = kwargs.get("owner", None)
        assert d.name == args[0]
        assert isinstance(d.options, OptionsDictionary)
        assert d.options["verbose"] == 0
        assert d.start_time == 0
        assert d.owner is owner
        assert d.recorder is None

    else:
        pattern = expected.get("match", None)
        with pytest.raises(error, match=pattern):
            Driver(*args, **kwargs)


def test_Driver_from_System():
    s = System('boss')
    d = s.add_driver(Driver('a_driver'))
    assert isinstance(d.options, OptionsDictionary)
    assert d.options["verbose"] == 0
    assert d.start_time == 0
    assert d.owner is s
    assert d.recorder is None


def test_Driver_owner():
    """Test getter/setter for attribute `Driver.owner`
    """
    def make_objects():
        return Driver('d'), System('s')
    
    d = Driver('d')
    assert d.owner is None

    d, s = make_objects()
    s.add_driver(d)
    assert d.owner is s

    d, s = make_objects()
    d.owner = s
    assert d.owner is s

    d, s = make_objects()
    foo = d.add_child(Driver('foo'))
    bar = d.add_child(Driver('bar'))
    sub = foo.add_child(Driver('sub'))
    assert d.owner is None
    assert foo.owner is None
    assert bar.owner is None
    assert sub.owner is None
    d.owner = s
    assert d.owner is s
    assert foo.owner is s
    assert bar.owner is s
    assert sub.owner is s

    d = Driver('d')
    foo = Driver('foo')
    with pytest.raises(TypeError):
        d.owner = foo


def test_Driver___repr__(driver: Driver):
    assert repr(driver) == "driver (alone) - Driver"

    boss = System('boss')
    d = boss.add_driver(Driver('d'))
    assert repr(d) == "d (on System 'boss') - Driver"


def test_Driver__setattr__(driver: Driver):
    # Error is raised when setting an absent attribute
    with pytest.raises(AttributeError):
        driver.ftol = 1e-5


def test_Driver__dir__(driver: Driver):
    """Test function dir(), useful for autocompletion
    """
    members = dir(driver)
    assert 'owner' in members
    assert 'children' in members
    assert 'recorder' in members


@pytest.mark.skip(reason="TODO")
def test_Driver__precompute():
    pytest.fail()


@pytest.mark.skip(reason="TODO")
def test_Driver__postcompute():
    pytest.fail()


@pytest.mark.parametrize("owner", [None, System('s')])
def test_Driver_add_child(driver: Driver, owner: Optional[System]):
    driver.owner = owner
    assert driver.owner is owner
    assert len(driver.children) == 0
    assert 'foo' not in dir(driver)
    assert 'bar' not in dir(driver)

    foo = driver.add_child(Driver('foo'))
    bar = driver.add_child(Driver('bar'))
    assert set(driver.children) == {'foo', 'bar'}
    assert driver.children['foo'] is foo
    assert driver.children['bar'] is bar
    assert foo.owner is driver.owner
    assert bar.owner is driver.owner
    assert 'foo' in dir(driver)
    assert 'bar' in dir(driver)

    with pytest.raises(ValueError, match="Module already contains an object with the same name"):
        driver.add_child(Driver('foo'))

    with pytest.raises(TypeError):
        driver.add_child(Driver)

    with pytest.raises(TypeError):
        driver.add_child(System('oops'))


@pytest.mark.parametrize("owner", [None, System('s')])
def test_Driver_add_driver(driver: Driver, owner: Optional[System]):
    driver.owner = owner
    assert driver.owner is owner
    assert len(driver.children) == 0
    assert 'foo' not in dir(driver)
    assert 'bar' not in dir(driver)

    foo = driver.add_driver(Driver('foo'))
    bar = driver.add_driver(Driver('bar'))
    assert set(driver.children) == {'foo', 'bar'}
    assert driver.children['foo'] is foo
    assert driver.children['bar'] is bar
    assert foo.owner is driver.owner
    assert bar.owner is driver.owner
    assert 'foo' in dir(driver)
    assert 'bar' in dir(driver)

    with pytest.raises(ValueError, match="Module already contains an object with the same name"):
        driver.add_driver(Driver('foo'))

    with pytest.raises(TypeError):
        driver.add_driver(Driver)

    with pytest.raises(TypeError):
        driver.add_driver(System('oops'))


def test_Driver_add_recorder(driver: Driver):
    rec = DataFrameRecorder()

    driver.add_recorder(rec)
    assert driver.recorder is rec

    rec2 = DataFrameRecorder()
    driver.add_recorder(rec2)
    assert driver.recorder is rec2
    assert driver.recorder is not rec

    with pytest.raises(TypeError):
        driver.add_recorder(System('oops'))


def test_Driver_recorder(driver: Driver):
    assert driver.recorder is None

    rec = DataFrameRecorder()
    with pytest.raises(AttributeError):
        driver.recorder = rec

    driver.add_recorder(rec)
    assert driver.recorder is rec


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
def test_Driver_check_owner_attr(ExtendedMultiply, attr, ok):
    bogus = ExtendedMultiply('bogus')
    driver = Driver('driver', bogus)

    if ok:
        assert attr in driver.owner
        driver.check_owner_attr(attr)
    else:
        with pytest.raises(AttributeError, match="'.*' not found in System 'bogus'"):
            driver.check_owner_attr(attr)
