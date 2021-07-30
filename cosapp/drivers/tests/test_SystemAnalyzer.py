import pytest
from contextlib import nullcontext as does_not_raise

from cosapp.drivers.utils import SystemAnalyzer
from cosapp.systems import System


@pytest.fixture
def dummy():
    return System('dummy')


@pytest.mark.parametrize("system, expected", [
    (None, does_not_raise()),
    (System('dummy'), does_not_raise()),
    (0, pytest.raises(TypeError)),
    ('string', pytest.raises(TypeError)),
])
def test_SystemAnalyzer__init__(system, expected):
    with expected:
        handler = SystemAnalyzer(system)
        assert handler.system is system
        assert handler.data == dict()


def test_SystemAnalyzer_system(dummy):
    """Test getter/setter for attribute `system`
    """
    handler = SystemAnalyzer()
    assert handler.system is None
    assert handler.data == dict()
    handler.data['foo'] = 'bar'
    assert handler.data == dict(foo='bar')

    # Check that changing system resets data
    handler.system = dummy
    assert handler.system is dummy
    assert handler.data == dict()
    handler.data['foo'] = 'bar'
    assert handler.data == dict(foo='bar')

    # Check `system` setter does not reset data
    # if system is unchanged
    handler.system = dummy
    assert handler.system is dummy
    assert handler.data == dict(foo='bar')

    handler.system = None
    assert handler.system is None
    assert handler.data == dict()


@pytest.mark.parametrize("system, expected", [
    (System('dummy'), does_not_raise()),
    (None, pytest.raises(ValueError)),
])
def test_SystemAnalyzer_check_system(system, expected):
    handler = SystemAnalyzer(system)
    with expected:
        handler.check_system()
