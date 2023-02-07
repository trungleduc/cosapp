import pytest
from unittest import mock
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


@mock.patch.object(SystemAnalyzer, 'clear')
def test_SystemAnalyzer_system(clear_method: mock.MagicMock, dummy):
    """Test getter/setter for attribute `system`
    """
    handler = SystemAnalyzer()
    assert handler.system is None

    # Check that changing system triggers method `clear`
    clear_method.reset_mock()
    handler.system = dummy
    assert clear_method.called
    assert handler.system is dummy

    # Check `system` setter does not trigger `clear` if system is unchanged
    clear_method.reset_mock()
    handler.system = dummy
    assert not clear_method.called
    assert handler.system is dummy

    clear_method.reset_mock()
    handler.system = None
    assert clear_method.called
    assert handler.system is None


@pytest.mark.parametrize("system, expected", [
    (System('dummy'), does_not_raise()),
    (None, pytest.raises(ValueError)),
])
def test_SystemAnalyzer_check_system(system, expected):
    handler = SystemAnalyzer(system)
    with expected:
        handler.check_system()
