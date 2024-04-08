import pytest
from unittest import mock
from cosapp.base import Port, System
from cosapp.core.numerics.basics import BaseProblem
from cosapp.utils.testing import no_exception


@pytest.fixture(autouse=True)
def PatchBaseProblem():
    """Patch BaseProblem to make it instanciable for tests"""
    patcher = mock.patch.multiple(
        BaseProblem,
        __abstractmethods__ = set(),
    )
    patcher.start()
    yield
    patcher.stop()


def test_BaseProblem__init__():
    # Empty case
    pb = BaseProblem('test', None)
    assert pb.name == 'test'
    assert pb.context is None

    s = System('s')
    pb = BaseProblem('pb', s)
    assert pb.name == 'pb'
    assert pb.context is s


def test_BaseProblem_name():
    s = System('s')
    pb = BaseProblem('pb', s)
    with pytest.raises(AttributeError):
        pb.name = 'new_name'


def test_BaseProblem_context():
    a, b = System('a'), System('b')

    pb = BaseProblem('test', a)
    assert pb.context is a

    with no_exception():
        pb.context = a
    with pytest.raises(ValueError, match="Context is already set to 'a'"):
        pb.context = b
    assert pb.context is a

    # Same with initially None context
    pb = BaseProblem('test', None)
    assert pb.context is None

    pb.context = a
    assert pb.context is a
    with no_exception():
        pb.context = a
    with pytest.raises(ValueError, match="Context is already set to 'a'"):
        pb.context = b
    assert pb.context is a
