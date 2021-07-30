import pytest
from contextlib import nullcontext as does_not_raise

from cosapp.drivers.utils import UnknownAnalyzer
from cosapp.systems import System
from cosapp.core.numerics.basics import MathematicalProblem


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
        handler = UnknownAnalyzer(system)
        assert handler.system is system
        assert handler.data == dict()


def test_SystemAnalyzer_filter_problem_undefined(dummy):
    """Test method `filter_problem` with undefined system
    """
    handler = UnknownAnalyzer(None)
    assert handler.system is None

    problem = MathematicalProblem('problem', dummy)
    
    with pytest.raises(ValueError):
        handler.filter_problem(problem)
