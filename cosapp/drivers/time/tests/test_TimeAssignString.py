import pytest
import numpy as np

from cosapp.drivers.time.scenario import TimeAssignString
from .conftest import ScalarOde, VectorOde


def test_TimeAssignString__init__():
    ode = ScalarOde('ode')
    interpolator = lambda t: t if t < 1 else 3 - 2 * t
    bc = TimeAssignString('df', interpolator, ode)
    assert bc.rhs is interpolator

    with pytest.raises(TypeError, match="must be a callable function"):
        TimeAssignString('df', 0.5, ode)


def test_TimeAssignString_exec_scalar(clock):
    ode = ScalarOde('ode')
    interpolator = lambda t: t if t < 1 else 3 - 2 * t
    bc = TimeAssignString('df', interpolator, ode)

    clock.time = 999

    for t in (0, 0.8, 1, 2, 3.4):
        bc.exec(t)
        assert ode.df == interpolator(t)
        assert ode.time == 999


def test_TimeAssignString_exec_vector(clock):
    ode = VectorOde('ode')
    interpolator = lambda t: np.array([t, np.exp(-t), np.sin(t)])
    bc = TimeAssignString('dv', interpolator, ode)
    assert bc.rhs is interpolator

    clock.time = 999

    for t in (0, 0.8, 1, 2, 3.4, -2.5):
        bc.exec(t)
        assert np.array_equal(ode.dv, interpolator(t))
        assert ode.time == 999


def test_TimeAssignString_exec_masked_array(clock):
    ode = VectorOde('ode')
    ode.dv[:] = -0.99
    interpolator = lambda t: np.array([t, np.exp(-t)])
    bc = TimeAssignString('dv[::2]', interpolator, ode)
    assert bc.rhs is interpolator

    clock.time = 999

    for t in (0, 0.8, 1, 2, 3.4, -2.5):
        bc.exec(t)
        assert np.array_equal(ode.dv[::2], interpolator(t))
        assert ode.dv[1] == -0.99
        assert ode.time == 999
