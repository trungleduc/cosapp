import pytest

import numpy as np

from cosapp.core.numerics.boundary import AbstractTimeUnknown
from cosapp.core.eval_str import EvalString
from cosapp.systems import System


class DummyTimeUnknown(AbstractTimeUnknown):
    """Minimalist concrete implementation of AbstractTimeUnknown"""
    def __init__(self, context, der, max_time_step=np.inf, max_abs_step=np.inf):
        self.context = context
        self.__der = EvalString(der, context)
        self.__max_dt = EvalString(max_time_step, context)
        self.__max_dx = EvalString(max_abs_step, context)

    @property
    def der(self) -> EvalString:
        """Expression of the time derivative, given as an EvalString"""
        return self.__der

    @property
    def max_time_step_expr(self) -> EvalString:
        """Expression of the maximum admissible time step, given as an EvalString."""
        return self.__max_dt

    @property
    def max_abs_step_expr(self) -> EvalString:
        """Expression of the maximum admissible variable step, given as an EvalString."""
        return self.__max_dx

    def reset(self):
        """Reset transient unknown to a reference value"""
        pass

    def touch(self):
        """Set owner port as 'dirty'."""
        pass


class SubSystem(System):
    def setup(self):
        self.add_inward('x', 1.0)
        self.add_inward('y', 1.0)


class DummySystem(System):
    def setup(self):
        self.add_child(SubSystem('sub1'))
        self.add_child(SubSystem('sub2'))

        self.add_inward('u', np.zeros(3))


@pytest.fixture(scope="function")
def system():
    s = DummySystem("system")
    s.u = np.r_[0.1, 0.2, 0.3]
    s.sub1.x = 3
    s.sub1.y = 1.23
    s.sub2.x = -2.4
    s.sub2.y = 0.5
    return s


@pytest.fixture(scope="function")
def unknowns(system):
    """Returns a list of dummy unknowns"""
    a = DummyTimeUnknown(system, der=0.5)
    b = DummyTimeUnknown(system, der="sub1.x")
    c = DummyTimeUnknown(system, der="sub1.y**2", max_time_step="0.1 * sub1.x")
    return [a, b, c]


@pytest.mark.parametrize("ctor_data, expected", [
    (dict(der=0.5), dict(d_dt=0.5)),
    (dict(der="0.5"), dict(d_dt=0.5)),
    (dict(der="sub1.x"), dict(d_dt=3)),
    (dict(der="1 / 0"), dict(error=ZeroDivisionError)),
    (dict(der="sub1.x", max_time_step="1 / 0"), dict(error=ZeroDivisionError)),
    (dict(der="sub1.x", max_time_step=0.01), dict(constrained=True, d_dt=3, dt_max=0.01)),
    (dict(der="sub1.x", max_time_step="inf"), dict(constrained=False, d_dt=3)),
    (dict(der="sub1.y**2", max_time_step="0.1 * sub1.x"), dict(d_dt=1.5129, constrained=True, dt_max=0.3)),
    (dict(der="sub1.y**2", max_time_step="min(abs(sub2.x), abs(sub2.y))"), dict(d_dt=1.5129, constrained=True, dt_max=0.5)),
    (dict(der="sub1.y**2", max_time_step="-abs(sub2.y)"), dict(d_dt=1.5129, constrained=True, dt_max=-0.5)),  # no value check at this point
    (dict(der="sub1.y**2", max_time_step=0), dict(d_dt=1.5129, constrained=True, dt_max=0)),  # no value check at this point
])
def test_AbstractTimeUnknown__init__(system, ctor_data, expected):
    error = expected.get('error', None)

    if error is None:
        unknown = DummyTimeUnknown(system, **ctor_data)
        assert str(unknown.der) == str(ctor_data['der'])
        assert unknown.d_dt == pytest.approx(expected['d_dt'], rel=1e-15)
        assert unknown.constrained == expected.get('constrained', False)
        max_time_step = unknown.max_time_step
        try:
            assert max_time_step == pytest.approx(expected['dt_max'], rel=1e-15)
        except KeyError:
            assert not np.isfinite(max_time_step)
    else:
        pattern = expected.get('match', None)
        with pytest.raises(error, match=pattern):
            DummyTimeUnknown(system, **ctor_data)


@pytest.mark.parametrize("state, ctor_data, expected", [
    (
        dict(),
        dict(der=0.5),
        dict(d_dt=0.5),
    ),
    (
        {"sub1.x": -0.2, "sub2.y": 5},
        dict(der="sub1.x * sub2.y"),
        dict(d_dt=-1)
    ),
    (
        {"sub1.x": -0.2, "sub2.y": 5},
        dict(der="sub1.x", max_time_step=0.01),
        dict(d_dt=-0.2, dt_max=0.01)
    ),
    (
        {"sub1.x": -0.2, "sub2.y": 5, "u": np.r_[0, 2.5, -10]},
        dict(der="sub1.x", max_time_step="0.01 * abs(sub2.y) / norm(u, inf)"),
        dict(d_dt=-0.2, dt_max=0.005)
    ),
])
def test_AbstractTimeUnknown_max_time_step(system, state, ctor_data, expected):
    unknown = DummyTimeUnknown(system, **ctor_data)

    for var, value in state.items():
        system[var] = value

    error = expected.get('error', None)

    if error is None:
        assert unknown.d_dt == pytest.approx(expected['d_dt'], rel=1e-15)
        max_time_step = unknown.max_time_step
        try:
            assert max_time_step == pytest.approx(expected['dt_max'], rel=1e-15)
        except KeyError:
            assert not np.isfinite(max_time_step)
    else:
        pattern = expected.get('match', None)
        with pytest.raises(error, match=pattern):
            unknown.max_time_step


@pytest.mark.parametrize("state, ctor_data, step, expected", [
    (
        dict(),
        dict(der=0.5), 1,
        dict(dt_max=pytest.approx(2, rel=1e-15)),
    ),
    (
        {"u": np.r_[-0.1, 1.2, 2.5]},
        dict(der="0.1 * u"), np.ones(3),
        dict(dt_max=pytest.approx(4, rel=1e-15))
    ),
    (
        {"u": np.r_[-0.1, 1.2, 2.5]},
        dict(der="-0.1 * u"), np.ones(3),
        dict(dt_max=pytest.approx(4, rel=1e-15))
    ),
    (
        {"u": np.r_[-0.1, 1.2, 2.5]},
        dict(der="-0.1 * u"), 0.1,  # Note: vector quantity, with scalar step (OK)
        dict(dt_max=pytest.approx(0.4, rel=1e-15))
    ),
    (
        {"u": np.zeros(3)},
        dict(der="-0.1 * u"), np.ones(3),
        dict(dt_max=np.inf)
    ),
    (
        {"u": np.r_[0.1, 0, -2]},
        dict(der="-0.1 * u"), np.ones(3),
        dict(dt_max=pytest.approx(5, rel=1e-15))
    ),
    (
        {"sub1.x": -0.2, "sub2.y": 5},
        dict(der="sub1.x"), 0,
        dict(dt_max=0)
    ),
    (
        {"sub1.x": 0, "sub2.y": 5},
        dict(der="sub1.x * sub2.y"), 0.5,
        dict(dt_max=np.inf)
    ),
    (
        {"sub1.x": 0.1, "sub2.y": 5},
        dict(der="sub1.x * sub2.y"), 0.2,
        dict(dt_max=pytest.approx(0.4, rel=1e-15))
    ),
    (
        {"sub1.x": 0.1, "sub2.y": 5},
        dict(der="sub1.x * sub2.y"), np.inf,
        dict(dt_max=np.inf)
    ),
])
def test_AbstractTimeUnknown_extrapolated_time_step(system, state, ctor_data, step, expected):
    unknown = DummyTimeUnknown(system, **ctor_data)

    for var, value in state.items():
        system[var] = value

    dt_max = unknown.extrapolated_time_step(step)
    assert dt_max >= 0
    try:
        assert dt_max == expected['dt_max']
    except KeyError:
        assert not np.isfinite(dt_max)
