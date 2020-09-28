import pytest

import numpy as np
from numbers import Number

from cosapp.systems import System
from cosapp.core.numerics.boundary import TimeDerivative
from cosapp.core.eval_str import EvalString
from cosapp.utils.testing import get_args


class BogusDynamicSystem(System):
    def setup(self):
        self.add_inward('h', 0.5)
        self.add_inward('x', np.r_[0.5, 2, 1.3])
        self.add_outward('norm_x')

        self.add_rate('dh_dt', source='h')
        self.add_rate('u', source='norm_x')
        self.add_rate('v', source='x')
        self.add_rate('c', source=2)  # stupid, but possible

    def compute(self):
        self.norm_x = np.linalg.norm(self.x)


@pytest.fixture(scope='function')
def bogus():
    s = BogusDynamicSystem('bogus')
    assert len(s.inwards) == 6
    return s


@pytest.mark.parametrize("ctor_data, state, expected", [
    (
        get_args('dh_dt', source=-1),
        dict(),
        dict(
            port = 'inwards',
            name = 'inwards.dh_dt',
            desc = 'd(-1)/dt',
            source = -1,
        )
    ),
    (
        get_args('dh_dt', source='2 * h', initial_value=-0.1),
        dict(h = 0.35),
        dict(
            port = 'inwards',
            name = 'inwards.dh_dt',
            desc = 'd(h)/dt',
            source = 0.7,
            value = -0.1,
        )
    ),
    (
        get_args('v', source='-0.1 * x'),
        dict(x = np.r_[0.5, 2, 1.3]),
        dict(
            port = 'inwards',
            name = 'inwards.v',
            desc = 'd(v)/dt',
            source = [-0.05, -0.2, -0.13],
        )
    ),
    (
        get_args('v', source='-0.1 * x', initial_value=np.r_[1.6, 2, 3]),
        dict(x = np.r_[0.5, 2, 1.3]),
        dict(
            port = 'inwards',
            name = 'inwards.v',
            desc = 'd(v)/dt',
            source = [-0.05, -0.2, -0.13],
            value = [1.6, 2, 3],
        )
    ),
    (
        get_args('dh_dt', source='x[-1]', initial_value='x[0] / 10'),
        dict(x = np.r_[0.5, 2, 1.3]),
        dict(
            port = 'inwards',
            name = 'inwards.dh_dt',
            desc = 'd(h)/dt',
            source = 1.3,
            value = 0.05,
        )
    ),
    (
        get_args('dh_dt', source='x[-1]', initial_value=np.r_[1.6, 2, 3]),
        dict(),
        dict(error=TypeError, match=r"Expression .* is incompatible with declared type Number")
    ),
    (
        get_args('v', source='x', initial_value=np.ones(5)),
        dict(),
        dict(error=ValueError, match=r"Expression .* should be an array of shape \(3,\)")
    ),
    (
        get_args('norm_x', source='h'),
        dict(),
        dict(error=ValueError, match="Only variables in input ports can be used as boundaries")
    ),
])
def test_TimeDerivative___init__(bogus, ctor_data, state, expected):
    args, kwargs = ctor_data
    error = expected.get('error', None)

    if error is None:
        der = TimeDerivative(bogus, *args, **kwargs)
        # set state of owner system
        for var, value in state.items():
            bogus[var] = value
        # check TimeDerivative object
        assert der.context is bogus
        assert der.port == expected['port']
        assert der.name == expected['name']
        # assert der.desc == expected['desc']
        assert der.source == pytest.approx(expected['source'], rel=1e-14)
        value = expected.get('value', None)
        if value is None:
            assert der.value is None 
            assert der.initial_value is None
        else:
            if isinstance(value, (list, np.ndarray)):
                assert isinstance(der.value, np.ndarray)
                assert isinstance(der.initial_value, np.ndarray)
                assert all(isinstance(comp, float) for comp in der.value)
            assert der.value == pytest.approx(value, rel=1e-14)
            assert der.initial_value == pytest.approx(value, rel=1e-14)
        assert der.mask == expected.get('mask', None)

    else:  # erroneous case
        with pytest.raises(error, match=expected.get('match', None)):
            TimeDerivative(bogus, *args, **kwargs)


def test_TimeDerivative___init__alt(bogus):
    assert len(bogus.inwards) == 6
    # Construction from evaluable strings
    # This situaton occurs when a time rate is transferred to its owner's parent,
    # owing to a 'pulling' statement. In this case, the evaluation context of the source
    # is preserved, and thus differs from the unknown's context. Also applies to 'initial_value'.
    b = BogusDynamicSystem('b')
    c = BogusDynamicSystem('c')
    der = TimeDerivative(bogus, 'v',
        source = EvalString('x', b),
        initial_value = EvalString('full(3, 0.1)', c),
    )
    assert der.context is not der.source_expr.eval_context
    assert der.context is not der.initial_value_expr.eval_context
    assert der.context is bogus
    assert der.source_expr.eval_context is b
    assert der.initial_value_expr.eval_context is c


def test_TimeDerivative_source(bogus):
    der = TimeDerivative(bogus, 'dh_dt', source=-1)
    assert der.source == -1
    der.source = 'h'
    assert bogus.h == 0.5
    assert der.source == bogus.h
    bogus.h = -2.4
    assert der.source == bogus.h


def test_TimeDerivative_value(bogus):
    der = TimeDerivative(bogus, 'dh_dt', source='h')
    with pytest.raises(RuntimeError, match="cannot be explicitly set"):
        der.value = 0.2


def test_TimeDerivative_reset(bogus):
    der = TimeDerivative(bogus, 'dh_dt', source='h')
    assert der.initial_value is None
    assert der.value == der.initial_value

    der = TimeDerivative(bogus, 'dh_dt', source='h', initial_value=2.7)
    assert der.initial_value == 2.7
    assert der.value == der.initial_value
    bogus.h = 0.2
    der.update(0.01)
    assert der.value == pytest.approx(-30)
    # reset time derivative
    der.reset()
    assert der.value == der.initial_value
    assert der.initial_value == 2.7
    bogus.h = 0.2
    der.update(0.01)
    assert der.value == pytest.approx(0)
    # reset with new initial value
    der.reset(5.25)
    assert der.value == der.initial_value
    assert der.initial_value == 5.25
    bogus.h = 0.25
    der.update(0.01)
    assert der.value == pytest.approx(5)


def test_TimeDerivative_update(bogus):
    dh_dt = TimeDerivative(bogus, 'dh_dt', source='h')
    dx_dt = TimeDerivative(bogus, 'v', source='x')

    dt = 1e-2
    assert bogus.h == 0.5
    bogus.h = 0.6
    dh_dt.update(dt)
    assert dh_dt.value == pytest.approx(10)
    dh_dt.update(dt)
    assert dh_dt.value == pytest.approx(0)
    bogus.h = 0.605
    dh_dt.update(dt)
    assert dh_dt.value == pytest.approx(0.5)


@pytest.mark.parametrize("name, options, expected", [
    ('dh_dt', dict(source=-1), dict(name='inwards.dh_dt')),
    ('v', dict(source='-0.1 * x'), dict(name='inwards.v')),
    ('dh_dt', dict(source='x[-1]'), dict(name='inwards.dh_dt')),
    ('h', dict(source=0.5), dict(name='inwards.h')),
])
def test_TimeDerivative_to_dict(bogus, name, options, expected):
    der = TimeDerivative(bogus, name, **options)
    der_dict = der.to_dict()

    assert der_dict["context"] == bogus.contextual_name
    assert der_dict["name"] == expected.get("name", name)
    assert der_dict["source"] == expected.get("source", str(options["source"]))
    assert der_dict["initial_value"] == expected.get("initial", str(der.initial_value_expr))
