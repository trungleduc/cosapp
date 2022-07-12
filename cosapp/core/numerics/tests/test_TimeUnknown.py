import pytest

import numpy as np
from numbers import Number

from cosapp.ports import Port
from cosapp.systems import System
from cosapp.core.numerics.boundary import TimeUnknown
from cosapp.core.eval_str import EvalString


class BogusPort(Port):
    def setup(self):
        self.add_variable('var', 0.0)

class DynamicSystem(System):
    def setup(self, **kwargs):
        self.add_input(BogusPort, 'in_')
        self.add_inward('x', 0.5)
        self.add_inward('y', 2.4)
        self.add_inward('acc', np.r_[0, 0, -9.81])
        self.add_outward('q')

        self.add_transient('h', der='y / x')
        self.add_transient('vel', der='acc')


@pytest.mark.parametrize("ctor_data, state, expected", [
    (
        dict(name='h', der=-1), dict(),
        dict(name='h', dtype=float, d_dt=-1),
    ),
    (
        dict(name='inwards.h', der=-1), dict(),
        dict(name='h', dtype=float, d_dt=-1),
    ),
    (
        dict(name='in_.var', der='h / 2'), dict(h=3),
        dict(port='in_', name='in_.var', dtype=float, d_dt=1.5),
    ),
    (
        dict(name='vel', der='-acc', max_time_step='0.1 * norm(acc, inf)'),
        dict(acc=np.r_[1, 2, -3]),
        dict(
            name = 'vel',
            d_dt = pytest.approx([-1, -2, 3], rel=1e-15),
            dtype = np.ndarray,
            max_time_step = pytest.approx(0.3, rel=1e-15),
        ),
    ),
    (
        dict(name='h', der='-acc[0]', max_time_step='0.1 * norm(acc, inf)'),
        dict(acc=np.r_[1, 2, -3]),
        dict(
            name = 'h',
            d_dt = pytest.approx(-1, rel=1e-15),
            max_time_step = pytest.approx(0.3, rel=1e-15),
        ),
    ),
    (
        dict(name='h', der='-acc[0]', max_time_step='0.1 * norm(acc, inf)', max_abs_step='1e-3'),
        dict(acc=np.r_[1, 2, -3]),
        dict(
            name = 'h',
            d_dt = pytest.approx(-1, rel=1e-15),
            max_time_step = pytest.approx(1e-3, rel=1e-15),  # limited by `max_abs_step`
            max_abs_step = 1e-3,
        ),
    ),
    (
        dict(name='h', der=1.5, max_time_step=-1e-2), dict(),
        dict(error=ValueError, match="max_time_step must be strictly positive"),
    ),
    (
        dict(name='h', der=1.5, max_abs_step=-0.5), dict(),
        dict(error=ValueError, match="max_abs_step must be strictly positive"),
    ),
    (
        dict(name='h', der=True), dict(),
        dict(error=TypeError, match="Derivative expressions may only be numbers or array-like collections"),
    ),
    (
        dict(name='h', der=dict(cool=False)), dict(),
        dict(error=TypeError, match="Derivative expressions may only be numbers or array-like collections"),
    ),
])
def test_TimeUnknown___init__(ctor_data, state, expected):
    a = DynamicSystem('a')
    for var, value in state.items():
        a[var] = value
    error = expected.get('error', None)

    if error is None:
        u = TimeUnknown(a, **ctor_data)
        assert u.context is a
        assert u.port is a[expected.get('port', 'inwards')]
        assert u.name == expected['name']
        assert u.d_dt == expected['d_dt']
        assert isinstance(u.value, expected.get('dtype', Number))
        if isinstance(u.value, Number):
            assert u.mask is None
        else:
            assert np.array_equal(u.mask, np.ones_like(u.value, dtype=bool))
        assert u.max_time_step == expected.get('max_time_step', np.inf)
        assert u.max_abs_step == expected.get('max_abs_step', np.inf)

    else:
        with pytest.raises(error, match=expected.get('match', None)):
            TimeUnknown(a, **ctor_data)


def test_TimeUnknown___init__eval_str():
    a = DynamicSystem('a')
    # Construction from evaluable strings
    # This situaton occurs when a time unknown is transferred to its owner's parent,
    # owing to a 'pulling' statement. In this case, the evaluation context of the derivative
    # is preserved, and thus differs from the unknown's context. Also applies to 'max_time_step'.
    b = DynamicSystem('b')
    c = DynamicSystem('c')
    u = TimeUnknown(a, 'h',
        der = EvalString('y', b),
        max_time_step = EvalString('0.1 * max(x, y)', c)
    )
    assert u.context is not u.der.eval_context
    assert u.context is not u.max_time_step_expr.eval_context
    assert u.context is a
    assert u.der.eval_context is b
    assert u.max_time_step_expr.eval_context is c


@pytest.mark.parametrize("attr", ["max_abs_step", "max_time_step"])
@pytest.mark.parametrize("value, expected", [
    (0,    dict(ok=False)),
    (-1,   dict(ok=False)),
    (1,    dict(ok=True)),
    (5e-3, dict(ok=True)),
    ("0",    dict(ok=False)),
    ('-1',   dict(ok=False)),
    ('1',    dict(ok=True, value=1)),
    ("5e-3", dict(ok=True, value=5e-3)),
    ("cos(pi)", dict(ok=False)),
    ("-cos(pi)", dict(ok=True, value=1)),
    # Negative, but context-dependent expressions should not raise any exception:
    ("-0.1 * norm(acc)", dict(ok=True, value=-0.981)),
    ("-abs(y)", dict(ok=True, value=-2.4)),
])
def test_TimeUnknown_max_step(attr, value, expected):
    """
    Test getter and setter for attributes `max_time_step` and `max_abs_step`.
    Note that this parametrization assumes that one of the two is set individually (`attr`),
    while the other is unspecified (that is infinity).
    """
    a = DynamicSystem('a')
    u = TimeUnknown(a, 'h', der='-h')

    if expected['ok']:
        assert u.max_time_step == np.inf
        assert u.max_abs_step == np.inf
        setattr(u, attr, value)
        expected_value = expected.get('value', value)
        assert getattr(u, attr) == pytest.approx(expected_value, rel=1e-14)
    else:
        with pytest.raises(ValueError, match=f"{attr} must be strictly positive"):
            setattr(u, attr, value)


def test_TimeUnknown_max_time_step():
    """
    Test the coupling between attributes `max_time_step` and `max_abs_step`.
    """
    a = DynamicSystem('a')
    a.h = 1.0
    u = TimeUnknown(a, 'h', der='-h')
    assert u.max_time_step == np.inf
    assert u.max_abs_step == np.inf
    u.max_time_step = 0.1
    assert u.max_time_step == 0.1
    assert u.max_abs_step == np.inf
    u.max_abs_step = 0.5
    assert u.max_time_step == 0.1
    assert u.max_abs_step == 0.5
    a.h = 10
    # assert u.d_dt == -10
    assert u.max_time_step == 0.05
    assert u.max_abs_step == 0.5


def test_TimeUnknown_max_time_step_expr():
    a = DynamicSystem('a')
    u = TimeUnknown(a, 'h', der=1.5)
    assert u.max_time_step == np.inf

    # Tests with evaluable expressions
    u = TimeUnknown(a, 'vel', der='-acc')
    u.max_time_step = '0.1 * norm(acc, inf)'
    a.acc = np.r_[0.1, 2, -3]
    assert u.max_time_step == pytest.approx(0.3)
    a.acc[2] = 0
    assert u.max_time_step == pytest.approx(0.2)
    u.max_time_step = '0.05 * abs(acc[0])'
    assert u.max_time_step == pytest.approx(0.005)
    u.max_time_step = 'max(0.01, 0.05 * abs(acc[-1]))'
    a.acc[2] = 1e-5
    assert u.max_time_step == pytest.approx(0.01)
    a.acc[2] = 0.5
    assert u.max_time_step == pytest.approx(0.025)


def test_TimeUnknown_der_type():
    sa = DynamicSystem('a')
    sa.x = -0.5
    sa.y = 6.28

    estring, value, dtype = TimeUnknown.der_type(-1, sa)
    assert isinstance(estring, EvalString)
    assert dtype is Number, "type of transient should be Number, even though its derivative is an integer"
    assert isinstance(value, int)
    assert value == -1

    estring, value, dtype = TimeUnknown.der_type('x * y', sa)
    assert dtype is Number
    assert value == estring.eval()
    assert value == -3.14

    estring, value, dtype = TimeUnknown.der_type('-acc', sa)
    assert dtype is np.ndarray
    assert value.shape == sa.acc.shape
    assert value == pytest.approx(-sa.acc)

    with pytest.raises(SyntaxError):
        TimeUnknown.der_type('cos(', sa)


def test_TimeUnknown_d_dt():
    a = DynamicSystem('a')
    a.x = -0.1
    a.acc = np.array([1, 2, 3], dtype=int)

    u = TimeUnknown(a, 'vel', der='x * acc')

    assert u.d_dt == pytest.approx([-0.1, -0.2, -0.3])
    assert isinstance(u.value, np.ndarray)
    assert len(u.value) == 3

    u.value = 1.23
    assert u.value == pytest.approx(np.full(3, 1.23))

    with pytest.raises(ValueError):
        u.value = [0.2, 0.1]
    
    u.d_dt = [0, 0.5, 1]
    assert u.d_dt == pytest.approx([0, 0.5, 1])

    u.d_dt = "0.01 * acc"
    assert u.d_dt == pytest.approx([0.01, 0.02, 0.03])

    with pytest.raises(TypeError, match="incompatible with declared type"):
        u.d_dt = 0.5

    with pytest.raises(TypeError, match="incompatible with declared type"):
        u.d_dt = "0.1 * norm(acc, inf)"

    with pytest.raises(ValueError, match="shape"):
        u.d_dt = [0.5, 0.1]
    
    with pytest.raises(ValueError, match="shape"):
        u.d_dt = "acc[1:]"

    with pytest.raises(TypeError):
        u.d_dt = dict(cool=False)


def test_TimeUnknown_copy():
    a = DynamicSystem('a')
    original = TimeUnknown(a, 'vel', der='x * acc')
    copy = original.copy()
    assert copy is not original
    assert copy.context is original.context
    for attr in ['name', 'der', 'max_time_step', 'max_time_step_expr']:
        assert getattr(copy, attr) == getattr(original, attr), f"(attribute '{attr}')"


@pytest.mark.parametrize("name, options, expected", [
    ('h', dict(der=-1), dict(name='h')),
    ('vel', dict(der='-acc', max_time_step='0.1 * norm(acc, inf)'), dict(name='vel', max_time_step='0.1 * norm(acc, inf)')),
    ('h', dict(der='acc[-1]'), dict(name='h')),
    ('in_.var', dict(der=0.5), dict()),
])
def test_TimeUnknown_to_dict(name, options, expected):
    a = DynamicSystem('a')
    
    u = TimeUnknown(a, name, **options)
    u_dict = u.to_dict()
    assert u_dict["context"] == a.contextual_name
    assert u_dict["name"] == expected.get("name", name)
    assert u_dict["der"] == expected.get("der", str(options["der"]))
    assert u_dict["max_time_step"] == expected.get("max_time_step", 'inf')


@pytest.mark.parametrize("name, options, expected", [
    ('h', dict(der=-1), False),
    ('vel', dict(der='acc', max_time_step='inf'), False),
    ('vel', dict(der='acc', max_abs_step=0.5), True),
    ('vel', dict(der='acc', max_abs_step='0.5 * norm(acc)'), True),
    ('vel', dict(der='acc', max_time_step=np.inf), False),
    ('vel', dict(der='acc', max_time_step=0.1), True),
    ('vel', dict(der='acc', max_time_step='0.1 + cos(pi / 6)'), True),
    ('vel', dict(der='acc', max_time_step='0.1 * norm(acc, inf)'), True),
    ('h', dict(der='acc[-1]'), False),
    ('in_.var', dict(der=0.5), False),
])
def test_TimeUnknown_constrained(name, options, expected):
    a = DynamicSystem('a')
    u = TimeUnknown(a, name, **options)

    assert u.constrained == expected
