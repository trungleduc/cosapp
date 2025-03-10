import pytest

import math
import copy
import numpy as np
import warnings

from cosapp.core.eval_str import EvalString
from cosapp.systems import System
from typing import Dict, Set


def test_EvalString__init__(eval_context):
    e = EvalString('3.14159', eval_context)
    assert e.eval_context is eval_context
    assert e.constant
    assert len(e.locals) == 0

    e = EvalString('22 + a', eval_context)
    assert e.eval_context is eval_context
    assert not e.constant
    assert set(e.locals) == {'a'}
    assert e.locals['a'] == eval_context.a


def test_EvalString_functions(ufunc_test_data):
    """Check that all tested symbols are actually implemented"""
    tested = set(ufunc_test_data)
    available = set(EvalString.available_symbols())
    assert tested <= available


@pytest.mark.parametrize("fname", filter(
    lambda funcname: not funcname.startswith("u"),
    EvalString.available_symbols()
))
def test_EvalString_exec_comp_value(fname, ufunc_test_data):
    class Test(System):
        def setup(self, **kwargs):
            for name, value in kwargs.items():
                self.add_inward(name, value=value)

    try:
        test_data = ufunc_test_data[fname]
    except KeyError:
        pytest.fail(reason=f"function {fname!r} is not tested")

    try:
        func = test_data['func']
    except KeyError:
        expected = test_data['check_val']
    else:
        expected = func(*test_data['args'].values())

    with warnings.catch_warnings():
        warnings.simplefilter('error')  # 'default', 'ignore', or 'error'
        s = Test('tmp', **test_data.get('args', {}))
        eval_str = EvalString(test_data['expr'], s)

    assert eval_str.eval() == pytest.approx(expected, rel=1e-14)


@pytest.mark.skip(reason='TODO')
def test_EvalString_residue_as_context(self):
    pass


@pytest.mark.parametrize("expression, expected", [
    (1, 1),
    (-1, -1),
    (0.2, 0.2),
    ("--1", 1),
    ("0.2", 0.2),
    ("0.2 / 0.5", pytest.approx(0.4, rel=1e-14)),
    ("cos(pi)", pytest.approx(-1, rel=1e-14)),
    ("sin(pi / 2)", pytest.approx(1, rel=1e-14)),
    ("log(exp(2))", pytest.approx(2, rel=1e-14)),
    ("log(e)", pytest.approx(1, rel=1e-14)),
    (None, None),
    ("None", None),
    (dict(), dict()),
    ("[]", list()),
    (True, True),
    (False, False),
    ("True", True),
    ("False", False),
    ("0.3 < 0.7", True),
    ("0.3 <= 0.3", True),
    ("0.3 < 0.7 < 0.6", False),
    ("0.3 + 0.7 == 1", True),
    ("[0., 0., 0.]", [0., 0., 0.]),
    ("[0.] * 3", [0., 0., 0.]),
    ("array([0, 0, 0], dtype=float)", [0., 0., 0.]),
    ("ones(4, dtype=int)", [1, 1, 1, 1]),
    ("ones(2, dtype=bool)", [True, True]),
    (repr(np.zeros(3)), [0., 0., 0.]),
    ("zeros(3)", [0., 0., 0.]),
    ("ones(3)", [1., 1., 1.]),
    (np.zeros(3), [0., 0., 0.]),
    (np.zeros((2, 5, 3)), np.zeros((2, 5, 3))),
    (np.ones(3), [1., 1., 1.]),
    ("-e", pytest.approx(-math.e, rel=1e-14)),
    ("9 + 3 + 6", 18),
    ("9 + 3 / 11", pytest.approx(9 + 3 / 11, rel=1e-14)),
    ("(9 + 3) / 11", pytest.approx(12 / 11, rel=1e-14)),
    ("(9 + 3)", 12),
    ("9 - 12 - 6", -9),
    ("9 - (12 - 6)", 9 - (12 - 6)),
    ("2 * 3.14159",  pytest.approx(2 * 3.14159, rel=1e-14)),
    ("3.1415926535 * 3.1415926535 / 10", pytest.approx(3.1415926535 * 3.1415926535 / 10, rel=1e-14)),
    ("pi * pi / 10", pytest.approx(math.pi * math.pi / 10, rel=1e-14)),
    ("pi**2 / 10", pytest.approx(math.pi**2 / 10, rel=1e-14)),
    ("pi**2", pytest.approx(math.pi ** 2, rel=1e-14)),
    ("round(pi**2)", pytest.approx(round(math.pi ** 2), rel=1e-14)),
    ("6.02E23 * 8.048", pytest.approx(6.02e23 * 8.048, rel=1e-14)),
    ("e / 3", pytest.approx(math.e / 3, rel=1e-14)),
    ("round(e)", pytest.approx(round(math.e), rel=1e-14)),
    ("round(-e)", pytest.approx(round(-math.e), rel=1e-14)),
    ("e**pi", pytest.approx(math.e ** math.pi, rel=1e-14)),
    ("2**3**2", pytest.approx(2 ** 3 ** 2, rel=1e-14)),
    ("2**3+2", pytest.approx(2 ** 3 + 2, rel=1e-14)),
    ("2**9", pytest.approx(2 ** 9, rel=1e-14)),
    ("{1, 2, 3, 3, 2, }", {1, 2, 3}),
    ("acos(-1)", pytest.approx(np.pi, rel=1e-14)),
    ("arccos(-1)", pytest.approx(np.pi, rel=1e-14)),
])
def test_EvalString_constant_expr(eval_context, expression, expected):
    """Test expressions expected to be interpreted as constant"""
    s = EvalString(expression, eval_context)
    assert s.eval_context is eval_context
    assert s.constant
    if isinstance(expected, (list, np.ndarray)):
        assert np.array_equal(s.eval(), expected)
    else:
        assert s.eval() == expected


@pytest.mark.parametrize("expression, expected", [
    ("-2 * g", dict(value=pytest.approx(-2 * 9.80665, rel=1e-14), constant=True)),
    ("1e-23 * NA + g", dict(value=pytest.approx(15.828790), constant=True)),
    ("g * x", dict(constant=False)),
])
def test_EvalString_with_constants(eval_context, expression, expected):
    """Test expressions with system constants, expected to be interpreted as constant"""
    s = EvalString(expression, eval_context)
    assert s.eval_context is eval_context
    assert s.constant == expected.get('constant', False)
    if 'value' in expected:
        assert s.eval() == expected['value']


def test_EvalString_subsystem_constant():
    """Test expressions involving sub-system constants"""
    class SystemWithConstants(System):
        def setup(self, constants: dict={}):
            for name, value in constants.items():
                self.add_property(name, value)

    top = System('top')
    mid = top.add_child(SystemWithConstants('mid', constants={'g': 9.81}))
    sub = mid.add_child(SystemWithConstants('sub', constants={'c': 0.10}))

    s = EvalString('2 * c', sub)
    assert s.variables == set()
    assert s.all_variables == {'c'}
    assert s.constant
    assert s.eval() == pytest.approx(0.2, rel=1e-14)

    s = EvalString('2 * sub.c', mid)
    assert s.variables == set()
    assert s.all_variables == {'sub.c'}
    assert s.constant
    assert s.eval() == pytest.approx(0.2, rel=1e-14)

    s = EvalString('2 * mid.sub.c', top)
    assert s.variables == set()
    assert s.all_variables == {'mid.sub.c'}
    assert s.constant
    assert s.eval() == pytest.approx(0.2, rel=1e-14)

    s = EvalString('mid.g * mid.sub.c', top)
    assert s.variables == set()
    assert s.all_variables == {'mid.g', 'mid.sub.c'}
    assert s.constant
    assert s.eval() == pytest.approx(0.981, rel=1e-14)


@pytest.mark.parametrize("expression, expected", [
    ("norm(x, inf)", 3.14),
    ("0.2 * a / 0.5", pytest.approx(0.8, rel=1e-14)),
    ("log10(x[0])", pytest.approx(-1, rel=1e-14)),
    ("a + b - x[1]", pytest.approx(2.7, rel=1e-14)),
    ("a - a + 1", 1),
    ("9 + sub.in_.q / 10", pytest.approx(9.5, rel=1e-14)),
    ("9 + sin(sub.in_.q / 11)", pytest.approx(9 + math.sin(5 / 11), rel=1e-14)),
    ("out.q - sub.in_.q", pytest.approx(-4.5, rel=1e-14)),
    ("out. q - sub.in_ .  q", pytest.approx(-4.5, rel=1e-14)),
    ("concatenate((x, [-a, sub.in_.q]))", pytest.approx([0.1, -0.2, -3.14, -2, 5], rel=1e-14)),
    ("len(out)", 1),   # not constant, by convention
])
def test_EvalString_nonconstant_expr(eval_context, expression, expected):
    """Test expressions expected to be interpreted as non-constant"""
    s = EvalString(expression, eval_context)
    assert s.eval_context is eval_context
    assert not s.constant
    assert s.eval() == expected


@pytest.mark.parametrize("expression, exception", [
    ("a = 2.0", SyntaxError),
    ("cos(pi", SyntaxError),
    ("a + .  5", SyntaxError),
    ("1 + _", NameError),
    ("1 / 0", ZeroDivisionError),
    ("foo.bar(1)", NameError),
    ("a + ['v', True]", TypeError),
    ("out.monkey", AttributeError),
    ("sub.out.q", AttributeError),
    ("sub.in_.", SyntaxError),
    ("sub.in_..", SyntaxError),
    ("", ValueError),
    ("  ", ValueError),
])
def test_EvalString_erroneous_expr(eval_context, expression, exception):
    """Test erroneous expressions expected to raise an exception at instantiation"""
    with pytest.raises(exception):
        eval_string = EvalString(expression, eval_context)
        eval_string.eval()


@pytest.mark.parametrize("expression, expected", [
    ("1", "1"),
    ("1.2", "1.2"),
    (8, "8"),
    (1.3, "1.3"),
    ([0.1, 0.2, 0.3], "[0.1, 0.2, 0.3]"),
    (np.r_[0.1, 0.2, 0.3], "array([0.1, 0.2, 0.3])"),
    ("g * cos(pi * x) + sub.z", "g * cos(pi * x) + sub.z"),
    ("out. q - sub.in_   .  q  -  sub   .z * .5", "out.q - sub.in_.q  -  sub.z * .5"),
])
def test_EvalString_str_repr(eval_context, expression, expected):
    s = EvalString(expression, eval_context)
    assert str(s) == expected
    assert repr(s) == repr(expected)  # same repr as built-in Python strings


@pytest.mark.parametrize("expression, patterns, results", [
    ("norm(x, inf)", ["norm", "inf", "x", "foo"], [True] * 3 + [False]),
    ("0.2 * a / 0.5", ["0.", "a", "x"], [True, True, False]),
    ("x", ["x", "Y", "foo"], [True, False, False]),
    ("[a, 2]", ["a", "2", "1", "foo"], [True, True, False, False]),
    ("[1, 2]", ["a", "2", "1", "1, 2"], [False, True, True, True]),
])
def test_EvalString_contains(eval_context, expression, patterns, results):
    """Test expressions expected to be interpreted as non-constant"""
    s = EvalString(expression, eval_context)
    for pattern, expected in zip(patterns, results):
        assert (pattern in s) == expected


@pytest.mark.parametrize("var", [
    "a", "b", "out.q", "sub.z", "sub.in_.q", ])
def test_EvalString_multiple_eval(eval_context, var):
    s = EvalString(var, eval_context)
    assert s.eval() == eval_context[var]
    for eval_context[var] in [0, 0.1, -22]:
        assert s.eval() == eval_context[var]


@pytest.mark.parametrize("expression, expected", [
    ("norm(x, inf)", "norm(x, inf)"),
    ("123.456", "123.456"),
    (123.456, "123.456"),
    ([1, 2, 3, ], "[1, 2, 3]"),
    ((1, 2, 3, ), "(1, 2, 3)"),
    ({1, 2, 3}, "{1, 2, 3}"),
    ("{1, 2, 3, 3, 2, }", "{1, 2, 3, 3, 2, }"),
    (np.r_[0.1, 0.2, 0.3], "array([0.1, 0.2, 0.3])"),
    (np.array([0, 0, 0], dtype=float), "array([0., 0., 0.])"),
    (np.ones(2, dtype=int), "array([1, 1])"),
    (None, "None"),
    ("out. q - sub.in_   .  q  -  sub   .z * .5", "out.q - sub.in_.q  -  sub.z * .5"),
])
def test_EvalString_string(eval_context, expression, expected):
    assert EvalString.string(expression) == expected
    s = EvalString(expression, eval_context)
    assert EvalString.string(s) == expected


@pytest.mark.parametrize("expression, expected", [
    ("norm(x, inf)", {'x'}),
    ("g * cos(pi * x)", {'x', 'g'}),
    ("g * cos(pi * x) + sub.z", {'x', 'g', 'sub_z'}),
    ("g * cos(pi * x) + sub.z + out.q", {'x', 'g', 'sub_z', 'out_q'}),
    ("g * cos(pi * x) / out.q + sub.z * sub.in_.q + out.q", {'x', 'g', 'out_q', 'sub_z', 'sub_in__q'}),
])
def test_EvalString_locals(eval_context, expression, expected):
    e = EvalString(expression, eval_context)
    assert set(e.locals) == expected


@pytest.mark.parametrize("expression, expected", [
    ("norm(x, inf)", dict(attributes={'x'})),
    ("g * cos(pi * x)", dict(attributes={'x'}, constants={'g'})),
    ("g * cos(pi * x) + sub.z", dict(attributes={'x', 'sub.z'}, constants={'g'})),
    (
        "g * cos(pi * x) / out.q + sub.z * sub.in_.q + out.q",
        dict(attributes={'x', 'sub.z', 'sub.in_.q', 'out.q'}, constants={'g'})
    ),
    (
        "(g * cos(pi * x), out.q, sub.z, sub.in_.q, 2 * out.q)",
        dict(attributes={'x', 'sub.z', 'sub.in_.q', 'out.q'}, constants={'g'}),
    ),
    ("out.q + sub.z * sub.in_.q + B52.in_.q", dict(attributes={'sub.z', 'sub.in_.q', 'out.q', 'B52.in_.q'})),
    ("out.q + exp(sub.z * sub.in_.q) + cos(B52.in_.q)", dict(attributes={'sub.z', 'sub.in_.q', 'out.q', 'B52.in_.q'})),
    ("out.q * exp(-y[1] / 2)", dict(attributes={'y', 'out.q'})),
    ("out. q + sub  .z * sub . in_.q + out  . q", dict(attributes={'sub.z', 'sub.in_.q', 'out.q'})),
    ("len(out)", dict(attributes={"out"})),
    ("asarray([out.q, a]).sum()", dict(attributes={'out.q', 'a'})),
    ("sub.add_constant(a)", dict(attributes={'sub', 'a'})),
    ("distance(x)", dict(attributes={'distance', 'x'})),
])
def test_EvalString_variables(eval_context, expression, expected: Dict[str, Set[str]]):
    """Test attributes collected during ast visit."""
    expected.setdefault('constants', set())
    all_required = expected['attributes'].union(expected['constants'])
    
    e = EvalString(expression, eval_context)
    assert e.variables == expected['attributes']
    assert e.all_variables == all_required


def test_EvalString_enum(eval_context):
    """Test the evaluation of enum members."""
    from enum import Enum

    class Mode(Enum):
        A = 0
        B = 1

    s = EvalString(Mode.A, eval_context)
    assert s.constant
    assert s.eval() is Mode.A


@pytest.mark.parametrize(
    "expression", [
        "norm(x, inf)",
        "g * cos(pi * x)",
        "g * cos(pi * x) + sub.z",
        "g * cos(pi * x) / out.q + sub.z * sub.in_.q + out.q",
    ]
)
def test_EvalString_equality(eval_context, expression):
    """Test attributes collected during ast visit."""
    e1 = EvalString(expression, eval_context)
    e2 = EvalString(expression, eval_context)
    assert e1 == e2
