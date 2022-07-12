import pytest

import numpy as np
import logging, re
from collections import OrderedDict
from numbers import Number
from typing import Tuple, Dict, Any

from cosapp.base import System, Port
from cosapp.core.numerics.basics import MathematicalProblem
from cosapp.core.numerics.boundary import Unknown, TimeUnknown
from cosapp.core.numerics.residues import DeferredResidue, Residue
from cosapp.utils.testing import get_args, no_exception, ArgsKwargs


class BogusPort(Port):
    def setup(self):
        self.add_variable('m')

class SystemA(System):
    def setup(self):
        self.add_property('n', 12)

        self.add_input(BogusPort, 'in_')
        self.add_output(BogusPort, 'out')
        self.add_inward('a', 1.)
        self.add_inward('b', [1., 2.])
        self.add_inward('c', np.arange(4, dtype=float))
        self.add_inward('d', -2.7)

        self.add_outward('g', 3.5)
        self.add_outward('h', [1., 2.])
        self.add_outward('i', 5.)

class SystemB(System):
    def setup(self):
        self.add_inward('x', 1.)
        self.add_inward('y', [1., 2.])

        self.add_outward('u', 3.5)
        self.add_outward('v', [1., 2.])

class DynamicSystemC(System):
    def setup(self):
        self.add_inward('q', 1)
        self.add_inward('v', np.zeros(3))
        self.add_transient('A', 'q')
        self.add_transient('x', 'v')


@pytest.fixture(scope='function')
def test_objects() -> Tuple[System, MathematicalProblem]:
    system = SystemA('system_a')
    return system, MathematicalProblem('math_pb', system)


def test_MathematicalProblem__init__():
    # Empty case
    m = MathematicalProblem('test', None)
    assert m.name == 'test'
    assert m.context is None
    assert len(m.unknowns) == 0
    assert len(m.residues) == 0
    assert len(m.transients) == 0
    assert len(m.deferred_residues) == 0


def test_MathematicalProblem_name(test_objects: Tuple[System, MathematicalProblem]):
    s, m = test_objects
    with pytest.raises(AttributeError):
        setattr(m, 'name', 'banana')


def test_MathematicalProblem_context():
    sa, sb = SystemA('a'), SystemB('b')

    m = MathematicalProblem('sa', None)
    assert m.context is None
    m.context = sa
    assert m.context is sa
    with pytest.raises(ValueError, match="Context is already set to .*"):
        m.context = sb
    assert m.context is sa

    with no_exception():
        m.context = m.context

    m = MathematicalProblem('test', sa)
    assert m.context is sa
    with pytest.raises(ValueError, match="Context is already set to .*"):
        m.context = sb
    assert m.context is sa


def test_MathematicalProblem_noSetters(test_objects: Tuple[System, MathematicalProblem]):
    s, m = test_objects

    with pytest.raises(AttributeError):
        m.unknowns = [Unknown(s, "a"), ]
    
    with pytest.raises(AttributeError):
        m.residues = OrderedDict(a=Residue(s, "a == 0"))
    
    ds = DynamicSystemC('ds')
    with pytest.raises(AttributeError):
        m.transients = [TimeUnknown(ds, "A", "2.5 * q"), ]


def test_MathematicalProblem_residues_vector(test_objects: Tuple[System, MathematicalProblem]):
    s, m = test_objects
    m.add_equation([
        dict(equation="g == 0"),
        dict(equation="h == array([22., 4.2])", name="h equation", reference=24.)
    ])
    m.add_unknown([
        dict(name="a", max_rel_step=1e-5),
        dict(name="c", max_abs_step=1e-2)
    ])

    assert len(m.residues_vector) == 3
    assert m.residues_vector == pytest.approx(
        np.concatenate(([m.residues['g == 0'].value], m.residues['h equation'].value)))


@pytest.mark.parametrize("case, expected", [
    (
        dict(unknowns='c', equations='g == 0'),
        dict(n_unknowns=4, n_equations=1),
    ),
    (
        dict(unknowns='a', equations='h == zeros(2)'),
        dict(n_unknowns=1, n_equations=2),
    ),
    (
        dict(unknowns=['a', 'c']), dict(n_unknowns=5),
    ),
    (
        dict(unknowns=['a', 'c[::2]']), dict(n_unknowns=3),
    ),
    (
        dict(unknowns=['a', 'c'], equations=['g == 0', 'h == zeros(2)']),
        dict(n_unknowns=5, n_equations=3),
    ),
    (
        dict(equations='g == 0', targets='out.m'),
        dict(n_equations=2),
    ),
    (
        dict(equations='g == 0', targets=['out.m', 'h']),
        dict(n_equations=4),
    ),
    (
        dict(targets=['out.m', 'h']),
        dict(n_equations=3),
    ),
])
def test_MathematicalProblem_shape(test_objects: Tuple[System, MathematicalProblem], case, expected: Dict[str, Any]):
    system, problem = test_objects
    problem.add_equation(case.get('equations', []))
    problem.add_unknown(case.get('unknowns', []))
    problem.add_target(case.get('targets', []))
    # Retrieve expected values
    n_unknowns = expected.get('n_unknowns', 0)
    n_equations = expected.get('n_equations', 0)

    assert problem.n_unknowns == n_unknowns
    assert problem.n_equations == n_equations
    assert problem.shape == (n_unknowns, n_equations)


def test_MathematicalProblem_add_methods():
    # Single element case
    s = SystemA('a')
    m = MathematicalProblem('test', s)
    m.add_equation("g == 0")
    m.add_unknown("a")
    assert m.context is s
    assert list(m.unknowns) == ['a']
    assert list(m.residues) == ['g == 0']
    assert m.shape == (1, 1)

    unknown = m.unknowns['a']
    assert isinstance(unknown, Unknown)
    assert unknown.context is s

    residue = m.residues['g == 0']
    assert isinstance(residue, Residue)
    assert residue.context is s

    # Multiple case
    s = SystemA('a')
    m = MathematicalProblem('tests', s)
    m.add_equation([
        dict(equation="g == 0"),
        dict(equation="h == array([22., 4.2])", name="h equation", reference=24.)
    ])
    m.add_unknown([
        dict(name="a", max_rel_step=1e-5),
        dict(name="c", max_abs_step=1e-2)
    ])

    assert m.context is s
    assert list(m.unknowns) == ['a', 'c']
    assert list(m.residues) == ['g == 0', 'h equation']
    assert list(m.transients) == []

    unknown = m.unknowns['c']
    assert isinstance(unknown, Unknown)
    assert unknown.context is s
    assert unknown.max_abs_step == 1e-2

    residue = m.residues['h equation']
    assert isinstance(residue, Residue)
    assert residue.context is s
    assert residue.name == 'h equation'


@pytest.mark.parametrize("args_kwargs, expected", [
    (
        get_args('g == i'),
        {
            'g == i': dict(),
        }
    ),
    (
        get_args('g == i', 'gi_balance', reference=0.1),
        {
            'gi_balance': dict(reference=0.1),
        }
    ),
    (
        get_args(['g == i', 'h == 0']),
        {
            'g == i': dict(),
            'h == 0': dict(),
        }
    ),
    (
        get_args(['g == i', dict(equation='h == 0', name='h_eqn')]),
        {
            'g == i': dict(),
            'h_eqn': dict(),
        }
    ),
    (
        get_args(['g == i', dict(equation='h == 0', name='h_eqn', reference=10)]),
        {
            'g == i': dict(),
            'h_eqn': dict(reference=10),
        }
    ),
    (
        get_args(['g == i', dict(equation='h == 0', name='h_eqn'), 'sum(c) == 1']),
        {
            'g == i': dict(),
            'h_eqn': dict(),
            'sum(c) == 1': dict(),
        }
    ),
])
def test_MathematicalProblem_add_equation(
    test_objects: Tuple[System, MathematicalProblem],
    args_kwargs: ArgsKwargs,
    expected: Dict[str, dict],
):
    s, m = test_objects
    args, kwargs = args_kwargs
    m.add_equation(*args, **kwargs)

    residues = m.residues
    assert set(residues.keys()) == set(expected.keys())

    for name, properties in expected.items():
        residue = residues[name]
        message = f"residue {name!r}"
        assert isinstance(residue, Residue), message
        assert residue.context is s, message
        assert residue.name == name, message
        assert residue.reference == properties.get('reference', 1), message


@pytest.mark.parametrize("args_kwargs, expected", [
    (
        get_args('g'),
        dict(equations=['g == 3.5']),
    ),
    (
        get_args('g', reference=10),
        dict(equations=['g == 3.5'], reference=10),
    ),
    (
        get_args('g - 1'),
        dict(equations=['g - 1 == 2.5']),
    ),
    (
        get_args('n * g'),
        dict(equations=['n * g == 42.0']),
    ),
    (
        get_args('cos(pi * g / 3.5)'),
        dict(equations=['cos(pi * g / 3.5) == -1.0']),
    ),
    (
        get_args('a'),
        dict(equations=['a == 1.0']),
    ),
    (
        get_args('a * g'),
        dict(error=NotImplementedError, match="Targets are only supported for single variables", equations=['a * g == 3.5']),
    ),
])
def test_MathematicalProblem_add_target(
    test_objects: Tuple[System, MathematicalProblem],
    args_kwargs: ArgsKwargs,
    expected: Dict[str, Any],
):
    context, problem = test_objects
    assert len(problem.deferred_residues) == 0
    args, kwargs = args_kwargs
    error = expected.get('error', None)

    if error is None:
        problem.add_target(*args, **kwargs)
        assert len(problem.deferred_residues) == 1

        equations = problem.get_target_equations()
        residues = problem.get_target_residues()
        assert equations == expected['equations']
        assert len(residues) == 1

        for name, residue in residues.items():
            message = f"residue {name!r}"
            assert isinstance(residue, Residue), message
            assert residue.context is context, message
            assert residue.reference == kwargs.get('reference', 1), message
            assert residue.value == pytest.approx(0, abs=1e-15), message

    else:
        with pytest.raises(error, match=expected.get('match', None)):
            problem.add_target(*args, **kwargs)


@pytest.mark.parametrize("args_kwargs, expected", [
    (
        get_args('a'),
        {
            'a': dict(),
        }
    ),
    (
        get_args('inwards.a'),
        {
            'a': dict(),
        }
    ),
    (
        get_args(['a', 'd']),
        {
            'a': dict(),
            'd': dict(),
        }
    ),
    (
        get_args(["in_.m", dict(name="d", max_rel_step=0.1)]),
        {
            'in_.m': dict(),
            'd': dict(max_rel_step=0.1),
        }
    ),
    # Vector unknowns:
    (
        get_args('c'),
        {
            'unknown_names': tuple(f"c[{i}]" for i in range(4)),
            'c': dict(mask=[True, True, True, True]),
        }
    ),
    (
        get_args('c[:]'),
        {
            'unknown_names': tuple(f"c[{i}]" for i in range(4)),
            'c[:]': dict(mask=[True, True, True, True]),
        }
    ),
    (
        get_args('c[:-1]'),
        {
            'unknown_names': tuple(f"c[{i}]" for i in range(3)),
            'c[:-1]': dict(mask=[True, True, True, False]),
        }
    ),
    (
        get_args('c[::2]'),
        {
            'unknown_names': ('c[0]', 'c[2]',),
            'c[::2]': dict(mask=[True, False, True, False]),
        }
    ),
    (
        get_args('c[1]'),
        {
            'unknown_names': ('c[1]',),
            'c[1]': dict(mask=[False, True, False, False]),
        }
    ),
    (
        get_args('c', mask=[True, False, True, True]),
        {
            'unknown_names': ('c[0]', 'c[2]', 'c[3]',),
            'c': dict(mask=[True, False, True, True]),
        }
    ),
    (
        get_args([
            dict(name="c", mask=[True, False, True, True]),
            dict(name='in_.m', max_abs_step=0.5, lower_bound=-2),
            'a',
        ]),
        {
            'unknown_names': ('c[0]', 'c[2]', 'c[3]', 'in_.m', 'a'),
            'c': dict(mask=[True, False, True, True]),
            'a': dict(),
            'in_.m': dict(max_abs_step=0.5, lower_bound=-2),
        }
    ),
])
def test_MathematicalProblem_add_unknown(
    test_objects: Tuple[System, MathematicalProblem],
    args_kwargs: ArgsKwargs,
    expected: Dict[str, Any],
):
    context, problem = test_objects
    args, kwargs = args_kwargs
    problem.add_unknown(*args, **kwargs)

    if 'unknown_names' in expected:
        assert problem.unknowns_names == expected.pop('unknown_names')

    unknowns = problem.unknowns
    assert set(unknowns.keys()) == set(expected.keys())

    for name, properties in expected.items():
        unknown = unknowns[name]
        message = f"unknown {name!r}"
        assert isinstance(unknown, Unknown), message
        assert unknown.context is context, message
        assert unknown.name == name, message
        assert unknown.max_abs_step == properties.get('max_abs_step', np.inf), message
        assert unknown.max_rel_step == properties.get('max_rel_step', np.inf), message
        assert unknown.lower_bound == properties.get('lower_bound', -np.inf), message
        assert unknown.upper_bound == properties.get('upper_bound', np.inf), message
        mask = properties.get('mask', None)
        if mask is None:
            assert unknown.mask is None, message
        else:
            assert tuple(unknown.mask) == tuple(mask), message


def test_MathematicalProblem_add_unknown_repeated(test_objects: Tuple[System, MathematicalProblem], caplog):
    """Check that defining the same unknown several times does not raise any exception"""
    m = test_objects[1]
    m.add_unknown('a', max_abs_step=1)
    assert m.unknowns['a'].max_rel_step == np.inf
    assert m.unknowns['a'].max_abs_step == 1

    caplog.clear()
    with caplog.at_level(logging.INFO):
        m.add_unknown('a', max_rel_step=0.1)
        m.add_unknown('inwards.a', max_abs_step=2)
    
    assert len(caplog.records) == 2
    pattern = "Variable '{}' is already declared as unknown"
    assert re.match(pattern.format('a'), caplog.records[0].message)
    assert re.match(pattern.format('inwards.a'), caplog.records[1].message)
    # Check that unknown properties have not changed
    assert m.unknowns['a'].max_rel_step == np.inf
    assert m.unknowns['a'].max_abs_step == 1


def test_MathematicalProblem_add_transient():
    s = DynamicSystemC('s')
    m = MathematicalProblem('test', s)
    m.add_transient('A', der='q')
    assert list(m.transients) == ['A']
    A = m.transients['A']
    assert isinstance(A.value, Number)
    assert A.context is s
    assert A.name == 'A'
    assert A.d_dt == 1

    m.add_transient('x', der='v / q**2')
    assert list(m.transients) == ['A', 'x']
    x = m.transients['x']
    assert isinstance(x.value, np.ndarray)
    s.v = np.r_[1, 2, 3]
    s.q = 2.0
    assert x.context is s
    assert x.name == 'x'
    assert x.d_dt == pytest.approx(np.r_[0.25, 0.5, 0.75], rel=1e-15)


def test_MathematicalProblem_extend():
    def local_test_objects():
        r, s = SystemA('r'), SystemB('s')
        mr = MathematicalProblem('test', r)
        ms = MathematicalProblem('test', s)
        # Define mathematical problem 'mr'
        mr.add_equation([
            dict(equation="g == 0"),
            dict(equation="h == array([22., 4.2])", name="h equation", reference=24.)
        ])
        mr.add_target("i")
        mr.add_unknown([
            dict(name="a", max_rel_step=1e-5),
            dict(name="c", max_abs_step=1e-2),
        ])
        # Define mathematical problem 'ms'
        ms.add_equation("v == 0").add_unknown("y")
        ms.add_target("u")

        return r, s, mr, ms

    # Test default extension (should copy unknowns and residues)
    r, s, mr, ms = local_test_objects()
    assert list(mr.unknowns) == ['a', 'c']
    assert list(mr.residues) == ['g == 0', 'h equation']
    assert [deferred.target for deferred in mr.deferred_residues.values()] == ['i']

    with no_exception():
        extended = mr.extend(mr, copy=False)
    assert extended is mr
    
    with pytest.raises(ValueError):
        mr.extend(mr, copy=True)

    with pytest.raises(ValueError):
        mr.extend(mr)
    
    # Check that option `overwrite=True` avoids the exception:
    with no_exception():
        extended = mr.extend(mr, overwrite=True)
    assert extended is mr
    assert list(mr.unknowns) == ['a', 'c']
    assert list(mr.residues) == ['g == 0', 'h equation']
    assert [deferred.target for deferred in mr.deferred_residues.values()] == ['i']

    with pytest.raises(ValueError, match=r".* is not a child of .*\."):
        mr.extend(ms)

    r.add_child(s)
    extended = mr.extend(ms)
    assert extended is mr
    assert mr.context is r
    assert list(mr.unknowns) == ['a', 'c', 's.y']
    assert list(mr.residues) == ['g == 0', 'h equation', 's: v == 0']
    assert len(mr.transients) == 0
    assert len(mr.rates) == 0
    assert len(mr.deferred_residues) == 2
    assert all(
        isinstance(obj.deferred, DeferredResidue)
        for obj in mr.deferred_residues.values()
    )
    assert [deferred.target for deferred in mr.deferred_residues.values()] == ['i', 's.u']

    assert mr.unknowns['s.y'] is not ms.unknowns['y']
    assert mr.residues['s: v == 0'] is not ms.residues['v == 0']

    # Test with pulled output
    r, s, mr, ms = local_test_objects()

    r.add_child(s, pulling='u')
    extended = mr.extend(ms)
    assert extended is mr
    assert mr.context is r
    assert list(mr.unknowns) == ['a', 'c', 's.y']
    assert list(mr.residues) == ['g == 0', 'h equation', 's: v == 0']
    assert len(mr.transients) == 0
    assert len(mr.rates) == 0
    assert len(mr.deferred_residues) == 2
    assert all(
        isinstance(obj.deferred, DeferredResidue)
        for obj in mr.deferred_residues.values()
    )
    assert [deferred.target for deferred in mr.deferred_residues.values()] == ['i', 'u']

    assert mr.unknowns['s.y'] is not ms.unknowns['y']
    assert mr.residues['s: v == 0'] is not ms.residues['v == 0']

    # Test extension with option copy = False
    r, s, mr, ms = local_test_objects()

    r.add_child(s)
    mr.extend(ms, copy=False)

    assert mr.context is r
    assert list(mr.unknowns) == ['a', 'c', 's.y']
    assert list(mr.residues) == ['g == 0', 'h equation', 's: v == 0']
    assert len(mr.transients) == 0
    assert len(mr.rates) == 0
    assert len(mr.deferred_residues) == 2
    assert all(
        isinstance(obj.deferred, DeferredResidue)
        for obj in mr.deferred_residues.values()
    )
    assert [deferred.target for deferred in mr.deferred_residues.values()] == ['i', 's.u']

    assert mr.unknowns['s.y'] is ms.unknowns['y']
    assert mr.residues['s: v == 0'] is ms.residues['v == 0']


def test_MathematicalProblem_extend_partial():
    """Test partial extension, disregarding either unknowns or equations"""
    def local_test_objects():
        r, s = SystemA('r'), SystemB('s')
        mr = MathematicalProblem('test', r)
        ms = MathematicalProblem('test', s)
        # Define mathematical problem 'mr'
        mr.add_equation([
            dict(equation="g == 0"),
            dict(equation="h == array([22., 4.2])", name="h equation", reference=24.)
        ])
        mr.add_target("i")
        mr.add_unknown([
            dict(name="a", max_rel_step=1e-5),
            dict(name="c", max_abs_step=1e-2),
        ])
        # Define mathematical problem 'ms'
        ms.add_equation("v == 0").add_unknown("y")
        ms.add_target("u")

        return r, s, mr, ms

    # Discard equations + default mode (copy)
    r, s, mr, ms = local_test_objects()

    r.add_child(s)
    extended = mr.extend(ms, equations=False)
    assert extended is mr
    assert mr.context is r
    assert list(mr.unknowns) == ['a', 'c', 's.y']
    assert list(mr.residues) == ['g == 0', 'h equation']
    assert len(mr.transients) == 0
    assert len(mr.rates) == 0
    assert len(mr.deferred_residues) == 1
    assert all(
        isinstance(obj.deferred, DeferredResidue)
        for obj in mr.deferred_residues.values()
    )
    assert [deferred.target for deferred in mr.deferred_residues.values()] == ['i']
    assert mr.unknowns['s.y'] is not ms.unknowns['y']

    # Discard equations + no copy
    r, s, mr, ms = local_test_objects()

    r.add_child(s)
    extended = mr.extend(ms, equations=False, copy=False)
    assert extended is mr
    assert mr.context is r
    assert list(mr.unknowns) == ['a', 'c', 's.y']
    assert list(mr.residues) == ['g == 0', 'h equation']
    assert len(mr.transients) == 0
    assert len(mr.rates) == 0
    assert len(mr.deferred_residues) == 1
    assert all(
        isinstance(obj.deferred, DeferredResidue)
        for obj in mr.deferred_residues.values()
    )
    assert [deferred.target for deferred in mr.deferred_residues.values()] == ['i']
    assert mr.unknowns['s.y'] is ms.unknowns['y']

    # Discard unknowns + default mode (copy)
    r, s, mr, ms = local_test_objects()

    r.add_child(s)
    extended = mr.extend(ms, unknowns=False)
    assert extended is mr
    assert mr.context is r
    assert list(mr.unknowns) == ['a', 'c']
    assert list(mr.residues) == ['g == 0', 'h equation', 's: v == 0']
    assert len(mr.transients) == 0
    assert len(mr.rates) == 0
    assert len(mr.deferred_residues) == 2
    assert all(
        isinstance(obj.deferred, DeferredResidue)
        for obj in mr.deferred_residues.values()
    )
    assert [deferred.target for deferred in mr.deferred_residues.values()] == ['i', 's.u']
    assert mr.residues['s: v == 0'] is not ms.residues['v == 0']

    # Discard unknowns + no copy
    r, s, mr, ms = local_test_objects()

    r.add_child(s)
    extended = mr.extend(ms, unknowns=False, copy=False)
    assert extended is mr
    assert mr.context is r
    assert list(mr.unknowns) == ['a', 'c']
    assert list(mr.residues) == ['g == 0', 'h equation', 's: v == 0']
    assert len(mr.transients) == 0
    assert len(mr.rates) == 0
    assert len(mr.deferred_residues) == 2
    assert all(
        isinstance(obj.deferred, DeferredResidue)
        for obj in mr.deferred_residues.values()
    )
    assert [deferred.target for deferred in mr.deferred_residues.values()] == ['i', 's.u']
    assert mr.residues['s: v == 0'] is ms.residues['v == 0']


def test_MathematicalProblem_extend_pulled_target(caplog):
    def local_test_objects():
        r, s = SystemA('r'), SystemB('s')
        # Define mathematical problem on `r`
        mr = MathematicalProblem('test', r)
        mr.add_target(["g", "2 * h", "out.m"])
        # Define mathematical problem on `s`
        ms = MathematicalProblem('test', s)
        ms.add_target(["u", "norm(v)"])

        return r, s, mr, ms

    # Test 1
    r, s, mr, ms = local_test_objects()
    r.add_child(s, pulling={'v': 'v_alias'})

    caplog.clear()
    with caplog.at_level(logging.INFO):
        mx = mr.extend(ms)
    
    assert len(caplog.records) == 1
    assert re.match(
        "Target on 'norm\(s.v\)' will be based on 'norm\(v_alias\)'",
        caplog.records[0].message
    )
    assert mx is mr
    assert mx.context is r
    assert len(mr.deferred_residues) == 5
    assert all(
        isinstance(obj.deferred, DeferredResidue)
        for obj in mx.deferred_residues.values()
    )
    assert [deferred.target for deferred in mx.deferred_residues.values()] == [
        'g', '2 * h', 'out.m', 's.u', 'norm(v_alias)',
    ]

    # Test 2
    r, s, mr, ms = local_test_objects()
    r.add_child(s, pulling={'u': 'U'})

    caplog.clear()
    with caplog.at_level(logging.INFO):
        mx = mr.extend(ms)
    
    assert len(caplog.records) == 1
    assert re.match(
        "Target on 's.u' will be based on 'U'",
        caplog.records[0].message
    )
    assert mx is mr
    assert mr.context is r
    assert len(mx.deferred_residues) == 5
    assert all(
        isinstance(obj.deferred, DeferredResidue)
        for obj in mx.deferred_residues.values()
    )
    assert [deferred.target for deferred in mx.deferred_residues.values()] == [
        'g', '2 * h', 'out.m', 'U', 'norm(s.v)',
    ]

    # Test 3
    r, s, mr, ms = local_test_objects()
    s.add_child(r, pulling={'g': 'g', 'h': 'H'})

    caplog.clear()
    with caplog.at_level(logging.INFO):
        mx = ms.extend(mr)
    
    assert len(caplog.records) == 2
    assert re.match(
        "Target on 'r\.g' will be based on 'g'",
        caplog.records[0].message
    )
    assert re.match(
        "Target on '2 \* r\.h' will be based on '2 \* H'",
        caplog.records[1].message
    )

    assert mx is ms
    assert mx.context is s
    assert len(mx.deferred_residues) == 5
    assert all(
        isinstance(obj.deferred, DeferredResidue)
        for obj in mx.deferred_residues.values()
    )
    assert [deferred.target for deferred in mx.deferred_residues.values()] == [
        'u', 'norm(v)', 'g', '2 * H', 'r.out.m',
    ]

    # Test 4
    r, s, mr, ms = local_test_objects()
    s.add_child(r, pulling={'out': 'bogus_out', 'g': 'G'})

    caplog.clear()
    with caplog.at_level(logging.INFO):
        mx = ms.extend(mr)
    
    assert len(caplog.records) == 2
    assert re.match(
        "Target on 'r\.g' will be based on 'G'",
        caplog.records[0].message
    )
    assert re.match(
        "Target on 'r\.out\.m' will be based on 'bogus_out\.m'",
        caplog.records[1].message
    )
    assert mx is ms
    assert mx.context is s
    assert len(mx.deferred_residues) == 5
    assert all(
        isinstance(obj.deferred, DeferredResidue)
        for obj in mx.deferred_residues.values()
    )
    assert [deferred.target for deferred in mx.deferred_residues.values()] == [
        'u', 'norm(v)', 'G', '2 * r.h', 'bogus_out.m',
    ]


def test_MathematicalProblem_clear(test_objects: Tuple[System, MathematicalProblem]):
    s, m = test_objects
    m.add_equation([
        dict(equation="g == 0"),
        dict(equation="h == array([22., 4.2])", name="h equation", reference=24.)
    ])
    m.add_unknown([
        dict(name="a", max_rel_step=1e-5),
        dict(name="c", max_abs_step=1e-2)
    ])

    assert m.context is s
    assert list(m.unknowns) == ['a', 'c']
    assert list(m.residues) == ['g == 0', 'h equation']
    assert len(m.transients) == 0
    assert len(m.rates) == 0
    assert len(m.deferred_residues) == 0

    for unknown in m.unknowns.values():
        assert unknown.context is s
    assert m.unknowns['a'].max_rel_step == 1e-5
    assert m.unknowns['c'].max_abs_step == 1e-2

    name = 'h equation'
    assert name in m.residues
    assert m.residues[name].context is s
    assert m.residues[name].name is name

    m.add_transient('d', der='g**2')
    assert list(m.transients) == ['d']
    assert list(m.rates) == []
    assert list(m.deferred_residues) == []

    m.add_rate('a', source='b[0]')
    assert list(m.transients) == ['d']
    assert list(m.rates) == ['a']
    assert list(m.deferred_residues) == []

    m.add_target('g')
    assert list(m.transients) == ['d']
    assert list(m.rates) == ['a']
    assert [deferred.target for deferred in m.deferred_residues.values()] == ['g']

    m.clear()
    assert m.context is s
    assert len(m.unknowns) == 0
    assert len(m.residues) == 0
    assert len(m.transients) == 0
    assert len(m.rates) == 0
    assert len(m.deferred_residues) == 0


def test_MathematicalProblem_copy(test_objects: Tuple[System, MathematicalProblem]):
    context, original = test_objects
    original.add_equation([
        dict(equation="g == 0"),
        dict(equation="h == array([22., 4.2])", name="h equation", reference=24.)
    ])
    original.add_unknown([
        dict(name="a", max_rel_step=1e-5),
        dict(name="c", max_abs_step=1e-2)
    ])
    original.add_target('i')

    copy = original.copy()
    assert copy is not original

    assert copy.context is original.context
    assert copy.context is context

    assert list(copy.unknowns) == list(original.unknowns)
    assert list(copy.residues) == list(original.residues)
    assert list(copy.unknowns) == ['a', 'c']
    assert list(copy.residues) == ['g == 0', 'h equation']
    assert all(
        unknown is not original.unknowns[key]
        for key, unknown in copy.unknowns.items()
    )
    assert all(
        residue is not original.residues[key]
        for key, residue in copy.residues.items()
    )
    assert list(copy.deferred_residues) == list(original.deferred_residues)
    assert list(copy.deferred_residues) == ['i (target)']
    
    unknown = copy.unknowns['c']
    assert isinstance(unknown, Unknown)
    assert unknown.context is context
    assert unknown.max_abs_step == 1e-2
    
    name = 'h equation'
    residue = copy.residues[name]
    assert isinstance(residue, Residue)
    assert residue.context is context
    assert residue.name == name


def test_MathematicalProblem_to_dict(test_objects: Tuple[System, MathematicalProblem]):
    problem = test_objects[1]
    problem.add_equation([
        dict(equation="g == 0"),
        dict(equation="h == array([22., 4.2])", name="h equation", reference=24.)
    ])
    problem.add_unknown([
        dict(name="a", max_rel_step=1e-5),
        dict(name="c", max_abs_step=1e-2)
    ])

    problem_dict = problem.to_dict()

    assert "unknowns" in problem_dict
    assert len(problem_dict["unknowns"]) == 2
    assert problem_dict["unknowns"]["a"] == problem.unknowns["a"].to_dict()
    assert "equations" in problem_dict
    assert len(problem_dict["equations"]) == 2
    assert problem_dict["equations"]["g == 0"] == problem.residues["g == 0"].to_dict()
    assert "transients" in problem_dict
    assert "rates" in problem_dict
    assert len(problem_dict) == 4


def test_MathematicalProblem_validate(test_objects: Tuple[System, MathematicalProblem]):
    s, problem = test_objects
    problem.add_equation("g == 0").add_unknown("c[:2]")

    assert problem.n_unknowns == 2
    assert problem.n_equations == 1
    with pytest.raises(ArithmeticError, match= r"Nonlinear problem .* error: Mismatch between numbers of params .* and residues .*"):
        problem.validate()

    problem.add_unknown("a")
    problem.add_equation("h == array([22., 4.2])", name="h equation", reference=24.)
    assert problem.n_unknowns == 3
    assert problem.n_equations == 3
    assert problem.validate() is None
