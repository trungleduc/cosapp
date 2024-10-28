import pytest

import numpy as np
from numbers import Number
from typing import Tuple

from cosapp.base import System, Port
from cosapp.core.numerics.basics import TimeProblem
from cosapp.core.numerics.boundary import TimeUnknown, TimeDerivative
from cosapp.utils.testing import no_exception


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
        self.add_inward('nd_in', np.ones((2, 4, 3), dtype=float))

        self.add_outward('g', 3.5)
        self.add_outward('h', [1., 2.])
        self.add_outward('i', 5.)
        self.add_outward('nd_out', np.zeros_like(self.nd_in))


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
def test_objects() -> Tuple[System, TimeProblem]:
    system = SystemA('system_a')
    return system, TimeProblem('time_pb', system)


def is_TimeUnknown(obj):
    return isinstance(obj, TimeUnknown)


def is_TimeDerivative(obj):
    return isinstance(obj, TimeDerivative)


def test_TimeProblem__init__():
    # Empty case
    m = TimeProblem('test', None)
    assert m.name == 'test'
    assert m.context is None
    assert len(m.transients) == 0
    assert len(m.rates) == 0


def test_TimeProblem_name(test_objects: Tuple[System, TimeProblem]):
    s, m = test_objects
    with pytest.raises(AttributeError):
        setattr(m, 'name', 'banana')


def test_TimeProblem_noSetters(test_objects: Tuple[System, TimeProblem]):
    s, m = test_objects

    ds = DynamicSystemC('ds')
    with pytest.raises(AttributeError):
        m.transients = [TimeUnknown(ds, "A", "2.5 * q"), ]


def test_TimeProblem_add_transient():
    s = DynamicSystemC('s')
    m = TimeProblem('test', s)
    m.add_transient('A', der='q')
    assert list(m.transients) == ['A']
    A = m.transients['A']
    assert isinstance(A.context[A.basename], Number)
    assert A.context is s
    assert A.name == 'A'
    assert A.d_dt == 1

    m.add_transient('x', der='v / q**2')
    assert list(m.transients) == ['A', 'x']
    x = m.transients['x']
    assert isinstance(x.context[x.basename], np.ndarray)
    s.v = np.r_[1, 2, 3]
    s.q = 2.0
    assert x.context is s
    assert x.name == 'x'
    assert x.d_dt == pytest.approx(np.r_[0.25, 0.5, 0.75], rel=1e-15)


def test_TimeProblem_extend():
    def local_test_objects():
        r, s = SystemA('r'), SystemB('s')
        pr = TimeProblem('test', r)
        ps = TimeProblem('test', s)
        # Define mathematical problem 'mr'
        pr.add_transient("a", der="h")
        pr.add_transient("c", der="g + i")
        pr.add_rate("d", source="nd_out[0]")
        # Define mathematical problem 'ms'
        ps.add_transient("x", der="u")
        ps.add_rate("y", source="v")

        return r, s, pr, ps

    # Test default extension (should copy unknowns and residues)
    r, s, pr, ps = local_test_objects()
    assert list(pr.transients) == ['a', 'c']
    assert list(pr.rates) == ['d']

    with no_exception():
        extended = pr.extend(pr, copy=False)
    assert extended is pr
    
    with pytest.raises(ValueError):
        pr.extend(pr, copy=True)

    with pytest.raises(ValueError):
        pr.extend(pr)
    
    # Check that option `overwrite=True` avoids the exception:
    with no_exception():
        extended = pr.extend(pr, overwrite=True)
    assert extended is pr
    assert list(pr.transients) == ['a', 'c']
    assert list(pr.rates) == ['d']

    with pytest.raises(ValueError, match=r".* is not a child of .*\."):
        pr.extend(ps)

    r.add_child(s)
    extended = pr.extend(ps)
    assert extended is pr
    assert pr.context is r
    assert list(pr.transients) == ['a', 'c', 's.x']
    assert list(pr.rates) == ['d', 's.y']
    assert all(map(is_TimeUnknown, pr.transients.values()))
    assert all(map(is_TimeDerivative, pr.rates.values()))

    assert pr.transients['s.x'] is not ps.transients['x']
    assert pr.rates['s.y'] is not ps.rates['y']

    # Test with pulled output
    r, s, pr, ps = local_test_objects()

    r.add_child(s, pulling='u')
    extended = pr.extend(ps)
    assert extended is pr
    assert pr.context is r
    assert list(pr.transients) == ['a', 'c', 's.x']
    assert list(pr.rates) == ['d', 's.y']
    assert all(map(is_TimeUnknown, pr.transients.values()))
    assert all(map(is_TimeDerivative, pr.rates.values()))

    assert pr.transients['s.x'] is not ps.transients['x']
    assert pr.rates['s.y'] is not ps.rates['y']

    # Test extension with option copy = False
    r, s, pr, ps = local_test_objects()

    r.add_child(s)
    pr.extend(ps, copy=False)

    assert pr.context is r
    assert list(pr.transients) == ['a', 'c', 's.x']
    assert list(pr.rates) == ['d', 's.y']
    assert all(map(is_TimeUnknown, pr.transients.values()))
    assert all(map(is_TimeDerivative, pr.rates.values()))

    assert pr.transients['s.x'] is ps.transients['x']
    assert pr.rates['s.y'] is ps.rates['y']


def test_TimeProblem_clear(test_objects: Tuple[System, TimeProblem]):
    s, m = test_objects
    assert m.is_empty()
    assert m.context is s

    m.add_transient('d', der='g**2')
    assert list(m.transients) == ['d']
    assert list(m.rates) == []
    assert not m.is_empty()

    m.add_rate('a', source='b[0]')
    assert list(m.transients) == ['d']
    assert list(m.rates) == ['a']
    assert not m.is_empty()

    m.clear()
    assert m.is_empty()
    assert m.context is s
    assert len(m.transients) == 0
    assert len(m.rates) == 0


def test_TimeProblem_copy(test_objects: Tuple[System, TimeProblem]):
    context, original = test_objects
    original.add_transient('d', der='g**2')
    original.add_rate('a', source='b[0]')

    copy = original.copy()
    assert copy is not original

    assert copy.context is original.context
    assert copy.context is context

    assert list(copy.transients) == list(original.transients)
    assert list(copy.rates) == list(original.rates)
    assert list(copy.transients) == ['d']
    assert list(copy.rates) == ['a']
    assert all(
        transient is not original.transients[key]
        for key, transient in copy.transients.items()
    )
    assert all(
        rate is not original.rates[key]
        for key, rate in copy.rates.items()
    )


def test_TimeProblem_to_dict(test_objects: Tuple[System, TimeProblem]):
    problem = test_objects[1]
    problem.add_transient('d', der='g**2')
    problem.add_rate('a', source='b[0]')

    problem_dict = problem.to_dict()

    assert set(problem_dict) == {"transients", "rates"}
    assert set(problem_dict["transients"]) == {"d"}
    assert problem_dict["transients"]["d"] == problem.transients["d"].to_dict()
    assert set(problem_dict["rates"]) == {"a"}
    assert problem_dict["rates"]["a"] == problem.rates["a"].to_dict()
