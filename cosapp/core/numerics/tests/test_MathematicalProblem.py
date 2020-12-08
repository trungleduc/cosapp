import pytest

from collections import OrderedDict
from numbers import Number
import numpy as np

from cosapp.systems import System
from cosapp.ports import Port
from cosapp.core.numerics.basics import MathematicalProblem
from cosapp.core.numerics.boundary import Unknown, TimeUnknown
from cosapp.core.numerics.residues import Residue


class BogusPort(Port):
    def setup(self):
        self.add_variable('m')

class SystemA(System):
    def setup(self):
        self.add_input(BogusPort, 'in_')
        self.add_inward('a', 1.)
        self.add_inward('b', [1., 2.])
        self.add_inward('c', np.asarray([1., 2.]))
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
def test_objects():
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


def test_MathematicalProblem_name(test_objects):
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

    m = MathematicalProblem('test', sa)
    assert m.context is sa
    with pytest.raises(ValueError, match="Context is already set to .*"):
        m.context = sb
    assert m.context is sa


def test_MathematicalProblem_noSetters(test_objects):
    s, m = test_objects

    with pytest.raises(AttributeError):
        m.unknowns = [Unknown(s, "a"), ]
    
    with pytest.raises(AttributeError):
        m.residues = OrderedDict(a=Residue(s, "a == 0"))
    
    ds = DynamicSystemC('ds')
    with pytest.raises(AttributeError):
        m.transients = [TimeUnknown(ds, "A", "2.5 * q"), ]


def test_MathematicalProblem_residues_vector(test_objects):
    s, m = test_objects
    m.add_equation([
        dict(equation="g == 0"),
        dict(equation="h == array([22., 4.2])", name="h - gorilla", reference=24.)
    ])
    m.add_unknown([
        dict(name="a", max_rel_step=1e-5),
        dict(name="c", max_abs_step=1e-2)
    ])

    assert len(m.residues_vector) == 3
    assert m.residues_vector == pytest.approx(
        np.concatenate(([m.residues['g == 0'].value], m.residues['h - gorilla'].value)))


def test_MathematicalProblem_shape(test_objects):
    s, m = test_objects
    m.add_equation("g == 0")
    m.add_unknown("c")
    assert m.shape == (2, 1)

    m.add_unknown("a")
    m.add_equation("h == array([22., 4.2])", name="h - gorilla", reference=24.)
    assert m.shape == (3, 3)


def test_MathematicalProblem_add_methods():
    # Single element case
    s = SystemA('a')
    m = MathematicalProblem('test', s)
    m.add_equation("g == 0")
    m.add_unknown("a")
    assert m.context is s
    assert len(m.unknowns) == 1
    assert len(m.residues) == 1
    assert m.shape == (1, 1)

    unknown = m.unknowns['inwards.a']
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
        dict(equation="h == array([22., 4.2])", name="h - gorilla", reference=24.)
    ])
    m.add_unknown([
        dict(name="a", max_rel_step=1e-5),
        dict(name="c", max_abs_step=1e-2)
    ])

    assert m.context is s
    assert len(m.unknowns) == 2
    assert len(m.residues) == 2
    assert len(m.transients) == 0

    assert 'inwards.c' in m.unknowns
    unknown = m.unknowns['inwards.c']
    assert isinstance(unknown, Unknown)
    assert unknown.context is s
    assert unknown.max_abs_step == 1e-2

    residue = m.residues['h - gorilla']
    assert isinstance(residue, Residue)
    assert residue.context is s
    assert residue.name == 'h - gorilla'


def test_MathematicalProblem_add_equations():
    r = SystemA('a')
    m = MathematicalProblem('test', r)

    assert len(m.residues) == 0
    m.add_equation("g == 0")
    assert len(m.residues) == 1
    assert 'g == 0' in m.residues
    assert m.residues['g == 0'].name == 'g == 0'

    r = SystemA('a')
    m = MathematicalProblem('test', r)
    m.add_equation("g == i", "my_res")
    assert len(m.residues) == 1
    assert 'my_res' in m.residues
    assert m.residues['my_res'].name == 'my_res'

    r = SystemA('a')
    m = MathematicalProblem('test', r)
    m.add_equation(["g == i", "h == 0"])
    assert len(m.residues) == 2

    for name in ['h == 0', 'g == i']:
        message = f"residue {name!r}"
        assert name in m.residues
        residue = m.residues[name]
        assert isinstance(residue, Residue), message
        assert residue.name == name, message

    r = SystemA('a')
    m = MathematicalProblem('test', r)
    m.add_equation(["g == i", dict(equation="h == 0", name="2nd")])
    assert len(m.residues) == 2

    for name in ['g == i', '2nd']:
        message = f"residue {name!r}"
        assert name in m.residues
        residue = m.residues[name]
        assert isinstance(residue, Residue), message
        assert residue.name == name, message
        assert residue.context is r, message


def test_MathematicalProblem_add_unknowns():
    r = SystemA('a')
    m = MathematicalProblem('test', r)
    m.add_unknown("a")
    assert len(m.unknowns) == 1

    for name in ['inwards.a', ]:
        assert name in m.unknowns
        unknown = m.unknowns[name]
        message = f"unknown {name}"
        assert isinstance(unknown, Unknown), message
        assert unknown.name == name, message
        assert unknown.context is r, message

    r = SystemA('a')
    m = MathematicalProblem('test', r)
    m.add_unknown(["a", "d"])
    assert len(m.unknowns) == 2

    for name in ['inwards.a', 'inwards.d']:
        assert name in m.unknowns
        unknown = m.unknowns[name]
        message = f"unknown {name}"
        assert isinstance(unknown, Unknown), message
        assert unknown.name == name, message
        assert unknown.context is r, message

    r = SystemA('a')
    m = MathematicalProblem('test', r)
    m.add_unknown(["in_.m", dict(name="d", max_rel_step=0.1)])
    assert len(m.unknowns) == 2

    for name in ['in_.m', 'inwards.d']:
        assert name in m.unknowns
        unknown = m.unknowns[name]
        message = f"unknown {name}"
        assert isinstance(unknown, Unknown), message
        assert unknown.name == name, message
        assert unknown.context is r, message
    unknown = m.unknowns['inwards.d']
    assert unknown.max_rel_step == 0.1

    m.add_unknown('a')
    assert len(m.unknowns) == 3
    assert 'inwards.a' in m.unknowns

    # Vector
    r = SystemA('r')
    m = MathematicalProblem('test_vector', r)
    m.add_unknown("c")
    assert len(m.unknowns) == 1
    assert len(m.unknowns_names) == len(r.c)

    r = SystemA('r')
    m = MathematicalProblem('test_vector', r)
    m.add_unknown("c[1]")
    assert len(m.unknowns) == 1
    assert len(m.unknowns_names) == 1

    r = SystemA('r')
    m = MathematicalProblem('test_vector', r)
    m.add_unknown("c", mask=[False, True])
    assert len(m.unknowns) == 1
    assert len(m.unknowns_names) == 1

    r = SystemA('r')
    m = MathematicalProblem('test_vector', r)
    m.add_unknown([dict(name="c", mask=[False, True])])
    assert len(m.unknowns) == 1
    assert len(m.unknowns_names) == 1

def test_MathematicalProblem_add_transient():
    s = DynamicSystemC('s')
    m = MathematicalProblem('test', s)
    m.add_transient('A', der='q')
    assert len(m.transients) == 1
    assert 'A' in m.transients
    A = m.transients['A']
    assert A.context is s
    assert A.name == 'inwards.A'
    assert A.d_dt == 1
    assert isinstance(A.value, Number)

    m.add_transient('x', der='v / q**2')
    assert len(m.transients) == 2
    assert 'x' in m.transients
    x = m.transients['x']
    s.v = np.r_[1, 2, 3]
    s.q = 2.0
    assert x.context is s
    assert x.name == 'inwards.x'
    assert x.d_dt == pytest.approx(np.r_[0.25, 0.5, 0.75], rel=1e-15)


def test_MathematicalProblem_extend():
    def local_test_objects():
        r, s = SystemA('asyst'), SystemB('bsyst')
        m = MathematicalProblem('test', r)
        n = MathematicalProblem('test', s)
        # Define mathematical problem 'm'
        m.add_equation([
            dict(equation="g == 0"),
            dict(equation="h == array([22., 4.2])", name="h - gorilla", reference=24.)
        ])
        m.add_unknown([
            dict(name="a", max_rel_step=1e-5),
            dict(name="c", max_abs_step=1e-2)
        ])
        # Define mathematical problem 'n'
        n.add_equation("v == 0").add_unknown("y")

        return r, s, m, n

    # Test default extension (should copy unknowns and residues)
    r, s, m, n = local_test_objects()

    with pytest.raises(ValueError, match=r".* is not a child of .*\."):
        m.extend(n)

    r.add_child(s)
    m.extend(n)
    assert m.context is r
    assert len(m.unknowns) == 3
    assert len(m.residues) == 3
    assert len(m.transients) == 0
    assert len(m.rates) == 0

    for name in ('inwards.a', 'inwards.c', 'bsyst.inwards.y'):
        assert name in m.unknowns
    
    for name in ('g == 0', 'h - gorilla', 'bsyst.(v == 0)'):
        assert name in m.residues

    assert m.unknowns['bsyst.inwards.y'] is not n.unknowns['inwards.y']
    assert m.residues['bsyst.(v == 0)'] is not n.residues['v == 0']

    # Test extension with option copy = False
    r, s, m, n = local_test_objects()

    r.add_child(s)
    m.extend(n, copy=False)

    assert m.context is r
    assert len(m.unknowns) == 3
    assert len(m.residues) == 3
    assert len(m.transients) == 0
    assert len(m.rates) == 0

    for name in ('inwards.a', 'inwards.c', 'bsyst.inwards.y'):
        assert name in m.unknowns
    
    for name in ('g == 0', 'h - gorilla', 'bsyst.(v == 0)'):
        assert name in m.residues

    assert m.unknowns['bsyst.inwards.y'] is n.unknowns['inwards.y']
    assert m.residues['bsyst.(v == 0)'] is n.residues['v == 0']


def test_MathematicalProblem_clear(test_objects):
    s, m = test_objects
    m.add_equation([
        dict(equation="g == 0"),
        dict(equation="h == array([22., 4.2])", name="h - gorilla", reference=24.)
    ])
    m.add_unknown([
        dict(name="a", max_rel_step=1e-5),
        dict(name="c", max_abs_step=1e-2)
    ])

    assert m.context is s
    assert len(m.unknowns) == 2
    assert len(m.residues) == 2
    assert len(m.transients) == 0
    assert len(m.rates) == 0
    for name in ['inwards.a', 'inwards.c']:
        assert name in m.unknowns
        assert m.unknowns[name].context is s
    assert m.unknowns['inwards.a'].max_rel_step == 1e-5
    assert m.unknowns['inwards.c'].max_abs_step == 1e-2

    name = 'h - gorilla'
    assert name in m.residues
    assert m.residues[name].context is s
    assert m.residues[name].name is name

    m.add_transient('d', der='g**2')
    assert len(m.transients) == 1
    assert len(m.rates) == 0

    m.add_rate('a', source='b[0]')
    assert len(m.transients) == 1
    assert len(m.rates) == 1

    m.clear()
    assert m.context is s
    assert len(m.unknowns) == 0
    assert len(m.residues) == 0
    assert len(m.transients) == 0
    assert len(m.rates) == 0


def test_MathematicalProblem_copy(test_objects):
    s, original = test_objects
    original.add_equation([
        dict(equation="g == 0"),
        dict(equation="h == array([22., 4.2])", name="h - gorilla", reference=24.)
    ])
    original.add_unknown([
        dict(name="a", max_rel_step=1e-5),
        dict(name="c", max_abs_step=1e-2)
    ])

    copy = original.copy()
    assert copy is not original

    assert original.context is s
    assert copy.context is original.context

    assert len(original.unknowns) == 2
    assert len(original.residues) == 2
    assert copy.unknowns.keys() == original.unknowns.keys()
    assert copy.residues.keys() == original.residues.keys()
    
    name = 'inwards.c'
    assert name in copy.unknowns
    unknown = copy.unknowns[name]
    assert isinstance(unknown, Unknown)
    assert unknown.context is s
    assert unknown.max_abs_step == 1e-2
    
    name = 'h - gorilla'
    assert name in copy.residues
    residue = copy.residues[name]
    assert isinstance(residue, Residue)
    assert residue.context is s
    assert residue.name == name


def test_MathematicalProblem_to_dict(test_objects):
    _, problem = test_objects
    problem.add_equation([
        dict(equation="g == 0"),
        dict(equation="h == array([22., 4.2])", name="h - gorilla", reference=24.)
    ])
    problem.add_unknown([
        dict(name="a", max_rel_step=1e-5),
        dict(name="c", max_abs_step=1e-2)
    ])

    problem_dict = problem.to_dict()

    assert "unknowns" in problem_dict
    assert len(problem_dict["unknowns"]) == 2
    assert problem_dict["unknowns"]["inwards.a"] == problem.unknowns["inwards.a"].to_dict()
    assert "equations" in problem_dict
    assert len(problem_dict["equations"]) == 2
    assert problem_dict["equations"]["g == 0"] == problem.residues["g == 0"].to_dict()
    assert "transients" in problem_dict
    assert "rates" in problem_dict
    assert len(problem_dict) == 4


def test_MathematicalProblem_validate(test_objects):
    s, m = test_objects
    m.add_equation("g == 0").add_unknown("c")

    with pytest.raises(ArithmeticError, match= r"Nonlinear problem .* error: Mismatch between numbers of params .* and residues .*"):
        m.validate()

    m.add_unknown("a")
    m.add_equation("h == array([22., 4.2])", name="h - gorilla", reference=24.)
    assert m.validate() is None
