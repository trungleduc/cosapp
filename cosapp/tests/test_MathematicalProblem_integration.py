import pytest
import numpy as np
import logging, re
from typing import Type

from cosapp.base import System, Port
from cosapp.core import MathematicalProblem
from cosapp.drivers import NonLinearSolver, RunSingleCase
from cosapp.utils.testing import get_args, DummySystemFactory


class AbcPort(Port):
    def setup(self):
        self.add_variable('a', 1.0)
        self.add_variable('b', 1.0)
        self.add_variable('c', np.zeros(4))


class XyzPort(Port):
    def setup(self):
        self.add_variable('x', 1.0)
        self.add_variable('y', 1.0)
        self.add_variable('z', 1.0)


@pytest.fixture
def A_factory() -> Type[System]:
    def factory(classname, **settings):
        return DummySystemFactory(
            classname,
            inputs = get_args(XyzPort, 'p_in'),
            outputs = get_args(AbcPort, 'q_out'),
            inwards = [
                get_args('x', 1.0),
                get_args('y', 0.5),
                get_args('u', np.ones(3)),
            ],
            outwards = [get_args('z', 0.0)],
            **settings,
        )
    return factory


@pytest.fixture
def B_factory() -> Type[System]:
    def factory(classname, **settings):
        return DummySystemFactory(
            classname,
            inwards = [
                get_args('length', 1.0),
                get_args('width', 0.5),
            ],
            outwards = [
                get_args('area', 0.0),
                get_args('v', np.zeros(4)),
            ],
            inputs = get_args(XyzPort, 'p_in'),
            **settings,
        )
    return factory


@pytest.fixture
def A(A_factory) -> Type[System]:
    return A_factory("A")


@pytest.fixture
def Aun(A_factory) -> Type[System]:
    return A_factory("Aun",
        unknowns = "p_in.x",
    )


@pytest.fixture
def Atrg(A_factory) -> Type[System]:
    return A_factory("Atrg",
        targets = [
            "z",
            "q_out.a",
            "abs(q_out.b)",
            "q_out.c[::2]",
        ],
    )


@pytest.fixture
def B(B_factory) -> Type[System]:
    return B_factory("B")


@pytest.fixture
def Beq(B_factory) -> Type[System]:
    return B_factory("Beq",
        equations = "v[::2] == [0, 1]",
    )


@pytest.fixture
def Bun(B_factory) -> Type[System]:
    return B_factory("Bun",
        unknowns = get_args(["p_in.x", "p_in.y"]),
    )


@pytest.fixture
def s1(A, Aun, B, Beq):
    """Generates composite system:

             top
         _____|_____
        |     |     |
        a     b    foo
        |           |
       aun         beq
    """
    a: System = A('a')
    a.add_child(Aun('aun'), pulling={'p_in': 'p'})
    top = System('s1')
    top.add_child(a, pulling=['x', 'y', 'q_out'])
    top.add_child(B('b'))
    foo = System('foo')
    foo.add_child(Beq('beq'), pulling={'v': 'v', 'p_in': 'xyz_in'})
    top.add_child(foo)
    return top


@pytest.fixture
def s2(Atrg):
    """Generates simple composite system `top/sub`.
    Sub-system `sub` has targets, some of which are pulled.
    """
    top = System('s2')
    top.add_child(Atrg('sub'), pulling={'q_out': 'out'})
    # Initialize outputs manually
    top.sub.z = 0.12
    top.out.set_values(
        a = 3.14,
        b = -2.3,
        c = np.r_[0.1, 0.2, 0.3, 0.4],
    )
    top.sub.q_out.set_values(
        a = 0.0,
        b = 0.0,
        c = np.zeros(4),
    )
    return top


def test_s1(s1):
    """Check fixture `s1` tree"""
    get_name = lambda system: system.name
    assert list(s1.children) == ['a', 'b', 'foo']
    assert list(map(get_name, s1.tree())) == [
        'aun', 'a', 'b', 'beq', 'foo', 's1',
    ]


def test_MathematicalProblem_repr_1(s1: System, caplog):
    """Test off-design unknown and equation assembly in system `s1`"""
    solver = s1.add_driver(NonLinearSolver('solver'))

    with caplog.at_level(logging.INFO):
        for driver in solver.tree():
            driver.setup_run()
    
    problem = solver.problem
    assert problem is not None
    assert isinstance(problem, MathematicalProblem)
    assert len(caplog.records) == 0
    assert set(problem.residues) == {
        "foo.beq: v[::2] == [0, 1]",
    }
    assert set(problem.unknowns) == {
        "a.p.x",
    }
    pattern = "\n".join([
        r"Unknowns \[1\]",
        r"  a\.p\.x = .*",
        r"Equations \[2\]",
        r"  foo\.beq: v\[::2\] == \[0, 1\] := \[.* .*\]",
    ])
    assert re.match(pattern, repr(problem))


def test_MathematicalProblem_repr_2(s1: System, caplog):
    """Same as `test_MathematicalProblem_keys_1`, with additional
    unknowns and equations added at solver level.
    """
    solver = s1.add_driver(NonLinearSolver('solver'))
    solver.add_unknown(['b.width', 'a.aun.p_in.y'])
    solver.add_equation("b.area == 10")

    with caplog.at_level(logging.INFO):
        for driver in solver.tree():
            driver.setup_run()
    
    problem = solver.problem
    assert problem is not None

    # print("", problem, sep="\n")
    assert len(caplog.records) > 0
    assert re.match(
        "Replace unknown 'a.aun.p_in.y' by 'a.p.y'",
        caplog.records[0].message
    )
    # Note: in a single-point problem, all equations
    # and off-design unknowns should be plainly displayed,
    # i.e. no wrapping of the kind "runner[b.area == 10]".
    assert set(problem.residues) == {
        "foo.beq: v[::2] == [0, 1]",
        "b.area == 10",
    }
    assert set(problem.unknowns) == {
        "a.p.x",
        "a.p.y",
        "b.width",
    }
    pattern = "\n".join([
        r"Unknowns \[3\]",
        r"  b\.width = .*",
        r"  a\.p\.y = .*",
        r"  a\.p\.x = .*",
        r"Equations \[3\]",
        r"  b\.area == 10 := .*",
        r"  foo\.beq: v\[::2\] == \[0, 1\] := \[.* .*\]",
    ])
    # print(pattern, problem, sep="\n")
    assert re.match(pattern, repr(problem))


def test_MathematicalProblem_repr_3(s1: System, caplog):
    """Check unknowns and equations in a multi-point solver.
    """
    solver = s1.add_driver(NonLinearSolver('solver'))
    point1 = solver.add_driver(RunSingleCase('point1'))
    point2 = solver.add_driver(RunSingleCase('point2'))

    solver.add_unknown(['b.width', 'b.length'])
    solver.add_equation("q_out.a == x + y")
    point1.add_unknown('a.aun.u[-1]').add_equation("a.z == 0")
    point2.design.add_unknown('foo.beq.p_in.y').add_equation("q_out.b == 0")

    with caplog.at_level(logging.INFO):
        for driver in solver.tree():
            driver.setup_run()
    
    assert len(caplog.records) > 0
    assert re.match(
        "Replace unknown 'foo.beq.p_in.y' by 'foo.xyz_in.y'",
        caplog.records[0].message
    )
    problem = solver.problem
    assert problem is not None
    # print("", problem, sep="\n")

    assert set(problem.residues) == {
        "point1[foo.beq: v[::2] == [0, 1]]",
        "point2[foo.beq: v[::2] == [0, 1]]",
        "point1[q_out.a == x + y]",
        "point2[q_out.a == x + y]",
        "point2[q_out.b == 0]",
        "point1[a.z == 0]",
    }
    assert set(problem.unknowns) == {
        # Design unknowns:
        "b.width",
        "b.length",
        "foo.xyz_in.y",
        # Off-design unknowns:
        "point1[a.p.x]",
        "point2[a.p.x]",
        "point1[a.aun.u[-1]]",
    }
    pattern = "\n".join([
        r"Unknowns \[6\]",
        r"  b\.width = .*",
        r"  b\.length = .*",
        r"  point1\[a\.aun\.u\[-1\]\] = \[.*\]",
        r"  point1\[a\.p\.x\] = .*",
        r"  foo\.xyz_in\.y = .*",
        r"  point2\[a\.p\.x\] = .*",
        r"Equations \[8\]",
        r"  point1\[a\.z == 0\] := .*",
        r"  point1\[foo\.beq: v\[::2\] == \[0, 1\]\] := \[.* .*\]",
        r"  point1\[q_out\.a == x \+ y\] := .*",
        r"  point2\[q_out\.b == 0\] := .*",
        r"  point2\[foo\.beq: v\[::2\] == \[0, 1\]\] := \[.* .*\]",
        r"  point2\[q_out\.a == x \+ y\] := .*",
    ])
    # print(pattern, problem, sep="\n")
    assert re.match(pattern, repr(problem))


def test_MathematicalProblem_repr_4(s2: System):
    """Test representation of a math problem with targets.
    """
    problem = s2.assembled_problem()
    assert repr(problem) == "\n".join([
        "Equations [5]",
        "  sub.z == 0.12 (target)",
        "  out.a == 3.14 (target)",
        "  abs(out.b) == 2.3 (target)",
        "  out.c[::2] == array([0.1, 0.3]) (target)",
    ])


def test_MathematicalProblem_repr_5(s2: System):
    """Test math problem representation.
    Case: single-point design.
    """
    solver = s2.add_driver(NonLinearSolver('solver'))
    for driver in solver.tree():
        driver.setup_run()
    problem = solver.problem
    assert repr(problem) == "\n".join([
        "Equations [5]",
        "  sub.z == 0.12 := 0.0",
        "  out.a == 3.14 := 0.0",
        "  abs(out.b) == 2.3 := 0.0",
        "  out.c[::2] == array([0.1, 0.3]) := [0. 0.]",
    ])


def test_MathematicalProblem_repr_6(s2: System):
    """Test math problem representation.
    Case: multi-point design with targets.
    """
    solver = s2.add_driver(NonLinearSolver('solver'))
    point1 = solver.add_driver(RunSingleCase('point1'))
    point2 = solver.add_driver(RunSingleCase('point2'))

    point1.set_init({
        'sub.z': 0.12,
        'out.a': 3.14,
        'out.b': -7.3,
        'out.c': np.r_[0.1, 0.2, 0.3, 0.4],
    })
    point2.set_init({
        'sub.z': 0.0,
        'out.a': 0.5,
        'out.b': 0.0,
        'out.c': np.ones(4),
    })

    for driver in solver.tree():
        driver.setup_run()

    # print(solver.problem)
    assert repr(solver.problem) == "\n".join([
        "Equations [10]",
        #  point1 equations
        "  point1[sub.z == 0.12] := 0.0",
        "  point1[out.a == 3.14] := 0.0",
        "  point1[abs(out.b) == 7.3] := 0.0",
        "  point1[out.c[::2] == array([0.1, 0.3])] := [0. 0.]",
        #  point2 equations
        "  point2[sub.z == 0.0] := 0.0",
        "  point2[out.a == 0.5] := 0.0",
        "  point2[abs(out.b) == 0.0] := 0.0",
        "  point2[out.c[::2] == array([1., 1.])] := [0. 0.]",
    ])
