import pytest
import numpy as np
import logging, re
from typing import Type

from cosapp.base import System, Port
from cosapp.core import MathematicalProblem
from cosapp.drivers import NonLinearSolver, RunSingleCase
from cosapp.utils.testing import get_args


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
def A_factory(DummySystemFactory) -> Type[System]:
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
def B_factory(DummySystemFactory) -> Type[System]:
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
        unknowns = get_args("p_in.x"),
    )


@pytest.fixture
def B(B_factory) -> Type[System]:
    return B_factory("B")


@pytest.fixture
def Beq(B_factory) -> Type[System]:
    return B_factory("Beq",
        equations = get_args("v[::2] == [0, 1]"),
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


def test_s1(s1):
    """Check fixture `s1` tree"""
    get_name = lambda system: system.name
    assert list(s1.children) == ['a', 'b', 'foo']
    assert list(map(get_name, s1.tree())) == [
        'aun', 'a', 'b', 'beq', 'foo', 's1',
    ]


def test_MathematicalProblem_keys_1(s1: System, caplog):
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


def test_MathematicalProblem_keys_2(s1: System, caplog):
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


def test_MathematicalProblem_keys_3(s1: System, caplog):
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
