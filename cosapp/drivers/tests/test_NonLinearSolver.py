import pytest
import logging, re
import numpy as np

from collections import OrderedDict
from unittest.mock import MagicMock, patch, DEFAULT
from contextlib import nullcontext as does_not_raise

from cosapp.core.numerics.residues import Residue
from cosapp.core.numerics.basics import MathematicalProblem
from cosapp.drivers import (
    Driver,
    NonLinearMethods,
    NonLinearSolver,
    RunOnce,
    RunSingleCase,
)
from cosapp.ports import Port
from cosapp.systems import System
from cosapp.utils.logging import LogFormat, LogLevel
from cosapp.tests.library.systems import (
    Merger1d,
    Multiply2,
    MultiplyVector2,
    Splitter1d,
    Strait1dLine,
)


@pytest.fixture
def FixedPointArray():
    """Factory creating a cyclic assembly of
    two systems `a` and `b` exchanging numpy arrays.

    Solution for each component of `a.x.value` is
    0.5 * (3 +/- np.sqrt(5)), depending on initial guess.
    """
    class Port3D(Port):
        def setup(self):
            self.add_variable('value', np.r_[-0.2, 1.1, 5.2])

    class SystemA(System):
        def setup(self):
            self.add_input(Port3D, 'x')
            self.add_output(Port3D, 'y')
        
        def compute(self):
            self.y.value = 1 - self.x.value

    class SystemB(System):
        def setup(self):
            self.add_input(Port3D, 'u')
            self.add_output(Port3D, 'v')
        
        def compute(self):
            self.v.value = self.u.value**2

    def factory(name):
        top = System(name)
        a = top.add_child(SystemA('a'))
        b = top.add_child(SystemB('b'))
        top.connect(a.x, b.v)
        top.connect(a.y, b.u)
        return top

    return factory


class QuadraticFunction(System):
    """Simple function y = k * x^2 - a
    """
    def setup(self):
        self.add_inward('a', 0.0)
        self.add_inward('k', 1.0)
        self.add_inward('x', 1.0)
        self.add_outward('y', 0.0)

        design = self.add_design_method('y')
        design.add_unknown('x').add_target('y')
    
    def compute(self) -> None:
        self.y = self.k * self.x**2 - self.a


class AbcdFunction(System):
    """Simple function x -> (a, b, c, d)
    used in tests on singular problems.
    """
    def setup(self):
        self.add_inward('x', np.ones(4))
        self.add_outward('a', 0.0)
        self.add_outward('b', 0.0)
        self.add_outward('c', 0.0)
        self.add_outward('d', 0.0)

    def compute(self) -> None:
        x = self.x
        self.a = x[0] + x[1]
        self.b = x[3]
        self.c = x[3]**2
        self.d = x[0] + x[3]


@pytest.mark.parametrize(
    "method", NonLinearMethods
)
def test_NonLinearSolver_method(method):
    system = Multiply2("s")
    solver = system.add_driver(NonLinearSolver("solver", method=method))

    assert solver.owner is system
    assert solver.method is method
    assert len(solver.children) == 0


def test_NonLinearSolver__setattr__():
    # Error is raised when setting an absent attribute
    d = NonLinearSolver("driver")
    with pytest.raises(AttributeError):
        d.ftol = 1e-5


def test_NonLinearSolver_add_child():
    d = NonLinearSolver("driver")
    d.compute_jacobian = False
    assert len(d.children) == 0
    assert d.compute_jacobian == False

    subdriver_name = "subdriver"
    sub_driver = d.add_child(Driver(subdriver_name))
    assert set(d.children) == {subdriver_name}
    assert d.children[subdriver_name] is sub_driver
    assert d.compute_jacobian == True


def test_NonLinearSolver_is_standalone():
    d = NonLinearSolver("driver")
    assert d.is_standalone()


def test_NonLinearSolver__fresidues(set_master_system):

    # Simple system with no design equations
    system = Multiply2("MyMult")
    solver = system.add_driver(NonLinearSolver("solver"))
    solver.add_unknown(["K1", "K2", "p_in.x"])
    system.call_setup_run()
    solver._precompute()

    init = np.random.rand(len(solver.problem.unknowns))
    residues = solver._fresidues(init)
    set_init = [var.default_value for var in solver.problem.unknowns.values()]
    assert set_init == list(init)
    mask_unknowns = [var.mask for var in solver.problem.unknowns.values()]
    assert mask_unknowns == list((None, None, None))
    assert isinstance(residues, np.ndarray)
    assert len(residues) == 0

    # More realistic single case
    system = Multiply2("MyMult")
    solver = system.add_driver(NonLinearSolver("solver"))
    runner = solver.add_child(RunSingleCase("runner"))
    solver.add_unknown("p_in.x").add_equation("p_in.x == 2")  # design unknown, off-design eq.
    runner.design.add_unknown("K1").add_equation("p_out.x == 20")
    system.call_setup_run()
    solver._precompute()

    init = np.random.rand(len(solver.problem.unknowns))
    residues = solver._fresidues(init)
    set_init = [var.default_value for var in solver.problem.unknowns.values()]
    assert set_init == list(init)
    mask_unknowns = [var.mask for var in solver.problem.unknowns.values()]
    assert mask_unknowns == [None, None]
    assert isinstance(residues, np.ndarray)
    assert len(residues) == 2

    # Multiple cases with off-design and design problems
    system = Multiply2("MyMult")
    solver = system.add_driver(NonLinearSolver("solver"))
    # Design points
    point1 = solver.add_child(RunSingleCase("point1"))
    point2 = solver.add_child(RunSingleCase("point2"))

    solver.add_unknown("p_in.x").add_equation("p_in.x == K2")  # design unknown, off-design eq.
    point1.design.add_unknown("K1").add_equation("p_out.x == 20")
    point2.design.add_unknown("K2").add_equation("K2 == 2 * K1")

    system.call_setup_run()
    solver._precompute()

    init = np.random.rand(len(solver.problem.unknowns))
    residues = solver._fresidues(init)
    variables = [var.default_value for var in solver.problem.unknowns.values()]
    assert variables == list(init)
    mask_unknowns = [var.mask for var in solver.problem.unknowns.values()]
    assert mask_unknowns == [None, None, None]
    assert isinstance(residues, np.ndarray)
    assert len(residues) == 4
    assert solver.problem.n_unknowns == 3
    assert solver.problem.n_equations == 4

    # Multiple cases with off-design and design problems (2)
    system = Multiply2("MyMult")
    solver = system.add_driver(NonLinearSolver("solver"))
    # Design points
    point1 = solver.add_child(RunSingleCase("point1"))
    point2 = solver.add_child(RunSingleCase("point2"))

    solver.add_unknown(["K1", "K2"])  # design unknowns
    # Local problems:
    point1.add_unknown("p_in.x").add_equation("p_in.x == 2")
    point2.add_unknown("p_in.x").add_equation("p_in.x**2 == 6")
    point1.add_equation("p_out.x == 20")
    point2.add_equation("K2 == 2 * K1")

    system.call_setup_run()
    solver._precompute()

    init = np.random.rand(len(solver.problem.unknowns))
    residues = solver._fresidues(init)
    variables = [var.default_value for var in solver.problem.unknowns.values()]
    assert variables == list(init)
    mask_unknowns = [var.mask for var in solver.problem.unknowns.values()]
    assert mask_unknowns == [None, None, None, None]
    assert isinstance(residues, np.ndarray)
    assert len(residues) == 4
    assert solver.problem.n_unknowns == 4
    assert solver.problem.n_equations == 4

    # Multiple cases, with design and local problems
    system = Multiply2("MyMult")
    solver = system.add_driver(NonLinearSolver("solver"))
    point1 = solver.add_child(RunSingleCase("point1"))
    point2 = solver.add_child(RunSingleCase("point2"))

    # Design constraints:
    point1.design.add_unknown("K1").add_equation("p_out.x == 20")
    point2.design.add_unknown("K2").add_equation("p_out.x == 15")
    # Local conditions:
    point1.add_unknown("p_in.x").add_equation("p_in.x == 2")
    point2.add_unknown("p_in.x").add_equation("p_in.x**2 == 6")

    system.call_setup_run()
    solver._precompute()

    init = np.random.rand(len(solver.problem.unknowns))
    residues = solver._fresidues(init)
    variables = [var.default_value for var in solver.problem.unknowns.values()]
    assert variables == list(init)
    mask_unknowns = [var.mask for var in solver.problem.unknowns.values()]
    assert mask_unknowns == [None, None, None, None]
    assert isinstance(residues, np.ndarray)
    assert len(residues) == 4
    assert solver.problem.n_unknowns == 4
    assert solver.problem.n_equations == 4


def test_NonLinearSolver_setup_run(caplog, set_master_system):
    # Simple system with no design equations
    system = Multiply2("MyMult")
    solver = NonLinearSolver("solver")
    system.add_driver(solver)
    solver.add_unknown(["K1", "K2", "p_in.x"])
    system.call_setup_run()
    solver._precompute()
    assert set(solver.problem.unknowns) == {
        "K1", "K2", "p_in.x",
    }
    assert set(solver.problem.residues) == set()

    # More realistic single case
    system = Multiply2("MyMult")
    solver = system.add_driver(NonLinearSolver("solver"))
    runner = solver.add_child(RunSingleCase("runner"))
    runner.offdesign.add_unknown("p_in.x").add_equation("p_in.x == 2")
    runner.design.add_unknown("K1").add_equation("p_out.x == 20")
    init = np.random.rand(2)
    runner.set_init({"K1": init[0], "p_in.x": init[1]})
    assert set(runner.initial_values) == {"K1", "p_in.x"}
    assert runner.initial_values["K1"].default_value == init[0]
    assert runner.initial_values["p_in.x"].default_value == init[1]
    system.call_setup_run()
    solver._precompute()
    assert set(solver.problem.unknowns) == {
        "K1", "p_in.x",
    }
    assert set(solver.problem.residues) == {
        "p_out.x == 20",
        "p_in.x == 2",
    }
    assert list(solver.initial_values) == list(init)
    assert runner.initial_values["K1"].default_value == init[0]
    assert runner.initial_values["p_in.x"].default_value == init[1]

    system.K1 = 0.8
    system.call_setup_run()
    assert list(solver.initial_values) == list(init)
    solver.runner.solution["K1"] = 0.8
    system.call_setup_run()
    assert list(solver.initial_values) == [0.8, init[1]]

    solver.force_init = True
    system.call_setup_run()
    assert list(solver.initial_values) == list(init)

    # Multiple cases, with off-design and design problems
    # defined at solver and case level.
    system = Multiply2("MyMult")
    solver = system.add_driver(NonLinearSolver("solver"))
    point1 = solver.add_child(RunSingleCase("point1"))
    point2 = solver.add_child(RunSingleCase("point2"))
    
    solver.add_unknown(["K1", "K2"]).add_equation("p_out.x == 20")
    point1.add_unknown("p_in.x").add_equation("p_in.x == 2")     # local
    point2.add_unknown("p_in.x").add_equation("p_in.x**2 == 6")  # local

    init = np.random.rand(4)
    point1.set_init({"K1": init[0], "p_in.x": init[1]})
    point2.set_init({"K2": init[2], "p_in.x": init[3]})
    # Initial values for K1 and K2 come first, as they
    # are declared as design unknowns at solver level
    expected_init = [
        init[0], init[2],  # initial values for K1, K2
        init[1], init[3],
    ]

    system.call_setup_run()
    solver._precompute()

    assert list(solver.problem.unknowns) == [
        "K1", "K2",  # design unknowns
        "point1[p_in.x]", "point2[p_in.x]",
    ]
    assert set(solver.problem.residues) == {
        "point1[p_out.x == 20]",
        "point2[p_out.x == 20]",
        "point1[p_in.x == 2]",
        "point2[p_in.x**2 == 6]",
    }
    assert list(solver.initial_values) == expected_init

    system.K1 = 0.123
    system.call_setup_run()
    assert list(solver.initial_values) == [0.123] + expected_init[1:]

    system.K1 = expected_init[0]
    solver.point1.solution["K1"] = 0.123
    system.call_setup_run()
    assert list(solver.initial_values) == expected_init

    # Multiple cases, with design and local problems
    system = Multiply2("MyMult")
    solver = system.add_driver(NonLinearSolver("solver"))
    point1 = solver.add_child(RunSingleCase("point1"))
    point2 = solver.add_child(RunSingleCase("point2"))

    point1.design.add_unknown("K1").add_equation("p_out.x == 20")  # design (global)
    point2.design.add_unknown("K2").add_equation("p_out.x == 15")  # design (global)
    point1.add_unknown("p_in.x").add_equation("p_in.x == 2")
    point2.add_unknown("p_in.x").add_equation("p_in.x**2 == 6")

    init = np.random.rand(4)
    point1.set_init({"K1": init[0], "p_in.x": init[1]})
    point2.set_init({"K2": init[2], "p_in.x": init[3]})

    system.call_setup_run()

    assert set(solver.problem.unknowns) == {
        "K1", "point1[p_in.x]",
        "K2", "point2[p_in.x]",
    }
    assert list(solver.initial_values) == list(init)

    system.K1 = 0.2
    system.call_setup_run()
    assert list(solver.initial_values) == list(init)
    solver.point1.solution["K1"] = 0.2
    system.call_setup_run()
    assert list(solver.initial_values) == [0.2, init[1], init[2], init[3]]

    # Children without iterative
    system = Multiply2("MyMult")
    solver = NonLinearSolver("solver")
    system.add_driver(solver)
    point1 = solver.add_child(RunSingleCase("point1"))
    point2 = solver.add_child(RunOnce("point2"))

    solver.add_unknown("p_in.x").add_equation("p_in.x == 2")
    point1.add_unknown("K1").add_equation("p_out.x == 20")
    caplog.clear()
    with caplog.at_level(logging.WARNING, NonLinearSolver.__name__):
        system.call_setup_run()
        solver._precompute()

    warning_messages = list(
        map(
            lambda r: r.msg,
            filter(lambda r: r.levelno == logging.WARNING, caplog.records),
        )
    )
    assert len(warning_messages) == 0


def test_NonLinearSolver_compute_log_validation(caplog, set_master_system):
    # Simple system with no design equations
    system = Multiply2("MyMult")
    solver = system.add_driver(NonLinearSolver("solver"))
    solver.add_unknown(["p_in.x", "K1", "K2"])
    system.call_setup_run()
    solver._precompute()

    caplog.clear()
    with caplog.at_level(logging.ERROR):
        with pytest.raises(
            ArithmeticError,
            match=r"Nonlinear problem \w+ error: Mismatch between numbers of params \[\d\] and residues \[\d\]",
        ):
            solver.compute()
    error_messages = list(
        map(
            lambda r: r.msg,
            filter(lambda r: r.levelno == logging.ERROR, caplog.records),
        )
    )
    assert len(error_messages) == 3
    assert (
        re.search(
            r"Nonlinear problem \w+ error: Mismatch between numbers of params \[\d\] and residues \[\d\]",
            error_messages[0],
        )
        is not None
    )
    assert re.search(r"Residues: .*", error_messages[1]) is not None
    assert re.search(r"Variables: .*", error_messages[2]) is not None

    # More realistic single case
    system = Multiply2("MyMult")
    # Set max_iter to force failure
    solver = system.add_driver(NonLinearSolver("solver", method=NonLinearMethods.NR, max_iter=0))
    runner = solver.add_child(RunSingleCase("runner"))
    solver.add_unknown("K1").add_equation("p_out.x == 20")
    runner.add_unknown("p_in.x").add_equation("p_in.x == 2")
    init = np.random.rand(2)
    runner.set_init({"K1": init[0], "p_in.x": init[1]})
    system.call_setup_run()
    solver._precompute()
    caplog.clear()
    with caplog.at_level(logging.ERROR):
        solver.compute()

    error_messages = list(
        map(
            lambda r: r.msg,
            filter(lambda r: r.levelno == logging.ERROR, caplog.records),
        )
    )
    assert len(error_messages) == 1
    assert re.search(r"The solver failed: .*", error_messages[0]) is not None

    assert solver.solution == dict(zip(solver.problem.unknowns, init))

    # Multipoint cases
    system = Multiply2("MyMult")
    solver = system.add_driver(NonLinearSolver("solver", method=NonLinearMethods.NR, max_iter=0))
    point1 = solver.add_child(RunSingleCase("point1"))
    point2 = solver.add_child(RunSingleCase("point2"))

    solver.add_unknown(["K1", "K2"]).add_equation("p_in.x == 2")
    point1.add_unknown("p_in.x").add_equation("p_out.x == 20")
    point2.add_unknown("p_in.x").add_equation("p_out.x == 15")

    init = np.random.rand(4)
    point1.set_init({"K1": init[0], "p_in.x": init[1]})
    point2.set_init({"K2": init[2], "p_in.x": init[3]})
    expected_init = [init[i] for i in [0, 2, 1, 3]]

    system.call_setup_run()
    solver._precompute()
    caplog.clear()
    with caplog.at_level(logging.ERROR):
        solver.compute()

    error_messages = list(
        map(
            lambda r: r.msg,
            filter(lambda r: r.levelno == logging.ERROR, caplog.records),
        )
    )
    assert len(error_messages) == 1
    assert re.search(r"The solver failed: .*", error_messages[0]) is not None
    assert solver.solution == dict(zip(solver.problem.unknowns, expected_init))

    system = Multiply2("MyMult")
    solver = system.add_driver(NonLinearSolver("solver", method=NonLinearMethods.NR, max_iter=0))
    runner = solver.add_child(RunSingleCase("runner"))
    solver.add_unknown("K1").add_equation("p_out.x == 20")
    runner.add_unknown("p_in.x").add_equation("p_in.x == 2")
    init = np.random.rand(2)
    runner.set_init({"K1": init[0], "p_in.x": init[1]})
    expected_init = init

    system.call_setup_run()
    solver._precompute()
    caplog.clear()
    with caplog.at_level(logging.ERROR):
        solver.compute()

    error_messages = list(filter(lambda r: r.levelno == logging.ERROR, caplog.records))
    assert len(error_messages) == 1
    assert solver.solution == dict(zip(solver.problem.unknowns, expected_init))


def test_NonLinearSolver_compute_empty(caplog, set_master_system):
    # Child without iterative
    system = System("empty")
    solver = system.add_driver(NonLinearSolver("solver"))
    system.call_setup_run()
    solver._precompute()

    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        solver.compute()

    debug_messages = list(
        map(
            lambda r: r.msg,
            filter(lambda r: r.levelno == logging.DEBUG, caplog.records),
        )
    )
    assert len(debug_messages) > 0
    assert (
        re.search(
            r"No parameters/residues to solve. Fallback to children execution\.",
            debug_messages[0],
        )
        is not None
    )


def test_NonLinearSolver_compute(caplog, set_master_system):
    # Single case
    system = Multiply2("MyMult")
    solver = system.add_driver(NonLinearSolver("solver", method=NonLinearMethods.POWELL))
    solver.add_unknown("p_in.x").add_equation("p_in.x == 2")
    solver.add_child(RunSingleCase("case"))
    solver.case.add_unknown("K1").add_equation("p_out.x == 20")
    init = np.random.rand(2)
    solver.case.set_init({"K1": init[0], "p_in.x": init[1]})
    system.call_setup_run()
    solver._precompute()
    caplog.clear()
    with caplog.at_level(logging.INFO):
        solver.compute()

    info_messages = list(record.msg
        for record in filter(lambda r: r.levelno == logging.INFO, caplog.records)
    )
    assert len(info_messages) == 1
    assert re.search(r"solver : \w+The solution converged\.", info_messages[0]) is not None

    assert len(solver.solution) == len(init)
    np.testing.assert_allclose(list(solver.solution.values()), (2.0, 2.0))


def test_NonLinearSolver__postcompute(set_master_system):
    # Simple system with no design equations
    system = Multiply2("MyMult")
    solver = NonLinearSolver("solver")
    system.add_driver(solver)
    solver.add_unknown(["K1", "K2", "p_in.x"])
    system.call_setup_run()
    solver._precompute()
    solver._postcompute()
    assert set(solver.problem.unknowns) == {
        "K1", "K2", "p_in.x",
    }
    assert set(solver.problem.residues) == set()

    # More realistic single case
    system = Multiply2("MyMult")
    solver = system.add_driver(NonLinearSolver("solver"))
    runner = solver.add_child(RunSingleCase("runner"))
    solver.add_unknown("K1").add_equation("p_out.x == 20")
    runner.add_unknown("p_in.x").add_equation("p_in.x == 2")
    system.call_setup_run()
    solver._precompute()
    solver._postcompute()
    assert set(solver.problem.unknowns) == {
        "K1", "p_in.x",
    }
    assert set(solver.problem.residues) == {
        "p_out.x == 20",
        "p_in.x == 2",
    }

    # Multiple cases
    system = Multiply2("MyMult")
    solver = system.add_driver(NonLinearSolver("solver"))
    point1 = solver.add_child(RunSingleCase("point1"))
    point2 = solver.add_child(RunSingleCase("point2"))

    solver.add_unknown(["K1", "K2"]).add_equation("p_out.x == 20")
    point1.add_unknown("p_in.x").add_equation("p_in.x == 2")
    point2.add_unknown("p_in.x").add_equation("p_in.x**2 == 6")

    system.call_setup_run()
    solver._precompute()
    solver._postcompute()
    assert set(solver.problem.unknowns) == {
        "K1", "point1[p_in.x]",
        "K2", "point2[p_in.x]",
    }
    assert set(solver.problem.residues) == {
        "point1[p_out.x == 20]", "point1[p_in.x == 2]",
        "point2[p_out.x == 20]", "point2[p_in.x**2 == 6]",
    }

    # Multiple cases, with local problems
    system = Multiply2("MyMult")
    solver = system.add_driver(NonLinearSolver("solver"))
    point1 = solver.add_child(RunSingleCase("point1"))
    point2 = solver.add_child(RunSingleCase("point2"))

    point1.design.add_unknown("K1").add_equation("p_out.x == 20")
    point2.design.add_unknown("K2").add_equation("p_out.x == 15")

    point1.add_unknown("p_in.x").add_equation("p_in.x == 2")
    point2.add_unknown("p_in.x").add_equation("p_in.x**2 == 6")

    system.call_setup_run()
    solver._precompute()
    solver._postcompute()
    assert set(solver.problem.unknowns) == {
        "K1", "point1[p_in.x]",
        "K2", "point2[p_in.x]",
    }
    assert set(solver.problem.residues) == {
        "point1[p_out.x == 20]", "point1[p_in.x == 2]",
        "point2[p_out.x == 15]", "point2[p_in.x**2 == 6]",
    }


def test_NonLinearSolver_residues_singlePoint(set_master_system):
    # Simple system with no design equations
    system = Multiply2("MyMult")
    solver = system.add_driver(NonLinearSolver("solver"))
    solver.add_unknown(["K1", "K2"])
    system.call_setup_run()
    solver._precompute()
    init = np.random.rand(len(solver.problem.unknowns))
    solver._fresidues(init)
    assert isinstance(solver.problem.residues, OrderedDict)
    assert len(solver.problem.residues) == 0

    # More realistic single case
    system = Multiply2("MyMult")
    solver = system.add_driver(NonLinearSolver("solver"))
    runner = solver.add_child(RunSingleCase("runner"))
    solver.add_unknown("K1").add_equation("p_out.x == 20")
    runner.add_unknown("p_in.x").add_equation("p_in.x == 2")
    solver.owner.call_setup_run()
    solver._precompute()
    solver.compute()
    assert set(solver.problem.residues) == {
        "p_in.x == 2",
        "p_out.x == 20",
    }
    for v in solver.problem.residues.values():
        assert isinstance(v, Residue)

    # Same, with runner-level problems only
    system = Multiply2("MyMult")
    solver = system.add_driver(NonLinearSolver("solver"))
    runner = solver.add_child(RunSingleCase("runner"))
    runner.design.add_unknown("K1").add_equation("p_out.x == 20")
    runner.add_unknown("p_in.x").add_equation("p_in.x == 2")
    solver.owner.call_setup_run()
    solver._precompute()
    solver.compute()
    assert set(solver.problem.residues) == {
        "p_in.x == 2",
        "p_out.x == 20",
    }
    for v in solver.problem.residues.values():
        assert isinstance(v, Residue)


def test_NonLinearSolver_problem_multiPoint(set_master_system):
    """Test combinations of design, local and off-design problems.
    Problems are not solved (some are ill-posed); only unknowns and
    equations are checked.
    """
    def make_case():
        system = Multiply2("MyMult")
        # Add solver and design points
        solver = system.add_driver(NonLinearSolver("solver", max_iter=50))
        point1 = solver.add_child(RunSingleCase("point1"))
        point2 = solver.add_child(RunSingleCase("point2"))
        return solver, point1, point2

    solver, point1, point2 = make_case()
    solver.add_unknown(["K1", "K2"]).add_equation("p_in.x == 2")
    point1.add_unknown("p_in.x").add_equation("p_out.x == 20")  # local
    point2.add_unknown("p_in.x").add_equation("Ksum == 10")     # local

    solver.owner.call_setup_run()
    solver._precompute()
    assert set(solver.problem.unknowns) == {
        "K1",
        "K2",
        "point1[p_in.x]",
        "point2[p_in.x]",
    }
    assert set(solver.problem.residues) == {
        "{}[{}]".format("point1", "p_out.x == 20"),
        "{}[{}]".format("point2", "Ksum == 10"),
        "{}[{}]".format("point1", "p_in.x == 2"),
        "{}[{}]".format("point2", "p_in.x == 2"),
    }
    for v in solver.problem.residues.values():
        assert isinstance(v, Residue)

    # Same, with local problems
    solver, point1, point2 = make_case()

    point1.design.add_unknown("K1").add_equation("p_out.x == 20")
    point2.add_unknown("K2").add_equation("Ksum == 10")
    point1.add_unknown("p_in.x").add_equation("p_in.x == 2")
    point2.add_unknown("p_in.x").add_equation("p_in.x**2 == 6")
    solver.owner.call_setup_run()
    solver._precompute()
    assert set(solver.problem.unknowns) == {
        "K1",
        "point2[K2]",
        "point1[p_in.x]",
        "point2[p_in.x]",
    }
    assert set(solver.problem.residues) == {
        "{}[{}]".format("point1", "p_out.x == 20"),
        "{}[{}]".format("point2", "Ksum == 10"),
        "{}[{}]".format("point1", "p_in.x == 2"),
        "{}[{}]".format("point2", "p_in.x**2 == 6"),
    }
    for v in solver.problem.residues.values():
        assert isinstance(v, Residue)

    # Same, combining off-design, design and local problems
    solver, point1, point2 = make_case()

    solver.add_unknown("K1")  # common design unknown
    point1.add_equation("p_out.x == 20")  # design constraint on point1
    point1.add_unknown("p_in.x").add_equation("p_in.x == 2")
    point2.add_unknown("p_in.x").add_equation("p_in.x**2 == 6")
    solver.owner.call_setup_run()
    solver._precompute()
    assert set(solver.problem.unknowns) == {
        "K1",
        "point1[p_in.x]",
        "point2[p_in.x]",
    }
    assert set(solver.problem.residues) == {
        "{}[{}]".format("point1", "p_out.x == 20"),
        "{}[{}]".format("point1", "p_in.x == 2"),
        "{}[{}]".format("point2", "p_in.x**2 == 6"),
    }
    for v in solver.problem.residues.values():
        assert isinstance(v, Residue)


def test_NonLinearSolver_solve_multiPoint():
    def make_case():
        system = Multiply2("s")
        system.p_in.x = 1
        # Add solver and design points
        solver = system.add_driver(NonLinearSolver("solver", max_iter=100))
        point1 = solver.add_child(RunSingleCase("point1"))
        point2 = solver.add_child(RunSingleCase("point2"))
        return system, solver, point1, point2

    s = Multiply2("s")
    solver = s.add_driver(NonLinearSolver("solver", max_iter=50))
    solver.add_unknown("K1").add_equation("p_out.x == 6")
    solver.add_unknown("K2").add_equation("Ksum == 8")

    s.p_in.x = 0.5
    s.K1 = 0.5
    s.K2 = 3.1

    s.run_drivers()
    assert s.K1 == pytest.approx(2)
    assert s.K2 == pytest.approx(6)

    s, solver, point1, point2 = make_case()
    # Define design unknowns and off-design equation
    solver.add_unknown(["K1", "K2"]).add_equation("2 * p_in.x == 1")
    # Define design point local problems
    point1.add_unknown("p_in.x").add_equation("p_out.x == 6")
    point2.add_unknown("p_in.x").add_equation("Ksum == 8")

    s.K1 = 0.5
    s.K2 = 3.1

    s.run_drivers()
    assert s.K1 == pytest.approx(2)
    assert s.K2 == pytest.approx(6)

    # Design problem with different values of `s.p_in.x`
    s, solver, point1, point2 = make_case()
    point1.design.add_unknown("K1").add_equation("p_out.x == 6")
    point2.design.add_unknown("K2").add_equation("p_out.x == 3 * Ksum")

    point1.set_values({"p_in.x": 0.5})
    point2.set_values({"p_in.x": 2.0})
    s.K1 = 0.5
    s.K2 = 3.1

    s.run_drivers()
    assert s.K1 == pytest.approx(2)
    assert s.K2 == pytest.approx(6)

    # Same, with design unknowns declared at solver level,
    # and equations declared in design points
    s, solver, point1, point2 = make_case()
    
    solver.add_unknown(["K1", "K2"])
    point1.add_equation("p_out.x == 6")
    point2.add_equation("p_out.x == 3 * Ksum")

    point1.set_values({"p_in.x": 0.5})
    point2.set_values({"p_in.x": 2.0})
    s.K1 = 0.5
    s.K2 = 3.1

    s.run_drivers()
    assert s.K1 == pytest.approx(2)
    assert s.K2 == pytest.approx(6)

    # Same, with local problems
    s, solver, point1, point2 = make_case()
    point1.design.add_unknown("K1").add_equation("p_out.x == 6")
    point2.design.add_unknown("K2").add_equation("p_out.x == 3 * Ksum")
    point1.add_unknown("p_in.x").add_equation("2 * p_in.x == 1")
    point2.add_unknown("p_in.x").add_equation("p_in.x**3 == 8")

    s.K1 = 0.5
    s.K2 = 5.1

    s.run_drivers()
    assert s.K1 == pytest.approx(2)
    assert s.K2 == pytest.approx(6)


def test_NonLinearSolver_multiPoint_error():
    """Test error caused by declaring variables as design and of-design unknowns"""
    def make_case():
        system = Multiply2("s")
        system.p_in.x = 1
        # Add solver and design points
        solver = system.add_driver(NonLinearSolver("solver", max_iter=100))
        point1 = solver.add_child(RunSingleCase("point1"))
        point2 = solver.add_child(RunSingleCase("point2"))
        return system, solver, point1, point2

    # Add local case unknown conflicting with solver unknowns (1)
    s, solver, point1, point2 = make_case()
    
    solver.add_unknown(["K1", "K2"])
    point1.add_equation("p_out.x == 6")
    point2.add_equation("p_out.x == 3 * Ksum")
    point1.add_unknown("K1")  # incompatible with solver design unknowns

    with pytest.raises(
        ValueError,
        match="'K1' is defined as design and off-design unknown in 'point1'"
    ):
        s.run_drivers()

    # Add local case unknown conflicting with solver unknowns (2)
    s, solver, point1, point2 = make_case()
    
    solver.add_unknown(["K1", "K2"])
    point1.add_equation("p_out.x == 6")
    point2.add_equation("p_out.x == 3 * Ksum")
    point2.add_unknown(["K1", "K2"])  # incompatible with solver design unknowns

    with pytest.raises(
        ValueError,
        match="\('K1', 'K2'\) are defined as design and off-design unknowns in 'point2'"
    ):
        s.run_drivers()

    # Add case design unknown conflicting with solver unknowns
    s, solver, point1, point2 = make_case()
    
    solver.add_unknown(["K1", "K2"])  # design unknwons
    point1.add_equation("p_out.x == 6")
    point2.add_equation("p_out.x == 3 * Ksum")
    point2.design.add_unknown("K1")  # redundant design unknown

    with pytest.raises(
        ValueError,
        match="'K1' already exists in 'solver'"
    ):
        s.run_drivers()

    # Conflicting case unknowns (1)
    s, solver, point1, point2 = make_case()
    
    point1.design.add_unknown("K1").add_equation("p_out.x == 6")
    point2.design.add_unknown("K2").add_equation("p_out.x == 3 * Ksum")
    point2.add_unknown("K1")  # incompatible with point1 design unknown

    with pytest.raises(
        ValueError,
        match="'K1' is defined as design and off-design unknown in 'point2'"
    ):
        s.run_drivers()

    # Conflicting case unknowns (2)
    s, solver, point1, point2 = make_case()
    
    point1.design.add_unknown("K1").add_equation("p_out.x == 6")
    point2.design.add_unknown("K2").add_equation("p_out.x == 3 * Ksum")
    point1.add_unknown("K2")  # incompatible with point2 design unknown
    point2.add_unknown("K1")  # incompatible with point1 design unknown

    # Error is raised for point2, after point1 has been parsed
    with pytest.raises(
        ValueError,
        match="'K1' is defined as design and off-design unknown in 'point2'"
    ):
        s.run_drivers()

    # Conflict within a single point
    s, solver, point1, point2 = make_case()
    
    point1.design.add_unknown("K1").add_equation("p_out.x == 6")
    point2.design.add_unknown("K2").add_equation("p_out.x == 3 * Ksum")
    point1.add_unknown("K1")  # incompatible with point1 design unknown

    with pytest.raises(
        ValueError,
        match="'K1' is defined as design and off-design unknown in 'point1'"
    ):
        s.run_drivers()


def test_NonLinearSolver_add_target_solver():
    """Test target definition at solver level"""
    f = QuadraticFunction('f')

    solver = f.add_driver(NonLinearSolver('solver', tol=1e-9))
    solver.add_unknown('x').add_target('y')

    f.k = 1.0
    f.a = 2.0
    f.y = 0.0   # set target value by setting output variable
    f.run_drivers()
    assert f.x == pytest.approx(np.sqrt(2))
    assert f.y == pytest.approx(0)

    f.y = -0.5   # update target value
    f.run_drivers()
    assert f.x == pytest.approx(np.sqrt(1.5))
    assert f.y == pytest.approx(-0.5)


@pytest.mark.parametrize("design", [True, False])
def test_NonLinearSolver_add_target_case(design):
    """Test target definition at `RunSingleCase` driver level"""
    f = QuadraticFunction('f')

    solver = f.add_driver(NonLinearSolver('solver', tol=1e-9))
    case = solver.add_child(RunSingleCase('case'))
    # Use design or off-design problem according to test parameter
    problem = case.design if design else case.offdesign
    problem.add_unknown('x').add_target('y')

    f.k = 1.0
    f.a = 2.0
    f.y = 0.0   # set target value by setting output variable
    f.run_drivers()
    assert f.x == pytest.approx(np.sqrt(2))
    assert f.y == pytest.approx(0)

    f.y = -0.5   # update target value
    f.run_drivers()
    assert f.x == pytest.approx(np.sqrt(1.5))
    assert f.y == pytest.approx(-0.5)


def test_NonLinearSolver_add_target_fix_values():
    """Test multi-point design problem with targets.

    Test purpose:
    -------------
    When target values are specified in `RunSingleCase.set_values`,
    these values are used in the mathematical problem, regardless of
    values set interactively.
    """
    f = QuadraticFunction('f')

    solver = f.add_driver(NonLinearSolver('solver'))
    solver.force_init = True
    case = solver.add_child(RunSingleCase('case'))
    
    case.add_unknown('x').add_target('y')
    case.set_values({
        'k': 0.5,
        'a': 1.2,
        'y': 0.0,  # target value
    })

    f.x = 2.0
    f.y = 1.0  # should not be used as target
    f.run_drivers()

    assert f.y == pytest.approx(0)
    assert f.x == pytest.approx(np.sqrt(2.4))
    # Check that actual RHS is determined by case values
    assert set(solver.problem.residues) == {"y == 0.0"}

    # Re-run with a different initial guess
    f.x = -2.0
    f.y = 1.0  # should not be used as target
    f.run_drivers()

    assert f.y == pytest.approx(0)
    assert f.x == pytest.approx(-np.sqrt(2.4))
    assert set(solver.problem.residues) == {"y == 0.0"}


def test_NonLinearSolver_add_target_multipoint_1():
    """Test multi-point design problem with targets.

    Case 1: targets declared at design point level.
    """
    f = QuadraticFunction('f')

    solver = f.add_driver(NonLinearSolver('solver'))
    point1 = solver.add_child(RunSingleCase('point1'))
    point2 = solver.add_child(RunSingleCase('point2'))
    
    solver.add_unknown(['x', 'a'])

    point1.add_target('y')
    point2.add_target('y')

    point1.set_values({'k': 2.0, 'y': 0.0})
    point2.set_values({'k': -0.5, 'y': -1.0})

    f.run_drivers()
    assert set(solver.problem.residues) == {
        "point1[y == 0.0]",
        "point2[y == -1.0]",
    }
    assert f.a == pytest.approx(0.8)
    assert f.x == pytest.approx(np.sqrt(0.5 * 0.8))

    # Dynamically change case values
    point1.set_values({'k': 2.0, 'y': 0.75})
    point2.set_values({'k': 0.5, 'y': 0.0})

    f.a = 1.0
    f.x = 2.0
    f.run_drivers()
    assert set(solver.problem.residues) == {
        "point1[y == 0.75]",
        "point2[y == 0.0]",
    }
    assert f.a == pytest.approx(0.25)
    assert f.x == pytest.approx(np.sqrt(2 * 0.25))

    # Same with negative initial x
    f.a = 1.0
    f.x = -2.0
    f.run_drivers()
    assert set(solver.problem.residues) == {
        "point1[y == 0.75]",
        "point2[y == 0.0]",
    }
    assert f.a == pytest.approx(0.25)
    assert f.x == pytest.approx(-np.sqrt(2 * 0.25))


def test_NonLinearSolver_add_target_multipoint_2():
    """Test multi-point design problem with targets.

    Case 2: Single target declared at solver level.
    Target is expected to be transmitted to all design points,
    with different target values.
    """
    f = QuadraticFunction('f')

    solver = f.add_driver(NonLinearSolver('solver'))
    point1 = solver.add_child(RunSingleCase('point1'))
    point2 = solver.add_child(RunSingleCase('point2'))
    
    solver.add_unknown(['x', 'a']).add_target('y')

    point1.set_values({'k': 2.0, 'y': 0.0})
    point2.set_values({'k': -0.5, 'y': -1.0})

    f.run_drivers()
    assert set(solver.problem.residues) == {
        "point1[y == 0.0]",
        "point2[y == -1.0]",
    }
    assert f.a == pytest.approx(0.8)
    assert f.x == pytest.approx(np.sqrt(0.5 * 0.8))

    # Dynamically change case values
    point1.set_values({'k': 2.0, 'y': 0.75})
    point2.set_values({'k': 0.5, 'y': 0.0})

    f.a = 1.0
    f.x = 2.0
    f.run_drivers()
    assert set(solver.problem.residues) == {
        "point1[y == 0.75]",
        "point2[y == 0.0]",
    }
    assert f.a == pytest.approx(0.25)
    assert f.x == pytest.approx(np.sqrt(2 * 0.25))

    # Same with negative initial x
    f.a = 1.0
    f.x = -2.0
    f.run_drivers()
    assert set(solver.problem.residues) == {
        "point1[y == 0.75]",
        "point2[y == 0.0]",
    }
    assert f.a == pytest.approx(0.25)
    assert f.x == pytest.approx(-np.sqrt(2 * 0.25))


def test_NonLinearSolver_add_target_multipoint_3():
    """Test multi-point design problem with targets.

    Case 3: specify target values in `set_init`,
    rather than `set_values`.
    """
    f = QuadraticFunction('f')

    solver = f.add_driver(NonLinearSolver('solver'))
    point1 = solver.add_child(RunSingleCase('point1'))
    point2 = solver.add_child(RunSingleCase('point2'))
    
    solver.add_unknown(['x', 'a']).add_target('y')

    point1.set_init({'y': 0.0})
    point2.set_init({'y': -1.0})
    point1.set_values({'k': 2.0})
    point2.set_values({'k': -0.5})

    f.run_drivers()
    assert f.a == pytest.approx(0.8)
    assert f.x == pytest.approx(np.sqrt(0.5 * 0.8))


def test_NonLinearSolver_vector1d_system():
    s = System("hat")
    one = s.add_child(Strait1dLine("one"), pulling="in_")
    two = s.add_child(Strait1dLine("two"), pulling="out")
    s.connect(two.in_, one.out)

    # Test transfer value
    s.in_.x *= 3.0
    assert not np.array_equal(s.in_.x, s.out.x)
    s.run_drivers()
    assert np.array_equal(s.in_.x, s.out.x)

    # Test design and offdesign vector variables
    s.drivers.clear()
    solver = s.add_driver(NonLinearSolver("solver", method=NonLinearMethods.POWELL))
    solver.add_unknown("one.a[1]").add_equation("one.out.x[1] == 42")
    s.run_drivers()
    assert one.a == pytest.approx([1, 7, 1])

    s.drivers.clear()
    solver = s.add_driver(NonLinearSolver("solver", method=NonLinearMethods.POWELL))
    solver.add_unknown(["one.a[1]", "two.a"]).add_equation(
        [
            {"equation": "one.out.x[1] == 9"},
            {"equation": "out.x == array([5., 1., 3.])"},
        ]
    )
    s.run_drivers()
    assert one.a == pytest.approx([1, 1.5, 1])
    assert two.a == pytest.approx([5 / 3, 1 / 9, 1 / 3])

    one.a = np.r_[1.0, 1.0, 1.0]
    two.a = np.r_[1.0, 1.0, 1.0]

    s.drivers.clear()
    solver = s.add_driver(NonLinearSolver("solver", method=NonLinearMethods.POWELL))
    case = solver.add_child(RunSingleCase("case"))
    case.add_unknown("one.a[1]").add_equation("one.out.x[1] == 9")
    solver.add_unknown("two.a").add_equation("out.x == array([5., 1., 3.])")
    s.run_drivers()
    assert one.a == pytest.approx([1.0, 1.5, 1.0])
    assert two.a == pytest.approx([5.0 / 3.0, 1.0 / 9.0, 1.0 / 3.0])

    one.a = np.r_[1.0, 1.0, 1.0]
    two.a = np.r_[1.0, 1.0, 1.0]
    s.drivers.clear()
    solver = s.add_driver(NonLinearSolver("solver", method=NonLinearMethods.POWELL))
    case = solver.add_child(RunSingleCase("case"))
    case.add_unknown(["one.a[1]", "two.a"]).add_equation(
        ["one.out.x[1] == 9", "out.x == array([5., 1., 3.])"]
    )
    s.run_drivers()
    assert one.a == pytest.approx([1.0, 1.5, 1.0])
    assert two.a == pytest.approx([5.0 / 3.0, 1.0 / 9.0, 1.0 / 3.0])

    # Test iterative connector with vector variable
    s = System("hat")
    one = s.add_child(Merger1d("one"), pulling={"in1": "in_"})
    two = s.add_child(Splitter1d("two"), pulling={"out1": "out"})
    s.connect(two.in_, one.out)
    s.connect(one.in2, two.out2)
    s.drivers.clear()
    solver = s.add_driver(NonLinearSolver("solver", tol=1e-7))

    s.run_drivers()
    assert one.out.x == pytest.approx(s.in_.x / two.s)
    assert s.out.x == pytest.approx(s.in_.x)


@patch.object(NonLinearSolver, "_postcompute")
def test_NonLinearSolver_init_handling_design(mock_postcompute):
    s2 = MultiplyVector2("multvector")

    solver = s2.add_driver(
        NonLinearSolver("solver", method=NonLinearMethods.POWELL)
    )
    point1 = solver.add_child(RunSingleCase("point1"))

    s2.k1 = 11.0
    s2.k2 = 8.0

    point1.set_values({"p_in.x1": 4.0, "p_in.x2": 10.0})
    point1.design.add_unknown("k1").add_equation("p_out.x == 100")

    # Read from current system
    s2.run_drivers()
    assert s2.k1 == pytest.approx(5, abs=1e-5)
    assert solver.initial_values == pytest.approx([11])
    assert list(point1.solution.values()) == pytest.approx([5])
    mock_postcompute.assert_called_once()
    mock_postcompute.reset_mock()

    # Use init
    point1.set_init({"k1": 10.0})
    s2.run_drivers()
    assert solver.initial_values == pytest.approx([10])
    assert list(point1.solution.values()) == pytest.approx([5])

    # Use latest solution
    s2.run_drivers()
    assert solver.initial_values == pytest.approx([5])
    assert list(point1.solution.values()) == pytest.approx([5])

    # Reuse init
    solver.force_init = True
    s2.run_drivers()
    assert solver.initial_values == pytest.approx([10])
    assert list(point1.solution.values()) == pytest.approx([5])
    solver.force_init = False

    point2 = solver.add_child(RunSingleCase("point2"))
    point2.set_values({"p_in.x1": 2, "p_in.x2": 8.0})
    point2.design.add_unknown("k2").add_equation("p_out.x == 76")

    # Use latest solution for run1 but current system for run2
    mock_postcompute.reset_mock()
    s2.run_drivers()
    assert solver.initial_values == pytest.approx([5, 8])
    assert list(point1.solution.values()) == pytest.approx([10 / 3])
    assert list(point2.solution.values()) == pytest.approx([26 / 3])
    mock_postcompute.assert_called_once()
    assert s2.k1 == pytest.approx(10 / 3, abs=1e-5)
    assert s2.k2 == pytest.approx(26 / 3, abs=1e-5)

    # Use latest solution for run1 but init for run2
    point2.set_init({"k2": 7.0})
    s2.run_drivers()
    assert solver.initial_values == pytest.approx([10 / 3, 7])
    assert list(point1.solution.values()) == pytest.approx([10 / 3])
    assert list(point2.solution.values()) == pytest.approx([26 / 3])

    # Use latest solution for all points
    s2.run_drivers()
    assert solver.initial_values == pytest.approx([10 / 3, 26 / 3])
    assert list(point1.solution.values()) == pytest.approx([10 / 3])
    assert list(point2.solution.values()) == pytest.approx([26 / 3])

    # Reuse init for all points
    solver.force_init = True
    s2.run_drivers()
    assert solver.initial_values == pytest.approx([10, 7])
    assert list(point1.solution.values()) == pytest.approx([10 / 3])
    assert list(point2.solution.values()) == pytest.approx([26 / 3])
    solver.force_init = False


def test_NonLinearSolver_init_handling_offdesign(test_library, test_data):
    s = System.load(test_data / "system_config_pressureloss121.json")

    s.exec_order = "p1", "sp", "p2", "p3", "mx", "p4"
    s.run_once()

    assert s.p4.flnum_out.Pt == 74800.0

    solver = s.add_driver(NonLinearSolver("solver", tol=5.0e-8))
    # design.add_unknown("sp.x")

    # System value
    s.run_drivers()
    assert solver.initial_values == pytest.approx([0.8])
    assert list(solver.solution.values()) == pytest.approx([0.5], rel=1e-4)
    assert solver.results.x == pytest.approx([0.5], rel=1e-4)

    # Reuse solution
    s.run_drivers()
    assert solver.initial_values == pytest.approx([0.5])
    assert list(solver.solution.values()) == pytest.approx([0.5], rel=1e-4)
    assert solver.results.x == pytest.approx([0.5], rel=1e-4)

    # Initial value
    s.sp.x = 2.0
    s.run_drivers()
    assert solver.initial_values == pytest.approx([2], abs=0)
    assert list(solver.solution.values()) == pytest.approx([0.5], rel=1e-4)
    assert solver.results.x == pytest.approx([0.5], rel=1e-4)

    # Reuse solution
    s.run_drivers()
    assert solver.initial_values == pytest.approx([0.5])
    assert list(solver.solution.values()) == pytest.approx([0.5], rel=1e-4)
    assert solver.results.x == pytest.approx([0.5], rel=1e-4)


@pytest.mark.parametrize("format", LogFormat)
@pytest.mark.parametrize("msg, kwargs, trace, to_log, emitted", [
    ("zombie call_setup_run", dict(), list(), False, None),
    ("useless start call_clean_run", dict(activate=True), list(), False, None),
    (
        f"{NonLinearSolver.CONTEXT_EXIT_MESSAGE} call_clean_run",
        dict(activate=False),
        list(),
        False,
        dict(levelno=LogLevel.DEBUG, patterns=[r"Compute calls for [\w\.]+: \d+",])
    ),
    ("other message with activation", dict(activate=True), list(), False, None),
    (
        "second message with deactivation",
        dict(activate=False),
        list((dict(x=np.zeros(2), residues=np.zeros(2)), dict(x=np.zeros(2), residues=np.zeros(2)))), 
        False, 
        dict(levelno=LogLevel.FULL_DEBUG, patterns=[r"Unknowns\n", r"Residues\n"])
    ),
    (
        "second message with deactivation",
        dict(activate=False),
        list((dict(x=np.zeros(2), residues=np.zeros(2), jac=np.eye(2)),)), 
        False, 
        dict(levelno=LogLevel.FULL_DEBUG, patterns=[r"Unknowns\n", r"Residues\n", r"Iteration \d+\n", r"New Jacobian matrix:\n,"])
    ),
    ("common message", dict(), list(), True, None),
])
def test_NonLinearSolver_log_debug_message(format, msg, kwargs, trace, to_log, emitted):
    handler = MagicMock(level=LogLevel.DEBUG, log=MagicMock())
    rec = logging.getLogRecordFactory()("log_test", LogLevel.INFO, __file__, 22, msg, (), None)
    for key, value in kwargs.items():
        setattr(rec, key, value)
    
    s = NonLinearSolver("dummy")
    s._NonLinearSolver__trace = trace
    s.problem = MagicMock(
        spec=MathematicalProblem,
        n_equations=2,
        n_unknowns=2,
    )
    s.problem.residue_names.return_value = ("r", "s")
    s.problem.unknown_names.return_value = ("x", "y")
    assert s.log_debug_message(handler, rec, format) == to_log

    if "activate" in kwargs and not msg.endswith("_run"):
        assert s.options["history"] == kwargs["activate"]

    if emitted:
        handler.log.assert_called_once()
        args = handler.log.call_args[0]
        assert args[0] == emitted["levelno"]
        for pattern in emitted["patterns"]:
            assert re.search(pattern, args[1]) is not None
    else:
        handler.log.assert_not_called()


@pytest.mark.parametrize("settings, rtol", [
    (dict(), 1e-15),  # default settings
    (dict(method=NonLinearMethods.NR, tol='auto'), 1e-15),
    (dict(method=NonLinearMethods.NR, tol=1e-9), 1e-9),
    (dict(method=NonLinearMethods.POWELL, tol=1e-9), 1e-9),
])
def test_NonLinearSolver_fixedPointArray(FixedPointArray, settings, rtol):
    """Integration test for `NonLinearSolver` driver:
    Solve cyclic dependency between two systems
    exchanging numpy arrays.
    """
    top = FixedPointArray('top')

    solver = top.add_driver(NonLinearSolver('solver', **settings))
    solver.add_child(RunSingleCase('runner'))
    solver.runner.set_init({'a.x.value': np.r_[-0.25, 1.04, 5.2]})
    top.run_drivers()

    expected = np.r_[
        0.5 * (3 - np.sqrt(5)),
        0.5 * (3 - np.sqrt(5)),
        0.5 * (3 + np.sqrt(5)),
    ]
    assert top.a.x.value == pytest.approx(expected, rel=rtol)


def test_NonLinearSolver_fixedPointArray_rerun(FixedPointArray, caplog):
    """Integration test for `NonLinearSolver` driver:
    Solve cyclic dependency between two systems
    exchanging numpy arrays.
    """
    top = FixedPointArray('top')

    solver = top.add_driver(NonLinearSolver('solver', tol=1e-9, verbose=True))
    solver.add_child(RunSingleCase('runner'))
    solver.runner.set_init({'a.x.value': np.r_[-0.25, 1.04, 5.2]})

    # First driver execution
    caplog.clear()
    with caplog.at_level(logging.INFO):
        top.run_drivers()
    # Check solution
    expected = np.r_[
        0.5 * (3 - np.sqrt(5)),
        0.5 * (3 - np.sqrt(5)),
        0.5 * (3 + np.sqrt(5)),
    ]
    assert top.a.x.value == pytest.approx(expected)
    # Check log record
    assert len(caplog.records) > 0
    assert any(
        re.match(r".*Converged \(.*\) in \d iterations", record.message)
        for record in caplog.records
    )

    # Second driver execution: already converged
    solver.force_init = False
    caplog.clear()
    with caplog.at_level(logging.INFO):
        top.run_drivers()
    assert len(caplog.records) > 0
    assert any(
        re.match(r".*Converged \(.*\) in 0 iteration", record.message)
        for record in caplog.records
    )
    # Check solution
    assert top.a.x.value == pytest.approx(expected)


@pytest.mark.parametrize("tol, expected", [
    ('auto', does_not_raise()),
    (1e-6, does_not_raise()),
    (0.0, does_not_raise()),
    (None, does_not_raise()),
    (1, pytest.raises(TypeError)),
    (True, pytest.raises(TypeError)),
    (-1e-6, pytest.raises(ValueError)),
    ('foo', pytest.raises(ValueError)),
])
def test_NonLinearSolver_tol(FixedPointArray, tol, expected):
    """Check expected behaviour for various values of `tol`.
    Note:
    -----
    At present, errors may be raised at driver construction
    or at driver execution. This should be fixed in a future revision,
    as illegal types or values should be captured at construction.
    """
    system = FixedPointArray('system')

    with expected:
        system.add_driver(NonLinearSolver('solver', tol=tol))
        system.a.x.value = np.r_[-0.25, 1.04, 5.2]
        system.run_drivers()


@pytest.mark.parametrize("options, expected", [
    (dict(tol_to_noise_ratio=1), does_not_raise()),
    (dict(tol_to_noise_ratio=8), does_not_raise()),
    (dict(tol_to_noise_ratio=8.0), does_not_raise()),
    (dict(tol_to_noise_ratio=0.5), pytest.raises(ValueError)),
    (dict(tol_to_noise_ratio=None), pytest.raises(TypeError)),
    (dict(tol_update_period=1), does_not_raise()),
    (dict(tol_update_period=10), does_not_raise()),
    (dict(tol_update_period=0), pytest.raises(ValueError)),
    (dict(tol_update_period=-1), pytest.raises(ValueError)),
    (dict(tol_update_period=8.0), pytest.raises(TypeError)),
    (dict(tol_update_period=None), pytest.raises(TypeError)),
    (dict(tol_update_period='auto'), pytest.raises(TypeError)),
])
def test_NonLinearSolver_options(options, expected):
    """Check expert options such as `tol_to_noise_ratio`
    and `tol_update_period`.
    """
    with expected:
        NonLinearSolver('solver', **options)


def get_module(cls: type):
    """Utility function returning the full module path of `cls`"""
    return f"{cls.__module__}.{cls.__qualname__}"


def test_NonLinearSolver_empty_problem_mock():
    """Test solver behaviour with a plain, direct system.
    Related to https://gitlab.com/cosapp/cosapp/-/issues/99
    """
    s = System("system")
    s.add_driver(NonLinearSolver('solver'))

    specs = dict(
        compute = DEFAULT,
    )
    
    with patch.multiple(get_module(System), **specs) as mocked:
        s.run_drivers()
        assert mocked['compute'].call_count > 0


def test_NonLinearSolver_empty_problem_mock_multipoint():
    """Test multi-point solver behaviour with a plain, direct system.
    Related to https://gitlab.com/cosapp/cosapp/-/issues/99
    """
    s = System("system")
    solver = s.add_driver(NonLinearSolver('solver'))
    solver.add_child(RunSingleCase('point1'))
    solver.add_child(RunSingleCase('point2'))

    specs = dict(
        get_init = DEFAULT,
        apply_values = DEFAULT,
    )
    
    with patch.multiple(get_module(RunSingleCase), **specs) as mocked:
        s.run_drivers()
        assert mocked['get_init'].call_count == 2
        assert mocked['apply_values'].call_count == 2


@pytest.mark.parametrize("method_name", [
    "set_init",
    "set_values",
])
def test_NonLinearSolver_empty_problem_case(method_name):
    """Test solver with a direct system, with case values
    set through `RunSingleCase.set_init` and `set_values`.

    Expected behaviour: case values are transferred to system,
    and system is updated.

    Related to https://gitlab.com/cosapp/cosapp/-/issues/99
    """
    s = QuadraticFunction("system")
    s.a = s.k = s.x = 0.0

    solver = s.add_driver(NonLinearSolver('solver'))
    point = solver.add_child(RunSingleCase('point'))

    set_point = getattr(point, method_name)
    set_point({
        'a': 1.0,
        'k': 0.5,
        'x': 0.1,
    })

    s.run_drivers()

    assert s.x == 0.1
    assert s.k == 0.5
    assert s.a == 1.0
    assert s.y == pytest.approx(-0.995)
    assert solver.problem.is_empty()


@pytest.mark.parametrize(
    "shape", [
        3,
        (3, 2),
        (3, 2, 4),
    ]
)
def test_NonLinearSolver_ndarray_residue(shape):
    """Test solver with a mathematical problem involving
    array unknowns and residues of various shapes.

    Related to https://gitlab.com/cosapp/cosapp/-/issues/122
    """
    class NdArraySystem(System):
        def setup(self):
            self.add_inward('x', np.ones(shape))
            self.add_outward('y', np.ones(shape))

        def compute(self):
            self.y = 10 * self.x

    s = NdArraySystem('s')

    solver = s.add_driver(NonLinearSolver('solver'))
    solver.add_unknown('x').add_equation(f"y == full({shape}, 2.0)")

    s.run_drivers()

    assert s.x == pytest.approx(np.full(shape, 0.2))
    assert s.y == pytest.approx(np.full(shape, 2.0))


def test_NonLinearSolver_singular_problem_1(caplog):
    """Singular problem, owing to unaffected residue.
    """
    system = QuadraticFunction('system')
    solver = system.add_driver(NonLinearSolver('solver'))

    solver.add_equation(["y == 2", "k == 1"])
    solver.add_unknown(["x", "a"])

    system.x = 1.0
    system.a = -0.5
    system.k = 0.1

    caplog.clear()
    with caplog.at_level(logging.ERROR, NonLinearSolver.__name__):
        system.run_drivers()

    assert solver.results.success == False
    assert len(caplog.records) == 1
    message = caplog.records[0].msg
    assert re.match(
        r".*residue.* not influenced: \['k == 1'\]",
        message,
        flags=re.DOTALL,  # match "\n" with "."
    )


def test_NonLinearSolver_singular_problem_2(caplog):
    """Same as `test_NonLinearSolver_singular_problem_1`,
    with several unaffected residues.
    """
    f = AbcdFunction('f')

    f.x = np.r_[1.0, 2.0, 3.0, 4.0]

    solver = f.add_driver(NonLinearSolver('solver'))
    solver.add_equation(["a == 0", "b == 0", "c == 0"])
    solver.add_unknown(["x[:3]"])

    caplog.clear()
    with caplog.at_level(logging.ERROR, NonLinearSolver.__name__):
        f.run_drivers()

    assert solver.results.success == False
    assert len(caplog.records) == 1
    message = caplog.records[0].msg
    assert re.match(
        r".*residue.* not influenced: \['b == 0', 'c == 0'\]",
        message,
        flags=re.DOTALL,  # match "\n" with "."
    )


def test_NonLinearSolver_singular_problem_3(caplog):
    """Singular problem, owing to unknown with no influence.
    """
    f = AbcdFunction('f')

    f.x = np.r_[1.0, 2.0, 3.0, 4.0]

    solver = f.add_driver(NonLinearSolver('solver'))
    solver.add_equation(["a == 1", "d == 1"])
    solver.add_unknown(["x[0]", "x[2]"])

    caplog.clear()
    with caplog.at_level(logging.ERROR, NonLinearSolver.__name__):
        f.run_drivers()

    assert solver.results.success == False
    assert len(caplog.records) == 1
    message = caplog.records[0].msg
    assert re.match(
        r".*parameter.* no influence: \['x\[2\]'\]",
        message,
        flags=re.DOTALL,  # match "\n" with "."
    )
