import pytest
import logging, re
import numpy as np

from collections import OrderedDict
from unittest.mock import MagicMock, patch

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


def test_NonLinearSolver__setattr__():
    # Error is raised when setting an absent attribute
    d = NonLinearSolver("driver")
    with pytest.raises(AttributeError):
        d.ftol = 1e-5


def test_NonLinearSolver_add_child():
    d = NonLinearSolver("driver")
    subdriver_name = "subdriver"
    sub_driver = Driver(subdriver_name)
    d.compute_jacobian = False
    assert set(d.children) == set((d._default_driver_name, ))
    assert isinstance(d.children[d._default_driver_name], RunSingleCase)
    assert d.compute_jacobian == False

    d.add_child(sub_driver)
    assert set(d.children) == set((subdriver_name,))
    assert d.children[subdriver_name] is sub_driver
    assert d.compute_jacobian == True


def test_NonLinearSolver_is_standalone():
    d = NonLinearSolver("driver")
    assert d.is_standalone()


def test_NonLinearSolver__fresidues(set_master_system):

    # Simple system with no design equations
    system = Multiply2("MyMult")
    solver = NonLinearSolver("solver")
    system.add_driver(solver)
    solver.add_unknown(["K1", "K2", "p_in.x"])
    system.call_setup_run()
    solver._precompute()

    init = np.random.rand(len(solver.problem.unknowns))
    residues = solver._fresidues(init, "init")
    set_init = [var.default_value for var in solver.problem.unknowns.values()]
    assert set_init == list(init)
    mask_unknowns = [var.mask for var in solver.problem.unknowns.values()]
    assert mask_unknowns == list((None, None, None))
    assert isinstance(residues, np.ndarray)
    assert len(residues) == 0

    # More realistic single case
    system = Multiply2("MyMult")
    solver = NonLinearSolver("solver")
    system.add_driver(solver)
    runner = solver.children[solver._default_driver_name]
    solver.add_unknown("p_in.x").add_equation("p_in.x == 2")  # design unknown, off-design eq.
    runner.design.add_unknown("K1").add_equation("p_out.x == 20")
    system.call_setup_run()
    solver._precompute()

    init = np.random.rand(len(solver.problem.unknowns))
    residues = solver._fresidues(init, "init")
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
    residues = solver._fresidues(init, "init")
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
    residues = solver._fresidues(init, "init")
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
    residues = solver._fresidues(init, "init")
    variables = [var.default_value for var in solver.problem.unknowns.values()]
    assert variables == list(init)
    mask_unknowns = [var.mask for var in solver.problem.unknowns.values()]
    assert mask_unknowns == [None, None, None, None]
    assert isinstance(residues, np.ndarray)
    assert len(residues) == 4
    assert solver.problem.n_unknowns == 4
    assert solver.problem.n_equations == 4


def test_NonLinearSolver__precompute(caplog, set_master_system):
    # Simple system with no design equations
    system = Multiply2("MyMult")
    solver = NonLinearSolver("solver")
    system.add_driver(solver)
    solver.add_unknown(["K1", "K2", "p_in.x"])
    system.call_setup_run()
    solver._precompute()
    assert set(solver.problem.unknowns) == {
        "inwards.K1", "inwards.K2", "p_in.x",
    }
    assert set(solver.problem.residues) == set()

    # More realistic single case
    system = Multiply2("MyMult")
    solver = system.add_driver(NonLinearSolver("solver"))
    runner = solver.children[solver._default_driver_name]
    runner.offdesign.add_unknown("p_in.x").add_equation("p_in.x == 2")
    runner.design.add_unknown("K1").add_equation("p_out.x == 20")
    init = np.random.rand(2)
    runner.set_init({"K1": init[0], "p_in.x": init[1]})
    assert set(runner.initial_values) == {"inwards.K1", "p_in.x"}
    assert runner.initial_values["inwards.K1"].default_value == init[0]
    assert runner.initial_values["p_in.x"].default_value == init[1]
    system.call_setup_run()
    solver._precompute()
    assert set(solver.problem.unknowns) == {
        "inwards.K1", f"{runner.name}[p_in.x]",
    }
    assert set(solver.problem.residues) == {
        f"{runner.name}[p_out.x == 20]",
        f"{runner.name}[p_in.x == 2]",
    }
    assert list(solver.initial_values) == list(init)
    assert runner.initial_values["inwards.K1"].default_value == init[0]
    assert runner.initial_values["p_in.x"].default_value == init[1]

    system.K1 = 0.8
    solver._precompute()
    assert list(solver.initial_values) == list(init)
    solver.children[solver._default_driver_name].solution["inwards.K1"] = 0.8
    solver._precompute()
    assert list(solver.initial_values) == [0.8, init[1]]

    solver.force_init = True
    solver._precompute()
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
        "inwards.K1", "inwards.K2",  # design unknowns
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
    solver._precompute()
    assert list(solver.initial_values) == [0.123] + expected_init[1:]

    system.K1 = expected_init[0]
    solver.point1.solution["inwards.K1"] = 0.123
    solver._precompute()
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
    solver._precompute()

    assert set(solver.problem.unknowns) == {
        "inwards.K1", "point1[p_in.x]",
        "inwards.K2", "point2[p_in.x]",
    }
    assert list(solver.initial_values) == list(init)

    system.K1 = 0.2
    solver._precompute()
    assert list(solver.initial_values) == list(init)
    solver.point1.solution["inwards.K1"] = 0.2
    solver._precompute()
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
    assert len(warning_messages) == 1
    assert (
        re.search(
            r"Including Driver '\w+' without iteratives in Driver '\w+' is not numerically advised.",
            warning_messages[0],
        )
        is not None
    )


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
    runner = solver.children[solver._default_driver_name]
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
    runner = solver.children[solver._default_driver_name]
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
    s = Multiply2("MyMult")
    d = NonLinearSolver("solver", method=NonLinearMethods.POWELL)
    s.add_driver(d)
    d.add_unknown("p_in.x").add_equation("p_in.x == 2")
    default = d._default_driver_name
    d.children[default].add_unknown("K1").add_equation("p_out.x == 20")
    init = np.random.rand(2)
    d.children[default].set_init({"K1": init[0], "p_in.x": init[1]})
    s.call_setup_run()
    d._precompute()
    caplog.clear()
    with caplog.at_level(logging.INFO):
        d.compute()

    info_messages = list(record.msg
        for record in filter(lambda r: r.levelno == logging.INFO, caplog.records)
    )
    assert len(info_messages) == 1
    assert re.search(r"solver : \w+The solution converged\.", info_messages[0]) is not None

    assert len(d.solution) == len(init)
    np.testing.assert_allclose(list(d.solution.values()), (2.0, 2.0))


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
        "inwards.K1", "inwards.K2", "p_in.x",
    }
    assert set(solver.problem.residues) == set()

    # More realistic single case
    system = Multiply2("MyMult")
    solver = system.add_driver(NonLinearSolver("solver"))
    runner = solver.runner
    solver.add_unknown("K1").add_equation("p_out.x == 20")
    runner.add_unknown("p_in.x").add_equation("p_in.x == 2")
    system.call_setup_run()
    solver._precompute()
    solver._postcompute()
    assert set(solver.problem.unknowns) == {
        "inwards.K1", f"{runner.name}[p_in.x]",
    }
    assert set(solver.problem.residues) == {
        f"{runner.name}[p_out.x == 20]",
        f"{runner.name}[p_in.x == 2]",
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
        "inwards.K1", "point1[p_in.x]",
        "inwards.K2", "point2[p_in.x]",
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
        "inwards.K1", "point1[p_in.x]",
        "inwards.K2", "point2[p_in.x]",
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
    solver._fresidues(init, "init")
    assert isinstance(solver.problem.residues, OrderedDict)
    assert len(solver.problem.residues) == 0

    # More realistic single case
    system = Multiply2("MyMult")
    solver = system.add_driver(NonLinearSolver("solver"))
    runner = solver.runner
    solver.add_unknown("K1").add_equation("p_out.x == 20")
    runner.add_unknown("p_in.x").add_equation("p_in.x == 2")
    solver.owner.call_setup_run()
    solver._precompute()
    solver.compute()
    assert set(solver.problem.residues) == set(
        f"{runner.name}[{eq}]"
        for eq in ("p_in.x == 2", "p_out.x == 20",)
    )
    for v in solver.problem.residues.values():
        assert isinstance(v, Residue)

    # Same, with runner-level problems only
    system = Multiply2("MyMult")
    solver = system.add_driver(NonLinearSolver("solver"))
    runner = solver.runner
    runner.design.add_unknown("K1").add_equation("p_out.x == 20")
    runner.add_unknown("p_in.x").add_equation("p_in.x == 2")
    solver.owner.call_setup_run()
    solver._precompute()
    solver.compute()
    assert set(solver.problem.residues) == set(
        f"{runner.name}[{eq}]"
        for eq in ("p_in.x == 2", "p_out.x == 20",)
    )
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
        "inwards.K1",
        "inwards.K2",
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
        "inwards.K1",
        "point2[inwards.K2]",
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
        "inwards.K1",
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
        match="'inwards\.K1' is defined as design and off-design unknown in 'point1'"
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
        match="\('inwards\.K1', 'inwards\.K2'\) are defined as design and off-design unknowns in 'point2'"
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
        match="'inwards\.K1' already exists in 'solver'"
    ):
        s.run_drivers()

    # Conflicting case unknowns (1)
    s, solver, point1, point2 = make_case()
    
    point1.design.add_unknown("K1").add_equation("p_out.x == 6")
    point2.design.add_unknown("K2").add_equation("p_out.x == 3 * Ksum")
    point2.add_unknown("K1")  # incompatible with point1 design unknown

    with pytest.raises(
        ValueError,
        match="'inwards\.K1' is defined as design and off-design unknown in 'point2'"
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
        match="'inwards\.K1' is defined as design and off-design unknown in 'point2'"
    ):
        s.run_drivers()

    # Conflict within a single point
    s, solver, point1, point2 = make_case()
    
    point1.design.add_unknown("K1").add_equation("p_out.x == 6")
    point2.design.add_unknown("K2").add_equation("p_out.x == 3 * Ksum")
    point1.add_unknown("K1")  # incompatible with point1 design unknown

    with pytest.raises(
        ValueError,
        match="'inwards\.K1' is defined as design and off-design unknown in 'point1'"
    ):
        s.run_drivers()


def test_NonLinearSolver_vector1d_system():
    s = System("hat")
    one = s.add_child(Strait1dLine("one"), pulling="in_")
    two = s.add_child(Strait1dLine("two"), pulling="out")
    s.connect(two.in_, one.out)

    # Test transfer value
    s.in_.x *= 3.0
    assert np.array_equal(s.in_.x, s.out.x) == False
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

    design = s2.add_driver(NonLinearSolver("design", method=NonLinearMethods.POWELL))

    run1 = design.add_child(RunSingleCase("run 1"))

    s2.inwards.k1 = 11.0
    s2.inwards.k2 = 8.0

    run1.set_values({"p_in.x1": 4.0, "p_in.x2": 10.0})
    run1.design.add_unknown("inwards.k1").add_equation("p_out.x == 100.")

    # Read from current system
    s2.run_drivers()
    assert s2.inwards.k1 == pytest.approx(5.0, abs=10e-5)
    np.testing.assert_array_almost_equal(design.initial_values, np.asarray([11.0]))
    np.testing.assert_allclose(
        list(run1.solution.values()), np.asarray([5.0]), rtol=1e-4
    )
    mock_postcompute.assert_called_once()
    mock_postcompute.reset_mock()

    # Use init
    run1.set_init({"k1": 10.0})
    s2.run_drivers()
    np.testing.assert_array_almost_equal(design.initial_values, np.asarray([10.0]))
    np.testing.assert_allclose(
        list(run1.solution.values()), np.asarray([5.0]), rtol=1e-4
    )

    # Use latest solution
    s2.run_drivers()
    np.testing.assert_array_almost_equal(design.initial_values, np.asarray([5.0]))
    np.testing.assert_allclose(
        list(run1.solution.values()), np.asarray([5.0]), rtol=1e-4
    )

    # Reuse init
    design.force_init = True
    s2.run_drivers()
    np.testing.assert_array_almost_equal(design.initial_values, np.asarray([10.0]))
    np.testing.assert_allclose(
        list(run1.solution.values()), np.asarray([5.0]), rtol=1e-4
    )
    design.force_init = False

    run2 = design.add_child(RunSingleCase("run 2"))
    run2.set_values({"p_in.x1": 2, "p_in.x2": 8.0})
    run2.design.add_unknown("inwards.k2").add_equation("p_out.x == 76.")

    # Use latest solution for run1 but current system for run2
    mock_postcompute.reset_mock()
    s2.run_drivers()
    np.testing.assert_array_almost_equal(design.initial_values, np.asarray([5.0, 8.0]))
    np.testing.assert_allclose(
        list(run1.solution.values()), np.asarray([10.0 / 3.0]), rtol=1e-4
    )
    np.testing.assert_allclose(
        list(run2.solution.values()), np.asarray([26.0 / 3.0]), rtol=1e-4
    )
    mock_postcompute.assert_called_once()
    assert s2.inwards.k1 == pytest.approx(10.0 / 3.0, abs=10e-5)
    assert s2.inwards.k2 == pytest.approx(26.0 / 3.0, abs=10e-5)

    # Use latest solution for run1 but init for run2
    run2.set_init({"k2": 7.0})
    s2.run_drivers()
    np.testing.assert_array_almost_equal(
        design.initial_values, np.asarray([10.0 / 3.0, 7.0])
    )
    np.testing.assert_allclose(
        list(run1.solution.values()), np.asarray([10.0 / 3.0]), rtol=1e-4
    )
    np.testing.assert_allclose(
        list(run2.solution.values()), np.asarray([26.0 / 3.0]), rtol=1e-4
    )

    # Use latest solution for all points
    s2.run_drivers()
    np.testing.assert_array_almost_equal(
        design.initial_values, np.asarray([10.0 / 3.0, 26.0 / 3.0])
    )
    np.testing.assert_allclose(
        list(run1.solution.values()), np.asarray([10.0 / 3.0]), rtol=1e-4
    )
    np.testing.assert_allclose(
        list(run2.solution.values()), np.asarray([26.0 / 3.0]), rtol=1e-4
    )

    # Reuse init for all points
    design.force_init = True
    s2.run_drivers()
    np.testing.assert_array_almost_equal(design.initial_values, np.asarray([10.0, 7.0]))
    np.testing.assert_allclose(
        list(run1.solution.values()), np.asarray([10.0 / 3.0]), rtol=1e-4
    )
    np.testing.assert_allclose(
        list(run2.solution.values()), np.asarray([26.0 / 3.0]), rtol=1e-4
    )
    design.force_init = False


def test_NonLinearSolver_init_handling_offdesign(test_library, test_data):
    s = System.load(str(test_data / "system_config_pressureloss121.json"))

    s.exec_order = "p1", "sp", "p2", "p3", "mx", "p4"
    s.run_once()

    assert s.p4.flnum_out.Pt == 74800.0

    design = s.add_driver(NonLinearSolver("design", tol=5.0e-8))
    # design.add_unknown("sp.x")

    # System value
    s.run_drivers()
    assert design.initial_values == pytest.approx([0.8])
    assert list(design.runner.solution.values()) == pytest.approx([0.5], rel=1e-4)

    # Reuse solution
    s.run_drivers()
    assert design.initial_values == pytest.approx([0.5])
    assert list(design.runner.solution.values()) == pytest.approx([0.5], rel=1e-4)

    # Initial value
    design.runner.set_init({"sp.inwards.x": 2.0})
    s.run_drivers()
    assert design.initial_values == pytest.approx([2], abs=0)
    assert list(design.runner.solution.values()) == pytest.approx([0.5], rel=1e-4)

    # Reuse solution
    s.run_drivers()
    assert design.initial_values == pytest.approx([0.5])
    assert list(design.runner.solution.values()) == pytest.approx([0.5], rel=1e-4)


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
        residues_names=["r", "s"],
        shape=(2, 2),
        unknowns_names=["x", "y"],
    )
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


def test_NonLinearSolver_fixedPointArray():
    """Integration test for `NonLinearSolver` driver:
    Solve cyclic dependency between two systems
    exchanging numpy arrays.
    """
    class Port3D(Port):
        def setup(self):
            self.add_variable('value', np.r_[-0.2, 1.1, 5.2])

    class SystemA(System):
        def setup(self):
            self.add_input(Port3D, 'x')
            self.add_output(Port3D, 'y')
        
        def compute(self):
            print(self.x.value)
            self.y.value = 1 - self.x.value

    class SystemB(System):
        def setup(self):
            self.add_input(Port3D, 'u')
            self.add_output(Port3D, 'v')
        
        def compute(self):
            self.v.value = self.u.value**2

    top = System('top')
    a = top.add_child(SystemA('a'))
    b = top.add_child(SystemB('b'))
    top.connect(a.x, b.v)
    top.connect(a.y, b.u)

    solver = top.add_driver(NonLinearSolver('solver', tol=1e-9))
    solver.runner.set_init({'a.x.value[:-1]': np.r_[-0.25, 1.04]})
    top.run_drivers()

    assert a.x.value[0] == pytest.approx(0.5 * (3 - np.sqrt(5)))
    assert a.x.value[1] == pytest.approx(0.5 * (3 - np.sqrt(5)))
    assert a.x.value[2] == pytest.approx(0.5 * (3 + np.sqrt(5)))
