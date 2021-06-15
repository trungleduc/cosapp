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
    s = Multiply2("MyMult")
    d = NonLinearSolver("run_drivers")
    s.add_driver(d)
    d.runner.offdesign.add_unknown(["K1", "K2", "p_in.x"])
    s.call_setup_run()
    d._precompute()

    init = np.random.rand(len(d.problem.unknowns))
    residues = d._fresidues(init, "init")
    set_init = [var.default_value for var in d.problem.unknowns.values()]
    assert set_init == list(init)
    mask_unknowns = [var.mask for var in d.problem.unknowns.values()]
    assert mask_unknowns == list((None, None, None))
    assert isinstance(residues, np.ndarray)
    assert len(residues) == 0

    # More realistic single case
    s = Multiply2("MyMult")
    d = NonLinearSolver("run_drivers")
    s.add_driver(d)
    d.children[d._default_driver_name].offdesign.add_unknown("p_in.x").add_equation(
        "p_in.x == 2."
    )
    d.children[d._default_driver_name].design.add_unknown("K1").add_equation(
        "p_out.x == 20."
    )
    s.call_setup_run()
    d._precompute()

    init = np.random.rand(len(d.problem.unknowns))
    residues = d._fresidues(init, "init")
    set_init = [var.default_value for var in d.problem.unknowns.values()]
    assert set_init == list(init)
    mask_unknowns = [var.mask for var in d.problem.unknowns.values()]
    assert mask_unknowns == [None, None]
    assert isinstance(residues, np.ndarray)
    assert len(residues) == 2

    # Multiple cases
    s = Multiply2("MyMult")
    d = NonLinearSolver("run_drivers")
    s.add_driver(d)
    p1 = RunSingleCase("pt1")
    d.add_child(p1)
    p1.offdesign.add_unknown("p_in.x").add_equation("p_in.x == 2.")
    p1.design.add_unknown("K1").add_equation("p_out.x == 20.")
    p2 = RunSingleCase("pt2")
    d.add_child(p2)
    p2.offdesign.add_unknown("p_in.x").add_equation("p_in.x == 6.")
    p2.design.add_unknown("K2").add_equation("p_out.x == 15.")
    s.call_setup_run()
    d._precompute()

    init = np.random.rand(len(d.problem.unknowns))
    residues = d._fresidues(init, "init")
    variables = [var.default_value for var in d.problem.unknowns.values()]
    assert variables == list(init)
    mask_unknowns = [var.mask for var in d.problem.unknowns.values()]
    assert mask_unknowns == [None, None, None, None]
    assert isinstance(residues, np.ndarray)
    assert len(residues) == 4


def test_NonLinearSolver__precompute(caplog, set_master_system):
    # Simple system with no design equations
    s = Multiply2("MyMult")
    d = NonLinearSolver("run_drivers")
    s.add_driver(d)
    d.runner.offdesign.add_unknown(["K1", "K2", "p_in.x"])
    s.call_setup_run()
    d._precompute()

    vars = [
        f"{d._default_driver_name}[{u}]"
        for u in ("inwards.K1", "inwards.K2", "p_in.x")
    ]
    assert set(d.problem.unknowns) == set(vars)

    # More realistic single case
    s = Multiply2("MyMult")
    d = NonLinearSolver("run_drivers")
    s.add_driver(d)
    d.children[d._default_driver_name].offdesign.add_unknown("p_in.x").add_equation(
        "p_in.x == 2."
    )
    d.children[d._default_driver_name].design.add_unknown("K1").add_equation(
        "p_out.x == 20."
    )
    init = np.random.rand(2)
    d.children[d._default_driver_name].set_init({"K1": init[0], "p_in.x": init[1]})
    assert set(d.children[d._default_driver_name].initial_values) == set(
        ("inwards.K1", "p_in.x")
    )
    assert (
        d.children[d._default_driver_name].initial_values["inwards.K1"].default_value
        == init[0]
    )
    assert (
        d.children[d._default_driver_name].initial_values["p_in.x"].default_value
        == init[1]
    )
    s.call_setup_run()
    d._precompute()
    assert set(d.problem.unknowns) == set(
        ("inwards.K1", d._default_driver_name + "[p_in.x]")
    )
    assert list(d.initial_values) == list(init)
    assert (
        d.children[d._default_driver_name].initial_values["inwards.K1"].default_value
        == init[0]
    )
    assert (
        d.children[d._default_driver_name].initial_values["p_in.x"].default_value
        == init[1]
    )

    s.K1 = 0.8
    d._precompute()
    assert list(d.initial_values) == list(init)
    d.children[d._default_driver_name].solution["inwards.K1"] = 0.8
    d._precompute()
    assert list(d.initial_values) == [0.8, init[1]]

    d.force_init = True
    d._precompute()
    assert list(d.initial_values) == list(init)

    # Multiple cases
    s = Multiply2("MyMult")
    d = NonLinearSolver("run_drivers")
    s.add_driver(d)
    p1 = RunSingleCase("pt1")
    init = np.random.rand(4)
    d.add_child(p1)
    p1.offdesign.add_unknown("p_in.x").add_equation("p_in.x == 2.")
    p1.design.add_unknown("K1").add_equation("p_out.x == 20.")
    p1.set_init({"K1": init[0], "p_in.x": init[1]})
    p2 = RunSingleCase("pt2")
    d.add_child(p2)
    p2.offdesign.add_unknown("p_in.x").add_equation("p_in.x == 6.")
    p2.design.add_unknown("K2").add_equation("p_out.x == 15.")
    p2.set_init({"K2": init[2], "p_in.x": init[3]})
    s.call_setup_run()
    d._precompute()

    assert set(d.problem.unknowns) == set(
        ("inwards.K1", "pt1[p_in.x]", "inwards.K2", "pt2[p_in.x]")
    )
    assert list(d.initial_values) == list(init)

    s.K1 = 0.2
    d._precompute()
    assert list(d.initial_values) == list(init)
    d.pt1.solution["inwards.K1"] = 0.2
    d._precompute()
    assert list(d.initial_values) == [0.2, init[1], init[2], init[3]]

    # Children without iterative
    s = Multiply2("MyMult")
    d = NonLinearSolver("run_drivers")
    s.add_driver(d)
    p1 = RunSingleCase("pt1")
    d.add_child(p1)
    p1.offdesign.add_unknown("p_in.x").add_equation("p_in.x == 2.")
    p1.design.add_unknown("K1").add_equation("p_out.x == 20.")
    p2 = RunOnce("pt2")
    d.add_child(p2)
    caplog.clear()
    with caplog.at_level(logging.WARNING, NonLinearSolver.__name__):
        s.call_setup_run()
        d._precompute()

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
    s = Multiply2("MyMult")
    d = NonLinearSolver("run_drivers")
    s.add_driver(d)
    d.runner.offdesign.add_unknown(["p_in.x", "K1", "K2"])
    s.call_setup_run()
    d._precompute()

    caplog.clear()
    with caplog.at_level(logging.ERROR):
        with pytest.raises(
            ArithmeticError,
            match=r"Nonlinear problem \w+ error: Mismatch between numbers of params \[\d\] and residues \[\d\]",
        ):
            d.compute()
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
    s = Multiply2("MyMult")
    # Set max_iter to force failure
    d = NonLinearSolver("run_drivers", method=NonLinearMethods.NR, max_iter=0)
    s.add_driver(d)
    d.children[d._default_driver_name].offdesign.add_unknown("p_in.x").add_equation(
        "p_in.x == 2."
    )
    d.children[d._default_driver_name].design.add_unknown("K1").add_equation(
        "p_out.x == 20."
    )
    init = np.random.rand(2)
    d.children[d._default_driver_name].set_init({"K1": init[0], "p_in.x": init[1]})
    s.call_setup_run()
    d._precompute()
    caplog.clear()
    with caplog.at_level(logging.ERROR):
        d.compute()

    error_messages = list(
        map(
            lambda r: r.msg,
            filter(lambda r: r.levelno == logging.ERROR, caplog.records),
        )
    )
    assert len(error_messages) == 1
    assert re.search(r"The solver failed: .*", error_messages[0]) is not None

    all_variables = [key for key in d.problem.unknowns]
    expected = dict([(var, init[i]) for i, var in enumerate(all_variables)])
    assert d.solution == expected

    # Multiple cases
    s = Multiply2("MyMult")
    d = NonLinearSolver("run_drivers", method=NonLinearMethods.NR, max_iter=0)
    s.add_driver(d)
    p1 = RunSingleCase("pt1")
    d.add_child(p1)
    p1.offdesign.add_unknown("p_in.x").add_equation("p_in.x == 2.")
    p1.design.add_unknown("K1").add_equation("p_out.x == 20.")
    p2 = RunSingleCase("pt2")
    d.add_child(p2)
    p2.offdesign.add_unknown("p_in.x").add_equation("p_in.x == 6.")
    p2.design.add_unknown("K2").add_equation("p_out.x == 15.")
    init = np.random.rand(4)
    d.children["pt1"].set_init({"K1": init[0], "p_in.x": init[1]})
    d.children["pt2"].set_init({"K2": init[2], "p_in.x": init[3]})
    s.call_setup_run()
    d._precompute()
    caplog.clear()
    with caplog.at_level(logging.ERROR):
        d.compute()

    error_messages = list(
        map(
            lambda r: r.msg,
            filter(lambda r: r.levelno == logging.ERROR, caplog.records),
        )
    )
    assert len(error_messages) == 1
    assert re.search(r"The solver failed: .*", error_messages[0]) is not None

    all_variables = [key for key in d.problem.unknowns]
    expected = dict([(var, init[i]) for i, var in enumerate(all_variables)])
    assert d.solution == expected

    s = Multiply2("MyMult")
    d = NonLinearSolver("run_drivers", method=NonLinearMethods.NR, max_iter=0)
    s.add_driver(d)
    d.children[d._default_driver_name].offdesign.add_unknown("p_in.x").add_equation(
        "p_in.x == 2."
    )
    d.children[d._default_driver_name].design.add_unknown("K1").add_equation(
        "p_out.x == 20."
    )
    init = np.random.rand(2)
    d.children[d._default_driver_name].set_init({"K1": init[0], "p_in.x": init[1]})

    s.call_setup_run()
    d._precompute()
    caplog.clear()
    with caplog.at_level(logging.ERROR):
        d.compute()

    error_messages = list(filter(lambda r: r.levelno == logging.ERROR, caplog.records))
    assert len(error_messages) == 1

    all_variables = [key for key in d.problem.unknowns]
    expected = dict([(var, init[i]) for i, var in enumerate(all_variables)])
    assert d.solution == expected

    # Child without iterative
    s = System("empty")
    d = NonLinearSolver("run_drivers")
    s.add_driver(d)
    pt = RunSingleCase("pt")
    d.add_child(pt)
    s.call_setup_run()
    d._precompute()

    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        d.compute()

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
            debug_messages[-1],
        )
        is not None
    )


def test_NonLinearSolver_compute(caplog, set_master_system):
    # Single case
    s = Multiply2("MyMult")
    d = NonLinearSolver("run_drivers", method=NonLinearMethods.POWELL)
    s.add_driver(d)
    d.children[d._default_driver_name].offdesign.add_unknown("p_in.x").add_equation(
        "p_in.x == 2."
    )
    d.children[d._default_driver_name].design.add_unknown("K1").add_equation(
        "p_out.x == 20."
    )
    init = np.random.rand(2)
    d.children[d._default_driver_name].set_init({"K1": init[0], "p_in.x": init[1]})
    s.call_setup_run()
    d._precompute()
    caplog.clear()
    with caplog.at_level(logging.INFO):
        d.compute()

    info_messages = list(
        map(
            lambda r: r.msg, filter(lambda r: r.levelno == logging.INFO, caplog.records)
        )
    )
    assert len(info_messages) == 1
    assert (
        re.search(r"solver : \w+The solution converged\.", info_messages[0]) is not None
    )

    assert len(d.solution) == len(init)
    np.testing.assert_allclose(list(d.solution.values()), (2.0, 2.0))


def test_NonLinearSolver__postcompute(set_master_system):
    # Simple system with no design equations
    s = Multiply2("MyMult")
    d = NonLinearSolver("run_drivers")
    s.add_driver(d)
    d.runner.offdesign.add_unknown(["K1", "K2", "p_in.x"])
    s.call_setup_run()
    d._precompute()
    d._postcompute()
    vars = [
        f"{d._default_driver_name}[{u}]"
        for u in ("inwards.K1", "inwards.K2", "p_in.x")
    ]
    assert set(d.problem.unknowns) == set(vars)

    # More realistic single case
    s = Multiply2("MyMult")
    d = NonLinearSolver("run_drivers")
    s.add_driver(d)
    d.children[d._default_driver_name].offdesign.add_unknown("p_in.x").add_equation(
        "p_in.x == 2."
    )
    d.children[d._default_driver_name].design.add_unknown("K1").add_equation(
        "p_out.x == 20."
    )
    s.call_setup_run()
    d._precompute()
    d._postcompute()
    assert set(d.problem.unknowns), set(
        ("inwards.K1", d._default_driver_name + "[p_in.x]")
    )

    # Multiple cases
    s = Multiply2("MyMult")
    d = NonLinearSolver("run_drivers")
    s.add_driver(d)
    p1 = RunSingleCase("pt1")
    d.add_child(p1)
    p1.offdesign.add_unknown("p_in.x").add_equation("p_in.x == 2.")
    p1.design.add_unknown("K1").add_equation("p_out.x == 20.")
    p2 = RunSingleCase("pt2")
    d.add_child(p2)
    p2.offdesign.add_unknown("p_in.x").add_equation("p_in.x == 6.")
    p2.design.add_unknown("K2").add_equation("p_out.x == 15.")
    s.call_setup_run()
    d._precompute()
    d._postcompute()
    assert set(d.problem.unknowns) == set(
        ("inwards.K1", "pt1[p_in.x]", "inwards.K2", "pt2[p_in.x]")
    )


def test_NonLinearSolver_residues_singlePoint(set_master_system):
    # Simple system with no design equations
    s = Multiply2("MyMult")
    d = NonLinearSolver("run_drivers")
    s.add_driver(d)
    d.runner.offdesign.add_unknown(["K1", "K2"])
    s.call_setup_run()
    d._precompute()
    init = np.random.rand(len(d.problem.unknowns))
    d._fresidues(init, "init")
    assert isinstance(d.problem.residues, OrderedDict)
    assert len(d.problem.residues) == 0

    # More realistic single case
    s = Multiply2("MyMult")
    d = NonLinearSolver("run_drivers")
    s.add_driver(d)
    d.runner.offdesign.add_unknown("p_in.x")
    d.children[d._default_driver_name].offdesign.add_equation("p_in.x == 2.")
    d.children[d._default_driver_name].design.add_unknown("K1").add_equation(
        "p_out.x == 20."
    )
    d.owner.call_setup_run()
    d._precompute()
    d.compute()
    assert set(d.problem.residues) == set(f"{d._default_driver_name}[{eq}]"
            for eq in ("p_in.x == 2.", "p_out.x == 20.",)
    )
    for v in d.problem.residues.values():
        assert isinstance(v, Residue)


def test_NonLinearSolver_residues_multiPoint(set_master_system):
    s = Multiply2("MyMult")
    d = NonLinearSolver("run_drivers")
    s.add_driver(d)
    # First point
    p1 = d.add_child(RunSingleCase("pt1"))
    p1.offdesign.add_unknown("p_in.x").add_equation("p_in.x == 2.")
    m1 = p1.design.add_unknown("K1").add_equation("p_out.x == 20.")
    # Second point
    p2 = d.add_child(RunSingleCase("pt2"))
    p2.offdesign.add_unknown("p_in.x").add_equation("p_in.x == 6.")
    d.owner.call_setup_run()
    d._precompute()
    d.compute()
    assert set(d.problem.residues) == set(
        (
            "{}[{}]".format("pt1", "p_in.x == 2."),
            "{}[{}]".format("pt1", "p_out.x == 20."),
            "{}[{}]".format("pt2", "p_in.x == 6."),
        )
    )
    for v in d.problem.residues.values():
        assert isinstance(v, Residue)


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
    case = RunSingleCase("case")
    solver.add_child(case)
    case.offdesign.add_unknown("one.a[1]").add_equation("one.out.x[1] == 42.")
    s.run_drivers()
    assert np.allclose(one.a, np.r_[1.0, 7.0, 1.0])

    solver.pop_child("case")
    case = RunSingleCase("case")
    solver.add_child(case)
    case.offdesign.add_unknown(["one.a[1]", "two.a"]).add_equation(
        [
            {"equation": "one.out.x[1] == 9"},
            {"equation": "out.x == array([5., 1., 3.])"},
        ]
    )
    s.run_drivers()
    assert np.allclose(one.a, np.r_[1.0, 1.5, 1.0])
    assert np.allclose(two.a, np.r_[5.0 / 3.0, 1.0 / 9.0, 1.0 / 3.0])

    one.a = np.r_[1.0, 1.0, 1.0]
    two.a = np.r_[1.0, 1.0, 1.0]
    solver.pop_child("case")
    case = RunSingleCase("case")
    solver.add_child(case)
    case.design.add_unknown("one.a[1]").add_equation("one.out.x[1] == 9")
    case.offdesign.add_unknown("two.a").add_equation("out.x == array([5., 1., 3.])")
    s.run_drivers()
    assert np.allclose(one.a, np.r_[1.0, 1.5, 1.0])
    assert np.allclose(two.a, np.r_[5.0 / 3.0, 1.0 / 9.0, 1.0 / 3.0])

    one.a = np.r_[1.0, 1.0, 1.0]
    two.a = np.r_[1.0, 1.0, 1.0]
    solver.pop_child("case")
    case = RunSingleCase("case")
    solver.add_child(case)
    case.design.add_unknown(["one.a[1]", "two.a"]).add_equation(
        ["one.out.x[1] == 9", "out.x == array([5., 1., 3.])"]
    )
    s.run_drivers()
    assert np.allclose(one.a, np.r_[1.0, 1.5, 1.0])
    assert np.allclose(two.a, np.r_[5.0 / 3.0, 1.0 / 9.0, 1.0 / 3.0])

    # Test iterative connector with vector variable
    s = System("hat")
    one = s.add_child(Merger1d("one"), pulling={"in1": "in_"})
    two = s.add_child(Splitter1d("two"), pulling={"out1": "out"})
    s.connect(two.in_, one.out)
    s.connect(one.in2, two.out2)
    s.drivers.clear()
    solver = s.add_driver(NonLinearSolver("solver", tol=1e-7))
    case = RunSingleCase("case")
    solver.add_child(case)

    s.run_drivers()
    assert np.allclose(one.out.x, s.in_.x / two.s)
    assert np.allclose(s.out.x, s.in_.x)


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

    s.exec_order = ["p1", "sp", "p2", "p3", "mx", "p4"]
    s.run_once()

    assert s.p4.flnum_out.Pt == 74800.0

    design = s.add_driver(NonLinearSolver("design", tol=5.0e-8))
    design.runner.offdesign.add_unknown("sp.x")

    # System value
    s.run_drivers()
    np.testing.assert_array_almost_equal(design.initial_values, [0.8])
    np.testing.assert_allclose(
        list(design.children[design._default_driver_name].solution.values()),
        [0.5],
        rtol=1e-4,
    )

    # Reuse solution
    s.run_drivers()
    np.testing.assert_array_almost_equal(design.initial_values, [0.5])
    np.testing.assert_allclose(
        list(design.children[design._default_driver_name].solution.values()),
        [0.5],
        rtol=1e-4,
    )

    # Initial value
    design.children[design._default_driver_name].set_init({"sp.inwards.x": 2.0})
    s.run_drivers()
    np.testing.assert_array_almost_equal(design.initial_values, [2])
    np.testing.assert_allclose(
        list(design.children[design._default_driver_name].solution.values()),
        [0.5],
        rtol=1e-4,
    )

    # Reuse solution
    s.run_drivers()
    np.testing.assert_array_almost_equal(design.initial_values, [0.5])
    np.testing.assert_allclose(
        list(design.children[design._default_driver_name].solution.values()),
        [0.5],
        rtol=1e-4,
    )


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
