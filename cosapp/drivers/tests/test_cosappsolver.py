import pytest

import logging
import re
import numpy as np
from scipy.linalg import lu_factor

from cosapp.core.numerics import root
from cosapp.drivers import NonLinearMethods, NonLinearSolver, RunSingleCase
from cosapp.systems import System
from cosapp.tests.library.systems import (
    IterativeNonLinear,
    Multiply2,
    MultiplyVector2,
    MultiplyVector3,
    NonLinear3,
)

@pytest.fixture
def caplog_messages(caplog):
    """Extracts record messages from fixture `caplog`"""
    def extractor(level):
        records = filter(lambda r: r.levelno == level, caplog.records)
        return [record.msg for record in records]
    return extractor


@pytest.mark.parametrize("settings, rtol", [
    (dict(), 1e-15),  # default settings
    (dict(method=NonLinearMethods.NR, tol='auto'), 1e-15),
    (dict(method=NonLinearMethods.NR, tol=1e-6), 1e-8),
])
def test_singlept1(set_master_system, settings, rtol):
    s = Multiply2("MyMult")
    d = s.add_driver(NonLinearSolver("solver", **settings))

    s.p_in.x = 1.0
    s.K1 = s.K2 = 1.0

    d.add_unknown("K1").add_equation("p_out.x == 100")
    s.call_setup_run()
    d.run_once()
    assert s.K1 == pytest.approx(100, rel=rtol)


def test_singlept2(set_master_system):
    s = Multiply2("MyMult")
    d = s.add_driver(NonLinearSolver("solver", method=NonLinearMethods.NR))

    s.p_in.x = 1.0
    s.K1 = s.K2 = 1.0

    d.add_unknown(["K1", "K2"]).add_equation(["p_out.x == 100", "K2 == K1"])

    s.K1 = 1.0
    s.call_setup_run()
    d.run_once()

    assert s.K1 == pytest.approx(10, rel=1e-15)
    assert s.K2 == pytest.approx(10, rel=1e-15)


def test_multipts1():
    s = MultiplyVector2("multvector")

    design = s.add_driver(NonLinearSolver("design", method=NonLinearMethods.NR))

    run1 = design.add_child(RunSingleCase("run1"))

    s.k1 = 10.0
    s.k2 = 8.0

    run1.set_values({"p_in.x1": 4.0, "p_in.x2": 10.0})
    run1.design.add_unknown("k1").add_equation("p_out.x == 100")

    s.run_drivers()
    assert s.inwards.k1 == pytest.approx(5.0, abs=1e-4)

    run2 = design.add_child(RunSingleCase("run2"))

    run2.set_values({"p_in.x1": 2, "p_in.x2": 8.0})
    run2.design.add_unknown("k2").add_equation("p_out.x == 76")

    s.run_drivers()
    assert s.inwards.k1 == pytest.approx(10 / 3, abs=1e-3)
    assert s.inwards.k2 == pytest.approx(26 / 3, abs=1e-3)


def test_multipts2():
    sys = System("s")
    sys.add_child(MultiplyVector3("s3"))

    design = sys.add_driver(NonLinearSolver("design", method=NonLinearMethods.NR))

    run1 = design.add_child(RunSingleCase("run1"))
    run2 = design.add_child(RunSingleCase("run2"))
    run3 = design.add_child(RunSingleCase("run3"))

    run1.set_values({"s3.p_in.x1": 4.0, "s3.p_in.x2": 10.0, "s3.p_in.x3": 1.0})
    run1.design.add_unknown("s3.k1").add_equation("s3.p_out.x == 100")

    run2.set_values({"s3.p_in.x1": 2, "s3.p_in.x2": 8.0, "s3.p_in.x3": 1.0})
    run2.design.add_unknown("s3.k2").add_equation("s3.p_out.x == 76")

    run3.set_values({"s3.p_in.x1": 5, "s3.p_in.x2": 12.0, "s3.p_in.x3": 1.0})
    run3.design.add_unknown("s3.k3").add_equation("s3.p_out.x == 150")

    sys.run_drivers()
    assert sys.s3.k1 == pytest.approx(-26, abs=1e-3)
    assert sys.s3.k2 == pytest.approx(38.0, abs=1e-3)
    assert sys.s3.k3 == pytest.approx(-176.0, abs=1e-2)


def test_multipts_nonlinear3():
    sys = System("s")
    sys.add_child(NonLinear3("s3"))

    design = sys.add_driver(NonLinearSolver("design", method=NonLinearMethods.NR))

    run1 = design.add_child(RunSingleCase("run1"))
    run2 = design.add_child(RunSingleCase("run2"))
    run3 = design.add_child(RunSingleCase("run3"))

    run1.set_values({"s3.p_in.x1": 4.0, "s3.p_in.x2": 10.0, "s3.p_in.x3": 1.0})
    run1.design.add_unknown("s3.k1").add_equation("s3.p_out.x == 100")

    run2.set_values({"s3.p_in.x1": 2, "s3.p_in.x2": 8.0, "s3.p_in.x3": 1.0})
    run2.design.add_unknown("s3.k2").add_equation("s3.p_out.x == 76")

    run3.set_values({"s3.p_in.x1": 5, "s3.p_in.x2": 12.0, "s3.p_in.x3": 1.0})
    run3.design.add_unknown("s3.k3").add_equation("s3.p_out.x == 150")

    sys.run_drivers()
    assert sys.s3.k1 == pytest.approx(227.4029139, abs=1e-2)
    assert sys.s3.k2 == pytest.approx(72465.89971, abs=10)
    assert sys.s3.k3 == pytest.approx(-26454.1762, abs=10)


def test_multipts_iterative_nonlinear():
    snl = IterativeNonLinear("nl")

    design = snl.add_driver(NonLinearSolver("design", method=NonLinearMethods.NR))

    snl.splitter.inwards.split_ratio = 0.1
    snl.mult2.inwards.K1 = 1
    snl.mult2.inwards.K2 = 1
    snl.nonlinear.inwards.k1 = 1
    snl.nonlinear.inwards.k2 = 0.5

    run1 = design.add_child(RunSingleCase("run1"))
    run2 = design.add_child(RunSingleCase("run2"))

    run1.set_values({"p_in.x": 1.0})
    run1.design.add_unknown("nonlinear.inwards.k1").add_equation("splitter.p2_out.x == 10")

    run2.set_values({"p_in.x": 10.0})
    run2.design.add_unknown(
        ["mult2.inwards.K1", "nonlinear.inwards.k2", "splitter.inwards.split_ratio"]
    ).add_equation(
        ["splitter.p2_out.x == 50", "merger.p_out.x == 30", "splitter.p1_out.x == 5"]
    )

    snl.run_drivers()

    assert snl.mult2.K1 == pytest.approx(1.833333333, abs=1e-4)
    assert snl.nonlinear.k1 == pytest.approx(5.0, abs=1e-4)
    assert snl.nonlinear.k2 == pytest.approx(0.861353116, abs=1e-4)
    assert snl.splitter.split_ratio == pytest.approx(0.090909091, abs=1e-4)


def test_completejacobian(caplog, caplog_messages):
    s = Multiply2("MyMult")
    d = s.add_driver(
        NonLinearSolver("solver", method=NonLinearMethods.NR, verbose=True)
    )

    s.p_in.x = 2.0
    s.K1 = s.K2 = 1.0

    d.add_unknown("inwards.K1").add_equation("p_out.x == 100")

    caplog.clear()
    with caplog.at_level(logging.INFO, root.__name__):
        s.run_drivers()
    assert s.K1 == pytest.approx(50, rel=1e-5)

    info_messages = caplog_messages(logging.INFO)
    assert "Jacobian matrix: full update" in info_messages


def test_reusejacobian(caplog, caplog_messages, set_master_system):
    s = Multiply2("MyMult")
    d = s.add_driver(NonLinearSolver("solver", method=NonLinearMethods.NR))

    s.p_in.x = 1.0
    s.K1 = 1.0
    s.K2 = 1.0

    d.add_unknown("inwards.K1").add_equation("p_out.x == 100")
    s.call_setup_run()
    d.run_once()
    assert s.K1 == pytest.approx(100)

    s.K1 = 1.0
    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        d.run_once()
    assert s.K1 == pytest.approx(100)

    info_messages = caplog_messages(logging.INFO)
    assert len(info_messages) >= 2
    matches = map(
        lambda message: re.search(
            r"Converged \((\d)+(?:\.\d*)(?:[eE][+-]\d+)\) in \d+ iterations, 0 complete, 0 partial Jacobian and 0 Broyden",
            message,
        ),
        info_messages,
    )
    assert any(matches)

    debug_messages = caplog_messages(logging.DEBUG)
    assert "Reuse of previous Jacobian matrix" in debug_messages


def test_partialjacobian(caplog, caplog_messages, set_master_system):
    s = Multiply2("MyMult")
    d = s.add_driver(
        NonLinearSolver("solver", method=NonLinearMethods.NR, verbose=True)
    )

    s.p_in.x = 1.0
    s.K1 = s.K2 = 1.0

    d.options["tol"] = 1e-6
    d.add_unknown("inwards.K1").add_equation("p_out.x == 100")
    s.call_setup_run()
    d.run_once()
    assert s.K1 == pytest.approx(100)

    s.K1 = 1.0
    d.jac = np.linalg.inv(np.array([[10.0]]))
    d.jac_lup = lu_factor(d.jac)
    d.runner.solution.clear()

    caplog.clear()
    with caplog.at_level(logging.INFO):
        d.run_once()
    assert s.K1 == pytest.approx(100)

    info_messages = caplog_messages(logging.INFO)
    assert len(info_messages) >= 2
    matches = map(
        lambda r: re.search(r"Jacobian matrix: \d+ over \d+ derivative\(s\) updated", r),
        info_messages,
    )
    assert not any(matches)
    assert "Jacobian matrix: full update" in info_messages
    matches = map(
        lambda r: re.search(
            r"Converged \((\d)+(?:\.\d*)(?:[eE][+-]\d+)\) in \d+ iterations, 1 complete, 0 partial Jacobian and 0 Broyden",
            r,
        ),
        info_messages,
    )
    assert any(matches)


def test_partialjacobian_coupledmatrix(caplog, caplog_messages, set_master_system):
    """Trivial linear problem with imposed incorrect Jacobian matrix"""
    s = Multiply2("MyMult")
    d = s.add_driver(
        NonLinearSolver("solver", method=NonLinearMethods.NR, factor=0.1)
    )

    s.p_in.x = 1.0
    s.K1 = s.K2 = 1.0

    run = d.add_child(RunSingleCase("run"))
    run.add_unknown(["K1", "K2"])
    run.add_equation("K1 == 100", reference="norm")
    run.add_equation("K2 == 50", reference="norm")
    s.call_setup_run()
    # Set tolerance level for Jacobian update criterion
    d.options['jac_update_tol'] = 0.05
    d.run_once()
    assert s.K1 == pytest.approx(100, rel=1e-5)
    assert s.K2 == pytest.approx(50, rel=1e-5)

    s.K1 = s.K2 = 1.0

    d.jac = np.array([[1, 0], [0, 0.077]])
    d.jac_lup = lu_factor(d.jac)
    d.run.solution.clear()

    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        d.run_once()
    
    info_messages = caplog_messages(logging.INFO)
    assert len(info_messages) >= 2
    matches = map(
        lambda r: re.search(
            r"Converged \((\d)+(?:\.\d*)(?:[eE][+-]\d+)\) in \d+ iterations, 0 complete, 1 partial Jacobian and \d+ Broyden",
            r,
        ),
        info_messages,
    )
    assert any(matches)

    debug_messages = caplog_messages(logging.DEBUG)
    jacobian_related = list(filter(lambda msg: "Jacobian" in msg, debug_messages))
    assert "Jacobian matrix: 1 over 2 derivative(s) updated" in jacobian_related
    assert "Reuse of previous Jacobian matrix" in jacobian_related
    assert "Perturb unknown 0" in debug_messages


def test_partialjacobian_independentmatrix(caplog, caplog_messages, set_master_system):
    """Trivial linear problem with imposed, incorrect Jacobian matrix"""
    s = Multiply2("MyMult")
    d = s.add_driver(
        NonLinearSolver("solver", method=NonLinearMethods.NR, factor=1.0, tol='auto')
    )

    s.p_in.x = 1.0
    s.K1 = 1.0
    s.K2 = 1.0

    d.add_unknown(["K1", "K2"])
    d.add_equation("K1 == 100", reference=1.0)
    d.add_equation("K2 == 50", reference=1.0)
    s.call_setup_run()
    # Set tolerance level for Jacobian update criterion
    d.options['jac_update_tol'] = 0.1
    d.run_once()
    assert s.K1 == pytest.approx(100, rel=1e-5)
    assert s.K2 == pytest.approx(50, rel=1e-5)

    s.K1 = s.K2 = 1.0
    # Impose Jacobian matrix; exact matrix is identity
    d.jac = np.array([[1, 0], [0, -3.2]])
    d.jac_lup = lu_factor(d.jac)
    d.runner.solution.clear()

    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        d.run_once()
    
    info_messages = caplog_messages(logging.INFO)
    assert len(info_messages) >= 2
    matches = map(
        lambda r: re.search(
            r"Converged \((\d)+(?:\.\d*)(?:[eE][+-]\d+)\) in \d+ iterations, 0 complete, 1 partial Jacobian and \d+ Broyden",
            r,
        ),
        info_messages,
    )
    # print("\n".join(info_messages))
    assert any(matches)

    debug_messages = caplog_messages(logging.DEBUG)
    jacobian_related = list(filter(lambda msg: "Jacobian" in msg, debug_messages))
    assert "Jacobian matrix: 1 over 2 derivative(s) updated" in jacobian_related
    assert "Reuse of previous Jacobian matrix" in jacobian_related
    assert "Perturb unknown 1" in debug_messages


def test_NumericalSolver_linear_nonlinear_diag(caplog, caplog_messages, set_master_system):
    """
    Test related to issue https://gitlab.com/cosapp/cosapp/-/issues/22
    Mathematical problem with one linear equation, and one highly nonlinear equation.
    Each residue depends on one parameter only, such that the jacobian matrix
    remains diagonal.
    """
    s = Multiply2("MyMult")
    solver = s.add_driver(
        NonLinearSolver("solver", method=NonLinearMethods.NR, tol=1e-6)
    )

    solver.add_unknown(["K1", "K2"]).add_equation(["K1 == 2", "K2**4 == 1"])

    s.call_setup_run()
    s.p_in.x = 1.0
    s.K1 = s.K2 = 100.0  # start far from solution (2, 1)

    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        solver.run_once()

    assert s.K1 == pytest.approx(2)
    assert s.K2 == pytest.approx(1)
    
    info_messages = caplog_messages(logging.INFO)
    debug_messages = caplog_messages(logging.DEBUG)
    jacobian_related = list(filter(lambda msg: "Jacobian" in msg, debug_messages))
    assert "Jacobian matrix: 1 over 2 derivative(s) updated" in jacobian_related
    assert "Perturb unknown 1" in debug_messages


def test_NumericalSolver_linear_nonlinear(caplog, caplog_messages, set_master_system):
    """
    Test related to issue https://gitlab.com/cosapp/cosapp/-/issues/22
    Mathematical problem with one linear equation, and one highly nonlinear equation.
    The nonlinear residue depends on both unknowns, such that no partial jacobian
    update is possible, even though the other residue is linear.
    """
    s = Multiply2("MyMult")
    solver = s.add_driver(
        NonLinearSolver("solver", method=NonLinearMethods.NR, tol=1e-6)
    )

    solver.add_unknown(["K1", "K2"]).add_equation(["K1 == 2", "K1 * K2**4 == 2"])

    s.call_setup_run()
    s.p_in.x = 1.0
    s.K1 = s.K2 = 100.0  # start far from solution (2, 1)

    caplog.clear()
    with caplog.at_level(logging.INFO):
        solver.run_once()

    assert s.K1 == pytest.approx(2)
    assert s.K2 == pytest.approx(1)
    
    matches = map(
        lambda record: re.search(
            r"Converged \((\d)+(?:\.\d*)(?:[eE][+-]\d+)\) in \d+ iterations, \d+ complete, 0 partial Jacobian and \d+ Broyden",
            record.msg,
        ),
        caplog.records,
    )
    assert any(matches)
