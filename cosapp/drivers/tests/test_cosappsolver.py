import logging
from collections import OrderedDict
import re

import numpy as np
import pytest
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


def test_singlept1(set_master_system):
    s = Multiply2("MyMult")
    d = s.add_driver(NonLinearSolver("run_drivers", method=NonLinearMethods.NR))

    s.p_in.x = 1.0
    s.inwards.K1 = 1.0
    s.inwards.K2 = 1.0

    d.add_child(RunSingleCase("run")).design.add_unknown("inwards.K1").add_equation(
        "p_out.x == 100."
    )
    s.call_setup_run()
    d.run_once()
    assert s.inwards.K1 == pytest.approx(100.0, abs=1e-3)


def test_singlept2(set_master_system):
    s = Multiply2("MyMult")
    d = s.add_driver(NonLinearSolver("run_drivers", method=NonLinearMethods.NR))

    s.p_in.x = 1.0
    s.inwards.K1 = 1.0
    s.inwards.K2 = 1.0

    d.add_child(RunSingleCase("run")).design.add_unknown(
        ["inwards.K1", "inwards.K2"]
    ).add_equation(["p_out.x == 100.", "inwards.K2 == inwards.K1"])

    s.inwards.K1 = 1.0
    s.call_setup_run()
    d.run_once()

    assert s.inwards.K1 == pytest.approx(10.0, abs=1e-4)
    assert s.inwards.K2 == pytest.approx(10.0, abs=1e-4)


def test_multipts1():
    s2 = MultiplyVector2("multvector")

    design = s2.add_driver(NonLinearSolver("design", method=NonLinearMethods.NR))

    run1 = design.add_child(RunSingleCase("run 1"))

    s2.inwards.k1 = 10.0
    s2.inwards.k2 = 8.0

    run1.set_values({"p_in.x1": 4.0, "p_in.x2": 10.0})
    run1.design.add_unknown("inwards.k1").add_equation("p_out.x == 100.")

    s2.run_drivers()
    assert s2.inwards.k1 == pytest.approx(5.0, abs=1e-4)

    run2 = design.add_child(RunSingleCase("run 2"))

    run2.set_values({"p_in.x1": 2, "p_in.x2": 8.0})
    run2.design.add_unknown("inwards.k2").add_equation("p_out.x == 76.")

    s2.run_drivers()
    assert s2.inwards.k1 == pytest.approx(10 / 3, abs=1e-3)
    assert s2.inwards.k2 == pytest.approx(26 / 3, abs=1e-3)


def test_multipts2():
    sys = System("s")
    sys.add_child(MultiplyVector3("s3"))

    design = sys.add_driver(NonLinearSolver("design", method=NonLinearMethods.NR))

    run1 = design.add_child(RunSingleCase("run 1"))
    run2 = design.add_child(RunSingleCase("run 2"))
    run3 = design.add_child(RunSingleCase("run 3"))

    run1.set_values({"s3.p_in.x1": 4.0, "s3.p_in.x2": 10.0, "s3.p_in.x3": 1.0})
    run1.design.add_unknown("s3.inwards.k1").add_equation("s3.p_out.x == 100.")

    run2.set_values({"s3.p_in.x1": 2, "s3.p_in.x2": 8.0, "s3.p_in.x3": 1.0})
    run2.design.add_unknown("s3.inwards.k2").add_equation("s3.p_out.x == 76.")

    run3.set_values({"s3.p_in.x1": 5, "s3.p_in.x2": 12.0, "s3.p_in.x3": 1.0})
    run3.design.add_unknown("s3.inwards.k3").add_equation("s3.p_out.x == 150.")

    sys.run_drivers()
    assert sys.s3.inwards.k1 == pytest.approx(-26, abs=1e-3)
    assert sys.s3.inwards.k2 == pytest.approx(38.0, abs=1e-3)
    assert sys.s3.inwards.k3 == pytest.approx(-176.0, abs=1e-2)


def test_multipts_nonlinear3():
    sys = System("s")
    sys.add_child(NonLinear3("s3"))

    design = sys.add_driver(NonLinearSolver("design", method=NonLinearMethods.NR))

    run1 = design.add_child(RunSingleCase("run 1"))
    run2 = design.add_child(RunSingleCase("run 2"))
    run3 = design.add_child(RunSingleCase("run 3"))

    run1.set_values({"s3.p_in.x1": 4.0, "s3.p_in.x2": 10.0, "s3.p_in.x3": 1.0})
    run1.design.add_unknown("s3.inwards.k1").add_equation("s3.p_out.x == 100.")

    run2.set_values({"s3.p_in.x1": 2, "s3.p_in.x2": 8.0, "s3.p_in.x3": 1.0})
    run2.design.add_unknown("s3.inwards.k2").add_equation("s3.p_out.x == 76.")

    run3.set_values({"s3.p_in.x1": 5, "s3.p_in.x2": 12.0, "s3.p_in.x3": 1.0})
    run3.design.add_unknown("s3.inwards.k3").add_equation("s3.p_out.x == 150.")

    sys.run_drivers()
    assert sys.s3.inwards.k1 == pytest.approx(227.4029139, abs=1e-2)
    assert sys.s3.inwards.k2 == pytest.approx(72465.89971, abs=10)
    assert sys.s3.inwards.k3 == pytest.approx(-26454.1762, abs=10)


def test_multipts_iterative_non_linear():
    snl = IterativeNonLinear("nl")

    design = snl.add_driver(NonLinearSolver("design", method=NonLinearMethods.NR))

    snl.splitter.inwards.split_ratio = 0.1
    snl.mult2.inwards.K1 = 1
    snl.mult2.inwards.K2 = 1
    snl.nonlinear.inwards.k1 = 1
    snl.nonlinear.inwards.k2 = 0.5

    run1 = design.add_child(RunSingleCase("run 1"))
    run2 = design.add_child(RunSingleCase("run 2"))

    run1.set_values({"p_in.x": 1.0})
    run1.design.add_unknown("nonlinear.inwards.k1").add_equation(
        "splitter.p2_out.x == 10."
    )

    run2.set_values({"p_in.x": 10.0})
    run2.design.add_unknown(
        ["mult2.inwards.K1", "nonlinear.inwards.k2", "splitter.inwards.split_ratio"]
    ).add_equation(
        ["splitter.p2_out.x == 50.", "merger.p_out.x == 30.", "splitter.p1_out.x == 5."]
    )

    snl.run_drivers()

    assert snl.mult2.inwards.K1 == pytest.approx(1.833333333, abs=1e-4)
    assert snl.nonlinear.inwards.k1 == pytest.approx(5.0, abs=1e-4)
    assert snl.nonlinear.inwards.k2 == pytest.approx(0.861353116, abs=1e-4)
    assert snl.splitter.inwards.split_ratio == pytest.approx(0.090909091, abs=1e-4)


def test_completejacobian(caplog, set_master_system):
    s = Multiply2("MyMult")
    d = s.add_driver(
        NonLinearSolver("run_drivers", method=NonLinearMethods.NR, verbose=True)
    )

    s.p_in.x = 1.0
    s.inwards.K1 = 1.0
    s.inwards.K2 = 1.0

    d.add_child(RunSingleCase("run")).design.add_unknown("inwards.K1").add_equation(
        "p_out.x == 100."
    )

    caplog.clear()
    with caplog.at_level(logging.INFO, root.__name__):
        s.call_setup_run()
        d.run_once()
    assert s.inwards.K1 == pytest.approx(100.0, abs=1e-3)

    assert "Jacobian matrix: full update" in map(
        lambda r: r.msg, filter(lambda r: r.levelno == logging.INFO, caplog.records)
    )


def test_reusejacobian(caplog, set_master_system):
    s = Multiply2("MyMult")
    d = s.add_driver(NonLinearSolver("run_drivers", method=NonLinearMethods.NR))

    s.p_in.x = 1.0
    s.inwards.K1 = 1.0
    s.inwards.K2 = 1.0

    d.add_child(RunSingleCase("run")).design.add_unknown("inwards.K1").add_equation(
        "p_out.x == 100."
    )
    s.call_setup_run()
    d.run_once()

    s.inwards.K1 = 1.0
    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        d.run_once()
    assert s.inwards.K1 == pytest.approx(100.0, abs=1e-3)

    info_messages = list(
        map(
            lambda r: r.msg,
            filter(lambda r: r.levelno == logging.INFO, caplog.records),
        )
    )
    assert len(info_messages) >= 2
    matches = map(
        lambda r: re.search(
            r"   -> Converged \((\d)+(?:\.\d*)(?:[eE][-]\d+)\) in \d+ iterations, 0 complete, 0 partial Jacobian and 0 Broyden evaluation\(s\)",
            r,
        ),
        info_messages,
    )
    assert any(matches)

    debug_messages = list(
        map(
            lambda r: r.msg,
            filter(lambda r: r.levelno == logging.DEBUG, caplog.records),
        )
    )
    assert "Reuse of previous Jacobian matrix" in debug_messages


def test_partialjacobian(caplog, set_master_system):
    s = Multiply2("MyMult")
    d = s.add_driver(
        NonLinearSolver("run_drivers", method=NonLinearMethods.NR, verbose=True)
    )

    s.p_in.x = 1.0
    s.inwards.K1 = 1.0
    s.inwards.K2 = 1.0

    d.options["tol"] = 1e-5
    d.add_child(RunSingleCase("run")).design.add_unknown("inwards.K1").add_equation(
        "p_out.x == 100."
    )
    s.call_setup_run()
    d.run_once()
    assert s.inwards.K1 == pytest.approx(100.0, abs=1e-3)

    s.inwards.K1 = 1.0
    d.jac = np.linalg.inv(np.array([[10.0]]))
    d.jac_lup = lu_factor(d.jac)
    d.run.solution.clear()

    caplog.clear()
    with caplog.at_level(logging.INFO):
        d.run_once()
    assert s.inwards.K1 == pytest.approx(100.0, abs=1e-3)

    info_messages = list(
        map(
            lambda r: r.msg,
            filter(lambda r: r.levelno == logging.INFO, caplog.records),
        )
    )
    assert len(info_messages) >= 2
    matches = map(
        lambda r: re.search(r"Jacobian matrix: \d+ over \d+ derivative\(s\) updated", r),
        info_messages,
    )
    assert not any(matches)
    assert "Jacobian matrix: full update" in info_messages
    matches = map(
        lambda r: re.search(
            r"   -> Converged \((\d)+(?:\.\d*)(?:[eE][-]\d+)\) in \d+ iterations, 1 complete, 0 partial Jacobian and 0 Broyden evaluation\(s\)",
            r,
        ),
        info_messages,
    )
    assert any(matches)

def test_partialjacobian_coupledmatrix(caplog, set_master_system):
    s = Multiply2("MyMult")
    d = s.add_driver(
        NonLinearSolver("run_drivers", method=NonLinearMethods.NR, factor=0.1)
    )

    s.p_in.x = 1.0
    s.inwards.K1 = 1.0
    s.inwards.K2 = 1.0

    run = d.add_child(RunSingleCase("run"))
    run.design.add_unknown(["K1", "K2"]).add_equation(
        "K1 == 100", reference="norm"
    ).add_equation("K2 == 50", reference="norm")
    s.call_setup_run()
    d.run_once()
    assert s.inwards.K1 == pytest.approx(100.0, abs=1e-3)
    assert s.inwards.K2 == pytest.approx(50.0, abs=1e-3)

    s.inwards.K1 = s.inwards.K2 = 1.0

    d.jac = np.array([[1, 0], [0.0, 0.077]])
    d.jac_lup = lu_factor(d.jac)
    d.run.solution.clear()

    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        d.run_once()
    
    info_messages = list(
        map(
            lambda r: r.msg,
            filter(lambda r: r.levelno == logging.INFO, caplog.records),
        )
    )
    assert len(info_messages) >= 2
    matches = map(
        lambda r: re.search(
            r"   -> Converged \((\d)+(?:\.\d*)(?:[eE][-]\d+)\) in \d+ iterations, 1 complete, 1 partial Jacobian and \d+ Broyden evaluation\(s\)",
            r,
        ),
        info_messages,
    )
    assert any(matches)

    debug_messages = list(
        map(
            lambda r: r.msg,
            filter(lambda r: r.levelno == logging.DEBUG, caplog.records),
        )
    )
    assert "Jacobian matrix: 1 over 2 derivative(s) updated" in debug_messages
    assert "Reuse of previous Jacobian matrix" in debug_messages
    assert "Perturb unknown 0" in debug_messages


def test_partialjacobian_independentmatrix(caplog, set_master_system):
    s = Multiply2("MyMult")
    d = s.add_driver(
        NonLinearSolver("run_drivers", method=NonLinearMethods.NR, factor=0.1)
    )

    s.p_in.x = 1.0
    s.inwards.K1 = 1.0
    s.inwards.K2 = 1.0

    run = d.add_child(RunSingleCase("run"))
    run.design.add_unknown(["K1", "K2"]).add_equation(
        "K1 == 100", reference="norm"
    ).add_equation("K2 == 50", reference="norm")
    s.call_setup_run()
    d.run_once()
    assert s.inwards.K1 == pytest.approx(100.0, abs=1e-3)
    assert s.inwards.K2 == pytest.approx(50.0, abs=1e-3)

    s.inwards.K1 = s.inwards.K2 = 1.0
    d.jac = np.array([[0.039, 0], [0.0, -3]])
    d.jac_lup = lu_factor(d.jac)
    d.run.solution.clear()

    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        d.run_once()
    
    info_messages = list(
        map(
            lambda r: r.msg,
            filter(lambda r: r.levelno == logging.INFO, caplog.records),
        )
    )
    assert len(info_messages) >= 2
    matches = map(
        lambda r: re.search(
            r"   -> Converged \((\d)+(?:\.\d*)(?:[eE][-]\d+)\) in \d+ iterations, 0 complete, 1 partial Jacobian and \d+ Broyden evaluation\(s\)",
            r,
        ),
        info_messages,
    )
    assert any(matches)

    debug_messages = list(
        map(
            lambda r: r.msg,
            filter(lambda r: r.levelno == logging.DEBUG, caplog.records),
        )
    )
    assert "Jacobian matrix: 1 over 2 derivative(s) updated" in debug_messages
    assert "Reuse of previous Jacobian matrix" in debug_messages
    assert "Perturb unknown 1" in debug_messages
