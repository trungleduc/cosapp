import numpy as np
import pytest

from cosapp.drivers import NonLinearMethods, NonLinearSolver, RunSingleCase
from cosapp.systems import System
from cosapp.core.numerics.residues import Residue
from cosapp.tests.library.systems import (
    Multiply2,
    MultiplyVector2,
    MultiplyVector3,
    NonLinear3,
    IterativeNonLinear,
)


def test_ScipySolverHybr_get_residues():
    system = Multiply2("MyMult")
    solver = system.add_driver(NonLinearSolver("solver", method=NonLinearMethods.POWELL))
    solver.add_unknown("K1").add_equation("p_out.x == 100")
    solver.owner.call_setup_run()
    solver._precompute()
    rname = "p_out.x == 100"
    residues = solver.problem.residues
    assert set(residues) == {rname}
    assert residues[rname].value == Residue._evaluate_numerical_residue(system.p_out.x, 100.0)


@pytest.mark.parametrize("method", NonLinearMethods)
def test_ScipySolver_fresidues_init(method):
    s = Multiply2("s")
    d = s.add_driver(NonLinearSolver("solver", method=method))

    init = np.random.rand(3).tolist()
    s.call_setup_run()
    d._precompute()
    d._fresidues(init)

    for idx, iterative in enumerate(d.problem.unknowns.values()):
        assert iterative.value == init[idx]


def test_ScipySolverHybr_singleptHybr(set_master_system):
    s = Multiply2("s")
    d = s.add_driver(NonLinearSolver("solver", method=NonLinearMethods.POWELL))

    s.p_in.x = 1.0
    s.K2 = 1.0

    d.add_unknown("K1").add_equation("p_out.x == 100")
    s.call_setup_run()
    d.run_once()
    assert s.K1 == pytest.approx(100, abs=1e-5)

    s = Multiply2("s")
    d = s.add_driver(NonLinearSolver("solver", method=NonLinearMethods.POWELL))
    d.add_unknown(["K1", "K2"]).add_equation(["p_out.x == 100", "K2 == K1"])
    s.K1 = 1.0
    s.call_setup_run()
    d.run_once()

    # Note: solution is K1^2 = K2^2 = 100
    assert abs(s.K1) == pytest.approx(10, abs=1e-12)
    assert s.K2 == pytest.approx(s.K1, abs=1e-12)


def test_ScipySolverHybr_multipts1():
    s2 = MultiplyVector2("multvector")

    solver = s2.add_driver(NonLinearSolver("solver", method=NonLinearMethods.POWELL))
    run1 = solver.add_child(RunSingleCase("run1"))

    s2.k1 = 10.0
    s2.k2 = 8.0

    run1.set_values({"p_in.x1": 4.0, "p_in.x2": 10.0})
    run1.design.add_unknown("k1").add_equation("p_out.x == 100")

    s2.run_drivers()
    assert s2.k1 == pytest.approx(5)

    run2 = solver.add_child(RunSingleCase("run2"))

    run2.set_values({"p_in.x1": 2, "p_in.x2": 8.0})
    run2.design.add_unknown("k2").add_equation("p_out.x == 76")

    s2.run_drivers()
    assert s2.k1 == pytest.approx(10 / 3)
    assert s2.k2 == pytest.approx(26 / 3)


def test_ScipySolverHybr_multipts2():
    sys = System("s")
    sys.add_child(MultiplyVector3("s3"))

    solver = sys.add_driver(NonLinearSolver("solver", method=NonLinearMethods.POWELL))

    run1 = solver.add_child(RunSingleCase("run1"))
    run2 = solver.add_child(RunSingleCase("run2"))
    run3 = solver.add_child(RunSingleCase("run3"))

    run1.set_values({"s3.p_in.x1": 4.0, "s3.p_in.x2": 10.0, "s3.p_in.x3": 1.0})
    run1.design.add_unknown("s3.k1").add_equation("s3.p_out.x == 100")

    run2.set_values({"s3.p_in.x1": 2, "s3.p_in.x2": 8.0, "s3.p_in.x3": 1.0})
    run2.design.add_unknown("s3.k2").add_equation("s3.p_out.x == 76")

    run3.set_values({"s3.p_in.x1": 5, "s3.p_in.x2": 12.0, "s3.p_in.x3": 1.0})
    run3.design.add_unknown("s3.k3").add_equation("s3.p_out.x == 150")

    sys.run_drivers()
    assert sys.s3.k1 == pytest.approx(-26)
    assert sys.s3.k2 == pytest.approx(38)
    assert sys.s3.k3 == pytest.approx(-176)


def test_ScipySolverHybr_multipts_nonlinear3():
    sys = System("s")
    sys.add_child(NonLinear3("s3"))

    solver = sys.add_driver(NonLinearSolver("solver", method=NonLinearMethods.POWELL))

    run1 = solver.add_child(RunSingleCase("run1"))
    run2 = solver.add_child(RunSingleCase("run2"))
    run3 = solver.add_child(RunSingleCase("run3"))

    run1.set_values({"s3.p_in.x1": 4.0, "s3.p_in.x2": 10.0, "s3.p_in.x3": 1.0})
    run1.design.add_unknown("s3.k1").add_equation("s3.p_out.x == 100")

    run2.set_values({"s3.p_in.x1": 2, "s3.p_in.x2": 8.0, "s3.p_in.x3": 1.0})
    run2.design.add_unknown("s3.k2").add_equation("s3.p_out.x == 76")

    run3.set_values({"s3.p_in.x1": 5, "s3.p_in.x2": 12.0, "s3.p_in.x3": 1.0})
    run3.design.add_unknown("s3.k3").add_equation("s3.p_out.x == 150")

    sys.run_drivers()
    assert sys.s3.k1 == pytest.approx(227.4029139, abs=1e-5)
    assert sys.s3.k2 == pytest.approx(72465.89971, abs=1e-5)
    assert sys.s3.k3 == pytest.approx(-26454.1762, abs=1e-5)


def test_ScipySolverHybr_multipts_iterative_non_linear():
    snl = IterativeNonLinear("nl")

    solver = snl.add_driver(NonLinearSolver("solver", method=NonLinearMethods.POWELL))

    snl.splitter.split_ratio = 0.1
    snl.mult2.K1 = 1
    snl.mult2.K2 = 1
    snl.nonlinear.k1 = 1
    snl.nonlinear.k2 = 0.5

    run1 = solver.add_child(RunSingleCase("run1"))
    run2 = solver.add_child(RunSingleCase("run2"))

    run1.set_values({"p_in.x": 1})
    run1.design.add_unknown("nonlinear.k1").add_equation("splitter.p2_out.x == 10")

    run2.set_values({"p_in.x": 10})
    run2.design.add_unknown(
        ["mult2.K1", "nonlinear.k2", "splitter.split_ratio"]
    ).add_equation(
        "splitter.p2_out.x == 50", reference="norm",
    ).add_equation(
        "splitter.p1_out.x == 5", reference="norm",
    ).add_equation(
        "merger.p_out.x == 30", reference="norm",
    )

    snl.run_drivers()

    assert snl.mult2.K1 == pytest.approx(1.833333333)
    assert snl.nonlinear.k1 == pytest.approx(5)
    assert snl.nonlinear.k2 == pytest.approx(0.861353116)
    assert snl.splitter.split_ratio == pytest.approx(0.090909091)


def test_ScipySolverHybr_option_aliases():
    system = IterativeNonLinear("nl")

    system.drivers.clear()
    solver = system.add_driver(
        NonLinearSolver("solver", method=NonLinearMethods.POWELL)
    )
    to_test = {"max_eval": 0, "tol": 1e-7, "factor": 0.1}
    assert {key: solver.options[key] for key in to_test} == to_test

    system.drivers.clear()
    solver = system.add_driver(
        NonLinearSolver(
            "design",
            method=NonLinearMethods.POWELL,
            tol=1e-9,
            max_eval=99,
            factor=12.3,
        )
    )
    to_test = {"max_eval": 99, "tol": 1e-9, "factor": 12.3}
    assert {key: solver.options[key] for key in to_test} == to_test

    system.drivers.clear()
    with pytest.raises(RuntimeError, match="Unknown option 'foo'"):
        solver = system.add_driver(
            NonLinearSolver(
                "design",
                method=NonLinearMethods.POWELL,
                tol=1e-9,
                foo=12.3,
            )
        )


def test_ScipySolverBroyden1_get_residues():
    s = Multiply2("s")
    solver = s.add_driver(
        NonLinearSolver("solver", method=NonLinearMethods.BROYDEN_GOOD)
    )
    solver.add_unknown("K1").add_equation("p_out.x == 100")
    solver.owner.call_setup_run()
    solver._precompute()
    rname = "p_out.x == 100"
    residues = solver.problem.residues
    assert set(residues) == {rname}
    assert residues[rname].value == Residue._evaluate_numerical_residue(s.p_out.x, 100.0)


def test_ScipySolverBroyden1_singlept():
    s = Multiply2("s")
    solver = s.add_driver(
        NonLinearSolver("solver", method=NonLinearMethods.BROYDEN_GOOD)
    )

    s.p_in.x = 1.0
    s.K2 = 1.0

    solver.add_child(RunSingleCase("case"))
    solver.case.design.add_unknown("K1").add_equation("p_out.x == 100", reference=10)

    s.run_drivers()

    assert s.K1 == pytest.approx(
        100, rel=1e-15
    ), "Floating-point accuracy is expected in a linear problem"

    solver.pop_child("case")
    solver.add_child(RunSingleCase("case"))

    # d.case.design.add_unknown(['K1', 'K2']).add_equation(['p_out.x == 100', 'K2 == K1'])  # Fails
    solver.case.design.add_unknown(['K1', 'K2']).add_equation(['p_out.x == 100', 'K1 == K2'])  # Passes!

    s.K1 = 5
    s.run_drivers()

    # Note: solution is K1^2 = K2^2 = 100
    assert abs(s.K1) == pytest.approx(10, abs=1e-12)
    assert s.K2 == pytest.approx(s.K1, abs=1e-12)


def test_ScipySolverBroyden1_multipts1():
    s = MultiplyVector2("s")

    solver = s.add_driver(
        NonLinearSolver("solver", method=NonLinearMethods.BROYDEN_GOOD)
    )
    run1 = solver.add_child(RunSingleCase("run1"))

    s.k1 = 10.0
    s.k2 = 8.0

    run1.set_values({"p_in.x1": 4.0, "p_in.x2": 10.0})
    run1.design.add_unknown("k1").add_equation("p_out.x == 100")

    s.run_drivers()
    assert s.k1 == pytest.approx(5, abs=1e-12)
    run1.solution.clear()

    run2 = solver.add_child(RunSingleCase("run2"))

    run2.set_values({"p_in.x1": 2, "p_in.x2": 8.0})
    run2.design.add_unknown("k2").add_equation("p_out.x == 76")

    s.k1 = 1.0
    s.run_drivers()
    assert s.k1 == pytest.approx(10 / 3, abs=1e-12)
    assert s.k2 == pytest.approx(26 / 3, abs=1e-12)


def test_ScipySolverBroyden1_multipts2():
    s = System("s")
    s.add_child(MultiplyVector3("s3"))

    solver = s.add_driver(
        NonLinearSolver("solver", method=NonLinearMethods.BROYDEN_GOOD)
    )
    run1 = solver.add_child(RunSingleCase("run1"))
    run2 = solver.add_child(RunSingleCase("run2"))
    run3 = solver.add_child(RunSingleCase("run3"))

    run1.set_values({"s3.p_in.x1": 4.0, "s3.p_in.x2": 10.0, "s3.p_in.x3": 1.0})
    run1.design.add_unknown("s3.k1").add_equation("s3.p_out.x == 100")

    run2.set_values({"s3.p_in.x1": 2, "s3.p_in.x2": 8.0, "s3.p_in.x3": 1.0})
    run2.design.add_unknown("s3.k2").add_equation("s3.p_out.x == 76")

    run3.set_values({"s3.p_in.x1": 5, "s3.p_in.x2": 12.0, "s3.p_in.x3": 1.0})
    run3.design.add_unknown("s3.k3").add_equation("s3.p_out.x == 150")

    s.run_drivers()
    assert s.s3.k1 == pytest.approx(-26, rel=1e-12)
    assert s.s3.k2 == pytest.approx(38, rel=1e-12)
    assert s.s3.k3 == pytest.approx(-176, rel=1e-12)


def test_ScipySolverBroyden1_multipts_nonlinear3():
    s = System("s")
    s.add_child(NonLinear3("s3"))

    solver = s.add_driver(
        NonLinearSolver("solver", method=NonLinearMethods.BROYDEN_GOOD)
    )
    run1 = solver.add_child(RunSingleCase("run1"))
    run2 = solver.add_child(RunSingleCase("run2"))
    run3 = solver.add_child(RunSingleCase("run3"))

    run1.set_values({"s3.p_in.x1": 4.0, "s3.p_in.x2": 10.0, "s3.p_in.x3": 1.0})
    run1.design.add_unknown("s3.k1").add_equation("s3.p_out.x == 100")

    run2.set_values({"s3.p_in.x1": 2, "s3.p_in.x2": 8.0, "s3.p_in.x3": 1.0})
    run2.design.add_unknown("s3.k2").add_equation("s3.p_out.x == 76")

    run3.set_values({"s3.p_in.x1": 5, "s3.p_in.x2": 12.0, "s3.p_in.x3": 1.0})
    run3.design.add_unknown("s3.k3").add_equation("s3.p_out.x == 150")

    s.run_drivers()
    assert s.s3.k1 == pytest.approx(227.4029139, abs=1e-5)
    assert s.s3.k2 == pytest.approx(72465.89971, abs=1e-5)
    assert s.s3.k3 == pytest.approx(-26454.1762, abs=1e-5)


@pytest.mark.skip("TODO this test does not converge with Good Broyden")
def test_ScipySolverBroyden1_multipts_iterative_non_linear():
    snl = IterativeNonLinear("snl")

    solver = snl.add_driver(
        NonLinearSolver("solver", method=NonLinearMethods.BROYDEN_GOOD)
    )

    snl.splitter.split_ratio = 0.1
    snl.mult2.K1 = 1
    snl.mult2.K2 = 1
    snl.nonlinear.k1 = 1
    snl.nonlinear.k2 = 0.5

    run1 = solver.add_child(RunSingleCase("run1"))
    run2 = solver.add_child(RunSingleCase("run2"))

    run1.set_values({"p_in.x": 1})
    run1.design.add_unknown("nonlinear.k1").add_equation("splitter.p2_out.x == 10")

    run2.set_values({"p_in.x": 10})
    run2.design.add_unknown(
        ["mult2.K1", "nonlinear.k2", "splitter.split_ratio"]
    ).add_equation(
        ["splitter.p2_out.x == 50", "merger.p_out.x == 30", "splitter.p1_out.x == 5"]
    )

    snl.run_drivers()

    assert snl.mult2.K1 == pytest.approx(1.833333333, abs=1e-9)
    assert snl.nonlinear.k1 == pytest.approx(5, abs=1e-13)
    assert snl.nonlinear.k2 == pytest.approx(0.861353116, abs=1e-9)
    assert snl.splitter.split_ratio == pytest.approx(0.090909091, abs=1e-9)


def test_ScipySolverBroyden1_option_aliases():
    system = IterativeNonLinear("nl")

    system.drivers.clear()
    solver = system.add_driver(
        NonLinearSolver("solver", method=NonLinearMethods.BROYDEN_GOOD)
    )
    to_test =  {"num_iter": 100, "max_iter": 100, "tol": 6e-6}
    assert {key: solver.options[key] for key in to_test} == to_test

    system.drivers.clear()
    solver = system.add_driver(
        NonLinearSolver(
            "solver",
            method=NonLinearMethods.BROYDEN_GOOD,
            tol=1e-9,
            max_iter=44,
            num_iter=10,
        )
    )
    to_test =  {"num_iter": 10, "max_iter": 44, "tol": 1e-9}
    assert {key: solver.options[key] for key in to_test} == to_test

    system.drivers.clear()
    with pytest.raises(RuntimeError, match="Unknown option 'max_eval'"):
        system.add_driver(
            NonLinearSolver(
                "solver",
                method=NonLinearMethods.BROYDEN_GOOD,
                max_eval=200,
            )
        )

    system.drivers.clear()
    with pytest.raises(RuntimeError, match="Unknown option 'foobar'"):
        system.add_driver(
            NonLinearSolver(
                "solver",
                method=NonLinearMethods.BROYDEN_GOOD,
                foobar=3.14,
            )
        )
