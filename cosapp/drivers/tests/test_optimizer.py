import pytest
import logging, re
import numpy as np

from cosapp.systems import System
from cosapp.drivers import Optimizer, RunSingleCase, NonLinearSolver, NonLinearMethods


class Rosenbrock(System):
    """Case taken from https://docs.scipy.org/doc/scipy-1.0.0/reference/tutorial/optimize.html"""

    def setup(self, **kwargs):
        # Solution is xi = 1. => init away from the solution
        self.add_inward("x", np.asarray([1.3, 0.7, 0.8, 1.9, 1.2]))
        self.add_outward("rosenbrock", np.ones(4))

    def compute(self):
        self.rosenbrock = (
            100.0 * (self.x[1:] - self.x[:-1] ** 2.0) ** 2.0 + (1 - self.x[:-1]) ** 2.0
        )


class Constraint(System):
    """Case taken from https://docs.scipy.org/doc/scipy-1.0.0/reference/tutorial/optimize.html"""

    def setup(self, **kwargs):
        self.add_inward("x1", 1.3)
        self.add_inward("x2", 0.7)
        self.add_outward("objective")

    def compute(self):
        self.objective = (
            2 * self.x1 * self.x2 + 2 * self.x1 - self.x1 ** 2 - 2 * self.x2 ** 2
        )


class ExempleOptim(System):
    def setup(self, **kwargs):
        self.add_inward("x1", 1.3)
        self.add_inward("x2", 0.7)
        self.add_inward("a", 1.0)

        self.add_outward("fcost", 1.0)

    def compute(self):
        self.fcost = (self.x1 - self.a) ** 2 + (self.x2 - self.a) ** 2


class CubicFunction(System):
    def setup(self):
        self.add_inward('a')
        self.add_inward('x')
        self.add_outward('y')
    
    def compute(self):
        self.y = self.x**3 - (2 * self.a)**3


# Note: scipy.optimize.minimize raises `OptimizeWarning`, as
# the underlying solver does not know either 'gtol' or 'ftol'.
# Since the choice of resolution method may not be known a priori
# (depending if the optimization problem has constraints or none),
# we cannot know in advance which option name will be correct.
# We simply ignore this warning, using the pytest.mark decorator below
@pytest.mark.filterwarnings("ignore:Unknown solver options. .tol")
def test_Optimizer_compute():
    # Simple optimization
    s = Rosenbrock("rosenbrock")
    opt = Optimizer("optimization", verbose=True)
    s.add_driver(opt)

    opt.runner.set_objective("rosenbrock.sum()")
    opt.runner.add_unknown("x")

    s.run_drivers()

    assert s.x == pytest.approx(np.ones(5), abs=1e-4)

    # Constrained optimization
    s = Constraint("constraint")
    opt = Optimizer("optimization", verbose=1)
    s.add_driver(opt)

    opt.runner.set_objective("-objective")
    opt.runner.add_unknown(["x1", "x2"])

    # Not constrained
    s.run_drivers()

    assert s.x1 == pytest.approx(2, abs=5e-5)
    assert s.x2 == pytest.approx(1, abs=5e-5)

    # Constrained
    opt.runner.add_constraints("x1**3 - x2", False)  # Equality
    opt.runner.add_constraints("x1 - 1")  # Inequality
    s.run_drivers()

    assert s.x1 == pytest.approx(1, abs=5e-5)
    assert s.x2 == pytest.approx(1, abs=5e-5)


@pytest.mark.filterwarnings("ignore:Unknown solver options. .tol")
def test_Optimizer_compute_with_optimizer():
    # Simple optimizer
    s = ExempleOptim("system")
    opt = Optimizer("optimization", verbose=1)
    s.add_driver(opt)

    opt.runner.set_objective("fcost")
    opt.runner.add_unknown(["x1", "x2"])

    s.x1 = 10.0
    s.x2 = 100.0

    s.a = 7.0
    s.run_drivers()

    assert s.x1 == pytest.approx(s.a, abs=5e-5)
    assert s.x2 == pytest.approx(s.a, abs=5e-5)


@pytest.mark.filterwarnings("ignore:Unknown solver options. .tol")
def test_Optimizer_compute_with_optimizer_and_solver_1():
    s = ExempleOptim("system")
    opt = Optimizer("optimization", verbose=1)
    s.add_driver(opt)
    solver = NonLinearSolver("solver", method=NonLinearMethods.NR, verbose=1, tol=1e-10)
    run = solver.add_driver(RunSingleCase("run", verbose=1))
    opt.runner.add_driver(solver)

    s.a = 15.0
    s.x1 = 10.0
    s.x2 = 100.0
    opt.options["ftol"] = 1.0e-5

    opt.runner.set_objective("fcost")
    opt.runner.add_unknown(["x1", "x2"])

    run.design.add_unknown("a").add_equation("a**3 == -27")

    s.run_drivers()

    assert s.a == pytest.approx(-3, abs=1e-4)
    assert s.x1 == pytest.approx(s.a, abs=1e-4)
    assert s.x2 == pytest.approx(s.a, abs=1e-4)


@pytest.mark.filterwarnings("ignore:Unknown solver options. .tol")
def test_Optimizer_compute_with_optimizer_and_solver_2():
    head = CubicFunction("head")
    optim = head.add_driver(Optimizer("optim", verbose=1))
    solver = optim.runner.add_driver(NonLinearSolver("solver", tol=1e-10))

    # Optim problem
    optim.runner.set_objective("2 + (x - 1)**2")
    optim.runner.add_unknown("a")

    # Solver problem (solution is x = 2a)
    solver.add_unknown("x").add_equation("y == 0")

    head.a = -5.0
    head.x = 10.0

    head.run_drivers()

    assert head.a == pytest.approx(0.5)  # minimizes 2 + (x - 1)**2, when x = 2a
    assert head.x == pytest.approx(1)
    assert head.y == pytest.approx(0, abs=1e-10)


@pytest.mark.filterwarnings("ignore:Unknown solver options. .tol")
def test_Optimizer_compute_with_optimizer_and_solver_3(caplog):
    """Same as test #2, where both optimizer and solver unknowns are pulled up"""
    head = System("head")
    head.add_child(CubicFunction("sub"), pulling=['a', 'x'])

    optim = head.add_driver(Optimizer("optim", verbose=1))
    solver = optim.runner.add_driver(NonLinearSolver("solver", tol=1e-10))

    # Optim problem
    optim.runner.set_objective("2 + (x - 1)**2")
    optim.runner.add_unknown("sub.a")  # `sub.a` is pulled

    # Solver problem (solution is x = 2a)
    solver.add_equation("sub.y == 0").add_unknown("sub.x")  # `sub.x` is pulled

    head.a = -5.0
    head.x = 10.0

    caplog.clear()
    with caplog.at_level(logging.INFO):
        head.run_drivers()
    
    assert len(caplog.records) > 2
    messages = [record.message for record in caplog.records]
    assert any(
        re.match(
            "Replace unknown 'sub.inwards.a' by 'inwards.a'",
            message
        )
        for message in messages[:2]
    )
    assert any(
        re.match(
            "Replace unknown 'sub.inwards.x' by 'inwards.x'",
            message
        )
        for message in messages[:2]
    )
    assert head.sub.a == pytest.approx(0.5)  # minimizes 2 + (x - 1)**2, when x = 2a
    assert head.sub.x == pytest.approx(1)
    assert head.sub.y == pytest.approx(0, abs=1e-10)
