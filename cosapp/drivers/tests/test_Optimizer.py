import pytest
import logging, re
import numpy as np

from cosapp.systems import System
from cosapp.drivers import Optimizer, NonLinearSolver


class Rosenbrock(System):
    """Case taken from https://docs.scipy.org/doc/scipy-1.0.0/reference/tutorial/optimize.html"""

    def setup(self):
        # Solution is xi = 1. => init away from the solution
        self.add_inward("x", np.array([1.3, 0.7, 0.8, 1.9, 1.2]))
        self.add_outward("rosenbrock", np.ones(4))

    def compute(self):
        self.rosenbrock = (
            100.0 * (self.x[1:] - self.x[:-1] ** 2.0) ** 2.0 + (1 - self.x[:-1]) ** 2.0
        )


class Constraint(System):
    """Case taken from https://docs.scipy.org/doc/scipy-1.0.0/reference/tutorial/optimize.html"""

    def setup(self):
        self.add_inward("x1", 1.3)
        self.add_inward("x2", 0.7)
        self.add_outward("objective")

    def compute(self):
        self.objective = 2 * self.x2 * (self.x1 - self.x2) + self.x1 * (2 - self.x1)


class CostFunction(System):
    def setup(self):
        self.add_inward("x1", 1.3)
        self.add_inward("x2", 0.7)
        self.add_inward("a", 1.0)

        self.add_outward("cost", 1.0)

    def compute(self):
        self.cost = (self.x1 - self.a)**2 + (self.x2 - self.a)**2


class CubicFunction(System):
    def setup(self):
        self.add_inward('a')
        self.add_inward('x')
        self.add_outward('y')
    
    def compute(self):
        self.y = self.x**3 - (2 * self.a)**3


@pytest.fixture
def optim():
    return Optimizer("optim", verbose=True)


def test_Optimizer_available_methods():
    methods = Optimizer.available_methods()
    assert set(methods) == {
        'Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG',
        'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP', 'dogleg',
        'trust-constr', 'trust-ncg', 'trust-exact', 'trust-krylov',
    }


def test_Optimizer__init__(optim: Optimizer):
    assert optim.owner is None
    assert optim.problem is None
    assert optim.raw_problem.shape == (0, 0)
    assert optim.objective is None


def test_Optimizer_set_objective(optim: Optimizer):
    with pytest.raises(AttributeError, match="Owner system is required"):
        optim.set_minimum("foo")
    
    s = CostFunction('s')
    s.a = 0.0
    s.x1 = s.x2 = 1.0
    s.run_once()
    s.add_driver(optim)
    assert optim.owner is s
    assert s.cost > 0
    
    with pytest.warns(UserWarning, match="deprecated"):
        optim.set_objective('cost')
        assert optim.objective == s.cost

    optim.set_minimum('cost')
    assert optim.objective == s.cost

    optim.set_minimum('cost + 1')
    assert optim.objective == s.cost + 1

    optim.set_maximum('cost')
    assert optim.objective == -s.cost


def test_Optimizer_add_unknown(optim: Optimizer):
    with pytest.raises(AttributeError, match="Owner system is required"):
        optim.add_unknown("foo")
    
    s = CostFunction('s')
    s.run_once()
    s.add_driver(optim)
    assert optim.owner is s
    
    optim.add_unknown(['x1', 'x2'])
    assert optim.objective is None
    assert optim.raw_problem.shape == (2, 0)

    with pytest.raises(ArithmeticError, match="objective was not specified"):
        s.run_drivers()


def test_Optimizer_Rosenbrock():
    # Simple optimization
    s = Rosenbrock("s")
    optim = Optimizer("optim", verbose=True)
    s.add_driver(optim)

    optim.set_minimum("rosenbrock.sum()")
    optim.add_unknown("x")

    s.x = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
    s.run_drivers()

    assert s.x == pytest.approx(np.ones(5), abs=1e-4)


def test_Optimizer_compute():
    # Simple optimizer
    s = CostFunction("system")

    optim = s.add_driver(Optimizer("optim", verbose=1))
    optim.set_minimum("cost")
    optim.add_unknown(["x1", "x2"])

    s.a = 7.0
    s.x1 = 10.0
    s.x2 = 100.0
    s.run_drivers()

    assert s.x1 == pytest.approx(s.a, rel=1e-6)
    assert s.x2 == pytest.approx(s.a, rel=1e-6)


def test_Optimizer_constrained():
    """Constrained optimization"""
    s = Constraint("s")

    optim = s.add_driver(Optimizer("optim", verbose=1))
    optim.set_maximum("objective")
    optim.add_unknown(["x1", "x2"])

    # Not constrained
    s.run_drivers()

    assert s.x1 == pytest.approx(2, rel=1e-6)
    assert s.x2 == pytest.approx(1, rel=1e-6)

    # Constrained
    optim.add_constraints([
        "x1**3 == x2",
        "x1 >= 1",
    ])
    s.run_drivers()

    assert s.x1 == pytest.approx(1, rel=1e-6)
    assert s.x2 == pytest.approx(1, rel=1e-6)


def test_Optimizer_compute_with_equality():
    s = CostFunction("system")
    optim = s.add_driver(Optimizer("optim", tol=1e-12))

    optim.set_minimum("cost")
    optim.add_unknown(["x1", "x2", "a"])
    optim.add_constraints("a**3 == -27")

    s.a = 15.0
    s.x1 = 10.0
    s.x2 = 100.0

    s.run_drivers()

    assert s.a == pytest.approx(-3, rel=1e-8)
    assert s.x1 == pytest.approx(s.a, rel=1e-7)
    assert s.x2 == pytest.approx(s.a, rel=1e-7)


def test_Optimizer_compute_with_solver_1():
    s = CostFunction("system")
    optim = s.add_driver(Optimizer("optim", verbose=1))
    solver = NonLinearSolver("solver", verbose=1, tol=1e-10)
    optim.add_driver(solver)

    s.a = 15.0
    s.x1 = 10.0
    s.x2 = 100.0
    optim.options["ftol"] = 1.0e-5

    optim.set_minimum("cost")
    optim.add_unknown(["x1", "x2"])

    solver.add_unknown("a").add_equation("a**3 == -27")

    s.run_drivers()

    assert s.a == pytest.approx(-3)
    assert s.x1 == pytest.approx(s.a)
    assert s.x2 == pytest.approx(s.a)


def test_Optimizer_compute_with_solver_2():
    head = CubicFunction("head")
    optim = head.add_driver(Optimizer("optim", verbose=1))
    solver = optim.add_driver(NonLinearSolver("solver", tol='auto'))

    # Optimization problem
    optim.set_minimum("2 + (x - 1)**2")
    optim.add_unknown("a")

    # Solver problem (solution is x = 2a)
    solver.add_unknown("x").add_equation("y == 0")

    head.a = -5.0
    head.x = 10.0

    head.run_drivers()

    assert head.a == pytest.approx(0.5)  # minimizes 2 + (x - 1)**2, when x = 2a
    assert head.x == pytest.approx(1)
    assert head.y == pytest.approx(0, abs=1e-10)


def test_Optimizer_compute_with_solver_3(caplog):
    """Same as test #2, where both optimizer and solver unknowns are pulled up"""
    head = System("head")
    head.add_child(CubicFunction("sub"), pulling=['a', 'x'])

    optim = head.add_driver(Optimizer("optim", verbose=1))
    solver = optim.add_driver(NonLinearSolver("solver", tol='auto'))

    # Optimization problem
    optim.set_minimum("2 + (x - 1)**2")
    optim.add_unknown("sub.a")  # `sub.a` is pulled

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
            "Replace unknown 'sub.a' by 'a'",
            message
        )
        for message in messages[:2]
    )
    assert any(
        re.match(
            "Replace unknown 'sub.x' by 'x'",
            message
        )
        for message in messages[:2]
    )
    assert head.sub.a == pytest.approx(0.5)  # minimizes 2 + (x - 1)**2, when x = 2a
    assert head.sub.x == pytest.approx(1)
    assert head.sub.y == pytest.approx(0, abs=1e-10)
