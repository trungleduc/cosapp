"""Integration tests for `cosapp.drivers.Optimizer`
"""
import pytest

from cosapp.systems import System
from cosapp.drivers import NonLinearSolver, NonLinearMethods, Optimizer
from cosapp.tests.library.systems.multiply import Merger, ExponentialLoad, Splitter


class ToOptimize(System):
    def setup(self):
        self.add_child(Merger("merger"), pulling={"p1_in": "p_in"})
        self.add_child(Splitter("splitter"), pulling={"p2_out": "p_out"})
        self.add_child(ExponentialLoad("main_load"))
        self.add_child(ExponentialLoad("secondary_load"))
        self.add_outward("load_", 0.0)

        self.connect(self.merger.p_out, self.main_load.p_in)
        self.connect(self.main_load.p_out, self.splitter.p_in)
        self.connect(self.splitter.p1_out, self.secondary_load.p_in)
        self.connect(self.secondary_load.p_out, self.merger.p2_in)

        self.exec_order = ["merger", "main_load", "splitter", "secondary_load"]

    def compute(self):
        self.load_ = self.main_load.load_ + self.secondary_load.load_


@pytest.fixture
def system():
    """System used in integration tests"""
    s = ToOptimize("system")
    s.splitter.split_ratio = 0.1
    s.main_load.a = -1.0
    s.main_load.b = 3.0
    s.secondary_load.c = 4.0
    return s


@pytest.fixture
def solver():
    """Solver used in integration tests"""
    return NonLinearSolver("solver", method=NonLinearMethods.NR, factor=0.8)


def test_system_solve(system, solver):
    """Benchmark: solve base case."""
    system.add_driver(solver)
    system.run_drivers()

    assert system.p_out.x == pytest.approx(1)
    assert system.merger.p2_in.x == pytest.approx(1 / 9)
    assert system.secondary_load.p_in.x / system.main_load.p_in.x == pytest.approx(0.1, rel=1e-14)
    assert system.load_ == system.main_load.load_ + system.secondary_load.load_
    assert system.load_ == pytest.approx(11.729536980212018)


def test_Optimizer_integration(system, solver):
    system.add_driver(solver)
    system.run_drivers()
    system.drivers.clear()

    optim = system.add_driver(Optimizer("optim"))

    optim.set_minimum("load_")
    optim.add_unknown("p_in.x")
    optim.add_child(solver)

    system.run_drivers()

    assert system.p_in.x == pytest.approx(4.338487, abs=1e-5)
    assert system.secondary_load.p_in.x / system.main_load.p_in.x == pytest.approx(0.1, rel=1e-14)
    assert system.load_ == pytest.approx(5.78134, abs=1e-5)


def test_Optimizer_integration_upper_bound(system, solver):
    system.add_driver(solver)
    system.run_drivers()

    system.drivers.clear()

    optim = system.add_driver(Optimizer("optim"))

    optim.set_minimum("load_")
    optim.add_unknown("p_in.x", upper_bound=4.2)
    optim.add_child(solver)

    system.run_drivers()

    assert system.p_in.x == pytest.approx(4.2, abs=1e-5)
    assert system.secondary_load.p_in.x / system.main_load.p_in.x == pytest.approx(0.1, rel=1e-14)
    assert system.load_ == pytest.approx(5.78355, abs=1e-5)


def test_Optimizer_integration_lower_bound(system, solver):
    system.add_driver(solver)
    system.run_drivers()

    system.drivers.clear()

    optim = system.add_driver(Optimizer("optim"))

    optim.set_minimum("load_")
    optim.add_unknown("p_in.x", lower_bound=4.5)
    optim.add_child(solver)

    system.run_drivers()

    assert system.p_in.x == pytest.approx(4.5, abs=1e-5)
    assert system.secondary_load.p_in.x / system.main_load.p_in.x == pytest.approx(0.1, rel=1e-14)
    assert system.load_ == pytest.approx(5.78406, abs=1e-5)
