import pytest
import numpy as np

from cosapp.base import System
from cosapp.drivers import NonLinearSolver, RungeKutta, CrankNicolson
from cosapp.tests.library.systems.sellar import Sellar


class Tank(System):
    def setup(self):
        self.add_inward("mass_flowrate", 0.0)
        self.add_transient("mass", der="mass_flowrate")


class Assembly(System):
    def setup(self):
        self.add_child(Tank("tank"))
        self.add_child(Sellar("sellar"), pulling=["z", "x"])


def test_NonLinearSolver_ImplicitTimeDriver():
    """Check that outer solver does not interfere with inner driver
    by unduely solving the intrinsic problem of the owner system.

    Here, a design problem is set on top of a time-dependent problem,
    for a system containing cyclic dependencies (within subsystem `sellar`).

    Test case: implicit time driver nested within the nonlinear solver.
    The implicit time driver solves the Sellar problem at each time step,
    while the outer nonlinear solver sets a design problem on top of it.
    """
    top = Assembly("top")
    top.z = np.array([1.0, 5.0])
    top.x = -1.0

    solver = top.add_driver(NonLinearSolver("solver"))
    driver = solver.add_child(CrankNicolson("driver", dt=0.125, time_interval=(0.0, 1.0)))

    solver.add_unknown("tank.mass_flowrate").add_equation("tank.mass == 1.0")
    driver.set_scenario(
        init={"tank.mass": 10.0},
    )

    top.run_drivers()

    assert solver.problem.shape == (1, 1)
    assert driver.problem.shape == (2, 1)
    assert set(solver.problem.unknowns.keys()) == {"tank.mass_flowrate"}
    assert set(driver.problem.unknowns.keys()) == {"tank.mass", "sellar.d1.y2"}
    assert top.tank.mass_flowrate == pytest.approx(-9.0)
    assert top.sellar.d1.y2 == pytest.approx(top.sellar.d2.y2)


def test_NonLinearSolver_time_inner_solver():
    """Check that outer solver does not interfere with inner driver
    by unduely solving the intrinsic problem of the owner system.

    Here, a design problem is set on top of a time-dependent problem,
    for a system containing cyclic dependencies (within subsystem `sellar`).

    Test case: explicit time driver with an inner solver, nested within the outer nonlinear solver.
    The inner solver solves the Sellar problem at each time step,
    while the outer nonlinear solver sets a design problem on top of it.
    """
    top = Assembly("top")
    top.z = np.array([1.0, 5.0])
    top.x = -1.0

    solver = top.add_driver(NonLinearSolver("solver"))
    driver = solver.add_child(RungeKutta("driver", order=2, dt=0.125, time_interval=(0.0, 1.0)))
    inner = driver.add_child(NonLinearSolver("inner"))

    solver.add_unknown("tank.mass_flowrate").add_equation("tank.mass == 1.0")
    driver.set_scenario(
        init={"tank.mass": 10.0},
    )

    top.run_drivers()
    assert solver.problem.shape == (1, 1)
    assert inner.problem.shape == (1, 1)
    assert set(solver.problem.unknowns.keys()) == {"tank.mass_flowrate"}
    assert set(inner.problem.unknowns.keys()) == {"sellar.d1.y2"}
    assert top.tank.mass_flowrate == pytest.approx(-9.0)
    assert top.sellar.d1.y2 == pytest.approx(top.sellar.d2.y2)
