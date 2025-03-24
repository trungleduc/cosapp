import pytest

import numpy as np
from cosapp.systems import System
from cosapp.drivers import CrankNicolson
from cosapp.recorders import DataFrameRecorder
from .conftest import CoupledTanks


class Ode(System):
    """ODE of the kind df/dt = df
    """
    def setup(self, varname="f", **options):
        self.add_inward(f"d{varname}", 0.0)
        self.add_transient(varname, der=f"d{varname}", **options)


class MultimodeOde(Ode):
    """Multimode ODE of the kind df/dt = df,
    with event `snap` (undefined by default).
    """
    def setup(self, varname="f", **options):
        super().setup(varname=varname, **options)
        self.add_event("snap")
        self.add_outward_modevar("snapped", init=False)

    def transition(self):
        if self.snap.present:
            self.snapped = True


@pytest.fixture
def ode_case():
    ode = Ode("ode")
    driver = ode.add_driver(CrankNicolson())
    driver.add_recorder(DataFrameRecorder())
    return ode, driver


def test_CrankNicolson_ode_exp(ode_case):
    ode, driver = ode_case
    driver.set_scenario(
        init={"f": 1.0},
        values={"df": "f"},
    )
    driver.add_recorder(
        DataFrameRecorder(includes=["*", "f - exp(t)"]),
        period=0.1,
    )
    driver.time_interval = [0.0, 1.0]
    driver.dt = 0.01
    ode.run_drivers()

    data = driver.recorder.export_data()

    # Expected "exact" numerical solution:
    # For this particular case, the numerical solution is expected to be
    # a geometrical series f_{n+1} = alpha * f_{n}, with f_{0} = 1, and
    # alpha = (2 + dt) / (2 - dt)
    alpha = (2 + driver.dt) / (2 - driver.dt)
    n = 101
    times = np.linspace(0, 1, n)
    f_exact = np.exp(times[::10])
    f_expected = [1.0]
    for i in range(n - 1):
        f_expected.append(alpha * f_expected[-1])
    f_expected = np.asarray(f_expected)[::10]
    f_simu = np.asarray(data["f"])
    error = f_simu - f_exact
    assert f_simu == pytest.approx(f_expected, rel=1e-14)
    assert np.linalg.norm(error, np.inf) == pytest.approx(2.26e-5, rel=1e-2)


@pytest.mark.parametrize("delta", [0.01, 0.001])
def test_CrankNicolson_ode_stiff(delta):
    """Test Crank-Nicolson driver on a stiff ODE:
    dy/dt = y^2 - y^3
    with y(0) = 1 / delta,
    t in [0, 2 / delta]
    """
    ode = Ode("ode", varname="y", max_abs_step=0.02)
    driver = ode.add_driver(CrankNicolson(record_dt=True))
    driver.set_scenario(
        init={"y": delta},
        values={"dy": "y * y * (1 - y)"},
    )
    driver.add_recorder(
        DataFrameRecorder(includes=["*"]),
    )
    driver.time_interval = [0.0, 2.0 / delta]
    driver.dt = 0.05 / delta

    ode.run_drivers()

    data = driver.recorder.export_data()

    y = np.asarray(data["y"])
    assert y[-1] == pytest.approx(1, abs=1e-5)


@pytest.mark.parametrize("dt", [0.1, 0.01, 0.005])
def test_CrankNicolson_tanks(two_tank_solution, dt):
    """Test Crank-Nicolson driver on a model with an intrinsic problem (loop).
    """
    system = CoupledTanks("system")
    driver = system.add_driver(CrankNicolson())

    h1_0, h2_0 = init = (3, 1)

    driver.set_scenario(
        init = {
            # initial conditions
            "tank1.height": h1_0,
            "tank2.height": h2_0,
        },
        values = {
            "pipe.D": 0.07,
            "pipe.L": 2.5,
            "tank1.area": 2.0,
            "tank2.area": 1.0,
        },
    )
    driver.add_recorder(
        DataFrameRecorder(includes=["tank?.height"]),
    )
    driver.time_interval = (0, 5)
    driver.dt = dt
    system.run_drivers()

    assert driver.problem.shape == (4, 2)

    data = driver.recorder.export_data()

    assert system.tank1.height == pytest.approx(system.tank2.height, abs=1e-3)
    assert system.tank1.height == pytest.approx(2.333, abs=1e-3)

    data = driver.recorder.export_data()
    solution = two_tank_solution(system, init)
    assert solution.characteristic_time == pytest.approx(0.5766040)
    assert driver.dt < solution.characteristic_time
    time = np.asarray(data['time'])
    exact = solution(time)
    error = np.vstack([
        np.asarray(data["tank1.height"]) - exact[0],
        np.asarray(data["tank2.height"]) - exact[1],
    ])
    error = np.linalg.norm(error[0], np.inf)
    assert error < 0.07 * dt**2
    assert error > 0.06 * dt**2


def test_CrankNicolson_new_problem():
    """Test a transition which brings new intrinsic constraints
    """
    class MultimodeSystem(System):

        def setup(self):
            self.add_inward("x", 0.0)
            self.add_inward("df", 0.0)
            self.add_outward("y", 0.0)

            self.add_transient("f", der="df")
            self.add_event("tada")

        def compute(self):
            self.y = self.f - self.x**2

        def transition(self):
            if self.tada.present:
                self.problem.clear()
                self.add_unknown("x").add_equation("y == 0")
                self.x = 1.0

    system = MultimodeSystem("system")
    driver = system.add_driver(CrankNicolson("driver"))
    driver.time_interval = (0, 1)
    driver.dt = 1e-2

    driver.add_recorder(
        DataFrameRecorder(),
        period=0.1,
    )
    driver.set_scenario(
        init={
            "f": 0.0,
            "x": 0.0,
        },
        values={
            "df": "8 * t",
        },
    )
    system.tada.trigger = "f > 2"
    system.run_drivers()

    data = driver.recorder.export_data()
    x = np.asarray(data["x"])
    y = np.asarray(data["y"])

    assert x == pytest.approx(
        [0., 0., 0., 0., 0., 0., 0., 0., 0., np.sqrt(2), 1.6, 1.8, 2.],
        rel=1e-14,
    )
    assert y == pytest.approx(
        [0., 0.04, 0.16, 0.36, 0.64, 1., 1.44, 1.96, 2., 0., 0., 0., 0.],
        rel=1e-14,
    )
