import pytest
import numpy as np

from cosapp.systems import System
from cosapp.drivers import BdfIntegrator
from cosapp.recorders import DataFrameRecorder

from .conftest import CoupledTanks, ScalarOde


@pytest.fixture
def ode_case():
    ode = ScalarOde("ode", max_abs_step=0.1)
    driver = ode.add_driver(BdfIntegrator())
    driver.add_recorder(DataFrameRecorder())
    return ode, driver


@pytest.mark.parametrize("dt", [0.01, 2.5e-3])
@pytest.mark.parametrize("order, bounds", [
    (1, (0.3, 0.6)),
    (2, (0.3, 0.4)),
    (3, (0.7, 0.8)),
    (4, (0.3, 0.5)),
])
def test_BdfIntegrator_ode_exp(ode_case, order, bounds, dt):
    ode, driver = ode_case
    driver.order = order
    driver.set_scenario(
        init={"f": 1.0},
        values={"df": "f"},
    )
    driver.add_recorder(
        DataFrameRecorder(includes=["*", "f - exp(t)"]),
        period=0.1,
    )
    driver.time_interval = [0.0, 1.0]
    driver.dt = dt

    assert driver.order == order

    ode.run_drivers()

    data = driver.recorder.export_data()
    data = data.drop(['Section', 'Status', 'Error code'], axis=1)

    f = np.asarray(data["f"])
    exact = np.exp(np.asarray(data["time"]))
    error = f / exact - 1.0
    error_max = float(np.linalg.norm(error, np.inf))
    lower, upper = bounds
    assert error_max > lower * dt**min(order, 3)
    assert error_max < upper * dt**min(order, 3)


@pytest.mark.parametrize("delta", [0.01, 0.001])
@pytest.mark.parametrize("order", [3, 4])
def test_BdfIntegrator_ode_stiff(delta, order):
    """Test BDF driver on a stiff ODE:
    dy/dt = y^2 - y^3
    with y(0) = 1 / delta,
    t in [0, 2 / delta]
    """
    ode = ScalarOde("ode", varname="y", max_abs_step=0.01)
    driver = ode.add_driver(BdfIntegrator(order=order, record_dt=True))
    driver.set_scenario(
        init={"y": delta},
        values={"dy": "y * y * (1 - y)"},
    )
    driver.add_recorder(
        DataFrameRecorder(includes=["*"]),
    )
    driver.time_interval = [0.0, 2.0 / delta]
    driver.dt = 0.05 / delta

    assert driver.order == order

    ode.run_drivers()

    data = driver.recorder.export_data()

    y = np.asarray(data["y"])
    assert y[-1] == pytest.approx(1, abs=1e-6)


@pytest.mark.parametrize("dt", [0.05, 0.01])
@pytest.mark.parametrize("order, bounds", [
    (1, (0.07, 0.09)),
    (2, (0.09, 0.10)),
    (3, (0.35, 0.65)),
    (4, (0.35, 0.65)),
])
def test_BdfIntegrator_tanks(two_tank_solution, order, bounds, dt):
    """Test BDF driver on a model with an intrinsic problem (loop).
    """
    system = CoupledTanks("system")
    driver = system.add_driver(BdfIntegrator(order=order))

    assert driver.order == order

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

    assert system.tank1.height == pytest.approx(system.tank2.height, abs=1e-3)
    assert system.tank1.height == pytest.approx(2.333, abs=1e-3)

    solution = two_tank_solution(system, init)
    assert solution.characteristic_time == pytest.approx(0.5766040)
    assert driver.dt < solution.characteristic_time

    data = driver.recorder.export_data()
    time = np.asarray(data['time'])
    exact = solution(time)
    error = np.vstack([
        np.asarray(data["tank1.height"]) / exact[0] - 1.0,
        np.asarray(data["tank2.height"]) / exact[1] - 1.0,
    ])
    error_max = float(np.linalg.norm(error[0], np.inf))
    lower, upper = bounds
    assert error_max > lower * dt**min(order, 3)
    assert error_max < upper * dt**min(order, 3)


@pytest.mark.parametrize("order", [2, 3, 4])
def test_BdfIntegrator_new_problem(order):
    """Test a transition which brings new intrinsic constraints
    generating discontinuities in the transient solution
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
    driver = system.add_driver(BdfIntegrator("driver", order=order))
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
    f = np.asarray(data["f"])
    t = np.asarray(data["time"])

    assert f == pytest.approx(4 * t**2, rel=1e-14)
    assert x == pytest.approx(
        [0., 0., 0., 0., 0., 0., 0., 0., 0., np.sqrt(2), 1.6, 1.8, 2.],
        rel=1e-14,
    )
    assert y == pytest.approx(
        [0., 0.04, 0.16, 0.36, 0.64, 1., 1.44, 1.96, 2., 0., 0., 0., 0.],
        rel=1e-14,
    )
