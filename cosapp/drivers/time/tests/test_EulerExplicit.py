import pytest

import numpy as np
from cosapp.drivers import EulerExplicit, NonLinearSolver, RunSingleCase
import cosapp.recorders as recorders


def test_EulerExplicit_init_default():
    driver = EulerExplicit()
    assert driver.owner is None
    assert driver.dt is None
    assert driver.time_interval is None
    assert driver.name == "Explicit Euler time driver"


def test_EulerExplicit_ode_solve_1(ode_case_1):
    # First run
    ode, driver = ode_case_1(EulerExplicit, dt=1e-3, time_interval=[0, 1])

    ode.run_drivers()
    start, end = driver.time_interval
    assert ode.f == pytest.approx(ode(end), 1e-3)
    assert end == 1

    # Second run, where end time is not a multiple of time step
    ode, driver = ode_case_1(EulerExplicit, dt=1e-3, time_interval=[0, 1 + 2e-4])
    ode.run_drivers()
    start, end = driver.time_interval
    assert ode.f == pytest.approx(ode(end), 1e-3)
    assert end == pytest.approx(1.0002)


@pytest.mark.parametrize("dt", [
    0.1, 0.01,
    # 1e-3,  # long
])
def test_EulerExplicit_twoTanks(two_tank_case, two_tank_solution, dt):
    system, driver = two_tank_case(EulerExplicit, dt=dt, time_interval=[0, 5])
    solver = driver.add_child(NonLinearSolver('solver', factor=1.0))

    h1_0, h2_0 = init = (3, 1)

    driver.set_scenario(
        name = 'run',
        init = {
            'tank1.height': h1_0,  # initial conditions
            'tank2.height': h2_0,
        },
        values = {
            'pipe.D': 0.07,  # fixed values
            'pipe.L': 2.5,
            'tank1.area': 2,
        }
    )
    
    assert driver.scenario.context is system
    assert driver.scenario.name == 'run'
    
    recorder = driver.add_recorder(recorders.DataFrameRecorder(includes='tank?.height'), period=0.1)
    # recorder = driver.add_recorder(recorders.DSVRecorder('twoTanks_Euler.csv', includes=['tank?.height', ]), period=0.1)
    assert driver.recording_period == 0.1

    system.run_drivers()

    assert system.tank1.height < 3
    assert system.tank2.height > 1
    assert system.tank1.height == pytest.approx(system.tank2.height, rel=1e-3)

    assert len(recorder.data) == 51
    heights = recorder.data.values[:, -2:]
    solution = two_tank_solution(system, init)
    assert solution.characteristic_time == pytest.approx(0.5766040318109212)
    error = 0
    for i, h1 in enumerate(heights[1:, 0]):
        t = (i + 1) * driver.recording_period
        exact = solution(t)
        error = max(error, abs(h1 - exact[0]))
    # Test that maximum error ~ dt
    assert error < 0.3 * dt
    assert error > 0.2 * dt
