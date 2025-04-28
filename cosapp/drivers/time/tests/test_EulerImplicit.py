import pytest

import numpy as np
from cosapp.drivers import EulerImplicit
from cosapp.recorders import DataFrameRecorder
from .conftest import ScalarOde


@pytest.fixture
def ode_case():
    ode = ScalarOde("ode")
    driver = ode.add_driver(EulerImplicit())
    driver.add_recorder(DataFrameRecorder())
    return ode, driver


@pytest.mark.parametrize("dt", [0.1, 0.01, 0.0025])
def test_EulerImplicit_ode_exp(ode_case, dt):
    ode, driver = ode_case
    driver.set_scenario(
        init={"f": 1.0},
        values={"df": "f"},
    )
    driver.add_recorder(DataFrameRecorder(), period=0.1)
    driver.time_interval = [0.0, 1.0]
    driver.dt = dt

    ode.run_drivers()

    data = driver.recorder.export_data()
    time = np.asarray(data["time"])
    f_simu = np.asarray(data["f"])
    f_exact = np.exp(time)
    error_max = np.linalg.norm(f_simu - f_exact, np.inf)
    assert error_max < 1.5 * dt
    assert error_max > 1.3 * dt
