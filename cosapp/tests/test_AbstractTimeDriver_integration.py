import pytest
import numpy as np

from cosapp.drivers import CrankNicolson, EulerImplicit, BdfIntegrator, RungeKutta
from cosapp.recorders import DataFrameRecorder
from .conftest import MassFreeFall, FreeFallSolution


def error(simu, exact):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(exact != 0, np.absolute(simu / exact - 1), np.absolute(simu - exact))


@pytest.mark.parametrize("integrator_cls, dt, expected_error", [
    (EulerImplicit, 0.01, dict(x=1e-1, v=2e-2, a=5e-3)),
    (CrankNicolson, 0.01, dict(x=5e-5, v=5e-5, a=1e-5)),
    (BdfIntegrator, 0.01, dict(x=1e-4, v=1e-4, a=5e-5)),
    (RungeKutta, 0.01, dict(x=5e-5, v=5e-5, a=2e-5)),
])
def test_MassFreeFall_arrays(integrator_cls, dt, expected_error):
    """Test of `MassFreeFall` system where N points are followed,
    using (Nx3) 2D arrays for positions, velocities and accelerations.
    Solver: various explicit and implicit time drivers.
    """
    points = MassFreeFall("points")

    shape = (4, 3)  # 4 points in 3D space
    points.x = np.zeros(shape)
    points.v = np.zeros(shape)
    points.run_once()

    assert points.a.shape == shape

    driver = points.add_driver(integrator_cls(time_interval=[0, 1], dt=dt))

    # Add a recorder to capture time evolution in a dataframe
    driver.add_recorder(
        DataFrameRecorder(includes=["x", "v", "a"]),
        period=0.05,
    )

    # Define a simulation scenario
    x0 = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 2.0],
        [0.0, 0.0, 0.0],
    ])
    v0 = np.array([
        [8.0, 0.0, 9.5],
        [8.0, 0.0, 9.5],
        [8.0, 0.0, 3.5],
        [1.0, 0.5, 2.0],
    ])

    driver.set_scenario(
        init = {
            "x": x0,
            "v": v0,
        },
        values = {
            "mass": np.r_[1.5, 0.2, 1.0, 1.0],
            "cf": 0.2,
        },
    )

    points.run_drivers()

    # Retrieve recorded data
    data = driver.recorder.export_data()

    a = np.asarray(data["a"].tolist())
    v = np.asarray(data["v"].tolist())
    x = np.asarray(data["x"].tolist())
    time = np.asarray(data["time"])

    n_steps = 21
    assert time.shape == (n_steps,)
    assert a.shape == (n_steps,) + shape
    assert v.shape == (n_steps,) + shape
    assert x.shape == (n_steps,) + shape
    assert points.a.shape == shape
    assert points.v.shape == shape
    assert points.x.shape == shape
    assert time[-1] == points.time

    solution = FreeFallSolution(points, v0=v0, x0=x0)

    a_exact = solution.a(time[-1])
    v_exact = solution.v(time[-1])
    x_exact = solution.x(time[-1])

    assert error(points.a, a_exact).max() <= expected_error["a"]
    assert error(points.v, v_exact).max() <= expected_error["v"]
    assert error(points.x, x_exact).max() <= expected_error["x"]
