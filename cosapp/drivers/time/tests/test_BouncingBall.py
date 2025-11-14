import pytest
import numpy as np

from cosapp.systems import System
from cosapp.recorders import DataFrameRecorder
from cosapp.drivers.time import RungeKutta, CrankNicolson
from cosapp.drivers.time.base import AbstractTimeDriver
from typing import Tuple


class PointDynamics(System):
    """Point mass dynamics"""
    def setup(self):
        self.add_inward("mass", 1.0)
        self.add_inward("acc_ext", np.zeros(3))
        self.add_inward("force_ext", np.zeros(3))

        self.add_outward("force", np.zeros(3))
        self.add_outward("acc", np.zeros(3))

    def compute(self):
        mass = np.expand_dims(self.mass, axis=-1)
        self.force = self.force_ext + mass * self.acc_ext
        self.acc = self.force / mass


class PointFriction(System):
    """Point mass ~ v**2 friction model"""
    def setup(self):
        self.add_inward('v', np.zeros(3), desc="Velocity")
        self.add_inward('cf', 0.1, desc="Friction coefficient")

        self.add_outward("force", np.zeros(3))

    def compute(self):
        v = self.v
        k = self.cf * np.linalg.norm(v, axis=-1)
        self.force = -np.expand_dims(k, axis=-1) * v


class PointMass(System):
    """Free fall model with friction.
    """
    def setup(self):
        self.add_child(PointFriction('friction'), pulling=['cf', 'v'])
        self.add_child(PointDynamics('dynamics'), pulling=[
            'mass', 'force',
            {'acc_ext': 'g', 'acc': 'a'},
        ])

        self.connect(self.friction, self.dynamics, {"force": "force_ext"})

        self.add_transient('v', der='a')
        self.add_transient('x', der='v')

        self.g = np.r_[0, 0, -9.81]
        self.exec_order = ['friction', 'dynamics']


class BouncingBall(PointMass):
    """Bouncing ball model, combining a free fall model with a rebound event.
    """
    def setup(self):
        super().setup()
        self.add_event("rebound", trigger="x[2] <= 0")
        self.add_inward("cr", 1.0, limits=(0, 1), desc="Rebound coefficient")

    def transition(self):
        if self.rebound.present:
            cr = np.clip(self.cr, 0.0, 1.0)
            v = self.v
            if abs(v[2]) < 1e-6:
                v[2] = 0.0
            else:
                v[2] *= -cr


@pytest.fixture
def ball() -> BouncingBall:
    return BouncingBall('ball')


@pytest.fixture
def ball_case(ball: BouncingBall) -> Tuple[BouncingBall, RungeKutta]:
    """Bouncing ball + driver test case"""
    driver = ball.add_driver(RungeKutta(order=3))
    setup_driver(driver)
    return ball, driver


def setup_driver(driver: AbstractTimeDriver):
    driver.time_interval = (0, 4)
    driver.dt = 0.01

    # Add a recorder to capture time evolution in a dataframe
    driver.add_recorder(
        DataFrameRecorder(includes=['x', 'v', 'a']),
        period=0.05,
    )
    # Define a simulation scenario
    driver.set_scenario(
        init = {
            'x': np.array([0, 0, 0]),
            'v': np.array([8, 0, 9.5]),
        },
        values = {'mass': 1.5, 'cf': 0.2, 'cr': 0.98},
    )


def test_BouncingBall(ball_case):
    ball, driver = ball_case

    ball.run_drivers()

    # Retrieve recorded data
    data = driver.recorder.export_data()

    assert len(driver.recorded_events) == 3
    assert [record.time for record in driver.recorded_events] == pytest.approx(
        [1.457205, 2.517932, 3.3959818]
    )
    # Check that all positions are above ground level, within numerical tolerance
    x = np.asarray(data['x'].tolist())
    assert min(x[:, 2]) > -1e-13
    # Check positions, velocities and accelerations recorded at event times
    event_data = driver.event_data
    assert len(event_data) == 6
    ae = np.asarray(event_data['a'].tolist())
    ve = np.asarray(event_data['v'].tolist())
    xe = np.asarray(event_data['x'].tolist())
    np.testing.assert_allclose(
        xe, [
            [6.416269698, 0.0, 0.0],
            [6.416269698, 0.0, 0.0],  # no jump in x
            [8.508373346, 0.0, 0.0],
            [8.508373346, 0.0, 0.0],  # no jump in x
            [9.674569363, 0.0, 0.0],
            [9.674569363, 0.0, 0.0],  # no jump in x
        ],
        rtol=1e-9,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        ve, [
            [2.515184307, 0., -5.929093249],
            [2.515184307, 0.,  5.810511384],  # jump in vz
            [1.550781610, 0., -4.717679606],
            [1.550781610, 0.,  4.623326014],  # jump in vz
            [1.138267878, 0., -4.030487952],
            [1.138267878, 0.,  3.949878193],  # jump in vz
        ],
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        ae, [
            [-2.1598793362, 0., -4.718474119],
            [-2.1233265999, 0., -14.71525221],  # jump in a
            [-1.0268297864, 0., -6.686250073],
            [-1.0083142767, 0., -12.81607486],  # jump in a
            [-0.6356294918, 0., -7.559302200],
            [-0.6238647968, 0., -11.97485943],  # jump in a
        ],
        rtol=1e-9,
    )
    # Check that `driver.recorder` and `driver.event_data`
    # contain the same data at event times
    for record in driver.recorded_events:
        t = record.time
        data_t = data[data["time"] == t]
        data_e = event_data[event_data["time"] == t]
        assert len(data_t) == 2
        assert len(data_e) == 2
        assert list(data_e.columns) == list(data_t.columns)
        for name in ["a", "v", "x"]:
            field_t = np.asarray(data_t[name].tolist())
            field_e = np.asarray(data_e[name].tolist())
            assert np.array_equal(field_e, field_t), f"{name} @ {t = }"


def test_BouncingBall_final(ball_case):
    """Bouncing ball case with final rebound event.
    """
    ball, driver = ball_case
    ball.rebound.final = True

    ball.run_drivers()

    # Retrieve recorded data
    data = driver.recorder.export_data()
    x = np.asarray(data['x'].tolist())
    assert min(x[:, 2]) > -1e-13
    assert ball.x == pytest.approx([6.41626970, 0, 0])
    assert len(driver.recorded_events) == 1
    assert driver.recorded_events[0].time == pytest.approx(1.457205)


def test_BouncingBall_stop(ball: BouncingBall):
    """Bouncing ball case with stop criterion based on rebound event.
    """
    driver = ball.add_driver(RungeKutta(order=3))
    driver.time_interval = (0, 4)
    driver.dt = 0.01

    # Add a recorder to capture time evolution in a dataframe
    driver.add_recorder(
        DataFrameRecorder(includes=['x', 'v', 'a']),
        period=0.05,
    )
    # Define a simulation scenario
    driver.set_scenario(
        init = {
            'x': np.array([0, 0, 0]),
            'v': np.array([8, 0, 9.5]),
        },
        values = {'mass': 1.5, 'cf': 0.2, 'cr': 0.98},
        stop = ball.rebound,
    )

    ball.run_drivers()

    # Retrieve recorded data
    data = driver.recorder.export_data()
    x = np.asarray(data['x'].tolist())
    assert min(x[:, 2]) > -1e-13
    assert len(driver.recorded_events) == 1
    record = driver.recorded_events[-1]
    assert len(record.events) == 2
    assert record.time == pytest.approx(1.457205)
    assert record.events[0] is ball.rebound
    assert record.events[1] is driver.scenario.stop


def test_BouncingBall_frictionless(ball: BouncingBall):
    """Bouncing ball case with stop criterion based on rebound event.
    Check analytical solution for frictionless motion.
    Order 2 Runge-Kutta solution is expected to be exact.
    """
    driver = ball.add_driver(RungeKutta(order=2))
    driver.time_interval = (0, 4)
    driver.dt = 0.01
    # Add a recorder to capture time evolution in a dataframe
    driver.add_recorder(
        DataFrameRecorder(includes=['x', 'v', 'a']),
        period=0.05,
    )
    # Initial conditions
    x0 = np.r_[0, 0, 2]
    v0 = np.r_[8, 0, 9.5]

    # Define a simulation scenario
    driver.set_scenario(
        init = {'x': np.array(x0), 'v': np.array(v0)},
        values = {'mass': 1.0, 'cf': 0.0, 'cr': 1.0},
        stop = ball.rebound,
    )

    ball.run_drivers()

    # Retrieve recorded data
    data = driver.recorder.export_data()
    x = np.asarray(data['x'].tolist())
    v = np.asarray(data['v'].tolist())
    g = ball.g

    # Analytical solution
    t_rebound = -v0[2] / g[2] * (1 + np.sqrt(1 - 2 * g[2] * x0[2] / v0[2]**2))
    v_exact = lambda t: v0 + g * t
    x_exact = lambda t: x0 + (v0 + 0.5 * g * t) * t

    assert min(x[:, 2]) > -1e-14
    assert len(driver.recorded_events) == 1
    record = driver.recorded_events[-1]
    assert len(record.events) == 2
    assert record.time == pytest.approx(t_rebound, rel=1e-15)
    assert record.events[0] is ball.rebound
    assert record.events[1] is driver.scenario.stop

    # Check rebound point (before transition, hence -2 index)
    assert v[-2, :] == pytest.approx(v_exact(t_rebound), rel=1e-14)
    assert x[-2, :] == pytest.approx(x_exact(t_rebound), rel=1e-14)


@pytest.mark.parametrize("z0, z1, z2", [
    (0.1, 0.11, 0.77),  # All contacts expected within the first time step
    (1.1, 1.11, 1.77),  # All contacts expected within the second time step
    (0.6, 1.11, 1.77),
])
def test_BouncingBall_close_events(z0, z1, z2):
    """Set of 3 falling balls, with closely occuring contacts.
    Parameters z0 < z1 < z3 denote the initial heights of the points.
    All points have a vertical velocity of -1.
    """
    class Marbles(System):
        def setup(self, n_points=2):
            self.add_property('n_points', n_points)

            for i in range(n_points):
                self.add_child(BouncingBall(f"p{i}"), pulling=["g", "cf", "cr"])

    s = Marbles('s', n_points=3)

    driver = s.add_driver(RungeKutta(order=2, time_interval=(0, 2), dt=1.0))

    vz = -1.0
    z0, z1, z2 = sorted((z0, z1, z2))

    driver.set_scenario(
        init={
            'p0.x': [0., 0., z0],  # expected to hit the ground @ t = -z0 / vz
            'p1.x': [1., 0., z1],  # expected to hit the ground @ t = -z1 / vz
            'p2.x': [2., 0., z2],  # expected to hit the ground @ t = -z2 / vz
            'p0.v': [0., 0., vz],
            'p1.v': [0., 0., vz],
            'p2.v': [0., 0., vz],
        },
        values={
            'g': np.zeros(3),  # no gravity: rectilinear movement
            'cf': 0.0,  # frictionless motion
            'cr': 1.0,  # lossless rebound
        }
    )

    s.run_drivers()

    recorded_events = driver.recorded_events
    assert len(recorded_events) == 3
    assert len(recorded_events[0].events) == 1
    assert len(recorded_events[1].events) == 1
    assert len(recorded_events[2].events) == 1
    assert recorded_events[0].events[0] is s.p0.rebound
    assert recorded_events[0].time == pytest.approx(-z0 / vz, rel=1e-14)
    assert recorded_events[1].events[0] is s.p1.rebound
    assert recorded_events[1].time == pytest.approx(-z1 / vz, rel=1e-14)
    assert recorded_events[2].events[0] is s.p2.rebound
    assert recorded_events[2].time == pytest.approx(-z2 / vz, rel=1e-14)


def test_BouncingBall_early_stop(ball: BouncingBall):
    """Bouncing ball case with stop criterion occurring in the first time step.
    """
    driver = ball.add_driver(RungeKutta(order=2))
    driver.time_interval = (0, 1)
    driver.dt = 1.0

    # Define a simulation scenario
    driver.set_scenario(
        init = {'x': [0, 0, 2], 'v': [8, 0, 9.5]},
        values = {'mass': 1.0, 'cf': 0.0},
        stop = "t == 0.123",
    )

    ball.run_drivers()

    assert len(driver.recorded_events) == 1
    record = driver.recorded_events[-1]
    assert len(record.events) == 1
    assert record.time == pytest.approx(0.123, rel=1e-15)
    assert record.events[0] is driver.scenario.stop


def test_PointMass_array():
    """Test of `PointMass` system where N points are followed,
    using (Nx3) 2D arrays for positions, velocities and accelerations.
    """
    points = PointMass("points")

    points.x = np.zeros((4, 3))
    points.v = np.zeros((4, 3))
    points.run_once()

    assert points.a.shape == (4, 3)
    assert points.force.shape == (4, 3)

    driver = points.add_driver(RungeKutta(order=2))
    driver.time_interval = (0, 1)
    driver.dt = 0.01

    # Add a recorder to capture time evolution in a dataframe
    driver.add_recorder(
        DataFrameRecorder(includes=['x', 'v', 'a']),
        period=0.05,
    )
    # Define a simulation scenario
    driver.set_scenario(
        init = {
            'x': np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 2.], [0., 0., 0.]]),
            'v': np.array([[8., 0., 9.5], [8., 0., 9.5], [8., 0., 3.5], [1., 0.5, 2.]]),
        },
        values = {'mass': np.r_[1.5, 0.2, 1.0, 1.0], 'cf': 0.2},
    )

    points.run_drivers()

    # Retrieve recorded data
    data = driver.recorder.export_data()
    a = np.asarray(data['a'].tolist())
    v = np.asarray(data['v'].tolist())
    x = np.asarray(data['x'].tolist())

    assert a.shape == (21, 4, 3)
    assert v.shape == (21, 4, 3)
    assert x.shape == (21, 4, 3)
    assert points.a.shape == (4, 3)
    assert points.v.shape == (4, 3)
    assert points.x.shape == (4, 3)

    assert points.a.ravel() == pytest.approx([
        -2.18764161, 0.0, -7.93226721,
        -1.10919385, 0.0, -0.78464155,
        -2.96084805, 0.0, -4.25519848,
        -0.61001083, -0.30500542, -3.41086269,
    ])
    assert points.v.ravel() == pytest.approx([
        3.52845768, 0.0, -3.02860426,
        0.36783040, 0.0, -2.99298553,
        2.63885675, 0.0, -4.95071859,
        0.53769438, 0.26884719, -5.64052300,
    ])
    assert points.x.ravel() == pytest.approx([
        5.03500913, 0.0, 2.10452024,
        1.76360485, 0.0, -0.39983429,
        4.64287564, 0.0, 0.46830942,
        0.81043518, 0.40521759, -2.4484154,
    ])


def test_BouncingBall_implicit():
    """Test of `PointMass` system where N points are followed,
    using (Nx3) 2D arrays for positions, velocities and accelerations.
    """
    ball = BouncingBall("ball")
    driver = ball.add_driver(CrankNicolson())
    setup_driver(driver)

    ball.run_drivers()

    # Retrieve recorded data
    data = driver.recorder.export_data()

    assert len(driver.recorded_events) == 3
    assert [record.time for record in driver.recorded_events] == pytest.approx(
        [1.457205, 2.517932, 3.3959818], abs=1e-4,  # values from RK3 simulation
    )
    # Check that all positions are above ground level, within numerical tolerance
    x = np.asarray(data['x'].tolist())
    assert min(x[:, 2]) > -1e-13
    # Check positions, velocities and accelerations recorded at event times
    event_data = driver.event_data
    assert len(event_data) == 6
    ae = np.asarray(event_data['a'].tolist())
    ve = np.asarray(event_data['v'].tolist())
    xe = np.asarray(event_data['x'].tolist())
    # Note: reference values obtained from RK3 simulation
    np.testing.assert_allclose(
        xe, [
            [6.416269698, 0.0, 0.0],
            [6.416269698, 0.0, 0.0],  # no jump in x
            [8.508373346, 0.0, 0.0],
            [8.508373346, 0.0, 0.0],  # no jump in x
            [9.674569363, 0.0, 0.0],
            [9.674569363, 0.0, 0.0],  # no jump in x
        ],
        rtol=1e-4,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        ve, [
            [2.515184307, 0.0, -5.929093249],
            [2.515184307, 0.0,  5.810511384],  # jump in vz
            [1.550781610, 0.0, -4.717679606],
            [1.550781610, 0.0,  4.623326014],  # jump in vz
            [1.138267878, 0.0, -4.030487952],
            [1.138267878, 0.0,  3.949878193],  # jump in vz
        ],
        rtol=1e-4,
    )
    np.testing.assert_allclose(
        ae, [
            [-2.1598793362, 0.0, -4.718474119],
            [-2.1233265999, 0.0, -14.71525221],  # jump in a
            [-1.0268297864, 0.0, -6.686250073],
            [-1.0083142767, 0.0, -12.81607486],  # jump in a
            [-0.6356294918, 0.0, -7.559302200],
            [-0.6238647968, 0.0, -11.97485943],  # jump in a
        ],
        rtol=1e-4,
    )
    # Check that `driver.recorder` and `driver.event_data`
    # contain the same data at event times
    for record in driver.recorded_events:
        t = record.time
        data_t = data[data["time"] == t]
        data_e = event_data[event_data["time"] == t]
        assert len(data_t) == 2
        assert len(data_e) == 2
        assert list(data_e.columns) == list(data_t.columns)
        for name in ["a", "v", "x"]:
            field_t = np.asarray(data_t[name].tolist())
            field_e = np.asarray(data_e[name].tolist())
            assert np.array_equal(field_e, field_t), f"{name} @ {t = }"


def test_PointMass_array_implicit():
    """Test of `PointMass` system where N points are followed,
    using (Nx3) 2D arrays for positions, velocities and accelerations.
    Solver: Crank-Nicolson implicit time driver.
    """
    points = PointMass("points")

    driver = points.add_driver(CrankNicolson(time_interval=[0, 1], dt=0.01))

    points.x = np.zeros((4, 3))
    points.v = np.zeros((4, 3))
    points.run_once()

    assert points.a.shape == (4, 3)
    assert points.force.shape == (4, 3)

    # Add a recorder to capture time evolution in a dataframe
    driver.add_recorder(
        DataFrameRecorder(includes=['x', 'v', 'a']),
        period=0.05,
    )
    # Define a simulation scenario
    driver.set_scenario(
        init = {
            'x': np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 2.], [0., 0., 0.]]),
            'v': np.array([[8., 0., 9.5], [8., 0., 9.5], [8., 0., 3.5], [1., 0.5, 2.]]),
        },
        values = {
            'mass': np.r_[1.5, 0.2, 1.0, 1.0],
            'cf': 0.2,
        },
    )

    points.run_drivers()

    # Retrieve recorded data
    data = driver.recorder.export_data()
    a = np.asarray(data['a'].tolist())
    v = np.asarray(data['v'].tolist())
    x = np.asarray(data['x'].tolist())

    assert a.shape == (21, 4, 3)
    assert v.shape == (21, 4, 3)
    assert x.shape == (21, 4, 3)
    assert points.a.shape == (4, 3)
    assert points.v.shape == (4, 3)
    assert points.x.shape == (4, 3)

    assert points.a.ravel() == pytest.approx([
        -2.18750138, 0.0, -7.93189682,
        -1.10530540, 0.0, -0.77917504,
        -2.96061570, 0.0, -4.25477496,
        -0.61002496, -0.30501248, -3.41060707,
    ])
    assert points.v.ravel() == pytest.approx([
        3.52814899, 0.0, -3.02913081,
        0.36644113, 0.0, -2.99398313,
        2.63859424, 0.0, -4.95099205,
        0.53769614, 0.26884807, -5.64063619,
    ])
    assert points.x.ravel() == pytest.approx([
        5.03492155, 0.0, 2.10438271,
        1.76020420, 0.0, -0.40512246,
        4.64278130, 0.0, 0.46830410,
        0.81041907, 0.40520953, -2.44828186,
    ])
