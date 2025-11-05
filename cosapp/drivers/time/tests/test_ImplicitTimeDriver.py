import pytest
from unittest.mock import patch

import numpy as np

from cosapp.drivers.time.implicit import ImplicitTimeDriver
from cosapp.recorders import DataFrameRecorder
from cosapp.systems import System


@pytest.fixture(autouse=True)
def PatchImplicitTimeDriver():
    """Patch ImplicitTimeDriver to make it instanciable for tests"""
    patcher = patch.multiple(
        ImplicitTimeDriver,
        __abstractmethods__ = set(),
        _time_residues=lambda self, dt, current: np.array([]),
    )
    patcher.start()
    yield
    patcher.stop()


class SimpleSystem(System):

    def setup(self):
        self.add_inward("a", 1.0)
        self.add_inward("b", 1.0)
        self.add_outward("c", 0.0)

    def compute(self):
        self.c = 2 * self.a + self.b


def test_ImplicitTimeDriver_init_default():
    driver = ImplicitTimeDriver()
    assert driver.name == "Implicit time driver"
    assert driver.owner is None
    assert driver.dt is None
    assert driver.time_interval is None


def test_ImplicitTimeDriver_design_no_owner():
    """Check that design problem methods raise if no owner system is set.
    """
    driver = ImplicitTimeDriver(
        name="driver",
        time_interval=(0, 10),
        dt=0.5,
    )
    assert driver.name == "driver"
    assert driver.owner is None
    assert driver.dt == 0.5
    assert driver.time_interval == (0, 10)
    with pytest.raises(RuntimeError, match="driver has no owner system"):
        driver.add_unknown("x")
    with pytest.raises(RuntimeError, match="driver has no owner system"):
        driver.add_equation("y == 0")


@pytest.mark.parametrize("init", [
    {"c": 10.0},
    {"a": 3.5, "b": 6.5, "c": "a + b"},
])
def test_ImplicitTimeDriver_target_init(init):
    """Check target handling in implicit integrators.

    In this test, target 'c' is initialized to 10 at t=0,
    either from a constant value, or from the sum of 'a' and 'b' that equals 10.
    """
    class SystemWithTarget(SimpleSystem):
        """Extension of SimpleSystem with intrinsic unknown & target.
        """
        def setup(self):
            super().setup()
            self.add_unknown("a").add_target("c")

    system = SystemWithTarget("system")
    driver = system.add_driver(ImplicitTimeDriver("driver", time_interval=(0, 1), dt=1))
    driver.set_scenario(
        init=init,
        values={
            "b": "0.5 - 0.1 * t",
        },
    )
    driver.add_recorder(DataFrameRecorder())
    system.a = 1.0  # should not affect the result
    system.b = 2.5  # will be dynamically updated by driver
    system.c = 0.0  # will be reset to 10.0 as an initial condition
    system.run_drivers()

    assert driver._intrinsic_problem.shape == (1, 1)

    data = driver.recorder.export_data()
    a = np.asarray(data["a"])
    b = np.asarray(data["b"])
    c = np.asarray(data["c"])
    t = np.asarray(data["time"])

    assert a == pytest.approx((c - b) / 2)
    assert b == pytest.approx(0.5 - 0.1 * t)
    assert c == pytest.approx(np.full_like(t, 10.0), rel=1e-15)


def test_ImplicitTimeDriver_add_unknown():

    class SystemWithEquation(SimpleSystem):
        """Extension of SimpleSystem with an intrinsic equation.
        """
        def setup(self):
            super().setup()
            self.add_equation("c == 10")

    system = SystemWithEquation("system")
    driver = system.add_driver(ImplicitTimeDriver("driver", time_interval=(0, 1), dt=1.0))
    driver.add_unknown("a")

    driver.set_scenario(
        values={
            "b": "0.5 - 0.1 * t",
        },
    )
    driver.add_recorder(DataFrameRecorder())
    system.a = 1.0  # should not affect the result
    system.b = 2.5  # will be dynamically updated by driver
    system.c = 0.0  # will be calculated to 10.0 by the equation
    system.run_drivers()

    assert driver._intrinsic_problem.shape == (1, 1)

    data = driver.recorder.export_data()
    a = np.asarray(data["a"])
    b = np.asarray(data["b"])
    c = np.asarray(data["c"])
    t = np.asarray(data["time"])

    assert a == pytest.approx((c - b) / 2)
    assert b == pytest.approx(0.5 - 0.1 * t)
    assert c == pytest.approx(np.full_like(t, 10.0), rel=1e-15)


def test_ImplicitTimeDriver_add_equation():

    class SystemWithUnknown(SimpleSystem):
        """Extension of SimpleSystem with an intrinsic unknown.
        """
        def setup(self):
            super().setup()
            self.add_unknown("a")

    system = SystemWithUnknown("system")
    driver = system.add_driver(ImplicitTimeDriver("driver", time_interval=(0, 1), dt=1.0))
    driver.add_equation("c == 10")

    driver.set_scenario(
        values={
            "b": "0.5 - 0.1 * t",
        },
    )
    driver.add_recorder(DataFrameRecorder())
    system.a = 1.0  # should not affect the result
    system.b = 2.5  # will be dynamically updated by driver
    system.c = 0.0  # will be calculated to 10.0 by the equation
    system.run_drivers()

    assert driver._intrinsic_problem.shape == (1, 1)

    data = driver.recorder.export_data()
    a = np.asarray(data["a"])
    b = np.asarray(data["b"])
    c = np.asarray(data["c"])
    t = np.asarray(data["time"])

    assert a == pytest.approx((c - b) / 2)
    assert b == pytest.approx(0.5 - 0.1 * t)
    assert c == pytest.approx(np.full_like(t, 10.0), rel=1e-15)


def test_ImplicitTimeDriver_design():

    system = SimpleSystem("system")
    driver = system.add_driver(ImplicitTimeDriver("driver", time_interval=(0, 1), dt=1.0))

    driver.add_unknown("a").add_equation("c == 10")

    driver.set_scenario(
        values={
            "b": "0.5 - 0.1 * t",
        },
    )
    driver.add_recorder(DataFrameRecorder())
    system.a = 1.0  # should not affect the result
    system.b = 2.5  # will be dynamically updated by driver
    system.c = 0.0  # will be calculated to 10.0 by the equation
    system.run_drivers()

    assert driver._intrinsic_problem.shape == (1, 1)

    data = driver.recorder.export_data()
    a = np.asarray(data["a"])
    b = np.asarray(data["b"])
    c = np.asarray(data["c"])
    t = np.asarray(data["time"])

    assert a == pytest.approx((c - b) / 2)
    assert b == pytest.approx(0.5 - 0.1 * t)
    assert c == pytest.approx(np.full_like(t, 10.0), rel=1e-15)


def test_ImplicitTimeDriver_add_problem():

    class SystemWithDesignMethod(SimpleSystem):
        """Extension of SimpleSystem with a design method.
        """
        def setup(self):
            super().setup()
            design = self.add_design_method("a")
            design.add_unknown("a").add_equation("c == 10")

    system = SystemWithDesignMethod("system")
    driver = system.add_driver(ImplicitTimeDriver("driver", time_interval=(0, 1), dt=1.0))
    driver.add_problem(system.design_methods["a"])

    driver.set_scenario(
        values={
            "b": "0.5 - 0.1 * t",
        },
    )
    driver.add_recorder(DataFrameRecorder())
    system.a = 1.0  # should not affect the result
    system.b = 2.5  # will be dynamically updated by driver
    system.c = 0.0  # will be calculated to 10.0 by the equation
    system.run_drivers()

    assert driver._intrinsic_problem.shape == (1, 1)

    data = driver.recorder.export_data()
    a = np.asarray(data["a"])
    b = np.asarray(data["b"])
    c = np.asarray(data["c"])
    t = np.asarray(data["time"])

    assert a == pytest.approx((c - b) / 2)
    assert b == pytest.approx(0.5 - 0.1 * t)
    assert c == pytest.approx(np.full_like(t, 10.0), rel=1e-15)


def test_ImplicitTimeDriver_clear_problem():
    """Test a combination of `add_unknown` and `add_equation`,
    plus method `clear_problem`.
    """
    system = SimpleSystem("system")
    driver = system.add_driver(ImplicitTimeDriver("driver", time_interval=(0, 1), dt=1.0))

    driver.add_unknown("a").add_equation("c == 10")

    driver.set_scenario(
        values={
            "b": "0.5 - 0.1 * t",
        },
    )
    driver.add_recorder(DataFrameRecorder())
    system.a = 1.0  # should not be modified
    system.b = 2.5  # will be dynamically updated by driver
    system.c = 0.0  # will be calculated as an output variable

    driver.clear_problem()
    system.run_drivers()

    assert driver._intrinsic_problem.is_empty()

    data = driver.recorder.export_data()
    a = np.asarray(data["a"])
    b = np.asarray(data["b"])
    c = np.asarray(data["c"])
    t = np.asarray(data["time"])

    assert a == pytest.approx(np.full_like(t, 1.0), abs=0.0)
    assert b == pytest.approx(0.5 - 0.1 * t, rel=1e-15)
    assert c == pytest.approx(2 * a + b, rel=1e-15)
