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


def test_ImplicitTimeDriver_init_default():
    driver = ImplicitTimeDriver()
    assert driver.name == "Implicit time driver"
    assert driver.owner is None
    assert driver.dt is None
    assert driver.time_interval is None


@pytest.mark.parametrize("init", [
    {"c": 10.0},
    {"a": 3.5, "b": 6.5, "c": "a + b"},
])
def test_ImplicitTimeDriver_target_init(init):
    """Check target handling in implicit integrators.

    In this test, target 'c' is initialized to 10 at t=0,
    either from a constant value, or from the sum of 'a' and 'b' that equals 10.
    """
    class SimpleSystem(System):

        def setup(self):
            self.add_inward("a", 1.0)
            self.add_inward("b", 1.0)

            self.add_outward("c", 0.0)

            self.add_unknown("a").add_target("c")

        def compute(self):
            self.c = 2 * self.a + self.b

    system = SimpleSystem("system")
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

    data = driver.recorder.export_data()
    a = np.asarray(data["a"])
    b = np.asarray(data["b"])
    c = np.asarray(data["c"])
    t = np.asarray(data["time"])

    assert a == pytest.approx((c - b) / 2)
    assert b == pytest.approx(0.5 - 0.1 * t)
    assert c == pytest.approx(np.full_like(t, 10.0), rel=1e-15)
