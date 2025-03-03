import pytest

from cosapp.systems import System
from cosapp.drivers import RungeKutta, EulerExplicit, CrankNicolson
from cosapp.recorders import DataFrameRecorder
from cosapp.utils.testing import pickle_roundtrip, are_same


class TimeDriverSystem(System):
    def setup(self):
        self.add_inward('area', 1.0, desc='Cross-section area')
        self.add_inward('x', 1.0)
        self.add_inward('z', 1.0)
        self.add_inward('flowrate', 1.0)
        self.add_transient('height', der='flowrate * x / area')

        self.add_event("low_x", trigger="x < 0.1")


drivers = [RungeKutta, EulerExplicit, CrankNicolson]


@pytest.fixture(params=drivers)
def system(request):
    system = TimeDriverSystem('s')
    driver = request.param
    system.add_driver(driver("driver", dt=1., time_interval=[0, 10.]))
    return system

@pytest.mark.parametrize("driver", drivers)
def test_standalone(driver):
    """Test pickling of standalone driver."""

    driver = driver('driver')
    driver_copy = pickle_roundtrip(driver)
    assert are_same(driver, driver_copy)


def test_default(system):
    """Test driver with default options."""

    system_copy = pickle_roundtrip(system)
    assert are_same(system, system_copy)


def test_scenario(system):
    """Test driver with a scenario setting."""

    driver = system.drivers["driver"]
    driver.set_scenario(values={'x': 'exp(-2 * t)', 'z': 't'})

    system_copy = pickle_roundtrip(system)
    assert are_same(system, system_copy)

    driver_copy = system_copy.drivers["driver"]
    assert driver_copy.scenario.case_values[0].lhs == "x"
    assert driver_copy.scenario.case_values[0].rhs == "exp(-2 * t)"
    assert driver_copy.scenario.case_values[1].lhs == "z"
    assert driver_copy.scenario.case_values[1].rhs == "t"


def test_recorder(system):
    """Test pickling of driver with recorder."""

    driver = system.drivers["driver"]
    driver.add_recorder(
        DataFrameRecorder(hold=True, includes=['x', 'height', 'flowrate']),
        period=0.05,
    )
    assert are_same(system, pickle_roundtrip(system))

    system_copy = pickle_roundtrip(system)
    driver_copy = system_copy.drivers["driver"]
    assert driver_copy.recorder.hold
    assert sorted(driver_copy.recorder.includes) == sorted(['x', 'height', 'flowrate', 'time'])
    assert driver_copy.recorder._owner is driver_copy
    assert driver_copy.recorder._watch_object is system_copy


def test_execution(system):
    """Test execution of pickled driver."""

    driver = system.drivers["driver"]
    driver.set_scenario(values={'x': 'exp(-2 * t)', 'z': 't'})
    driver.add_recorder(
        DataFrameRecorder(hold=True, includes=['x', 'height', 'flowrate'], raw_output=True),
        period=1.
    )
    system.run_drivers()

    system_copy = pickle_roundtrip(system)
    driver_copy = system_copy.drivers["driver"]
    df = driver_copy.recorder.export_data()
    assert df.shape == (13, 8)
    assert df.iloc[-1]["time"] == 10.0
    assert df.iloc[-1]["height"] > 1.5
    assert all(df["flowrate"] == 1.0)
    assert 1.0 <= df["height"].all() < 1.52
    assert df.iloc[-1]["x"] < 1e-8
    assert all(df.iloc[2:4]["x"].round(2) == 0.1)
