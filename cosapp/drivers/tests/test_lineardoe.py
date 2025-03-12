import sys
import numpy as np
import pytest

from cosapp.base import System
from cosapp.drivers import LinearDoE, RunOnce, NonLinearSolver
from cosapp.recorders import DataFrameRecorder
from cosapp.tests.library.systems import Multiply2, MultiplySystem2
from cosapp.core.execution import ExecutionPolicy, ExecutionType, WorkerStartMethod
from cosapp.utils.testing import pickle_roundtrip, are_same


def test_LinearDoE_init():
    s = Multiply2("mult")
    d = s.add_driver(LinearDoE("doe"))
    assert isinstance(d, LinearDoE)


def test_LinearDoE_add_input_var():
    s = Multiply2("mult")
    d = s.add_driver(LinearDoE("doe"))

    d.add_input_var("K1", 0.0, 10.0)
    assert d.input_vars == {"K1": {"lower": 0.0, "upper": 10.0, "count": 2}}

    d.add_input_var({"K1": {"lower": 0.2, "upper": 20.0}})
    assert d.input_vars == {"K1": {"lower": 0.2, "upper": 20.0, "count": 2}}

    d.add_input_var(
        {"K1": {"lower": 0.2, "upper": 20.0}, "K2": {"lower": 0.1, "upper": 200.0}}
    )
    assert d.input_vars == {
        "K1": {"lower": 0.2, "upper": 20.0, "count": 2},
        "K2": {"lower": 0.1, "upper": 200.0, "count": 2},
    }

    with pytest.raises(TypeError):
        d.add_input_var({"myvar": {"upper": 20.0}})
    with pytest.raises(TypeError):
        d.add_input_var({"myvar": 20.0})

    with pytest.raises(AttributeError):
        d.add_input_var("myvar", 0.0, 10.0)
    with pytest.raises(AttributeError):
        d.add_input_var({"myvar": {"lower": 0.2, "upper": 20.0}})


def test_LinearDoE__build_cases():
    s = Multiply2("mult")
    d = s.add_driver(LinearDoE("doe"))

    d._build_cases()
    assert d.cases == [()]

    d.add_input_var("K1", lower=0.2, upper=20.0)
    d.add_input_var("K2", lower=0.1, upper=200.0)
    d._build_cases()
    assert np.allclose(
        np.asarray(d.cases),
        np.asarray([(0.2, 0.1), (0.2, 200.0), (20.0, 0.1), (20.0, 200.0)]),
        rtol=1e-9,
    )

    d.add_input_var("K1", lower=0.2, upper=20.0, count=3)
    d.add_input_var("K2", lower=0.1, upper=200.0)
    d._build_cases()
    assert np.allclose(
        np.asarray(d.cases),
        np.asarray(
            [
                (0.2, 0.1),
                (0.2, 200.0),
                (10.1, 0.1),
                (10.1, 200.0),
                (20.0, 0.1),
                (20.0, 200.0),
            ]
        ),
        rtol=1e-9,
    )


def test_LinearDoE_execution():
    """Test normal execution of driver `LinearDoE`"""
    s = Multiply2("mult")
    doe = s.add_driver(LinearDoE("doe"))
    doe.add_child(RunOnce("run"))
    doe.add_recorder(DataFrameRecorder(includes=["K?", "p_out.x"], raw_output=True))

    doe.add_input_var(
        {
            "K1": dict(lower=0.0,  upper=20.0, count=3),
            "K2": dict(lower=0.0,  upper=200.0, count=3),
        }
    )
    s.run_drivers()

    assert s.K1 == 20.0
    assert s.K2 == 200.0
    df = doe.recorder.export_data()
    assert len(df) == 9
    assert df.iloc[4]["K1"] == 10.0
    assert df.iloc[4]["K2"] == 100.0
    assert df.iloc[4]["p_out.x"] == 1000.0
    assert df.iloc[8]["K1"] == 20.0
    assert df.iloc[8]["K2"] == 200.0
    assert df.iloc[8]["p_out.x"] == 4000.0


class TestLinearDoEPickling:
  
    @pytest.fixture
    def system(self):
        s: System = Multiply2('s')
        s.add_driver(LinearDoE('doe'))
        return s

    def test_standalone(self):
        """Test pickling of standalone driver."""

        doe = LinearDoE('doe')

        doe_copy = pickle_roundtrip(doe)
        assert are_same(doe, doe_copy)

    def test_default(self, system):
        """Test driver with default options."""

        system_copy = pickle_roundtrip(system)
        assert are_same(system, system_copy)

    def test_input_vars(self, system):
        """Test driver with input vars."""

        doe = system.drivers["doe"]
        doe.add_input_var("K1", 0.0, 10.0, 11)
        doe.add_input_var({"K2": {"lower": 0.1, "upper": 200.0, "count": 8}})
        system_copy = pickle_roundtrip(system)
        assert are_same(system, system_copy)

        doe_copy = system_copy.drivers["doe"]
        assert doe_copy.input_vars == {
            "K1": {"lower": 0.0, "upper": 10.0, "count": 11},
            "K2": {"lower": 0.1, "upper": 200.0, "count": 8},
        }

    def test_recorder(self, system):
        """Test pickling of driver with recorder."""

        doe = system.drivers["doe"]
        doe.add_recorder(DataFrameRecorder(hold=True, includes="a*"))
        assert are_same(system, pickle_roundtrip(system))

        system_copy = pickle_roundtrip(system)
        doe_copy = system_copy.drivers["doe"]
        assert doe_copy.recorder.hold
        assert doe_copy.recorder.includes == ["a*"]
        assert doe_copy.recorder._owner is doe_copy
        assert doe_copy.recorder._watch_object is system_copy

    def test_execution(self, system):
        """Test execution of pickled driver."""

        doe = system.drivers["doe"]
        doe.add_child(RunOnce("run"))
        doe.add_recorder(DataFrameRecorder(includes=["K?", "p_out.x"], raw_output=True))

        doe.add_input_var(
            {
                "K1": dict(lower=0.0,  upper=20.0, count=3),
                "K2": dict(lower=0.0,  upper=200.0, count=3),
            }
        )
        system.run_drivers()

        system_copy = pickle_roundtrip(system)
        doe_copy = system_copy.drivers["doe"]
        assert system_copy.K1 == 20.0
        assert system_copy.K2 == 200.0

        df = doe_copy.recorder.export_data()
        assert len(df) == 9
        assert df.iloc[4]["K1"] == 10.0
        assert df.iloc[4]["K2"] == 100.0
        assert df.iloc[4]["p_out.x"] == 1000.0
        assert df.iloc[8]["K1"] == 20.0
        assert df.iloc[8]["K2"] == 200.0
        assert df.iloc[8]["p_out.x"] == 4000.0

def _get_start_methods():
    if sys.platform == "win32":
        return (WorkerStartMethod.SPAWN, )
    return (WorkerStartMethod.FORK, WorkerStartMethod.SPAWN)


@pytest.mark.parametrize(argnames="nprocs", argvalues=[2, 4])    
@pytest.mark.parametrize("start_method", _get_start_methods())
def test_LinearDoE_multiprocessing(nprocs, start_method):
    """Tests the execution of a MonteCarlo on multiple (sub)processes."""
    s = MultiplySystem2("s")
    s.run_once()

    doe = s.add_driver(
        LinearDoE(
            "doe", 
            execution_policy=ExecutionPolicy(
                workers_count=nprocs,
                execution_type=ExecutionType.MULTI_PROCESSING,
                start_method=start_method,
            )
        )
    )
    nls = doe.add_child(NonLinearSolver('solver'))
    doe.add_child(RunOnce("runonce"))
    rec = nls.add_recorder(DataFrameRecorder("*x", hold=True))

    doe.add_input_var("mult1.p_in.x", 0.9, 1.1, 16)

    s.run_drivers()
    data = rec.export_data()
    assert len(data) == 16

    assert np.array_equal(data["mult1.p_in.x"], np.linspace(0.9, 1.1, 16))
