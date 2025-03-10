import sys
import pytest
import numpy as np

from cosapp.utils.distributions import Normal, Uniform
from cosapp.drivers import (
    MonteCarlo,
    NonLinearMethods,
    NonLinearSolver,
    RunOnce,
    RunSingleCase,
    EulerExplicit
)
from cosapp.recorders import DataFrameRecorder
from cosapp.systems import System
from cosapp.tests.library.systems import (
    IterativeNonLinear,
    Multiply2,
    Multiply4,
    MultiplySystem2,
)
from cosapp.utils.execution import ExecutionPolicy, ExecutionType, WorkerStartMethod
from cosapp.utils.testing import no_exception, pickle_roundtrip, are_same, has_keys


class SimpleCentered(System):
    def setup(self):
        self.add_inward("x", 1.0)
        self.add_inward("K1", value=5, distribution=Uniform(worst=-0.1, best=0.1))
        self.add_outward("y")

    def compute(self):
        self.y = self.x * self.K1


class SimpleUncentered(System):
    def setup(self):
        self.add_inward("x", 1.0)
        self.add_inward("K1", value=5, distribution=Uniform(worst=-2, best=1))
        self.add_outward("y")

    def compute(self):
        self.y = self.x * self.K1


class SystemTime(System):
    def setup(self):
        self.add_inward("a", 0.0, distribution=Uniform(0.0, 1.0))
        self.add_inward("m", 0.0)
        self.add_transient("m", der="1.")


class SystemEvent(System):
    def setup(self):
        self.add_inward("m", 10.0)
        self.add_inward("event_time", -1.0, distribution=Uniform(worst=0.0, best=6.0))

        self.add_transient("m", der="-1.")
        self.add_event("m_event", trigger="time == event_time")

    def transition(self):
        if self.m_event.present:
            self.m = 0.0


def test_MonteCarlo_setup():
    mc = MonteCarlo("statistics")
    assert not mc.linear
    assert len(mc.random_variables) == 0
    assert len(mc.responses) == 0
    assert mc.cases is None


def test_MonteCarlo__build_cases():
    s = Multiply2("mult")
    s.K1 = 5.0
    s.K2 = 10.0

    mc = s.add_driver(MonteCarlo("mc"))

    mc.add_random_variable("K1")

    mc._build_cases()
    assert mc.cases.shape == (mc.draws, len(mc.random_variables))

    mc.draws = 33
    mc.add_random_variable("K2")
    mc._build_cases()
    assert mc.cases.shape == (mc.draws, len(mc.random_variables))


def test_MonteCarlo_add_random_variable():
    s = Multiply2("mult")
    s.K1 = 5.0
    s.K2 = 10.0

    mc = s.add_driver(MonteCarlo("mc"))

    mc.add_random_variable({"K1", "K2"})
    assert set(mc.random_variables) == {"K1", "K2"}
    
    assert mc.random_variables["K1"].variable is s.name2variable["K1"]
    assert mc.random_variables["K1"].connector is None
    assert mc.random_variables["K1"].distribution is s.inwards.get_details("K1").distribution

    assert mc.random_variables["K2"].variable is s.name2variable["K2"]
    assert mc.random_variables["K2"].connector is None
    assert mc.random_variables["K2"].distribution is s.inwards.get_details("K2").distribution

    mc.random_variables.clear()
    mc.add_random_variable(["K1"])
    assert set(mc.random_variables) == {"K1"}

    mc.random_variables.clear()
    mc.add_random_variable("K1")
    assert set(mc.random_variables) == {"K1"}

    # Don't duplicate input
    mc.add_random_variable("K1")
    assert set(mc.random_variables) == {"K1"}

    # Protection
    with pytest.raises(TypeError):
        mc.add_random_variable(1.0)
    with pytest.raises(
        TypeError, match=r"'p_out.x' is not an input variable"
    ):
        mc.add_random_variable("p_out.x")

    with pytest.raises(
        AttributeError, match=r"'[\w\.]+' not found in System '[\w\.]+'"
    ):
        mc.add_random_variable("x")

    with pytest.raises(TypeError, match=r"'[\w\.]+' is not a variable\."):
        mc.add_random_variable("p_out")

    with pytest.raises(
        ValueError, match=r"No distribution specified for '[\w\.]+\.\w+'"
    ):
        mc.add_random_variable("p_in.x")

    # Add a connected variable
    s = Multiply2("mult")
    t = System("top")
    t.add_child(s, pulling="p_in")
    dummy = Normal(best=2.0, worst=0.0)
    s.p_in.get_details("x").distribution = dummy
    mc = t.add_driver(MonteCarlo("mc"))
    mc.add_random_variable("mult.p_in.x")
    connector = list(filter(lambda c: c.sink is s.p_in, t.all_connectors()))[0]
    assert set(mc.random_variables) == {"mult.p_in.x"}
    random_variable = mc.random_variables["mult.p_in.x"]
    assert random_variable.variable is t.name2variable["mult.p_in.x"]
    assert random_variable.connector is connector
    assert random_variable.distribution is dummy


def test_MonteCarlo_add_response():
    s = Multiply2("mult")
    mc = s.add_driver(MonteCarlo("mc"))

    mc.add_response("K1")
    assert "K1" in mc.responses
    assert set(mc.responses) == {"K1"}

    mc.add_response("K1")
    assert set(mc.responses) == {"K1"}

    mc.add_response("K2")
    assert set(mc.responses) == {"K1", "K2"}

    mc.responses.clear()
    mc.add_response(["K1", "K2"])
    assert mc.responses == ["K1", "K2"]
    assert set(mc.responses) == {"K1", "K2"}

    mc.responses.clear()
    mc.add_response({"K1", "K2"})
    assert set(mc.responses) == {"K1", "K2"}

    mc.responses.clear()
    with pytest.raises(TypeError):
        mc.add_response(1)
    with pytest.raises(AttributeError, match=r"'[\w\.]+' not found in System '[\w\.]+'"):
        mc.add_response("x")
    with pytest.raises(TypeError):
        mc.add_response([1])


def test_MonteCarlo__precompute():
    s = Multiply2("mult")
    s.K1 = 5.0
    s.K2 = 10.0

    # Classical use case
    mc = s.add_driver(MonteCarlo("mc"))
    mc.draws = 10
    mc.add_random_variable({"K1", "K2"})

    assert mc.cases is None
    mc.setup_run()
    assert mc.cases.shape == (mc.draws, len(mc.random_variables))

    # Linear use case
    s.drivers.clear()
    mc = s.add_driver(MonteCarlo("mc"))
    mc.add_driver(RunOnce("run"))
    mc.draws = 10
    mc.linear = True
    mc.add_random_variable(["K1", "K2"])
    mc.add_response("p_out.x")

    assert mc.cases is None
    assert mc.X0 is None
    assert mc.Y0 is None
    assert mc.A is None
    mc.setup_run()
    mc._precompute()
    assert mc.cases.shape == (mc.draws, len(mc.random_variables))
    assert len(mc.X0) == 2
    np.testing.assert_array_equal(mc.X0, np.r_[s.K1, s.K2])
    assert len(mc.Y0) == 1
    np.testing.assert_array_equal(mc.Y0, np.r_[s.p_out.x])
    assert mc.A.shape == (mc.Y0.size, mc.X0.size)


def test_MonteCarlo__precase():
    s = Multiply2("mult")
    s.K1 = 5.0
    s.K2 = 10.0

    # No connected variable
    mc = s.add_driver(MonteCarlo("mc"))
    mc.draws = 2
    mc.add_random_variable({"K1", "K2"})

    mc.setup_run()
    mc._precase(1, mc.cases[1])
    assert s.K1 != pytest.approx(5, abs=1e-4)
    assert s.K2 != pytest.approx(10, abs=1e-4)

    # With connected variable
    t = System("top")
    s = t.add_child(Multiply2("mult"), pulling="p_in")
    s.p_in.x = 22.0
    s.K2 = 10.0
    s.p_in.get_details("x").distribution = Normal(best=-1.0, worst=-3.0)
    mc = t.add_driver(MonteCarlo("mc"))
    mc.add_random_variable({"mult.p_in.x", "mult.K2"})

    mc.setup_run()
    mc._precase(1, mc.cases[1])
    t.run_once()
    assert t.mult.p_in.x != pytest.approx(22, abs=1e-4)
    assert t.mult.K2 != pytest.approx(10, abs=1e-4)


def test_MonteCarlo__postcase():
    s = Multiply2("mult")
    s.K1 = 5.0
    s.K2 = 10.0

    # No connected variable
    mc = s.add_driver(MonteCarlo("mc"))
    mc.draws = 2
    mc.add_random_variable({"K1", "K2"})

    mc.setup_run()
    mc._precase(1, mc.cases[1])
    mc._postcase(1, mc.cases[1])
    assert s.K1 == pytest.approx(5.0, abs=1e-4)
    assert s.K2 == pytest.approx(10.0, abs=1e-4)

    s.drivers.clear()
    # With connected variable
    s.p_in.x = 22.0
    s.K2 = 10.0
    t = System("top")
    s.p_in.get_details("x").distribution = Normal(best=-1.0, worst=-3.0)
    t.add_child(s, pulling="p_in")
    mc = t.add_driver(MonteCarlo("mc"))
    mc.add_random_variable({"mult.p_in.x", "mult.K2"})

    mc.setup_run()
    mc._precase(1, mc.cases[1])
    s.run_once()
    mc._postcase(1, mc.cases[1])
    s.run_once()
    assert t.mult.p_in.x == pytest.approx(22.0, abs=1e-4)
    assert t.mult.K2 == pytest.approx(10.0, abs=1e-4)


def test_MonteCarlo_cases_centered():
    s = SimpleCentered("s")
    distribution = Normal(best=0.1, worst=-0.1)
    s.inwards.get_details("K1").distribution = distribution
    mc = s.add_driver(MonteCarlo("mc"))
    rec = mc.add_recorder(DataFrameRecorder(includes=["K1"], raw_output=True))
    mc.add_child(RunOnce("run"))

    mc.add_random_variable({"K1"})

    mc.draws = 100
    s.run_drivers()
    df = rec.export_data()
    assert df["K1"].mean() == pytest.approx(s.K1, abs=1e-3)
    # Trick to get std from scipy object
    assert df["K1"].std() == pytest.approx(distribution._rv.kwds["scale"], abs=1e-2)


def test_MonteCarlo_cases_uncentered():

    s = SimpleCentered("s")
    distribution = Normal(best=0.0, worst=-0.4)
    s.inwards.get_details("K1").distribution = distribution
    mc = s.add_driver(MonteCarlo("mc"))
    rec = mc.add_recorder(DataFrameRecorder(includes=["K1"], raw_output=True))
    mc.add_child(RunOnce("run"))

    mc.add_random_variable({"K1"})

    mc.draws = 100
    s.run_drivers()
    df = rec.export_data()
    # Trick to get mean and std from scipy object
    assert df["K1"].mean() == pytest.approx(s.K1 + distribution._rv.kwds["loc"], abs=1e-3)
    assert df["K1"].std() == pytest.approx(distribution._rv.kwds["scale"], abs=1e-2)


def test_MonteCarlo_run_driver_perturbation_internal():

    s = MultiplySystem2("s")
    s.run_once()  # Initialize internal values
    initial_value = s.mult2.p_in.x
    distribution = Uniform(best=0.2, worst=-0.2)
    std = distribution._rv.kwds["scale"] / np.sqrt(12.0)
    s.mult2.p_in.get_details("x").distribution = distribution

    mc = s.add_driver(MonteCarlo("mc"))
    mc.add_child(RunOnce("run"))

    mc.add_random_variable({"mult2.p_in.x"})
    rec = mc.add_recorder(DataFrameRecorder(includes=["mult2.p_in.x"], raw_output=True))

    mc.draws = 1000
    s.run_drivers()
    df = rec.export_data()
    assert df["mult2.p_in.x"].mean() == pytest.approx(initial_value, abs=1e-3)
    assert df["mult2.p_in.x"].std() == pytest.approx(std, abs=2e-3)


def test_MonteCarlo_run_driver_perturbation_input():

    s = MultiplySystem2("s")
    distribution = Uniform(best=0.1, worst=-0.1)
    std = distribution._rv.kwds["scale"] / np.sqrt(12.0)
    s.mult1.p_in.get_details("x").distribution = distribution

    mc = s.add_driver(MonteCarlo("mc"))
    mc.add_recorder(
        DataFrameRecorder(includes=["mult1.p_in.x", "mult2.p_out.x"], raw_output=True)
    )
    mc.add_child(RunOnce("run"))
    mc.add_random_variable({"mult1.p_in.x"})
    mc.draws = 1000

    s.run_drivers()
    df = mc.recorder.export_data()
    assert df["mult1.p_in.x"].mean() == pytest.approx(s.mult1.p_in.x, abs=1e-3)
    assert df["mult1.p_in.x"].std() == pytest.approx(std, abs=1e-3)
    assert df["mult2.p_out.x"].std() != pytest.approx(0.0, abs=1e-2)


def test_MonteCarlo_run_driver_perturbation_combined():

    s = MultiplySystem2("s")
    s.run_once()
    init_mult1_p_out_x = s.mult1.p_out.x
    init_mult2_p_in_x = s.mult2.p_in.x
    distribution1 = Normal(best=0.1, worst=-0.1)
    std1 = distribution1._rv.kwds["scale"]
    s.mult1.p_in.get_details("x").distribution = distribution1
    distribution2 = Normal(best=0.2, worst=-0.2)
    s.mult2.p_in.get_details("x").distribution = distribution2
    std2 = distribution2._rv.kwds["scale"]

    mc = s.add_driver(MonteCarlo("mc"))
    mc.add_recorder(
        DataFrameRecorder(includes=["mult?.p_in.x", "mult?.p_out.x"], raw_output=True)
    )
    mc.add_child(RunOnce("run"))

    mc.add_random_variable({"mult1.p_in.x", "mult2.p_in.x"})

    mc.draws = 1000

    s.run_drivers()
    df = mc.recorder.export_data()

    assert df["mult1.p_in.x"].mean() == pytest.approx(s.mult1.p_in.x, abs=1e-3)
    assert df["mult1.p_in.x"].std() == pytest.approx(std1, abs=1e-2)
    assert df["mult1.p_out.x"].mean() == pytest.approx(init_mult1_p_out_x, abs=1e-3)
    assert df["mult1.p_out.x"].std() != pytest.approx(0, abs=1e-2)
    assert df["mult2.p_in.x"].mean() == pytest.approx(init_mult2_p_in_x, abs=1e-3)
    assert df["mult2.p_in.x"].std() >= std2
    assert df["mult2.p_out.x"].std() != pytest.approx(0, abs=1e-2)


def test_MonteCarlo_multipts_iterative_nonlinear():
    snl = IterativeNonLinear("nl")

    design = NonLinearSolver("design", method=NonLinearMethods.NR, factor=0.4)
    design = snl.add_driver(design)

    snl.splitter.inwards.split_ratio = 0.1
    snl.mult2.inwards.K1 = 1
    snl.mult2.inwards.K2 = 1
    snl.nonlinear.inwards.k1 = 1
    snl.nonlinear.inwards.k2 = 0.5

    run1 = design.add_child(RunSingleCase("run1"))
    run2 = design.add_child(RunSingleCase("run2"))

    run1.set_values({"p_in.x": 1})
    run1.design.add_unknown("nonlinear.inwards.k1").add_equation("splitter.p2_out.x == 10")

    run2.set_values({"p_in.x": 10})
    run2.design.add_unknown(
        ["mult2.inwards.K1", "nonlinear.inwards.k2", "splitter.inwards.split_ratio"]
    ).add_equation(
        ["splitter.p2_out.x == 50", "merger.p_out.x == 30", "splitter.p1_out.x == 5"]
    )

    mc = MonteCarlo("mc")
    rec = design.add_recorder(
        DataFrameRecorder(
            includes=["mult2.inwards.K1", "mult2.inwards.K2", "p_out.x"],
            raw_output=True,
            hold=True,
        )
    )
    snl.add_driver(mc)
    mc.add_driver(design)

    distribution = Uniform(best=0.1, worst=-0.1)
    std = distribution._rv.kwds["scale"] / np.sqrt(12)
    snl.mult2.inwards.get_details("K2").distribution = distribution

    mc.add_random_variable({"mult2.inwards.K2"})

    mc.draws = 64

    snl.run_drivers()

    df = rec.export_data()
    # Skip the initial execution
    stat = df.iloc[4:].groupby("Reference").describe()
    run1 = stat.loc["run1"]
    run2 = stat.loc["run2"]
    # Check the imposed distribution
    assert run1[("mult2.K2", "mean")] == pytest.approx(snl.mult2.K2, abs=5e-3)
    assert run1[("mult2.K2", "std")] == pytest.approx(std, abs=1e-2)
    assert run2[("mult2.K2", "mean")] == pytest.approx(snl.mult2.K2, abs=5e-3)
    assert run2[("mult2.K2", "std")] == pytest.approx(std, abs=1e-2)

    # snl.p_out.x == snl.spliter.p2_out.x => Always constraint
    assert run1[("p_out.x", "mean")] == pytest.approx(10, abs=1e-3)
    assert run1[("p_out.x", "std")] == pytest.approx(0, abs=1e-2)
    assert run2[("p_out.x", "mean")] == pytest.approx(50, abs=1e-3)
    assert run2[("p_out.x", "std")] == pytest.approx(0, abs=1e-2)

    # Check that the design variables are impacted
    assert run1[("mult2.K1", "std")] != pytest.approx(0, abs=1e-3)
    assert run2[("mult2.K1", "std")] != pytest.approx(0, abs=1e-3)


@pytest.mark.skip("TODO linearization mcs not support multipoint cases")
def test_MonteCarlo_multipts_iterative_nonlinear_linearized():
    snl = IterativeNonLinear("nl")

    design = NonLinearSolver("design", method=NonLinearMethods.NR)
    design = snl.add_driver(design)

    snl.splitter.split_ratio = 0.1
    snl.mult2.K1 = 1
    snl.mult2.K2 = 1
    snl.nonlinear.k1 = 1
    snl.nonlinear.k2 = 0.5

    run1 = design.add_child(RunSingleCase("run 1"))
    run2 = design.add_child(RunSingleCase("run 2"))

    run1.set_values({"p_in.x": 1})
    run1.design.add_unknown("nonlinear.inwards.k1").add_equation(
        "splitter.p2_out.x == 10"
    )

    run2.set_values({"p_in.x": 10})
    run2.design.add_unknown(
        ["mult2.inwards.K1", "nonlinear.inwards.k2", "splitter.inwards.split_ratio"]
    ).add_equation(
        ["splitter.p2_out.x == 50", "merger.p_out.x == 30", "splitter.p1_out.x == 5"]
    )

    mc = MonteCarlo("mc")
    rec = design.add_recorder(
        DataFrameRecorder(
            includes=["mult2.inwards.K1", "mult2.inwards.K2", "p_out.x"],
            raw_output=True,
            hold=True,
        )
    )

    snl.add_driver(mc)
    mc.add_driver(design)

    distribution = Uniform(best=0.2, worst=-0.2)
    std = distribution._rv.kwds["scale"] / np.sqrt(12)
    snl.mult2.inwards.get_details("K2").distribution = distribution
    mc.add_random_variable({"mult2.inwards.K2"})
    mc.add_response(["mult2.inwards.K1", "mult2.inwards.K2", "p_out.x"])

    run1.set_values({"p_in.x": 1})
    mc.draws = 1000
    mc.linear = True

    snl.run_drivers()

    df = rec.export_data()
    # Skip the initial execution
    stat = df.iloc[4:].groupby("Reference").describe()
    run1 = stat.loc["run 1"]
    run2 = stat.loc["run 2"]
    # Check the imposed distribution
    assert run1[("mult2.K2", "mean")] == pytest.approx(snl.mult2.K2, abs=5e-3)
    assert run1[("mult2.K2", "std")] == pytest.approx(std, abs=1e-2)
    assert run2[("mult2.K2", "mean")] == pytest.approx(snl.mult2.K2, abs=5e-3)
    assert run2[("mult2.K2", "std")] == pytest.approx(std, abs=1e-2)

    # snl.p_out.x == snl.spliter.p2_out.x => Always constraint
    assert run1[("p_out.x", "mean")] == pytest.approx(10, abs=1e-3)
    assert run1[("p_out.x", "std")] == pytest.approx(0, abs=1e-2)
    assert run2[("p_out.x", "mean")] == pytest.approx(50, abs=1e-3)
    assert run2[("p_out.x", "std")] == pytest.approx(0, abs=1e-2)

    # Check that the design variables are impacted
    assert run1[("mult2.K1", "std")] != pytest.approx(0, abs=1e-3)
    assert run2[("mult2.K1", "std")] != pytest.approx(0, abs=1e-3)


def test_MonteCarlo_embedded_solver():
    """
    Test case exposing bug reported in https://gitlab.com/cosapp/cosapp/-/issues/44

    Same as `test_MonteCarlo_run_driver_perturbation_combined`,
    with a NonLinearSolver child added before the RunOnce child.

    Note:
        Values are not tested, as the only behaviour tested is
        the absence of exception during `run_drivers`.
    """
    s = MultiplySystem2("s")
    s.run_once()
    s.mult1.p_in.get_details("x").distribution = Normal(best=0.1, worst=-0.1)
    s.mult2.p_in.get_details("x").distribution = Normal(best=0.2, worst=-0.2)

    mc = s.add_driver(MonteCarlo("mc"))
    mc.add_child(NonLinearSolver('solver'))
    mc.add_child(RunOnce("runonce"))  # last mc child is *not* a solver

    mc.add_random_variable({"mult1.p_in.x", "mult2.p_in.x"})
    mc.draws = 10

    with no_exception():
        s.run_drivers()


def test_MonteCarlo_with_time_driver():
    s = SystemTime("s")
    mc = s.add_driver(MonteCarlo("mc"))
    mc.add_random_variable("a")
    euler = mc.add_driver(EulerExplicit("euler", dt=0.1, time_interval=(0.0, 10.0)))
    euler.add_recorder(DataFrameRecorder(hold=True))
    mc.draws = 5

    s.run_drivers()
    results = euler.recorder.export_data()

    assert pytest.approx(results["m"].iloc[-1]) == 10.0
    assert pytest.approx(results["m"].iloc[101]) == 0.0
    assert len(results["Section"]) == 606


def test_MonteCarlo_with_event():
    s = SystemEvent("s")
    mc = s.add_driver(MonteCarlo("mc"))
    mc.add_random_variable("event_time")
    euler = mc.add_driver(EulerExplicit("euler", dt=1.0, time_interval=(0.0, 10.0)))
    euler.add_recorder(DataFrameRecorder(hold=True))
    mc.draws = 5

    s.run_drivers()
    results = euler.recorder.export_data()

    assert len(results) == 77
    assert results["time"].iloc[0] == 0.0
    assert results["time"].iloc[10] == 10.0
    assert  all(results["time"].iloc[13:16] == 2.0)


def _get_start_methods():
    if sys.platform == "win32":
        return (WorkerStartMethod.SPAWN, )

    return (WorkerStartMethod.FORK, WorkerStartMethod.SPAWN)

@pytest.mark.parametrize(argnames="nprocs", argvalues=[2, 4])    
@pytest.mark.parametrize("start_method", _get_start_methods())
def test_MonteCarlo_multiprocessing(nprocs, start_method):
    """Tests the execution of a MonteCarlo on multiple (sub)processes."""
    s = MultiplySystem2("s")
    s.run_once()
    s.mult1.p_in.get_details("x").distribution = Normal(best=0.1, worst=-0.1)
    s.mult2.p_in.get_details("x").distribution = Normal(best=0.2, worst=-0.2)

    mc = s.add_driver(MonteCarlo(
        "mc", 
        execution_policy=ExecutionPolicy(
            workers_count=nprocs,
            execution_type=ExecutionType.MULTI_PROCESSING,
            start_method=start_method
            )
        )
    )
    nls = mc.add_child(NonLinearSolver('solver'))
    mc.add_child(RunOnce("runonce"))  # last mc child is *not* a solver
    rec = nls.add_recorder(DataFrameRecorder("*x", hold=True))

    mc.add_random_variable({"mult1.p_in.x", "mult2.p_in.x"})
    mc.draws = 1 << 4

    with no_exception():
        s.run_drivers()

    assert len(rec.export_data()) == mc.draws + 1


def test_MonteCarlo_multiprocessing_with_time_driver():
    s = SystemTime("s")
    mc = s.add_driver(
        MonteCarlo(
            "mc",
            execution_policy=ExecutionPolicy(
                workers_count=4,
                execution_type=ExecutionType.MULTI_PROCESSING,
            ),
        )
    )
    mc.add_random_variable("a")
    euler = mc.add_driver(EulerExplicit("euler", dt=1.0, time_interval=(0.0, 10.0)))
    euler.add_recorder(DataFrameRecorder(hold=True))
    mc.draws = 5

    s.run_drivers()
    results = euler.recorder.export_data()

    assert pytest.approx(results["m"].iloc[-1]) == 10.0
    assert pytest.approx(results["m"].iloc[11]) == 0.0
    assert len(results["Section"]) == 66


def test_MonteCarlo_with_event_parallel():
    s = SystemEvent("s")
    mc = s.add_driver(
        MonteCarlo(
            "mc",
            execution_policy=ExecutionPolicy(
                workers_count=4,
                execution_type=ExecutionType.MULTI_PROCESSING,
            ),
        )
    )
    mc.add_random_variable("event_time")
    euler = mc.add_driver(EulerExplicit("euler", dt=1.0, time_interval=(0.0, 10.0)))
    euler.add_recorder(DataFrameRecorder(hold=True))
    mc.draws = 5

    s.run_drivers()
    results = euler.recorder.export_data()

    assert len(results) == 77
    assert results["time"].iloc[0] == 0.0
    assert results["time"].iloc[10] == 10.0
    assert  all(results["time"].iloc[13:16] == 2.0)

class TestMonteCarloPickling:
  
    @pytest.fixture
    def system(self):
        s: System = Multiply2('s')
        s.add_driver(MonteCarlo('mc'))
        return s

    def test_standalone(self):
        """Test pickling of standalone driver."""

        mc = MonteCarlo('mc')

        mc_copy = pickle_roundtrip(mc)
        assert are_same(mc, mc_copy)

    def test_default(self, system):
        """Test driver with default options."""

        system_copy = pickle_roundtrip(system)
        assert are_same(system, system_copy)

    def test_random_variables(self, system):
        """Test driver with input vars."""

        mc = system.drivers["mc"]
        distribution = Uniform(best=0.2, worst=-0.2)
        system.p_in.get_details("x").distribution = distribution
        mc.add_random_variable({"p_in.x"})

        system_copy = pickle_roundtrip(system)
        assert are_same(system, system_copy)

        mc_copy = system_copy.drivers["mc"]
        assert has_keys(mc_copy.random_variables, "p_in.x")
        random_var = mc_copy.random_variables["p_in.x"]
        assert are_same(random_var.distribution, distribution)
        assert are_same(system_copy.p_in.get_details("x").distribution, distribution)


    def test_responses(self, system):
        """Test driver with response vars."""

        mc = system.drivers["mc"]
        mc.add_response("K1")
        mc.add_response("K2")

        system_copy = pickle_roundtrip(system)
        assert are_same(system, system_copy)

        mc_copy = system_copy.drivers["mc"]
        assert set(mc_copy.responses) == {"K1", "K2"}


    def test_recorder(self, system):
        """Test pickling of driver with recorder."""

        mc = system.drivers["mc"]
        mc.add_recorder(DataFrameRecorder(hold=True, includes="a*"))
        assert are_same(system, pickle_roundtrip(system))

        system_copy = pickle_roundtrip(system)
        mc_copy = system_copy.drivers["mc"]
        assert mc_copy.recorder.hold
        assert mc_copy.recorder.includes == ["a*"]
        assert mc_copy.recorder._owner is mc_copy
        assert mc_copy.recorder._watch_object is system_copy

    def test_execution(self, system):
        """Test execution of pickled driver."""

        mc = system.drivers["mc"]
        mc.add_child(RunOnce("run"))
        mc.add_recorder(DataFrameRecorder(includes=["K?", "p_in.x"], raw_output=True))

        distribution = Uniform(best=0.2, worst=-0.2)
        system.p_in.get_details("x").distribution = distribution
        mc.add_random_variable({"p_in.x"})
        mc.draws = 10
        system.run_drivers()

        system_copy = pickle_roundtrip(system)
        mc_copy = system_copy.drivers["mc"]
        df = mc_copy.recorder.export_data()
        assert len(df) == 10
        assert df.iloc[4]["K1"] == 5.0
        assert df.iloc[4]["K2"] == 5.0
        assert 0.0 < df.iloc[4]["p_in.x"] < 2.0
        assert df.iloc[8]["K1"] == 5.0
        assert df.iloc[8]["K2"] == 5.0
        assert 0.0 < df.iloc[8]["p_in.x"] < 2.0
