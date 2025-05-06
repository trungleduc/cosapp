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
    EulerExplicit,
    Driver,
)
from cosapp.recorders import DataFrameRecorder
from cosapp.systems import System
from cosapp.tests.library.systems import (
    IterativeNonLinear,
    Multiply2,
    Multiply4,
    MultiplySystem2,
)
from cosapp.core.execution import ExecutionPolicy, ExecutionType, WorkerStartMethod
from cosapp.utils.testing import no_exception, pickle_roundtrip, are_same


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
    assert len(mc.response_varnames) == 0
    assert mc.cases is None


def test_MonteCarlo_add_child():
    mc = MonteCarlo("mc")
    solver = mc.add_child(NonLinearSolver("solver"))
    driver = mc.add_child(Driver("driver"))
    assert mc.solver is solver
    assert mc.driver is driver


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
    mult = Multiply2("mult")
    mult.K1 = 5.0
    mult.K2 = 10.0

    mc = mult.add_driver(MonteCarlo("mc"))

    mc.add_random_variable(["K1", "K2"])
    assert set(mc.random_variable_names) == {"K1", "K2"}
    
    for random_variable in mc.random_variables:
        name = random_variable.name
        assert random_variable._ref is mult.name2variable[name]
        assert random_variable.connector is None
        assert random_variable.distribution is mult.get_variable(name).distribution

    mc.clear_random_variables()
    assert len(mc.random_variable_names) == 0
    mc.add_random_variable(["K1"])
    assert set(mc.random_variable_names) == {"K1"}

    mc.clear_random_variables()
    assert len(mc.random_variable_names) == 0
    mc.add_random_variable("K1")
    assert set(mc.random_variable_names) == {"K1"}

    # Don't duplicate input
    mc.add_random_variable("K1")
    assert set(mc.random_variable_names) == {"K1"}

    # Protection
    with pytest.raises(TypeError, match="Variable name"):
        mc.add_random_variable(1.0)
    
    with pytest.raises(TypeError, match=r"'p_out.x' is not an input variable"):
        mc.add_random_variable("p_out.x")

    with pytest.raises(AttributeError, match=r"'[\w\.]+' not found in System '[\w\.]+'"):
        mc.add_random_variable("x")

    with pytest.raises(TypeError, match=r"'[\w\.]+' is not a variable of '[\w\.]+'\."):
        mc.add_random_variable("p_in")

    with no_exception():
        mc.add_random_variable("p_in.x")

    with pytest.raises(ValueError, match=r"No distribution was specified for [\w\.]+\.\w+"):
        mc.setup_run()


def test_MonteCarlo_add_random_variable_connected():
    class Assembly(System):
        """Bogus assembly"""
        def setup(self):
            m1 = self.add_child(Multiply2("m1"), pulling="p_in")
            m2 = self.add_child(Multiply2("m2"), pulling="p_out")

            self.connect(m1.p_out, m2.p_in)
    
    top = Assembly("top")
    mc = top.add_driver(MonteCarlo("mc"))

    top.get_variable("m2.p_in.x").distribution = distribution = Normal(best=2.0, worst=0.0)

    assert len(mc.random_variable_names) == 0

    mc.add_random_variable("m2.p_in.x")
    assert set(mc.random_variable_names) == {"m2.p_in.x"}
    assert mc.random_variable_data() == {"m2.p_in.x": distribution}

    random_variables = list(mc.random_variables)
    connector = list(filter(lambda c: c.sink is top.m2.p_in, top.all_connectors()))[0]

    assert len(random_variables) == 1
    random_variable = random_variables[0]
    assert random_variable.name == "m2.p_in.x"
    assert random_variable._ref is top.name2variable["m2.p_in.x"]
    assert random_variable.connector is connector
    assert random_variable.distribution is distribution


def test_MonteCarlo_add_random_variable_pulled():
    top = System("top")
    top.add_child(Multiply2("mult"), pulling="p_in")
    top.get_variable("mult.p_in.x").distribution = distribution = Normal(best=2.0, worst=0.0)

    mc = top.add_driver(MonteCarlo("mc"))

    mc.add_random_variable("mult.p_in.x")
    assert set(mc.random_variable_names) == {"mult.p_in.x"}
    assert mc.random_variable_data() == {"mult.p_in.x": distribution}

    random_variables = list(mc.random_variables)
    assert len(random_variables) == 1
    random_variable = random_variables[0]
    assert random_variable.name == "mult.p_in.x"
    assert random_variable._ref is top.name2variable["p_in.x"]  # p_in.x is the free variable, here
    assert random_variable.connector is None  # due to pulling
    assert random_variable.distribution is distribution


def test_MonteCarlo_add_random_variable_with_distribution_1():
    """Test method `MonteCarlo_add_random_variable`.
    Case 1: one variables and one distribution at a time.
    """
    s = Multiply2("s")
    mc = s.add_driver(MonteCarlo("mc"))

    normal = Normal(best=-1.0, worst=3.0)
    uniform = Uniform(worst=-0.1, best=0.1)
    
    mc.add_random_variable("K1", normal)
    assert mc.random_variable_data() == {"K1": normal}
    
    mc.add_random_variable("K2", uniform)
    assert mc.random_variable_data() == {"K1": normal, "K2": uniform}


@pytest.mark.parametrize("collection", [tuple, list, set])
def test_MonteCarlo_add_random_variable_with_distribution_2(collection):
    """Test method `MonteCarlo_add_random_variable`.
    Case 2: several variables and a single distribution.
    """
    s = Multiply2("s")
    mc = s.add_driver(MonteCarlo("mc"))

    normal = Normal(best=-1.0, worst=3.0)
    
    mc.add_random_variable(collection(["K1", "K2"]), normal)
    assert mc.random_variable_data() == {"K1": normal, "K2": normal}


def test_MonteCarlo_add_random_variable_with_distribution_3():
    """Test method `MonteCarlo_add_random_variable`.
    Case 3: several variables with associated distributions.
    """
    s = Multiply2("s")
    mc = s.add_driver(MonteCarlo("mc"))

    normal = Normal(best=-1.0, worst=3.0)
    uniform = Uniform(worst=-0.1, best=0.1)
    
    mc.add_random_variable({"K1": normal, "K2": uniform})
    assert mc.random_variable_data() == {"K1": normal, "K2": uniform}
    
    # Add existing varname with a different distribution
    mc.add_random_variable({"K2": normal})
    assert mc.random_variable_data() == {"K1": normal, "K2": normal}

    # Clear and check erroneous case
    s.drivers.clear()
    mc = s.add_driver(MonteCarlo("mc"))
    
    with pytest.raises(TypeError, match=r"Distribution for 's\.K2'"):
        mc.add_random_variable({"K1": normal, "K2": 3.14})

    # assert list(var_data) == ["K1"]

    # Clear and check erroneous case
    s.drivers.clear()
    mc = s.add_driver(MonteCarlo("mc"))
    
    with pytest.raises(TypeError, match=r"Distribution for 's\.K1'"):
        mc.add_random_variable({"K1": 3.14, "K2": normal})

    # assert list(var_data) == []


def test_MonteCarlo_add_response():
    s = Multiply2("mult")
    mc = s.add_driver(MonteCarlo("mc"))

    mc.add_response("K1")
    assert "K1" in mc.response_varnames
    assert set(mc.response_varnames) == {"K1"}

    mc.add_response("K1")
    assert set(mc.response_varnames) == {"K1"}

    mc.add_response("K2")
    assert set(mc.response_varnames) == {"K1", "K2"}

    mc.response_varnames.clear()
    mc.add_response(["K1", "K2"])
    assert mc.response_varnames == ["K1", "K2"]
    assert set(mc.response_varnames) == {"K1", "K2"}

    mc.response_varnames.clear()
    mc.add_response({"K1", "K2"})
    assert set(mc.response_varnames) == {"K1", "K2"}

    mc.response_varnames.clear()
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
    s.get_variable("p_in.x").distribution = Normal(best=-1.0, worst=-3.0)
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
    s.get_variable("p_in.x").distribution = Normal(best=-1.0, worst=-3.0)
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
    mc = s.add_driver(MonteCarlo("mc"))
    rec = mc.add_recorder(DataFrameRecorder(includes=["K1"], raw_output=True))
    mc.add_child(RunOnce("run"))

    distribution = Normal(best=0.1, worst=-0.1)
    mc.add_random_variable("K1", distribution)

    mc.draws = 100
    s.run_drivers()
    df = rec.export_data()
    assert df["K1"].mean() == pytest.approx(s.K1, abs=1e-3)
    # Trick to get std from scipy object
    assert df["K1"].std() == pytest.approx(distribution._rv.kwds["scale"], abs=1e-2)


def test_MonteCarlo_cases_uncentered():

    s = SimpleCentered("s")
    mc = s.add_driver(MonteCarlo("mc"))
    rec = mc.add_recorder(DataFrameRecorder(includes=["K1"], raw_output=True))
    mc.add_child(RunOnce("run"))

    distribution = Normal(best=0.0, worst=-0.4)
    mc.add_random_variable("K1", distribution)

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
    s.get_variable("mult2.p_in.x").distribution = distribution

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
    s.get_variable("mult1.p_in.x").distribution = distribution

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
    s.get_variable("mult1.p_in.x").distribution = distribution1 = Normal(best=0.1, worst=-0.1)
    s.get_variable("mult2.p_in.x").distribution = distribution2 = Normal(best=0.2, worst=-0.2)
    std1 = distribution1._rv.kwds["scale"]
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

    snl.splitter.split_ratio = 0.1
    snl.mult2.K1 = 1
    snl.mult2.K2 = 1
    snl.nonlinear.k1 = 1
    snl.nonlinear.k2 = 0.5

    run1 = design.add_child(RunSingleCase("run1"))
    run2 = design.add_child(RunSingleCase("run2"))

    run1.set_values({"p_in.x": 1})
    run1.design.add_unknown("nonlinear.k1").add_equation("splitter.p2_out.x == 10")

    run2.set_values({"p_in.x": 10})
    run2.design.add_unknown(
        ["mult2.K1", "nonlinear.k2", "splitter.split_ratio"]
    ).add_equation(
        ["splitter.p2_out.x == 50", "merger.p_out.x == 30", "splitter.p1_out.x == 5"]
    )

    mc = MonteCarlo("mc")
    rec = design.add_recorder(
        DataFrameRecorder(
            includes=["mult2.K1", "mult2.K2", "p_out.x"],
            raw_output=True,
            hold=True,
        )
    )
    snl.add_driver(mc)
    mc.add_driver(design)

    distribution = Uniform(best=0.1, worst=-0.1)
    std = distribution._rv.kwds["scale"] / np.sqrt(12)

    mc.add_random_variable("mult2.K2", distribution)
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

    design = snl.add_driver(NonLinearSolver("design"))

    snl.splitter.split_ratio = 0.1
    snl.mult2.K1 = 1
    snl.mult2.K2 = 1
    snl.nonlinear.k1 = 1
    snl.nonlinear.k2 = 0.5

    run1 = design.add_child(RunSingleCase("run 1"))
    run2 = design.add_child(RunSingleCase("run 2"))

    design.add_unknown([
        "mult2.K1",
        "nonlinear.k1",
        "nonlinear.k2",
        "splitter.split_ratio",
    ])

    run1.set_values({"p_in.x": 1.0})
    run1.add_equation("splitter.p2_out.x == 10")

    run2.set_values({"p_in.x": 10.0})
    run2.design.add_equation([""
        "merger.p_out.x == 30",
        "splitter.p1_out.x == 5",
        "splitter.p2_out.x == 50",
    ])

    mc = MonteCarlo("mc")
    rec = design.add_recorder(
        DataFrameRecorder(
            includes=["mult2.K?", "p_out.x"],
            raw_output=True,
            hold=True,
        )
    )

    snl.add_driver(mc)
    mc.add_driver(design)

    distribution = Uniform(best=0.2, worst=-0.2)
    std = distribution._rv.kwds["scale"] / np.sqrt(12)
    mc.add_random_variable("mult2.K2", distribution)
    mc.add_response(["mult2.K1", "mult2.K2", "p_out.x"])

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

    mc = s.add_driver(MonteCarlo("mc"))
    solver = mc.add_child(NonLinearSolver("solver"))
    mc.add_child(RunOnce("runonce"))  # last mc child is *not* a solver

    mc.add_random_variable({
        "mult1.p_in.x": Normal(best=0.1, worst=-0.1),
        "mult2.p_in.x": Normal(best=0.2, worst=-0.2),
    })
    mc.draws = 10

    with no_exception():
        s.run_drivers()

    assert mc._solver is solver


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


def test_MonteCarlo_with_time_driver_subsystem():
    """Same as `test_MonteCarlo_with_time_driver`, with `SystemTime` encapsulated in a top system"""
    s = System("s")
    s.add_child(SystemTime("sub"))
    mc = s.add_driver(MonteCarlo("mc"))
    mc.add_random_variable("sub.a")
    euler = mc.add_driver(EulerExplicit("euler", dt=0.1, time_interval=(0.0, 10.0)))
    euler.add_recorder(DataFrameRecorder(hold=True))
    mc.draws = 5

    s.run_drivers()
    results = euler.recorder.export_data()

    assert pytest.approx(results["sub.m"].iloc[-1]) == 10.0
    assert pytest.approx(results["sub.m"].iloc[101]) == 0.0
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
        return (WorkerStartMethod.SPAWN,)
    else:
        return (WorkerStartMethod.FORK, WorkerStartMethod.SPAWN)


@pytest.mark.parametrize(argnames="nprocs", argvalues=[2, 4])    
@pytest.mark.parametrize("start_method", _get_start_methods())
def test_MonteCarlo_multiprocessing(nprocs, start_method):
    """Tests the execution of a MonteCarlo on multiple (sub)processes."""
    s = MultiplySystem2("s")
    s.run_once()
    s.get_variable("mult1.p_in.x").distribution = Normal(best=0.1, worst=-0.1)
    s.get_variable("mult2.p_in.x").distribution = Normal(best=0.2, worst=-0.2)

    mc = s.add_driver(
        MonteCarlo(
            "mc", 
            execution_policy=ExecutionPolicy(
                workers_count=nprocs,
                execution_type=ExecutionType.MULTI_PROCESSING,
                start_method=start_method
            ),
        ),
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
        """Test driver with default options.
        """
        system_copy = pickle_roundtrip(system)
        assert are_same(system, system_copy)

    def test_random_variables(self, system):
        """Test driver with input vars.
        """
        mc: MonteCarlo = system.drivers["mc"]
        distribution = Uniform(best=0.2, worst=-0.2)
        mc.add_random_variable("p_in.x", distribution)

        system_copy = pickle_roundtrip(system)
        assert are_same(system, system_copy)

        mc_copy = system_copy.drivers["mc"]
        assert isinstance(mc_copy, MonteCarlo)
        assert set(mc_copy.random_variable_names) == {"p_in.x"}
        random_variables = list(mc_copy.random_variables)
        random_variable = random_variables[0]
        assert random_variable.name == "p_in.x"
        assert are_same(random_variable.distribution, distribution)

    def test_responses(self, system):
        """Test driver with response vars.
        """
        mc: MonteCarlo = system.drivers["mc"]
        mc.add_response("K1")
        mc.add_response("K2")

        system_copy = pickle_roundtrip(system)
        assert are_same(system, system_copy)

        mc_copy = system_copy.drivers["mc"]
        assert isinstance(mc_copy, MonteCarlo)
        assert set(mc_copy.response_varnames) == {"K1", "K2"}

    def test_recorder(self, system):
        """Test pickling of driver with recorder."""

        mc: MonteCarlo = system.drivers["mc"]
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

        mc: MonteCarlo = system.drivers["mc"]
        mc.add_child(RunOnce("run"))
        mc.add_recorder(DataFrameRecorder(includes=["K?", "p_in.x"], raw_output=True))

        system.get_variable("p_in.x").distribution = Uniform(best=0.2, worst=-0.2)
        mc.add_random_variable("p_in.x")
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
