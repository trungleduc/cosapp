import pytest
import numpy as np

from cosapp.utils.distributions import Normal, Uniform
from cosapp.drivers import (
    MonteCarlo,
    NonLinearMethods,
    NonLinearSolver,
    RunOnce,
    RunSingleCase,
)
from cosapp.recorders import DataFrameRecorder
from cosapp.systems import System
from cosapp.tests.library.systems import (
    IterativeNonLinear,
    Multiply2,
    Multiply4,
    MultiplySystem2,
)
from cosapp.utils.testing import no_exception


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
    assert mc.random_variables["K1"] == (
        s.name2variable["K1"],
        None,
        s.inwards.get_details("K1").distribution,
    )
    assert mc.random_variables["K2"] == (
        s.name2variable["K2"],
        None,
        s.inwards.get_details("K2").distribution,
    )

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
    assert mc.random_variables["mult.p_in.x"] == (
        t.name2variable["mult.p_in.x"],
        connector,
        dummy,
    )


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
    mc._precompute()
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

    mc._precompute()
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

    mc._precompute()
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

    mc._precompute()
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

    mc._precompute()
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


# @pytest.mark.skip("Incorrect behaviour - remove test altogether")
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

    mc.draws = 50

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


@pytest.mark.skip("TODO linearization does not support multipoint cases")
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
