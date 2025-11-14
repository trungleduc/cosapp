import numpy as np
import pandas as pd
import pytest

from cosapp.systems import System
from cosapp.drivers import RunSingleCase, NonLinearSolver, NonLinearMethods, Influence
from cosapp.tests.library.systems import Multiply2, IterativeNonLinear


def test_Influence_reset_input_vars():
    s = Multiply2("MyMult")
    influence = s.add_driver(Influence("influence", verbose=True))
    influence.add_input_vars({"K1", "K2"})
    assert set(influence.input_vars) == {"K1", "K2"}

    influence.reset_input_vars()
    assert influence.input_vars == ["*"]


def test_Influence_reset_response_vars():
    s = Multiply2("MyMult")
    influence = s.add_driver(Influence("influence", verbose=True))
    influence.add_response_vars({"p_out.x", "Ksum"})
    assert set(influence.response_vars) == {"p_out.x", "Ksum"}

    influence.reset_response_vars()
    assert influence.response_vars == ["*"]


def test_Influence_add_input_vars():
    s = Multiply2("MyMult")
    influence = s.add_driver(Influence("influence", verbose=True))
    assert influence.input_vars == ["*"]

    influence.add_input_vars("K1")
    assert influence.input_vars == ["K1"]

    influence.reset_input_vars()
    influence.add_input_vars({"K1", "K2"})
    assert set(influence.input_vars) == {"K1", "K2"}

    with pytest.raises(TypeError):
        influence.add_input_vars(1.0)
    with pytest.raises(TypeError):
        influence.add_input_vars({"a": 1.0})
    with pytest.raises(TypeError):
        influence.add_input_vars({1.0})
    with pytest.raises(TypeError):
        influence.add_input_vars([1.0])


def test_Influence_add_response_vars():
    s = Multiply2("MyMult")
    influence = s.add_driver(Influence("influence", verbose=True))
    assert influence.response_vars == ["*"]

    influence.add_response_vars("K1")
    assert influence.response_vars == ["K1"]

    influence.reset_response_vars()
    influence.add_response_vars({"K1", "K2"})
    assert set(influence.response_vars) == {"K1", "K2"}

    influence.reset_response_vars()
    influence.add_response_vars(["K2"])
    assert influence.response_vars == ["K2"]

    with pytest.raises(TypeError):
        influence.add_response_vars(1.0)
    with pytest.raises(TypeError):
        influence.add_response_vars({"a": 1.0})
    with pytest.raises(TypeError):
        influence.add_response_vars({1.0})
    with pytest.raises(TypeError):
        influence.add_response_vars([1.0])


def test_Influence__build_cases():
    s = Multiply2("MyMult")
    s.p_in.x = 1.0

    influence = s.add_driver(Influence("influence", verbose=True))
    solver = influence.add_driver(
        NonLinearSolver("solver", factor=1.0, method=NonLinearMethods.NR)
    )
    solver.add_driver(RunSingleCase("run"))

    influence._build_cases()
    assert len(influence.cases) == 4
    assert set(influence.found_input_vars) == {"K1", "K2", "p_in.x"}
    assert set(influence.found_response_vars) == {"Ksum", "p_out.x"}

    influence.reset_input_vars()
    influence.reset_response_vars()
    influence.add_input_vars("K1")
    influence.add_response_vars("K*")
    influence._build_cases()
    assert len(influence.cases) == 2
    assert influence.found_input_vars == ["K1"]
    assert influence.found_response_vars == ["Ksum"]

    influence.reset_input_vars()
    influence.reset_response_vars()
    influence.add_input_vars("K?")
    influence.add_response_vars("*.x")
    influence._build_cases()
    assert len(influence.cases) == 3
    assert set(influence.found_input_vars) == {"K1", "K2"}
    assert influence.found_response_vars == ["p_out.x"]


def test_Influence_show_influence_matrix():
    s = Multiply2("MyMult")
    influence = s.add_driver(Influence("influence", verbose=True))
    influence.add_driver(
        NonLinearSolver("solver", factor=1.0, method=NonLinearMethods.NR)
    )

    s.run_drivers()

    assert isinstance(influence.show_influence_matrix(), pd.DataFrame)
    assert isinstance(
        influence.show_influence_matrix(styling=True), pd.io.formats.style.Styler
    )

    assert np.allclose(
        influence.show_influence_matrix(cleaned=True).values,
        [[0.5, 1.0], [0.5, 1.0], [0.0, 1.0]],
    )

    influence.influence_min_threshold = 0.7
    assert np.allclose(
        influence.show_influence_matrix(cleaned=True).values, [[1.0], [1.0], [1.0]]
    )

    influence.influence_min_threshold = 1.2
    assert influence.show_influence_matrix(cleaned=True).values.size == 0


def test_Influence__precase():
    s = Multiply2("MyMult")
    influence = s.add_driver(Influence("influence", verbose=True))
    s.K1 = 10.0
    s.K2 = 10.0
    influence._build_cases()

    influence._precase(2, influence.cases[2])
    assert 10.0 == s.K1
    assert 10.0 * (1 + influence.delta) == s.K2

    influence._precase(2, influence.cases[1])
    assert 10.0 * (1 + influence.delta) == s.K1
    assert 10.0 == s.K2


def test_Influence__run_reference():
    s = Multiply2("MyMult")
    influence = s.add_driver(Influence("influence", verbose=True))
    s.p_in.x = 1.0
    s.K1 = 10.0
    s.K2 = 10.0
    influence._build_cases()
    assert influence.reference.values.size == 0

    influence._run_reference()
    assert np.array_equal(influence.reference.values, [[10.0, 10.0, 0.0, 1.0, 1.0]])


class ZeroDivisionSystem(System):
    def setup(self):
        self.add_inward("a", 1.0)
        self.add_inward("b", 0.0)
        self.add_outward("res", 1.0)
        self.add_outward("fake", 0.0)
        self.add_outward("boolvar", False)

    def compute(self):
        self.res = self.a + self.b


def test_integration_Influence_singlept1():
    s = Multiply2("MyMult")

    s.p_in.x = 1.0
    s.K1 = 11.0
    s.K2 = 10.0

    influence = s.add_driver(Influence("influence", verbose=True))
    solver = influence.add_driver(
        NonLinearSolver("solver", factor=1.0, method=NonLinearMethods.NR)
    )
    solver.add_unknown("K1").add_equation("K1 == K2")

    s.run_drivers()

    assert len(influence.cases) == 3
    assert influence.influence_matrix["p_out.x"]["K2"] == pytest.approx(2.0, abs=1.0e-2)
    assert influence.influence_matrix.shape == (3, 2)


def test_integration_Influence_nonlinear():
    snl = IterativeNonLinear("nl")
    design = snl.add_driver(NonLinearSolver("design", method=NonLinearMethods.NR))

    snl.splitter.split_ratio = 0.1
    snl.mult2.K1 = 1.0
    snl.mult2.K2 = 1.0
    snl.nonlinear.k1 = 1.0
    snl.nonlinear.k2 = 0.5

    run1 = design.add_child(RunSingleCase("run1"))

    run1.set_values({"p_in.x": 1.0})
    run1.add_unknown("nonlinear.k1").add_equation("splitter.p2_out.x == 10")

    influence = snl.add_driver(Influence("influence", verbose=True))
    influence.add_driver(design)

    snl.run_drivers()
    assert influence.influence_matrix.shape == (11, 7)


def test_integration_Influence_zerodivisionerror():
    s = ZeroDivisionSystem("s")

    design = s.add_driver(NonLinearSolver("design", method=NonLinearMethods.NR))
    design.add_child(RunSingleCase("run"))

    influence = s.add_driver(Influence("influence", verbose=True))
    influence.add_driver(design)

    s.run_drivers()
    assert influence.influence_matrix.shape == (2, 2)
