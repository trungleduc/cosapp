import pytest
import numpy as np

from cosapp.systems import System
from cosapp.drivers import RunSingleCase
from cosapp.core.numerics.basics import MathematicalProblem
from cosapp.core.numerics.boundary import Boundary
from cosapp.core.numerics.residues import Residue
from cosapp.utils.testing import assert_keys, assert_all_type


# TODO unit tests for vectors
# Test with a vector variable and a partial set vector variable
#   test for 1d and 2d vectors
# - set_values
# - set_init
# - design equations
# - local equations

# <codecell>

def check_problem(problem: MathematicalProblem, n_unknowns, n_equations):
    """Utility function used in tests below"""
    assert isinstance(problem, MathematicalProblem)
    assert problem.n_unknowns == n_unknowns
    assert problem.n_equations == n_equations

# <codecell>

def test_RunSingleCase_setup():
    d = RunSingleCase("case", System('s'))
    assert len(d.children) == 0
    assert isinstance(d.case_values, list)
    assert len(d.case_values) == 0
    assert isinstance(d.initial_values, dict)
    assert len(d.initial_values) == 0
    assert d.problem is None
    assert d.design.is_empty()
    assert d.offdesign.is_empty()


def test_RunSingleCase_setup_run(DummyFactory):
    def Dummy(name):
        return DummyFactory(name, unknown=["p_in.x", "K2"])

    mult = Dummy("mult")
    d = RunSingleCase("case")
    mult.add_driver(d)
    assert d.owner is mult

    mult.K1 = 12.0
    mult.K2 = 42.0
    assert mult.K1 == 12
    
    d.add_value("K1", 15.0)
    assert d.problem is None
    d.setup_run()
    check_problem(d.problem, 2, 0)
    assert set(d.problem.unknowns) == {
        "K2", "p_in.x"
    }
    for name, unknown in d.problem.unknowns.items():
        assert unknown.name == name
        assert unknown.context is mult

    # Test with subsystems
    s = System("top")
    s.add_child(Dummy("mult"))
    d = RunSingleCase("case")
    s.add_driver(d)

    s.mult.K1 = 12.0
    s.mult.K2 = 42.0

    d.add_value("mult.K1", 15.0)
    assert d.problem is None
    d.setup_run()
    check_problem(d.problem, 2, 0)
    assert set(d.problem.unknowns) == set(
        f"mult.{name}"
        for name in ("K2", "p_in.x")
    )
    for name in ("K2", "p_in.x"):
        key = f"mult.{name}"
        unknown = d.problem.unknowns[key]
        assert unknown.name == name
        assert unknown.context is s.mult


def test_RunSingleCase__precompute_boundary_cdts(DummyFactory):
    def Dummy(name):
        return DummyFactory(name, unknown="K2")

    # Test with top system
    mult = Dummy("mult")
    d = RunSingleCase("case")
    mult.add_driver(d)

    mult.K1 = 12.0
    mult.K2 = 42.0
    assert mult.K1 == 12
    
    d.add_value("K1", 15.0)
    d.setup_run()
    d._precompute()
    assert mult.K1 == 15

    # Test with subsystems
    s = System("compute")
    s.add_child(Dummy("mult"))
    d = RunSingleCase("case")
    s.add_driver(d)

    s.mult.K1 = 12.0
    s.mult.K2 = 42.0
    assert s.mult.K1 == 12

    d.add_value("mult.K1", 15.0)
    d.setup_run()
    d._precompute()

    assert s.mult.K1 == 15


def test_RunSingleCase__precompute_equations(DummyFactory):
    def Dummy(name):
        return DummyFactory(name, unknown=["K1"])

    s = System("compute")
    s.add_child(Dummy("mult"))
    d = RunSingleCase("case")
    s.add_driver(d)

    d.add_equation("mult.p_out.x == 50")

    d.setup_run()
    d._precompute()
    check_problem(d.problem, 1, 1)
    assert set(d.problem.unknowns) == {"mult.K1"}


def test_RunSingleCase_set_values(DummyFactory, hat_case):
    def Dummy(name):
        return DummyFactory(name, unknown=["K2"])

    s = Dummy("mult")
    d = RunSingleCase("case")
    s.add_driver(d)

    d.add_value("K1", 11.5)
    s.run_drivers()
    assert s.K1 == 11.5

    with pytest.raises(TypeError, match="Only variables can be used in mathematical algorithms"):
        d.add_value("inwards", 10.0)

    d.add_value("K1", 9.5)
    s.run_drivers()
    assert s.K1 == 9.5

    with pytest.raises(TypeError):
        d.add_value(s.K1, 10.0)

    with pytest.raises(AttributeError):
        d.add_value("C", 10.0)

    d = RunSingleCase("case")

    with pytest.raises(AttributeError, match="Driver '\w+' must be attached to a System to set case values."):
        d.add_value("K1", 11.5)

    with pytest.raises(AttributeError, match="Driver '\w+' must be attached to a System to set case values."):
        d.add_values({"K1": 11.5})

    # Test vector variables
    s, case = hat_case(RunSingleCase)
    case.add_value("in_.x", np.r_[-1.0, -2.0, -3.0])
    s.run_drivers()
    assert np.allclose(s.in_.x, [-1, -2, -3], atol=0)

    s, case = hat_case(RunSingleCase)
    case.add_value("in_.x[0]", 42.0)
    s.run_drivers()
    assert np.allclose(s.in_.x, [42, -2, -3], atol=0)

    s, case = hat_case(RunSingleCase)
    case.add_value("in_.x[1:]", 24.0)
    s.run_drivers()
    assert np.allclose(s.in_.x, [42, 24, 24], atol=0)

    s, case = hat_case(RunSingleCase)
    s.in_.x = np.zeros(3)
    case.add_values({"in_.x[0]": 22.0})
    case.add_values({"in_.x[1:]": 33.0})
    s.run_drivers()
    assert np.allclose(s.in_.x, [22, 33, 33], atol=0)

    s, case = hat_case(RunSingleCase)
    s.in_.x = np.zeros(3)
    case.add_values({"in_.x[0]": 22.0})
    case.set_values({"in_.x[1:]": 33.0})  # purges case values
    s.run_drivers()
    assert np.allclose(s.in_.x, [0, 33, 33], atol=0)


def test_RunSingleCase_run_once(DummyFactory):
    def make_case():
        s = System("compute")
        s.add_child(DummyFactory("mult"))
        s.exec_order = ["mult"]
        d = RunSingleCase("case")
        s.add_driver(d)
        return s, d

    s, d = make_case()
    s.mult.p_in.x = 1.0
    s.mult.K1 = 5.0
    s.mult.K2 = 2.0
    s.mult.p_out.x = 1.0

    d.offdesign.add_unknown("mult.inwards.K1").add_equation("mult.p_out.x == 50")

    assert s.mult.p_out.x == 1

    d.setup_run()
    d.run_once()
    assert s.mult.p_out.x == 10

    s, d = make_case()
    s.mult.p_in.x = 1.0
    s.mult.K1 = 5.0
    s.mult.K2 = 2.0
    s.mult.p_out.x = 1.0
    d.add_value("mult.K1", 10.0)
    d.setup_run()
    d.run_once()
    assert s.mult.p_out.x == 20


def test_RunSingleCase_owner():
    case = RunSingleCase("case")
    assert case.owner is None
    assert case.problem is None
    assert case.offdesign.shape == (0, 0)
    assert case.design.shape == (0, 0)

    class Dummy(System):
        def setup(self):
            self.add_inward('x', 0.0)
            self.add_inward('y', 0.0)
            self.add_outward('z', 0.0)

    a = Dummy("a")
    a.add_driver(case)
    assert case.owner is a
    assert case.problem is None
    assert case.offdesign.shape == (0, 0)
    assert case.design.shape == (0, 0)
    case.design.add_unknown('x').add_equation('z == 0')
    case.add_unknown('y')
    assert case.problem is None
    assert case.offdesign.shape == (1, 0)
    assert case.design.shape == (1, 1)

    b = System("b")
    b.add_driver(case)
    assert case.owner is b
    assert case.problem is None
    assert case.offdesign.shape == (0, 0)
    assert case.design.shape == (0, 0)


def test_RunSingleCase_add_working_equations(DummyFactory, hat_case):
    # TODO Fred test partial couple - only variable or only equation
    s = System("compute")
    s.add_child(DummyFactory("mult"))
    d = RunSingleCase("case")
    s.add_driver(d)

    d.offdesign.add_unknown("mult.K1").add_equation("mult.p_out.x == 50")

    assert d.problem is None
    check_problem(d.design, 0, 0)
    check_problem(d.offdesign, 1, 1)

    assert d.offdesign.context is d.owner
    assert_keys(d.offdesign.unknowns, "mult.K1")
    assert_keys(d.offdesign.residues, "mult.p_out.x == 50")
    assert_all_type(d.offdesign.residues, Residue)
    assert d.offdesign.residues["mult.p_out.x == 50"].context is d.owner

    d.setup_run()
    check_problem(d.problem, 1, 1)
    assert_keys(d.problem.unknowns, "mult.K1")
    assert_keys(d.problem.residues, "mult.p_out.x == 50")

    d.offdesign.add_unknown("mult.K2").add_equation("mult.p_out.x == 30")
    check_problem(d.offdesign, 2, 2)
    assert_keys(d.offdesign.unknowns, "mult.K1", "mult.K2")
    assert_keys(d.offdesign.residues, "mult.p_out.x == 30", "mult.p_out.x == 50")

    d.setup_run()
    check_problem(d.design, 0, 0)
    check_problem(d.problem, 2, 2)
    assert_keys(d.problem.unknowns, "mult.K1", "mult.K2")
    assert_keys(d.problem.residues, "mult.p_out.x == 30", "mult.p_out.x == 50")

    d = RunSingleCase("case")
    with pytest.raises(AttributeError, match="Owner System is required to define unknowns"):
        d.add_unknown("mult.K1")

    with pytest.raises(AttributeError, match="Owner System is required to define equations"):
        d.add_equation("mult.p_out.x == 50")

    # Test full vector variable
    s, case = hat_case(RunSingleCase)

    case.add_unknown("one.a").add_equation("out.x == [20, -2, 10]")
    check_problem(case.design, 0, 0)
    check_problem(case.offdesign, 3, 3)
    assert_keys(case.offdesign.unknowns, "one.a")
    assert_keys(case.offdesign.residues, "out.x == [20, -2, 10]")

    case.setup_run()
    check_problem(case.design, 0, 0)
    check_problem(case.problem, 3, 3)
    assert_keys(case.problem.unknowns, "one.a")
    assert_keys(case.problem.residues, "out.x == [20, -2, 10]")
    unknown = case.problem.unknowns["one.a"]
    assert np.array_equal(unknown.mask, [True, True, True])

    # Test vector variable with mask
    s, case = hat_case(RunSingleCase)

    case.add_unknown("one.a[1]").add_equation("out.x[1] == 42")
    check_problem(case.design, 0, 0)
    check_problem(case.offdesign, 1, 1)
    assert_keys(case.offdesign.unknowns, "one.a[1]")
    assert_keys(case.offdesign.residues, "out.x[1] == 42")
    unknown = case.offdesign.unknowns["one.a[1]"]
    assert np.array_equal(unknown.mask, [False, True, False])

    assert case.problem is None
    case.setup_run()
    check_problem(case.problem, 1, 1)
    assert_keys(case.problem.unknowns, "one.a[1]")
    assert_keys(case.problem.residues, "out.x[1] == 42")
    unknown = case.problem.unknowns["one.a[1]"]
    assert np.array_equal(unknown.mask, [False, True, False])


def test_RunSingleCase_add_design_equations(DummyFactory, hat_case):
    # TODO Fred test partial couple - only variable or only equation
    s = System("compute")
    s.add_child(DummyFactory("mult"))
    d = RunSingleCase("case")
    s.add_driver(d)

    d.design.add_unknown("mult.inwards.K1").add_equation("mult.p_out.x == 40")

    check_problem(d.design, 1, 1)
    check_problem(d.offdesign, 0, 0)
    assert_keys(d.design.unknowns, "mult.K1")
    assert_keys(d.design.residues, "mult.p_out.x == 40")

    d.design.add_unknown("mult.inwards.K2").add_equation("mult.p_out.x == 30")
    check_problem(d.design, 2, 2)
    check_problem(d.offdesign, 0, 0)
    assert_keys(d.design.unknowns, "mult.K1", "mult.K2")
    assert_keys(d.design.residues, "mult.p_out.x == 30", "mult.p_out.x == 40")

    d = RunSingleCase("case")
    with pytest.raises(AttributeError, match="Owner System is required to define unknowns"):
        d.design.add_unknown("mult.K1")

    with pytest.raises(AttributeError, match="Owner System is required to define equations"):
        d.design.add_equation("mult.p_out.x == 50")

    # Test full vector variable
    s, case = hat_case(RunSingleCase)
    case.design.add_unknown("one.a").add_equation("out.x == [20, -2, 10]")
    check_problem(case.design, 3, 3)
    check_problem(case.offdesign, 0, 0)
    assert_keys(case.design.unknowns, "one.a")
    assert_keys(case.design.residues, "out.x == [20, -2, 10]")

    # Test vector variable with mask
    s, case = hat_case(RunSingleCase)
    case.design.add_unknown("one.a[1]").add_equation("out.x[1] == 42")
    check_problem(case.design, 1, 1)
    check_problem(case.offdesign, 0, 0)
    assert_keys(case.design.unknowns, "one.a[1]")
    assert_keys(case.design.residues, "out.x[1] == 42")
    unknown = case.design.unknowns["one.a[1]"]
    assert np.array_equal(unknown.mask, [False, True, False])


def test_RunSingleCase_clean_run(DummyFactory):
    def Dummy(name):
        return DummyFactory(name, unknown=["p_in.x", "K2"])

    mult = Dummy("mult")
    case = RunSingleCase("case")
    mult.add_driver(case)

    mult.K1 = 12.0
    mult.K2 = 42.0
    assert mult.K1 == 12

    case.add_value("K1", 15.0)
    assert case.problem is None
    case.setup_run()
    check_problem(case.problem, 2, 0)
    assert_keys(case.problem.unknowns, "K2", "p_in.x")
    for key in ("K2", "p_in.x"):
        unknown = case.problem.unknowns[key]
        assert unknown.name == key
        assert unknown.context is mult

    case.clean_run()
    assert case.problem is None

    # Test with subsystems
    s = System("compute")
    s.add_child(Dummy("mult"))
    case = RunSingleCase("case")
    s.add_driver(case)

    s.mult.K1 = 12.0
    s.mult.K2 = 42.0
    
    case.add_value("mult.K1", 15.0)
    assert case.problem is None
    case.setup_run()
    check_problem(case.problem, 2, 0)
    for name in ("K2", "p_in.x"):
        key = f"mult.{name}"
        unknown = case.problem.unknowns[key]
        assert unknown.name == name
        assert unknown.context is s.mult

    case.clean_run()
    assert case.problem is None


def test_RunSingleCase_get_problem(DummyFactory):
    def make_case():
        s = System("compute")
        s.add_child(DummyFactory("mult"))
        d = RunSingleCase("case")
        s.add_driver(d)
        return s, d

    # Test design iteratives
    s, d = make_case()
    d.design.add_unknown("mult.K1").add_equation("mult.p_out.x == 50")
    d.setup_run()
    m = d.get_problem()
    assert_keys(m.unknowns, "mult.K1")

    # Test offdesign iteratives
    s, d = make_case()
    d.offdesign.add_unknown("mult.K1").add_equation("mult.p_out.x == 50")
    d.setup_run()
    m = d.get_problem()
    assert_keys(m.unknowns, "mult.K1")

    # Test residues
    s, d = make_case()

    s.mult.p_out.x = 1.0
    d.setup_run()
    m = d.get_problem()
    assert len(m.residues) == 0

    d.design.add_unknown(["mult.K1", "mult.K2"]).add_equation(
        [
            {"equation": "mult.p_out.x == 30"},
            {"equation": "mult.p_out.x == 40"},
        ]
    )
    d.setup_run()
    m = d.get_problem()
    assert_keys(m.residues, "mult.p_out.x == 40", "mult.p_out.x == 30")
    residue = m.residues["mult.p_out.x == 40"]
    assert residue.value == Residue.evaluate_residue(s.mult.p_out.x, 40.0)
    residue = m.residues["mult.p_out.x == 30"]
    assert residue.value == Residue.evaluate_residue(s.mult.p_out.x, 30.0)

    # TODO write more tests
    #   - checking case of mixture of design and offdesign equations


def test_RunSingleCase_get_init(DummyFactory):
    def make_case():
        s = System("compute")
        s.add_child(DummyFactory("mult"))
        d = RunSingleCase("case")
        s.add_driver(d)
        return s, d

    # Test design iteratives
    s, d = make_case()
    d.design.add_unknown("mult.K1").add_equation("mult.p_out.x == 50")

    d.setup_run()
    init_array = d.get_init()
    assert np.array_equal(init_array, np.asarray([s.mult.K1]))
    d.clean_run()

    d.set_init({"mult.K1": 42})
    d.setup_run()
    init_array = d.get_init()
    assert np.array_equal(init_array, np.asarray([42]))
    d.clean_run()

    d.set_init({"mult.K1": 41, "mult.K2": 43})
    d.setup_run()
    init_array = d.get_init()
    assert np.array_equal(init_array, np.asarray([41]))
    d.clean_run()

    # Test offdesign iteratives
    s, d = make_case()

    d.offdesign.add_unknown("mult.K1").add_equation("mult.p_out.x == 50")
    d.setup_run()
    init_array = d.get_init()
    assert np.array_equal(init_array, [s.mult.K1])
    d.clean_run()

    s.mult.K1 = 10
    d.set_init({"mult.K1": 33})
    assert_keys(d.initial_values, "mult.K1")
    assert_all_type(d.initial_values, Boundary)
    assert s.mult.K1 == 33
    d.setup_run()
    init_array = d.get_init()
    assert np.array_equal(init_array, [33])
    d.clean_run()

    s.mult.K1 = 10
    assert_keys(d.initial_values, "mult.K1")
    assert_all_type(d.initial_values, Boundary)
    d.setup_run()
    init_array = d.get_init()
    assert np.array_equal(init_array, [33])
    d.clean_run()
    d.solution["mult.K1"] = 10.0
    d.setup_run()
    init_array = d.get_init()
    assert np.array_equal(init_array, [10])
    d.clean_run()

    s.mult.K1 = 10
    d.setup_run()
    init_array = d.get_init(force_init=True)
    assert np.array_equal(init_array, [33])
    d.clean_run()

    d.set_init({"mult.K1": 32, "mult.K2": 34})
    assert_keys(d.initial_values, "mult.K1", "mult.K2")
    assert_all_type(d.initial_values, Boundary)

    d.setup_run()
    init_array = d.get_init()
    assert np.array_equal(init_array, [32])
    d.clean_run()
    assert_keys(d.initial_values, "mult.K1", "mult.K2")
    assert_all_type(d.initial_values, Boundary)

    d.design.add_unknown("mult.K2").add_equation("mult.Ksum == 20")
    s.mult.K1 = 10
    d.setup_run()
    init_array = d.get_init()
    assert_keys(d.initial_values, "mult.K1", "mult.K2")
    assert_all_type(d.initial_values, Boundary)
    assert np.array_equal(init_array, [34, 32])
    d.clean_run()

    d.solution["mult.K1"] = 22
    d.solution["mult.K2"] = 42
    d.setup_run()
    init_array = d.get_init()
    assert np.array_equal(init_array, [42, 22])
    d.clean_run()
