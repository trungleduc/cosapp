import pytest

from cosapp.drivers import NonLinearMethods, NonLinearSolver
from cosapp.tests.library.systems import (
    SimpleTurbofan,
    AdvancedTurbofan,
    ComplexTurbofan,
)


@pytest.fixture
def simple_turbofan():
    engine = SimpleTurbofan("engine")

    engine.exec_order = ["atm", "inlet", "fan", "duct", "noz"]
    engine.duct.cst_loss = 0.98
    engine.fan.mech_in.XN = 80

    solver = engine.add_driver(NonLinearSolver("solver"))
    solver.add_unknown("fan.mech_in.PW")

    return engine


@pytest.fixture
def advanced_turbofan():
    engine = AdvancedTurbofan("engine")

    engine.exec_order = ["atm", "inlet", "fan", "merger", "duct", "bleed", "noz"]
    engine.duct.cst_loss = 0.98
    engine.fan.mech_in.XN = 80

    solver = engine.add_driver(NonLinearSolver("solver"))
    solver.add_unknown("fan.mech_in.PW")

    return engine


@pytest.fixture
def complex_turbofan():
    def factory(with_solver=False):
        engine = ComplexTurbofan("engine")
        engine.duct.cst_loss = 0.98
        engine.fanC.ductC.duct.cst_loss = 0.98
        engine.fanC.ductC.bleed.split_ratio = 0.995
        engine.fanC.mech_in.XN = 80
        engine.bleed.split_ratio = 0.99

        if with_solver:
            solver = engine.add_driver(NonLinearSolver("solver", tol=1e-7))
            solver.add_unknown("fanC.mech_in.PW")

        return engine

    return factory


def test_SimpleTurbofan_set_inputs(simple_turbofan):
    engine = simple_turbofan

    assert engine.fan.mech_in.XN == 80.0
    assert engine.duct.cst_loss == 0.98


def test_SimpleTurbofan_solve_80(simple_turbofan):
    """Solve simple turbofan @ XN = 80"""
    engine = simple_turbofan
    engine.fan.mech_in.XN = 80
    engine.run_drivers()

    assert engine.fan.mech_in.XN == 80.0
    assert engine.fan.gh == pytest.approx(0.0385792701, rel=1e-5)
    assert engine.fan.mech_in.PW == pytest.approx(8279856, rel=1e-5)
    # a second run changes fan power because of poor fsolve behavior
    # first run gives fan power residue bigger than 1e-5
    assert engine.inlet.W_in.W == pytest.approx(161.463, rel=1e-4)


def test_SimpleTurbofan_solve_100(simple_turbofan):
    """Solve simple turbofan @ XN = 100"""
    engine = simple_turbofan
    engine.fan.mech_in.XN = 100
    engine.run_drivers()

    assert engine.fan.mech_in.XN == 100.0
    assert engine.fan.gh == pytest.approx(0.1566921207, rel=1e-5)
    assert engine.fan.mech_in.PW == pytest.approx(10874232, rel=1e-5)
    assert engine.inlet.W_in.W == pytest.approx(177.034, rel=1e-5)


def test_AdvancedTurbofan_set_inputs(advanced_turbofan: AdvancedTurbofan):
    engine = advanced_turbofan

    assert engine.fan.mech_in.XN == 80.0
    assert engine.duct.cst_loss == 0.98


def test_AdvancedTurbofan_solve_80(advanced_turbofan: AdvancedTurbofan):
    """Solve advanced turbofan @ XN = 80"""
    engine = advanced_turbofan
    engine.fan.mech_in.XN = 80
    engine.run_drivers()

    assert engine.fan.mech_in.XN == 80.0
    assert engine.fan.gh == pytest.approx(0.038579270081290681, rel=1e-5)
    assert engine.fan.mech_in.PW == pytest.approx(8279856, rel=1e-5)
    assert engine.inlet.W_in.W == pytest.approx(161.463, rel=1e-5)

    assert engine.bleed.fl2_out.W == pytest.approx(1.630942952, rel=1e-5)
    assert engine.merger.fl2_in.W == pytest.approx(1.630942952, rel=1e-5)
    assert engine.bleed.fl2_out.Pt == pytest.approx(180023, rel=1e-5)
    assert engine.merger.fl2_in.Pt == pytest.approx(180023, rel=1e-5)
    assert engine.bleed.fl2_out.Tt == pytest.approx(324.2258, rel=1e-5)
    assert engine.merger.fl2_in.Tt == pytest.approx(324.2258, rel=1e-5)


def test_AdvancedTurbofan_solve_100(advanced_turbofan: AdvancedTurbofan):
    """Solve advanced turbofan @ XN = 100"""
    engine = advanced_turbofan
    engine.fan.mech_in.XN = 100
    engine.run_drivers()

    assert engine.fan.mech_in.XN == 100.0
    assert engine.fan.gh == pytest.approx(0.15669212068943689, rel=1e-5)
    assert engine.fan.mech_in.PW == pytest.approx(10874232, rel=1e-5)
    assert engine.inlet.W_in.W == pytest.approx(177.034, rel=1e-5)

    assert engine.bleed.fl2_out.W == pytest.approx(1.7882221046, rel=1e-5)
    assert engine.merger.fl2_in.W == pytest.approx(1.7882221046, rel=1e-5)
    assert engine.bleed.fl2_out.Pt == pytest.approx(200435.4, rel=1e-5)
    assert engine.merger.fl2_in.Pt == pytest.approx(200435.4, rel=1e-5)
    assert engine.bleed.fl2_out.Tt == pytest.approx(334.330, rel=1e-5)
    assert engine.merger.fl2_in.Tt == pytest.approx(334.330, rel=1e-5)


def test_ComplexTurbofan_set_inputs(complex_turbofan):
    engine: ComplexTurbofan = complex_turbofan(with_solver=False)

    assert engine.fanC.mech_in.XN == 80.0
    assert engine.duct.cst_loss == 0.98
    assert engine.fanC.ductC.duct.cst_loss == 0.98
    assert engine.fanC.ductC.bleed.split_ratio == 0.995
    assert engine.bleed.split_ratio == 0.99


def test_ComplexTurbofan_solve_80(complex_turbofan):
    """Solve complex turbofan @ XN = 80 (ref case).
    """
    engine: ComplexTurbofan = complex_turbofan(with_solver=True)
    engine.run_drivers()

    assert engine.fanC.mech_in.XN == 80.0
    assert engine.fanC.gh == pytest.approx(0.03857927008129068, rel=1e-5)
    assert engine.fanC.mech_in.PW == pytest.approx(8114258.93, rel=1e-5)
    assert engine.inlet.W_in.W == pytest.approx(158.2341, rel=1e-5)

    assert engine.bleed.fl2_out.W == pytest.approx(1.59832409, rel=1e-5)
    assert engine.merger.fl2_in.W == pytest.approx(1.59832409, rel=1e-5)
    assert engine.bleed.fl2_out.Pt == pytest.approx(176422.55, rel=1e-5)
    assert engine.merger.fl2_in.Pt == pytest.approx(176422.55, rel=1e-5)
    assert engine.bleed.fl2_out.Tt == pytest.approx(324.2258, rel=1e-5)
    assert engine.merger.fl2_in.Tt == pytest.approx(324.2258, rel=1e-5)

    assert engine.fanC.ductC.bleed.fl2_out.W == pytest.approx(0.79514614358, rel=1e-5)
    assert engine.fanC.ductC.merger.fl2_in.W == pytest.approx(0.79514614358, rel=1e-5)
    assert engine.fanC.ductC.bleed.fl2_out.Pt == pytest.approx(98802.0075, rel=1e-5)
    assert engine.fanC.ductC.merger.fl2_in.Pt == pytest.approx(98802.0075, rel=1e-5)
    assert engine.fanC.ductC.bleed.fl2_out.Tt == pytest.approx(273.15, rel=1e-5)
    assert engine.fanC.ductC.merger.fl2_in.Tt == pytest.approx(273.15, rel=1e-5)


def test_ComplexTurbofan_inner_solver_ductC_solve_80(complex_turbofan):
    """Complex turbofan with local loop solved with an inner solver
    (subsystem `ductC`).
    """
    engine: ComplexTurbofan = complex_turbofan(with_solver=True)

    # Add inner driver to solve local loop
    engine.fanC.ductC.add_driver(NonLinearSolver("design", method=NonLinearMethods.POWELL))
    engine.run_drivers()

    assert engine.fanC.mech_in.XN == 80.0
    assert engine.fanC.gh == pytest.approx(0.03857927008, rel=1e-5)
    assert engine.fanC.mech_in.PW == pytest.approx(8114258.932, rel=1e-5)
    assert engine.inlet.W_in.W == pytest.approx(158.2341, rel=1e-5)

    assert engine.bleed.fl2_out.W == pytest.approx(1.59832409, rel=1e-5)
    assert engine.merger.fl2_in.W == pytest.approx(1.59832409, rel=1e-5)
    assert engine.bleed.fl2_out.Pt == pytest.approx(176422.55, rel=1e-5)
    assert engine.merger.fl2_in.Pt == pytest.approx(176422.55, rel=1e-5)
    assert engine.bleed.fl2_out.Tt == pytest.approx(324.2258, rel=1e-5)
    assert engine.merger.fl2_in.Tt == pytest.approx(324.2258, rel=1e-5)

    assert engine.fanC.ductC.bleed.fl2_out.W == pytest.approx(0.79514614358, rel=1e-5)
    assert engine.fanC.ductC.merger.fl2_in.W == pytest.approx(0.79514614358, rel=1e-5)
    assert engine.fanC.ductC.bleed.fl2_out.Pt == pytest.approx(98802.0075, rel=1e-5)
    assert engine.fanC.ductC.merger.fl2_in.Pt == pytest.approx(98802.0075, rel=1e-5)
    assert engine.fanC.ductC.bleed.fl2_out.Tt == pytest.approx(273.15, rel=1e-5)
    assert engine.fanC.ductC.merger.fl2_in.Tt == pytest.approx(273.15, rel=1e-5)


def test_ComplexTurbofan_inner_solver_fanC_solve_80(complex_turbofan):
    """Complex turbofan with local loop solved with an inner solver
    (subsystem `fanC`).
    """
    engine: ComplexTurbofan = complex_turbofan(with_solver=False)
    engine.add_driver(NonLinearSolver("solver", tol=1e-7))

    # Add inner solver to solve local loop
    fan_solver = engine.fanC.add_driver(NonLinearSolver("solver", tol=1e-7))
    fan_solver.add_unknown("mech_in.PW")

    engine.run_drivers()

    assert engine.fanC.mech_in.XN == 80.0
    assert engine.fanC.gh == pytest.approx(0.03857927008, rel=1e-5)
    assert engine.fanC.mech_in.PW == pytest.approx(8114258.932, rel=1e-5)
    assert engine.inlet.W_in.W == pytest.approx(158.2341, rel=1e-5)

    assert engine.bleed.fl2_out.W == pytest.approx(1.59832409, rel=1e-5)
    assert engine.merger.fl2_in.W == pytest.approx(1.59832409, rel=1e-5)
    assert engine.bleed.fl2_out.Pt == pytest.approx(176422.55, rel=1e-5)
    assert engine.merger.fl2_in.Pt == pytest.approx(176422.55, rel=1e-5)
    assert engine.bleed.fl2_out.Tt == pytest.approx(324.2258, rel=1e-5)
    assert engine.merger.fl2_in.Tt == pytest.approx(324.2258, rel=1e-5)

    assert engine.fanC.ductC.bleed.fl2_out.W == pytest.approx(0.79514614358, rel=1e-5)
    assert engine.fanC.ductC.merger.fl2_in.W == pytest.approx(0.79514614358, rel=1e-5)
    assert engine.fanC.ductC.bleed.fl2_out.Pt == pytest.approx(98802.0075, rel=1e-5)
    assert engine.fanC.ductC.merger.fl2_in.Pt == pytest.approx(98802.0075, rel=1e-5)
    assert engine.fanC.ductC.bleed.fl2_out.Tt == pytest.approx(273.15, rel=1e-5)
    assert engine.fanC.ductC.merger.fl2_in.Tt == pytest.approx(273.15, rel=1e-5)


def test_ComplexTurbofan_solve_100(complex_turbofan):
    """Solve complex turbofan @ XN = 100.
    """
    engine: ComplexTurbofan = complex_turbofan(with_solver=True)
    engine.fanC.mech_in.XN = 100.0
    engine.run_drivers()

    assert engine.fanC.mech_in.XN == 100.0
    assert engine.fanC.gh == pytest.approx(0.15669212069, rel=1e-5)
    assert engine.fanC.mech_in.PW == pytest.approx(10656747.6, rel=1e-5)
    assert engine.inlet.W_in.W == pytest.approx(173.4933, rel=1e-5)

    assert engine.bleed.fl2_out.W == pytest.approx(1.75245766, rel=1e-5)
    assert engine.merger.fl2_in.W == pytest.approx(1.75245766, rel=1e-5)
    assert engine.bleed.fl2_out.Pt == pytest.approx(196426.7, rel=1e-5)
    assert engine.merger.fl2_in.Pt == pytest.approx(196426.7, rel=1e-5)
    assert engine.bleed.fl2_out.Tt == pytest.approx(334.330, rel=1e-5)
    assert engine.merger.fl2_in.Tt == pytest.approx(334.330, rel=1e-5)


def test_ComplexTurbofan_to_html(tmp_path, complex_turbofan):
    engine: ComplexTurbofan = complex_turbofan(with_solver=True)
    engine.to_html(tmp_path / "complexturbofan.html")
