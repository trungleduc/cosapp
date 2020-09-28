import pytest

from cosapp.drivers import NonLinearMethods, NonLinearSolver
from cosapp.tests.library.systems import (
    SimpleTurbofan,
    AdvancedTurbofan,
    ComplexTurbofan,
)


def test_SimpleTurbofan_set_inputs():
    s = SimpleTurbofan("MySimpleTurbofan")

    s.exec_order = ["atm", "inlet", "fan", "duct", "noz"]
    s.duct.inwards.cst_loss = 0.98

    design = s.add_driver(NonLinearSolver("design"))
    design.runner.offdesign.add_unknown("fan.mech_in.PW")

    s.fan.mech_in.XN = 80
    # test inlet XN value
    assert s.fan.mech_in.XN == 80.0
    assert s.duct.inwards.cst_loss == 0.98


def test_SimpleTurbofan_running_80():
    s = SimpleTurbofan("MySimpleTurbofan")

    s.exec_order = ["atm", "inlet", "fan", "duct", "noz"]
    s.duct.inwards.cst_loss = 0.98

    design = s.add_driver(NonLinearSolver("design"))
    design.runner.offdesign.add_unknown("fan.mech_in.PW")

    s.fan.mech_in.XN = 80
    s.run_drivers()
    assert s.fan.mech_in.XN == 80.0
    assert s.fan.gh / 0.038579270081290681 == pytest.approx(1.0, abs=1e-5)
    assert s.fan.mech_in.PW / 8279856.0537550747 == pytest.approx(1.0, abs=1e-5)
    # a second run changes fan power because of poor fsolve behavior
    # first run gives fan power residue bigger than 1e-5
    assert s.inlet.W_in.W / 161.46335209778994 == pytest.approx(1.0, abs=1e-4)


def test_SimpleTurbofan_running_100():
    s = SimpleTurbofan("MySimpleTurbofan")

    s.exec_order = ["atm", "inlet", "fan", "duct", "noz"]
    s.duct.inwards.cst_loss = 0.98

    design = s.add_driver(NonLinearSolver("design"))
    design.runner.offdesign.add_unknown("fan.mech_in.PW")

    s.fan.mech_in.XN = 100
    s.run_drivers()

    assert s.fan.mech_in.XN == 100.0
    assert s.fan.gh / 0.15669212068943697 == pytest.approx(1.0, abs=1e-5)
    assert s.fan.mech_in.PW / 10874232.300347026 == pytest.approx(1.0, abs=1e-5)
    assert s.inlet.W_in.W / 177.03398835527886 == pytest.approx(1.0, abs=1e-5)


def test_AdvancedTurbofan_set_inputs():
    s = AdvancedTurbofan("MyAdvancedTurbofan")

    s.exec_order = ["atm", "inlet", "fan", "merger", "duct", "bleed", "noz"]
    s.duct.inwards.cst_loss = 0.98

    design = s.add_driver(NonLinearSolver("design"))
    design.runner.offdesign.add_unknown("fan.mech_in.PW")

    s.fan.mech_in.XN = 80
    # test inlet XN value
    assert s.fan.mech_in.XN == 80.0
    assert s.duct.inwards.cst_loss == 0.98


def test_AdvancedTurbofan_running_80():
    s = AdvancedTurbofan("MyAdvancedTurbofan")

    s.exec_order = ["atm", "inlet", "fan", "merger", "duct", "bleed", "noz"]
    s.duct.inwards.cst_loss = 0.98

    design = s.add_driver(NonLinearSolver("design"))
    design.runner.offdesign.add_unknown("fan.mech_in.PW")

    s.fan.mech_in.XN = 80
    s.run_drivers()
    assert s.fan.mech_in.XN == 80.0
    assert s.fan.gh / 0.038579270081290681 == pytest.approx(1.0, abs=1e-5)
    assert s.fan.mech_in.PW / 8279856.0537550747 == pytest.approx(1.0, abs=1e-5)
    assert s.inlet.W_in.W / 161.46335209778994 == pytest.approx(1.0, abs=1e-5)

    assert s.bleed.fl2_out.W / 1.6309429518683161 == pytest.approx(1.0, abs=1e-5)
    assert s.merger.fl2_in.W / 1.6309429518683161 == pytest.approx(1.0, abs=1e-5)
    assert s.bleed.fl2_out.Pt / 180023.00671059423 == pytest.approx(1.0, abs=1e-5)
    assert s.merger.fl2_in.Pt / 180023.00671059423 == pytest.approx(1.0, abs=1e-5)
    assert s.bleed.fl2_out.Tt / 324.22579195039475 == pytest.approx(1.0, abs=1e-5)
    assert s.merger.fl2_in.Tt / 324.22579195039475 == pytest.approx(1.0, abs=1e-5)


def test_AdvancedTurbofan_running_100():
    s = AdvancedTurbofan("MyAdvancedTurbofan")

    s.exec_order = ["atm", "inlet", "fan", "merger", "duct", "bleed", "noz"]
    s.duct.inwards.cst_loss = 0.98

    design = s.add_driver(NonLinearSolver("design"))
    design.runner.offdesign.add_unknown("fan.mech_in.PW")

    s.fan.mech_in.XN = 100
    s.run_drivers()

    assert s.fan.mech_in.XN == 100.0
    assert s.fan.gh / 0.15669212068943689 == pytest.approx(1.0, abs=1e-5)
    assert s.fan.mech_in.PW / 10874232.300347026 == pytest.approx(1.0, abs=1e-5)
    assert s.inlet.W_in.W / 177.03398835527886 == pytest.approx(1.0, abs=1e-5)

    assert s.bleed.fl2_out.W / 1.7882221045987776 == pytest.approx(1.0, abs=1e-5)
    assert s.merger.fl2_in.W / 1.7882221045987776 == pytest.approx(1.0, abs=1e-5)
    assert s.bleed.fl2_out.Pt / 200435.42510736716 == pytest.approx(1.0, abs=1e-5)
    assert s.merger.fl2_in.Pt / 200435.42510736716 == pytest.approx(1.0, abs=1e-5)
    assert s.bleed.fl2_out.Tt / 334.32982635034119 == pytest.approx(1.0, abs=1e-5)
    assert s.merger.fl2_in.Tt / 334.32982635034119 == pytest.approx(1.0, abs=1e-5)


def test_ComplexTurbofan_set_inputs():
    s = ComplexTurbofan("MyComplexTurbofan")
    s.duct.inwards.cst_loss = 0.98
    s.fanC.ductC.duct.inwards.cst_loss = 0.98
    s.fanC.ductC.bleed.inwards.split_ratio = 0.995
    s.bleed.inwards.split_ratio = 0.99

    design = s.add_driver(NonLinearSolver("design", tol=1e-7))
    design.runner.offdesign.add_unknown("fanC.mech_in.PW")

    s.fanC.mech_in.XN = 80
    # test inlet XN value
    assert s.fanC.mech_in.XN == 80.0
    assert s.duct.inwards.cst_loss == 0.98
    assert s.fanC.ductC.duct.inwards.cst_loss == 0.98


def test_ComplexTurbofan_running_80():
    s = ComplexTurbofan("MyComplexTurbofan")
    s.duct.inwards.cst_loss = 0.98
    s.fanC.ductC.duct.inwards.cst_loss = 0.98
    s.fanC.ductC.bleed.inwards.split_ratio = 0.995
    s.bleed.inwards.split_ratio = 0.99

    design = s.add_driver(NonLinearSolver("design", tol=1e-7))
    design.runner.offdesign.add_unknown("fanC.mech_in.PW")

    s.fanC.mech_in.XN = 80
    s.run_drivers()
    assert s.fanC.mech_in.XN == 80.0
    assert s.fanC.gh / 0.038579270081290681 == pytest.approx(1.0, abs=1e-5)
    assert s.fanC.mech_in.PW / 8114258.9326799763 == pytest.approx(1.0, abs=1e-5)
    assert s.inlet.W_in.W / 158.23408418127087 == pytest.approx(1, abs=1e-5)

    assert s.bleed.fl2_out.W / 1.598324092287366 == pytest.approx(1.0, abs=1e-5)
    assert s.merger.fl2_in.W / 1.598324092287366 == pytest.approx(1.0, abs=1e-5)
    assert s.bleed.fl2_out.Pt / 176422.54890477096 == pytest.approx(1.0, abs=1e-5)
    assert s.merger.fl2_in.Pt / 176422.54890477096 == pytest.approx(1.0, abs=1e-5)
    assert s.bleed.fl2_out.Tt / 324.22579195039475 == pytest.approx(1.0, abs=1e-5)
    assert s.merger.fl2_in.Tt / 324.22579195039475 == pytest.approx(1.0, abs=1e-5)


def test_ComplexTurbofan_internal_loop_ductC_running_80():
    # Another run with an internal loop at ductC level
    s = ComplexTurbofan("MyComplexTurbofan")
    s.duct.inwards.cst_loss = 0.98
    s.fanC.ductC.duct.inwards.cst_loss = 0.98
    s.fanC.ductC.bleed.inwards.split_ratio = 0.995
    s.bleed.inwards.split_ratio = 0.99

    design = s.add_driver(NonLinearSolver("design", tol=1e-7))
    design.runner.offdesign.add_unknown("fanC.mech_in.PW")

    s.fanC.mech_in.XN = 80
    s.fanC.ductC.add_driver(NonLinearSolver("design", method=NonLinearMethods.POWELL))
    s.run_drivers()

    assert s.fanC.mech_in.XN == 80.0
    assert s.fanC.gh / 0.038579270081290681 == pytest.approx(1, abs=1e-5)
    assert s.fanC.mech_in.PW / 8114258.9326799763 == pytest.approx(1.0, abs=1e-5)
    assert s.inlet.W_in.W / 158.23408418127087 == pytest.approx(1, abs=1e-5)

    assert s.bleed.fl2_out.W / 1.598324092287366 == pytest.approx(1.0, abs=1e-5)
    assert s.merger.fl2_in.W / 1.598324092287366 == pytest.approx(1.0, abs=1e-5)
    assert s.bleed.fl2_out.Pt / 176422.54890477096 == pytest.approx(1.0, abs=1e-5)
    assert s.merger.fl2_in.Pt / 176422.54890477096 == pytest.approx(1.0, abs=1e-5)
    assert s.bleed.fl2_out.Tt / 324.22579195039475 == pytest.approx(1.0, abs=1e-5)
    assert s.merger.fl2_in.Tt / 324.22579195039475 == pytest.approx(1.0, abs=1e-5)

    assert s.fanC.ductC.bleed.fl2_out.W / 0.79514614357830227 == pytest.approx(
        1.0, abs=1e-5
    )
    assert s.fanC.ductC.merger.fl2_in.W / 0.79514614357830227 == pytest.approx(
        1.0, abs=1e-5
    )
    assert s.fanC.ductC.bleed.fl2_out.Pt / 98802.0075 == pytest.approx(1.0, abs=1e-5)
    assert s.fanC.ductC.merger.fl2_in.Pt / 98802.0075 == pytest.approx(1.0, abs=1e-5)
    assert s.fanC.ductC.bleed.fl2_out.Tt / 273.15 == pytest.approx(1.0, abs=1e-5)
    assert s.fanC.ductC.merger.fl2_in.Tt / 273.15 == pytest.approx(1.0, abs=1e-5)


def test_ComplexTurbofan_internal_loop_fanC_running_80():
    # Another run with an internal loop at fanC level
    s = ComplexTurbofan("MyComplexTurbofan")
    s.duct.inwards.cst_loss = 0.98
    s.fanC.ductC.duct.inwards.cst_loss = 0.98
    s.fanC.ductC.bleed.inwards.split_ratio = 0.995
    s.bleed.inwards.split_ratio = 0.99

    s.add_driver(NonLinearSolver("design", tol=1e-7))

    s.fanC.mech_in.XN = 80
    design = s.fanC.add_driver(NonLinearSolver("design", tol=1e-7))
    design.runner.offdesign.add_unknown("mech_in.PW")
    s.run_drivers()

    assert s.fanC.mech_in.XN == 80.0
    assert s.fanC.gh / 0.038579270081290681 == pytest.approx(1, abs=1e-5)
    assert s.fanC.mech_in.PW / 8114258.9326799763 == pytest.approx(1.0, abs=1e-5)
    assert s.inlet.W_in.W / 158.23408418127087 == pytest.approx(1, abs=1e-5)

    assert s.bleed.fl2_out.W / 1.598324092287366 == pytest.approx(1.0, abs=1e-5)
    assert s.merger.fl2_in.W / 1.598324092287366 == pytest.approx(1.0, abs=1e-5)
    assert s.bleed.fl2_out.Pt / 176422.54890477096 == pytest.approx(1.0, abs=1e-5)
    assert s.merger.fl2_in.Pt / 176422.54890477096 == pytest.approx(1.0, abs=1e-5)
    assert s.bleed.fl2_out.Tt / 324.22579195039475 == pytest.approx(1.0, abs=1e-5)
    assert s.merger.fl2_in.Tt / 324.22579195039475 == pytest.approx(1.0, abs=1e-5)

    assert s.fanC.ductC.bleed.fl2_out.W / 0.79514614357830227 == pytest.approx(
        1.0, abs=1e-5
    )
    assert s.fanC.ductC.merger.fl2_in.W / 0.79514614357830227 == pytest.approx(
        1.0, abs=1e-5
    )
    assert s.fanC.ductC.bleed.fl2_out.Pt / 98802.0075 == pytest.approx(1.0, abs=1e-5)
    assert s.fanC.ductC.merger.fl2_in.Pt / 98802.0075 == pytest.approx(1.0, abs=1e-5)
    assert s.fanC.ductC.bleed.fl2_out.Tt / 273.15 == pytest.approx(1.0, abs=1e-5)
    assert s.fanC.ductC.merger.fl2_in.Tt / 273.15 == pytest.approx(1.0, abs=1e-5)


def test_ComplexTurbofan_running_100():
    s = ComplexTurbofan("MyComplexTurbofan")
    s.duct.inwards.cst_loss = 0.98
    s.fanC.ductC.duct.inwards.cst_loss = 0.98
    s.fanC.ductC.bleed.inwards.split_ratio = 0.995
    s.bleed.inwards.split_ratio = 0.99

    design = s.add_driver(NonLinearSolver("design", tol=1e-7))
    design.runner.offdesign.add_unknown("fanC.mech_in.PW")

    s.fanC.mech_in.XN = 100
    s.run_drivers()

    assert s.fanC.mech_in.XN == 100.0
    assert s.fanC.gh / 0.15669212068943694 == pytest.approx(1.0, abs=1e-5)
    assert s.fanC.mech_in.PW / 10656747.654340087 == pytest.approx(1.0, abs=1e-5)
    assert s.inlet.W_in.W / 173.49330858817328 == pytest.approx(1.0, abs=1e-5)

    assert s.bleed.fl2_out.W / 1.75245766250680226 == pytest.approx(1.0, abs=1e-5)
    assert s.merger.fl2_in.W / 1.75245766250680226 == pytest.approx(1.0, abs=1e-5)
    assert s.bleed.fl2_out.Pt / 196426.71660521982 == pytest.approx(1.0, abs=1e-5)
    assert s.merger.fl2_in.Pt / 196426.71660521982 == pytest.approx(1.0, abs=1e-5)
    assert s.bleed.fl2_out.Tt / 334.32982635034119 == pytest.approx(1.0, abs=1e-5)
    assert s.merger.fl2_in.Tt / 334.32982635034119 == pytest.approx(1.0, abs=1e-5)


def to_ComplexTurbofan_html():
    s = ComplexTurbofan("MyComplexTurbofan")
    s.duct.inwards.cst_loss = 0.98
    s.fanC.ductC.duct.inwards.cst_loss = 0.98
    s.fanC.ductC.bleed.inwards.split_ratio = 0.995
    s.bleed.inwards.split_ratio = 0.99

    s.add_driver(NonLinearSolver("design", tol=1e-7))

    s.to_html("complexturbofan.html")
