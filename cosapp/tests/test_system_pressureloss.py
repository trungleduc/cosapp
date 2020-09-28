from pathlib import Path

import pytest

from cosapp.systems import System
from cosapp.drivers import NonLinearSolver


def test_system0(test_library, test_data):
    #
    #  pressureLoss0D
    #
    s = System.load(test_data / "system_config_pressureloss.json")
    s.run_once()
    assert s.flnum_out.Pt == 90000.0


def test_system1(test_library, test_data):
    #
    # pressureLossSys (pressureLoss0D)
    #
    s = System.load(test_data / "system_config_pressureloss1.json")
    s.exec_order = ["p1"]
    s.run_once()

    assert s.p1.flnum_out.Pt == 90000.0


def test_system2(test_library, test_data):
    #
    # pressureLossSys (pressureLoss0D + pressureLoss0D)
    #
    s = System.load(test_data / "system_config_pressureloss2.json")
    s.exec_order = ["p1", "p2"]
    s.run_once()

    assert s.p2.flnum_out.Pt == 80000.0


def test_system2tank(test_library, test_data):
    #
    # pressureLossSys (pressureLoss0D + tank + pressureLoss0D a pressureLossSys)
    #

    s = System.load(test_data / "system_config_pressureloss2tank.json")

    s.exec_order = ["p1", "tank1", "p2"]

    # tank is half full
    s.tank1.vol = 10.0
    s.run_once()
    assert s.p2.flnum_out.Pt == 89900.0
    assert s.tank1.vol == 10.9
    assert s.tank1.flnum_out.W == 1.0

    # tank is empty
    s.tank1.vol = 0.0
    s.run_once()
    assert s.p2.flnum_out.Pt == 89900.0
    assert s.tank1.vol == 0.9
    assert s.tank1.flnum_out.W == 1.0

    # tank is full
    s.tank1.vol = 100.0
    s.run_once()
    assert s.p2.flnum_out.Pt == 80000.0
    assert s.tank1.vol == pytest.approx(100.0)
    assert s.tank1.flnum_out.W == 10.0


def test_system11(test_library, test_data):
    #
    # pressureLossSys (pressureLossSys(pressureLoss0D))
    # with direct connexion of module
    #
    s = System.load(test_data / "system_config_pressureloss11.json")
    s.exec_order = ["p1"]
    s.p1.exec_order = ["p11"]
    s.run_once()

    assert s.p1.p11.flnum_out.Pt / 90000.0 == pytest.approx(1.0, abs=5.0e-8)


def test_system11bis(test_library, test_data):
    #
    # pressureLossSys (pressureLossSys(pressureLoss0D))
    # with connexion of module using ".."
    #
    s = System.load(test_data / "system_config_pressureloss11bis.json")
    s.run_once()

    assert s.p2.p21.flnum_out.Pt / 80000.0 == pytest.approx(1.0, abs=5.0e-8)
    assert s.p3.flnum_out.Pt / 70000.0 == pytest.approx(1.0, abs=5.0e-8)


def test_system12(test_library, test_data):
    #
    # pressureLossSys (pressureLossSys(pressureLoss0D + pressureLoss0D))
    #
    s = System.load(test_data / "system_config_pressureloss12.json")
    s.exec_order = ["p1"]
    s.p1.exec_order = ["p11", "p12"]
    s.run_once()

    assert s.p1.p12.flnum_out.Pt / 80000.0 == pytest.approx(1.0, abs=5.0e-8)


def test_system22(test_library, test_data):
    #
    # pressureLossSys (pressureLossSys(pressureLoss0D + pressureLoss0D) + pressureLoss0D)
    #
    s = System.load(test_data / "system_config_pressureloss22.json")
    s.exec_order = ["p1", "p2"]
    s.p1.exec_order = ["p11", "p12"]
    s.run_once()

    assert s.p2.flnum_out.Pt / 70000.0 == pytest.approx(1.0, abs=5.0e-8)


def test_system121(test_library, test_data):
    #
    # pressureLossSys (pressureLoss0D + Splitter12 + pressureLoss0D//pressureLoss0D + Mixer21 pressureLoss0D)
    #
    s = System.load(test_data / "system_config_pressureloss121.json")

    s.exec_order = ["p1", "sp", "p2", "p3", "mx", "p4"]
    s.run_once()

    assert s.p4.flnum_out.Pt == 74800.0

    d = s.add_driver(NonLinearSolver("design", tol=5.0e-8))
    d.runner.offdesign.add_unknown("sp.x")

    s.run_drivers()

    # test children of the system are listed in the execution order list
    assert len(s.children) == len(s.exec_order)
    for child in s.children:
        assert child in s.exec_order

    # test iterative loop detection
    assert len(s.residues) == 0
    residues = s.get_unsolved_problem().residues
    assert "mx.(epsilon == 0)" in residues

    assert s.p4.flnum_out.Pt / 77500.0 == pytest.approx(1.0, abs=1e-6)
    assert s.p2.flnum_out.W / 5.0 == pytest.approx(1.0, abs=1e-6)
    assert s.mx.flnum_out.W / (s.mx.flnum_in1.W + s.mx.flnum_in2.W) == pytest.approx(
        1.0, abs=1e-5
    )


def test_system131(test_library, test_data):
    #
    # pressureLossSys (pressureLoss0D + Mixer21 + pressureLoss0D//pressureLoss0D(backward) + Splitter12 + pressureLoss0D)
    #
    # ! Do not suppress this unit test, it's the only one doing new computation with resetting the system
    s = System.load(test_data / "system_config_pressureloss131.json")

    s.exec_order = ["p1", "mx", "p2", "sp", "p3", "p4"]

    s.p3.K = -12100
    s.p2.K = 100
    s.sp.x = 0.5

    d = s.add_driver(NonLinearSolver("design"))

    s.run_drivers()
    assert s.p4.flnum_out.Pt / 67900.0 == pytest.approx(1.0, abs=1e-5)
    assert s.p2.flnum_out.W / 11.0 == pytest.approx(1.0, abs=1e-5)
    assert s.p4.flnum_out.W / 10.0 == pytest.approx(1.0, abs=1e-5)
    assert s.sp.x / (10 / 11.0) == pytest.approx(1.0, abs=2e-5)

    s.p3.K = -3600
    s.run_drivers()
    assert s.p4.flnum_out.Pt / 65600.0 == pytest.approx(1.0, abs=1e-5)
    assert s.p2.flnum_out.W / 12.0 == pytest.approx(1.0, abs=1e-5)
    assert s.p4.flnum_out.W / 10.0 == pytest.approx(1.0, abs=1e-5)
    assert s.sp.x / (10 / 12.0) == pytest.approx(1.0, abs=1e-5)

    s.p3.K = -12100
    d.options["factor"] = 0.7
    s.run_drivers()
    assert s.p4.flnum_out.Pt / 67900.0 == pytest.approx(1.0, abs=1e-5)
    assert s.p2.flnum_out.W / 11.0 == pytest.approx(1.0, abs=1e-5)
    assert s.p4.flnum_out.W / 10.0 == pytest.approx(1.0, abs=1e-5)
    assert s.sp.x / (10 / 11.0) == pytest.approx(1.0, abs=1e-5)

    s.p3.K = -260100
    # d.options["factor"] = 0.9
    s.run_drivers()

    assert s.p2.flnum_out.W / 10.2 == pytest.approx(1.0, abs=1e-5)
    assert s.p4.flnum_out.W / 10.0 == pytest.approx(1.0, abs=1e-5)
    assert s.sp.x / (10 / 10.2) == pytest.approx(1.0, abs=1e-5)
    assert s.p4.flnum_out.Pt / 69596.0 == pytest.approx(1.0, abs=1e-5)


def test_system222(test_library, test_data):
    #
    # pLS(pl0 + mx + (pLS (pL0 + mx + (pL0+pL0//pl0(b) + sp + pl0))//pl0(b)) + sp + pL0)
    #
    s = System.load(test_data / "system_config_pressureloss222.json")

    s.exec_order = ["p1", "m1", "p2", "s1", "p3", "p4"]

    s.flnum_in.W = 9.0
    s.flnum_in.Pt = 108100
    s.p2.p22.p221.K = 50
    s.p2.p22.p222.K = 50
    s.p3.K = -32100
    s.p2.p23.K = -12100

    d = s.add_driver(NonLinearSolver("design"))
    d.runner.offdesign.add_unknown(["p2.s21.x", "s1.x"])
    s.run_drivers()

    assert s.p2.p24.flnum_out.Pt / 67900.0 == pytest.approx(1.0, abs=1e-5)
    assert s.p2.flnum_out.W / 10.0 == pytest.approx(1.0, abs=1e-5)
    assert s.p4.flnum_out.W / 9.0 == pytest.approx(1.0, abs=1e-5)
    assert s.s1.x / (9 / 10) == pytest.approx(1.0, abs=1e-5)
    assert s.p2.s21.x / (10 / 11) == pytest.approx(1.0, abs=1e-5)

