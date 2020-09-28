import pytest

from cosapp.systems import System
from cosapp.drivers import NonLinearSolver, NonLinearMethods, Optimizer
from cosapp.tests.library.systems.multiply import Merger, ExponentialLoad, Splitter


class ToOptimize(System):
    def setup(self):
        self.add_child(Merger("merger"), pulling={"p1_in": "p_in"})
        self.add_child(ExponentialLoad("main_load"))
        self.add_child(ExponentialLoad("secondary_load"))
        self.add_outward("load_", 0.0)
        self.add_child(Splitter("splitter"), pulling={"p2_out": "p_out"})

        self.connect(self.merger.p_out, self.main_load.p_in)
        self.connect(self.main_load.p_out, self.splitter.p_in)
        self.connect(self.splitter.p1_out, self.secondary_load.p_in)
        self.connect(self.secondary_load.p_out, self.merger.p2_in)

        self.exec_order = ["merger", "main_load", "splitter", "secondary_load"]

    def compute(self):
        self.load_ = self.main_load.load_ + self.secondary_load.load_


def test_integration_Optimizer_solve():
    s = ToOptimize("s")
    s.add_driver(NonLinearSolver("solver", method=NonLinearMethods.NR, factor=0.1))

    s.splitter.split_ratio = 0.1
    s.main_load.a = -1.0
    s.main_load.b = 3.0
    s.secondary_load.c = 4.0

    s.run_drivers()

    assert s.p_out.x == pytest.approx(1.0, abs=1e-5)
    assert s.secondary_load.p_in.x / s.main_load.p_in.x == 0.1
    assert s.load_ == pytest.approx(11.72947, abs=1e-5)


@pytest.mark.filterwarnings("ignore:Unknown solver options. .tol")
def test_integration_Optimizer_optimization():
    s = ToOptimize("s")
    d = NonLinearSolver("solver", method=NonLinearMethods.NR, factor=0.1)
    s.splitter.split_ratio = 0.1
    s.main_load.a = -1.0
    s.main_load.b = 3.0
    s.secondary_load.c = 4.0

    s.add_driver(d)
    s.run_drivers()

    s.drivers.clear()

    opt = s.add_driver(Optimizer("optimizer"))

    opt.runner.set_objective("load_")
    opt.runner.add_unknown("p_in.x")
    opt.runner.add_child(d)

    s.run_drivers()

    assert s.p_in.x == pytest.approx(4.338487, abs=1e-5)
    assert s.secondary_load.p_in.x / s.main_load.p_in.x == 0.1
    assert s.load_ == pytest.approx(5.78134, abs=1e-5)


def test_integration_Optimizer_upper_bound():
    s = ToOptimize("s")
    d = NonLinearSolver("solver", method=NonLinearMethods.NR, factor=0.1)
    s.splitter.split_ratio = 0.1
    s.main_load.a = -1.0
    s.main_load.b = 3.0
    s.secondary_load.c = 4.0

    s.add_driver(d)
    s.run_drivers()

    s.drivers.clear()

    opt = s.add_driver(Optimizer("optimizer"))

    opt.runner.set_objective("load_")
    opt.runner.add_unknown("p_in.x", upper_bound=4.2)
    opt.runner.add_child(d)

    s.run_drivers()

    assert s.p_in.x == pytest.approx(4.2, abs=1e-5)
    assert s.secondary_load.p_in.x / s.main_load.p_in.x == 0.1
    assert s.load_ == pytest.approx(5.78355, abs=1e-5)


def test_integration_Optimizer_lower_bound():
    s = ToOptimize("s")
    d = NonLinearSolver("solver", method=NonLinearMethods.NR, factor=0.1)
    s.splitter.split_ratio = 0.1
    s.main_load.a = -1.0
    s.main_load.b = 3.0
    s.secondary_load.c = 4.0

    s.add_driver(d)
    s.run_drivers()

    s.drivers.clear()

    opt = s.add_driver(Optimizer("optimizer"))

    opt.runner.set_objective("load_")
    opt.runner.add_unknown("p_in.x", lower_bound=4.5)
    opt.runner.add_child(d)

    s.run_drivers()

    assert s.p_in.x == pytest.approx(4.5, abs=1e-5)
    assert s.secondary_load.p_in.x / s.main_load.p_in.x == 0.1
    assert s.load_ == pytest.approx(5.78406, abs=1e-5)
