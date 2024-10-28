from cosapp.base import System
from cosapp.drivers import NonLinearSolver
from cosapp.core.eval_str import EvalString
from cosapp.core.numerics.residues import Residue
import numpy as np


class ScalarSystem(System):
    def setup(self):
        self.add_inward("x", 1.0)
        self.add_inward("a", 1.0)
        self.add_inward("b", 1.0)
        self.add_outward("y", 1.0)

    def compute(self):
        self.y = self.a * self.x + self.b


class ArraySystem(System):
    def setup(self):
        self.add_inward("x", np.random.rand(3))
        self.add_outward("y", np.zeros(3))


class BenchScalarResidue:
    number = 1000
    repeat = 10

    def setup(self):
        self.s = s = ScalarSystem("s")
        nls = s.add_driver(NonLinearSolver("nls"))
        nls.add_equation("y * 2. - a == 10.").add_unknown("x")

        s.run_drivers()
        self.res = list(nls._raw_problem._residues.values())[0]
        self.string = EvalString("({'y * 2. - a'}, {'10.'})", s)

        EvalString.available_symbols()

        if hasattr(self.res, "evaluate_residue"):
            self.method = self.res.evaluate_residue
        else:
            self.method = self.res.residue_method

    def time_res_update(self):
        self.res.update()

    def time_sides_eval(self):
        self.string.eval()

    def time_eval_residue(self):
        self.method(2.0, 10.0, 1.0)

    def time_eval_init(self):
        EvalString("y * 2. - a == 10.", self.s)

    def time_res_init(self):
        Residue(self.s, "y * 2. - a == 10.")


class BenchArrayResidue:
    number = 1000
    repeat = 10

    def setup(self):
        self.lhs = np.random.rand(3)
        self.rhs = np.random.rand(3)

        self.s = s = ArraySystem("s")
        nls = s.add_driver(NonLinearSolver("nls"))
        nls.add_equation("x[1:] == y[1:]").add_unknown("x[1:]")

        s.run_drivers()
        self.res = list(nls._raw_problem._residues.values())[0]
        self.string = EvalString("x[1:] == y[1:]", s)

        EvalString.available_symbols()

        if hasattr(self.res, "evaluate_residue"):
            self.method = self.res.evaluate_residue
        else:
            self.method = self.res.residue_method

    def time_res_update(self):
        self.res.update()

    def time_sides_eval(self):
        self.string.eval()

    def time_eval_residue(self):
        self.method(self.lhs, self.rhs, 1.0)

    def time_eval_init(self):
        EvalString("x[1:] == y[1:]", self.s)

    def time_res_init(self):
        Residue(self.s, "x[1:] == y[1:]")
