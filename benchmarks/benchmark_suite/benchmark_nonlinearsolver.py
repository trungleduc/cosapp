from cosapp.base import System
from cosapp.drivers import NonLinearSolver

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


class BenchNonLinearSolverScalar:
    number = 100
    repeat = 10

    def setup(self):
        self.n_iter = 100
        self.x = np.array([5.])

        self.s = s = ScalarSystem("s")
        self.nls = nls = s.add_driver(NonLinearSolver("nls"))
        nls.add_equation("y * 2. - a == 10.").add_unknown("x")
        s.run_drivers()

    def time_set_iteratives(self):
        self.nls.set_iteratives(self.x)

    def time_resolution_method(self):
        with System.set_master("dummy"):
            x0 = np.array([5.])
            self.nls.compute_jacobian = True
            self.nls.resolution_method(self.nls._fresidues, x0, options=self.nls._get_solver_options())


class BenchNonLinearSolverArrays:
    number = 100
    repeat = 10

    def setup(self):
        self.n_iter = 100
        self.values = np.ones(3)

        s = ArraySystem("s")
        self.nls = nls = s.add_driver(NonLinearSolver("nls"))
        nls.add_equation("x[1:] == y[1:]").add_unknown("x[1:]")
        s.run_drivers()

    def time_set_iteratives(self):
        self.nls.set_iteratives(self.values)

    def time_resolution_method(self):
        with System.set_master("dummy"):
            x0 = self.values
            self.nls.compute_jacobian = True
            self.nls.resolution_method(self.nls._fresidues, x0, options=self.nls._get_solver_options())
