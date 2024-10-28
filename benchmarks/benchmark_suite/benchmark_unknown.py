from cosapp.base import System
from cosapp.drivers import NonLinearSolver
from cosapp.core.numerics.boundary import Boundary

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


class BenchUnknownScalar:
    number = 1000
    repeat = 10

    def setup(self):
        self.s = s = ScalarSystem("s")
        self.nls = nls = s.add_driver(NonLinearSolver("nls"))
        nls.add_equation("y * 2. - a == 10.").add_unknown("x")
        self.unknowns = list(nls._raw_problem._unknowns.values())[0]

    def time_None_default_value(self):
        self.unknowns.update_default_value(value=None)

    def time_scalar_default_value(self):
        self.unknowns.update_default_value(value=self.number)


class BenchUnknownArrays:
    number = 1000
    repeat = 10

    def setup(self):
        s = ArraySystem("s")
        nls = s.add_driver(NonLinearSolver("nls"))
        nls.add_equation("x[1:] == y[1:]").add_unknown("x[1:]")
        s.run_drivers()
        
        self.unknowns = list(nls._raw_problem._unknowns.values())[0]
        self.values = np.random.rand(3)

    def time_nparray_default_value(self):
        self.unknowns.update_default_value(value=self.values)


class BenchBoundary:
    number = 1000
    repeat = 10

    def setup(self):
        self.n_iter = 1_000
        self.default = np.ones(3)

        self.scalar = ScalarSystem("scalar")
        self.array = ArraySystem("array")

        self.b_scalar = Boundary(self.scalar, "x")
        self.b_array = Boundary(self.array, "x")

        if hasattr(self.b_scalar, "update_default_value"):
            self.s_default_method = self.b_scalar.update_default_value 
            self.a_default_method = self.b_array.update_default_value 
        else:
            self.s_default_method = self.b_scalar.set_default_value 
            self.a_default_method = self.b_array.set_default_value 

    def time_init_scalar(self):
        Boundary(self.scalar, "x")

    def time_init_array(self):
        Boundary(self.array, "x")

    def time_getter_scalar(self):
        self.b_scalar.value
    
    def time_getter_array(self):
        self.b_array.value

    def time_setter_scalar(self):
        self.b_scalar.value = 5.
    
    def time_setter_array(self):
        self.b_array.value = self.default

    def time_set_default_scalar(self):
        self.s_default_method(5.)
    
    def time_set_default_array(self):
        self.a_default_method(self.default)

