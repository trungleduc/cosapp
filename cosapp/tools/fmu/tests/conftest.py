import pytest

import numpy as np
from cosapp.ports import Port
from cosapp.systems import System

from cosapp.tests.library.systems.multiply import IterativeNonLinear


def case_factory(system_cls, name, **kwargs):
    """Case factory used in test fixtures below"""
    return system_cls(name, **kwargs)


class Float(Port):
    def setup(self):
        self.add_variable('x', 1.)


class AllCausality(System):
    def setup(self):
        self.add_input(Float, "in_")
        self.add_inward("in_x", 22.)
        self.add_output(Float, "out")
        self.add_outward("out_y", 42.)

@pytest.fixture(scope='function')
def allcausality():
    return case_factory(AllCausality, 'test')


class TestType(System):

    def setup(self):
        self.add_inward("a", True)
        self.add_inward("b", 22)
        self.add_inward("c", 2.54)
        self.add_inward("d", "string")
        self.add_inward("f", np.array(True))
        self.add_inward("g", np.array(22))
        self.add_inward("h", np.array(2.54))
        self.add_inward("i", np.array("string"))

        self.add_inward("k", np.array([0., 1, 2, ]))
        self.add_inward("l", dict(a=1, b=2))

@pytest.fixture(scope='function')
def testtype():
    return case_factory(TestType, 'test')


class ExpRampOde(System):
    """
    System representing function f(t) = a * (1 - exp(-t / tau)),
    through ODE: tau * f' + f = a
    """
    def setup(self):
        self.add_inward('a', 1.0)
        self.add_inward('tau', 1.0)

        self.add_outward('df_dt', 0.0)
        self.add_transient('f', der='df_dt', max_time_step='tau / 5')

    def compute(self):
        self.df_dt = (self.a - self.f) / self.tau

    def __call__(self, t):
        """Analytical solution at time t"""
        return self.a * (1 - np.exp(-t / self.tau))

@pytest.fixture(scope='function')
def ode():
    return case_factory(ExpRampOde, 'ode')


class VectorSyst(System):

    def setup(self):
        self.add_inward("vi", [1, 2, 3])
        self.add_inward("vr", [1., 2., 3.])

@pytest.fixture(scope='function')
def vector_syst():
    return case_factory(VectorSyst, "syst")


@pytest.fixture(scope='function')
def iterativenonlinear():
    snl = IterativeNonLinear('nl')
    snl.splitter.inwards.split_ratio = 0.1
    snl.mult2.inwards.K1 = 1
    snl.mult2.inwards.K2 = 1
    snl.nonlinear.inwards.k1 = 1
    snl.nonlinear.inwards.k2 = 0.5

    return snl


class VectorProblem(System):

    def setup(self):
        self.add_inward("in_")
        self.add_outward("out")

        self.add_inward("dummy_coef", np.array([1., 2.]))

        self.add_unknown("dummy_coef[0]").add_equation("dummy_coef[0] == 0.5 * (in_ + out)")

    def compute(self):
        self.out = self.dummy_coef[1] + self.in_

@pytest.fixture(scope='function')
def vector_problem():
    return case_factory(VectorProblem, "problem")
