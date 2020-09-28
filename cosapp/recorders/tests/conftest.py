import pytest

import numpy as np
from cosapp.ports import Port
from cosapp.systems import System

# <codecell>

class V1dPort(Port):
    def setup(self):
        self.add_variable("x", np.r_[1., 2., 3.])


class _AllTypesSystem(System):
    def setup(self, **kwargs):
        self.add_input(V1dPort, "in_")
        self.add_inward("a", np.ones(3), unit="kg")
        self.add_inward("b", np.zeros(3), unit="N")
        self.add_inward("c", 23, unit="m")
        self.add_inward("e", "sammy")
        self.add_outward("d", list())
        self.add_output(V1dPort, "out")

    def compute(self, time_ref):
        self.out.x = self.a * self.in_.x + self.b
        self.d = [self.c, self.e]


@pytest.fixture(scope='function')
def AllTypesSystem():
    def _test_object(name, **kwargs):
        return _AllTypesSystem(name, **kwargs)
    return _test_object
