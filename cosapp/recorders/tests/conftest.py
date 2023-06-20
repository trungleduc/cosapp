import pytest

import numpy as np
from cosapp.ports import Port
from cosapp.systems import System

# <codecell>

class V1dPort(Port):
    def setup(self):
        self.add_variable("x", np.r_[1., 2., 3.])


class _AllTypesSystem(System):
    def setup(self):
        self.add_property('g', 9.81)
        self.add_input(V1dPort, "in_")
        self.add_inward("a", np.ones(3), unit="kg")
        self.add_inward("b", np.zeros(3), unit="N")
        self.add_inward("c", 23, unit="m")
        self.add_inward("e", "sammy")
        self.add_outward("d", list())
        self.add_output(V1dPort, "out")
        self.add_inward_modevar("m_in", 0.1)
        self.add_outward_modevar("m_out", False)
        self.add_event('beep')
        # Properties containing CoSApp objects
        self.add_property('v_ports', (self.in_, self.out))
        self.add_property('s_tuple', (
            self.add_child(System('foo')),
            self.add_child(System('bar')),
        ))

    def compute(self):
        self.out.x = self.a * self.in_.x + self.b
        self.d = [self.c, self.e]


@pytest.fixture(scope='function')
def AllTypesSystem():
    def _test_object(name):
        return _AllTypesSystem(name)
    return _test_object


@pytest.fixture(scope='function')
def SystemWithProps():
    """Returns a test system with properties"""
    class XyPort(Port):
        def setup(self):
            self.add_variable('x', 0.0)
            self.add_variable('y', 1.0)

        @property
        def xy_ratio(self):  # property matching '*_ratio' pattern
            return self.x / self.y

        def custom_ratio(self):  # method matching '*_ratio' pattern
            return 'not a property'

    class SystemWithProps(System):
        def setup(self):
            self.add_input(XyPort, 'in_')
            self.add_output(XyPort, 'out')
            self.add_outward('a', 0.0)

        @property
        def bogus_ratio(self):
            """Bogus property matching '*_ratio' name pattern"""
            return 2 * self.in_.x

        def compute(self):
            self.out.x = self.in_.x
            self.out.y = self.in_.y * 2
            self.a = 0.1 * self.out.xy_ratio

    def factory(name):
        return SystemWithProps(name)

    return factory
