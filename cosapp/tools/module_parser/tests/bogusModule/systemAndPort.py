from cosapp.base import System, Port
import numpy as np


class BogusPort(Port):
    pass


class AbPort(Port):
    def setup(self) -> None:
        self.add_variable('a', 1.0)
        self.add_variable('b', np.zeros(3))


class XyzPort(Port):
    def setup(self) -> None:
        self.add_variable('x', 3, desc='x var')
        self.add_variable('y', 2.0, unit='K')
        self.add_variable('z', 1.0)


class BogusSystem(System):
    """This is a markdown docstring,
    with an indent"""
    def setup(self) -> None:
        self.add_input(AbPort, 'p_in')
        self.add_output(XyzPort, 'p_out')
        self.add_inward('foo', 0)
        self.add_outward('bar', 1)


class SystemWithKwargs(System):
    def setup(self, n: int, r=None) -> None:
        self.add_property('n', n)
        self.add_inward('v', np.ones(n))
        if r is not None:
            self.add_inward('r', r)
