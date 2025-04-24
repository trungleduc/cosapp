import pytest
import numpy as np
from contextlib import nullcontext as does_not_raise

from cosapp.base import System, Port
from cosapp.tools.help import DocDisplay
from cosapp.utils.testing import get_args, ArgsKwargs


class XyzPort(Port):
    def setup(self):
        self.add_variable('x')
        self.add_variable('y')
        self.add_variable('z')


class SystemWithKwargs(System):
    def setup(self, xi, n=2) -> None:
        self.add_property('xi', xi)
        self.add_inward('v', np.zeros(n))


@pytest.mark.parametrize("args_kwargs, expected", [
    (get_args(Port), does_not_raise()),
    (get_args(System), does_not_raise()),
    (get_args(System('foo')), does_not_raise()),
    (get_args(SystemWithKwargs, xi=0.5), does_not_raise()),
    (get_args(SystemWithKwargs, xi=0.5, n=3), does_not_raise()),
    (get_args(SystemWithKwargs('foo', xi=0.5)), does_not_raise()),
    (get_args(SystemWithKwargs, 0.5), pytest.raises(TypeError, match=r"takes [\d] positional argument")),
    (get_args(SystemWithKwargs), pytest.raises(TypeError, match=r"missing [\d] required positional argument")),
])
def test_DocDisplay__init__(args_kwargs: ArgsKwargs, expected):
    args, kwargs = args_kwargs
    with expected:
        DocDisplay(*args, **kwargs)
