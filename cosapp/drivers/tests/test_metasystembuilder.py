import numpy as np
import pytest

from cosapp.ports import Port
from cosapp.systems import System
from cosapp.drivers import MetaSystemBuilder


class XPort(Port):
    def setup(self):
        self.add_variable("x")

class MyExp(System):
    def setup(self):
        self.add_input(XPort, "p_in")
        self.add_output(XPort, "p_out")

        self.add_inward({"K1": 1.0, "K2": 1.0})

    def compute(self):
        self.p_out.x = self.K1 * np.exp(self.p_in.x * self.K2)

def test_MetaSystemBuilder_add_input_var():
    s = MyExp("s")
    d = s.add_driver(MetaSystemBuilder("builder"))

    d.add_input_var("K1", 0.0, 10.0)
    assert d.doe.input_vars == {"K1": {"lower": 0.0, "upper": 10.0, "count": 2}}
    assert len(d.responses) == 0

    d.add_input_var({"K1": {"lower": 0.2, "upper": 20.0}})
    assert d.doe.input_vars == {"K1": {"lower": 0.2, "upper": 20.0, "count": 2}}
    assert len(d.responses) == 0

    d.add_input_var(
        {"K1": {"lower": 0.2, "upper": 20.0}, "K2": {"lower": 0.1, "upper": 200.0}}
    )
    assert d.doe.input_vars == {
            "K1": {"lower": 0.2, "upper": 20.0, "count": 2},
            "K2": {"lower": 0.1, "upper": 200.0, "count": 2},
        }
    assert len(d.responses) == 0

    with pytest.raises(TypeError):
        d.add_input_var({"myvar": {"upper": 20.0}})
    with pytest.raises(TypeError):
        d.add_input_var({"myvar": 20.0})

    with pytest.raises(AttributeError):
        d.add_input_var("myvar", 0.0, 10.0)
    with pytest.raises(AttributeError):
        d.add_input_var({"myvar": {"lower": 0.2, "upper": 20.0}})

def test_MetaSystemBuilder_add_response():
    s = MyExp("s")
    d = s.add_driver(MetaSystemBuilder("builder"))

    d.add_response("K1")

    with pytest.raises(TypeError):
        d.add_response(10.0)

    d.add_response(["K1", "K2"])

    with pytest.raises(TypeError):
        d.add_response(["K1", 55])

    with pytest.raises(AttributeError):
        d.add_response(["var4"])

def test_MetaSystemBuilder_compute():
    s = MyExp("s")
    d = s.add_driver(MetaSystemBuilder("builder"))
    assert d._metasystem is None

    d.add_input_var(
        {"K1": {"lower": 0.9, "upper": 1.1}, "K2": {"lower": 0.95, "upper": 1.05}}
    )
    d.add_response("p_out.x")
    s.run_drivers()
    # TODO this is light we should test the size of the system at least
    assert d._metasystem is not None
    
    # d._metasystem.run_once()
