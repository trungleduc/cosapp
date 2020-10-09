import pytest

import numpy as np
import copy

from cosapp.utils.testing import get_args, assert_keys
from cosapp.systems import System
from cosapp.ports.variable import Variable
from cosapp.core.numerics.distributions.distribution import Distribution
from cosapp.core.numerics.distributions.uniform import Uniform
from cosapp.core.connectors import Connector, ConnectorError
from cosapp.ports.port import PortType, Port


@pytest.mark.parametrize("direction", PortType)
@pytest.mark.parametrize("case_data, expected", [
    (
        dict(
            variables = [get_args("x"), get_args("y")],
            init = {"x": 0.5, "y": -4.1}
        ),
        dict()
    ),
    (
        dict(
            variables = [get_args("x"), get_args("y")],
            init = {"x": 0.5, "y": -4.1}
        ),
        dict()
    ),
    (
        dict(
            variables = [get_args("x", 0.5), get_args("y", 3.14)],
            init = {"x": 0.12345}
        ),
        dict(values = {"x": 0.12345, "y": 3.14})
    ),
    (
        dict(variables = [get_args("x", 0.5), get_args("y", -3.14)]),
        dict(values = {"x": 0.5, "y": -3.14})
    ),
    (
        dict(
            variables = [get_args("x"), get_args("y")],
            init = {"foo": 0.5}
        ),
        dict(error=KeyError, match="foo does not exist in Port")
    ),
    (
        dict(  # construction with dictionaries
            variables =
                [ # Port variables:
                    get_args("Pt", 101325.0, unit="Pa"),
                    get_args("W", 1.0,
                        unit="kg/s",
                        valid_range=(0.0, 2.0),
                        limits=(-5, 3.0),
                        desc="my lovely W",
                    ),
                ],
            init =
                {
                    "W": dict(
                        value=22.0,
                        valid_range=(-2.0, 1.0),
                        invalid_comment="not valid",
                        limits=(-2, 2.0),
                        out_of_limits_comment="out of bounds",
                        distribution=Uniform(0.1, 0.2, 0.4),
                    ),
                    "Pt": dict(value=3),
                },
        ),
        dict( # expected data:
            values = {"W": 22, "Pt": 3},
            details = 
            {
                "W": dict(
                    valid_range=(-2.0, 1.0),
                    invalid_comment="not valid",
                    limits=(-2, 2.0),
                    out_of_limits_comment="out of bounds",
                ),
                "Pt": dict(
                    limits=(-np.inf, np.inf),
                    valid_range=(-np.inf, np.inf),
                    invalid_comment="",
                    out_of_limits_comment="",
                ),
            }
        )
    ),
    (
        dict(
            variables = get_args("W", unit="kg/s"),
            init = {"W": dict(unit="lbm")},
        ),
        dict(error=AttributeError, match="can't set attribute")
    ),
    (
        dict(
            variables = get_args("W", unit="kg/s"),
            init = {"W": dict(banana="split")},
        ),
        dict(error=AttributeError, match="object has no attribute 'banana'")
    ),
])
def test_Port__init__(DummyPort, direction, case_data, expected):
    settings = copy.deepcopy(case_data)
    error = expected.get("error", None)
    name = settings.pop("name", "dummy")
    init = settings.pop("init", dict())
    if error is None:
        port = DummyPort(name, direction, init, **settings)
        assert port.name == name
        assert port.direction is direction
        assert port.owner is settings.get("owner", None)
        # check variable initialization (if any)
        if "values" not in expected:
            expected["values"] = init
        for var, value in expected["values"].items():
            assert getattr(port, var) == value
        # check variable details
        expected_details = expected.get("details", dict())
        for var, details in expected_details.items():
            actual = port.get_details(var)
            assert isinstance(actual.distribution, (type(None), Distribution))
            for name, value in details.items():
                assert getattr(actual, name) == value
    else:
        pattern = expected.get("match", None)
        with pytest.raises(error, match=pattern):
            port = DummyPort(name, direction, init, **settings)


@pytest.mark.parametrize("direction", PortType)
def test_Port_setup(DummyPort, direction):
    p = DummyPort("dummy", direction,
        variables = [
            get_args("x", 22.0),
            get_args("y", 10.0),
            get_args("z", 3.14),
        ])
    assert p.x == 22
    assert p.y == 10
    assert p.z == 3.14


@pytest.mark.parametrize("direction", PortType)
def test_Port_owner(DummyPort, direction):
    s = System("foo")
    p = DummyPort("dummy", direction,
        variables = get_args("x", 22.0),
        owner = s,
        )
    assert p.x == 22
    assert p.owner is s


@pytest.mark.parametrize("direction", PortType)
def test_Port_add_variable(DummyPort, direction):
    p = DummyPort("dummy", direction, {"var1": 0.0, "var2": 100.1},
        variables = [get_args("var1"), get_args("var2")],
        )
    with pytest.raises(AttributeError):
        p.add_variable("foo", 3.14)


@pytest.mark.parametrize("direction", PortType)
def test_Port_remove_variable(DummyPort, direction):
    p = DummyPort("dummy", direction, {"var1": 0.0, "var2": 100.1},
        variables = [get_args("var1"), get_args("var2")],
        )
    with pytest.raises(AttributeError):
        p.remove_variable("var1")


@pytest.mark.parametrize("direction", PortType)
@pytest.mark.parametrize("copy_dir", PortType)
def test_Port_copy(DummyPort, direction, copy_dir):
    # p = DummyPort("p", PortType.IN)
    p = DummyPort("dummy", direction,
        variables = [
            get_args("Pt", 101325.0, unit="Pa"),
            get_args("W", 1.0,
                unit="kg/s",
                valid_range=(0.0, 2.0),
                limits=(-5, 3.0),
                desc="my lovely W",
            ),
        ],
    )
    c = p.copy()
    assert c.name == p.name
    assert c.Pt == 101325
    assert c.W == 1
    details = c.get_details()
    assert_keys(details, "Pt", "W")

    assert details["Pt"].valid_range == (-np.inf, np.inf)
    assert details["Pt"].limits == (-np.inf, np.inf)
    assert details["Pt"].description == ""
    assert details["Pt"].unit == "Pa"

    assert details["W"].valid_range == (0, 2)
    assert details["W"].limits == (-5, 3)
    assert details["W"].description == "my lovely W"
    assert details["W"].unit == "kg/s"

    c = p.copy("other_name", copy_dir)
    assert c.name == "other_name"


@pytest.mark.skip(reason="TODO")
def test_Port_morph():
    pytest.fail()


@pytest.mark.parametrize("direction", PortType)
@pytest.mark.parametrize("copy_dir", PortType)
def test_Port_to_dict_with_def(DummyPort, direction, copy_dir):
    p = DummyPort("dummy", direction,
        variables = [
            get_args("Pt", 101325.0, unit="Pa"),
            get_args("W", 1.0,
                unit="kg/s",
                valid_range=(0.0, 2.0),
                limits=(-5, 3.0),
                desc="my lovely W",
            ),
        ],
    )
    port_dict = p.to_dict(True)
    assert_keys(port_dict, "dummy")
    assert  port_dict["dummy"] ==  {'__class__': 'DummyPort.<locals>.Factory.<locals>.PrototypePort', 'Pt': 101325.0, 'W': 1.0}