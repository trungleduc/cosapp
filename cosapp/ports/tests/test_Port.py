import pytest

import numpy as np
import logging, re
import copy

from cosapp.base import Port, System
from cosapp.ports.port import PortType
from cosapp.utils.distributions import Distribution, Uniform
from cosapp.utils.testing import get_args, assert_keys
from typing import Type


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
        dict(error=AttributeError, match="can't set attribute|no setter")
    ),
    (
        dict(
            variables = get_args("W", unit="kg/s"),
            init = {"W": dict(banana="split")},
        ),
        dict(error=AttributeError, match="object has no attribute 'banana'")
    ),
    # Test on name validity
    (
        dict(name="inwards"), dict(error=None)
    ),
    (
        dict(name="outwards"), dict(error=None)
    ),
    (
        dict(name="time"),
        dict(error=ValueError, match="reserved")
    ),
    (
        dict(name="3.14"),
        dict(error=ValueError, match="Name must start with a letter")
    ),
    (
        dict(name=3.14), dict(error=TypeError)
    ),
])
def test_Port__init__(PortClassFactory, direction, case_data, expected: dict):
    settings = copy.deepcopy(case_data)
    name = settings.pop("name", "dummy")
    init = settings.pop("init", dict())
    # Generate test port class
    DummyPort: Type[Port] = PortClassFactory(
        classname="DummyPort",
        variables=settings.pop('variables', []),
    )
    error = expected.get("error", None)

    if error is None:
        port = DummyPort(name, direction, init, **settings)
        assert port.name == name
        assert port.direction is direction
        assert port.owner is settings.get("owner", None)
        # check variable initialization (if any)
        expected.setdefault("values", init)
        for varname, value in expected["values"].items():
            assert getattr(port, varname) == value, varname
        # check variable details
        expected_details = expected.get("details", dict())
        for varname, details in expected_details.items():
            actual = port.get_details(varname)
            assert isinstance(actual.distribution, (type(None), Distribution))
            for name, value in details.items():
                assert getattr(actual, name) == value

    else:
        pattern = expected.get("match", None)
        with pytest.raises(error, match=pattern):
            port = DummyPort(name, direction, init, **settings)


@pytest.mark.parametrize("direction", PortType)
def test_Port_setup(PortClassFactory, direction):
    DummyPort: Type[Port] = PortClassFactory(
        classname="DummyPort",
        variables=[
            get_args("x", 22.0),
            get_args("y", 10.0),
            get_args("z", 3.14),
        ],
    )
    p = DummyPort("dummy", direction)
    assert p.x == 22
    assert p.y == 10
    assert p.z == 3.14


@pytest.mark.parametrize("direction", PortType)
def test_Port_owner(PortFactory, direction):
    s = System("foo")
    p = PortFactory("dummy", direction,
        variables=get_args("x", 22.0),
        owner=s,
    )
    assert p.x == 22
    assert p.owner is s


@pytest.mark.parametrize("direction", PortType)
def test_Port_add_variable(PortFactory, direction):
    p = PortFactory("dummy", direction,
        variables=[get_args("var1"), get_args("var2")],
    )
    with pytest.raises(AttributeError):
        p.add_variable("foo", 3.14)


@pytest.mark.parametrize("direction", PortType)
def test_Port_remove_variable(PortFactory, direction):
    p = PortFactory(
        "dummy", direction,
        variables=[get_args("var1"), get_args("var2")],
    )
    with pytest.raises(AttributeError):
        p.remove_variable("var1")


@pytest.mark.parametrize("direction", PortType)
@pytest.mark.parametrize("copy_dir", PortType)
def test_Port_copy(PortFactory, direction, copy_dir):
    p: Port = PortFactory(
        "dummy", direction,
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
    assert isinstance(c, type(p))
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
def test_Port_to_dict_with_def(PortClassFactory, direction):
    DummyPort: Type[Port] = PortClassFactory(
        classname="DummyPort",
        variables=[
            get_args("Pt", 101325.0, unit="Pa"),
            get_args("W", 1.0,
                unit="kg/s",
                valid_range=(0.0, 2.0),
                limits=(-5, 3.0),
                desc="my lovely W",
            ),
        ],
    )
    p = DummyPort("dummy", direction)
    port_dict = p.to_dict(with_types=True, value_only=True)
    assert_keys(port_dict, "__class__", "name", "variables")
    assert port_dict["__class__"] == "DummyPort"
    assert port_dict["variables"] == {
        'Pt': 101325.0,
        'W': 1.0,
    }


@pytest.mark.parametrize("source_direction", PortType)
@pytest.mark.parametrize("target_direction", PortType)
@pytest.mark.parametrize("variables, check, expected", [
    # No name conflict
    (
        dict(x=0.1, v=[0.2, 0.1]), True,
        dict(x=0.1, v=[0.2, 0.1]),
    ),
    (
        dict(x=0.1, v=[0.2, 0.1]), False,
        dict(x=0.1, v=[0.2, 0.1]),
    ),
    # Source port with name conflicts: only works with prior check
    (
        dict(x=0.1, y=0.2, z='foo'), True,
        dict(x=0.1, v=[0, 0]),
    ),
    (
        dict(x=0.1, y=0.2, z='foo'), False,
        dict(error=AttributeError),
    ),
    # Source port has fewer variables, with no name conflict: OK with no check
    (
        dict(x=0.1), True,
        dict(x=0.1, v=[0, 0]),
    ),
    (
        dict(x=0.1), False,
        dict(x=0.1, v=[0, 0]),
    ),
])
def test_Port_set_from(source_direction, target_direction, variables, check, expected):
    # Construct source port
    class SourcePort(Port):
        def setup(self):
            for name, value in variables.items():
                self.add_variable(name, copy.copy(value))

    # Construct (x, v) test port `target`
    class TargetPort(Port):
        def setup(self):
            self.add_variable('x')
            self.add_variable('v', np.zeros(2))

    source = SourcePort('source', source_direction)
    target = TargetPort('target', target_direction)

    expected = expected.copy()
    error = expected.pop('error', None)

    if error is None:
        target.set_from(source, check_names=check)
        for name, value in expected.items():
            np.testing.assert_equal(target[name], value), name
    
    else:
        with pytest.raises(error, match=expected.get('match', None)):
            target.set_from(source, check_names=check)


@pytest.mark.parametrize("source_direction", PortType)
@pytest.mark.parametrize("target_direction", PortType)
def test_Port_set_from_peer(source_direction, target_direction):
    class SomePort(Port):
        def setup(self):
            self.add_variable('x')
            self.add_variable('v', np.zeros(2))

    source = SomePort('source', source_direction)
    port1 = SomePort('port1', target_direction)
    port2 = SomePort('port2', target_direction)

    source.x = 3.14
    source.v = np.r_[0.1, 0.2]
    port1.set_from(source)
    port2.set_from(source, copy.copy)

    assert port1.x == source.x
    assert port2.x == source.x
    assert np.array_equal(port1.v, source.v)
    assert np.array_equal(port2.v, source.v)
    assert port1.v is source.v
    assert port2.v is not source.v


@pytest.mark.parametrize("source_direction", PortType)
@pytest.mark.parametrize("target_direction", PortType)
def test_Port_set_from_common(source_direction, target_direction, caplog):
    class XyPort(Port):
        def setup(self):
            self.add_variable('x')
            self.add_variable('y')

    class XyzPort(XyPort):
        """Derived from `XyPort`, with additional variables"""
        def setup(self):
            super().setup()
            self.add_variable('z')

    class AbPort(Port):
        """Incompatible with `XyPort`"""
        def setup(self):
            self.add_variable('a')
            self.add_variable('b')

    source = XyzPort('source', source_direction)
    port1 = XyPort('port1', target_direction)
    port2 = AbPort('port2', target_direction)

    source.set_values(
        x = -1,
        y = 3.14,
        z = np.r_[0.1, 0.2],
    )
    
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        port1.set_from(source)
    assert len(caplog.records) == 0
    assert port1.x == source.x
    assert port1.y == source.y

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        port2.set_from(source)
    assert any(
        re.match("'port2' and 'source' have no common variables", record.message)
        for record in caplog.records
    )


@pytest.mark.parametrize("direction", PortType)
def test_Port_pop_variable(direction):
    class XyPort(Port):
        def setup(self):
            self.add_variable('x')
            self.add_variable('y')

    port = XyPort('source', direction)

    with pytest.raises(NotImplementedError, match="cannot remove variables from fixed-size port"):
        port.pop_variable('x')
