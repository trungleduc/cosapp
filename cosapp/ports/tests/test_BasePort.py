import pytest

from numbers import Number
import logging, re
import numpy as np

from cosapp.ports.variable import Variable
from cosapp.core.numerics.distributions.uniform import Uniform
from cosapp.ports.port import (
    PortType,
    BasePort,
    Validity,
    Scope,
)
from cosapp.ports.units import UnitError
from cosapp.ports.exceptions import ScopeError
from cosapp.systems import System
from cosapp.utils.testing import no_exception, get_args


@pytest.mark.parametrize("direction", PortType)
@pytest.mark.parametrize("name, error", [
    ("a", None),
    ("A", None),
    ("foobar", None),
    ("foo4bar", None),
    ("loveYou2", None),
    ("CamelBack", None),
    ("foo_bar", None),
    ("foobar_", None),
    ("_foobar", ValueError),
    ("foo bar", ValueError),
    ("foobar?", ValueError),
    ("foo.bar", ValueError),
    ("foo:bar", ValueError),
    ("foo/bar", ValueError),
    ("1foobar", ValueError),
    ("foobar-2", ValueError),
    ("foobar:2", ValueError),
    ("foobar.2", ValueError),
    (23, TypeError),
    (1.0, TypeError),
    (dict(a=True), TypeError),
    (list(), TypeError),
])
def test_BasePort__init__(name, direction, error):
    if error is None:
        port = BasePort(name, direction)
        assert port.name == name
        assert port.contextual_name == name
        assert port.direction is direction
        assert port.owner is None
    else:
        with pytest.raises(error):
            BasePort(name, direction)


@pytest.mark.parametrize("direction", PortType)
def test_BasePort__init__var(direction):
    port = BasePort("myPort", direction)
    port.add_variable("var1", 0.0)
    port.add_variable("var2", 100.1)
    port.add_variable("var3", 42.0)
    port.add_variable("var4", np.ones(5))
    assert port.var1 == 0
    assert port.var2 == 100.1
    assert port.var3 == 42
    assert np.array_equal(port["var4"], np.ones(5))


def test_BasePort_is_inOrOut():
    """Test properties `in_input` and `is_output`
    """
    port = BasePort("port", PortType.IN)
    assert port.is_input
    assert not port.is_output

    port = BasePort("port", PortType.OUT)
    assert not port.is_input
    assert port.is_output


@pytest.mark.parametrize("direction", [
    PortType.IN.name, PortType.IN.value,
    PortType.OUT.name, PortType.OUT.value,
    "IN", "OUT", "out", 3.14, [1, 2, 3], dict(),
])
def test_BasePort__init__wrong_dir(direction):
    with pytest.raises(TypeError):
        BasePort("dummy", direction)


@pytest.mark.parametrize("direction", PortType)
def test_BasePort_owner(direction):
    port = BasePort("myPort", direction)
    assert port.owner is None
    assert port.name == "myPort"
    assert port.contextual_name == "myPort"

    system = System("foo")
    port.owner = system
    assert port.owner is system
    assert port.name == "myPort"
    assert port.contextual_name == "foo.myPort"

    with pytest.raises(TypeError):
        port.owner = "blahblah"


@pytest.mark.parametrize("direction", PortType)
@pytest.mark.parametrize("owner", [
    "foo", BasePort("other", PortType.IN), -1, 3.14, [1, 2, 3], dict(),
])
def test_BasePort_owner_error(direction, owner):
    name = "myPort"
    port = BasePort(name, direction)

    with pytest.raises(TypeError):
        port.owner = owner

    assert port.owner is None
    assert port.name == name
    assert port.contextual_name == name


@pytest.mark.parametrize("direction", PortType)
@pytest.mark.parametrize("user_scope, errors", [
    (Scope.PUBLIC, dict(public=None, protected=ScopeError, private=ScopeError)),
    (Scope.PROTECTED, dict(private=ScopeError)),
    (Scope.PRIVATE, dict()),  # users with PRIVATE clearance can modify all vars
])
def test_BasePort_validate(direction, user_scope, errors):
    port = BasePort("myPort", direction)
    port.add_variable("public", 0.1, scope=Scope.PUBLIC)
    port.add_variable("protected", 0.1, scope=Scope.PROTECTED)
    port.add_variable("private", 0.1, scope=Scope.PRIVATE)
    var_names = ("public", "protected", "private")

    # set user's clearance level
    port.scope_clearance = user_scope

    # 1. Validation always passes when port.owner is None
    assert port.owner is None
    with no_exception():
        for key in var_names:
            port.validate(key, 3.14)

    # 2. Scope-dependent validation when port.owner is not None
    port.owner = System("system")
    assert port.owner is not None

    # make sure errors has the proper keys
    for key in var_names:
        errors.setdefault(key, None)

    for key, error in errors.items():
        var = port[key]
        if error is None:
            with no_exception():
                port.validate(key, 3.14)
                port.validate(key, None)
            with pytest.raises(TypeError):
                port.validate(key, [-1, 2.3])
        else:
            with pytest.raises(error):
                port.validate(key, 3.14)


@pytest.mark.parametrize("direction", PortType)
@pytest.mark.parametrize("user_scope, expected", [
    (Scope.PUBLIC, dict(public=False, protected=True, private=True)),
    (Scope.PROTECTED, dict(private=True)),
    (Scope.PRIVATE, dict()),  # users with PRIVATE clearance can modify all vars
])
def test_BasePort_out_of_scope(direction, user_scope, expected):
    port = BasePort("myPort", direction)
    port.add_variable("public", 0.1, scope=Scope.PUBLIC)
    port.add_variable("protected", 0.1, scope=Scope.PROTECTED)
    port.add_variable("private", 0.1, scope=Scope.PRIVATE)
    var_names = ("public", "protected", "private")

    # set user's clearance level
    port.scope_clearance = user_scope

    # 1. Never out-of-scope when port.owner is None
    assert port.owner is None
    for key in var_names:
        assert not port.out_of_scope(key)

    # 2. Scope-dependent access when port.owner is not None
    port.owner = System("system")
    assert port.owner is not None

    # make sure errors has the proper keys
    for key in var_names:
        expected.setdefault(key, False)

    for key, value in expected.items():
        var = port[key]
        assert port.out_of_scope(key) == value, f"variable {key!r}"


@pytest.mark.parametrize("direction", PortType)
def test_BasePort___setattr__(direction):
    port = BasePort("myPort", direction)
    port.add_variable("var1", 0.0)
    port.add_variable("var2", 100.1)

    port.var2 = 42.0
    assert port.var2 == 42.0

    port._dummy = "banana"
    assert port._dummy == "banana"
    with pytest.raises(AttributeError,
        match="variable .* can only be created using method 'add_variable'"):
        port.var3 = 12.0

    # Type test
    port = BasePort("myPort", direction)
    port.add_variable("var1", 2)
    port.var1 = 3.14159
    assert port.var1 == 3.14159
    port.var1 = True
    assert port.var1 == True

    port = BasePort("myPort", direction)
    port.add_variable("var1", "hello", dtype=(int, str))
    port.var1 = "banana"
    assert port.var1 == "banana"
    port.var1 = 23
    assert port.var1 == 23

    port = BasePort("myPort", direction)
    port.add_variable("var1", 2, dtype=int)
    with pytest.raises(TypeError):
        port.var1 = 2.5


@pytest.mark.parametrize("direction", PortType)
@pytest.mark.parametrize("user_scope", Scope)
def test_BasePort_scope_clearance(direction, user_scope):
    """Simple test on getter/setter. Full behaviour tested with method `out_of_scope`"""
    port = BasePort("myPort", direction)
    # set user's clearance level
    port.scope_clearance = user_scope
    assert port.scope_clearance is user_scope


@pytest.mark.parametrize("direction", PortType)
@pytest.mark.parametrize("wrong_value", [
    "PRIVATE", -1, 1.4, Scope.PROTECTED.name, Scope.PROTECTED.value,
])
def test_BasePort_scope_clearance_error(direction, wrong_value):
    """Simple test on getter/setter. Full behaviour tested with method `out_of_scope`"""
    port = BasePort("myPort", direction)
    with pytest.raises(TypeError):
        port.scope_clearance = wrong_value


@pytest.mark.parametrize("direction", PortType)
def test_BasePort_set_type_checking(direction):
    port = BasePort("myPort", direction)
    port.add_variable("var1", 2, dtype=int)

    assert port.var1 == 2
    port.var1 = 4
    assert port.var1 == 4

    pattern = r"Trying to set .* of type <class .*> with <class .*>"

    with pytest.raises(TypeError, match=pattern):
        port.var1 = 2.5

    BasePort.set_type_checking(False)
    port.var1 = 2.5
    assert port.var1 == 2.5

    BasePort.set_type_checking(True)

    with pytest.raises(TypeError, match=pattern):
        port.var1 = 3.14

    port.var1 = -1
    assert port.var1 == -1


@pytest.mark.parametrize("direction", PortType)
def test_BasePort___contains__(direction):
    port = BasePort("myPort", direction)
    port.add_variable("var1", 0.0)
    port.add_variable("var2", 100.1)
    assert "var1" in port
    assert "var2" in port
    assert "var3" not in port


@pytest.mark.parametrize("direction", PortType)
def test_BasePort___getitem__(direction):
    port = BasePort("myPort", direction)
    port.add_variable("x", 0.0)
    port.add_variable("y", 100.1)
    assert port["x"] is port.x
    assert port["y"] is port.y


@pytest.mark.parametrize("direction", PortType)
def test_BasePort___setitem__(direction):
    port = BasePort("myPort", direction)
    port.add_variable("var", 0.0)
    port["var"] = 42.0
    assert port.var == 42.0
    with pytest.raises(KeyError):
        port["banana"] = 0.0


@pytest.mark.parametrize("direction", PortType)
def test_BasePort___iter__(direction):
    port = BasePort("myPort", direction)
    port.add_variable("x", 0.0)
    port.add_variable("y", 100.1)
    assert set(iter(port)) == {"x", "y"}


@pytest.mark.parametrize("direction", PortType)
def test_BasePort___len__(direction):
    port = BasePort("myPort", direction)
    assert len(port) == 0
    port.add_variable("x", 0.0)
    port.add_variable("y", 100.1)
    assert len(port) == 2
    port.add_variable("z", 42.0)
    assert len(port) == 3


@pytest.mark.parametrize("direction", PortType)
@pytest.mark.parametrize("scope", Scope)
@pytest.mark.parametrize("options, expected", [
    (dict(), dict()),
    (dict(value=1.2), dict()),
    (dict(value=1.2, dtype=float), dict(dtype=float)),
    (dict(value=1.2, unit="kg/s"), dict()),
    (dict(value=True), dict(dtype=bool, limits=None, valid_range=None)),
    (dict(value=True, unit=""), dict(dtype=bool, limits=None, valid_range=None)),
    (dict(value=list()), dict(dtype=list, limits=None, valid_range=None)),
    (dict(value=dict()), dict(dtype=dict, limits=None, valid_range=None)),
    (dict(value=set()), dict(dtype=set, limits=None, valid_range=None)),
    (dict(value=np.zeros(2)), dict(dtype=np.ndarray)),
    (dict(value=np.zeros((2, 3))), dict(dtype=np.ndarray)),
    (dict(value=(1, 2)), dict(dtype=tuple)),
    (dict(unit="kg"), dict()),
    (dict(unit=1.2), dict(error=TypeError)),
    (dict(desc=1.2), dict(error=TypeError)),
    (dict(value=1.2, unit="nonsense"), dict(error=UnitError, match="Unknown unit")),
    (dict(value=1.2, dtype=int), dict(error=TypeError)),
    (
        dict(
            value = 0.5,
            unit = "m",
            valid_range = (0.0, 5.0),
            limits = (-5.0, 10.0),
            desc = "my little description",
        ),
        dict()
    ),
    (
        dict(value="hello", dtype=(int, str)),
        dict(limits=None, valid_range=None)  # no ranges for string variables
    ),
    (dict(value=2, dtype=int), dict(dtype=int)),
    (dict(value=2, dtype=(int, str)), dict()),
    (dict(value="hello", dtype=int), dict(error=TypeError)),
    (dict(value=0.123, valid_range=(0.0, 5.0), invalid_comment="Not acceptable"), dict()),
    (dict(value=0.123, valid_range=(0.0, 5.0), limits=(-5.0, 10.0)), dict()),
    ( # None `valid_range` set to prescribed `limits`
        dict(value=0.123, valid_range=None, limits=(0.1, 5.2)),
        dict(valid_range=(0.1, 5.2), limits=(0.1, 5.2))
    ),
    ( # Ranges given as (max, min) should be changed into (min, max)
        dict(value=0.123, valid_range=None, limits=(5.2, 0.1)),
        dict(valid_range=(0.1, 5.2), limits=(0.1, 5.2))
    ),
    ( # Extension of `limits` when not covering `valid_range`
        dict(value=0.123, valid_range=(0.0, 5.0), limits=(1.0, 10.0)),
        dict(valid_range=(0.0, 5.0), limits=(0.0, 10.0))
    ),
    (
        dict(value=0.123, valid_range=(0.0, 5.0), limits=(-5.0, 4.0)),
        dict(valid_range=(0.0, 5.0), limits=(-5.0, 5.0))
    ),
    (
        dict(value=0.123, valid_range=(0.0, 5.0), limits=(1.0, 4.0)),
        dict(valid_range=(0.0, 5.0), limits=(0.0, 5.0))
    ),
    (
        dict(value=0.123, valid_range=(0.0, 5.0), limits=(-5, None)),
        dict(valid_range=(0.0, 5.0), limits=(-5.0, np.inf))
    ),
    (
        dict(value=0.123, valid_range=(0.0, 5.0), limits=(None, 10)),
        dict(valid_range=(0.0, 5.0), limits=(-np.inf, 10))
    ),
    (
        dict(value=0.123, valid_range=(0.0, 5.0), limits=(None, 10)),
        dict(valid_range=(0.0, 5.0), limits=(-np.inf, 10))
    ),
    (
        dict(value="hello", valid_range=(0, 5), limits=(0, 5)),
        dict(limits=None, valid_range=None, dtype=str)
    ),
])
def test_BasePort_get_details(direction, scope, options, expected):
    error = expected.get("error", None)
    port = BasePort("myPort", direction)
    options["scope"] = scope

    if error is None:
        port.add_variable("var", **options)
        assert isinstance(port.get_details("var"), Variable)
        details = port.get_details()
        assert set(details) == {'var'}
        # Check that `details` is immutable
        with pytest.raises(TypeError):
            details["var"] = 0
        with pytest.raises(TypeError):
            details["newkey"] = 0
        assert port.get_details("var") is details["var"]
        assert len(details) == len(port)
        with pytest.raises(KeyError):
            assert port.get_details("foobar")

        default = dict(
            unit = "",
            dtype = (Number, np.ndarray),
            desc = "",
            limits = (-np.inf, np.inf),
            valid_range = (-np.inf, np.inf),
            invalid_comment = "",
            out_of_limits_comment = "",
        )
        default.update(options)
        for key, value in default.items():
            expected.setdefault(key, value)
        detail = port.get_details("var")
        assert detail.unit == expected["unit"]
        assert detail.dtype == expected["dtype"]
        assert detail.valid_range == expected["valid_range"]
        assert detail.limits == expected["limits"]
        assert detail.description == expected["desc"]
        assert detail.scope is scope
        assert detail.distribution is None

    else:
        pattern = expected.get("match", None)
        with pytest.raises(error, match=pattern):
            port.add_variable("var", **options)


@pytest.mark.parametrize("direction", PortType)
def test_BasePort_add_variable(direction, caplog):
    logging.disable(logging.NOTSET)  # enable all logging levels

    port = BasePort("dummy", direction)
    port.add_variable("var1", 0.0)
    port.add_variable("var2", 100.1)
    port.add_variable("var3")
    assert port.var1 == 0.0
    assert port.var2 == 100.1
    assert port.var3 == 1

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        port.add_variable("var3", 1.23)
        assert len(caplog.records) == 1
        assert re.match(
            r"Variable .* already exists in port .*\. It will be overwritten",
            caplog.messages[-1])

    with pytest.raises(TypeError):
        port.add_variable("var4", scope="PRIVATE")


    port = BasePort("myPort", direction)
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        port.add_variable("var1", np.array([1, 2]))
        assert len(caplog.records) == 1
        assert re.match(
            "Variable .* instantiates a numpy array with integer dtype. "
            "This may lead to unpredictible consequences.",
            caplog.messages[-1])

    # Variable with a distribution
    d = Uniform(-1, 2, 0.2)
    port = BasePort("myPort", direction)
    port.add_variable("var", 1.0, distribution=d)
    assert port.get_details("var").distribution is d

    with pytest.raises(TypeError,
            match="Random distribution should be of type 'Distribution'"):
        port.add_variable("foo", 2, distribution="Gaussian")


@pytest.mark.parametrize("direction", PortType)
@pytest.mark.parametrize("scope", Scope)
@pytest.mark.parametrize("options, message", [
    (
        dict(value="hello", unit="kg"),
        "A physical unit is defined for non-numerical variable '.*'; it will be ignored"
    ),
    (
        dict(value=False, unit="kg"),
        "A physical unit is defined for non-numerical variable '.*'; it will be ignored"
    ),
    (
        dict(value="hello", valid_range=(0.0, 5.0), invalid_comment="Not acceptable"),
        "Invalid comment specified for variable '.*' without validity range"
    ),
    (
        dict(value=True, out_of_limits_comment="Not acceptable"),
        "Out-of-limits comment specified for variable '.*' without limits"
    ),
    (
        dict(value=True, invalid_comment="Not acceptable"),
        "Invalid comment specified for variable '.*' without validity range"
    ),
])
def test_BasePort_add_variable_warning(caplog, direction, scope, options, message):
    logging.disable(logging.NOTSET)  # enable all logging levels
    options["scope"] = scope

    port = BasePort("dummy", direction)
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        port.add_variable("var", **options)
        assert len(caplog.records) == 1
        assert re.match(message, caplog.messages[-1])


@pytest.mark.parametrize("direction", PortType)
@pytest.mark.parametrize("scope1", Scope)
@pytest.mark.parametrize("scope2", Scope)
def test_BasePort_add_variable_multiscope(direction, scope1, scope2):
    port = BasePort("dummy", direction)
    port.add_variable("var1", 3.14, scope=scope1)
    port.add_variable("var2", True, scope=scope2)
    assert port.get_details("var1").scope is scope1
    assert port.get_details("var2").scope is scope2


@pytest.mark.parametrize("direction", PortType)
def test_BasePort_check(direction):
    port = BasePort("myPort", direction)
    port.add_variable(
        "var1",
        0.0,
        valid_range=(0.0, 5.0),
        limits=(-5.0, 10.0),
        desc="my little description.",
    )
    port.add_variable(
        "var2",
        -2.0,
        valid_range=(0.0, 5.0),
        limits=(-5.0, 10.0),
        desc="my little description.",
    )
    port.add_variable(
        "var3",
        42.0,
        valid_range=(0.0, 5.0),
        limits=(-5.0, 10.0),
        desc="my little description.",
    )
     
    assert port.check() == {
            "var1": Validity.OK,
            "var2": Validity.WARNING,
            "var3": Validity.ERROR,
    }

    assert port.check("var1") == Validity.OK
    port["var1"] = -1
    assert port.check("var1") == Validity.WARNING
    port["var1"] = 7
    assert port.check("var1") == Validity.WARNING
    port["var1"] = -10
    assert port.check("var1") == Validity.ERROR
    port["var1"] = 20
    assert port.check("var1") == Validity.ERROR


@pytest.mark.parametrize("direction1", PortType)
@pytest.mark.parametrize("direction2", PortType)
def test_BasePort_copy(direction1, direction2):
    port = BasePort("myPort", direction1)
    port.add_variable("var1", 0.0)
    port.add_variable("var2", 100.1)

    p_copy = port.copy()
    assert p_copy.name == port.name
    assert p_copy.direction == port.direction
    assert p_copy.var1 == 0.0
    assert p_copy.var2 == 100.1

    p2 = port.copy("myCopy", direction2)
    assert p2.name == "myCopy"
    assert p2.direction == direction2

    port = BasePort("myPort", direction1)
    port.add_variable("var1", 22.0,
        valid_range=(0.0, 5.0),
        limits=(-5.0, 42.0),
        desc="my stupid description",
    )
    p_copy = port.copy("new", direction2)
    details = p_copy.get_details()
    assert set(details.keys()) == {"var1"}

    assert details["var1"].valid_range == (-np.inf, np.inf)
    assert details["var1"].limits == (-np.inf, np.inf)
    assert details["var1"].description == "my stupid description"

    # Check that copy() copies variable details, except validation parameters
    port = BasePort("myPort", direction1)
    port.add_variable("var1", 0.0,
        valid_range=(1, 2),
        invalid_comment="invalid value",
        limits=(0, 4),
        out_of_limits_comment="too far",
        scope=Scope.PROTECTED,
        desc="banana",
    )
    port.add_variable("var2", True)

    p_copy = port.copy(direction=direction2)
    details = p_copy.get_details()
    assert set(details.keys()) == {"var1", "var2"}

    assert details["var1"].valid_range == (-np.inf, np.inf)
    assert details["var1"].limits == (-np.inf, np.inf)
    assert details["var1"].description == "banana"
    assert details["var1"].invalid_comment == ""
    assert details["var1"].out_of_limits_comment == ""
    assert details["var1"].scope == Scope.PROTECTED

    assert details["var2"].valid_range is None
    assert details["var2"].limits is None
    assert details["var2"].description == ""
    assert details["var2"].invalid_comment == ""
    assert details["var2"].out_of_limits_comment == ""
    assert details["var2"].scope == Scope.PRIVATE


@pytest.mark.parametrize("direction, expected", [
    (PortType.IN, {"dummy.x": 1.5, "dummy.y": 0.2}),
    (PortType.OUT, {}),
])
def test_BasePort_to_dict(direction, expected):
    port = BasePort("dummy", direction)
    port.add_variable("x", 1.5)
    port.add_variable("y", 0.2)
    assert port.to_dict() == expected

@pytest.mark.parametrize("direction, expected", [
    (PortType.IN, {'dummy': {'__class__': 'BasePort', 'x': 1.5, 'y': 0.2}}),
    (PortType.OUT, {'dummy': {'__class__': 'BasePort', 'x': 1.5, 'y': 0.2}}),
])
def test_BasePort_to_dict_with_def(direction, expected):
    port = BasePort("dummy", direction)
    port.add_variable("x", 1.5)
    port.add_variable("y", 0.2)
    assert port.to_dict(True) == expected

@pytest.mark.skip(reason="TODO")
def test_BasePort___json__():
    pytest.fail()


@pytest.mark.skip(reason="TODO")
def test_BasePort_serialize_data():
    pytest.fail()


@pytest.mark.parametrize("direction", PortType)
@pytest.mark.parametrize("data, expected", [
    (
        [get_args('x', 0.1)],
        "{'x': 0.1}"
    ),
    (
        [get_args('x', 0.1, scope=Scope.PROTECTED, unit="kg", limits=(0, 10))],
        "{'x': 0.1}"
    ),
    (
        [
            get_args('x', 0.1),
            get_args('y', [0.1, 0.2]),
        ],
        "{'x': 0.1, 'y': array([0.1, 0.2])}"
    ),
    (
        [
            get_args('y', [0.1, 0.2]),
            get_args('x', 0.1),
        ],
        "{'y': array([0.1, 0.2]), 'x': 0.1}"
    ),
    (
        [
            get_args('x', 0.1),
            get_args('y', [0.1, 0.2]),
            get_args('z', True, dtype=bool),
        ],
        "{'x': 0.1, 'y': array([0.1, 0.2]), 'z': True}"
    ),
])
def test_BasePort___repr__(direction, data, expected):
    port = BasePort("dummy", direction)
    for args, kwargs in data:
        port.add_variable(*args, **kwargs)
    assert repr(port) == f"BasePort: {expected}"


@pytest.mark.skip(reason="TODO")
def test_BasePort_morph():
    pytest.fail()
