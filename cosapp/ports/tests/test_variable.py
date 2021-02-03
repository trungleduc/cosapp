import logging
import re
from numbers import Number

from unittest import mock

import numpy as np
import pytest

from cosapp.ports.variable import Variable
from cosapp.ports.port import ExtensiblePort
from cosapp.core.numerics.distributions.distribution import Distribution
from cosapp.core.numerics.distributions.uniform import Uniform
from cosapp.ports.enum import Scope, Validity
from cosapp.ports.units import UnitError
from cosapp.utils.testing import  get_args


@pytest.fixture(scope='function')
def port():
    return mock.Mock(spec=ExtensiblePort)


@pytest.fixture(scope='function')
def plainvar(port):
    name = 'plain'
    value = 2.0
    setattr(port, name, value)
    return Variable(name, port, value)


@pytest.mark.parametrize("value, expected", [
    (True, None),
    (1, (-np.inf, np.inf)),
    (1.0e-5, (-np.inf, np.inf)),
    (np.asarray(1.0e-5), (-np.inf, np.inf)),
    ("string", None),
    ("", None),
    ([1, 2, 3], (-np.inf, np.inf)),
     (["a", "b", "c"], None),
    ([1, 2, "c"], None),
    ([], None),
    ([[]], None),
    ((1, 2, 3), (-np.inf, np.inf)), 
    (("a", "b", "c"), None),
    ((1, 2, "c"), None),    
    ((), None),  
    ({1, 2, 3}, (-np.inf, np.inf)), 
    ({"a", "b", "c"}, None),
    ({1, 2, "c"}, None),    
    ({}, None), 
    (frozenset([1, 2, 3]), (-np.inf, np.inf)),
    (frozenset(["a", "b", "c"]), None),
    (frozenset([1, 2, "c"]), None),
    (frozenset([]), None),
    ({"a": 1, "b": 2, "c": 3}, None),
    (np.ones(4), (-np.inf, np.inf)),
    (np.asarray(["a", "b", "c"]), None),
    (np.asarray([], dtype=float), (-np.inf, np.inf)),
    (np.asarray([], dtype=int), (-np.inf, np.inf)),
    (np.asarray([], dtype=str), None),
    ])
def test_Variable__get_limits_from_type(plainvar, value, expected):
    assert plainvar._get_limits_from_type(value) == expected


@pytest.mark.parametrize("limits, valid, value, expected", [
    (None, (0.0, 5.0), 0.0, ((-np.inf, np.inf), (0.0, 5.0))),
    (None, (5.0, 0.0), 0.0, ((-np.inf, np.inf), (0.0, 5.0))),
    (None, (0.0, None), 0.0, ((-np.inf, np.inf), (0.0, np.inf))),
    (None, (None, 5.0), 0.0, ((-np.inf, np.inf), (-np.inf, 5.0))),
    (None, (0.0, 5.0), "dummy string", (None, None)),
    ((0.0, 5.0), None, 0.0, ((0.0, 5.0), (0.0, 5.0))),
    ((-5.0, 10.0), (0.0, 5.0), 0.0, ((-5.0, 10.0), (0.0, 5.0))),
    ((1.0, 10.0), (0.0, 5.0), 0.0, ((0.0, 10.0), (0.0, 5.0))),
    ((-5.0, 4.0), (0.0, 5.0), 0.0, ((-5.0, 5.0), (0.0, 5.0))),
    ((1.0, 4.0), (0.0, 5.0), 0.0, ((0.0, 5.0), (0.0, 5.0))),
    ((-5, None), (0.0, 5.0), 0.0, ((-5, np.inf), (0.0, 5.0))),
    ((None, 10.0), (0.0, 5.0), 0.0, ((-np.inf, 10.0), (0.0, 5.0))),
    ((0.0, 5.0), (0.0, 5.0), "dummy string",  (None, None)),
])
def test_Variable__check_range(plainvar, limits, valid, value, expected):
    # Test validity range
    assert plainvar._check_range(limits, valid, value) == expected 


def test_Variable___init__(port, caplog):
    name = "var1"
    value = 2.0
    setattr(port, name, value)

    # Minimal constructor
    v = Variable(name, port, value)
    for a, b in (
        {
            "name": name,
            "unit": "",
            "dtype": (Number, np.ndarray),
            "valid_range": (-np.inf, np.inf),
            "invalid_comment": "",
            "limits": (-np.inf, np.inf),
            "out_of_limits_comment": "",
            "description": "",
            "scope": Scope.PRIVATE,
            "distribution": None,
        }
    ).items():
        assert getattr(v, a) == b

    # Full constructor
    w = Variable(
        name,
        port,
        value,
        unit="kg",
        dtype=float,
        valid_range=(-2, 0),
        invalid_comment="No valid",
        limits=(-4, 1),
        out_of_limits_comment="No so far!",
        desc="I'm a dummy donkey.",
        scope=Scope.PUBLIC,
        distribution=Uniform(1.0, 4.0, 0.2),
    )
    for a, b in (
        {
            "name": name,
            "unit": "kg",
            "dtype": float,
            "valid_range": (-2, 0),
            "invalid_comment": "No valid",
            "limits": (-4, 1),
            "out_of_limits_comment": "No so far!",
            "description": "I'm a dummy donkey.",
            "scope": Scope.PUBLIC,
        }
    ).items():
        assert getattr(w, a) == b
    assert isinstance(w.distribution, Distribution)

    # Test physical unit
    v = Variable(name, port, value, unit="")
    assert v.unit == ""

    v = Variable(name, port, value, unit="kg/s")
    assert v.unit == "kg/s"

    with pytest.raises(UnitError, match=r"Unknown unit [\w/]+."):
        Variable(name, port, value, unit="kg/gh")

    # Test unit for boolean
    port = mock.Mock(spec=ExtensiblePort)
    name = "var1"
    value = True
    setattr(port, name, value)

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        v = Variable(name, port, value, unit="kg/s")

    assert v.unit == ""
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelno == logging.WARNING
    expected_msg = r"A physical unit is defined for non-numerical variable '\w+'; it will be ignored."
    assert re.match(expected_msg, record.message)

    # Test unit for string
    port = mock.Mock(spec=ExtensiblePort)
    name = "var1"
    value = "hello"
    setattr(port, name, value)
    v = Variable(name, port, value, unit="")
    assert v.unit == ""

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        v = Variable(name, port, value, unit="m")
    assert v.unit == ""
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelno == logging.WARNING
    expected_msg = r"A physical unit is defined for non-numerical variable '\w+'; it will be ignored."
    assert re.match(expected_msg, record.message)

    # Test valid_range
    port = mock.Mock(spec=ExtensiblePort)
    name = "var1"
    value = 0.0
    setattr(port, name, value)
    v = Variable(name, port, value, valid_range=(0.0, 5.0))
    for a, b in ({"valid_range": (0.0, 5.0), "limits": (-np.inf, np.inf)}).items():
        assert getattr(v, a) == b

    v = Variable(name, port, value, valid_range=(5.0, 0.0))
    for a, b in ({"valid_range": (0.0, 5.0), "limits": (-np.inf, np.inf)}).items():
        assert getattr(v, a) == b

    v = Variable(name, port, value, valid_range=(0.0, None))
    for a, b in ({"valid_range": (0.0, np.inf), "limits": (-np.inf, np.inf)}).items():
        assert getattr(v, a) == b

    v = Variable(name, port, value, valid_range=(None, 5.0))
    for a, b in ({"valid_range": (-np.inf, 5.0), "limits": (-np.inf, np.inf)}).items():
        assert getattr(v, a) == b

    port = mock.Mock(spec=ExtensiblePort)
    name = "var1"
    value = "dummy string"
    setattr(port, name, value)
    v = Variable(name, port, value, valid_range=(0.0, 5.0))
    for a, b in ({"valid_range": None, "limits": None}).items():
        assert getattr(v, a) == b

    # Test invalid_comment
    port = mock.Mock(spec=ExtensiblePort)
    name = "var1"
    value = 0.0
    setattr(port, name, value)
    v = Variable(
        name, port, value, valid_range=(0.0, 5.0), invalid_comment="Not acceptable"
    )
    assert v.invalid_comment == "Not acceptable"

    port = mock.Mock(spec=ExtensiblePort)
    name = "var1"
    value = True
    setattr(port, name, value)
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        v = Variable(name, port, value, invalid_comment="Not acceptable")
    assert v.invalid_comment == "Not acceptable"
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelno == logging.WARNING
    expected_msg = (
        r"Invalid comment specified for variable '\w+' without validity range."
    )
    assert re.match(expected_msg, record.message)

    port = mock.Mock(spec=ExtensiblePort)
    name = "var1"
    value = ""
    setattr(port, name, value)
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        v = Variable(
            name, port, value, valid_range=(0.0, 5.0), invalid_comment="Not acceptable"
        )
    assert v.invalid_comment == "Not acceptable"
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelno == logging.WARNING
    expected_msg = (
        r"Invalid comment specified for variable '\w+' without validity range."
    )
    assert re.match(expected_msg, record.message)

    # Test limits
    port = mock.Mock(spec=ExtensiblePort)
    name = "var1"
    value = 0.0
    setattr(port, name, value)
    v = Variable(name, port, value, valid_range=None, limits=(0.0, 5.0))
    for a, b in ({"valid_range": (0.0, 5.0), "limits": (0.0, 5.0)}).items():
        assert getattr(v, a) == b

    v = Variable(name, port, value, valid_range=None, limits=(5.0, 0.0))
    for a, b in ({"valid_range": (0.0, 5.0), "limits": (0.0, 5.0)}).items():
        assert getattr(v, a) == b

    v = Variable(name, port, value, valid_range=(0.0, 5.0), limits=(-5.0, 10.0))
    for a, b in ({"valid_range": (0.0, 5.0), "limits": (-5.0, 10.0)}).items():
        assert getattr(v, a) == b

    v = Variable(name, port, value, valid_range=(0.0, 5.0), limits=(1.0, 10.0))
    for a, b in ({"valid_range": (0.0, 5.0), "limits": (0.0, 10.0)}).items():
        assert getattr(v, a) == b

    v = Variable(name, port, value, valid_range=(0.0, 5.0), limits=(-5.0, 4.0))
    for a, b in ({"valid_range": (0.0, 5.0), "limits": (-5.0, 5.0)}).items():
        assert getattr(v, a) == b

    v = Variable(name, port, value, valid_range=(0.0, 5.0), limits=(1.0, 4.0))
    for a, b in ({"valid_range": (0.0, 5.0), "limits": (0.0, 5.0)}).items():
        assert getattr(v, a) == b

    v = Variable(name, port, value, valid_range=(0.0, 5.0), limits=(-5, None))
    for a, b in ({"valid_range": (0.0, 5.0), "limits": (-5.0, np.inf)}).items():
        assert getattr(v, a) == b

    v = Variable(name, port, value, valid_range=(0.0, 5.0), limits=(None, 10.0))
    for a, b in ({"valid_range": (0.0, 5.0), "limits": (-np.inf, 10.0)}).items():
        assert getattr(v, a) == b

    port = mock.Mock(spec=ExtensiblePort)
    name = "var1"
    value = "dummy string"
    setattr(port, name, value)
    v = Variable(name, port, value, valid_range=(0.0, 5.0), limits=(0.0, 5.0))
    for a, b in ({"valid_range": None, "limits": None}).items():
        assert getattr(v, a) == b

    # Test out_of_limits_comment
    port = mock.Mock(spec=ExtensiblePort)
    name = "var1"
    value = 1.0
    setattr(port, name, value)
    v = Variable(
        name, port, value, limits=(0.0, 5.0), out_of_limits_comment="Not acceptable"
    )
    assert v.out_of_limits_comment == "Not acceptable"

    port = mock.Mock(spec=ExtensiblePort)
    name = "var1"
    value = True
    setattr(port, name, value)
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        v = Variable(
            name, port, value, limits=(0.0, 5.0), out_of_limits_comment="Not acceptable"
        )
    assert v.out_of_limits_comment == "Not acceptable"
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelno == logging.WARNING
    expected_msg = r"Out-of-limits comment specified for variable '\w+' without limits."
    assert re.match(expected_msg, record.message)

    port = mock.Mock(spec=ExtensiblePort)
    name = "var1"
    value = ""
    setattr(port, name, value)
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        v = Variable(
            name, port, value, limits=(0.0, 5.0), out_of_limits_comment="Not acceptable"
        )
    assert v.out_of_limits_comment == "Not acceptable"
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelno == logging.WARNING
    expected_msg = r"Out-of-limits comment specified for variable '\w+' without limits."
    assert re.match(expected_msg, record.message)

    # Test description
    port = mock.Mock(spec=ExtensiblePort)
    name = "var1"
    value = 2.5
    setattr(port, name, value)
    v = Variable(name, port, value, desc="my stupid description")
    for a, b in (
        {
            "valid_range": (-np.inf, np.inf),
            "limits": (-np.inf, np.inf),
            "description": "my stupid description",
        }
    ).items():
        assert getattr(v, a) == b
    with pytest.raises(TypeError):
        Variable(name, port, value, desc=42.0)

    # Test scope
    port = mock.Mock(spec=ExtensiblePort)
    name = "var1"
    value = 2.5
    setattr(port, name, value)
    v = Variable(name, port, value, scope=Scope.PRIVATE)
    assert v.scope == Scope.PRIVATE

    port = mock.Mock(spec=ExtensiblePort)
    name = "var1"
    value = 2.5
    setattr(port, name, value)
    v = Variable(name, port, value, scope=Scope.PROTECTED)
    assert v.scope == Scope.PROTECTED

    port = mock.Mock(spec=ExtensiblePort)
    name = "var1"
    value = 2.5
    setattr(port, name, value)
    v = Variable(name, port, value, scope=Scope.PUBLIC)
    assert v.scope == Scope.PUBLIC

    with pytest.raises(TypeError):
        Variable(name, port, value, scope="PRIVATE")

    # Test dtype
    port = mock.Mock(spec=ExtensiblePort)
    name = "var1"
    value = -2
    setattr(port, name, value)
    v = Variable(name, port, value)
    assert v.dtype == (Number, np.ndarray)

    port = mock.Mock(spec=ExtensiblePort)
    name = "var1"
    value = 2.5
    setattr(port, name, value)
    v = Variable(name, port, value)
    assert v.dtype == (Number, np.ndarray)

    port = mock.Mock(spec=ExtensiblePort)
    name = "var1"
    value = True
    setattr(port, name, value)
    v = Variable(name, port, value)
    assert v.dtype == bool

    port = mock.Mock(spec=ExtensiblePort)
    name = "var1"
    value = ""
    setattr(port, name, value)
    v = Variable(name, port, value)
    assert v.dtype == str

    port = mock.Mock(spec=ExtensiblePort)
    name = "var1"
    value = list()
    setattr(port, name, value)
    v = Variable(name, port, value)
    assert v.dtype == list

    port = mock.Mock(spec=ExtensiblePort)
    name = "var1"
    value = tuple()
    setattr(port, name, value)
    v = Variable(name, port, value)
    assert v.dtype == tuple

    port = mock.Mock(spec=ExtensiblePort)
    name = "var1"
    value = set()
    setattr(port, name, value)
    v = Variable(name, port, value)
    assert v.dtype == set

    port = mock.Mock(spec=ExtensiblePort)
    name = "var1"
    value = dict()
    setattr(port, name, value)
    v = Variable(name, port, value)
    assert v.dtype == dict

    port = mock.Mock(spec=ExtensiblePort)
    name = "var1"
    value = np.asarray([1, 2, 3])
    setattr(port, name, value)
    v = Variable(name, port, value)
    assert v.dtype == np.ndarray

    port = mock.Mock(spec=ExtensiblePort)
    name = "var1"
    value = 2.5
    setattr(port, name, value)
    v = Variable(name, port, value, dtype=float)
    assert v.dtype == float

    port = mock.Mock(spec=ExtensiblePort)
    name = "var1"
    value = 2
    setattr(port, name, value)
    v = Variable(name, port, value, dtype=int)
    assert v.dtype == int

    port = mock.Mock(spec=ExtensiblePort)
    name = "var1"
    value = 2.5
    setattr(port, name, value)
    with pytest.raises(TypeError, match=r"Cannot set .+ of type \w+ with a \w+"):
        Variable(name, port, value, dtype=int)

    port = mock.Mock(spec=ExtensiblePort)
    name = "var1"
    value = np.r_[1]
    setattr(port, name, value)
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        Variable(name, port, value)
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelno == logging.WARNING
    expected_msg = (
        r"Variable '\w+' instantiates a numpy array with integer dtype. "
        r"This may lead to unpredictible consequences."
    )
    assert re.match(expected_msg, record.message)

    port = mock.Mock(spec=ExtensiblePort)
    name = "var1"
    value = np.r_[1.0]
    setattr(port, name, value)
    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        Variable(name, port, value)
    assert len(caplog.records) == 0

    # Test distribution
    port = mock.Mock(spec=ExtensiblePort)
    name = "var1"
    value = 0.0
    setattr(port, name, value)
    v = Variable(name, port, value, distribution=None)
    assert v.distribution == None

    d = Uniform(-1, 2, 0.2)
    v = Variable(name, port, value, distribution=d)
    assert v.distribution is d

    with pytest.raises(
        TypeError,
        match=r"Random distribution should be of type 'Distribution'; got [\w\.]+.",
    ):
        Variable(name, port, value, distribution="Gaussian")


def test_Variable___str__(port):
    name = "var1"
    value = 2.0
    setattr(port, name, value)

    v = Variable(name, port, value)
    assert str(v) == name


@pytest.mark.parametrize("data, expected", [
    (
        get_args(), "var1 &#128274;&#128274; : 2"
    ),(
        get_args (unit="kg",
        dtype=float,
        valid_range=(-2, 0),
        invalid_comment="No valid",
        limits=(-4, 1),
        out_of_limits_comment="No so far!",
        desc="I'm a dummy donkey.",
        scope=Scope.PROTECTED,
        distribution=Uniform(1.0, 4.0, 0.2)),
        "var1 &#128274; : 2 kg;  &#10647; -4 &#10205; -2 &#10205;  value  &#10206; 0 &#10206; 1 &#10648;  # I'm a dummy donkey."
    )
])
def test_Variable___repr__(port, data, expected):
    name = "var1"
    value = 2.0
    setattr(port, name, value)
    v = Variable(name, port, value, **data[1])
    assert repr(v) == expected 


@pytest.mark.parametrize("data, expected", [
    (
        get_args(),
        {
            "value": 2.0,
            "valid_range": (-np.inf, np.inf),
            "invalid_comment": "",
            "limits": (-np.inf, np.inf),
            "out_of_limits_comment": "",
            "distribution": None,
        }
    ),(
        get_args (unit="kg",
        dtype=float,
        valid_range=(-2, 0),
        invalid_comment="No valid",
        limits=(-4, 1),
        out_of_limits_comment="No so far!",
        desc="I'm a dummy donkey.",
        scope=Scope.PROTECTED,
        distribution=Uniform(1.0, 4.0, 0.2)),
        {
            "value": 2.0,
            "valid_range": (-2, 0),
            "invalid_comment": "No valid",
            "limits": (-4, 1),
            "out_of_limits_comment": "No so far!",
            "distribution":{ "worst": 1.0, "pworst": 0.2, "best": 4.0, "pbest": 0.15 },
        }
        
    )
])
def test_Variable___json__(port, data, expected):
    name = "var1"
    value = 2.0
    setattr(port, name, value)

    v = Variable(name, port, value,**data[1])
    assert v.__json__() == expected


@pytest.mark.parametrize("data, expected", [
    (
        get_args(),
        {"value": 2.0}
    ),(
        get_args (unit="kg",
        dtype=float,
        valid_range=(-2, 0),
        invalid_comment="No valid",
        limits=(-4, 1),
        out_of_limits_comment="No so far!",
        desc="I'm a dummy donkey.",
        scope=Scope.PROTECTED,
        distribution=Uniform(1.0, 4.0, 0.2)),
        {
            "value": 2.0,
            "unit": "kg",
            "invalid_comment": "No valid",
            "out_of_limits_comment": "No so far!",
            "desc": "I'm a dummy donkey.",
            "distribution": { "worst": 1.0, "pworst": 0.2, "best": 4.0, "pbest": 0.15 },
            "valid_range": [-2, 0],
            'limits': [-4, 1]
        }
    )
])
def test_Variable_to_dict(port, data, expected):
    name = "var1"
    value = 2.0  
    setattr(port, name, value)
    w1 = Variable(name, port, value, **data[1])    
    assert w1.to_dict() == expected


@pytest.mark.parametrize("name, expected", [
    ("var1", dict()),
    ("&var", dict(error=ValueError)),
    ("foo.bar", dict(error=ValueError)),
    ("time", dict(error=ValueError, match="reserved")),
    ("inwards", dict(error=ValueError, match="invalid")),
    ("outwards", dict(error=ValueError, match="invalid")),
    (3.14159, dict(error=TypeError)),
])
def test_Variable_name(port, name, expected):
    value = 2.0
    setattr(port, str(name), value)
    error = expected.get('error', None)
    
    if error is None:
        v = Variable(name, port, value)
        assert v.name == name

    else:
        with pytest.raises(error, match=expected.get('match', None)):
            Variable(name, port, value)


def test_Variable_name_change(port):
    name = "var1"
    value = 2.0
    setattr(port, name, value)

    v = Variable(name, port, value)
    assert v.name == name
    with pytest.raises(AttributeError):
        v.name = "hello"


def test_Variable_unit(port):
    name = "var1"
    value = 2.0
    setattr(port, name, value)

    v = Variable(name, port, value)
    with pytest.raises(AttributeError):
        v.unit = "Pa"


def test_Variable_description(port):
    name = "var1"
    value = 2.0
    setattr(port, name, value)

    v = Variable(name, port, value)
    with pytest.raises(AttributeError):
        v.description = "This is the most beautiful"


def test_Variable_scope(port):
    name = "var1"
    value = 2.0
    setattr(port, name, value)

    v = Variable(name, port, value)
    with pytest.raises(AttributeError):
        v.scope = Scope.PUBLIC


def test_Variable_valid_range():
    port = mock.Mock(spec=ExtensiblePort)
    name = "var1"
    value = 2.0
    setattr(port, name, value)

    v = Variable(name, port, value)
    for a, b in (
        {"limits": (-np.inf, np.inf), "valid_range": (-np.inf, np.inf)}
    ).items():
        assert getattr(v, a) == b

    v.valid_range = (0.0, 5.0)
    for a, b in ({"limits": (-np.inf, np.inf), "valid_range": (0.0, 5.0)}).items():
        assert getattr(v, a) == b
    v.valid_range = (None, None)
    for a, b in (
        {"limits": (-np.inf, np.inf), "valid_range": (-np.inf, np.inf)}
    ).items():
        assert getattr(v, a) == b

    v = Variable(name, port, value)
    v.valid_range = (0.0, 5.0)
    v.valid_range = None
    for a, b in (
        {"limits": (-np.inf, np.inf), "valid_range": (-np.inf, np.inf)}
    ).items():
        assert getattr(v, a) == b

    v = Variable(name, port, value)
    v.valid_range = (5.0, 0.0)
    for a, b in ({"limits": (-np.inf, np.inf), "valid_range": (0.0, 5.0)}).items():
        assert getattr(v, a) == b

    v = Variable(name, port, value)
    v.valid_range = (0.0, None)
    for a, b in ({"limits": (-np.inf, np.inf), "valid_range": (0.0, np.inf)}).items():
        assert getattr(v, a) == b

    v = Variable(name, port, value)
    v.valid_range = (None, 5.0)
    for a, b in ({"limits": (-np.inf, np.inf), "valid_range": (-np.inf, 5.0)}).items():
        assert getattr(v, a) == b

    port = mock.Mock(spec=ExtensiblePort)
    name = "var1"
    value = "dummy string"
    setattr(port, name, value)
    v = Variable(name, port, value)
    v.valid_range = (0.0, 5.0)
    for a, b in ({"limits": None, "valid_range": None}).items():
        assert getattr(v, a) == b

    # With limits
    port = mock.Mock(spec=ExtensiblePort)
    name = "var1"
    value = 2.0
    setattr(port, name, value)

    v = Variable(name, port, value, limits=(1.0, 4.0))
    for a, b in ({"limits": (1.0, 4.0), "valid_range": (1.0, 4.0)}).items():
        assert getattr(v, a) == b

    v.valid_range = (0.0, 5.0)
    for a, b in ({"limits": (0.0, 5.0), "valid_range": (0.0, 5.0)}).items():
        assert getattr(v, a) == b


def test_Variable_invalid_comment(port):
    name = "var1"
    value = 2.0
    setattr(port, name, value)

    v = Variable(name, port, value, invalid_comment="Based comment")
    assert v.invalid_comment == "Based comment"
    v.invalid_comment = "This is really bad"
    assert v.invalid_comment == "This is really bad"


def test_Variable_limits():
    port = mock.Mock(spec=ExtensiblePort)
    name = "var1"
    value = 0.0
    setattr(port, name, value)
    v = Variable(name, port, value)
    for a, b in (
        {"limits": (-np.inf, np.inf), "valid_range": (-np.inf, np.inf)}
    ).items():
        assert getattr(v, a) == b

    v.limits = (0.0, 5.0)
    for a, b in ({"valid_range": (0.0, 5.0), "limits": (0.0, 5.0)}).items():
        assert getattr(v, a) == b

    v.limits = (-3.0, 7.0)
    for a, b in ({"valid_range": (0.0, 5.0), "limits": (-3.0, 7.0)}).items():
        assert getattr(v, a) == b

    v = Variable(name, port, value)
    v.limits = (5.0, 0.0)
    for a, b in ({"valid_range": (0.0, 5.0), "limits": (0.0, 5.0)}).items():
        assert getattr(v, a) == b

    v = Variable(name, port, value, valid_range=(0.0, 5.0))
    v.limits = (-5.0, 10.0)
    for a, b in ({"valid_range": (0.0, 5.0), "limits": (-5.0, 10.0)}).items():
        assert getattr(v, a) == b

    v = Variable(name, port, value, valid_range=(0.0, 5.0))
    v.limits = (1.0, 10.0)
    for a, b in ({"valid_range": (1.0, 5.0), "limits": (1.0, 10.0)}).items():
        assert getattr(v, a) == b

    v = Variable(name, port, value, valid_range=(0.0, 5.0))
    v.limits = (-5.0, 4.0)
    for a, b in ({"valid_range": (0.0, 4.0), "limits": (-5.0, 4.0)}).items():
        assert getattr(v, a) == b

    v = Variable(name, port, value, valid_range=(0.0, 5.0))
    v.limits = (1.0, 4.0)
    for a, b in ({"valid_range": (1.0, 4.0), "limits": (1.0, 4.0)}).items():
        assert getattr(v, a) == b

    v = Variable(name, port, value, valid_range=(-5.0, 5.0))
    v.limits = (-2.0, None)
    for a, b in ({"valid_range": (-2.0, 5.0), "limits": (-2.0, np.inf)}).items():
        assert getattr(v, a) == b

    v = Variable(name, port, value, valid_range=(-5.0, 5.0))
    v.limits = (None, 2.0)
    for a, b in ({"valid_range": (-5.0, 2.0), "limits": (-np.inf, 2.0)}).items():
        assert getattr(v, a) == b

    v = Variable(name, port, value, valid_range=(0.0, 5.0))
    v.limits = (-5, None)
    for a, b in ({"valid_range": (0.0, 5.0), "limits": (-5.0, np.inf)}).items():
        assert getattr(v, a) == b

    v = Variable(name, port, value, valid_range=(0.0, 5.0))
    v.limits = (None, 10.0)
    for a, b in ({"valid_range": (0.0, 5.0), "limits": (-np.inf, 10.0)}).items():
        assert getattr(v, a) == b

    v = Variable(name, port, value, valid_range=(0.0, 5.0))
    v.limits = (None, None)
    for a, b in ({"valid_range": (0.0, 5.0), "limits": (-np.inf, np.inf)}).items():
        assert getattr(v, a) == b

    v = Variable(name, port, value, valid_range=(0.0, 5.0))
    v.limits = None
    for a, b in ({"valid_range": (0.0, 5.0), "limits": (-np.inf, np.inf)}).items():
        assert getattr(v, a) == b

    port = mock.Mock(spec=ExtensiblePort)
    name = "var1"
    value = "dummy string"
    setattr(port, name, value)
    v = Variable(name, port, value, valid_range=(0.0, 5.0))
    v.limits = (0.0, 5.0)
    for a, b in ({"valid_range": None, "limits": None}).items():
        assert getattr(v, a) == b


def test_Variable_out_of_limits_comment(port):
    name = "var1"
    value = 2.0
    setattr(port, name, value)

    v = Variable(name, port, value, out_of_limits_comment="Based comment")
    assert v.out_of_limits_comment == "Based comment"
    v.out_of_limits_comment = "This is really bad"
    assert v.out_of_limits_comment == "This is really bad"


def test_Variable_distribution(port):
    name = "var1"
    value = 2.0
    setattr(port, name, value)
    v = Variable(name, port, value)
    assert v.distribution is None

    d = Uniform(0.0, 1.0)
    v.distribution = d
    assert v.distribution is d
    v.distribution = None
    assert v.distribution is None


def test_Variable_is_valid():
    port = mock.Mock(spec=ExtensiblePort)
    name = "var1"
    value = 2.0
    setattr(port, name, value)
    v = Variable(
        name,
        port,
        value,
        invalid_comment="Not valid",
        out_of_limits_comment="Get out now!",
    )

    assert v.is_valid() == Validity.OK

    v = Variable(
        name,
        port,
        value,
        valid_range=(1.0, 3.0),
        invalid_comment="Not valid",
        limits=(0.0, 4.0),
        out_of_limits_comment="Get out now!",
    )

    assert v.is_valid() == Validity.OK
    setattr(port, name, 0.5)
    assert v.is_valid() == Validity.WARNING
    setattr(port, name, 3.5)
    assert v.is_valid() == Validity.WARNING
    setattr(port, name, -0.5)
    assert v.is_valid() == Validity.ERROR
    setattr(port, name, 4.5)
    assert v.is_valid() == Validity.ERROR

    port = mock.Mock(spec=ExtensiblePort)
    name = "var1"
    value = True
    setattr(port, name, value)
    v = Variable(
        name,
        port,
        value,
        invalid_comment="Not valid",
        out_of_limits_comment="Get out now!",
    )

    assert v.is_valid() == Validity.OK


def test_Variable_get_validity_comment():
    port = mock.Mock(spec=ExtensiblePort)
    name = "var1"
    value = 2.0
    setattr(port, name, value)
    v = Variable(
        name,
        port,
        value,
        invalid_comment="Not valid",
        out_of_limits_comment="Get out now!",
    )

    assert v.get_validity_comment(Validity.OK) == ""
    assert v.get_validity_comment(Validity.WARNING) == "[-inf, inf] - Not valid"
    assert v.get_validity_comment(Validity.ERROR) == "[-inf, inf] - Get out now!"

    v = Variable(
        name,
        port,
        value,
        valid_range=(1.0, 3.0),
        invalid_comment="Not valid",
        limits=(0.0, 4.0),
        out_of_limits_comment="Get out now!",
    )

    assert v.get_validity_comment(Validity.OK) == ""
    assert v.get_validity_comment(Validity.WARNING) == "[1.0, 3.0] - Not valid"
    assert v.get_validity_comment(Validity.ERROR) == "[0.0, 4.0] - Get out now!"

    port = mock.Mock(spec=ExtensiblePort)
    name = "var1"
    value = True
    setattr(port, name, value)
    v = Variable(
        name,
        port,
        value,
        invalid_comment="Not valid",
        out_of_limits_comment="Get out now!",
    )

    assert v.get_validity_comment(Validity.OK) == ""
    assert v.get_validity_comment(Validity.WARNING) == "] , [ - Not valid"
    assert v.get_validity_comment(Validity.ERROR) == "] , [ - Get out now!"
