import pytest
import logging
import re
import numpy as np
from numbers import Number
from unittest import mock
from contextlib import nullcontext as does_not_raise

from cosapp.systems import System
from cosapp.ports.mode_variable import ModeVariable
from cosapp.ports import ModeVarPort
from cosapp.ports.enum import Scope
from cosapp.ports.units import UnitError
from cosapp.utils.testing import get_args


@pytest.fixture(scope='module')
def context():
    return System('context')


@pytest.fixture(scope='function')
def owner(context):
    return mock.Mock(spec=ModeVarPort, owner=context)


@pytest.mark.parametrize("name, error", [
    ("1var", ValueError),
    ("_var", ValueError),
    ("var-2", ValueError),
    ("var:2", ValueError),
    ("var.2", ValueError),
    ("inwards", ValueError),
    ("outwards", ValueError),
    (23, TypeError),
    (1.0, TypeError),
    (dict(a=True), TypeError),
    (list(), TypeError)
])
def test_ModeVariable___init___bad_name(owner, name, error):
    with pytest.raises(error):
        v = ModeVariable(name, owner)


def test_ModeVariable___init__(owner, caplog):
    name, value = "var1", 2.0

    # Minimal constructor
    v = ModeVariable(name, owner, value=value)
    for a, b in (
        {
            "name": name,
            "unit": "",
            "dtype": (Number, np.ndarray),
            "description": "",
            "scope": Scope.PRIVATE,
        }
    ).items():
        assert getattr(v, a) == b

    # Full constructor
    w = ModeVariable(
        name,
        owner,
        value=value,
        unit="kg",
        dtype=float,
        desc="Dummy yummy.",
        scope=Scope.PUBLIC,
    )
    for a, b in (
        {
            "name": name,
            "unit": "kg",
            "dtype": float,
            "description": "Dummy yummy.",
            "scope": Scope.PUBLIC,
        }
    ).items():
        assert getattr(w, a) == b

    # Test physical unit
    v = ModeVariable(name, owner, value=value, unit="")
    assert v.unit == ""

    v = ModeVariable(name, owner, value=value, unit="kg/s")
    assert v.unit == "kg/s"

    with pytest.raises(UnitError, match=r"Unknown unit [\w/]+."):
        ModeVariable(name, owner, value=value, unit="kg/gh")

    # Test unit for boolean
    name = "var1"
    value = True

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        v = ModeVariable(name, owner, value=value, unit="kg/s")

    assert v.unit == ""
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelno == logging.WARNING
    expected_msg = r"A physical unit is defined for non-numerical variable '\w+'; it will be ignored."
    assert re.match(expected_msg, record.message)

    # Test unit for string
    name = "var1"
    value = "hello"
    v = ModeVariable(name, owner, value=value, unit="")
    assert v.unit == ""

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        v = ModeVariable(name, owner, value=value, unit="m")
    assert v.unit == ""
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelno == logging.WARNING
    expected_msg = r"A physical unit is defined for non-numerical variable '\w+'; it will be ignored."
    assert re.match(expected_msg, record.message)

    # Test description
    name = "var1"
    value = 2.5
    v = ModeVariable(name, owner, value=value, desc="I am a lonely description.")
    assert getattr(v, "description") == "I am a lonely description."
    with pytest.raises(TypeError):
        ModeVariable(name, owner, value=value, desc=42.0)

    # Test scope
    name = "var1"
    value = 2.5
    v = ModeVariable(name, owner, value=value, scope=Scope.PRIVATE)
    assert v.scope == Scope.PRIVATE

    name = "var1"
    value = 2.5
    v = ModeVariable(name, owner, value=value, scope=Scope.PROTECTED)
    assert v.scope == Scope.PROTECTED

    name = "var1"
    value = 2.5
    v = ModeVariable(name, owner, value=value, scope=Scope.PUBLIC)
    assert v.scope == Scope.PUBLIC

    with pytest.raises(TypeError):
        ModeVariable(name, owner, value=value, scope="PRIVATE")

    # Test dtype
    name = "var1"
    value = -2
    v = ModeVariable(name, owner, value=value)
    assert v.dtype == (Number, np.ndarray)

    name = "var1"
    value = 2.5
    v = ModeVariable(name, owner, value=value)
    assert v.dtype == (Number, np.ndarray)

    name = "var1"
    value = True
    v = ModeVariable(name, owner, value=value)
    assert v.dtype == bool

    name = "var1"
    value = ""
    v = ModeVariable(name, owner, value=value)
    assert v.dtype == str

    name = "var1"
    value = list()
    v = ModeVariable(name, owner, value=value)
    assert v.dtype == list

    name = "var1"
    value = tuple()
    v = ModeVariable(name, owner, value=value)
    assert v.dtype == tuple

    name = "var1"
    value = set()
    v = ModeVariable(name, owner, value=value)
    assert v.dtype == set

    name = "var1"
    value = dict()
    v = ModeVariable(name, owner, value=value)
    assert v.dtype == dict

    name = "var1"
    value = np.asarray([1, 2, 3])
    v = ModeVariable(name, owner, value=value)
    assert v.dtype == np.ndarray

    name = "var1"
    value = 2.5
    v = ModeVariable(name, owner, value=value, dtype=float)
    assert v.dtype == float

    name = "var1"
    value = 2
    v = ModeVariable(name, owner, value=value, dtype=int)
    assert v.dtype == int

    name = "var1"
    value = 2.5
    with pytest.raises(TypeError, match=r"Cannot set .+ of type \w+ with a \w+"):
        ModeVariable(name, owner, value=value, dtype=int)

    name = "var1"
    value = np.r_[1.0]
    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        ModeVariable(name, owner, value=value)
    assert len(caplog.records) == 0


def test_ModeVariable___str__(owner):
    name, value = "var1", 2.0
    v = ModeVariable(name, owner, value)
    assert str(v) == name


@pytest.mark.parametrize("kwargs, expected", [
    (
        dict(), "**var1** &#128274;&#128274; : 2 |"
    ),
    (
        dict(
        unit="kg",
        dtype=float,
        desc="I'm a dummy donkey.",
        scope=Scope.PROTECTED),
        "**var1** &#128274; : 2 kg | I'm a dummy donkey."
    ),
])
def test_ModeVariable___repr__(owner, kwargs, expected):
    name, value = "var1", 2.0
    setattr(owner, name, value)

    v = ModeVariable(name, owner, value, **kwargs)
    assert repr(v) == expected 


@pytest.mark.parametrize("kwargs, expected", [
    (
        dict(), {'value': 2.0, "dtype": "(<class 'numbers.Number'>, <class 'numpy.ndarray'>)"}
    ),
    (
        dict(
            unit="kg",
            dtype=float,
            desc="I'm a dummy donkey.",
            scope=Scope.PROTECTED,
        ),
        {
            "value": 2.0, 
            "unit": "kg",
            "dtype": "<class 'float'>",
            "desc": "I'm a dummy donkey."
        },
    ),
])
def test_ModeVariable___json__(owner, kwargs, expected):
    name, value = "var1", 2.0
    setattr(owner, name, value)

    v = ModeVariable(name, owner, value, **kwargs)
    assert v.__json__() == expected


@pytest.mark.parametrize("kwargs, expected", [
    (
        dict(),
        {"value": 2.0, "dtype": "(<class 'numbers.Number'>, <class 'numpy.ndarray'>)"}
    ),
    (
        dict(
            unit="kg",
            dtype=float,
            desc="I'm a dummy donkey.",
            scope=Scope.PROTECTED,
        ),
        {
            "value": 2.0,
            "unit": "kg",
            "dtype": "<class 'float'>",
            "desc": "I'm a dummy donkey.",
        }
    ),
])
def test_ModeVariable_to_dict(owner, kwargs, expected):
    name, value = "var1", 2.0  
    setattr(owner, name, value)
    w1 = ModeVariable(name, owner, value, **kwargs)    
    assert w1.to_dict() == expected


@pytest.mark.parametrize("name, expected", [
    ("var1", does_not_raise()),
    ("&var", pytest.raises(ValueError)),
    ("foo.bar", pytest.raises(ValueError)),
    ("time", pytest.raises(ValueError, match="reserved")),
    ("inwards", pytest.raises(ValueError, match="invalid")),
    ("outwards", pytest.raises(ValueError, match="invalid")),
    (3.14159, pytest.raises(TypeError)),
])
def test_ModeVariable_name(owner, name, expected):
    value = 2.0
    with expected:
        v = ModeVariable(name, owner, value)
        assert v.name == name


def test_ModeVariable_name_change(owner):
    name, value = "var1", 2.0
    v = ModeVariable(name, owner, value)
    assert v.name == name
    with pytest.raises(AttributeError):
        v.name = "hello"


def test_ModeVariable_unit(owner):
    name, value = "var1", 2.0
    v = ModeVariable(name, owner, value)
    with pytest.raises(AttributeError):
        v.unit = "Pa"


def test_ModeVariable_description(owner):
    name, value = "var1", 2.0
    v = ModeVariable(name, owner, value)
    with pytest.raises(AttributeError):
        v.description = "This is the most beautiful"


def test_ModeVariable_scope(owner):
    name, value = "var1", 2.0
    v = ModeVariable(name, owner, value)
    with pytest.raises(AttributeError):
        v.scope = Scope.PUBLIC
