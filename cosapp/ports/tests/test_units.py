""" Unit tests for the units library."""
import os

import pytest

from cosapp.ports import units
from cosapp.ports.units import (
    NumberDict,
    PhysicalUnit,
    _find_unit,
    import_library,
    add_unit,
    add_offset_unit,
    UnitError,
    get_conversion,
    conversion_to_base_units,
    convert_units,
    is_compatible,
)


def test_NumberDict_UnknownKeyGives0():
    # a NumberDict instance should initialize using integer and non-integer indices
    # a NumberDict instance should initialize all entries with an initial
    # value of 0
    x = NumberDict()

    # integer test
    assert x[0] == 0

    # string test
    assert x["t"] == 0


def test_NumberDict__add__KnownValues():
    # __add__ should give known result with known input
    # for non-string data types, addition must be commutative

    x = NumberDict()
    y = NumberDict()
    x["t1"], x["t2"] = 1, 2
    y["t1"], y["t2"] = 2, 1

    result1, result2 = x + y, y + x
    assert (3, 3) == (result1["t1"], result1["t2"])
    assert (3, 3) == (result2["t1"], result2["t2"])


def test_NumberDict__sub__KnownValues():
    # __sub__ should give known result with known input
    # commuting the input should result in equal magnitude, opposite sign

    x = NumberDict()
    y = NumberDict()
    x["t1"], x["t2"] = 1, 2
    y["t1"], y["t2"] = 2, 1

    result1, result2 = x - y, y - x
    assert (-1, 1) == (result1["t1"], result1["t2"])
    assert (1, -1) == (result2["t1"], result2["t2"])


def test_NumberDict__mul__KnownValues():
    # __mul__ should give known result with known input

    x = NumberDict([("t1", 1), ("t2", 2)])
    y = 10

    result1, result2 = x * y, y * x
    assert (10, 20) == (result1["t1"], result1["t2"])
    assert (10, 20) == (result2["t1"], result2["t2"])


def test_NumberDict__div__KnownValues():
    # __div__ should give known result with known input

    x = NumberDict()
    x = NumberDict([("t1", 1), ("t2", 2)])
    y = 10.0
    result1 = x / y
    assert (0.1, 0.20) == (result1["t1"], result1["t2"])


with open(
    os.path.join(os.path.dirname(units.__file__), "unit_library.ini")
) as default_lib:
    _unitLib = import_library(default_lib)


def _get_powers(**powdict):
    powers = [0] * len(_unitLib.base_types)
    for name, power in powdict.items():
        powers[_unitLib.base_types[name]] = power
    return powers


def test_PhysicalUnit_repr_str():
    # __repr__should return a string which could be used to contruct the
    # unit instance, __str__ should return a string with just the unit
    # name for str

    u = _find_unit("d")

    assert repr(u) == "PhysicalUnit({'d': 1},86400.0,%s,0.0)" % _get_powers(time=1)
    assert str(u) == "<PhysicalUnit d>"


def test_PhysicalUnit_cmp():
    # should error for incompatible units, if they are compatible then it
    # should cmp on their factors

    x = _find_unit("d")
    y = _find_unit("s")
    z = _find_unit("ft")

    assert x > y
    assert x == x
    assert y < x

    with pytest.raises(UnitError, match=r"Unit \w+ is not compatible with \w+."):
        x < z


def test_PhysicalUnit_multiply():
    # multiplication should error for units with offsets

    x = _find_unit("g")
    y = _find_unit("s")
    z = _find_unit("degC")

    assert x * y == PhysicalUnit({"s": 1, "kg": 1}, 0.001, _get_powers(mass=1, time=1), 0)
    assert y * x == PhysicalUnit({"s": 1, "kg": 1}, 0.001, _get_powers(mass=1, time=1), 0)

    with pytest.raises(UnitError, match="cannot multiply units with non-zero offset"):
        x * z


def test_PhysicalUnit_division():
    # division should error when working with offset units

    w = _find_unit("kg")
    x = _find_unit("g")
    y = _find_unit("s")
    z = _find_unit("degC")

    quo = w / x
    quo2 = x / y

    assert quo == PhysicalUnit({"kg": 1, "g": -1}, 1000.0, _get_powers(), 0)
    assert quo2 == PhysicalUnit(
        {"s": -1, "g": 1}, 0.001, _get_powers(mass=1, time=-1), 0
    )
    quo = y / 2.0
    assert quo == PhysicalUnit({"s": 1, "2.0": -1}, 0.5, _get_powers(time=1), 0)

    quo = 2.0 / y
    assert quo == PhysicalUnit({"s": -1, "2.0": 1}, 2, _get_powers(time=-1), 0)

    with pytest.raises(UnitError, match="cannot divide units with non-zero offset"):
        x / z

def test_PhysicalUnit_pow():
    # power should error for offest units and for non-integer powers

    x = _find_unit("m")
    y = _find_unit("degF")

    z = x ** 3
    assert z == _find_unit("m**3")
    x = z ** (1.0 / 3.0)  # checks inverse integer units
    assert x == _find_unit("m")

    # test offset units:
    with pytest.raises(
        UnitError, match="cannot exponentiate units with non-zero offset"
    ):
        y ** 17

    # test non-integer powers
    with pytest.raises(
        TypeError, match="Only integer and inverse integer exponents allowed"
    ):
        x ** 1.2

    with pytest.raises(
        TypeError, match="Only integer and inverse integer exponents allowed"
    ):
        x ** (5.0 / 2.0)


def test_PhysicalUnit_conversion_tuple_to():
    # test_conversion_tuple_to should error when units have different power
    # lists

    w = _find_unit("cm")
    x = _find_unit("m")
    y = _find_unit("degF")
    z1 = _find_unit("degC")

    # check for non offset units
    assert w.conversion_tuple_to(x) == (1 / 100.0, 0)

    # check for offset units
    result = y.conversion_tuple_to(z1)
    assert result[0] == pytest.approx(0.556, abs=1e-3)
    assert result[1] == pytest.approx(-32.0, abs=1e-3)

    # check for incompatible units
    with pytest.raises(UnitError, match=r"Unit \w+ is not compatible with \w+."):
        x.conversion_tuple_to(z1)


def test_PhysicalUnit_name():
    # name should return a mathematically correct representation of the
    # unit
    x1 = _find_unit("m")
    x2 = _find_unit("kg")
    y = 1 / x1
    assert y.name() == "1/m"
    y = 1 / x1 / x1
    assert y.name() == "1/m**2"
    y = x1 ** 2
    assert y.name() == "m**2"
    y = x2 / (x1 ** 2)
    assert y.name() == "kg/m**2"


def test_add_unit():
    with pytest.raises(
        KeyError, match="'Unit ft already defined with different factor or powers'"
    ):
        add_unit("ft", "20*m")

    with pytest.raises(
        KeyError, match="'Unit degR already defined with different factor or powers'"
    ):
        add_offset_unit("degR", "degK", 20, 10)


def test_find_units():
    assert isinstance(_find_unit("m"), PhysicalUnit)
    assert isinstance(_find_unit("kg"), PhysicalUnit)
    assert isinstance(_find_unit("K"), PhysicalUnit)
    assert isinstance(_find_unit("A"), PhysicalUnit)
    assert isinstance(_find_unit("s"), PhysicalUnit)
    assert isinstance(_find_unit("mol"), PhysicalUnit)
    assert isinstance(_find_unit("cd"), PhysicalUnit)

    assert _find_unit("gh") is None
    assert _find_unit("") is None


def test_is_compatible():
    assert is_compatible("", "") == True
    assert is_compatible("", "m") == False
    assert is_compatible("kg", "") == False
    assert is_compatible("kg", "lbm") == True
    assert is_compatible("ft", "m") == True
    assert is_compatible("m", "cm") == True


def test_get_conversion():
    assert get_conversion("m", "ft") == (1.0 / 0.3048, 0.0)
    assert get_conversion("K", "degC") == (1.0, -273.15)
    assert get_conversion("K", "degR") == (pytest.approx(9.0 / 5.0), 0.0)
    assert get_conversion("m**2", "ft**2") == (1.0 / 0.3048 ** 2, 0.0)
    assert get_conversion("m/s", "ft/s") == (1.0 / 0.3048, 0.0)

    assert get_conversion("", "") is None
    assert get_conversion("K", "") is None
    assert get_conversion("", "m") is None

    with pytest.raises(UnitError, match=r"Unit \w+ is not compatible with \w+."):
        get_conversion("K", "m")


def test_conversion_to_base_units():
    assert conversion_to_base_units("cm") == (0.0, 1.0e-2)
    assert conversion_to_base_units("km") == (0.0, 1.0e3)


def test_convert_units():
    assert convert_units(3.0, "mm") == 3.0
    assert convert_units(3.0, "mm", "cm") == pytest.approx(3.0e-1)
    assert convert_units(100, "degC", "degF") == pytest.approx(212.0)

    assert convert_units(100, "degC", "") == 100.0
    assert convert_units(100, "", "degF") == 100.0
