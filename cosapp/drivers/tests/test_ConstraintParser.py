import pytest
from contextlib import nullcontext as does_not_raise

from cosapp.drivers.utils import ConstraintParser, Constraint
from cosapp.systems import System


def test_ConstraintParser_types():
    """Test class method ConstraintParser.types()"""
    types = ConstraintParser.types()
    assert set(types) == {"<", "<=", "==", ">=", ">"}


@pytest.mark.parametrize("expression, expected", [
    ("a < c - b", {Constraint("c - b", "a", True)}),
    ("a > c - b", {Constraint("a", "c - b", True)}),
    ("a <= c - b", {Constraint("c - b", "a", True)}),
    ("a >= c - b", {Constraint("a", "c - b", True)}),
    ("sin(a) > exp(x - y)", {Constraint("sin(a)", "exp(x - y)", True)}),
    ("0 < x < 1", {Constraint("x", "0", True), Constraint("1", "x", True)}),
    ("0 < x > 1", {Constraint("x", "0", True), Constraint("x", "1", True)}),
    ("a < b > a", {Constraint("b", "a", True)}),
    (
        "0 < x == a + b < 1",
        {
            Constraint("x", "0", True),
            Constraint("x", "a + b", False),
            Constraint("1", "a + b", True),
        },
    ),
    (
        "a < b < c == d",
        {
            Constraint("b", "a", True),
            Constraint("c", "b", True),
            Constraint("c", "d", False),
        },
    ),
    (
        "a < b > c > d <= e == f < g",
        {
            Constraint("b", "a", True),
            Constraint("b", "c", True),
            Constraint("c", "d", True),
            Constraint("e", "d", True),
            Constraint("e", "f", False),
            Constraint("g", "f", True),
        },
    ),
    (
        ["a < b < c", "x == y"],  # list of expressions
        {
            Constraint("b", "a", True),
            Constraint("c", "b", True),
            Constraint("x", "y", False),
        },
    ),
])
def test_ConstraintParser_parse(expression, expected):
    constraints = ConstraintParser.parse(expression)
    print(expected)
    assert constraints == expected


@pytest.mark.parametrize("expression", [
    "a < c = b",
    "a === b",
    "a <=> b",
    "a << b",
    "a >> b",
])
def test_ConstraintParser_parse_error(expression):
    with pytest.raises(ValueError, match="Invalid constraint"):
        ConstraintParser.parse(expression)
