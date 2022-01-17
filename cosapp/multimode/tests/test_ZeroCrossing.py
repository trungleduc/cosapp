import pytest

from cosapp.multimode.zeroCrossing import (
    ZeroCrossing,
    EventDirection,
)


def test_ZeroCrossing_operators():
    """Test class method ZeroCrossing.operators()"""
    operators = ZeroCrossing.operators()
    assert set(operators) == {"<", "<=", "==", ">=", ">"}
    assert all(
        isinstance(direction, EventDirection)
        for direction in operators.values()
    )


@pytest.mark.parametrize("comp, direction", [
    (">=", EventDirection.UP),
    ("==", EventDirection.UPDOWN),
    ("<=", EventDirection.DOWN),
    (">", EventDirection.UP),
    ("<", EventDirection.DOWN)
])
@pytest.mark.parametrize("lhs", ['a', '2 * a - b', 'c - a'])
@pytest.mark.parametrize("rhs", ['c - b', 'sin(a)', 'exp(b - a)', '1'])
def test_ZeroCrossing_from_comparison(comp, direction, lhs, rhs):
    zeroxing = ZeroCrossing.from_comparison(f"{lhs} {comp} {rhs}")
    assert zeroxing.direction == direction
    assert zeroxing.expression == f"{lhs.strip()} - ({rhs.strip()})"


@pytest.mark.parametrize("expression, exception", [
    ("a", ValueError),
    ("a == ", ValueError),
    (" == b", ValueError),
    ("a <== b", ValueError),
    ("a === b", ValueError),
    ("a <> b", ValueError),
    ("a == b == c", ValueError),
    ("a <= b > c", ValueError),
    ("a + b - c", ValueError),
])
def test_ZeroCrossing_from_comparison_error(expression, exception):
    """Test `ZeroCrossing.from_comparison` with erroneous expressions"""
    with pytest.raises(exception):
        ZeroCrossing.from_comparison(expression)


@pytest.mark.parametrize("direction", EventDirection)
@pytest.mark.parametrize("expression", [
    12,
    "nonsense",
    "syntax error(",
])
def test_ZeroCrossing_ctor(expression, direction):
    zeroxing = ZeroCrossing(expression, direction)
    assert zeroxing.expression == expression
    assert zeroxing.direction is direction


@pytest.mark.parametrize("method", ["up", "down", "updown"])
@pytest.mark.parametrize("expression", [
    12,
    "nonsense",
    "syntax error(",
])
def test_ZeroCrossing_factories(expression, method):
    """Test factories `ZeroCrossing.up`, `down` and `updown`"""
    factory = getattr(ZeroCrossing, method.lower())
    zeroxing = factory(expression)
    assert zeroxing.expression == expression
    assert zeroxing.direction is EventDirection[method.upper()]
