import pytest

from cosapp.utils.naming import has_time


@pytest.mark.parametrize("expression, expected", [
    ("t", True),
    ("'t'", True),
    ("time", False),
    ("T", False),
    ("-t", True),
    ("tt", False),
    ("foobar", False),
    ("t**2", True),
    ("2t", False),
    ("foo_t", False),
    ("2*t", True),
    ("2 *t", True),
    ("2 * t", True),
    (".t.", True),
    ("exp(-t / tau)", True),
    ("cos(w * t)", True),
    # Non-string arguments
    (23, False),
    (1.0, False),
    (dict(a=True), False),
    ([1, 2, 3], False),
    ([1, 2, 't'], True),
])
def test_has_time(expression, expected):
    assert has_time(expression) == expected


basic_symbols = ["", " ", "+", "-", "*", "/", "**"]

@pytest.mark.parametrize("left", basic_symbols + ["("], ids=lambda a: f'"{a}"')
@pytest.mark.parametrize("right", basic_symbols + [")"], ids=lambda a: f'"{a}"')
def test_has_time_math_symbols(left, right):
    assert has_time(left + "t" + right)
    assert not has_time(left + "T" + right)
