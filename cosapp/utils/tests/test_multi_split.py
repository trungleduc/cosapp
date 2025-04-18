import pytest

from cosapp.utils.parsing import multi_split, multi_join


@pytest.mark.parametrize("expression, symbols, expected", [
    (
        "a+b-c-d+e", list("+-"),
        dict(
            expressions = ['a', 'b', 'c', 'd', 'e'],
            separators = ['+', '-', '-', '+'],
        )
    ),
    (
        "abracadabra", ["a"],  # single separator: equivalent to `str.split`
        dict(
            expressions = "abracadabra".split("a"),
            separators = ["a"] * 5,
        )
    ),
    (
        "abracadabra", ["br"],
        dict(
            expressions = "abracadabra".split("br"),
            separators = ["br"] * 2,
        )
    ),
    (
        "abracadabra", ["a", "b"],
        dict(
            expressions = ["", "", "r", "c", "d", "", "r", ""],
            separators = ['a', 'b', 'a', 'a', 'a', 'b', 'a'],
        )
    ),
    (
        "xyz", list("abc"),  # no match
        dict(
            expressions = ["xyz"],
            separators = [],
        )
    ),
])
def test_multi_split(expression, symbols, expected):
    expressions, separators = multi_split(expression, symbols)
    assert len(expressions) == len(separators) + 1
    assert expressions == expected['expressions']
    assert separators == expected['separators']
    assert multi_join(expressions, separators) == expression


@pytest.mark.parametrize("expression, expected_terms, expected_ops", [
    ("a < c - b", ["a", "c - b"], ["<"]),
    ("a > c - b", ["a", "c - b"], [">"]),
    ("sin(a) > exp(x - y)", ["sin(a)", "exp(x - y)"], [">"]),
    ("0 < x < 1", ["0", "x", "1"], ["<", "<"]),
    ("0 < x > 1", ["0", "x", "1"], ["<", ">"]),
    ("0 > x < 1", ["0", "x", "1"], [">", "<"]),
    (
        "0 < x == a + b < 1",
        ["0", "x", "a + b", "1"],
        ["<", "==", "<"],
    ),
    (
        "a < b < c == d",
        ["a", "b", "c", "d"],
        ["<", "<", "=="],
    ),
    (
        "a < b > c > d < e == f < g",
        list("abcdefg"),
        ["<", ">", ">", "<", "==", "<"],
    ),
])
def test_multi_split_operators(expression, expected_terms, expected_ops):
    """Test multi_split with {"<", "==", ">"} separators.
    """
    terms, operators = multi_split(expression, separators={"<", "==", ">"})
    assert len(terms) == len(operators) + 1
    assert terms == expected_terms
    assert operators == expected_ops
