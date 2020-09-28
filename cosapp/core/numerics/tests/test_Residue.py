import pytest
import numpy as np

from cosapp.systems import System
from cosapp.core.numerics.residues import Residue


class LocalSystem(System):
    def setup(self):
        self.add_inward('a', 1.)
        self.add_inward('b', [1., 2.])
        self.add_inward('c', np.array([1., 2.]))


@pytest.mark.parametrize("expression, exception", [
    (12, TypeError),
    ([1, 2], TypeError),
    (['x', 'y'], TypeError),
    (dict(x=1, y=0), TypeError),
    ("x == y == z", ValueError),
    ("a === b", ValueError),
    ("a ==== = b", ValueError),
    ("a == == b", ValueError),
    (" == ", SyntaxError),
    ("x == ", SyntaxError),
    ("== x", SyntaxError),
    ("x    ==   ", SyntaxError),
    ("x + y / 2", ValueError),
])
def test_Residue_split_equation_error(expression, exception):
    with pytest.raises(exception):
        Residue.split_equation(expression)


@pytest.mark.parametrize("expression, lhs, rhs", [
    (" x == y + z ", "x", "y + z"),
    ("a + b == Z", "a + b", "Z"),
    ("a + 5 == 2.1", "a + 5", "2.1"),
    ("[a, 2, foo] == nonsense", "[a, 2, foo]", "nonsense"),
    ("nonsense == [a, 2, foo]", "nonsense", "[a, 2, foo]"),
    ("  nonsense  ==   [a, 2, foo] ", "nonsense", "[a, 2, foo]"),
])
def test_Residue_split_equation_ok(expression, lhs, rhs):
    assert Residue.split_equation(expression) == (lhs, rhs)


@pytest.mark.parametrize("args, kwargs, exception", [
    (("2. == 3.", 'foo', 20.), dict(), RuntimeWarning),
    (("4.",), dict(), ValueError),
    (("4.",), dict(reference=20.), ValueError),
    (("array([2., 4.]) == array([3., 6.])",), dict(reference=20.), RuntimeWarning),
    (("array([4., 5.]) == 0",), dict(), RuntimeWarning),
    (("array([4., 5.]) == a + 'foo'",), dict(), TypeError),
    (("[a, b] == [1, 2]",), dict(), TypeError),
    (("[a, b] == [1, [2, 3]]",), dict(), TypeError),
    # Note: behaviour of next cases may change with numpy version
    #       (comparison of two arrays of different sizes)
    (("c == []",), dict(), ValueError),
    (("c == [4., 5., 0.1]",), dict(), ValueError),
    (("c == array([4., 5., 0.1])",), dict(), ValueError),
])
def test_Residue___init__Error(args, kwargs, exception):
    s = LocalSystem('system')
    with pytest.raises(exception):
        Residue(s, *args, **kwargs)


@pytest.mark.parametrize("args, kwargs, name, value, reference", [
    (("a == 4",), dict(), "a == 4", -3, 1),
    (("c == 0",), dict(), "c == 0", [1, 2], 1),
    (("c == 0",), dict(reference="norm"), "c == 0", [1, 2], [1, 1]),
    (("a + b[0] == 0.5",), dict(), "a + b[0] == 0.5", 1.5, 1),
    (("a + b[0] == 0.5",), dict(name="balance"), "balance", 1.5, 1),
    (('a + b[0] == 0.5', 'foo',), dict(), 'foo', 1.5, 1),
    (('a + b[0] == 0.5', 'foo', 15,), dict(), 'foo', 0.1, 15),
    (("c == [1, 0]",), dict(), "c == [1, 0]", [0, 2], 1),
    (("c == [1, 0]",), dict(reference=[1, 10]), "c == [1, 0]", [0, 0.2], [1, 10]),
    (("c == [1, 0]",), dict(reference="norm"), "c == [1, 0]", [0, 2], [1, 1]),
    (("[a, b[0]] == [0, 2]", "weird"), dict(), "weird", [1, -1], 1),
])
def test_Residue___init__(args, kwargs, name, value, reference):
    s = LocalSystem('system')
    r = Residue(s, *args, **kwargs)
    assert r._context is s
    assert r.name == name
    assert r.value == pytest.approx(value, rel=1e-14)
    assert r.reference == pytest.approx(reference, rel=1e-14)


@pytest.mark.parametrize("lhs, rhs, expected", [
    (4, 5, 1),
    (4., 5., 1),
    (5, 8, 10),
    (2.5, 50, 10),
    (0.4, -0.3, 0.1),
    (-2, -4, 1),
    (-2e9, 0, 1e9),
    (-2e9, 100, 1e9),
    (2.5e9, 1e9, 1e9),
    (2.5e-6, 4e7, 1e7),
    (2.5e-6, 0, 1e-6),
    (2.5e-6, -1, 1),
    (np.r_[4., 0.4, -2.], np.r_[5., -0.3, -4], np.r_[1, 0.1, 1]),
    (np.r_[4., 7.1e-12, -2e9], np.r_[0, 0, 0], np.r_[1, 1e-12, 1e9]),
])
def test_Residue_residue_norm(lhs, rhs, expected):
    assert Residue.residue_norm(lhs, rhs) == pytest.approx(expected, rel=1e-14)
    assert Residue.residue_norm(rhs, lhs) == pytest.approx(expected, rel=1e-14)


@pytest.mark.parametrize("args, expected", [
    ((4., 5.), -1.),
    ((0.4, -0.3), 0.7),
    ((0.4, -0.3, 1), 0.7),
    ((0.4, -0.3, 0.1), 7),
    ((0.4, -0.3, 2), 0.35),
    ((-2.5, -4), 1.5),
    ((4., 5., 4.), -0.25),
    ((np.r_[4., 0.4, -2.], np.r_[5., -0.3, -4]), np.r_[-1, 0.7, 2]),
    ((np.r_[4., 0.4, -2.], np.r_[5., -0.3, -4], "norm"), np.r_[-1, 7, 2]),
    ((np.r_[4., 0.4, -2.], np.r_[5., -0.3, -4], 10.), np.r_[-0.1, 0.07, 0.2]),
    ((np.r_[4., 0.4, -2.], np.r_[5., -0.3, -4], np.r_[2., 7., 20.]), np.r_[-0.5, 0.1, 0.1]),
    ((np.r_[4., 7.1e-12, -2e9], np.r_[0, 0, 0]), np.r_[4, 7.1e-12, -2e9]),
    ((np.r_[4., 7.1e-12, -2e9], np.r_[0, 0, 0], "norm"), np.r_[4, 7.1, -2]),
])
def test_Residue_evaluate_residue(args, expected):
    assert Residue.evaluate_residue(*args) == pytest.approx(expected, rel=1e-14)


@pytest.mark.parametrize("args, kwargs", [
    (('a + b[0] == 0.5', 'foo', 15,), dict()),
    (('a + b[0] == 0.5', 'foo',), dict()),
    (('a + b[0] == 0.5',), dict()),
    (("c == 0",), dict()),
    (("a + b[0] == 0.5",), dict(name="balance")),
    (("[a, b[0]] == [0, 2]", "weird"), dict()),
    (("c == [1, 0]",), dict()),
])
def test_Residue_copy(args, kwargs):
    s = LocalSystem('system')
    r = Residue(s, *args, **kwargs)
    t = r.copy()
    assert t is not r
    assert t.context is r.context
    assert t.name == r.name
    assert t.value == pytest.approx(t.value, rel=1e-14)
    assert t.reference == pytest.approx(t.reference, rel=1e-14)


@pytest.mark.parametrize("options, expected", [
    (dict(equation="a == 4", ), dict()),
    (dict(equation="c == 0", ), dict()),
    (dict(equation="c == 0", reference="norm"), dict(reference="[1.0, 1.0]")),
    (dict(equation="a + b[0] == 0.5",), dict()),
    (dict(equation="a + b[0] == 0.5", name="balance"), dict()),
    (dict(equation='a + b[0] == 0.5', name='foo',), dict()),
    (dict(equation='a + b[0] == 0.5', name='foo', reference=15,), dict(reference="15")),
    (dict(equation="c == [1, 0]",), dict()),
    (dict(equation="c == [1, 0]", reference=[1, 10]), dict(reference="[1, 10]")),
    (dict(equation="c == [1, 0]", reference="norm"), dict(reference="[1.0, 1.0]")),
    (dict(equation="[a, b[0]] == [0, 2]", name="weird"), dict()),
])
def test_Residue_to_dict(options, expected):
    s = LocalSystem('system')
    r = Residue(s, **options)
    r_dict = r.to_dict()

    assert r_dict["context"] == s.contextual_name
    assert r_dict["name"] == options.get("name", options["equation"])
    assert r_dict["equation"] == options["equation"]
    assert r_dict["reference"] == expected.get("reference", "1")
