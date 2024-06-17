import pytest
import numpy as np

from cosapp.base import System, Port
from cosapp.core.numerics.residues import Residue
from cosapp.utils.testing import get_args


class LocalSystem(System):
    def setup(self):
        self.add_inward('a', 1.)
        self.add_inward('b', [1., 2.])
        self.add_inward('c', np.array([1., 2.]))


class XyzPort(Port):
    def setup(self):
        self.add_variable('x', 0.)
        self.add_variable('y', 0.)
        self.add_variable('z', np.array([1., 2.]))


class XyzSystem(System):
    def setup(self):
        self.add_input(XyzPort, 'p_in')
        self.add_output(XyzPort, 'p_out')
        self.add_inward('u', np.array([1., 2.]))
        self.add_outward('v', np.array([1., 2.]))


@pytest.fixture
def system():
    """Evaluation context for residues"""
    return LocalSystem('system')


@pytest.fixture
def composite():
    """Generates test system tree:

                 a
         ________|________
        |        |        |
       aa       ab        ac
      __|__           ____|____
     |     |         |    |    |
    aaa   aab       aca  acb  acc
    """
    def add_children(parent: System, **children):
        prefix = parent.name
        for name, sysclass in children.items():
            parent.add_child(sysclass(f"{prefix}{name}"))

    a = XyzSystem('a')
    add_children(a, a=XyzSystem, b=LocalSystem, c=XyzSystem)
    add_children(a.aa, a=XyzSystem, b=XyzSystem)
    add_children(a.ac, a=XyzSystem, b=XyzSystem, c=XyzSystem)

    return a


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


@pytest.mark.parametrize("args_kwargs, expected", [
    (get_args("2. == 3.", 'foo', 20.), pytest.raises(RuntimeWarning, match="trivially constant")),
    (get_args("4."), pytest.raises(ValueError, match="equation")),
    (get_args("4.", reference=20.), pytest.raises(ValueError, match="equation")),
    (get_args("array([2., 4.]) == array([3., 6.])", reference=20.), pytest.raises(RuntimeWarning, match="trivially constant")),
    (get_args("array([4., 5.]) == 0"), pytest.raises(RuntimeWarning, match="trivially constant")),
    (get_args("array([4., 5.]) == a + 'foo'"), pytest.raises(TypeError)),
    (get_args("[a, b] == [1, 2]"), pytest.raises(TypeError, match="not comparable")),
    (get_args("[a, b] == [1, [2, 3]]"), pytest.raises(TypeError, match="not comparable")),
    # Note: behaviour of next cases may change with numpy version
    #       (comparison of two arrays of different sizes)
    #       For numpy < 1.25, `ValueError` is raised
    #       For numpy >= 1.25, `TypeError` is raised
    (get_args("c == []"), pytest.raises((ValueError, TypeError))),
    (get_args("c == [4., 5., 0.1]"), pytest.raises((ValueError, TypeError))),
    (get_args("c == array([4., 5., 0.1])"), pytest.raises((ValueError, TypeError))),
])
def test_Residue___init__Error(system, args_kwargs, expected):
    args, kwargs = args_kwargs
    with expected:
        Residue(system, *args, **kwargs)


@pytest.mark.parametrize("args_kwargs, name, value, reference", [
    (get_args("a == 4"), "a == 4", -3, 1),
    (get_args("c == 0"), "c == 0", [1, 2], 1),
    (get_args("c == 0", reference="norm"), "c == 0", [1, 2], [1, 1]),
    (get_args("a + b[0] == 0.5"), "a + b[0] == 0.5", 1.5, 1),
    (get_args("a + b[0] == 0.5", name="balance"), "balance", 1.5, 1),
    (get_args('a + b[0] == 0.5', 'foo'), 'foo', 1.5, 1),
    (get_args('a + b[0] == 0.5', 'foo', 15), 'foo', 0.1, 15),
    (get_args("c == [1, 0]"), "c == [1, 0]", [0, 2], 1),
    (get_args("c == [1, 0]", reference=[1, 10]), "c == [1, 0]", [0, 0.2], [1, 10]),
    (get_args("c == [1, 0]", reference="norm"), "c == [1, 0]", [0, 2], [1, 1]),
    (get_args("[a, b[0]] == [0, 2]", "weird"), "weird", [1, -1], 1),
])
def test_Residue___init__(system, args_kwargs, name, value, reference):
    args, kwargs = args_kwargs
    r = Residue(system, *args, **kwargs)
    assert r._context is system
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
    (2.5e-6, 1e-3, 1e-3),
    (2.5e-6, 1, 1),
    (2.5e-6, -1, 1),
    (2.5e-1, 1, 1),
    (2.5e+1, 1, 10),
    (np.r_[4., 0.4, -2.], np.r_[5., -0.3, -4], np.r_[1, 0.1, 1]),
    (np.r_[4., 7.1e-12, -2e9], np.r_[0, 0, 0], np.r_[1, 1e-12, 1e9]),
    (np.r_[4., 7.1e-12, -2e9], np.r_[1, 1, 1], np.r_[1, 1, 1e9]),
    (np.logspace(-12, 0, 13), np.ones(13), np.ones(13)),
    (np.logspace(0, 12, 13), np.ones(13), np.logspace(0, 12, 13)),
    # cases involving zero
    (0, 0, 1),
    (0, 1, 1),
    (0, 123.456, 100),
    (0, None, 1),
])
def test_Residue_residue_norm(lhs, rhs, expected):
    assert Residue.residue_norm(lhs, rhs) == pytest.approx(expected, rel=1e-14)
    if rhs is not None:
        assert Residue.residue_norm(rhs, lhs) == pytest.approx(expected, rel=1e-14)


@pytest.mark.parametrize("factor", [1, 5, 9])
@pytest.mark.parametrize("magnitude", np.logspace(-12, 12, 25))
def test_Residue_residue_norm_single(magnitude, factor):
    """Test `Residue.residue_norm` with a single parameter"""
    r = factor * magnitude
    assert Residue.residue_norm(r) == pytest.approx(magnitude, rel=1e-14)
    assert Residue.residue_norm(-r) == pytest.approx(magnitude, rel=1e-14)


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


@pytest.mark.parametrize("args_kwargs", [
    get_args('a + b[0] == 0.5', 'foo', 15),
    get_args('a + b[0] == 0.5', 'foo'),
    get_args('a + b[0] == 0.5'),
    get_args("c == 0"),
    get_args("a + b[0] == 0.5", name="balance"),
    get_args("[a, b[0]] == [0, 2]", "weird"),
    get_args("c == [1, 0]"),
])
def test_Residue_copy(system, args_kwargs):
    args, kwargs = args_kwargs
    r = Residue(system, *args, **kwargs)
    c = r.copy()
    assert c is not r
    assert c.context is r.context
    assert c.name == r.name
    assert c.value == pytest.approx(r.value, abs=0)
    assert c.reference == pytest.approx(r.reference, abs=0)


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
def test_Residue_to_dict(system, options, expected):
    r = Residue(system, **options)
    r_dict = r.to_dict()

    assert r_dict["context"] == system.contextual_name
    assert r_dict["equation"] == options["equation"]
    assert r_dict["reference"] == expected.get("reference", "1.0")
    assert r_dict["name"] == options.get("name", options["equation"])


@pytest.mark.parametrize("args_kwargs, expected", [
    (get_args("u[0] == 0"), {"u"}),
    (get_args("u[0] == aa.p_out.x"), {"u", "aa.p_out.x"}),
    (get_args("u[0] == aa.p_out.x * ac.v[-1]"), {"u", "ac.v", "aa.p_out.x"}),
    (get_args("ac.u[0] == aa.aab.p_out.x"), {"ac.u", "aa.aab.p_out.x"}),
    (get_args("ac.u[0] + cos(aa.aab.p_out.x) == aa.aab.p_out.x"), {"ac.u", "aa.aab.p_out.x"}),
    (get_args("ac.u[0] + cos(aa.aab.p_out.x) == -2 * aa.p_in.y"), {"ac.u", "aa.aab.p_out.x",  "aa.p_in.y"}),
])
def test_Residue_variables_head(composite, args_kwargs, expected):
    """Test property `Residue.variables` for equations defined in head system."""
    args, kwargs = args_kwargs
    r = Residue(composite, *args, **kwargs)
    assert r.variables == expected


def test_Residue_variables_subsystems(composite):
    """Test property `Residue.variables` for equations defined in subsystems."""
    a = composite
    r = Residue(a, "ac.u[0] == ac.acb.p_out.x")
    assert r.context is a
    assert r.variables == {"ac.u", "ac.acb.p_out.x"}

    # Same equation, defined in subsystem a.ac
    r = Residue(a.ac, "u[0] == acb.p_out.x")
    assert r.context is a.ac
    assert r.variables == {"u", "acb.p_out.x"}
