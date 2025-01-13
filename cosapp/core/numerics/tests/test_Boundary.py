import pytest
import numpy as np

from cosapp.systems import System
from cosapp.core.numerics.boundary import Boundary, MaskedVarInfo, AttrRef, MaskedAttrRef, NumpyMaskedAttrRef
from cosapp.ports.port import Port, PortType
from cosapp.utils.testing import get_args
from contextlib import nullcontext as does_not_raise
from cosapp.tests.library.systems import Multiply2, Strait1dLine
from typing import Dict, Any, Optional, Tuple

class APort(Port):
    def setup(self):
        self.add_variable('m')
        self.add_variable('n', np.array([3., 4.]))


class BSyst(System):
    def setup(self, **kwargs):
        self.add_inward('cc', CustomClass())
        self.add_inward('z', 2.)
        self.add_inward('a', np.reshape(np.arange(12, dtype=float), (3, 4)))

class ASyst(System):
    def setup(self, **kwargs):
        self.add_input(APort, 'in_')
        self.add_inward('x')
        self.add_inward('y', np.array([1., 2.]))
        self.add_inward('u', np.zeros(5))
        self.add_inward('cc', CustomClass())
        self.add_inward('tuple', (5., 8., 10.))
        self.add_inward('dict', {"a": 5.})
        self.add_inward('string', "str_value")
        self.add_child(BSyst("b"))
        self.add_outward('v')

class MimicSeq:
    def __init__(self):
        self.x = (1., 2., 3.)

    def __getitem__(self, i):
        return self.x[i]

    def __setitem__(self, i, new):
        self.x = self.x[:i] + (new,) + self.x[i+1:]

    def __len__(self):
        return len(self.x)


class SubCustomClass:
    def __init__(self):
        self.d = 1.
        self.x = np.r_[1., 2., 3.]
        self.seq2 = MimicSeq()


class CustomClass:
    def __init__(self):
        self.r = np.ones(3)
        self.w = [4., 5., 6.]
        self.g = SubCustomClass()
        self.seq = MimicSeq()
        self.h = [0., SubCustomClass(), 1.]

    def get_g(self):
        return self.g


@pytest.fixture
def a():
    return ASyst('a')


def get_indices(context: System, name: str, mask: Optional[np.ndarray] = None) -> Tuple:
    basename, selector = Boundary.parse_expression(name)
    _, mask = Boundary.create_mask(context, basename, selector, mask)
    return (basename, selector, mask)

def test_Boundary_parse_scalar():
    s = Multiply2("mult")
    t = System("dummy")
    t.add_child(s)

    r = get_indices(t, "mult.K1")
    assert r == ("mult.K1", "", None)
    r = get_indices(s, "K1")
    assert r == ("K1", "", None)
    r = get_indices(t, "mult.p_in.x")
    assert r == ("mult.p_in.x", "", None)


@pytest.mark.parametrize("name, expected", [
    ("hat.one.a[[2, 4]", dict(error=SyntaxError, match="Bracket mismatch")),
    ("hat.one.a[2, 4]]", dict(error=SyntaxError, match="Bracket mismatch")),
    ("hat.one.a[[2, 4]]", dict(error=IndexError, match=r"Invalid selector '[\w\[\.\s,\]]+' for variable '\w+.[\w\.]+':.*")),
    ("hat.one.a[]", dict(error=SyntaxError, match="invalid syntax")),
    ("hat.one.a[2, 4, 'oups']", dict(error=IndexError, match="Invalid selector")),
    ("mult.K1[1]", dict(error=TypeError, match="Only non-empty arrays can be partially selected")),
    ("mult[0]", dict(error=TypeError, match="Only non-empty arrays can be partially selected")),
])
def test_Boundary_parse_error(name, expected: Dict[str, Any]):
    hat = System("hat")
    one = hat.add_child(Strait1dLine("one"), pulling=["in_", "a", "b"])
    two = hat.add_child(Strait1dLine("two"), pulling=["out", "a", "b"])
    hat.connect(two.in_, one.out)

    top = System("top")
    top.add_child(hat, pulling=['a'])

    s = Multiply2("mult")
    top.add_child(s)

    error = expected.get('error', None)
    with pytest.raises(error, match=expected.get('match', None)):
        get_indices(top, name)

@pytest.mark.parametrize("name, expected", [
    ("hat.a", dict(mask=[True, True, True], basename="hat.a", selector="")),
    ("hat.one.a", dict(mask=[True, True, True], basename="hat.one.a", selector="")),
    ("hat.one.a[0]", dict(mask=[True, False, False], basename="hat.one.a", selector="[0]")),
    ("hat.one.a[1:]", dict(mask=[False, True, True], basename="hat.one.a", selector="[1:]")),
    ("hat.one.a[[0, 2]]", dict(mask=[True, False, True], basename="hat.one.a", selector="[[0, 2]]")),
    ("hat.one.a[[True, False, True]]", dict(mask=[True, False, True], basename="hat.one.a", selector="[[True, False, True]]")),
    ("hat['one'].a", dict(mask=[True, True, True], basename="hat['one'].a", selector="")),
    ("hat['one'].a[1:]", dict(mask=[False, True, True], basename="hat['one'].a", selector="[1:]")),
    # TODO: ideally, cases below should work (basename is reformatted into `x.y.z`) - OK for now
    # ("hat['one'].a", dict(mask=[True, True, True], basename="hat.one.a", selector="")),
    # ("hat['one'].a[1:]", dict(mask=[False, True, True], basename="hat.one.a", selector="[1:]")),
    # ("hat['one.a'][1:]", dict(mask=[False, True, True], basename="hat.one.a", selector="[1:]")),
])
def test_Boundary_parse_array_1D(name, expected: Dict[str, Any]):
    """Test `get_indices` with vector variables"""
    hat = System("hat")
    one = hat.add_child(Strait1dLine("one"), pulling=["in_", "a", "b"])
    two = hat.add_child(Strait1dLine("two"), pulling=["out", "a", "b"])
    hat.connect(two.in_, one.out)
    top = System("top")
    top.add_child(hat, pulling=['a'])

    basename, selector, mask = get_indices(top, name)
    assert basename == expected.get('basename', name)
    assert selector == expected.get('selector', '')
    assert isinstance(mask, np.ndarray)
    assert np.array_equal(mask, expected['mask'])

@pytest.mark.parametrize("selector, expected", [
    ("", np.full((3, 4), True)),
    ("[:]", np.full((3, 4), True)),
    ("[0]", [[True] * 4, [False] * 4, [False] * 4]),
    ("[0][1]", [[False, True, False, False], [False] * 4, [False] * 4]),
    ("[::2]", [[True] * 4, [False] * 4, [True] * 4]),
    ("[-1][::2]", [[False] * 4, [False] * 4, [True, False, True, False]]),
    ("[:, -1]", [[False, False, False, True]] * 3),
    ("[:, 1:]", [[False, True, True, True]] * 3),
    ("[:, 1::2]", [[False, True, False, True]] * 3),
])
def test_Boundary_parse_array_2D(selector, expected):
    top = System('top')
    sub = top.add_child(BSyst('sub'))

    basename, selector, mask = get_indices(top, f"sub.a{selector}")
    r = MaskedVarInfo(basename, selector, mask)
    assert r.basename == "sub.a"
    assert r.selector == selector
    assert r.fullname == f"sub.a{selector.strip()}"
    assert np.array_equal(r.mask, expected)


@pytest.mark.parametrize("name, expected", [
    ('in_.m', dict(name='in_.m', basename="in_.m", variable='m', portname='in_.m')),
    ('x', dict(name='x', basename="x", variable='x', portname='x')),
    ('inwards.x', dict(name='x', basename="x", variable='x', portname='x')),
    ('y', dict(name='y', basename="y", variable='y', portname='y')),
    ('y[0]', dict(name='y[0]', basename="y", variable='y', portname='y')),
    ('tuple', dict(name='tuple', basename="tuple", variable='tuple', portname='tuple')),
    ('dict', dict(name='dict', basename="dict", variable='dict', portname='dict')),
    ('string', dict(name='string', basename="string", variable='string', portname='string')),
    ('cc.r[:2]', dict(name='cc.r[:2]', basename="cc.r", variable='r', portname='cc')),
    ('cc.g.d', dict(name='cc.g.d', basename="cc.g.d", variable='d', portname='cc')),
    ('b.cc.r', dict(name='b.cc.r', basename="b.cc.r", variable='r', portname='b.cc')),
    ('cc.get_g().x', dict(name='cc.get_g().x', basename="cc.get_g().x", variable='x', portname='cc')),
    ('cc.get_g().x[1]', dict(name='cc.get_g().x[1]', basename="cc.get_g().x", variable='x', portname='cc')),
    ('cc.w[:2]', dict(name='cc.w[:2]', basename="cc.w", variable='w', portname='cc')),
    ('cc.seq', dict(name='cc.seq', basename="cc.seq", variable='seq', portname='cc')),
    ('cc.seq[1]', dict(name='cc.seq[1]', basename="cc.seq", variable='seq', portname='cc')),
    ('cc.h[1].seq2[1]', dict(name='cc.h[1].seq2[1]', basename="cc.h[1].seq2", variable='seq2', portname='cc')),
])
def test_Boundary__init__names(a, name, expected):
    x = Boundary(a, name)

    # Test object attributes:
    assert x.context is a
    assert x.name == expected["name"]
    assert x.basename == expected["basename"]
    assert x.variable == expected["variable"]
    assert x.portname == expected["portname"]


@pytest.mark.parametrize("name, mask", [
    ('x', None),
    ('y', [True, True]),
    ('y[0]', [True, False]),
    ('y[1]', [False, True]),
    ('y[:]', [True, True]),
    ('y[1:]', [False, True]),
    ('y[:-1]', [True, False]),
    ('u[::2]',  [True, False, True, False, True]),
    ('cc.r[:2]', [True, True, False]),
    ('b.cc.r', [True, True, True]),
    ('cc.get_g().x',  [True, True, True]),
    ('cc.get_g().x[1]', [False, True, False]),
    ('cc.w',  [True, True, True]),
    ('cc.w[:2]', [True, True, False]),
    ('cc.seq', [True, True, True]),
    ('cc.seq[1]', [False, True, False]),
    ('cc.h[1].seq2[1]', [False, True, False]),
])
def test_Boundary__init__mask(a, name, mask):
    x = Boundary(a, name)

    # Test object attributes:
    if mask is None:
        assert not hasattr(x._ref, "mask")
    else:
        assert isinstance(x.mask, np.ndarray)
        assert np.all(x.mask == np.asarray(mask))


@pytest.mark.parametrize("name, init_mask, final_mask", [
    ('y', [True, False], [True, False]),
    ('y', (False, False), [False, False]),
    ('y', (True, True), [True, True]),
    ('y', np.full(2, True), [True, True]),
    ('y[1]', [True, False], [True, False]),
    ('u[::2]',  [True, True, False, False, False], [True, True, False, False, False]),
])
def test_Boundary__init__default_mask(a, name, init_mask, final_mask):
    x = Boundary(a, name, mask=init_mask)
    assert isinstance(x.mask, np.ndarray)
    assert np.all(x.mask == np.asarray(final_mask))


@pytest.mark.parametrize("name, value", [
    ('x', 4.),
    ('y', [0, 0]),
    ('y', np.zeros(2)),
    ('y', 1.234),
    ('y[1]', np.array([-2.])),
    ('y[1]', [-2.]),
    ('y[1]', -2.),
    ('y[1:]', -2.),
    ('y[:-1]', -2.),
    ('y[:]', np.array([-2., 1.])),
    ('y[:]', [-2., 1.]),
    ('u[::2]', [1., 2., 3.]),
    ('tuple', (2., 5., 9.)),
    ('dict', {"b": 6.}),
    ('string', "new_value"),
    ('cc.r[:2]', [1., 2.]),
    ('cc.g.d', 15.),
    ('b.cc.r', [5., 5., 5.]),
])
def test_Boundary__init__default_value(a, name, value):
    x = Boundary(a, name, default=value)
    assert x.default_value == pytest.approx(value, abs=0)


@pytest.mark.parametrize("name, cls", [
    ('in_.m', AttrRef),
    ('x', AttrRef),
    ('inwards.x', AttrRef),
    ('y', NumpyMaskedAttrRef),
    ('y[0]', NumpyMaskedAttrRef),
    ('cc.g.d', AttrRef),
    ('cc.r[:2]', NumpyMaskedAttrRef),
    ('b.cc.r', NumpyMaskedAttrRef),
    ('cc.get_g().x', NumpyMaskedAttrRef),
    ('cc.get_g().x[1]', NumpyMaskedAttrRef),
    ('cc.w[:2]', MaskedAttrRef),
    ('cc.seq', MaskedAttrRef),
    ('cc.seq[1]', MaskedAttrRef),
    ('cc.h', MaskedAttrRef),
    ('cc.h[1].seq2', MaskedAttrRef),
])
def test_Boundary_ref(a, name, cls):
    x = Boundary(a, name)
    assert isinstance(x._ref, cls)
    assert isinstance(x._ref, AttrRef)


@pytest.mark.parametrize("name", [
    'in_.m',
    'x',
    'inwards.x',
    'y',
    'y[0]',
    'cc.g.d',
    'cc.r[:2]',
    'b.cc.r',
    'cc.get_g().x',
    'cc.get_g().x[1]',
    'cc.w[:2]',
    'cc.seq',
    'cc.seq[1]',
    'cc.h',
    'cc.h[1].seq2',
])
def test_Boundary_equality(a, name):
    b1 = Boundary(a, name)
    b2 = Boundary(a, name)
    assert b1 is not b2
    assert b1._ref is not b2._ref
    assert b1._ref == b2._ref
    assert b1 == b2


@pytest.mark.parametrize("name, kwargs, expected", [
    ('_', dict(), pytest.raises(AttributeError)),
    ('foo', dict(), pytest.raises(AttributeError)),
    ('in_.x', dict(), pytest.raises(AttributeError)),
    ('inwards.m', dict(), pytest.raises(AttributeError)),
    ('v', dict(inputs_only=False), does_not_raise()),
    ('v', dict(), pytest.raises(ValueError, match="Only variables in input ports")),
    ('outwards.v', dict(), pytest.raises(ValueError, match="Only variables in input ports")),
    ('x', dict(mask=False), pytest.raises(TypeError, match="mask")),
    ('y', dict(mask="False"), pytest.raises(TypeError, match="mask")),
    ('y', dict(mask=0), pytest.raises(TypeError, match="mask")),
    ('x', dict(inputs_only=True), does_not_raise()),
    ('x', dict(inputs_only=False), does_not_raise()),
    ('in_', dict(), pytest.raises(ValueError)),
    ('inwards', dict(), pytest.raises(ValueError)),
    ('outwards', dict(), pytest.raises(ValueError)),
    ('y[', dict(), pytest.raises(SyntaxError)),
    ('y(', dict(), pytest.raises(SyntaxError)),  # TODO remove this test since tuple not allowed anymore
    ('cc.h[0].seq2[1]', dict(), pytest.raises(AttributeError)),
])
def test_Boundary___init__error(a, name, kwargs, expected):
    with expected:
        Boundary(a, name, **kwargs)


@pytest.mark.parametrize("attr, value", [
    ('context', ASyst('B')),
    ('port', 'outwards'),
    ('name', 'blade_runner'),
    ('variable', 'y'),
])
def test_Boundary_setattr_error(a, attr, value):
    x = Boundary(a, 'x')
    with pytest.raises(AttributeError):
        setattr(x, attr, value)


def test_Boundary_update_default_value_full_array(a):
    x = Boundary(a, 'u')

    x.mask = np.r_[True, False, True, False, True]
    x.update_default_value(np.r_[-3.14, 0.1, 0.2])

    assert np.all(x.mask == [True, False, True, False, True])
    assert x.default_value == pytest.approx([-3.14, 0.1, 0.2], abs=0)

    x.set_to_default()
    assert x.context[x.basename] == pytest.approx([-3.14, 0., 0.1, 0., 0.2], abs=0)


@pytest.mark.parametrize("name, mask, value", [
    ('x', None, 5),
    ('x', None, None),
    ('y', None, np.r_[0., 0.]),
    ('y', None, np.r_[1., 2.]),
    ('y', np.r_[True, False], np.r_[-3.14]),
    ('y', np.r_[False, True], np.r_[5.85]),
    ('y[1]', None, np.r_[-2.3]),
    ('y[1]', np.r_[False, True], np.r_[22]),
    ('y[1]', np.r_[True, False], np.r_[42.]),
    ('y[1]', None, -2.3),
    ('y[1]', np.r_[False, True], 22.),
    ('y[1]', np.r_[True, True], 42.),
    ('y[:]', None, np.r_[-2., -1.]),
    ('y[:]', np.r_[True, False], np.r_[-3.]),
    ('cc.r[:2]', np.r_[True, False, False], np.r_[-3.]),
    ('cc.g.d', None, 5.),
    ('b.cc.r', np.r_[True, False, True], np.r_[-3., 15.]),
    ('cc.get_g().x', None, np.r_[5., 6., 7.]),
    ('cc.get_g().x[1]', None, np.r_[2.]),
    ('cc.w', None, [4., 5., 6.]),
    ('cc.w[:2]', None, [10., 20.]),
    ('cc.w[:2]', np.r_[False, True, False], [10.]),
    ('cc.seq', None, [4., 5., 6.]),
    ('cc.seq[1]', None, [50.]),
    ('cc.h[1].seq2[1]', None, [50.]),
])
def test_Boundary_update_default_value(a, name, mask, value):
    x = Boundary(a, name)
    x.update_default_value(value, mask=mask)

    if mask is not None:
        assert np.all(x.mask == mask)
    assert x.default_value == pytest.approx(value, abs=0)


@pytest.mark.parametrize("name, value, context_value", [
    ('u[::2]', [1., 2., 3.], [1., 0., 2., 0., 3.]),
    ('in_.m', 1., 1.),
    ('x', 4., 4.),
    ('y', [1., 2.], [1., 2.]),
    ('y', np.zeros(2), [0, 0]),
    ('y[0]', [5.], [5., 2.]),
    ('y[1]', [5.], [1., 5.]),
    ('y[1]', np.array([-2.]), [1., -2.]),
    ('y[:]', [-3., 5.], [-3., 5.]),
    ('cc.r[:2]', np.r_[-3., 10.], [-3., 10., 1.]),
    ('cc.g.d', 5., 5.),
    ('b.cc.r', np.r_[-3., 10., 0.], [-3., 10., 0.]),
    ('cc.get_g().x', np.r_[-3., 10., 0.], [-3., 10., 0.]),
    ('cc.get_g().x[1]', np.r_[20.], [1., 20., 3.]),
    ('cc.w', [10., 20., 30.], [10., 20., 30.]),
    ('cc.w[:2]', [10., 20.], [10., 20., 6.]),
    ('cc.seq', [4., 5., 6.], [4., 5., 6.]),
    ('cc.seq[1]', [50.], [1., 50., 3.]),
    ('cc.h[1].seq2[1]', [50.], [1., 50., 3.]),
])
def test_Boundary_set_to_default(a, name, value, context_value):
    x = Boundary(a, name, default=value)
    x.set_to_default()

    ref_value = getattr(x.ref._obj, x.ref._key)
    if not isinstance(ref_value, np.ndarray) and Boundary.is_mutable_sequence(ref_value):
        ref_value = [ref_value.__getitem__(i) for i in range(x.ref._ref_size)]

    assert x.value == pytest.approx(value, rel=1e-14)
    if x._is_scalar:
        assert x.value == pytest.approx(context_value, rel=1e-14)
    else:
        assert ref_value == pytest.approx(context_value, rel=1e-14)


@pytest.mark.parametrize("name, value, clean", [
    ("in_.m", 0.1234, False),
    ("x", 1., True),
    ("x", -99, False),
    ("y", [1, 2], True),
    ("y", [0.7, 12.8], False),
    ("y[:]", [0.7, 12.8], False),
    ("y[0]", [1], True),
    ("y[0]", [-99], False),
    ("y[1]", [2], True),
    ("y[1]", [-99], False),
    ('cc.r[:2]', [1., 1.], True),
    ('cc.r[:2]', [-3., 4.], False),
    ('cc.g.d', 5., False),
    ('cc.get_g().x', [-3., 10., 0.], False),
    ('cc.get_g().x[1]', [2.], True),
    ('cc.w', [4., 5., 6.], True),
    ('cc.w[:2]', [10., 20.], False),
    ('cc.seq', [4., 5., 6.], False),
    ('cc.seq[1]', [2.], True),
    ('cc.h[1].seq2[1]', [50.], False),
])
def test_Boundary_clean_value(a, name, value, clean):
    x = Boundary(a, name)
    a.set_clean(PortType.IN)
    x.update_value(value)

    assert x.value == pytest.approx(value)
    assert a.is_clean(PortType.IN) == clean

@pytest.mark.parametrize("name, value, error", [
    ("x", [], TypeError),
    ("x", '0', TypeError),
    ("x", np.r_[5.], TypeError),
    ("y", [0.7], ValueError),
    ("y[1]", [0.5, 0.7], ValueError),
    ("y[1]", [[0.5], [0.7]], ValueError),
    ('cc.r[:2]', [-3.], ValueError),
    ('cc.seq', [10., 2.], ValueError),
    ('cc.seq', (5., 5., 5.), TypeError),
])
def test_Boundary_error_value(a, name, value, error):
    x = Boundary(a, name)
    a.set_clean(PortType.IN)

    with pytest.raises(error):
        x.update_value(value)
    assert a.is_clean(PortType.IN)

@pytest.mark.parametrize("name, mask, error", [
    ("y", [True], ValueError),
    ("y[1]", [True, True], ValueError),
    ("y[1]", [False, True, False], ValueError),
    ('cc.r[:2]', [False], ValueError),
    ('cc.r', [False], ValueError),
    ('cc.seq', [False], ValueError),
])
def test_Boundary_error_mask(a, name, mask, error):
    x = Boundary(a, name)
    x.update_default_value(x.value)
    with pytest.raises(error):
        x.mask = mask

@pytest.mark.parametrize("name", ["x", "y", "cc.seq", "cc.w"])
def test_Boundary_None_value(a, name):
    x = Boundary(a, name)
    init_value = x.value
    x.update_default_value(init_value)

    x.update_value(None)
    x.update_default_value(None)
    x.set_to_default()

    assert x.value == pytest.approx(init_value)
    assert x.default_value == pytest.approx(init_value)

@pytest.mark.parametrize("ctor_data", [
    (get_args('in_.m')),
    (get_args('x')),
    (get_args('inwards.x')),
    (get_args('y')),
    (get_args('y[0]')),
    (get_args('y[1]')),
    (get_args('y[:]')),
    (get_args('x', default=4.)),
    (get_args('y', default=[0, 0])),
    (get_args('y', default=np.zeros(2))),
    (get_args('y', default=1.234)),
    (get_args('y[1]', default=np.array([-2.]))),
    (get_args('y[1]', default=[-2.])),
    (get_args('y[1]', default=-2.)),
    (get_args('y[1:]', default=-2.)),
    (get_args('y[:-1]', default=-2.)),
    (get_args('y[:]', default=np.array([-2., 1.]))),
    (get_args('y[:]', default=[-2., 1.])),
    (get_args('y', mask=[True, False])),
    (get_args('y', mask=(False, False))),
    (get_args('y', mask=(True, True))),
    (get_args('y', mask=np.full(2, True))),
    (get_args('u[::2]')),
    (get_args('u[::2]', default=[1., 2., 3.])),
])
def test_Boundary_copy(a, ctor_data):
    args, kwargs = ctor_data
    old = Boundary(a, *args, **kwargs)
    new = old.copy()

    assert isinstance(new, Boundary)
    assert new is not old
    assert new.port is old.port
    assert new.context is old.context
    assert new.name == old.name
    assert new.variable == old.variable
    assert new.basename == old.basename
    assert np.array_equal(new.value, old.value)
    assert np.array_equal(new.default_value, old.default_value)
    assert new._ref is not old._ref 
    assert new._ref._obj is old._ref._obj
    assert new._ref._key is old._ref._key
    if not old._is_scalar:
        assert np.array_equiv(new.mask, old.mask)
        assert new.mask is not old.mask
    if isinstance(old.default_value, np.ndarray):
        assert new.default_value is not old.default_value
    if isinstance(old.value, np.ndarray):
        assert new.value is not old.value

