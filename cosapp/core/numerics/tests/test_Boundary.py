import pytest
import numpy as np

from cosapp.systems import System
from cosapp.core.numerics.boundary import Boundary, MaskedVarInfo
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

class CustomClass:
    def __init__(self):
        self.r = np.ones(3)
        self.w = [4., 5., 6.]
        self.g = SubCustomClass()
        self.seq = MimicSeq()

    def get_g(self):
        return self.g


@pytest.fixture
def a():
    return ASyst('a')


@pytest.mark.parametrize("name, kwargs, expected", [
    ('in_.m', dict(), dict(portname='in_', name='in_.m', variable='m')),
    ('x', dict(), dict(variable='x')),
    ('inwards.x', dict(), dict(variable='x', name='x')),
    ('y', dict(), dict(variable='y', mask=np.full(2, True))),
    ('y[0]', dict(), dict(variable='y', mask=[True, False])),
    ('y[1]', dict(), dict(variable='y', mask=[False, True])),
    ('y[:]', dict(), dict(variable='y', mask=[True, True])),
    ('x', dict(default=4.), dict(variable='x', default_value=4)),
    ('y', dict(default=[0, 0]), dict(variable='y', default_value=np.zeros(2), mask=[True, True])),
    ('y', dict(default=np.zeros(2)), dict(variable='y', default_value=np.zeros(2), mask=[True, True])),
    ('y', dict(default=1.234), dict(variable='y', default_value=np.full(2, 1.234), mask=[True, True])),
    ('y[1]', dict(default=np.array([-2.])), dict(variable='y', default_value=np.array([-2]), mask=[False, True])),
    ('y[1]', dict(default=[-2.]), dict(variable='y', default_value=np.array([-2]), mask=[False, True])),
    ('y[1]', dict(default=-2.), dict(variable='y', default_value=np.array([-2]), mask=[False, True])),
    ('y[1:]', dict(default=-2.), dict(variable='y', default_value=np.array([-2]), mask=[False, True])),
    ('y[:-1]', dict(default=-2.), dict(variable='y', default_value=np.array([-2]), mask=[True, False])),
    ('y[:]', dict(default=np.array([-2., 1.])), dict(variable='y', default_value=[-2, 1], mask=[True, True])),
    ('y[:]', dict(default=[-2., 1.]), dict(variable='y', default_value=[-2, 1], mask=[True, True])),
    ('y', dict(mask=[True, False]), dict(variable='y', mask=[True, False])),
    ('y', dict(mask=(False, False)), dict(variable='y', mask=[False, False])),
    ('y', dict(mask=(True, True)), dict(variable='y', mask=[True, True])),
    ('y', dict(mask=np.full(2, True)), dict(variable='y', mask=[True, True])),
    ('u[::2]', dict(), dict(variable='u', mask=[True, False, True, False, True])),
    ('u[::2]', dict(default=[1., 2., 3.]), dict(variable='u', default_value=[1., 2., 3.], mask=[True, False, True, False, True])),
    ('cc.r[:2]', dict(default=[1., 2.]), dict(basename='cc.r', variable='r', default_value=[1., 2.], mask=[True, True, False])),
    ('cc.g.d', dict(default=15.), dict(name='cc.g.d', variable='d', default_value=15.)),
    ('b.cc.r', dict(default=[5., 5., 5.]), dict(portname="b.inwards", name='b.cc.r', variable='r', default_value=[5., 5., 5.], mask=[True, True, True])),
    ('cc.get_g().x', dict(), dict(basename="cc.get_g().x", variable='x', mask=[True, True, True])),
    ('cc.get_g().x[1]', dict(), dict(basename="cc.get_g().x", variable='x', mask=[False, True, False])),
    ('cc.w', dict(), dict(basename="cc.w", variable='w', mask=[True, True, True])),
    ('cc.w[:2]', dict(), dict(basename="cc.w", variable='w', mask=[True, True, False])),
    ('cc.seq', dict(), dict(basename="cc.seq", variable='seq', mask=[True, True, True])),
    ('cc.seq[1]', dict(), dict(basename="cc.seq", variable='seq', mask=[False, True, False])),
    # ('', dict(), dict(variable='', default_value=, mask=[True, True])),
])
def test_Boundary___init__(a, name, kwargs, expected):
    x = Boundary(a, name, **kwargs)
    # Set of expected attributes:
    portname = expected.pop('portname', 'inwards')
    expected.setdefault('variable', name)
    expected.setdefault('name', name)
    expected.setdefault('default_value', None)
    # Test object attributes:
    assert x.context is a
    assert x.port is a[portname]
    mask = expected.pop('mask', None)
    if mask is None:
        assert not hasattr(x._ref, "mask")
    else:
        assert isinstance(x.mask, np.ndarray)
        assert np.all(x.mask == np.asarray(mask))
    for attr, value in expected.items():
        err_msg = f"for attribute {attr!r}"
        x_attr = getattr(x, attr)
        if isinstance(value, (str, type(None))):
            assert x_attr == value, err_msg
        else:
            assert x_attr == pytest.approx(value), err_msg


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
    ('cc.get_g()', dict(), pytest.raises(TypeError, match="Type of evaluated expression is incompatible")),
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


@pytest.mark.parametrize("name, kwargs, mask, expected", [
    ('x', dict(), [False], None),
    ('x', dict(), [True], None),
    ('y', dict(), [True, False], [True, False]),
    ('y', dict(), [False, False], [False, False]),
    ('y[1]', dict(), [True, True], [True, True]),
])
def test_Boundary_mask(a, name, kwargs, mask, expected):
    x = Boundary(a, name, **kwargs)
    x.mask = np.asarray(mask)
    if expected is None:
        assert not hasattr(x, "mask")
    else:
        assert np.all(x.mask == expected)

def test_Boundary_update_default_value_full_array(a):
    x = Boundary(a, 'u')

    x.mask = np.r_[True, False, True, False, True]
    x.update_default_value(np.r_[-3.14, 0.1, 0.2])

    assert np.all(x.mask == [True, False, True, False, True])
    assert x.default_value == pytest.approx([-3.14, 0.1, 0.2], abs=0)

    x.set_to_default()
    assert x.context[x.basename] == pytest.approx([-3.14, 0., 0.1, 0., 0.2], abs=0)

@pytest.mark.parametrize("name, kwargs, data", [
    ('x', dict(), dict(value=5)),
    ('x', dict(), dict(value=None)),
    ('y', dict(), dict(value=np.r_[0., 0.])),
    ('y', dict(), dict(value=np.r_[1., 2.])),
    ('y', dict(mask=np.r_[True, False]), dict(value=np.r_[-3.14], expected=[-3.14, 2])),
    ('y', dict(mask=np.r_[False, True]), dict(value=np.r_[5.85], expected=[1, 5.85])),
    ('y[1]', dict(), dict(value=np.r_[-2.3], expected=[1, -2.3], mask=[False, True])),
    ('y[1]', dict(mask=np.r_[False, True]), dict(value=np.r_[22], expected=[1., 22], mask=[False, True])),
    ('y[1]', dict(mask=np.r_[True, False]), dict(value=np.r_[42.], expected=np.r_[42., 2.], mask=[True, False])),
    ('y[1]', dict(), dict(value=-2.3, expected=[1., -2.3])),
    ('y[1]', dict(mask=np.r_[False, True]), dict(value=22., expected=[1., 22.])),
    ('y[1]', dict(mask=np.r_[True, True]), dict(value=42., expected=[42., 42.], mask=[True, True])),
    ('y[:]', dict(), dict(value=np.r_[-2., -1.])),
    ('y[:]', dict(mask=np.r_[True, False]), dict(value=np.r_[-3.], expected=[-3, 2])),
    ('cc.r[:2]', dict(mask=np.r_[True, False, False]), dict(value=np.r_[-3.], expected=[-3., 1., 1.])),
    ('cc.g.d', dict(), dict(value=5.)),
    ('b.cc.r', dict(mask=np.r_[True, False, True]), dict(value=np.r_[-3., 15.], expected=[-3., 1., 15.])),
    ('cc.get_g().x', dict(), dict(value=np.r_[5., 6., 7.], mask=[True, True, True])),
    ('cc.get_g().x[1]', dict(), dict(value=np.r_[2.], expected=[1., 2., 3.], mask=[False, True, False])),
    ('cc.w', dict(), dict(value=[4., 5., 6.], mask=[True, True, True])),
    ('cc.w[:2]', dict(), dict(value=[10., 20.], expected=[10., 20., 6.], mask=[True, True, False])),
    ('cc.w[:2]', dict(mask=np.r_[False, True, False]), dict(value=[10.], expected=[4., 10., 6.], mask=[False, True, False])),
    ('cc.seq', dict(), dict(value=[4., 5., 6.], expected=[4., 5., 6.], mask=[True, True, True])),
    ('cc.seq[1]', dict(), dict(value=[50.], expected=[1., 50., 3.], mask=[False, True, False])),
])
def test_Boundary_update_default_value(a, name, kwargs, data):
    x = Boundary(a, name)
    # Filter test data
    expected = data.pop('expected', data['value'])
    mask = data.pop('mask', None)  # assumes that x.mask is unchanged if not present in 'data'
    # Test method 'update_default_value'
    value = data['value']
    x.update_default_value(value, **kwargs)
    x.set_to_default()
    if mask is not None:
        assert np.all(x.mask == mask)
    if expected is None:
        assert x.default_value is None
    else:
        assert np.allclose(x.default_value, value, atol=0, equal_nan=True)

        ref_value = getattr(x.ref._obj, x.ref._key)
        if not isinstance(ref_value, np.ndarray) and Boundary.is_mutable_sequence(ref_value):
            ref_value = [ref_value.__getitem__(i) for i in range(x.ref._ref_size)]
            assert np.allclose(ref_value, expected, atol=0, equal_nan=True)
        else:
            assert np.allclose(ref_value, expected, atol=0, equal_nan=True)


@pytest.mark.parametrize("name, kwargs, expected", [
    ('u[::2]', dict(default=[1., 2., 3.]), dict(value=[1., 2., 3.], context_value=[1., 0., 2., 0., 3.])),
    ('in_.m', dict(), dict(value=1)),
    ('x', dict(), dict(value=1)),
    ('y',    dict(), dict(value=[1, 2])),
    ('y[:]', dict(), dict(value=[1, 2])),
    ('y[0]', dict(), dict(context_value=[1, 2], value=[1])),
    ('y[1]', dict(), dict(context_value=[1, 2], value=[2])),
    ('x', dict(default=4.), dict(value=4)),
    ('y', dict(default=np.zeros(2)), dict(value=[0, 0])),
    ('y[0]', dict(default=[-2.]), dict(value=[-2.], context_value=[-2., 2.])),
    ('y[1]', dict(default=[-2.]), dict(value=[-2.], context_value=[1., -2.])),
    ('y[1]', dict(default=np.array([-2.])), dict(value=[-2.], context_value=[1., -2.])),
    ('y[:]', dict(default=[-3., 5.]), dict(value=[-3., 5.])),
    ('cc.r[:2]', dict(default=np.array([-3., 10.])), dict(value=np.r_[-3., 10.], context_value=[-3., 10., 1.])),
    ('cc.g.d', dict(), dict(value=5.)),
    ('b.cc.r', dict(default=np.array([-3., 10., 0.])), dict(value=np.r_[-3., 10., 0.], context_value=[-3., 10., 0.])),
    ('cc.get_g().x', dict(default=np.array([-3., 10., 0.])), dict(value=np.r_[-3., 10., 0.], context_value=[-3., 10., 0.])),
    ('cc.get_g().x[1]', dict(default=np.r_[20.]), dict(value=np.r_[20.], context_value=[1., 20., 3.])),
    ('cc.w', dict(), dict(value=[10., 20., 30.], context_value=[10., 20., 30.])),
    ('cc.w[:2]', dict(default=[10., 20.]), dict(value=[10., 20.], context_value=[10., 20., 6.])),
    ('cc.seq', dict(default=[4., 5., 6.]), dict(value=[4., 5., 6.], context_value=[4., 5., 6.])),
    ('cc.seq[1]', dict(default=[50.]), dict(value=[50.], context_value=[1., 50., 3.])),
])
def test_Boundary_set_to_default(a, name, kwargs, expected):
    x = Boundary(a, name, **kwargs)
    # Set expected values
    expected.setdefault('context_value', expected['value'])
    # Test
    if "default" not in kwargs:
        x.update_default_value(expected['value'])
    x.set_to_default()
    assert x.value == pytest.approx(expected['value'], rel=1e-14)
    
    ref_value = getattr(x.ref._obj, x.ref._key)
    if not isinstance(ref_value, np.ndarray) and Boundary.is_mutable_sequence(ref_value):
        ref_value = [ref_value.__getitem__(i) for i in range(x.ref._ref_size)]
    if x._is_scalar:
        assert x.value == pytest.approx(expected['context_value'], rel=1e-14)
    else:
        assert ref_value == pytest.approx(expected['context_value'], rel=1e-14)


@pytest.mark.parametrize("name, kwargs, data", [
    ("in_.m", dict(), dict(value=0.1234)),
    ("x", dict(), dict(value=1, clean=True)),
    ("x", dict(), dict(value=0.1234)),
    ("y", dict(), dict(value=[1, 2], clean=True)),
    ("y", dict(), dict(value=[0.7, 12.8])),
    ("y", dict(), dict(value=[0.7], error=ValueError)),
    ("x", dict(), dict(value=[], error=TypeError)),
    ("x", dict(), dict(value='0', error=TypeError)),
    ("y[:]", dict(), dict(value=[0.7, 12.8])),
    ("y[0]", dict(), dict(value=[1], context_value=[1, 2], clean=True)),
    ("y[1]", dict(), dict(value=[2], context_value=[1, 2], clean=True)),
    ("y[0]", dict(), dict(value=[-99], context_value=[-99, 2])),
    ("y[1]", dict(), dict(value=[-99], context_value=[1, -99])),
    ("x", dict(mask=None), dict(value=-99, expected=-99, context_value=-99, clean=False)),
    ("y", dict(mask=[False, False]), dict(value=[], expected=[], context_value=[1, 2], clean=True)),
    ('cc.r[:2]', dict(), dict(value=[-3., 4.], context_value=[-3., 4., 1.])),
    ('cc.r[:2]', dict(), dict(value=[-3.],  error=ValueError)),
    ('cc.g.d', dict(), dict(value=5.)),
])
def test_Boundary_value(a, name, kwargs, data):
    x = Boundary(a, name, **kwargs)
    a.set_clean(PortType.IN)
    # Test
    value = data['value']
    error = data.get('error', None)

    if error is None:
        x.update_value(value)  # tested setter
        context_value = data.get('context_value', value)
        expected = data.get('expected', value)
        ref_value = getattr(x.ref._obj, x.ref._key)
        if x._is_scalar:
            assert x.value == pytest.approx(context_value)
        else:
            assert ref_value == pytest.approx(context_value)
        assert x.value == pytest.approx(expected)
        assert a.is_clean(PortType.IN) == data.get('clean', False)

    else:
        with pytest.raises(error):
            x.update_value(value)
        assert a.is_clean(PortType.IN)


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


def get_indices(context: System, name: str, mask: Optional[np.ndarray] = None) -> Tuple:
    basename, selector = Boundary.parse_expression(name)
    _, mask = Boundary.create_mask(context, basename, selector, mask)
    return MaskedVarInfo(basename, selector, mask)

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

    info = get_indices(top, name)
    assert info.basename == expected.get('basename', name)
    assert info.selector == expected.get('selector', '')
    assert info.fullname == name
    assert isinstance(info.mask, np.ndarray)
    assert np.array_equal(info.mask, expected['mask'])

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

    r = get_indices(top, f"sub.a{selector}")
    assert r.basename == "sub.a"
    assert r.selector == selector
    assert r.fullname == f"sub.a{selector.strip()}"
    assert np.array_equal(r.mask, expected)
