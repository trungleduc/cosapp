import pytest
import numpy as np

from cosapp.systems import System
from cosapp.core.numerics.boundary import Boundary
from cosapp.ports.port import Port, PortType
from cosapp.utils.testing import get_args
from contextlib import nullcontext as does_not_raise


class APort(Port):
    def setup(self):
        self.add_variable('m')
        self.add_variable('n', np.array([3., 4.]))

class ASyst(System):
    def setup(self, **kwargs):
        self.add_input(APort, 'in_')
        self.add_inward('x')
        self.add_inward('y', np.array([1., 2.]))
        self.add_inward('u', np.zeros(5))
        self.add_outward('v')


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
    ('x', dict(mask=[False]), dict(variable='x', mask=[False])),
    ('x', dict(mask=[True]), dict(variable='x', mask=None)),
    ('y', dict(mask=[True, False]), dict(variable='y', mask=[True, False])),
    ('y', dict(mask=(False, False)), dict(variable='y', mask=[False, False])),
    ('y', dict(mask=(True, True)), dict(variable='y', mask=[True, True])),
    ('y', dict(mask=np.full(2, True)), dict(variable='y', mask=[True, True])),
    ('u[::2]', dict(), dict(variable='u', mask=[True, False, True, False, True])),
    ('u[::2]', dict(default=[1., 2., 3.]), dict(variable='u', default_value=[1., 2., 3.], mask=[True, False, True, False, True])),
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
        assert x.mask is None
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
    ('in_', dict(), pytest.raises(TypeError)),
    ('inwards', dict(), pytest.raises(TypeError)),
    ('outwards', dict(), pytest.raises(TypeError)),
    ('y[', dict(), pytest.raises(SyntaxError)),
    ('y(', dict(), pytest.raises(SyntaxError)),
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
    ('x', dict(), [False], [False]),
    ('x', dict(), [True], None),
    ('y', dict(), [True, False], [True, False]),
    ('y', dict(), [False, False], [False, False]),
    ('y[1]', dict(), [True, True], [True, True]),
])
def test_Boundary_mask(a, name, kwargs, mask, expected):
    x = Boundary(a, name, **kwargs)
    x.mask = np.asarray(mask)
    if expected is None:
        assert x.mask is None
    else:
        assert np.all(x.mask == expected)


def test_Boundary_default_value(a):
    x = Boundary(a, 'x')
    with pytest.raises(AttributeError, match="can't set attribute|no setter"):
        setattr(x, 'default_value', 25.)


def test_Boundary_set_default_value_full_array(a):
    x = Boundary(a, 'y')

    x.set_default_value(np.r_[-3.14, 5.85], mask=np.r_[True, False])

    assert np.all(x.mask == [True, True])
    assert x.default_value == pytest.approx([-3.14, 2], abs=0)


@pytest.mark.parametrize("name, kwargs, data", [
    ('x', dict(), dict(value=5)),
    ('x', dict(), dict(value=None)),
    ('y', dict(), dict(value=np.r_[0., 0.])),
    ('y', dict(), dict(value=np.r_[1., 2.])),
    ('y', dict(mask=np.r_[True, False]), dict(value=np.r_[-3.14, 5.85], expected=[-3.14, 2])),
    ('y', dict(mask=np.r_[False, True]), dict(value=np.r_[-3.14, 5.85], expected=[1, 5.85])),
    ('y[1]', dict(), dict(value=np.r_[-2.3], mask=[False, True])),
    ('y[1]', dict(mask=np.r_[False, True]), dict(value=np.r_[-3, 22], expected=[22], mask=[False, True])),
    ('y[1]', dict(mask=np.r_[True, False]), dict(value=np.r_[42., 7.], expected=np.r_[42., 2.], mask=[True, True])),
    ('y[1]', dict(), dict(value=-2.3, expected=[-2.3])),
    ('y[1]', dict(mask=np.r_[False, True]), dict(value=22., expected=[22])),
    ('y[1]', dict(mask=np.r_[True, False]), dict(value=42., expected=[42., np.nan], mask=[True, True])),
    ('y[:]', dict(), dict(value=np.r_[-2., -1.])),
    ('y[:]', dict(mask=np.r_[True, False]), dict(value=np.r_[-3., 22.], expected=[-3, 2])),
    # (, dict(), dict(value=, expected=, mask=)),
])
def test_Boundary_set_default_value(a, name, kwargs, data):
    x = Boundary(a, name)
    # Filter test data
    expected = data.pop('expected', data['value'])
    mask = data.pop('mask', x.mask)  # assumes that x.mask is unchanged if not present in 'data'
    # Test method 'set_default_value'
    x.set_default_value(data['value'], **kwargs)
    assert np.all(x.mask == mask)
    if expected is None:
        assert x.default_value is None
    else:
        assert np.allclose(x.default_value, expected, atol=0, equal_nan=True)


@pytest.mark.parametrize("args, expected", [
    ((4., None, 5., None), dict(value=4, mask=None)),
    ((np.r_[0.1, 0.2], np.r_[True, True], np.r_[-3., -1.], np.r_[True, True]), dict(value=[0.1, 0.2], mask=[True, True])),
    ((np.r_[0.1, 0.2], np.r_[False, True], np.r_[-3., -1.], np.r_[True, False]), dict(value=[-3., 0.2], mask=[True, True])),
    ((np.r_[0.1, 0.2], np.r_[False, True], np.r_[-3., -1.], np.r_[False, False]), dict(value=[np.nan, 0.2], mask=[False, True])),
])
def test_Boundary__merge_masked_array(args, expected):
    value, mask = Boundary._merge_masked_array(*args)
    if isinstance(value, np.ndarray):
        assert np.allclose(value, expected['value'], atol=0, equal_nan=True)
    else:
        assert value == pytest.approx(expected['value'], abs=0)
    if expected['mask'] is None:
        assert mask is None
    else:
        assert isinstance(mask, np.ndarray)
        assert np.all(mask == expected['mask'])


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
])
def test_Boundary_set_to_default(a, name, kwargs, expected):
    x = Boundary(a, name, **kwargs)
    # Set expected values
    expected.setdefault('context_value', expected['value'])
    attr = x.basename
    # Test
    x.set_to_default()
    assert a[attr] == pytest.approx(expected['context_value'], rel=1e-14)
    assert x.value == pytest.approx(expected['value'], rel=1e-14)


@pytest.mark.parametrize("name, kwargs, data", [
    ("in_.m", dict(), dict(value=0.1234)),
    ("x", dict(), dict(value=1, clean=True)),
    ("x", dict(), dict(value=0.1234)),
    ("y", dict(), dict(value=[1, 2], clean=True)),
    ("y", dict(), dict(value=[0.7, 12.8])),
    # ("y", dict(), dict(value=[0.7], error=ValueError)),  # should arguably raise an exception, but does not
    ("x", dict(), dict(value=[], error=TypeError)),
    ("x", dict(), dict(value='0', error=TypeError)),
    ("y[:]", dict(), dict(value=[0.7, 12.8])),
    ("y[0]", dict(), dict(value=[1], context_value=[1, 2], clean=True)),
    ("y[1]", dict(), dict(value=[2], context_value=[1, 2], clean=True)),
    ("y[0]", dict(), dict(value=[-99], context_value=[-99, 2])),
    ("y[1]", dict(), dict(value=[-99], context_value=[1, -99])),
    ("x", dict(mask=[False]), dict(value=-99, expected=[], context_value=1, clean=True)),
    ("y", dict(mask=[False, False]), dict(value=[-0.1, 6.3], expected=[], context_value=[1, 2], clean=True)),
])
def test_Boundary_value(a, name, kwargs, data):
    x = Boundary(a, name, **kwargs)
    a.set_clean(PortType.IN)
    # Test
    value = data['value']
    error = data.get('error', None)

    if error is None:
        x.value = value  # tested setter
        attr = x.basename
        context_value = data.get('context_value', value)
        expected = data.get('expected', value)
        # assert np.shape(x.value) == np.shape(expected)
        assert x.value == pytest.approx(expected)
        assert a[attr] == pytest.approx(context_value)
        assert a.is_clean(PortType.IN) == data.get('clean', False)

    else:
        with pytest.raises(error):
            x.value = value
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
    (get_args('x', mask=[False])),
    (get_args('x', mask=[True])),
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
    assert new.ref is old.ref
    assert new.port is old.port
    assert new.context is old.context
    assert new.name == old.name
    assert new.variable == old.variable
    assert new.basename == old.basename
    assert np.array_equiv(new.mask, old.mask)
    assert np.array_equal(new.value, old.value)
    assert np.array_equal(new.default_value, old.default_value)
    if old.mask is not None:
        assert new.mask is not old.mask
    if isinstance(old.value, np.ndarray):
        assert new.value is not old.value
    if isinstance(old.default_value, np.ndarray):
        assert new.default_value is not old.default_value
