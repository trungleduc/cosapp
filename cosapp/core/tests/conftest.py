import pytest
import numpy
import scipy

from cosapp.systems import System
from cosapp.ports import Port
from cosapp.core.numerics.residues import Residue

# <codecell>

class BogusPort(Port):
    def setup(self):
        self.add_variable('q')

class FooBar(System):
    def setup(self):
        self.add_outward('z')
        self.add_input(BogusPort, 'in_', variables={'q': 5.})

class Bogus(System):
    def setup(self):
        self.add_property('g', 9.80665)
        self.add_property('NA', 6.02214076e23)
        self.add_inward('a', 2.0)
        self.add_inward('x', numpy.r_[0.1, -0.2, -3.14])
        self.add_inward('y', numpy.ones(3))
        self.add_outward('b', 0.5)

        self.add_child(FooBar('sub'))
        self.add_child(FooBar('B52'))
        self.add_output(BogusPort, 'out', variables={'q': 0.5})

@pytest.fixture(scope="function")
def eval_context():
    """Returns a bogus system used only to provide a valid context"""
    return Bogus('bogus')

# <codecell>

@pytest.fixture(scope="function")
def ufunc_test_data():
    ufunc_data = {
        'arange': {
            'expr': 'arange(0, 10, 2)',
            'check_val': numpy.arange(0, 10, 2),
        },
        'array': {
            'expr': 'array(x)',
            'func': numpy.array,
            'args': {'x': [-1., 0., 1.]},
        },
        'asarray': {
            'expr': 'asarray(x)',
            'func': numpy.asarray,
            'args': {'x': [-1., 0., 1.]},
        },
        'concatenate': {
            'expr': 'concatenate(x)',
            'func': numpy.concatenate,
            'args': {
                'x': (numpy.random.random(6), numpy.random.random(4)),
            },
        },
        'dot': {
            'expr': 'dot(x, y)',
            'func': numpy.dot,
            'args': {
                'x': numpy.random.random(6),
                'y': numpy.random.random(6),
            },
        },
        'evaluate_residue': {
            'expr': 'evaluate_residue(x, y)',
            'func': Residue.evaluate_residue,
            'args': {'x': 25., 'y': -12.},
        },
        'fmax': {
            'expr': 'fmax(x, y)',
            'func': numpy.fmax,
            'args': {
                'x': numpy.random.random(6),
                'y': numpy.random.random(6),
            },
        },
        'fmin': {
            'expr': 'fmin(x, y)',
            'func': numpy.fmin,
            'args': {
                'x': numpy.random.random(6),
                'y': numpy.random.random(6),
            },
        },
        'inner': {
            'expr': 'inner(x, y)',
            'func': numpy.inner,
            'args': {
                'x': numpy.random.random(6),
                'y': numpy.random.random(6),
            },
        },
        'isinf': {
            'expr': 'isinf(x)',
            'func': numpy.isinf,
            'args': {'x': [0, numpy.inf, 5.0]},
        },
        'isnan': {
            'expr': 'isnan(x)',
            'func': numpy.isnan,
            'args': {'x': [0, numpy.nan, numpy.nan]},
        },
        'kron': {
            'func': numpy.kron,
            'expr': 'kron(x, y)',
            'args': {
                'x': numpy.random.random(6),
                'y': numpy.random.random(6),
            },
        },
        'linspace': {
            'expr': 'linspace(0, 10, 50)',
            'check_val': numpy.linspace(0, 10, 50),
        },
        'matmul': {
            'expr': 'matmul(x, y)',
            'func': numpy.matmul,
            'args': {
                'x': numpy.random.random((3, 3)),
                'y': numpy.random.random((3, 1)),
            },
        },
        'maximum': {
            'expr': 'maximum(x, y)',
            'func': numpy.maximum,
            'args': {
                'x': numpy.random.random(6),
                'y': numpy.random.random(6),
            },
        },
        'minimum': {
            'expr': 'minimum(x, y)',
            'func': numpy.minimum,
            'args': {
                'x': numpy.random.random(6),
                'y': numpy.random.random(6),
            },
        },
        'ones': {
            'expr': 'ones(21)',
            'check_val': numpy.ones(21),
        },
        'full': {
            'expr': 'full((3, 7), 3.14159)',
            'check_val': numpy.full((3, 7), 3.14159),
        },
        'full_like': {
            'expr': 'full_like(x, 3.14159)',
            'check_val': numpy.full((3, 7), 3.14159),
            'args': {'x': numpy.random.random((3, 7))},
        },
        'cross': {
            'expr': 'cross(x, y)',
            'func': numpy.cross,
            'args': {
                'x': numpy.random.random(3),
                'y': numpy.random.random(3),
            },
        },
        'outer': {
            'expr': 'outer(x, y)',
            'func': numpy.outer,
            'args': {
                'x': numpy.random.random(6),
                'y': numpy.random.random(6),
            },
        },
        'power': {
            'expr': 'power(x, y)',
            'func': numpy.power,
            'args': {
                'x': numpy.random.random(6),
                'y': numpy.random.random(6) + 1.0,
            },
        },
        'residue_norm': {
            'expr': 'residue_norm(x, y)',
            'func': Residue.residue_norm,
            'args': {'x': 25., 'y': -12.},
            },
        'round': {
            'expr': 'round(x, y)',
            'func': numpy.round,
            'args': {
                'x': numpy.random.random(6),
                'y': 4,
            },
        },
        'tensordot': {
            'expr': 'tensordot(x, y)',
            'func': numpy.tensordot,
            'args': {
                'x': numpy.random.random((6, 6)),
                'y': numpy.random.random((6, 6)),
            }
        },
        'zeros': {
            'expr': 'zeros(21)',
            'check_val': numpy.zeros(21),
        },
    }

    # Add base types
    for dtype in (
        numpy.int8, numpy.int16, numpy.int32, numpy.int64,
        numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64,
        numpy.float32, numpy.float64,
        numpy.complex64, numpy.complex128,
    ):
        name = dtype.__name__
        ufunc_data[name] = {
            'expr': f'zeros(5, dtype={name})',
            'check_val': numpy.zeros(5, dtype=dtype)
        }

    def add_func(func, x=numpy.random.random(10), fname=None, alias=None, **kwargs):
        nonlocal ufunc_data
        fname = fname or func.__name__
        if alias is None:
            alias = fname
        args = {'x': x}
        args.update(kwargs)
        arg_list = ', '.join(args.keys())
        for name in {fname, alias}:
            ufunc_data[name] = {
                'func': func,
                'expr': f"{name}({arg_list})",
                'args': args,
            }

    # Add simple ufuncs
    for func in (
        numpy.degrees, numpy.radians,
        numpy.cos, numpy.sin, numpy.tan, numpy.arctan,
        numpy.cosh, numpy.sinh, numpy.tanh, numpy.arctanh,
        numpy.exp, numpy.expm1,
        numpy.log, numpy.log10, numpy.log1p,
        numpy.sqrt, numpy.cbrt,
        numpy.prod, numpy.sum,
        numpy.linalg.norm,
        scipy.special.factorial,
        scipy.special.erf,
        scipy.special.erfc,
    ):
        add_func(func, x=numpy.random.random(10))

    # Add common short names
    add_func(numpy.abs, fname='abs')
    add_func(numpy.arctan, alias='atan')
    add_func(numpy.arcsin, alias='asin', x=numpy.random.random(6) - 0.5)
    add_func(numpy.arccos, alias='acos', x=numpy.random.random(6) - 0.5)
    add_func(numpy.arctanh, alias='atanh')
    add_func(numpy.arcsinh, alias='asinh')
    add_func(numpy.arccosh, alias='acosh', x=numpy.random.random(6) + 1.1)
    add_func(
        numpy.arctan2,
        alias='atan2',
        x=numpy.random.random(10),
        y=numpy.random.random(10),
    )

    # Add constants
    for name in ('e', 'pi', 'inf'):
        ufunc_data[name] = {
            'expr': name,
            'check_val': getattr(numpy, name),
        }

    return ufunc_data
