import pytest

import enum
import numpy as np

from cosapp.utils.helpers import is_numerical, is_number, check_arg
from cosapp.utils.testing import no_exception


@pytest.mark.parametrize("value, expected", [
    (True, False),
    (1, True),
    (1.e-5, True),
    (np.asarray(True), False),
    (np.asarray(1), True),
    (np.asarray(1.e-5), True),
    (np.asarray('str'), False),
    ('string', False),
    ('', False),
    ([1, 2, 3], True),
    (['a', 'b', 'c'], False),
    ([1, 2, 'b'], False),
    ([], False),
    ([[]], False),
    ((1, 2, 3), True),
    (('a', 'b', 'c'), False),
    ((1, 2, 'b'), False),
    ((), False),
    ({'a', 'b', 'c'}, False),
    ({1, 2, 'b'}, False),
    ({1, 2, 3}, True),
    ({}, False),
    (frozenset([1, 2, 3]), True),
    (frozenset(['a', 'b', 'c']), False),
    (frozenset([1, 2, 'b']), False),
    (frozenset([]), False),
    ({'a': 1, 'b': 2, 'c': 3}, False),
    (np.ones(4), True),
    (np.asarray(['a', 'b', 'c']), False),
    (np.asarray([], dtype=bool), False),
    (np.asarray([], dtype=float), True),
    (np.asarray([], dtype=int), True),
    (np.asarray([], dtype=str), False),
])
def test_is_numerical(value, expected):
    assert is_numerical(value) == expected


@pytest.mark.parametrize("value, expected", [
    (True, False),
    (1, True),
    (1.e-5, True),
    (np.asarray(True), False),
    (np.asarray(1), True),
    (np.asarray(1.e-5), True),
    (np.asarray('str'), False),
    ('string', False),
    ('', False),
    ([1, 2, 3], False),
    (['a', 'b', 'c'], False),
    ([1, 2, 'b'], False),
    ([], False),
    ([[]], False),
    ((1, 2, 3), False),
    (('a', 'b', 'c'), False),
    ((1, 2, 'b'), False),
    ((), False),
    ({'a', 'b', 'c'}, False),
    ({1, 2, 'b'}, False),
    ({1, 2, 3}, False),
    ({}, False),
    (frozenset([1, 2, 3]), False),
    (frozenset(['a', 'b', 'c']), False),
    (frozenset([1, 2, 'b']), False),
    (frozenset([]), False),
    ({'a': 1, 'b': 2, 'c': 3}, False),
    (np.ones(4), False),
    (np.asarray(['a', 'b', 'c']), False),
    (np.asarray([], dtype=bool), False),
    (np.asarray([], dtype=float), False),
    (np.asarray([], dtype=int), False),
    (np.asarray([], dtype=str), False),
])
def test_is_number(value, expected):
    assert is_number(value) == expected


class DummyEnum(enum.Enum):
    A = enum.auto()
    B = enum.auto()
    C = enum.auto()


@pytest.mark.parametrize("args, expected", [
    ((0, 'var', int), dict()),
    ((0, 'var', (int, float)), dict()),
    ((0.1, 'var', int), dict(error=TypeError)),
    ((1, 'var', int, lambda n: 0 < n <= 2), dict()),
    ((2, 'var', int, lambda n: 0 < n <= 2), dict()),
    ((0, 'var', int, lambda n: 0 < n <= 2), dict(error=ValueError)),
    ((3, 'var', int, lambda n: 0 < n <= 2), dict(error=ValueError)),
    ((3.14, 'var', (int, str)), dict(error=TypeError)),
    ((3.14, 'var', (int, float, str)), dict()),
    ((3.14, 'var', float), dict()),
    ((3.14, 'var', (int, float, str), lambda x: abs(np.sin(x)) < 1e-12), dict(error=ValueError)),
    (('foo', 'var', (int, float)), dict(error=TypeError)),
    (('foo', 'var', (int, float, str)), dict()),
    (('_foo', 'var', str), dict()),
    (('_foo', 'var', str, lambda s: not s.startswith('_')), dict(error=ValueError)),
    (((1, 2), 'interval', (tuple, list), lambda it: len(it) == 2), dict()),
    (((1, 2, 3), 'interval', (tuple, list), lambda it: len(it) == 2), dict(error=ValueError)),
    (('string', 'interval', (tuple, list), lambda it: len(it) == 2), dict(error=TypeError, match=".*got str")),
    (('foo', 'var', DummyEnum), dict(error=TypeError)),
    ((DummyEnum.A, 'var', DummyEnum), dict())
])
def test_check_arg(args, expected):
    """Test of utility function check_arg"""
    error = expected.get('error', None)

    if error is None:
        with no_exception():
            check_arg(*args)

    else:
        with pytest.raises(error, match=expected.get('match', None)):
            check_arg(*args)
