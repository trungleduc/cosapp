import enum

import numpy as np
import pytest

from cosapp.utils.helpers import is_numerical, is_number, check_arg


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
    (np.asarray([], dtype=np.bool), False),
    (np.asarray([], dtype=np.float), True),
    (np.asarray([], dtype=np.int), True),
    (np.asarray([], dtype=np.unicode), False),
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
    (np.asarray([], dtype=np.bool), False),
    (np.asarray([], dtype=np.float), False),
    (np.asarray([], dtype=np.int), False),
    (np.asarray([], dtype=np.unicode), False),
])
def test_is_number(value, expected):
    assert is_number(value) == expected


class DummyEnum(enum.Enum):
    A = enum.auto()
    B = enum.auto()
    C = enum.auto()


@pytest.mark.parametrize("args, expected", [
    ((0, 'var', int), None),
    ((0, 'var', (int, float)), None),
    ((0.1, 'var', int), (TypeError, )),
    ((1, 'var', int, lambda n: 0 < n <= 2), None),
    ((2, 'var', int, lambda n: 0 < n <= 2), None),
    ((0, 'var', int, lambda n: 0 < n <= 2), (ValueError, )),
    ((3, 'var', int, lambda n: 0 < n <= 2), (ValueError, )),
    ((3.14, 'var', (int, str)), (TypeError, )),
    ((3.14, 'var', (int, float, str)), None),
    ((3.14, 'var', float), None),
    ((3.14, 'var', (int, float, str), lambda x: abs(np.sin(x)) < 1e-12), (ValueError, )),
    (('foo', 'var', (int, float)), (TypeError, )),
    (('foo', 'var', (int, float, str)), None),
    (('_foo', 'var', str), None),
    (('_foo', 'var', str, lambda s: not s.startswith('_')), (ValueError, )),
    (((1, 2), 'interval', (tuple, list), lambda it: len(it) == 2), None),
    (((1, 2, 3), 'interval', (tuple, list), lambda it: len(it) == 2), (ValueError, )),
    (('string', 'interval', (tuple, list), lambda it: len(it) == 2), (TypeError, ".*got str")),
    (('foo', 'var', DummyEnum), (TypeError, )),
    ((DummyEnum.A, 'var', DummyEnum), None)
])
def test_check_arg(args, expected):
    """Test of utility function check_arg"""

    def expect_pass(args):
        try:
            check_arg(*args)
        except Exception as ex:
            self.fail(str(ex) + "\nArguments: " + str(args))

    def expect_fail(args, exception, match=None):
        with pytest.raises(exception, match=match):
            check_arg(*args)

    if expected is None:
        expect_pass(args)
    else:
        expect_fail(args, *expected)
