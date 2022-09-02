"""Utility functions for testing purposes"""
import numpy
import itertools
from numbers import Number
from contextlib import contextmanager
from typing import Tuple, Dict, Any, Union, Iterable


ArgsKwargs = Tuple[Tuple[Any], Dict[str, Any]]


def assert_keys(dictionary, *keys):
    """Utility function to test dictionary keys"""
    if set(dictionary.keys()) != set(keys):
        actual = list(dictionary.keys())
        raise KeyError(f"actual: {actual}; expected: {list(keys)}") 


def assert_all_type(collection, dtype):
    """Asserts that all the elements of a collecion are of a given type"""
    def assert_type(name, value):
        if not isinstance(value, dtype):
            actual = type(value).__name__
            raise TypeError(
                f"{name} is of type {actual}; expected {dtype.__name__}"
            ) 
    if isinstance(collection, dict):
        for key, value in collection.items():
            assert_type(f"element with key '{key}'", value)
    else:
        for i, value in enumerate(collection):
            assert_type(f"element #{i}", value)


def rel_error(actual: Union[Number, Iterable], expected: Union[Number, Iterable]) -> Union[float, numpy.ndarray]:
    """Computes the relative error of `actual` compared to `expected`
    """
    error = lambda a, x: abs(a) if x == 0 else abs(a / x - 1)

    if isinstance(actual, Number):
        return error(actual, expected)
    
    actual = numpy.asarray(actual)

    if isinstance(expected, Number):
        iterator = (error(a, expected) for a in actual.flat)
    else:
        iterator = itertools.starmap(error,
            zip(actual.flat, numpy.ravel(expected))
        )
    errors = numpy.fromiter(
        iterator,
        dtype=float,
        count=actual.size,
    )
    return errors.reshape(actual.shape)


def get_args(*args, **kwargs) -> ArgsKwargs:
    """Utility function to collect args and kwargs in a tuple"""
    return args, kwargs


@contextmanager
def no_exception():
    """Context manager to assert that a block does not raise any exception"""
    try:
        yield

    except Exception as error:
        raise AssertionError(f"Unexpected exception raised: {error!r}")


@contextmanager
def not_raised(ExpectedException):
    """Context manager to assert that a block does not raise `ExpectedException`"""
    # https://gist.github.com/oisinmulvihill/45c14271fad7794a4a52516ecb784e69
    try:
        yield

    except ExpectedException as error:
        raise AssertionError(f"Raised {error!r} exception when it should not!")

    except Exception as error:
        raise AssertionError(f"Unexpected exception raised: {error!r}")
