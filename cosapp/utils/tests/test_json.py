import base64
import io
import json
import pickle

import numpy
import pytest

from cosapp.utils.json import JSONEncoder, decode_cosapp_dict


def encode(obj: bytes) -> str:
    return base64.b64encode(obj).decode("utf-8")


def decode(obj: str) -> bytes:
    return base64.b64decode(bytes(data, encoding="utf-8"))


def ndarray_to_bytes(arr) -> bytes:
    buf = io.BytesIO()
    numpy.save(buf, arr)
    return buf.getvalue()


class Jsonable:

    CONTENT = {"a": [1, 2]}

    def __json__(self):
        return json.dumps(Jsonable.CONTENT)


class AnyClass:

    def __init__(self):
        self.a = 22.
        self.b = object()

    def __eq__(self, o):
        r = False
        if isinstance(o, AnyClass):
            r = o.a == self.a and isinstance(o.b, object)
        return r


class UnPickable:

    def __getstate__(self):
        raise pickle.PicklingError()


@pytest.mark.parametrize("value, expected", [
    (b"hello", "data:text/plain;base64," + encode(b"hello")),
    (numpy.array([1., 1.]), "data:application/vnd.numpy.ndarray;base64," + encode(ndarray_to_bytes(numpy.array([1., 1.])))),
    (Jsonable(), json.dumps(Jsonable.CONTENT)),
    (AnyClass(), "data:application/vnd.python3.pickle;base64," + encode(pickle.dumps(AnyClass(), protocol=4))),
    (UnPickable(), TypeError),
])
def test_JSONEncoder(value, expected):
    encoder = JSONEncoder()
    
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            encoder.default(value)
    else:
        assert encoder.default(value) == expected


@pytest.mark.parametrize("value, expected", [
    ({"hello": "data:text/plain;base64," + encode(b"hello")}, {"hello": b"hello"}),
    ({1: {"hello": "data:text/plain;base64," + encode(b"hello")}}, {1: {"hello": b"hello"}}),
    ({"pickled": "data:application/vnd.python3.pickle;base64," + encode(pickle.dumps(AnyClass(), protocol=4))}, {"pickled": AnyClass()}),
    ({"hello": "data:application/unknown;base64," + encode(b"hello")}, ValueError),
])
def test_decode_cosapp_dict(value, expected):
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            decode_cosapp_dict(value)
    else:
        assert decode_cosapp_dict(value) == expected


def test_decode_cosapp_dict_array():
    value, expected = ({"array": "data:application/vnd.numpy.ndarray;base64," + encode(ndarray_to_bytes(numpy.array([1., 1.])))}, {"array": numpy.array([1., 1.])})

    decoded_value = decode_cosapp_dict(value)

    assert len(decoded_value) == len(expected)
    numpy.testing.assert_array_equal(decoded_value["array"], expected["array"])
