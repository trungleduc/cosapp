import pytest

import json
from cosapp.utils.orderedset import OrderedSet
from random import shuffle

@pytest.mark.parametrize("source, expected", [
    (None, dict()),
    ([], dict()),
    (tuple(), dict()),
    ([0, 1], dict(values=[0, 1])),
    ([1, 0], dict(values=[1, 0])),
    ([1, 0, 1, 2], dict(values=[1, 0, 2])),
    (['0', 1], dict(values=['0', 1])),
    ([1, '0'], dict(values=[1, '0'])),
    ([1, 2, 1.0, 2.3], dict(values=[1, 2, 2.3])),
    ((1, 2, 1.0, 2.3), dict(values=[1, 2, 2.3])),
    ("abc", dict(values=['a', 'b', 'c'])),
    ("cab", dict(values=['c', 'a', 'b'])),
    ("abracadabra", dict(values=['a', 'b', 'r', 'c', 'd'])),
    (['foo', 'bar', 'Foo'], dict(values=['foo', 'bar', 'Foo'])),
    (['foo', 'bar', 'Foo', 'bar'], dict(values=['foo', 'bar', 'Foo'])),
    ([True, False], dict(values=[True, False])),
    ([False, True], dict(values=[False, True])),
    ([True, False, False], dict(values=[True, False])),
    ({2}, dict(values=[2])),
    (dict(a=3.14, cool=True), dict(values=['a', 'cool'])),
    (2, dict(error=TypeError, match="object is not iterable")),
])
def test_OrderedSet__init__(source, expected):
    """Test constructor, method `len` and `__repr__"""
    error = expected.get("error", None)
    if error is None:
        s = OrderedSet(source)
        values = expected.get("values", [])
        assert list(s) == values
        assert len(s) == len(values)
        elems = []
        for e in (source or []):
            if e not in elems:
                elems.append(e)
        assert repr(s) == "OrderedSet({})".format(elems if elems else "")
    else:
        pattern = expected.get("match", None)
        with pytest.raises(error, match=pattern):
            OrderedSet(source)
    

@pytest.mark.parametrize("source, elem, expected", [
    (None, 'a', False),
    ('abracadabra', 'a', True),
    ('abracadabra', 'b', True),
    ('abracadabra', 'z', False),
    ('abracadabra', 'ab', False),
    ([0, 1], 'a', False),
    ([0, 1], '1', False),
    ([0, 1], 1, True),
    ([0, 1], False, True),  # as in built-in set
])
def test_OrderedSet__contains__(source, elem, expected):
    assert (elem in OrderedSet(source)) == expected


@pytest.mark.parametrize("source, elem, expected", [
    (None, None, dict(values=[None])),
    (None, 'a', dict(values=['a'])),
    (None, 'abc', dict(values=['abc'])),
    ('abc', 'a', dict(values=['a', 'b', 'c'])),
    ('abc', 'c', dict(values=['a', 'b', 'c'])),
    ('abc', 'abc', dict(values=['a', 'b', 'c', 'abc'])),
    ([3, 2, 'foo'], 'bar', dict(values=[3, 2, 'foo', 'bar'])),
    ([3, 2, 'foo'], True, dict(values=[3, 2, 'foo', True])),
    ([0, 1], 1, dict(values=[0, 1])),
    ([0, 1], True, dict(values=[0, 1])),
    ([0, 1], False, dict(values=[0, 1])),
    ([False, True], 1, dict(values=[False, True])),
    ([1, 2, 'foo'], {'pi': 3.14}, dict(error=TypeError, match="unhashable")),
    (None, [1, 2, 3], dict(error=TypeError, match="unhashable")),
])
def test_OrderedSet_add(source, elem, expected):
    s = OrderedSet(source)
    error = expected.get("error", None)
    if error is None:
        s.add(elem)
        assert elem in s
        assert list(s) == expected["values"]
    else:
        pattern = expected.get("match", None)
        with pytest.raises(error, match=pattern):
            s.add(elem)


@pytest.mark.parametrize("source, index, expected", [
    (None, 0, dict(value=0)),
    ('abcd', 0, dict(value=0)),
    ('abcd', 1, dict(value=1)),
    ('abcd', 3, dict(value=3)),
    ('abcd', -1, dict(value=3)),
    ('abcd', -99, dict(value=0)),
    ('abcd', 100, dict(value=4)),
    ('abcde', -1, dict(value=4)),
    ('abcde', -2, dict(value=3)),
    ('abcde', -5, dict(value=0)),
    ('abcde', 1.0, dict(error=TypeError)),
    ('abcde', '0', dict(error=TypeError)),
    ('abcde', None, dict(error=TypeError)),
])
def test_OrderedSet__insertion_index(source, index, expected):
    s = OrderedSet(source)
    error = expected.get("error", None)
    if error is None:
        assert s._insertion_index(index) == expected["value"]
    else:
        pattern = expected.get("match", None)
        with pytest.raises(error, match=pattern):
            s._insertion_index(index)


@pytest.mark.parametrize("source, index, elem, expected", [
    (None, 0, None, dict(values=[None])),
    (None, 0, 'a', dict(values=['a'])),
    (None, -9, 'a', dict(values=['a'])),
    (None, 99, 'a', dict(values=['a'])),
    ('abc', 0, 'x', dict(values=['x', 'a', 'b', 'c'])),
    ('abc', 2, 'x', dict(values=['a', 'b', 'x', 'c'])),
    ('abc', -1, 'x', dict(values=['a', 'b', 'x', 'c'])),
    ('abc', -2, 'x', dict(values=['a', 'x', 'b', 'c'])),
    ([1, 2, 'foo'], 0, 'bar', dict(values=['bar', 1, 2, 'foo'])),
    ([1, 2, 'foo'], 0, {'pi': 3.14}, dict(error=TypeError, match="unhashable")),
    ('abcd', 0, [1, 2, 3], dict(error=TypeError, match="unhashable")),
    ('abcd', None, 0, dict(error=TypeError, match="index must be int")),
])
def test_OrderedSet_insert(source, index, elem, expected):
    s = OrderedSet(source)
    error = expected.get("error", None)
    if error is None:
        s.insert(index, elem)
        assert list(s) == expected["values"]
    else:
        pattern = expected.get("match", None)
        with pytest.raises(error, match=pattern):
            s.insert(index, elem)


@pytest.mark.parametrize("source, elem, expected", [
    (None, None, dict(values=[])),
    ('abc', None, dict(values=['a', 'b', 'c'])),
    ('abc', 'a', dict(values=['b', 'c'])),
    ('bac', 'a', dict(values=['b', 'c'])),
    ('cab', 'a', dict(values=['c', 'b'])),
    ((0, 1, 2), True, dict(values=[0, 2])),
])
def test_OrderedSet_discard(source, elem, expected):
    s = OrderedSet(source)
    error = expected.get("error", None)
    if error is None:
        s.discard(elem)
        assert list(s) == expected["values"]
    else:
        pattern = expected.get("match", None)
        with pytest.raises(error, match=pattern):
            s.discard(elem)


@pytest.mark.parametrize("source, last, expected", [
    (None, True, dict(error=KeyError, match="empty")),
    (None, False, dict(error=KeyError, match="empty")),
    ('abc', True, dict(values=['a', 'b'], popped='c')),
    ('abc', False, dict(values=['b', 'c'], popped='a')),
])
def test_OrderedSet_pop(source, last, expected):
    s = OrderedSet(source)
    error = expected.get("error", None)
    if error is None:
        assert s.pop(last) == expected["popped"]
        assert list(s) == expected["values"]
    else:
        pattern = expected.get("match", None)
        with pytest.raises(error, match=pattern):
            s.pop(last)


def test_OrderedSet_unfold():
    s = OrderedSet("abc")
    assert (*s, "a") == ("a", "b", "c", "a")


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("source, expected", [
    (None, []),
    ("abracadabra", ['a', 'b', 'r', 'c', 'd']),
    ((1, 2, 1, 0, 3, 2, 1), [1, 2, 0, 3]),
])
def test_OrderedSet__iter__(source, reverse, expected):
    s = OrderedSet(source)
    if reverse:
        values = [value for value in reversed(s)]
        expected = list(reversed(expected))
    else:
        values = [value for value in s]
    assert values == expected


@pytest.mark.parametrize("source, other, expected", [
    (None, OrderedSet(), True),
    ("abracadabra", OrderedSet(['a', 'b', 'r', 'c', 'd']), True),
    ((1, 2, 1, 0, 3, 2, 1), OrderedSet([1, 2, 0, 3]), True),
    ((1, 2, 1, 0, 3, 2, 1), OrderedSet([1, 2]), False),
    ("Ready You Are Not", OrderedSet("You Are Not Ready"), False),  # the Yoda test
    ("ABC", OrderedSet("BCA"), False),
    ("ABC", set("ABC"), False),
    ("ABC", list("ABC"), False),
    ("ABC", tuple("ABC"), False),
    ((0, 1, 2), (0, 1, 2), False),
    ((0, 1, 2), {0, 1, 2}, False),
])
def test_OrderedSet__eq__(source, other, expected):
    s = OrderedSet(source)
    assert s == s
    assert (s == other) == expected


@pytest.mark.parametrize("source", [
    (None),
    ("abracadabra"),
    ([1, 2, 0, 3]),
    ("ABC"),
])
def test_OrderedSet_set(source):
    """Check conversion from OrderedSet to built-in set"""
    s = OrderedSet(source)
    s_set = set(s)
    for _ in range(2 * len(s)):
        trial = list(s)
        shuffle(trial)
        assert s_set == set(trial)


@pytest.mark.parametrize("src1, src2, expected", [
    (None, "abc", ['a', 'b', 'c']),
    ("XY", "abc", ['X', 'Y', 'a', 'b', 'c']),
])
def test_OrderedSet_binaryOrEq(src1, src2, expected):
    s1, s2 = OrderedSet(src1), OrderedSet(src2)
    s1 |= s2
    assert list(s1) == expected


@pytest.mark.parametrize("source, expected", [
    ('a', dict(first='a', last='a')),
    ('abcd', dict(first='a', last='d')),
    ([], dict(error=AttributeError)),
    (None, dict(error=AttributeError)),
])
def test_OrderedSet_first_last(source, expected):
    s = OrderedSet(source)
    error = expected.get("error", None)
    if error is None:
        assert s.first == expected["first"]
        assert s.last == expected["last"]
    else:
        pattern = expected.get("match", None)
        with pytest.raises(error, match=pattern):
            s.first
            s.last


@pytest.mark.parametrize("source, elem, expected", [
    ('a', 'a', dict(index=0)),
    ('abcd', 'a', dict(index=0)),
    ('abcd', 'b', dict(index=1)),
    ('abcd', 'c', dict(index=2)),
    ('abcd', 'd', dict(index=3)),
    ('abcd', 'x', dict(error=ValueError)),
    (None, 'foo', dict(error=ValueError)),
    ([], 'foo', dict(error=ValueError)),
])
def test_OrderedSet_index(source, elem, expected):
    s = OrderedSet(source)
    error = expected.get("error", None)
    if error is None:
        assert s.index(elem) == expected["index"]
    else:
        pattern = expected.get("match", None)
        with pytest.raises(error, match=pattern):
            s.index(elem)


@pytest.mark.skip(reason="JSON encoder/decoder not implemented")
def test_OrderedSet_json():
    s = OrderedSet("abracadabra")
    jd = json.dumps(s)
    assert jd == '["a", "b", "r", "c", "d"]'
