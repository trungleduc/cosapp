import numpy as np
import pytest

from cosapp.systems import System
from cosapp.tests.library.systems import Multiply2, Strait1dLine
from cosapp.utils.parsing import get_indices


def test__get_indices():
    s = Multiply2("mult")
    t = System("dummy")
    t.add_child(s)

    with pytest.raises(
        TypeError,
        match=r"Only non-empty numpy array can be partially selected; got '\w+\.[\w\[\.\]]+'\.",
    ):
        get_indices(t, "mult.K1[1]")

    with pytest.raises(
        TypeError,
        match=r"Only non-empty numpy array can be partially selected; got '\w+\.[\w\[\.\]]+'\.",
    ):
        get_indices(t, "mult[0]")

    r = get_indices(t, "mult.K1")
    assert r == ("mult.K1", None)
    r = get_indices(s, "K1")
    assert r == ("K1", None)
    r = get_indices(t, "mult.p_in.x")
    assert r == ("mult.p_in.x", None)

    s = Multiply2("mult")
    r = get_indices(s, "K1")
    assert r == ("K1", None)

    # Test vector variables
    s = System("hat")
    one = s.add_child(Strait1dLine("one"), pulling="in_")
    two = s.add_child(Strait1dLine("two"), pulling="out")
    s.connect(two.in_, one.out)

    r = get_indices(s, "one.a")
    assert r[0] == "one.a"
    assert isinstance(r[1], np.ndarray)
    assert np.array_equal(r[1], [True, True, True])

    r = get_indices(s, "one.a[0]")
    assert r[0] == "one.a"
    assert isinstance(r[1], np.ndarray)
    assert np.array_equal(r[1], [True, False, False])

    r = get_indices(s, "one.a[1:]")
    assert r[0] == "one.a"
    assert isinstance(r[1], np.ndarray)
    assert np.array_equal(r[1], [False, True, True])

    r = get_indices(s, "one.a[[0, 2]]")
    assert r[0] == "one.a"
    assert isinstance(r[1], np.ndarray)
    assert np.array_equal(r[1], [True, False, True])

    r = get_indices(s, "one.a[[True, False, True]]")
    assert r[0] == "one.a"
    assert isinstance(r[1], np.ndarray)
    assert np.array_equal(r[1], [True, False, True])

    with pytest.raises(
        IndexError,
        match=r"Selection '[\w\[\.\s,\]]+' for variable '\w+.[\w\.]+' is not valid:.*",
    ):
        get_indices(s, "one.a[[2, 4]]")

    with pytest.raises(
        SyntaxError,
        match=r"Selection '[\w\[\.\s,\]]+' for variable '\w+.[\w\.]+' is not valid:.*",
    ):
        get_indices(s, "one.a[[2, 4]")
