import pytest
import numpy as np

from cosapp.systems import System
from cosapp.tests.library.systems import Multiply2, Strait1dLine
from cosapp.utils.parsing import get_indices
from typing import Dict, Any


def test_get_indices_scalar():
    s = Multiply2("mult")
    t = System("dummy")
    t.add_child(s)

    with pytest.raises(
        TypeError,
        match=r"Only non-empty numpy arrays can be partially selected; got '\w+\.[\w\[\.\]]+'\.",
    ):
        get_indices(t, "mult.K1[1]")

    with pytest.raises(
        TypeError,
        match=r"Only non-empty numpy arrays can be partially selected; got '\w+\.[\w\[\.\]]+'\.",
    ):
        get_indices(t, "mult[0]")

    r = get_indices(t, "mult.K1")
    assert r == ("mult.K1", "", None)
    r = get_indices(s, "K1")
    assert r == ("K1", "", None)
    r = get_indices(t, "mult.p_in.x")
    assert r == ("mult.p_in.x", "", None)

    s = Multiply2("mult")
    r = get_indices(s, "K1")
    assert r == ("K1", "", None)


@pytest.mark.parametrize("name, expected", [
    ("hat.a", dict(mask=[True, True, True])),
    ("hat.one.a", dict(mask=[True, True, True])),
    ("hat.one.a[0]", dict(mask=[True, False, False], basename="hat.one.a", selector="[0]")),
    ("hat.one.a[1:]", dict(mask=[False, True, True], basename="hat.one.a", selector="[1:]")),
    ("hat.one.a[[0, 2]]", dict(mask=[True, False, True], basename="hat.one.a", selector="[[0, 2]]")),
    ("hat.one.a[[True, False, True]]", dict(mask=[True, False, True], basename="hat.one.a", selector="[[True, False, True]]")),
    ("hat['one'].a", dict(mask=[True, True, True], basename="hat['one'].a", selector="")),
    ("hat['one'].a[1:]", dict(mask=[False, True, True], basename="hat['one'].a", selector="[1:]")),
    ("hat.one.a[[2, 4]", dict(error=SyntaxError, match="Bracket mismatch")),
    ("hat.one.a[2, 4]]", dict(error=SyntaxError, match="Bracket mismatch")),
    ("hat.one.a[[2, 4]]", dict(error=IndexError, match=r"Invalid selector '[\w\[\.\s,\]]+' for variable '\w+.[\w\.]+':.*")),
    # TODO: ideally, cases below should work (basename is reformatted into `x.y.z`) - OK for now
    # ("hat['one'].a", dict(mask=[True, True, True], basename="hat.one.a", selector="")),
    # ("hat['one'].a[1:]", dict(mask=[False, True, True], basename="hat.one.a", selector="[1:]")),
    # ("hat['one.a'][1:]", dict(mask=[False, True, True], basename="hat.one.a", selector="[1:]")),
])
def test_get_indices_array_1D(name, expected: Dict[str, Any]):
    """Test `get_indices` with vector variables"""
    hat = System("hat")
    one = hat.add_child(Strait1dLine("one"), pulling=["in_", "a", "b"])
    two = hat.add_child(Strait1dLine("two"), pulling=["out", "a", "b"])
    hat.connect(two.in_, one.out)
    top = System("top")
    top.add_child(hat, pulling=['a'])

    error = expected.get('error', None)

    if error is None:
        info = get_indices(top, name)
        assert info.basename == expected.get('basename', name)
        assert info.selector == expected.get('selector', '')
        assert info.fullname == name
        assert isinstance(info.mask, np.ndarray)
        assert np.array_equal(info.mask, expected['mask'])

    else:
        with pytest.raises(error, match=expected.get('match', None)):
            get_indices(top, name)


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
def test_get_indices_array_2D(selector, expected):
    class Bogus(System):
        def setup(self):
            self.add_inward('a', np.reshape(np.arange(12, dtype=float), (3, 4)))

    top = System('top')
    sub = top.add_child(Bogus('sub'))
    r = get_indices(top, f"sub.a{selector}")
    assert r.basename == "sub.a"
    assert r.selector == selector
    assert r.fullname == f"sub.a{selector.strip()}"
    assert np.array_equal(r.mask, expected)
