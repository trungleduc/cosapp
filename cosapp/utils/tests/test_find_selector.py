import pytest

from cosapp.utils.parsing import find_selector


@pytest.mark.parametrize("expression, expected", [
    ("[]", dict(basename="", selector="[]")),
    ("x", dict()),
    ("_", dict()),
    (" ", dict(basename="")),
    ("a[]", dict(basename="a", selector="[]")),
    ("a[:]", dict(basename="a", selector="[:]")),
    ("a[-]", dict(basename="a", selector="[-]")),
    ("a[0:]", dict(basename="a", selector="[0:]")),
    ("a[::2]", dict(basename="a", selector="[::2]")),
    ("a[[0, 2]]", dict(basename="a", selector="[[0, 2]]")),
    ("a[1, [0, 2]]", dict(basename="a", selector="[1, [0, 2]]")),
    ("a[[[s]]]", dict(basename="a", selector="[[[s]]]")),
    ("a[[[1, 2], 0:, [s]]]", dict(basename="a", selector="[[[1, 2], 0:, [s]]]")),
    ("a[0][1, 2]", dict(basename="a", selector="[0][1, 2]")),
    ("foo[bar].a[0][1, 2]", dict(basename="foo[bar].a", selector="[0][1, 2]")),
    ("foo.bar[[[1, 2], 0:, [s]]]", dict(basename="foo.bar", selector="[[[1, 2], 0:, [s]]]")),
    ("hat['foo'].bar.[s].one.two[[0:2]]", dict(basename="hat['foo'].bar.[s].one.two", selector="[[0:2]]")),
    ("hat['foo'].bar.[s].one.two", dict(basename="hat['foo'].bar.[s].one.two", selector="")),
    ("foo[[].bar[[s]]", dict(basename="foo[[].bar", selector="[[s]]")),
    # Erroneous cases
    ("foo[[]", dict(error=ValueError, match="Bracket mismatch")),
    ("foo[][[]][]]", dict(error=ValueError, match="Bracket mismatch")),
    ("foo[", dict(error=ValueError, match="Bracket mismatch")),
    ("foo]", dict(error=ValueError, match="Bracket mismatch")),
])
def test_find_selector(expression, expected):
    error = expected.get('error', None)

    if error is None:
        basename, selector = find_selector(expression)
        assert basename == expected.get('basename', expression)
        assert selector == expected.get('selector', '')

    else:
        with pytest.raises(error, match=expected.get('match', None)):
            find_selector(expression)
