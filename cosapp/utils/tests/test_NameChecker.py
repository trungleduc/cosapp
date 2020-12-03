import pytest

from cosapp.utils.naming import NameChecker


@pytest.fixture(scope="function")
def default():
    return NameChecker()


@pytest.mark.parametrize("name, expected", [
    ("a", dict(error=None)),
    ("A", dict()),
    ("foobar", dict()),
    ("foo4bar", dict()),
    ("CamelBack", dict()),
    ("foo_bar", dict()),
    ("foobar_", dict()),
    ("teaFor2", dict()),
    ("2forTea", dict(error=ValueError)),
    ("foo@bar", dict(error=ValueError)),
    ("_foobar", dict(error=ValueError)),
    ("foo bar", dict(error=ValueError)),
    ("foobar?", dict(error=ValueError)),
    ("foo.bar", dict(error=ValueError)),
    ("foo:bar", dict(error=ValueError)),
    ("foo/bar", dict(error=ValueError)),
    ("1foobar", dict(error=ValueError)),
    ("foobar-2", dict(error=ValueError)),
    ("foobar:2", dict(error=ValueError)),
    ("foobar.2", dict(error=ValueError)),
    ("", dict(error=ValueError)),
    ("t_", dict(error=None)),
    ("t", dict(error=ValueError, match="reserved")),
    ("time", dict(error=ValueError, match="reserved")),
    (23, dict(error=TypeError)),
    (1.0, dict(error=TypeError)),
    (dict(a=True), dict(error=TypeError)),
    (list(), dict(error=TypeError)),
])
def test_NameChecker_default(default, name, expected):
    error = expected.get("error", None)
    if error is None:
        assert default.is_valid(name)
        assert default(name) == name
    else:
        assert not default.is_valid(name)
        pattern = expected.get("match", None)
        with pytest.raises(error, match=pattern):
            default(name)


def test_NameChecker_default_message(default):
    assert default.message == "Name must start with a letter, and contain only alphanumerics and '_'"


@pytest.mark.parametrize("message, expected", [
    ("Customized message", dict()),
    ("", dict()),
    (0.1234, dict(error=TypeError)),
    ([1, 2], dict(error=TypeError)),
])
def test_NameChecker_message(message, expected):
    checker = NameChecker()
    error = expected.get("error", None)
    if error is None:
        checker.message = message
        assert checker.message == expected.get("message", message)
    else:
        with pytest.raises(error):
            checker.message = message


def test_NameChecker_reserved():
    assert set(NameChecker.reserved()) == {"t", "time"}


def test_NameChecker_empty():
    checker = NameChecker(pattern="", message="")
    assert checker.is_valid("@foo(bar)")
    with pytest.raises(ValueError, match="reserved"):
        checker("t") == "t"


def test_NameChecker_excluded():
    checker = NameChecker(excluded=[])

    assert checker.excluded == tuple()
    assert checker.is_valid("foobar")
    assert checker.is_valid("inwards")
    assert checker.is_valid("outwards")

    checker.excluded = ["inwards", "outwards"]
    assert checker.excluded == ("inwards", "outwards")
    assert checker.is_valid("foobar")
    assert checker.is_valid("inwards_")
    assert not checker.is_valid("inwards")
    assert not checker.is_valid("outwards")

    with pytest.raises(ValueError):
        checker("inwards")
    
    with pytest.raises(ValueError):
        checker("outwards")

    checker.excluded = "foobar"
    assert checker.excluded == ("foobar", )
    assert not checker.is_valid("foobar")
    assert checker.is_valid("inwards")
    assert checker.is_valid("outwards")
    
    with pytest.raises(ValueError):
        checker("foobar")
    
    checker.excluded = None
    assert checker.excluded == tuple()

    with pytest.raises(TypeError):
        checker.excluded = 1

    with pytest.raises(ValueError):
        checker.excluded = ['foo', 3.14]
