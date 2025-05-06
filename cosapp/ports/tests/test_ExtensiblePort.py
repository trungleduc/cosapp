import pytest

from cosapp.ports.port import PortType, ExtensiblePort


@pytest.mark.parametrize("direction", PortType)
@pytest.mark.parametrize("name, error", [
    ("a", None),
    ("A", None),
    ("foobar", None),
    ("foo4bar", None),
    ("loveYou2", None),
    ("CamelBack", None),
    ("foo_bar", None),
    ("foobar_", None),
    ("_foobar", ValueError),
    ("foo bar", ValueError),
    ("foobar?", ValueError),
    ("foo.bar", ValueError),
    ("foo:bar", ValueError),
    ("foo/bar", ValueError),
    ("1foobar", ValueError),
    ("foobar-2", ValueError),
    ("foobar:2", ValueError),
    ("foobar.2", ValueError),
    (23, TypeError),
    (1.0, TypeError),
    (dict(a=True), TypeError),
    (list(), TypeError),
])
def test_ExtensiblePort__init__(name, direction, error):
    if error is None:
        port = ExtensiblePort(name, direction)
        assert port.name == name
        assert port.contextual_name == name
        assert port.direction is direction
        assert port.owner is None
    else:
        with pytest.raises(error):
            ExtensiblePort(name, direction)


@pytest.mark.parametrize("direction", PortType)
def test_ExtensiblePort_pop_variable(direction):
    port = ExtensiblePort("myPort", direction)
    port.add_variable("x", 0.0)
    port.add_variable("y", 100.1)

    vardict = port.variable_dict()

    assert len(port) == 2
    assert set(port) == {"x", "y"}
    assert set(vardict) == {"x", "y"}
    assert hasattr(port, "x")
    assert hasattr(port, "y")
    assert "x" in port
    assert "y" in port

    port.pop_variable("x")
    assert len(port) == 1
    assert set(port) == {"y"}
    assert set(vardict) == {"y"}
    assert hasattr(port, "y")
    assert not hasattr(port, "x")
    assert "x" not in port
    assert "y" in port

    with pytest.raises(AttributeError):
        port.pop_variable("z")
