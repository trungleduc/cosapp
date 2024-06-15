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
def test_ExtensiblePort_pop_variable(direction, caplog):
    port = ExtensiblePort("myPort", direction)
    port.add_variable("var1", 0.0)
    port.add_variable("var2", 100.1)

    assert len(port) == 2
    assert set(port) == {"var1", "var2"}
    assert hasattr(port, "var1")
    assert hasattr(port, "var2")
    assert "var1" in port
    assert "var1" in port.get_details()

    port.pop_variable("var1")
    assert len(port) == 1
    assert set(port) == {"var2"}
    assert hasattr(port, "var2")
    assert not hasattr(port, "var1")
    assert "var1" not in port
    assert "var1" not in port.get_details()

    with pytest.raises(AttributeError):
        port.pop_variable("var3")
