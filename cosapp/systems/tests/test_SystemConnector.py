import pytest
from contextlib import nullcontext as does_not_raise

from cosapp.patterns import Proxy
from cosapp.core.connectors import BaseConnector, Connector
from cosapp.systems.systemConnector import SystemConnector
from cosapp.ports.port import PortType, Port
from cosapp.systems import System


fake = System('fake')

class AbcdPort(Port):
    """Compatible with AbcPort, with extra variable 'd'."""
    def setup(self):
        self.add_variable('a', 1.0, unit='inch')
        self.add_variable('b', 1.0, unit='degF')
        self.add_variable('c', 1.0)
        self.add_variable('d', 1.0)
        self.owner = fake

class XyPort(Port):
    def setup(self):
        self.add_variable('x', 1.0, unit='m')
        self.add_variable('y', 1.0, unit='degC')
        self.owner = fake

class XyImperialPort(Port):
    def setup(self):
        self.add_variable('x', 1.0, unit='ft')
        self.add_variable('y', 1.0, unit='degF')
        self.owner = fake

class UvPort(Port):
    def setup(self):
        self.add_variable('u', 2.0, unit='m')
        self.add_variable('v', 4.0)
        self.owner = fake


class PlainConnector(BaseConnector):
    """Simple assignment connector.
    """
    def transfer(self) -> None:
        source, sink = self.source, self.sink

        for target, origin in self.mapping.items():
            value = getattr(source, origin)
            setattr(sink, target, value)


@pytest.fixture
def proxy():
    p1 = AbcdPort('p1', PortType.IN)
    p2 = AbcdPort('p2', PortType.OUT)
    return SystemConnector(Connector('p2_to_p1', p1, p2))


def test_Connector__init__():
    p1 = AbcdPort('p1', PortType.IN)
    p2 = AbcdPort('p2', PortType.OUT)
    connector = Connector('p2_to_p1', p1, p2)
    proxy = SystemConnector(connector)
    assert isinstance(proxy, Proxy)
    assert isinstance(proxy, Connector)
    assert isinstance(proxy, SystemConnector)
    assert proxy.is_active
    assert proxy.__wrapped__ is connector
    assert proxy.source is connector.source
    assert proxy.sink is connector.sink
    assert proxy.mapping == connector.mapping

    with pytest.raises(TypeError):
        SystemConnector('not a connector')


@pytest.mark.parametrize("wrappee, expected", [
    (Connector, does_not_raise()),
    (PlainConnector, does_not_raise()),
    (
        PlainConnector(
            'p2_to_p1',
            AbcdPort('p1', PortType.IN),
            AbcdPort('p2', PortType.OUT),
        ),
        does_not_raise()
    ),
    (BaseConnector, pytest.raises(ValueError)),
    (System, pytest.raises(ValueError)),
    (System('foo'), pytest.raises(TypeError)),
])
def test_Connector_check(wrappee, expected):
    with expected:
        SystemConnector.check(wrappee)


def test_Connector_activate(proxy):
    proxy.activate()
    assert proxy.is_active

    proxy.deactivate()
    assert not proxy.is_active


def test_SystemConnector_transfer():
    """Test transfer with and without activation"""
    p1 = XyPort('p1', PortType.IN, {'x': 1, 'y': 2})
    p2 = UvPort('p2', PortType.OUT, {'u': 2, 'v': 4})
    c = SystemConnector(Connector('p2_to_p1', p1, p2, dict(zip(p1, p2))))
    assert p1.x == 1
    assert p1.y == 2
    c.deactivate()
    c.transfer()
    assert p1.x == 1
    assert p1.y == 2
    c.activate()
    c.transfer()
    assert p1.x == 2  # Neutral unit conversion
    assert p1.y == 4  # No unit conversion

    p1 = XyPort('p1', PortType.IN, dict(x=0, y=0))
    p2 = XyImperialPort('p2', PortType.OUT, dict(x=10, y=100))
    c = SystemConnector(Connector('p2_to_p1', p1, p2))
    assert p1.x == 0
    assert p1.y == 0
    c.deactivate()
    c.transfer()
    assert p1.x == 0
    assert p1.y == 0
    c.activate()
    c.transfer()
    assert p1.x == pytest.approx(10 * 0.3048, rel=1e-15)
    assert p1.y == pytest.approx(5 / 9 * (100 - 32), rel=1e-15)
