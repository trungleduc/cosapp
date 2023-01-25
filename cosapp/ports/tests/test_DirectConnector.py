import pytest
import numpy as np

from cosapp.base import System, Port
from cosapp.ports.port import PortType
from cosapp.ports.connectors import (
    BaseConnector,
    PlainConnector,
    CopyConnector,
    DeepCopyConnector,
)
from typing import Type


# <codecell>

fake = System('fake')


class AbcdPort(Port):
    def setup(self):
        self.add_variable('a', 1.0, unit='inch')
        self.add_variable('b', 1.0, unit='degF')
        self.add_variable('c', np.ones(3), unit='psi')
        self.add_variable('d', {})
        self.owner = fake


class XyzPort(Port):
    def setup(self):
        self.add_variable('x', 0.0, unit='m')
        self.add_variable('y', 0.0, unit='degC')
        self.add_variable('z', np.zeros(3), unit='Pa')
        self.add_variable('d', {})
        self.owner = fake


def ConnectorFactory(
    ctype: Type[BaseConnector],
    port1: Port,
    port2: Port,
    mapping=None,
    name="p2_to_p1",
) -> BaseConnector:
    """Connector factory from construction settings"""
    return ctype(name, port1, port2, mapping)

# <codecell>

@pytest.mark.parametrize("ctype, expected", [
    (
        PlainConnector,
        dict(
            name="PlainConnector",
            docstring=(
                "Plain assignment connector, with no unit conversion."
                " Warning: may generate common references between sink and source variables."
            ),
        ),
    ),
    (
        CopyConnector,
        dict(name="CopyConnector", docstring="Shallow copy connector, with no unit conversion."),
    ),
    (
        DeepCopyConnector,
        dict(name="DeepCopyConnector", docstring="Deep copy connector, with no unit conversion."),
    ),
])
def test_DirectConnector_traits(ctype: Type[BaseConnector], expected: dict):
    assert issubclass(ctype, BaseConnector)
    assert ctype.__name__ == expected['name']
    assert ctype.__doc__ == expected['docstring']


@pytest.mark.parametrize("ctype", [
    PlainConnector,
    CopyConnector,
    DeepCopyConnector,
])
def test_DirectConnector_transfer(ctype: Type[BaseConnector]):
    """Test `transfer` method for various connector types.
    """
    class Foo:
        pass

    source = AbcdPort('p1',
        direction=PortType.OUT,
        variables=dict(
            a = 1.2,
            b = -0.5,
            c = np.linspace(0, 1, 3),
            d = {'u': np.ones(5), 'v': Foo()},
        ),
    )
    sink = XyzPort('p2',
        direction=PortType.IN,
        variables=dict(x=0.1, y=0.2, z=np.zeros(3), d={}),
    )

    # Check sink port before transfer
    assert sink.x == 0.1
    assert sink.y == 0.2
    assert np.array_equal(sink.z, [0, 0, 0])
    assert sink.d == dict()

    connector = ConnectorFactory(ctype, sink, source, dict(zip('xyzd', 'abcd')))
    assert isinstance(connector, BaseConnector)
    assert connector.source is source
    assert connector.sink is sink
    assert connector.mapping == {'x': 'a', 'y': 'b', 'z': 'c', 'd': 'd'}

    connector.transfer()

    # Check sink port after transfer
    assert sink.x == 1.2   # note: No unit conversion
    assert sink.y == -0.5  # note: No unit conversion
    assert sink.z == pytest.approx([0, 0.5, 1], rel=1e-15)
    assert set(sink.d) == {'u', 'v'}
    assert np.array_equal(sink.d['u'], np.ones(5))
    assert isinstance(sink.d['v'], Foo)

    if isinstance(connector, PlainConnector):
        assert sink.z is source.c
        assert sink.d is source.d

    elif isinstance(connector, CopyConnector):
        assert sink.z is not source.c
        assert sink.d is not source.d
        assert sink.d['u'] is source.d['u']
        assert sink.d['v'] is source.d['v']

    elif isinstance(connector, DeepCopyConnector):
        assert sink.z is not source.c
        assert sink.d is not source.d
        assert sink.d['u'] is not source.d['u']
        assert sink.d['v'] is not source.d['v']
