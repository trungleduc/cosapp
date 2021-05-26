import pytest
from unittest.mock import MagicMock
import logging

from cosapp.core import connectors
from cosapp.core.connectors import Connector, ConnectorError
from cosapp.ports.port import PortType, Port
from cosapp.systems import System
from cosapp.ports.units import UnitError
from cosapp.utils.testing import assert_keys

# <codecell>

fake = System('fake')

class PortWithNoOwner(Port):
    def setup(self):
        self.add_variable('a')
        self.add_variable('x', unit='m')

class APort(Port):
    def setup(self):
        self.add_variable('a')
        self.owner = fake

class AbcPort(Port):
    def setup(self):
        self.add_variable('a', 1.0, unit='mm')
        self.add_variable('b', 1.0, unit='K')
        self.add_variable('c', 1.0)
        self.owner = fake

class AbcdPort(Port):
    """Compatible with AbcPort, with extra variable 'd'."""
    def setup(self):
        self.add_variable('a', 1.0, unit='inch')
        self.add_variable('b', 1.0, unit='degF')
        self.add_variable('c', 1.0)
        self.add_variable('d', 1.0)
        self.owner = fake

class XPort(Port):
    def setup(self):
        self.add_variable('x', unit='m')
        self.owner = fake

class XMassPort(Port):
    def setup(self):
        self.add_variable('x', unit='kg')
        self.owner = fake

class YPort(Port):
    def setup(self):
        self.add_variable('y')
        self.owner = fake

class XYPort(Port):
    def setup(self):
        self.add_variable('x', unit='m')
        self.add_variable('y', unit='degC')
        self.owner = fake

class XYImperialPort(Port):
    def setup(self):
        self.add_variable('x', unit='ft')
        self.add_variable('y', unit='degF')
        self.owner = fake

class XYZPort(Port):
    def setup(self):
        self.add_variable('x')
        self.add_variable('y')
        self.add_variable('z', 'test')
        self.owner = fake

class P(Port):
    def setup(self):
        self.add_variable('v', 2, unit='m')
        self.add_variable('w', 4)
        self.owner = fake

class Q(Port):
    def setup(self):
        self.add_variable('x')
        self.add_variable('y')
        self.add_variable('z')
        self.owner = fake

class R(Port):
    def setup(self):
        self.add_variable('v')
        self.add_variable('w')
        self.add_variable('z')
        self.owner = fake

class T(Port):
    def setup(self):
        self.add_variable('v')
        self.add_variable('y')
        self.owner = fake

# <codecell>

@pytest.fixture(scope="function")
def ConnectorFactory():
    """Connector factory from construction settings"""
    def factory(port1, port2, name="p2_to_p1", mapping=lambda p1, p2: dict(zip(p1, p2))) -> Connector:
        return Connector(name, port1, port2, mapping(port1, port2))
    return factory


@pytest.mark.parametrize("settings, expected", [
    (
        dict(
            port1 = XPort('p1', PortType.IN),
            port2 = XPort('p2', PortType.OUT),
            mapping = lambda p1, p2: 'x',
        ),
        dict(mapping={'x': 'x'}, conversions={'x': (1, 0)})
    ),
    (
        dict(
            port1 = XPort('p1', PortType.IN),
            port2 = YPort('p2', PortType.OUT),
            mapping = lambda p1, p2: {'x': 'y'},
        ),
        dict(mapping={'x': 'y'}, conversions={'x': (1, 0)})
    ),
    (
        dict(
            port1 = XYPort('p1', PortType.IN),
            port2 = XYPort('p2', PortType.OUT),
            mapping = lambda p1, p2: list(p2),
        ),
        dict(mapping={'x': 'x', 'y': 'y'}, conversions={'x': (1, 0), 'y': (1, 0)})
    ),
    (
        dict(
            port1 = XYPort('p1', PortType.IN),
            port2 = P('p2', PortType.OUT),
            mapping = lambda p1, p2: dict(zip(p1, p2)),
        ),
        dict(mapping={'x': 'v', 'y': 'w'}, conversions={'x': (1, 0), 'y': (1, 0)})
    ),
    (
        dict(
            port1 = Q('p1', PortType.IN),
            port2 = XYPort('p2', PortType.OUT, {'x': 2, 'y': 4}),
            mapping = lambda p1, p2: list(p2),
        ),
        dict(mapping={'x': 'x', 'y': 'y'}, conversions={'x': (1, 0), 'y': (1, 0)})
    ),
    (
        dict(
            port1 = XYPort('p1', PortType.IN, {'x': 1, 'y': 2}),
            port2 = R('p2', PortType.OUT, {'v': 2, 'w': 4, 'z': 3}),
            mapping = lambda p1, p2: dict(zip(p1, ['w', 'z'])),
        ),
        dict(mapping={'x': 'w', 'y': 'z'}, conversions={'x': (1, 0), 'y': (1, 0)})
    ),
    (
        dict(
            port1 = Q('p1', PortType.IN, {'x': 1, 'y': 2, 'z': 3}),
            port2 = R('p2', PortType.OUT, {'v': 2, 'w': 4, 'z': 3}),
            mapping = lambda p1, p2: 'z',
        ),
        dict(mapping={'z': 'z'})
    ),
    (
        dict(
            port1 = Q('p1', PortType.IN, {'x': 1, 'y': 2, 'z': 3}),
            port2 = R('p2', PortType.OUT, {'v': 2, 'w': 4, 'z': 3}),
            mapping = lambda p1, p2: {'x': 'w'},
        ),
        dict(mapping={'x': 'w'})
    ),
    (
        dict(
            port1 = T('p1', PortType.IN, {'v': 1, 'y': 2}),
            port2 = P('p2', PortType.OUT, {'v': 2, 'w': 4}),
            mapping = lambda p1, p2: ['v', 'foo'],
        ),
        dict(error=ConnectorError, match="variable 'foo' does not exist in port P")
    ),
    (
        dict(
            port1 = XYPort('p1', PortType.IN, {'x': 1, 'y': 2}),
            port2 = P('p2', PortType.OUT, {'v': 2, 'w': 4}),
            mapping = lambda p1, p2: {'x': 'v', 'foo': 'w'},
        ),
        dict(error=ConnectorError, match="variable 'foo' does not exist in port XYPort")
    ),
    (
        dict(
            port1 = AbcPort('p1', PortType.IN),
            port2 = AbcdPort('p2', PortType.OUT),
            mapping = lambda p1, p2: None,
        ),
        dict(mapping={k: k for k in list('abc')}, conversions={'a': (25.4, 0), 'b': (5/9, 459.67), 'c': (1, 0)})
    ),
    (
        dict(
            port1 = XYPort('p1', PortType.IN),
            port2 = XYImperialPort('p2', PortType.OUT),
            mapping = lambda p1, p2: 'y',
        ),
        dict(mapping={'y': 'y'}, conversions={'y': (5/9, -32)})
    ),
    (
        dict(
            port1 = XYPort('p1', PortType.IN),
            port2 = XYImperialPort('p2', PortType.OUT),
            mapping = lambda p1, p2: ['x', 'y'],
        ),
        dict(mapping={'x': 'x', 'y': 'y'}, conversions={'x': (0.3048, 0), 'y': (5/9, -32)})
    ),
    (
        dict(
            port1 = XPort('p1', PortType.IN),
            port2 = XMassPort('p2', PortType.OUT),
            mapping = lambda p1, p2: 'x',
        ),
        dict(error=UnitError, match="Unit m is not compatible with kg")
    ),
    (
        dict(
            port1 = XPort('p1', PortType.IN),
            port2 = PortWithNoOwner('p2', PortType.OUT),
            mapping = lambda p1, p2: {'x': 'a'},
        ),
        dict(error=ConnectorError, match="Source owner is undefined")
    ),
    (
        dict(
            port1 = PortWithNoOwner('p1', PortType.OUT),
            port2 = XPort('p2', PortType.IN),
            mapping = lambda p1, p2: {'a': 'x'},
        ),
        dict(error=ConnectorError, match="Sink owner is undefined")
    ),
    (
        dict(port1={'p1': PortType.OUT}, port2=XPort('p2', PortType.IN)),
        dict(error=TypeError, match="sink")
    ),
    (
        dict(port2={'p2': PortType.IN}, port1=XPort('p1', PortType.OUT)),
        dict(error=TypeError, match="source")
    ),
])
def test_Connector__init__(ConnectorFactory, settings, expected):
    error = expected.get('error', None)

    if error is None:
        c = ConnectorFactory(**settings)
        assert c.name == settings.get('name', 'p2_to_p1')
        assert c.sink is settings['port1']
        assert c.source is settings['port2']
        assert c.mapping == expected['mapping']
        # conversion factors:
        conversions = expected.get('conversions',
            { var: (1, 0) for var in expected['mapping']})
        assert_keys(c._unit_conversions, *conversions.keys())
        for var, values in conversions.items():
            assert c._unit_conversions[var] == pytest.approx(values, rel=1e-15), f"var = {var!r}"

    else:
        pattern = expected.get('match', None)
        with pytest.raises(error, match=pattern):
            c = ConnectorFactory(**settings)


def test_Connector_empty():
    port1 = XYZPort('p1', PortType.IN)
    port2 = AbcPort('p2', PortType.OUT)
    c = Connector('p2_p1', port1, port2)
    assert c.mapping == dict()


@pytest.mark.parametrize("ptype", PortType)
def test_Connector__init__same_port(ptype):
    port = XYPort('port', ptype, {'x': 1, 'y': 2})
    with pytest.raises(ConnectorError, match="Source and sink cannot be the same object"):
        c = Connector('oops', port, port, ['x', 'y'])


def test_Connector_name():
    p1 = XPort('p1', PortType.IN, {'x': 1})
    p2 = XPort('p2', PortType.OUT, {'x': 1})
    connector_name = 'p2_to_p1'
    c = Connector(connector_name, p1, p2, 'x')
    assert c.name == connector_name
    with pytest.raises(AttributeError):
        c.name = 'new_name'


@pytest.mark.parametrize("attr", ["source", "sink"])
@pytest.mark.parametrize("port, expected", [
    (XPort("p3", PortType.IN, {'x': 1}), dict(error=None)),
    (XPort("p3", PortType.OUT, {'x': 1}), dict(error=None)),
    (APort("p3", PortType.IN, {'a': 2}), dict(error=ConnectorError, match="variable 'x' does not exist in port APort")),
    (APort("p3", PortType.OUT, {'a': 2}), dict(error=ConnectorError, match="variable 'x' does not exist in port APort")),
    (XYPort("p3", PortType.IN, {'x': 1, 'y': 2}), dict(error=None)),
    (XYPort("p3", PortType.OUT, {'x': 1, 'y': 2}), dict(error=None)),
    ("p3", dict(error=TypeError, match="should be BasePort")),
    ({'x': 1}, dict(error=TypeError, match="should be BasePort")),
])
def test_Connector_source_sink(attr, port, expected):
    """Test 'source' and 'sink' attributes"""
    p1 = XPort('p1', PortType.IN, {'x': -2})
    p2 = XPort('p2', PortType.OUT, {'x': 1})
    connector_name = 'p2_to_p1'
    c = Connector(connector_name, p1, p2, 'x')
    assert c.source is p2
    assert c.sink is p1

    error = expected.get("error", None)

    if error is None:
        setattr(c, attr, port)
        assert getattr(c, attr) is port
    else:
        pattern = expected.get("match", None)
        with pytest.raises(error, match=pattern):
            setattr(c, attr, port)


@pytest.mark.parametrize("attr, port, expected", [
    ('source', AbcPort('p3', PortType.IN), dict(error=None)),
    ('source', AbcdPort('p3', PortType.IN), dict(error=None)),  # source has extra variables `c` and `d` - OK
    ('source', PortWithNoOwner('p3', PortType.IN), dict(error=ConnectorError, match="owner is undefined")),
    ('source', XPort('p3', PortType.IN), dict(error=ConnectorError, match="variable 'a' does not exist in port XPort")),
    ('source', APort('p3', PortType.IN), dict(error=ConnectorError, match="variable 'b' does not exist in port APort")),
    ('sink', XYPort('p3', PortType.IN), dict(error=None)),
    ('sink', XYZPort('p3', PortType.IN), dict(error=None)),  # sink has extra variable `z` - OK
    ('sink', PortWithNoOwner('p3', PortType.IN), dict(error=ConnectorError, match="owner is undefined")),
    ('sink', XPort('p3', PortType.IN), dict(error=ConnectorError, match="variable 'y' does not exist in port XPort")),
    ('sink', APort('p3', PortType.IN), dict(error=ConnectorError, match="variable 'x' does not exist in port APort")),
])
def test_Connector_source_sink_mapping(attr, port, expected):
    """
    Test 'source' and 'sink' setters in partial connector AbcdPort -> XYPort,
    with mapping {'x': 'a', 'y': 'b'}
    """
    p1 = XYPort('p1', PortType.IN, {'x': -2, 'y': 0})
    p2 = AbcPort('p2', PortType.OUT, {'a': 1, 'b': 100, 'c': 0})
    c = Connector('p2_to_p1', p1, p2, {'x': 'a', 'y': 'b'})
    assert c.source is p2
    assert c.sink is p1

    error = expected.get("error", None)

    if error is None:
        setattr(c, attr, port)
        assert getattr(c, attr) is port

    else:
        pattern = expected.get("match", None)
        with pytest.raises(error, match=pattern):
            setattr(c, attr, port)


def test_Connector_source_sink_variables():
    p1 = XYPort('p1', PortType.IN, {'x': -2, 'y': 0})
    p2 = AbcPort('p2', PortType.OUT, {'a': 1, 'b': 100, 'c': 0})
    c = Connector('p2_to_p1', p1, p2, {'x': 'a', 'y': 'b'})
    assert c.source is p2
    assert c.sink is p1

    assert len(c) == 2
    assert len(c.sink_variables()) == len(c)
    assert len(c.source_variables()) == len(c)

    assert 'x' in c.sink_variables()
    assert 'y' in c.sink_variables()
    assert c.source_variable('x') == 'a'
    assert c.source_variable('y') == 'b'
    with pytest.raises(KeyError):
        c.source_variable('a')

    assert 'a' in c.source_variables()
    assert 'b' in c.source_variables()
    assert c.sink_variable('a') == 'x'
    assert c.sink_variable('b') == 'y'
    with pytest.raises(KeyError):
        c.sink_variable('x')


@pytest.mark.parametrize("settings, expected", [
    (
        dict(Port1=XYZPort),
        True
    ),
    (   # Same Port types with implicit full mapping
        dict(Port1=XYZPort, mapping=['x', 'y', 'z']),
        True
    ),
    (   # Same Port types with explicit full mapping
        dict(Port1=XYZPort, mapping={'x': 'x', 'y': 'y', 'z': 'z'}),
        True
    ),
    (   # Same Port types with incomplete mapping
        dict(Port1=XYZPort, mapping='x'),
        False
    ),
    (   # Same Port types with incomplete mapping
        dict(Port1=XYZPort, mapping=['x', 'y']),
        False
    ),
    (   # Same Port types with full mapping, but with name permutation
        dict(Port1=XYZPort, mapping={'x': 'z', 'y': 'x', 'z': 'y'}),
        False
    ),
    (   # Complete mapping, but different Port types
        dict(Port1=XYPort, Port2=XYImperialPort, mapping=['x', 'y']),
        False
    ),
    (
        dict(Port1=XYPort, Port2=XYImperialPort),
        False
    ),
])
def test_Connector_is_mirror(settings, expected):
    """Test method `Connector.is_mirror`"""
    Port1 = settings['Port1']
    Port2 = settings.get('Port2', Port1)
    port1 = Port1('p1', PortType.IN)
    port2 = Port2('p2', PortType.OUT)
    c = Connector('test', port1, port2, settings.get('mapping', None))
    assert c.is_mirror() == expected


def test_Connector_is_mirror_ExtensiblePort():
    """Test method `Connector.is_mirror` involving extensible ports"""
    class InwardSystem(System):
        def setup(self):
            self.add_inward('x', 1.0)
            self.add_inward('d', dict(cool=True))

    class OutwardSystem(System):
        def setup(self):
            self.add_outward('x', 0.0)
            self.add_outward('d', dict(pi=3.14))

    sub_in = InwardSystem('sub_in')
    sub_out = OutwardSystem('sub_out')
    c = Connector('test', sub_in.inwards, sub_out.outwards)

    assert type(c.sink) is type(c.source)
    assert len(c) == len(c.sink)
    assert all(target == origin for target, origin in c.mapping.items())
    assert not c.is_mirror()  # by convention


def test_Connector_source_unit():
    """Check that changing 'source' updates the conversion table"""
    p1 = XYPort('p1', PortType.IN, {'x': -2, 'y': 0})
    p2 = AbcPort('p2', PortType.OUT, {'a': 1, 'b': 100, 'c': 0})
    c = Connector('p2_to_p1', p1, p2, {'x': 'a', 'y': 'b'})
    assert c.source is p2
    assert c.sink is p1
    conversion = c._unit_conversions
    assert_keys(conversion, 'x', 'y')
    assert conversion['x'] == pytest.approx((1e-3, 0), rel=1e-14)
    assert conversion['y'] == pytest.approx((1, -273.15), rel=1e-14)

    # Change source
    c.source = port = AbcdPort('port', PortType.IN)

    assert c.source is port
    assert conversion['x'] == pytest.approx((0.0254, 0), rel=1e-14)
    assert conversion['y'] == pytest.approx((5/9, -32), rel=1e-14)


def test_Connector_sink_unit():
    """Check that changing 'sink' updates the conversion table"""
    p1 = XYPort('p1', PortType.IN, {'x': -2, 'y': 0})
    p2 = AbcPort('p2', PortType.OUT, {'a': 1, 'b': 100, 'c': 0})
    c = Connector('p2_to_p1', p1, p2, {'x': 'a', 'y': 'b'})
    assert c.source is p2
    assert c.sink is p1
    conversion = c._unit_conversions
    assert_keys(conversion, 'x', 'y')
    assert conversion['x'] == pytest.approx((1e-3, 0), rel=1e-14)
    assert conversion['y'] == pytest.approx((1, -273.15), rel=1e-14)

    # Change source
    c.sink = port = XYImperialPort('port', PortType.IN)

    assert c.sink is port
    assert conversion['x'] == pytest.approx((1/304.8, 0), rel=1e-14)
    assert conversion['y'] == pytest.approx((1.8, -2298.35 / 9), rel=1e-14)


@pytest.mark.parametrize("settings, expected", [
    (
        dict(
            port1 = XYPort('p1', PortType.IN, {'x': 1, 'y': 2}),
            port2 = P('p2', PortType.OUT, {'v': 2, 'w': 4}),
        ),
        dict(mapping={'x': 'v', 'y': 'w'})
    ),
    (
        dict(
            port1 = Q('p1', PortType.IN, {'x': 1, 'y': 2, 'z': 3}),
            port2 = R('p2', PortType.OUT, {'v': 2, 'w': 4, 'z': 3}),
            mapping = lambda p1, p2: 'z',
        ),
        dict(mapping={'z': 'z'})
    ),
])
def test_Connector_mapping(ConnectorFactory, settings, expected):
    c = ConnectorFactory(**settings)
    assert c.mapping == expected['mapping']


@pytest.mark.parametrize("settings, removed, expected", [
    (
        dict(
            port1 = XPort('p1', PortType.IN, {'x': 1}),
            port2 = XPort('p2', PortType.OUT, {'x': 1}),
            mapping = lambda p1, p2: dict(zip(p1, p2)),
        ),
        ['x'], 
        # expected output:
        dict(mapping={})
    ),
    (
        dict(
            port1 = XYPort('p1', PortType.IN, {'x': 1, 'y': 2}),
            port2 = XYPort('p2', PortType.OUT, {'x': 2, 'y': 4}),
            mapping = lambda p1, p2: list(p2),
        ),
        ['x'], dict(mapping={'y': 'y'})
    ),
    (
        dict(
            port1 = XYPort('p1', PortType.IN, {'x': 1, 'y': 2}),
            port2 = P('p2', PortType.OUT, {'v': 2, 'w': 4}),
        ),
        ['x'], dict(mapping={'y': 'w'})
    ),
    (
        dict(
            port1 = XYPort('p1', PortType.IN, {'x': 1, 'y': 2}),
            port2 = P('p2', PortType.OUT, {'v': 2, 'w': 4}),
        ),
        ['w'], dict(error=KeyError)
    ),
    (
        dict(
            port1 = Q('p1', PortType.IN, {'x': 1, 'y': 2, 'z': 3}),
            port2 = R('p2', PortType.OUT, {'v': 2, 'w': 4, 'z': 3}),
        ),
        ['x', 'z'], dict(mapping={'y': 'w'})
    ),
])
def test_Connector_remove_variables(ConnectorFactory, settings, removed, expected):
    c = ConnectorFactory(**settings)
    error = expected.get('error', None)
    if error is None:
        c.remove_variables(removed)
        assert c.mapping == expected['mapping']
    else:
        pattern = expected.get('match', None)
        with pytest.raises(error, match=pattern):
            c.remove_variables(removed)


@pytest.mark.parametrize("settings, expected", [
    (
        dict(
            port1 = XYPort('p1', PortType.IN, {'x': 1, 'y': 2}),
            port2 = P('p2', PortType.OUT, {'v': 2, 'w': 4}),
        ),
        # expected output:
        dict(conversions={'x': (1, 0), 'y': (1, 0)},
            message="Connector source 'fake.p2.w' is dimensionless, but target 'fake.p1.y' has physical unit degC")
    ),
    (
        dict(
            port1 = P('p2', PortType.OUT, {'v': 2, 'w': 4}),
            port2 = XYPort('p1', PortType.IN, {'x': 1, 'y': 2}),
        ),
        dict(conversions={'v': (1, 0), 'w': (1, 0)},
            message="Connector source 'fake.p1.y' has physical unit degC, but target 'fake.p2.w' is dimensionless")
    ),
    (
        dict(
            port1 = XYZPort('p3', PortType.IN, {'x': 1, 'y': 2, 'z': 'test'}),
            port2 = XYZPort('p3', PortType.IN, {'x': 10, 'y': 20, 'z': 'test2'}),
        ),
        dict(conversions={'x': (1, 0), 'y': (1, 0), 'z': None})
    ),
])
def test_Connector_update_unit_conversion(caplog, ConnectorFactory, settings, expected):
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=connectors.__name__):
        c = ConnectorFactory(**settings)
        assert c._unit_conversions == expected['conversions']

    records = caplog.records
    expected_msg = expected.get('message', None)
    if expected_msg is None:
        assert len(records) == 0
    else:
        assert len(records) == 1
        assert expected_msg in records[-1].message


def test_Connector_transfer():
    p1 = XYPort('p1', PortType.IN, {'x': 1, 'y': 2})
    p2 = P('p2', PortType.OUT, {'v': 2, 'w': 4})
    c = Connector('p2_to_p1', p1, p2, dict(zip(p1, p2)))
    assert p1.x == 1
    assert p1.y == 2
    c.transfer()
    assert p1.x == 2  # Neutral unit conversion
    assert p1.y == 4  # No unit conversion

    p1 = XYPort('p1', PortType.IN)
    p2 = XYImperialPort('p2', PortType.OUT)
    c = Connector('p2_to_p1', p1, p2, ['x', 'y'])
    p2.x = 10.
    p2.y = 100.
    assert p1.x == 1
    assert p1.y == 1
    c.transfer()
    assert p1.x == pytest.approx(10 * 0.3048, rel=1e-15)
    assert p1.y == pytest.approx(5 / 9 * (100 - 32), rel=1e-15)


@pytest.mark.parametrize("settings, expected", [
    (
        dict(
            port1 = XYPort('p1', PortType.IN, {'x': 1, 'y': 2}),
            port2 = P('p2', PortType.OUT, {'v': 2, 'w': 4}),
        ),
        ('fake.p1', 'fake.p2', {'x': 'v', 'y': 'w'})
    ),
    (
        dict(
            port1 = Q('p1', PortType.IN, {'x': 1, 'y': 2, 'z': 3}),
            port2 = R('p2', PortType.OUT, {'v': 2, 'w': 4, 'z': 3}),
            mapping = lambda p1, p2: 'z',
        ),
        ('fake.p1', 'fake.p2', {'z': 'z'})
    ),
])
def test_Connector_to_dict(ConnectorFactory, settings, expected):
    c = ConnectorFactory(**settings)
    d = c.to_dict()
    assert_keys(d, c.name)
    assert d[c.name] == expected


def test_Connector_to_dict_system():
    p1 = Q('p1', PortType.IN, {'x': 1, 'y': 2, 'z': 3})
    p2 = R('p2', PortType.OUT, {'v': 2, 'w': 4, 'z': 3})
    c = Connector('p2_to_p1', p1, p2, 'z')
    d = c.to_dict()
    assert_keys(d, c.name)
    assert d[c.name] == ('fake.p1', 'fake.p2', {'z': 'z'})

    s = System('system')
    s.parent = MagicMock()
    p1.owner = s
    s2 = System('system2')
    p2.owner = s2
    d = c.to_dict()
    assert_keys(d, c.name)
    assert d[c.name] == ('system.p1', 'system2.p2', {'z': 'z'})

    s.parent = s2
    d = c.to_dict()
    assert_keys(d, c.name)
    assert d[c.name] == ('system.p1', 'p2', {'z': 'z'})
