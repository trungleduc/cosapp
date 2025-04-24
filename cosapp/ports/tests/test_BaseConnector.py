import pytest
from unittest import mock
from types import MappingProxyType
from functools import partial

from cosapp.ports.connectors import BaseConnector, ConnectorError
from cosapp.ports.port import PortType, Port
from cosapp.systems import System
from cosapp.utils.testing import DummySystemFactory, get_args


@pytest.fixture(autouse=True)
def PatchBaseConnector():
    """Patch `BaseConnector` to make it instanciable for tests"""
    patcher = mock.patch.multiple(
        BaseConnector,
        __abstractmethods__ = set(),
    )
    patcher.start()
    yield
    patcher.stop()

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


class CustomPort(Port):
    """Port with peer-to-peer connector"""
    class Connector(BaseConnector):
        def transfer(self) -> None:
            pass

# <codecell>

@pytest.fixture(scope="function")
def ConnectorFactory():
    """BaseConnector factory from construction settings"""
    def factory(port1, port2, name="p2_to_p1", mapping=lambda p1, p2: dict(zip(p1, p2))) -> BaseConnector:
        return BaseConnector(name, port1, port2, mapping(port1, port2))
    return factory


@pytest.mark.parametrize("settings, expected", [
    (
        dict(
            port1 = XPort('p1', PortType.IN),
            port2 = XPort('p2', PortType.OUT),
            mapping = lambda p1, p2: 'x',
        ),
        dict(mapping={'x': 'x'})
    ),
    (
        dict(
            port1 = XPort('p1', PortType.IN),
            port2 = YPort('p2', PortType.OUT),
            mapping = lambda p1, p2: {'x': 'y'},
        ),
        dict(mapping={'x': 'y'})
    ),
    (
        dict(
            port1 = XYPort('p1', PortType.IN),
            port2 = XYPort('p2', PortType.OUT),
            mapping = lambda p1, p2: list(p2),
        ),
        dict(mapping={'x': 'x', 'y': 'y'})
    ),
    (
        dict(
            port1 = XYPort('p1', PortType.IN),
            port2 = P('p2', PortType.OUT),
            mapping = lambda p1, p2: dict(zip(p1, p2)),
        ),
        dict(mapping={'x': 'v', 'y': 'w'})
    ),
    (
        dict(
            port1 = Q('p1', PortType.IN),
            port2 = XYPort('p2', PortType.OUT, {'x': 2, 'y': 4}),
            mapping = lambda p1, p2: list(p2),
        ),
        dict(mapping={'x': 'x', 'y': 'y'})
    ),
    (
        dict(
            port1 = XYPort('p1', PortType.IN, {'x': 1, 'y': 2}),
            port2 = R('p2', PortType.OUT, {'v': 2, 'w': 4, 'z': 3}),
            mapping = lambda p1, p2: dict(zip(p1, ['w', 'z'])),
        ),
        dict(mapping={'x': 'w', 'y': 'z'})
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
        dict(mapping={k: k for k in list('abc')})
    ),
    (
        dict(
            port1 = XYPort('p1', PortType.IN),
            port2 = XYImperialPort('p2', PortType.OUT),
            mapping = lambda p1, p2: 'y',
        ),
        dict(mapping={'y': 'y'})
    ),
    (
        dict(
            port1 = XYPort('p1', PortType.IN),
            port2 = XYImperialPort('p2', PortType.OUT),
            mapping = lambda p1, p2: ['x', 'y'],
        ),
        dict(mapping={'x': 'x', 'y': 'y'})
    ),
    (
        dict(
            port1 = XPort('p1', PortType.IN),
            port2 = XMassPort('p2', PortType.OUT),
            mapping = lambda p1, p2: 'x',
        ),
        # no error **in base class** even though units are incompatible
        dict(mapping={'x': 'x'})
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
def test_BaseConnector__init__(ConnectorFactory, settings, expected):
    error = expected.get('error', None)

    if error is None:
        c = ConnectorFactory(**settings)
        assert c.name == settings.get('name', 'p2_to_p1')
        assert c.sink is settings['port1']
        assert c.source is settings['port2']
        assert c.mapping == expected['mapping']

    else:
        pattern = expected.get('match', None)
        with pytest.raises(error, match=pattern):
            c = ConnectorFactory(**settings)


def test_BaseConnector__repr__():
    p1 = Q('p1', PortType.IN, {'x': 1, 'y': 2, 'z': 3})
    p2 = R('p2', PortType.OUT, {'v': 2, 'w': 4, 'z': 3})

    c = BaseConnector('p2 -> p1', p1, p2, {'z': 'w'})
    assert repr(c) == "BaseConnector(fake.p1 <- fake.p2, {'z': 'w'})"

    c = BaseConnector('p2 -> p1', p1, p2, 'z')
    assert repr(c) == "BaseConnector(fake.p1 <- fake.p2, ['z'])"

    c = CustomPort.Connector('p2 -> p1', p1, p2, 'z')
    assert repr(c) == "CustomPort.Connector(fake.p1 <- fake.p2, ['z'])"


@pytest.mark.parametrize("data, expected", [
    ([], {}),
    ('', {}),
    ('x', {'x': 'x'}),
    (['x', 'y'], {'x': 'x', 'y': 'y'}),  # list
    (('x', 'y'), {'x': 'x', 'y': 'y'}),  # tuple
    ({'x', 'y'}, {'x': 'x', 'y': 'y'}),  # set
    ({'x': 'v', 'y': 'w'}, {'x': 'v', 'y': 'w'}),
    (
        MappingProxyType({'x': 'a', 'y': 'b', 'z': 'c'}),
        {'x': 'a', 'y': 'b', 'z': 'c'},
    ),
    # Mixed collections of str and str/str mappings
    (['x', 'y', {'a': 'u', 'b': 'v'}], {'x': 'x', 'y': 'y', 'a': 'u', 'b': 'v'}),
    (['x', {'a': 'u', 'b': 'v'}, 'y'], {'x': 'x', 'y': 'y', 'a': 'u', 'b': 'v'}),
    (['x', {'a': 'u'}, 'y', MappingProxyType({'b': 'v'})], {'x': 'x', 'y': 'y', 'a': 'u', 'b': 'v'}),
    # Erroneous cases
    (0, pytest.raises(TypeError)),
    (None, pytest.raises(TypeError)),
    (['x', ['y', 'z']], pytest.raises(TypeError)),
])
def test_BaseConnector_format_mapping(data, expected):
    """Test static method `BaseConnector.format_mapping`"""
    if isinstance(expected, dict):
        assert BaseConnector.format_mapping(data) == expected
    
    else:
        with expected:
            BaseConnector.format_mapping(data)


def test_BaseConnector_empty():
    port1 = XYZPort('p1', PortType.IN)
    port2 = AbcPort('p2', PortType.OUT)
    c = BaseConnector('p2_p1', port1, port2)
    assert c.mapping == dict()


@pytest.mark.parametrize("ptype", PortType)
def test_BaseConnector__init__same_port(ptype):
    port = XYPort('port', ptype, {'x': 1, 'y': 2})
    with pytest.raises(ConnectorError, match="Source and sink cannot be the same object"):
        c = BaseConnector('oops', port, port, ['x', 'y'])


def test_BaseConnector_name():
    p1 = XPort('p1', PortType.IN, {'x': 1})
    p2 = XPort('p2', PortType.OUT, {'x': 1})
    connector_name = 'p2_to_p1'
    c = BaseConnector(connector_name, p1, p2, 'x')
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
def test_BaseConnector_source_sink(attr, port, expected):
    """Test 'source' and 'sink' attributes"""
    p1 = XPort('p1', PortType.IN, {'x': -2})
    p2 = XPort('p2', PortType.OUT, {'x': 1})
    connector_name = 'p2_to_p1'
    c = BaseConnector(connector_name, p1, p2, 'x')
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
    ('source', XPort('p3', PortType.IN), dict(error=ConnectorError, match=r"variables \['a', 'b'\] do not exist in port XPort")),
    ('source', APort('p3', PortType.IN), dict(error=ConnectorError, match=r"variable 'b' does not exist in port APort")),
    ('sink', XYPort('p3', PortType.IN), dict(error=None)),
    ('sink', XYZPort('p3', PortType.IN), dict(error=None)),  # sink has extra variable `z` - OK
    ('sink', PortWithNoOwner('p3', PortType.IN), dict(error=ConnectorError, match="owner is undefined")),
    ('sink', XPort('p3', PortType.IN), dict(error=ConnectorError, match=r"variable 'y' does not exist in port XPort")),
    ('sink', APort('p3', PortType.IN), dict(error=ConnectorError, match=r"variables \['x', 'y'\] do not exist in port APort")),
])
def test_BaseConnector_source_sink_mapping(attr, port, expected):
    """
    Test 'source' and 'sink' setters in partial connector AbcdPort -> XYPort,
    with mapping {'x': 'a', 'y': 'b'}
    """
    p1 = XYPort('p1', PortType.IN, {'x': -2, 'y': 0})
    p2 = AbcPort('p2', PortType.OUT, {'a': 1, 'b': 100, 'c': 0})
    c = BaseConnector('p2_to_p1', p1, p2, {'x': 'a', 'y': 'b'})
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


def test_BaseConnector_source_sink_variables():
    p1 = XYPort('p1', PortType.IN, {'x': -2, 'y': 0})
    p2 = AbcPort('p2', PortType.OUT, {'a': 1, 'b': 100, 'c': 0})
    c = BaseConnector('p2_to_p1', p1, p2, {'x': 'a', 'y': 'b'})
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
def test_BaseConnector_is_mirror(settings, expected):
    """Test method `BaseConnector.is_mirror`"""
    Port1 = settings['Port1']
    Port2 = settings.get('Port2', Port1)
    port1 = Port1('p1', PortType.IN)
    port2 = Port2('p2', PortType.OUT)
    c = BaseConnector('test', port1, port2, settings.get('mapping', None))
    assert c.is_mirror() == expected


def test_BaseConnector_is_mirror_ExtensiblePort():
    """Test method `BaseConnector.is_mirror` involving extensible ports"""
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
    c = BaseConnector('test', sub_in.inwards, sub_out.outwards)

    assert type(c.sink) is type(c.source)
    assert len(c) == len(c.sink)
    assert all(target == origin for target, origin in c.mapping.items())
    assert not c.is_mirror()  # by convention


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
def test_BaseConnector_mapping(ConnectorFactory, settings, expected):
    c = ConnectorFactory(**settings)
    assert c.mapping == expected['mapping']
    # Check that connector mapping is immutable
    with pytest.raises(TypeError, match="does not support item assignment"):
        c.mapping['foo'] = 'bar'


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
def test_BaseConnector_remove_variables(ConnectorFactory, settings, removed, expected):
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
def test_BaseConnector_info(ConnectorFactory, settings, expected):
    """Test related methods `info` and `to_dict`."""
    c = ConnectorFactory(**settings)
    info = c.info()
    assert info == expected
    assert c.to_dict() == {c.name: expected}


def test_BaseConnector_info_system():
    p1 = Q('p1', PortType.IN, {'x': 1, 'y': 2, 'z': 3})
    p2 = R('p2', PortType.OUT, {'v': 2, 'w': 4, 'z': 3})
    c = BaseConnector('p2_to_p1', p1, p2, 'z')
    assert c.info() == ('fake.p1', 'fake.p2', {'z': 'z'})

    s = System('system')
    s.parent = mock.MagicMock()
    p1.owner = s
    s2 = System('system2')
    p2.owner = s2
    assert c.info() == ('system.p1', 'system2.p2', {'z': 'z'})

    s.parent = s2
    assert c.info() == ('system.p1', 'p2', {'z': 'z'})


def test_BaseConnector_contextual_name():
    Foo = DummySystemFactory(
        "Foo",
        inwards=[get_args('x', 0.0), get_args('u', 0.0)],
        outwards=[get_args('v', 0.0), get_args('z', 0.0)],
        inputs=[get_args(AbcdPort, 'a_in'), get_args(XYZPort, 'x_in'), get_args(AbcPort, 'p_in')],
        outputs=[get_args(AbcdPort, 'a_out'), get_args(XYZPort, 'x_out'), get_args(AbcPort, 'p_out')],
    )
    Bar = DummySystemFactory(
        "Bar",
        inwards=[get_args('u', 0.0), get_args('z', 0.0)],
        outwards=[get_args('result', 0.0)],
        inputs=[get_args(AbcdPort, 'a_in'), get_args(XYZPort, 'x_in'), get_args(AbcPort, 'p_in')],
        outputs=[get_args(AbcdPort, 'a_out'), get_args(XYZPort, 'x_out'), get_args(AbcPort, 'p_out')],
    )
    class Assembly(System):
        """Foo-Bar assembly, with various parent/child and sibling connectors"""
        def setup(self) -> None:
            foo = self.add_child(Foo('foo'), pulling=['p_in', {'x': 'X'}, 'u'])
            bar = self.add_child(Bar('bar'), pulling=['p_out', {'result': 'Y'}])

            self.connect(foo, bar, ['z', {'v': 'u'}])
            self.connect(bar.x_out, foo.x_in)
            self.connect(foo.a_out, bar.a_in, {'a': 'c', 'b': 'd'})

    # Encapsulate assembly of interest into a bogus system tree s0.s1.s2
    s0 = System('s0')
    s1 = s0.add_child(System('s1'))
    s2 = s1.add_child(System('s2'))
    s3 = s2.add_child(Assembly('s3'), pulling=['X', 'Y', 'p_out'])  # assembly of interest, containing connectors

    # Check pretty mapping
    assert sorted(map(BaseConnector.pretty_mapping, s3.all_connectors())) == [
        'X⟶x, u',
        'a, b, c',
        'a, b, c',
        'a⟶c, b⟶d',
        'result⟶Y',
        'x, y, z',
        'z, v⟶u',
    ]

    # No context specified -> context is full path s0.s1.s2.s3
    assert set(map(BaseConnector.contextual_name, s3.all_connectors())) == {
        # Pulling connectors
        's0.s1.s2.s3[p_in ⟶ foo.p_in]',
        's0.s1.s2.s3[bar.p_out ⟶ p_out]',
        's0.s1.s2.s3[inwards ⟶ foo.inwards] (X⟶x, u)',
        's0.s1.s2.s3[bar.outwards ⟶ outwards] (result⟶Y)',
        # sub-system connectors
        's0.s1.s2.s3[bar.x_out ⟶ foo.x_in]',
        's0.s1.s2.s3[foo.a_out ⟶ bar.a_in] (a⟶c, b⟶d)',
        's0.s1.s2.s3[foo.outwards ⟶ bar.inwards] (z, v⟶u)',
    }

    # Same without variable name mappings
    assert set(map(partial(BaseConnector.contextual_name, with_mapping=False), s3.all_connectors())) == {
        # Pulling connectors
        's0.s1.s2.s3[p_in ⟶ foo.p_in]',
        's0.s1.s2.s3[bar.p_out ⟶ p_out]',
        's0.s1.s2.s3[inwards ⟶ foo.inwards]',
        's0.s1.s2.s3[bar.outwards ⟶ outwards]',
        # sub-system connectors
        's0.s1.s2.s3[bar.x_out ⟶ foo.x_in]',
        's0.s1.s2.s3[foo.a_out ⟶ bar.a_in]',
        's0.s1.s2.s3[foo.outwards ⟶ bar.inwards]',
    }

    # Context is s0: connector names start with path s1.s2.s3
    assert set(map(partial(BaseConnector.contextual_name, context=s0), s3.all_connectors())) == {
        # Pulling connectors
        's1.s2.s3[p_in ⟶ foo.p_in]',
        's1.s2.s3[bar.p_out ⟶ p_out]',
        's1.s2.s3[inwards ⟶ foo.inwards] (X⟶x, u)',
        's1.s2.s3[bar.outwards ⟶ outwards] (result⟶Y)',
        # sub-system connectors
        's1.s2.s3[bar.x_out ⟶ foo.x_in]',
        's1.s2.s3[foo.a_out ⟶ bar.a_in] (a⟶c, b⟶d)',
        's1.s2.s3[foo.outwards ⟶ bar.inwards] (z, v⟶u)',
    }

    # Context is s0.s1: connector names start with path s2.s3
    assert set(map(partial(BaseConnector.contextual_name, context=s1), s3.all_connectors())) == {
        # Pulling connectors
        's2.s3[p_in ⟶ foo.p_in]',
        's2.s3[bar.p_out ⟶ p_out]',
        's2.s3[inwards ⟶ foo.inwards] (X⟶x, u)',
        's2.s3[bar.outwards ⟶ outwards] (result⟶Y)',
        # sub-system connectors
        's2.s3[bar.x_out ⟶ foo.x_in]',
        's2.s3[foo.a_out ⟶ bar.a_in] (a⟶c, b⟶d)',
        's2.s3[foo.outwards ⟶ bar.inwards] (z, v⟶u)',
    }

    # Context is s0.s1.s2: connector names start with path s3
    assert set(map(partial(BaseConnector.contextual_name, context=s2), s3.all_connectors())) == {
        # Pulling connectors
        's3[p_in ⟶ foo.p_in]',
        's3[bar.p_out ⟶ p_out]',
        's3[inwards ⟶ foo.inwards] (X⟶x, u)',
        's3[bar.outwards ⟶ outwards] (result⟶Y)',
        # sub-system connectors
        's3[bar.x_out ⟶ foo.x_in]',
        's3[foo.a_out ⟶ bar.a_in] (a⟶c, b⟶d)',
        's3[foo.outwards ⟶ bar.inwards] (z, v⟶u)',
    }

    # Context is s0.s1.s2.s3: brackets are omitted
    assert set(map(partial(BaseConnector.contextual_name, context=s3), s3.all_connectors())) == {
        # Pulling connectors
        'p_in ⟶ foo.p_in',
        'bar.p_out ⟶ p_out',
        'inwards ⟶ foo.inwards (X⟶x, u)',
        'bar.outwards ⟶ outwards (result⟶Y)',
        # sub-system connectors
        'bar.x_out ⟶ foo.x_in',
        'foo.a_out ⟶ bar.a_in (a⟶c, b⟶d)',
        'foo.outwards ⟶ bar.inwards (z, v⟶u)',
    }

    # Context is s0.s1, omitting mappings
    assert set(map(partial(BaseConnector.contextual_name, context=s1, with_mapping=False), s3.all_connectors())) == {
        # Pulling connectors
        's2.s3[p_in ⟶ foo.p_in]',
        's2.s3[bar.p_out ⟶ p_out]',
        's2.s3[inwards ⟶ foo.inwards]',
        's2.s3[bar.outwards ⟶ outwards]',
        # sub-system connectors
        's2.s3[bar.x_out ⟶ foo.x_in]',
        's2.s3[foo.a_out ⟶ bar.a_in]',
        's2.s3[foo.outwards ⟶ bar.inwards]',
    }

    # Check s2 connectors
    assert set(map(BaseConnector.contextual_name, s2.all_connectors())) == {
        # Pulling connectors
        's0.s1.s2[inwards ⟶ s3.inwards] (X)',
        's0.s1.s2[s3.outwards ⟶ outwards] (Y)',
        's0.s1.s2[s3.p_out ⟶ p_out]',
    }

    # Check s2 connectors in the context of its child s3 -> error
    with pytest.raises(ValueError, match="'s2' is not a child of 's3'"):
        set(map(partial(BaseConnector.contextual_name, context=s3), s2.all_connectors()))
