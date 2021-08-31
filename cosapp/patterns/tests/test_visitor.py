import pytest

from cosapp.patterns.visitor import Component, Visitor, send


class NamedComponent(Component):
    def __init__(self, name: str):
        self.name = str(name)


class PortMockup(NamedComponent):
    def accept(self, visitor: Visitor):
        visitor.visit_port(self)


class SystemMockup(NamedComponent):
    def accept(self, visitor: Visitor):
        visitor.visit_system(self)


class DriverMockup(NamedComponent):
    def accept(self, visitor: Visitor):
        visitor.visit_driver(self)


class NameCollector(Visitor):
    """Visitor collecting entity names"""
    def __init__(self):
        self.data = dict(
            (key, list())
            for key in ('systems', 'ports', 'drivers')
        )
    
    def visit_port(self, port) -> None:
        self.data['ports'].append(port.name)
    
    def visit_system(self, system) -> None:
        self.data['systems'].append(system.name)
    
    def visit_driver(self, driver) -> None:
        self.data['drivers'].append(driver.name)


class BogusCounter(Visitor):
    """Visitor incrementing counters depending on type"""
    def __init__(self):
        self.data = dict.fromkeys(
            ('systems', 'ports', 'drivers'), 0,
        )
    
    def visit_port(self, port) -> None:
        self.data['ports'] += 1
    
    def visit_system(self, system) -> None:
        self.data['systems'] += 2
    
    def visit_driver(self, driver) -> None:
        self.data['drivers'] += 3


@pytest.fixture
def components():
    """Simple component list"""
    return [
        SystemMockup('ab'),
        PortMockup('p1'),
        DriverMockup('foo'),
        PortMockup('p2'),
        SystemMockup('a'),
        SystemMockup('c'),
    ]


def test_visitor_send(components):
    """Test function `visitor.send`
    """
    visitor = NameCollector()
    send(visitor, components)
    
    assert visitor.data['ports'] == ['p1', 'p2']
    assert visitor.data['systems'] == ['ab', 'a', 'c']
    assert visitor.data['drivers'] == ['foo']

    counter = BogusCounter()
    send(counter, components)

    assert counter.data['ports'] == 2
    assert counter.data['systems'] == 6
    assert counter.data['drivers'] == 3
