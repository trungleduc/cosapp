import pytest

from cosapp.utils.graph_analysis import get_free_inputs
from cosapp.core.variableref import VariableReference
from cosapp.tests.library.systems import AllTypesSystem
from cosapp.ports.enum import PortType
from cosapp.ports.port import BasePort, Port
from cosapp.systems import System


class XyzPort(Port):
    def setup(self):
        self.add_variable('x', 0.0)
        self.add_variable('y', 0.0)
        self.add_variable('z', 0.0)


class AbcdPort(Port):
    def setup(self):
        self.add_variable('a', 0.0)
        self.add_variable('b', 0.0)
        self.add_variable('c', 0.0)
        self.add_variable('d', 0.0)


class XySystem(System):
    def setup(self):
        self.add_input(XyzPort, 'x_in')
        self.add_output(XyzPort, 'x_out')


class AbcdSystem(System):
    def setup(self):
        self.add_input(AbcdPort, 'a_in')
        self.add_output(AbcdPort, 'a_out')


@pytest.fixture
def assembly():
    """Unconnected assembly of `XySystem` and `AbcdSystem` components"""
    head = System('head')
    head.add_child(XySystem('foo1'), pulling='x_in')
    head.add_child(XySystem('foo2'))
    head.add_child(AbcdSystem('bar1'))
    head.add_child(AbcdSystem('bar2'), pulling='a_out')

    return head


def test_get_free_inputs_0(assembly):
    # Add connections
    assembly.connect(assembly.foo1.x_out, assembly.foo2.x_in)

    inputs = get_free_inputs(assembly)

    assert set(inputs) == {
        'x_in.x', 'x_in.y', 'x_in.z',
        'foo1.x_in.x', 'foo1.x_in.y', 'foo1.x_in.z',
        'bar1.a_in.a', 'bar1.a_in.b', 'bar1.a_in.c', 'bar1.a_in.d',
        'bar2.a_in.a', 'bar2.a_in.b', 'bar2.a_in.c', 'bar2.a_in.d',
    }

    for key, alias in inputs.items():
        assert isinstance(alias, VariableReference)
        assert isinstance(alias.mapping, BasePort)
        assert alias.mapping.direction is PortType.IN

    name2var = assembly.name2variable

    assert inputs['x_in.x'] is name2var['x_in.x']
    assert inputs['x_in.y'] is name2var['x_in.y']
    assert inputs['x_in.z'] is name2var['x_in.z']

    assert inputs['foo1.x_in.x'] is name2var['x_in.x']
    assert inputs['foo1.x_in.y'] is name2var['x_in.y']
    assert inputs['foo1.x_in.z'] is name2var['x_in.z']

    for subsystem in (assembly.bar1, assembly.bar2):
        for var in 'abcd':
            key = aka = f"{subsystem.name}.a_in.{var}"
            assert inputs[key] is name2var[aka], f"key {key!r} <-- {aka!r}"


def test_get_free_inputs_1(assembly):
    # Add connections
    assembly.connect(assembly.foo1.x_out, assembly.foo2.x_in)
    assembly.connect(assembly.foo2.x_out, assembly.bar1.a_in, {'x': 'b', 'y': 'c', 'z': 'd'})
    assembly.connect(assembly.bar1.a_out, assembly.bar2.a_in)

    inputs = get_free_inputs(assembly)

    assert set(inputs) == {
        'x_in.x', 'x_in.y', 'x_in.z',
        'foo1.x_in.x', 'foo1.x_in.y', 'foo1.x_in.z',
        'bar1.a_in.a',  # all of `bar1.a_in` connected to `foo2.x_out`, except variable 'a'
    }

    for key, alias in inputs.items():
        assert isinstance(alias, VariableReference)
        assert isinstance(alias.mapping, BasePort)
        assert alias.mapping.direction is PortType.IN

    name2var = assembly.name2variable

    for var in ('x', 'y', 'z'):
        key = f"x_in.{var}"
        aka = f"x_in.{var}"
        assert inputs[key] is name2var[aka], f"key {key!r} <-- {aka!r}"

    for var in ('x', 'y', 'z'):
        key = f"foo1.x_in.{var}"
        aka = f"x_in.{var}"
        assert inputs[key] is name2var[aka], f"key {key!r} <-- {aka!r}"

    assert inputs['bar1.a_in.a'] is name2var['bar1.a_in.a']


def test_get_free_inputs_inwards():
    class Component(System):
        def setup(self):
            self.add_inward('x', 1.0)
            self.add_outward('y', 0.0)

    head = System('head')
    head.add_child(Component('comp1'), pulling={'x': 'top_in'})
    head.add_child(Component('comp2'))
    head.add_child(Component('comp3'), pulling={'y': 'top_out'})

    head.connect(head.comp1.outwards, head.comp2.inwards, {'y': 'x'})

    inputs = get_free_inputs(head)

    assert set(inputs) == {
        'top_in', 'inwards.top_in',
        'comp1.x', 'comp1.inwards.x',
        'comp3.x', 'comp3.inwards.x',
    }

    name2var = head.name2variable

    for key in (
        'top_in',
        'inwards.top_in',
        'comp1.x',
        'comp1.inwards.x',
    ):
        assert inputs[key] is name2var['inwards.top_in'], f"key {key!r}"

    assert inputs['comp3.x'] is name2var['comp3.inwards.x']
    assert inputs['comp3.inwards.x'] is name2var['comp3.inwards.x']

    # Test function on sub-systems
    comp1_inputs = get_free_inputs(head.comp1)
    assert set(comp1_inputs) == {'x', 'inwards.x'}
    for key in ('x', 'inwards.x',):
        assert comp1_inputs[key] is name2var['inwards.top_in'], f"key {key!r}"

    assert get_free_inputs(head.comp2) == dict()  # connected component

    comp3_inputs = get_free_inputs(head.comp3)
    assert set(comp3_inputs) == {'x', 'inwards.x'}
    for key in ('x', 'inwards.x',):
        assert comp3_inputs[key] is name2var['comp3.inwards.x'], f"key {key!r}"


def test_get_free_inputs_serial_pulling():
    """Test `get_free_inputs` on a system with multi-level pulling
    """
    class Component(System):
        def setup(self):
            self.add_child(XySystem('sub'), pulling={'x_in': 'c_in', 'x_out': 'c_out'})

    head = System('head')
    head.add_child(Component('comp1'), pulling={'c_in': 'h_in'})
    head.add_child(Component('comp2'), pulling={'c_in': 'h_in'})
    head.add_child(Component('comp3'))
    head.add_child(Component('comp4'))

    head.connect(head.comp3.c_out, head.comp4.c_in)

    inputs = get_free_inputs(head)

    assert set(inputs) == {
        'h_in.x', 'h_in.y', 'h_in.z',
        'comp1.c_in.x', 'comp1.c_in.y', 'comp1.c_in.z',
        'comp2.c_in.x', 'comp2.c_in.y', 'comp2.c_in.z',
        'comp3.c_in.x', 'comp3.c_in.y', 'comp3.c_in.z',
        'comp1.sub.x_in.x', 'comp1.sub.x_in.y', 'comp1.sub.x_in.z',
        'comp2.sub.x_in.x', 'comp2.sub.x_in.y', 'comp2.sub.x_in.z',
        'comp3.sub.x_in.x', 'comp3.sub.x_in.y', 'comp3.sub.x_in.z',
    }

    name2var = head.name2variable

    for subsystem in [head.comp1, head.comp2]:
        for var in ('x', 'y', 'z'):
            aka = f"h_in.{var}"  # highest level alias

            # first level pulling
            key = f"{subsystem.name}.c_in.{var}"
            assert inputs[key] is name2var[aka], f"key {key!r} <-- {aka!r}"
            
            # second level pulling
            key = f"{subsystem.name}.sub.x_in.{var}"
            assert inputs[key] is name2var[aka], f"key {key!r} <-- {aka!r}"

    for subsystem in [head.comp3]:
        for var in ('x', 'y', 'z'):
            aka = f"{subsystem.name}.c_in.{var}"

            # first level - no pulling
            key = f"{subsystem.name}.c_in.{var}"
            assert inputs[key] is name2var[aka], f"key {key!r} <-- {aka!r}"

            # second level pulling
            key = f"{subsystem.name}.sub.x_in.{var}"
            assert inputs[key] is name2var[aka], f"key {key!r} <-- {aka!r}"

    # Test function on sub-systems
    assert get_free_inputs(head.comp1.sub) == {
        'x_in.x': name2var['h_in.x'],
        'x_in.y': name2var['h_in.y'],
        'x_in.z': name2var['h_in.z'],
    }
    assert get_free_inputs(head.comp4) == dict()
    assert get_free_inputs(head.comp4.sub) == dict()

    # Shift `head` system one level down
    top = System('top')
    top.add_child(head)
    assert get_free_inputs(top.head.comp1.sub) == {
        'x_in.x': name2var['h_in.x'],
        'x_in.y': name2var['h_in.y'],
        'x_in.z': name2var['h_in.z'],
    }
    assert get_free_inputs(top.head.comp4) == dict()
    assert get_free_inputs(top.head.comp4.sub) == dict()
