import pytest
from unittest import mock

from cosapp.core.variableref import VariableReference
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


@pytest.fixture
def assembly_inputs(assembly):
    """Computes expected value of `assembly.input_mapping`,
    and returns tuple (system, input_mapping).
    """
    keys_in = {
        'x_in.x', 'x_in.y', 'x_in.z',
        'foo1.x_in.x', 'foo1.x_in.y', 'foo1.x_in.z',
        'foo2.x_in.x', 'foo2.x_in.y', 'foo2.x_in.z',
        'bar1.a_in.a', 'bar1.a_in.b', 'bar1.a_in.c', 'bar1.a_in.d',
        'bar2.a_in.a', 'bar2.a_in.b', 'bar2.a_in.c', 'bar2.a_in.d',
    }
    name2var = assembly.name2variable
    input_mapping = dict(
        (key, name2var[key])
        for key in keys_in
    )
    # Account for the pulling of `foo1.x_in`
    for var in 'xyz':
        input_mapping[f"foo1.x_in.{var}"] = name2var[f"x_in.{var}"]
    
    return assembly, input_mapping


def test_System_input_mapping_lazy(assembly):
    """Check lazy evaluation of `System.input_mapping`.
    In this test, analysis function `get_free_inputs` is patched,
    and we track the number of times it has been called.
    """
    fname = "cosapp.utils.graph_analysis.get_free_inputs"

    with mock.patch(fname, return_value={}) as analysis:
        assert analysis.call_count == 0
        assembly.input_mapping
        assert analysis.call_count == 1
        # Calling `input_mapping` again does not call `get_free_inputs`
        assembly.input_mapping
        assert analysis.call_count == 1
        # Adding a child resets input mapping
        assembly.add_child(System('dummy'))
        assembly.input_mapping
        assert analysis.call_count == 2
        assembly.input_mapping
        assert analysis.call_count == 2
        # Adding a child anywhere in tree resets input mapping
        assembly.bar1.add_child(XySystem('sub'))
        assembly.input_mapping
        assert analysis.call_count == 3
        assembly.input_mapping
        assert analysis.call_count == 3
        # Adding a connection triggers a new analysis
        assembly.connect(assembly.foo1.x_out, assembly.foo2.x_in)
        assembly.input_mapping
        assert analysis.call_count == 4
        assembly.input_mapping
        assert analysis.call_count == 4
        # Shift `assembly` system one level down
        top = System('top')
        top.add_child(assembly)
        assembly.input_mapping
        assert analysis.call_count == 4


def test_System_input_mapping_lazy_top_pulling(assembly):
    """Check lazy evaluation of `System.input_mapping`.
    In this test, system `assembly` is added altogether
    as a child of another system, with a pulling.
    """
    fname = "cosapp.utils.graph_analysis.get_free_inputs"

    with mock.patch(fname, return_value={'what': 'ever'}) as analysis:
        assert analysis.call_count == 0
        assembly.input_mapping
        assert analysis.call_count == 1
        # Shift `assembly` system one level down
        top = System('top')
        top.add_child(assembly, pulling={'x_in': 'x_in'})
        assert 'x_in' in top
        assembly.input_mapping
        assert analysis.call_count == 2


def test_System_input_mapping_top_pulling(assembly_inputs):
    """Same as `test_System_input_mapping_lazy_top_pulling`,
    with full dictionary content check.
    """
    assembly, expected = assembly_inputs
    top = System('top')
    top.add_child(assembly, pulling={'x_in': 'p_in'})
    assert 'p_in' in top
    # Update expected mapping
    for var in 'xyz':
        expected[f"x_in.{var}"] = expected[f"foo1.x_in.{var}"] = top.name2variable[f"p_in.{var}"]

    assert assembly.input_mapping == expected


def test_System_input_mapping_0(assembly_inputs):
    """Bare system, with no additional connections"""
    assembly, expected = assembly_inputs

    input_mapping = assembly.input_mapping
    assert input_mapping == expected

    for key, alias in input_mapping.items():
        assert key in assembly
        assert isinstance(alias, VariableReference)
        assert isinstance(alias.mapping, BasePort)
        assert alias.mapping.is_input


def test_System_input_mapping_1(assembly_inputs):
    """Same as test 0, with new connections"""
    assembly, expected = assembly_inputs

    # Add connections
    assembly.connect(assembly.foo1.x_out, assembly.foo2.x_in)
    # Remove connected inputs from expected mapping
    for var in 'xyz':
        expected.pop(f"foo2.x_in.{var}")

    assert assembly.input_mapping == expected


def test_System_input_mapping_2(assembly_inputs):
    """Same as test 1, with partial connections"""
    assembly: System = assembly_inputs[0]
    expected: dict = assembly_inputs[1]

    # Add connections
    assembly.connect(assembly.foo1.x_out, assembly.foo2.x_in)
    assembly.connect(assembly.foo2.x_out, assembly.bar1.a_in, {'x': 'b', 'y': 'c', 'z': 'd'})
    assembly.connect(assembly.bar1.a_out, assembly.bar2.a_in)

    # Remove connected inputs from expected mapping
    for var in 'xyz':
        expected.pop(f"foo2.x_in.{var}")
    for var in 'bcd':
        expected.pop(f"bar1.a_in.{var}")
    for var in 'abcd':
        expected.pop(f"bar2.a_in.{var}")

    assert set(assembly.input_mapping) == {
        'x_in.x', 'x_in.y', 'x_in.z',
        'foo1.x_in.x', 'foo1.x_in.y', 'foo1.x_in.z',
        'bar1.a_in.a',  # all of `bar1.a_in` connected to `foo2.x_out`, except variable 'a'
    }
    assert assembly.input_mapping == expected
    assert assembly.foo1.input_mapping == {
        f"x_in.{var}": assembly.name2variable[f"x_in.{var}"]
        for var in 'xyz'
    }
    assert assembly.bar1.input_mapping == {
        'a_in.a': assembly.name2variable['bar1.a_in.a'],
    }
    assert assembly.name2variable['bar1.a_in.a'] is assembly.bar1.name2variable['a_in.a']
    assert assembly.foo2.input_mapping == {}
    assert assembly.bar2.input_mapping == {}


def test_System_input_mapping_add_child(assembly_inputs):
    assembly, expected = assembly_inputs

    assembly.bar1.add_child(XySystem('sub'))
    # Update expected mapping
    for var in 'xyz':
        key = f"bar1.sub.x_in.{var}"
        expected[key] = assembly.name2variable[key]

    assert assembly.input_mapping == expected


def test_System_input_mapping_pop_child(assembly_inputs):
    assembly, expected = assembly_inputs

    assembly.bar1.add_child(XySystem('sub'))
    # New keys added by new sub-system `bar1.sub`
    # Note: check keys only; values already tested in
    # `test_System_input_mapping_add_child`
    assert set(assembly.input_mapping) == set(expected).union(
        f"bar1.sub.x_in.{var}" for var in 'xyz'
    )
    # Revert to original mapping after popping `bar1.sub`
    assembly.bar1.pop_child('sub')
    assert assembly.input_mapping == expected


def test_System_input_mapping_inwards():
    class Component(System):
        def setup(self):
            self.add_inward('x', 1.0)
            self.add_outward('y', 0.0)

    head = System('head')
    head.add_child(Component('comp1'), pulling={'x': 'top_in'})
    head.add_child(Component('comp2'))
    head.add_child(Component('comp3'), pulling={'y': 'top_out'})

    head.connect(head.comp1.outwards, head.comp2.inwards, {'y': 'x'})

    input_mapping = head.input_mapping

    assert set(input_mapping) == {
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
        assert input_mapping[key] is name2var['inwards.top_in'], f"key {key!r}"

    assert input_mapping['comp3.x'] is name2var['comp3.inwards.x']
    assert input_mapping['comp3.inwards.x'] is name2var['comp3.inwards.x']

    # Test function on sub-systems
    comp1_inputs = head.comp1.input_mapping
    assert set(comp1_inputs) == {'x', 'inwards.x'}
    for key in ('x', 'inwards.x',):
        assert comp1_inputs[key] is name2var['inwards.top_in'], f"key {key!r}"

    assert head.comp2.input_mapping == dict()  # connected component

    comp3_inputs = head.comp3.input_mapping
    assert set(comp3_inputs) == {'x', 'inwards.x'}
    for key in ('x', 'inwards.x',):
        assert comp3_inputs[key] is name2var['comp3.inwards.x'], f"key {key!r}"


def test_System_input_mapping_serial_pulling():
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

    input_mapping = head.input_mapping

    assert set(input_mapping) == {
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
            assert input_mapping[key] is name2var[aka], f"key {key!r} <-- {aka!r}"
            
            # second level pulling
            key = f"{subsystem.name}.sub.x_in.{var}"
            assert input_mapping[key] is name2var[aka], f"key {key!r} <-- {aka!r}"

    for subsystem in [head.comp3]:
        for var in ('x', 'y', 'z'):
            aka = f"{subsystem.name}.c_in.{var}"

            # first level - no pulling
            key = f"{subsystem.name}.c_in.{var}"
            assert input_mapping[key] is name2var[aka], f"key {key!r} <-- {aka!r}"

            # second level pulling
            key = f"{subsystem.name}.sub.x_in.{var}"
            assert input_mapping[key] is name2var[aka], f"key {key!r} <-- {aka!r}"

    # Test function on sub-systems
    assert head.comp1.sub.input_mapping == {
        'x_in.x': name2var['h_in.x'],
        'x_in.y': name2var['h_in.y'],
        'x_in.z': name2var['h_in.z'],
    }
    assert head.comp4.input_mapping == dict()
    assert head.comp4.sub.input_mapping == dict()

    # Shift `head` system one level down
    top = System('top')
    top.add_child(head)
    assert top.head.comp1.sub.input_mapping == {
        'x_in.x': name2var['h_in.x'],
        'x_in.y': name2var['h_in.y'],
        'x_in.z': name2var['h_in.z'],
    }
    assert top.head.comp4.input_mapping == dict()
    assert top.head.comp4.sub.input_mapping == dict()
