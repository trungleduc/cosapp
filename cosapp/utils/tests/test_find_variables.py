import pytest

from cosapp.utils.find_variables import (
    natural_varname,
    make_wishlist,
    find_variables,
    find_variable_names,
    find_system_properties,
)
from cosapp.tests.library.systems import AllTypesSystem
from cosapp.base import System, Port
from cosapp.utils.naming import CommonPorts


class XyPort(Port):
    def setup(self):
        self.add_variable('x', 0.0)
        self.add_variable('y', 1.0)


class XyRatioPort(XyPort):
    def setup(self):
        super().setup()

    @property
    def xy_ratio(self):  # property matching '*_ratio' pattern
        return self.x / self.y

    def custom_ratio(self):  # method matching '*_ratio' pattern
        return 'not a property'


class SystemWithProps(System):
    def setup(self):
        self.add_input(XyRatioPort, 'in_')
        self.add_output(XyRatioPort, 'out')
        self.add_inward('a', 2.0)
        self.add_property('n', 12)
        self.add_property('const_ratio', 0.75)

    def compute(self):
        self.out.x = self.in_.x
        self.out.y = self.in_.y * 2

    @property
    def bogus_ratio(self) -> float:
        """Bogus property matching '*_ratio' name pattern"""
        return self.a * self.out.xy_ratio


@pytest.mark.parametrize("name, expected", [
    ('', ''),
    ('banana', 'banana'),
    ('foo.bar', 'foo.bar'),
    ('foo._bar', 'foo._bar'),
    ('foo.__bar', 'foo.__bar'),
    ('foo.inwards.a', 'foo.a'),
    ('foo.outwards.x', 'foo.x'),
    ('inwards.a', 'a'),
    ('outwards.x', 'x'),
    ('foo.modevars_in.a', 'foo.a'),
    ('foo.modevars_out.x', 'foo.x'),
    ('modevars_in.a', 'a'),
    ('modevars_out.x', 'x'),
    # Wildcard symbols left unchanged
    ('foo.?', 'foo.?'),
    ('*.foo.b?r', '*.foo.b?r'),
])
def test_natural_varname(name, expected):
    assert natural_varname(name) == expected


@pytest.mark.parametrize("case1", CommonPorts)
@pytest.mark.parametrize("case2", CommonPorts)
def test_natural_varname_weird(case1, case2):
    """Borderline cases like 'inwards.outwards.x',
    never supposed to appear in real variable names."""
    port1 = case1.value
    port2 = case2.value
    assert natural_varname(f"{port1}.{port2}.x") == "x"
    assert natural_varname(f"foo.{port1}.bar.{port2}.x") == "foo.bar.x"


@pytest.mark.parametrize("args, expected", [
    ([], dict(result=[])),
    (None, dict(result=[])),
    ('', dict(result=[''])),
    ('x', dict(result=['x'])),
    (['x', 'y'], dict()),
    ('foo.bar', dict(result=['foo.bar'])),
    ('foo.inwards.a', dict(result=['foo.a'])),
    ('foo.outwards.x', dict(result=['foo.x'])),
    ('inwards.a', dict(result=['a'])),
    ('outwards.x', dict(result=['x'])),
    # Wildcard symbols left unchanged
    ('foo.?', dict(result=['foo.?'])),
    ('*.foo.b?r', dict(result=['*.foo.b?r'])),
    (['*.foo.b?r', 'foo.?'], dict()),
    (set('abracadabra'), dict(result=['a', 'b', 'c', 'd', 'r'])),
    (dict(strange=True, weird=False), dict(result=['strange', 'weird'])),
    # Erroneous cases
    (3.14, dict(error=TypeError)),
    ([3.14], dict(error=TypeError)),
    (['x', 3.14], dict(error=TypeError)),
])
def test_make_wishlist(args, expected):
    error = expected.get('error', None)

    if error is None:
        wishlist = make_wishlist(args)
        assert isinstance(wishlist, list)
        assert set(wishlist) == set(expected.get('result', args))  # test content regardless of order

    else:
        with pytest.raises(error, match=expected.get('match', None)):
            make_wishlist(args)


@pytest.mark.parametrize("includes, expected", [
    ('', []),
    ('sub.a', ['sub.a']),
    ('sub.d', ['sub.d']),
    ('sub.inwards.a', ['sub.a']),
    ('inwards.a', ['a']),
    ('sub.outwards.d', ['sub.d']),
    ('outwards.d', ['d']),
    ('sub.?', ['sub.a', 'sub.b', 'sub.c', 'sub.e', 'sub.d']),
    ('sub.*', ['sub.in_.x', 'sub.a', 'sub.b', 'sub.c', 'sub.e', 'sub.d', 'sub.out.x']),
    (['sub.*', '*d', 'a'], ['a', 'd', 'sub.in_.x', 'sub.a', 'sub.b', 'sub.c', 'sub.e', 'sub.d', 'sub.out.x']),
    ('banana', []),
])
def test_find_variables__includes(includes, expected):
    sub = AllTypesSystem('sub')
    top = AllTypesSystem('top')
    top.add_child(sub)
    hits = find_variables(top, includes=includes, excludes=None)
    assert set(hits) == set(expected)  # test variable names regardless of order


@pytest.mark.parametrize("excludes, expected", [
    ('*', []),
    ('sub.a', [
        'in_.x', 'out.x', 'a', 'b', 'c', 'e', 'd',
        'sub.b', 'sub.c', 'sub.d', 'sub.e',
        'sub.in_.x', 'sub.out.x',
    ]),
    ('sub.inwards.a', [
        'in_.x', 'out.x', 'a', 'b', 'c', 'e', 'd',
        'sub.b', 'sub.c', 'sub.d', 'sub.e',
        'sub.in_.x', 'sub.out.x',
    ]),
    ('inwards.a', [
        'in_.x', 'out.x', 'sub.a', 'b', 'c', 'e', 'd',
        'sub.b', 'sub.c', 'sub.d', 'sub.e',
        'sub.in_.x', 'sub.out.x',
    ]),
    ('sub.d', [
        'in_.x', 'out.x', 'a', 'b', 'c', 'e', 'd',
        'sub.a', 'sub.b', 'sub.c', 'sub.e',
        'sub.in_.x', 'sub.out.x',
    ]),
    ('sub.outwards.d', [
        'in_.x', 'out.x', 'a', 'b', 'c', 'e', 'd',
        'sub.a', 'sub.b', 'sub.c', 'sub.e',
        'sub.in_.x', 'sub.out.x',
    ]),
    ('outwards.d', [
        'in_.x', 'out.x', 'a', 'b', 'c', 'e',
        'sub.d', 'sub.a', 'sub.b', 'sub.c', 'sub.e',
        'sub.in_.x', 'sub.out.x',
    ]),
    ('sub.?', [
        'a', 'b', 'c', 'e', 'd',
        'in_.x', 'out.x', 'sub.in_.x', 'sub.out.x',
    ]),
    ('sub.*', [
        'a', 'b', 'c', 'e', 'd',
        'in_.x', 'out.x',
    ]),
    (
        ['sub.*', '*d', 'a'],
        ['b', 'c', 'e', 'in_.x', 'out.x'],
    ),
    ('banana', [
        'a', 'b', 'c', 'e', 'd', 'in_.x', 'out.x',
        'sub.in_.x', 'sub.a', 'sub.b', 'sub.c', 'sub.e',
        'sub.d', 'sub.out.x',
    ]),
])
def test_find_variables__excludes(excludes, expected):
    sub = AllTypesSystem('sub')
    top = AllTypesSystem('top')
    top.add_child(sub)
    hits = find_variables(top, includes='*', excludes=excludes)
    assert set(hits) == set(expected)  # test variable names regardless of order


@pytest.mark.parametrize("criteria, expected", [
    (
        dict(includes=['*'], excludes=None),
        [
            'a', 'in_.x', 'in_.y', 'out.x', 'out.y',
            'in_.xy_ratio', 'out.xy_ratio', 'bogus_ratio',
        ],
    ),
    (
        dict(includes=['*'], excludes=None, include_const=True),
        [
            'a', 'in_.x', 'in_.y', 'out.x', 'out.y',
            'in_.xy_ratio', 'out.xy_ratio', 'bogus_ratio',
            'n', 'const_ratio',
        ],
    ),
    (
        dict(includes=['*_ratio'], excludes=None),
        ['in_.xy_ratio', 'out.xy_ratio', 'bogus_ratio'],
    ),
    (
        dict(includes=['*_ratio'], excludes='in_.*'),
        ['out.xy_ratio', 'bogus_ratio'],
    ),
    ( # System property 'bogus_ratio' not captured as 'output'
        dict(includes=['*_ratio'], excludes=None, outputs=False),
        ['in_.xy_ratio', 'bogus_ratio'],
    ),
    ( # Check that port method 'XyRatioPort.custom_ratio()' is not captured
        dict(includes=['custom*'], excludes=None), []
    ),
    (
        dict(includes=['*'], excludes='*_ratio'),
        ['a', 'in_.x', 'in_.y', 'out.x', 'out.y',],
    ),
    (
        dict(includes=['*x*'], excludes=None),
        ['in_.xy_ratio', 'out.xy_ratio', 'in_.x', 'out.x',],
    ),
])
def test_find_variables_property(criteria, expected):
    system = SystemWithProps("s")

    actual = find_variables(system, **criteria)
    assert set(actual) == set(expected)


@pytest.mark.parametrize("criteria, expected", [
    (
        dict(includes='*', excludes='*.*'),
        ['a', 'bogus_ratio'],
    ),
    (
        dict(includes='*', excludes=None),
        [
            'a', 'in_.x', 'in_.y', 'out.x', 'out.y',
            'in_.xy_ratio', 'out.xy_ratio', 'bogus_ratio',
            'foo.a', 'foo.in_.x', 'foo.in_.y', 'foo.out.x', 'foo.out.y',
            'foo.in_.xy_ratio', 'foo.out.xy_ratio', 'foo.bogus_ratio',
            'bar.x', 'bar.y', 'bar.out.x', 'bar.out.y',
        ],
    ),
])
def test_find_variables_composite_1(criteria, expected):
    class Bar(System):
        def setup(self):
            self.add_inward('x', 1.0)
            self.add_inward('y', 2.0)
            self.add_output(XyPort, 'out')

        def compute(self):
            self.out.x = self.x
            self.out.y = self.y

    top = SystemWithProps("top")
    top.add_child(SystemWithProps("foo"))
    top.add_child(Bar("bar"))

    actual = find_variables(top, **criteria)
    assert set(actual) == set(expected)


@pytest.mark.parametrize("criteria, expected", [
    (
        dict(includes='*', excludes='*.*'),
        ['a', 'b', 'c', 'd', 'e'],
    ),
    (
        dict(includes='*', excludes=None),
        [
            'a', 'b', 'c', 'd', 'e', 'in_.x', 'out.x',
            'foo.a', 'foo.in_.x', 'foo.in_.y', 'foo.out.x', 'foo.out.y',
            'foo.in_.xy_ratio', 'foo.out.xy_ratio', 'foo.bogus_ratio',
        ],
    ),
])
def test_find_variables_composite_2(criteria, expected):
    top = AllTypesSystem("top")
    top.add_child(SystemWithProps("foo"))

    actual = find_variables(top, **criteria)
    assert set(actual) == set(expected)


@pytest.mark.parametrize("criteria", [
        dict(includes='*', excludes='*.*'),
        dict(includes='*', excludes=None),
])
def test_find_variable_names(criteria):
    top = AllTypesSystem("top")
    top.add_child(SystemWithProps("foo"))

    matches = find_variables(top, **criteria)
    assert find_variable_names(top, **criteria) == sorted(matches)


@pytest.mark.parametrize("include_const, expected", [
    (False, {
        'bogus_ratio', 'foo.bogus_ratio',
    }),
    (True, {
        'n', 'const_ratio', 'bogus_ratio',
        'foo.n', 'foo.const_ratio', 'foo.bogus_ratio',
        'bar.magic_ratio',
    }),
])
def test_find_system_properties(include_const, expected):
    class Bar(System):
        def setup(self):
            self.add_inward('x', 1.0)
            self.add_inward('y', 2.0)
            self.add_output(XyPort, 'out')
            self.add_property('magic_ratio', 0.123)
            self.add_event('beep', trigger="y > x")
            self.add_inward_modevar('m_in', True)
            self.add_outward_modevar('m_out', init="y > x")

        def compute(self):
            self.out.x = self.x
            self.out.y = self.y

    top = SystemWithProps("top")
    top.add_child(SystemWithProps("foo"))
    top.add_child(Bar("bar"))

    actual = find_system_properties(top, include_const)
    assert actual == expected


@pytest.mark.parametrize("include_const, expected", [
    (False, {'bogus_ratio'}),
    (True,  {'bogus_ratio', 'const_ratio', 'n'}),
])
def test_find_system_properties_simple(include_const, expected):
    top = SystemWithProps("top")
    actual = find_system_properties(top, include_const)
    assert actual == expected


def test_find_variables_subsystem():
    """Related to https://gitlab.com/cosapp/cosapp/-/issues/143
    """
    top = System("top")
    top.add_child(SystemWithProps("foo"))

    options = dict(
        includes=["*in_.*", "*out.*"],
        excludes=[],
    )
    actual_top = set(find_variables(top, **options))
    actual_foo = set(find_variables(top.foo, **options))
    assert actual_top == {
        'foo.in_.x', 'foo.in_.y', 'foo.in_.xy_ratio',
        'foo.out.x', 'foo.out.y', 'foo.out.xy_ratio',
    }
    assert actual_foo == {
        'in_.x', 'in_.y', 'in_.xy_ratio',
        'out.x', 'out.y', 'out.xy_ratio',
    }


def test_find_variables_events():
    """Check that events are not picked up by `find_variables`."""
    class Bar(System):
        def setup(self):
            self.add_inward('a', 1.0)
            self.add_inward('b', 2.0)
            self.add_output(XyPort, 'p')
            self.add_outward_modevar('m_out', 2.0)
            self.add_property('pi', 3.14)
            self.add_event('beep')

    bar = Bar("bar")

    actual = find_variables(bar, includes="*", excludes=None)
    assert set(actual) == {'a', 'b', 'p.x', 'p.y', 'm_out'}
