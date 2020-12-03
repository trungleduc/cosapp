import pytest

from cosapp.utils.find_variables import natural_varname, make_wishlist, find_variables
from cosapp.tests.library.systems.vectors import AllTypesSystem
from cosapp.ports import Port
from cosapp.systems import System


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
    # Wildcard symbols left unchanged
    ('foo.?', 'foo.?'),
    ('*.foo.b?r', '*.foo.b?r'),
    # Borderline cases - never supposed to occur in real variable names
    ('foo.outwards.bar.outwards.x', 'foo.bar.x'),
    ('foo.outwards.bar.inwards.x', 'foo.bar.x'),
    ('inwards.outwards.x', 'x'),
    ('outwards.inwards.x', 'x'),
])
def test_natural_varname(name, expected):
    assert natural_varname(name) == expected


@pytest.mark.parametrize("args, expected", [
    ([], dict(list=[])),
    (None, dict(list=[])),
    ('', dict(list=[''])),
    ('x', dict(list=['x'])),
    (['x', 'y'], dict()),
    ('foo.bar', dict(list=['foo.bar'])),
    ('foo.inwards.a', dict(list=['foo.a'])),
    ('foo.outwards.x', dict(list=['foo.x'])),
    ('inwards.a', dict(list=['a'])),
    ('outwards.x', dict(list=['x'])),
    # Wildcard symbols left unchanged
    ('foo.?', dict(list=['foo.?'])),
    ('*.foo.b?r', dict(list=['*.foo.b?r'])),
    (['*.foo.b?r', 'foo.?'], dict()),
    (set('abracadabra'), dict(list=['a', 'b', 'c', 'd', 'r'])),
    (dict(strange=True, weird=False), dict(list=['strange', 'weird'])),
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
        assert set(wishlist) == set(expected.get('list', args))  # test content regardless of order

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
    assert set(hits) == set(expected)  # test lists regardless of order


@pytest.mark.parametrize("excludes, expected", [
    ('*', []),
    ('sub.a', ['in_.x', 'out.x', 'a', 'b', 'c', 'e', 'd',
                'sub.b', 'sub.c', 'sub.d', 'sub.e',
                'sub.in_.x', 'sub.out.x']),
    ('sub.inwards.a', ['in_.x', 'out.x', 'a', 'b', 'c', 'e', 'd',
                        'sub.b', 'sub.c', 'sub.d', 'sub.e',
                        'sub.in_.x', 'sub.out.x']),
    ('inwards.a', ['in_.x', 'out.x', 'sub.a', 'b', 'c', 'e', 'd',
                   'sub.b', 'sub.c', 'sub.d', 'sub.e',
                   'sub.in_.x', 'sub.out.x']),
    ('sub.d', ['in_.x', 'out.x', 'a', 'b', 'c', 'e', 'd',
                'sub.a', 'sub.b', 'sub.c', 'sub.e',
                'sub.in_.x', 'sub.out.x']),
    ('sub.outwards.d', ['in_.x', 'out.x', 'a', 'b', 'c', 'e', 'd',
                         'sub.a', 'sub.b', 'sub.c', 'sub.e',
                         'sub.in_.x', 'sub.out.x']),
    ('outwards.d', ['in_.x', 'out.x', 'a', 'b', 'c', 'e',
                    'sub.d', 'sub.a', 'sub.b', 'sub.c', 'sub.e',
                    'sub.in_.x', 'sub.out.x']),
    ('sub.?', ['a', 'b', 'c', 'e', 'd', 'in_.x', 'out.x', 'sub.in_.x', 'sub.out.x']),
    ('sub.*', ['a', 'b', 'c', 'e', 'd', 'in_.x', 'out.x']),
    (['sub.*', '*d', 'a'], ['b', 'c', 'e', 'in_.x', 'out.x']),
    ('banana', ['a', 'b', 'c', 'e', 'd', 'in_.x', 'out.x',
                'sub.in_.x', 'sub.a', 'sub.b', 'sub.c', 'sub.e',
                'sub.d', 'sub.out.x']),
])
def test_find_variables__excludes(excludes, expected):
    sub = AllTypesSystem('sub')
    top = AllTypesSystem('top')
    top.add_child(sub)
    hits = find_variables(top, includes='*', excludes=excludes)
    assert set(hits) == set(expected)  # test lists regardless of order


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
    ( # Check that port method 'XyPort.custom_ratio()' is not captured
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
    class XyPort(Port):
        def setup(self):
            self.add_variable('x', 0.0)
            self.add_variable('y', 1.0)

        @property
        def xy_ratio(self):  # property matching '*_ratio' pattern
            return self.x / self.y

        def custom_ratio(self):  # method matching '*_ratio' pattern
            return 'foo'

    class MySystem(System):
        def setup(self):
            self.add_input(XyPort, 'in_')
            self.add_output(XyPort, 'out')
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

    system = MySystem("s")

    assert set(find_variables(system, **criteria)) == set(expected)
