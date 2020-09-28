import pytest

import numpy as np
from cosapp.systems import System


def test_DynamicSystem_add_transient(funky):
    assert len(funky.inwards) == 4
    assert len(funky.outwards) == 1
    for var in ['m', 'v', 'x', 'foo']:
        assert var in funky.inwards
    funky.m = 1.0
    funky.run_once()
    assert funky.m == 1
    assert funky.y == pytest.approx(np.exp(1), rel=1e-15)
    assert type(funky.x) is type(funky.v)
    assert len(funky.x) == len(funky.v)
    assert funky.foo == pytest.approx(0)
    assert funky.x == pytest.approx(np.zeros(3))

    # Check that transients cannot be added outside `setup`
    with pytest.raises(AttributeError):
        funky.add_transient('banana', der='m**2')

    # Check that transient is uniquely defined
    Funky = funky.__class__
    class ErroneousSystem(Funky):
        def setup(self):
            super().setup()
            self.add_transient('x', der='[2 * m, 0, -m]')  # 'x' already declared in base class

    with pytest.raises(ArithmeticError, match="'x' is already defined as a time-dependent unknown of 'oops'"):
        erroneous = ErroneousSystem('oops')
    
    class FooBar(System):
        def setup(self):
            self.add_outward('v', np.ones(3))
            self.add_transient('x', der='v')

    s = FooBar('foobar')
    
    assert s.x is not s.v


@pytest.mark.parametrize("declared", [True, False])
def test_DynamicSystem_add_transient_onTheFlyInward(declared):
    class FooBar(System):
        def setup(self, add_f):
            if add_f:
                self.add_inward('f', np.full(3, 0.123))
            self.add_inward('v', np.ones(3))
            self.add_outward('k', 0.5)
            
            self.add_transient('f', der='v / k')

    s = FooBar('foobar', add_f=declared)
    assert len(s.inwards) == 2
    assert len(s.outwards) == 1
    assert 'f' in s.inwards
    if declared:
        assert np.array_equal(s.f, np.full(3, 0.123))
    else:
        assert np.array_equal(s.f, np.full_like(s.v, 2))


def test_DynamicSystem_add_transient_outward():
    class FooBar(System):
        def setup(self):
            self.add_inward('x', 0.1)
            self.add_outward('k', 0.5)
            self.add_transient('k', der='x')

    with pytest.raises(TypeError, match="Only input variables can be declared as transient"):
        FooBar('foobar')


def test_DynamicSystem_add_transient_outside():
    """Test the declaration of a transient unknown outside of system setup"""
    class FooBar(System):
        def setup(self):
            self.add_inward('x', np.ones(3))
            self.add_inward('v', np.ones(3))

    s = FooBar('foobar')
    assert len(s.inwards) == 2
    assert len(s.transients) == 0
    s.add_transient('x', der='v')
    assert len(s.transients) == 1

    with pytest.raises(AttributeError, match="`add_inward` cannot be called outside `setup`"):
        s.add_transient('b', der='v')


@pytest.mark.parametrize("transient, der, error", [
    ('h', '0.1', None),
    ('h', '0.1 * k', None),
    ('h', 'h', None),
    ('h', 'norm(v7)', None),
    ('f', 'f', None),
    ('f', 'v3', None),
    ('f', '-v3', None),
    ('f', 'k * v3', None),
    ('f', 'v3 - v7[::3]', None),
    ('f', 'm23[1, :]', None),
    ('f', 'k * array([1, 2, 3])', None),
    ('f', '[1, 2, 3]', None),
    # Erroneous cases (type or size incompatibility)
    ('h', '[1, 2, 3]', TypeError),
    ('f', 'k', TypeError),
    ('f', '3.14', TypeError),
    ('f', 'v7', TypeError),
    ('2 * f', 'v3', AttributeError),
    ('f', 'm23', TypeError),
    ('f', 'm23 * v7', ValueError),
    ('f', 'exp(v3', SyntaxError),
    ('h', 'banana', NameError),
    ('f', 'banana', NameError),
])
def test_DynamicSystem_add_transient_compatibility(transient, der, error):
    """Test compatibility between a transient variable and its declared derivative"""
    class FooBar(System):
        def setup(self):
            self.add_inward('h', 1.0)
            self.add_inward('f', np.full(3, 0.123))
            self.add_inward('v3', np.ones(3))
            self.add_inward('v7', np.ones(7))
            self.add_inward('m23', np.ones((2, 3)))
            self.add_outward('k', 0.5)

    s = FooBar('foobar')

    if error and issubclass(error, BaseException):
        with pytest.raises(error):
            s.add_transient(transient, der=der)
    else:
        s.add_transient(transient, der=der)


def test_DynamicSystem_transient_unknowns(funky, groovy):
    def check_keys(system, keys):
        context_msg = "system {!r}".format(system)
        problem = system.get_unsolved_problem()
        assert set(problem.transients.keys()) == set(keys), context_msg

    check_keys(funky, ['x', 'foo'])
    check_keys(groovy, ['F', 'G', 'brass.x', 'brass.foo'])

    Groovy = groovy.__class__
    class ThirdLevel(System):
        def setup(self):
            self.add_child(Groovy('sub'))
            self.add_transient('A', der='sub.brass.x')

    check_keys(ThirdLevel('third'), ['A', 'sub.F', 'sub.G', 'sub.brass.x', 'sub.brass.foo'])


def test_DynamicSystem_add_rate(jazzy):
    assert 'dB_dt' in jazzy
    assert 'dH_dt' in jazzy
    assert 'sub.dB_dt' in jazzy

    # Check that transients cannot be added outside `setup`
    with pytest.raises(AttributeError):
        jazzy.add_rate('banana', source='sub.bass')

    # Check that time derivative is uniquely defined
    Jazzy = jazzy.__class__
    class ErroneousSystem(Jazzy):
        def setup(self):
            super().setup()
            self.add_rate('bass', source='drums[0]')  # 'bass' already declared in base class

    with pytest.raises(ValueError, match="oops.bass already exists"):
        erroneous = ErroneousSystem('oops')

    # Check that initial value is compatible with source
    class ErroneousSystem(System):
        def setup(self):
            self.add_inward('x', np.zeros(6))
            self.add_rate('dx_dt', source='x', initial_value=np.ones((2, 3)))

    with pytest.raises(ValueError):
        erroneous = ErroneousSystem('oops')

    # Check that initial value is compatible with source
    class ErroneousSystem(System):
        def setup(self):
            self.add_inward('x', np.zeros(6))
            self.add_rate('dx_dt', source='x', initial_value=1)

    with pytest.raises(ValueError):
        erroneous = ErroneousSystem('oops')

    class Bogus(System):
        def setup(self):
            self.add_inward('x', np.ones(3))
            self.add_rate('dx_dt', source='x', initial_value=[1, 2, 3])
            self.add_rate('dy_dt', source='x**2', initial_value='2 * x')
    
    bogus = Bogus('ok')
    assert bogus.x == pytest.approx([1, 1, 1])
    assert bogus.dx_dt == pytest.approx([1, 2, 3])
    assert bogus.dy_dt == pytest.approx(np.full(3, 2))


def test_DynamicSystem_rates(funky, groovy, jazzy):
    def check_keys(system, keys):
        context_msg = "system {!r}".format(system)
        problem = system.get_unsolved_problem()
        assert set(problem.rates.keys()) == set(keys), context_msg

    check_keys(funky, [])
    check_keys(groovy, ['dB_dt'])
    check_keys(jazzy, ['dB_dt', 'dH_dt', 'sub.dB_dt'])


def test_DynamicSystem_get_unsolved_problem(groovy):
    problem = groovy.get_unsolved_problem()

    assert len(problem.transients) == 4
    assert "F" in problem.transients
    assert problem.transients["F"].pulled_from is None
    assert "G" in problem.transients
    assert problem.transients["G"].pulled_from is None
    assert "brass.x" in problem.transients
    assert problem.transients["brass.x"].pulled_from is None
    assert "brass.foo" in problem.transients
    assert problem.transients["brass.foo"].pulled_from is None
    assert len(problem.rates) == 1
    assert "dB_dt" in problem.rates


def test_DynamicSystem_get_unsolved_problem_with_pulling(funky):

    class GroovySystem(System):
        def setup(self):
            self.add_inward('bass', 0.0)
            self.add_inward('drums', np.zeros(3))

            self.add_child(funky, pulling="x")

            self.add_transient('F', der='bass')
            self.add_transient('G', der='drums + x')
            self.add_rate('dB_dt', source='x')

    jazzy = System("jazzy")
    jazzy.add_child(GroovySystem("groovy"), pulling=["G", "x"])
    problem = jazzy.get_unsolved_problem()

    assert len(problem.transients) == 4
    assert "groovy.F" in problem.transients
    assert problem.transients["groovy.F"].pulled_from is None
    assert "inwards.G" in problem.transients
    assert problem.transients["inwards.G"].pulled_from == jazzy.groovy.name2variable["inwards.G"]
    assert problem.transients["inwards.G"].pulled_from != jazzy.name2variable["inwards.G"]
    assert "inwards.x" in problem.transients
    assert problem.transients["inwards.x"].pulled_from == jazzy.groovy.funky.name2variable["inwards.x"] == jazzy.groovy.funky.name2variable["x"]
    assert problem.transients["inwards.x"].pulled_from != jazzy.name2variable["inwards.x"]
    assert len(problem.rates) == 1
    assert "groovy.dB_dt" in problem.rates
