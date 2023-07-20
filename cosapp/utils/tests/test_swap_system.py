import pytest
import numpy as np
import copy
import logging

from cosapp.base import System, Port
from cosapp.drivers import NonLinearSolver, EulerExplicit

from cosapp.utils.swap_system import swap_system


class AbcPort(Port):
    def setup(self):
        self.add_variable("a", 1.0)
        self.add_variable("b", 1.0)
        self.add_variable("c", np.zeros(3))


class BaseVersion(System):

    def setup(self):
        self.add_input(AbcPort, 'p_in')
        self.add_output(AbcPort, 'p_out')

        self.add_inward('x', 1.0)
        self.add_outward('y', 0.0)


class VersionA(BaseVersion):

    def compute(self):
        self.y = self.x
        self.p_out.set_from(self.p_in, transfer=copy.copy)


class VersionB(BaseVersion):

    def compute(self):
        p_in = self.p_in
        self.y = p_in.a + p_in.b * self.x**2
        self.p_out.set_values(
            a = 2 * p_in.a,
            b = 3 * p_in.b,
            c = -0.5 * p_in.c,
        )


class NonCopyable:
    def __deepcopy__(self, *args, **kwargs):
        raise copy.Error(self.__class__.__name__)


class VersionC(System):
    """System API compatible with `BaseVersion`,
    but where inward `x` is not copyable, and with
    extra variables `extra_in` and `extra_out`.
    """
    def setup(self):
        self.add_input(AbcPort, 'p_in')
        self.add_output(AbcPort, 'p_out')

        self.add_inward('x', NonCopyable())
        self.add_outward('y', 0.0)
        self.add_inward('extra_in', 0.123)
        self.add_outward('extra_out', 0.0)

    def compute(self):
        self.y = self.p_in.a
        self.p_out.set_from(self.p_in, transfer=copy.copy)
        self.extra_out = -self.extra_in


class CompositeSystem(System):
    def setup(self):
        foo = self.add_child(VersionA('foo'), pulling=['x', 'p_in'])
        bar = self.add_child(VersionA('bar'), pulling=['y', 'p_out'])

        self.connect(foo, bar, {'y': 'x', 'p_out': 'p_in'})


@pytest.fixture
def composite():
    s = CompositeSystem('composite')
    s.x = 0.1
    s.p_in.set_values(a=1.0, b=2.0, c=np.full(3, 0.5))
    return s



class ComponentTest(System):
    def setup(self):
        self.add_inward("x", 1.0)
        self.add_inward("m", 2.0)
        self.add_outward("y", 0.0)

    def compute(self):
        self.y = self.x * self.m


class SystemTest(System):
    def setup(self):
        self.add_child(ComponentTest("st11"), pulling=["x"])
        self.add_child(ComponentTest("st21"), pulling=["y"])

        self.connect(self.st11.outwards, self.st21.inwards, {"y": "x"})


class SystemTest2(System):
    def setup(self):
        self.add_child(SystemTest("st1"), pulling=["x"])
        self.add_child(SystemTest("st2"), pulling=["y"])

        self.connect(self.st1.outwards, self.st2.inwards, {"y": "x"})


class TestSwapComponents:

    def test_system(self):
        """Test systems swap"""
        sys = SystemTest("sys")

        old_system = sys.st11
        new_system = ComponentTest("st11")
        assert sys.st11 is old_system
        assert sys.st11 is not new_system

        swap_system(sys.st11, new_system)

        assert sys.st11 is not old_system
        assert sys.st11 is new_system

    def test_connectors(self):
        """Test systems swap"""
        sys = SystemTest("sys")

        old_connectors = sys.connectors()

        swap_system(sys.st11, SystemTest("st11"))

        new_connectors = sys.connectors()
        
        assert set(old_connectors.keys()) == set(new_connectors.keys())

        for key, old_connector in old_connectors.items():
            new_connector = new_connectors[key]
            assert old_connector.source is new_connector.source
            assert old_connector.sink is new_connector.sink

    def test_pop_then_add_child(self):
        """Test system swap"""
        sys = SystemTest("sys")
        old_system = sys.pop_child("st11")
        new_system = sys.add_child(ComponentTest("st11"))

        assert new_system.inwards is not old_system.inwards

    @pytest.mark.parametrize("init_values", [True, False])
    def test_do_not_init_values(self, init_values):
        """Test system swap init_values"""
        sys = SystemTest("sys")
        sys.st11.m += 1.0
        original_m = sys.st11.m

        assert sys.st11.m == original_m

        swap_system(sys.st11, ComponentTest("st11"), init_values=init_values)

        if init_values:
            assert sys.st11.m == original_m
        else:
            assert sys.st11.m != original_m

    @pytest.mark.parametrize("name", ["st11", "st21"])
    def test_run_once(self, name):
        """Test `run_once` before and after swapping"""
        sys = SystemTest("sys")

        sys.x = 2.0
        sys.y = 0.0
        sys.run_once()
        assert sys.y == 8.0

        original = swap_system(sys[name], ComponentTest(name))

        sys.x = 2.0
        sys.y = 0.0
        sys.run_once()
        assert sys.y == 8.0

        # Revert to original
        swap_system(sys[name], original)

        sys.x = 2.0
        sys.y = 0.0
        sys.run_once()
        assert sys.y == 8.0

    def test_run_driver_with_solver(self):
        """Test run once"""
        sys = SystemTest("sys")
        swap_system(sys.st11, ComponentTest("st11"))

        solver = sys.add_driver(NonLinearSolver("solver"))
        solver.add_unknown("st11.m")
        solver.add_equation("y == 2.")

        sys.x = 2.0
        sys.run_drivers()

        assert sys.st11.m == pytest.approx(0.5, rel=1e-12)


class TestSwapSystems:

    def test_connectors(self):
        """Test systems swap"""
        sys = SystemTest2("sys")
        old_connectors = sys.connectors()

        swap_system(sys.st1, SystemTest("st1"))
        new_connectors = sys.connectors()
        
        assert set(old_connectors.keys()) == set(new_connectors.keys())

        for key, old_connector in old_connectors.items():
            new_connector = new_connectors[key]
            assert old_connector.source is new_connector.source
            assert old_connector.sink is new_connector.sink

    @pytest.mark.parametrize("name", ["st1", "st2"])
    def test_run_once(self, name):
        """Test run once"""
        sys = SystemTest2("sys")

        sys.x = 2.0
        sys.y = 0.0
        sys.run_once()

        assert sys.st1.x == 2.0
        assert sys.st1.st11.x == 2.0
        assert sys.st1.st21.x == 4.0
        assert sys.st1.st21.y == 8.0
        assert sys.st2.st11.x == 8.0
        assert sys.st2.st21.x == 16.0
        assert sys.st1.y == 8.0
        assert sys.st2.x == 8.0
        assert sys.st2.y == 32.0
        assert sys.y == 32.0

        original = swap_system(sys[name], SystemTest(name))

        sys.x = 2.0
        sys.y = 0.0
        sys.run_once()

        assert sys.st1.x == 2.0
        assert sys.st1.st11.x == 2.0
        assert sys.st1.st21.x == 4.0
        assert sys.st1.st21.y == 8.0
        assert sys.st2.st11.x == 8.0
        assert sys.st2.st21.x == 16.0
        assert sys.st1.y == 8.0
        assert sys.st2.x == 8.0
        assert sys.st2.y == 32.0
        assert sys.y == 32.0

        swap_system(sys[name], original)

        sys.x = 2.0
        sys.y = 0.0
        sys.run_once()

        assert sys.st1.x == 2.0
        assert sys.st1.st11.x == 2.0
        assert sys.st1.st21.x == 4.0
        assert sys.st1.st21.y == 8.0
        assert sys.st2.st11.x == 8.0
        assert sys.st2.st21.x == 16.0
        assert sys.st1.y == 8.0
        assert sys.st2.x == 8.0
        assert sys.st2.y == 32.0
        assert sys.y == 32.0

    def test_run_driver_with_solver(self):
        """Test run driver"""
        sys = SystemTest2("sys")
        swap_system(sys.st2, SystemTest("st2"))

        solver = sys.add_driver(NonLinearSolver("solver"))
        solver.add_unknown("st1.st11.m")
        solver.add_equation("y == 2.")

        sys.x = 2.0
        sys.run_drivers()

        assert sys.st1.st11.m == 0.125

    def test_run_driver_with_solver_in_swapped_system(self):
        """Test run driver"""
        sys = SystemTest2("sys")

        solver = sys.st1.add_driver(NonLinearSolver("solver"))
        solver.add_unknown("st11.m")
        solver.add_equation("y == 2.")

        solver2 = sys.add_driver(NonLinearSolver("solver"))
        solver2.add_unknown("st2.st11.m")
        solver2.add_equation("y == 4.")

        sys.x = 2.0
        sys.run_drivers()

        assert sys.st1.y == pytest.approx(2.0, rel=1e-12)
        assert sys.st1.st11.m == pytest.approx(0.5, rel=1e-12)
        assert sys.st2.st11.m == pytest.approx(1.0, rel=1e-12)
        assert sys.st2.y == pytest.approx(4.0, rel=1e-12)
        assert sys.y == pytest.approx(4.0, rel=1e-12)

        st1_with_solver = swap_system(sys.st1, ComponentTest("st1"))

        sys.x = 2.0
        sys.run_drivers()

        assert sys.st1.y == pytest.approx(4.0, rel=1e-12)
        assert sys.st1.m == pytest.approx(2.0, rel=1e-12)
        assert sys.st2.st11.m == pytest.approx(0.5, rel=1e-12)
        assert sys.st2.y == pytest.approx(4.0, rel=1e-12)
        assert sys.y == pytest.approx(4.0, rel=1e-12)

        swap_system(sys.st1, st1_with_solver)

        sys.x = 2.0
        sys.run_drivers()

        assert sys.st1.y == pytest.approx(2.0, rel=1e-12)
        assert sys.st1.st11.m == pytest.approx(0.5, rel=1e-12)
        assert sys.st2.st11.m == pytest.approx(1.0, rel=1e-12)
        assert sys.st2.y == pytest.approx(4.0, rel=1e-12)
        assert sys.y == pytest.approx(4.0, rel=1e-12)



class TransientSystem(System):
    def setup(self):
        self.add_inward("t1", 0.0)
        self.add_transient("t1", der="1.")


class TestSwapSystemTransient:

    def test_system(self):
        """Test systems swap transient"""
        sys = System("sys")
        sys.add_child(TransientSystem("sys1"))
        swap_system(sys.sys1, TransientSystem("sys1"))

        ee = sys.add_driver(EulerExplicit(time_interval=[0, 1], dt=0.1))
        sys.run_drivers()
        assert sys.sys1.t1 == pytest.approx(1.0, rel=1e-12)

        # considering scenario
        ee.set_scenario(init={"sys1.t1": 0.0})
        sys.run_drivers()
        assert sys.sys1.t1 == pytest.approx(1.0, rel=1e-12)


class EventSystem(System):

    def setup(self):
        self.add_inward("x", 0.0)
        self.add_inward("x_max", 1.0)
        self.add_event("failure", trigger="x > x_max")
        self.add_outward_modevar("time_broken", init="time if x > x_max else inf")
        self.add_outward_modevar("broken", init="x > x_max")

    def transition(self):
        if self.failure.present:
            self.broken = True
            self.time_broken = self.time


class TestSwapSystemEvent:

    @pytest.mark.parametrize("init_values", [True, False, True])
    def test_system(self, init_values):
        """Test systems swap event"""
        sys = System("sys")
        sys.add_child(EventSystem("sub"))

        driver = sys.add_driver(EulerExplicit(time_interval=[0, 1], dt=0.1))
        driver.set_scenario(
            init={
                'sub.x_max': 0.52,
            },
            values={
                'sub.x': 't',
            },
        )

        assert sys.sub.x == 0.0
        assert not sys.sub.broken

        sys.run_drivers()

        assert sys.sub.broken
        assert sys.sub.time_broken == pytest.approx(0.52, rel=1e-14)
        assert len(driver.recorded_events) == 1
        record = driver.recorded_events[0]
        assert len(record.events) == 1
        assert record.time == sys.sub.time_broken
        assert record.events[0] is sys.sub.failure

        original = swap_system(sys.sub, EventSystem("sub"), init_values=init_values)

        if init_values:
            assert sys.sub.broken
        else:
            assert not sys.sub.broken

        sys.run_drivers()

        assert sys.sub.broken
        assert sys.sub.time_broken == pytest.approx(0.52, rel=1e-14)
        assert len(driver.recorded_events) == 1
        record = driver.recorded_events[0]
        assert len(record.events) == 1
        assert record.time == sys.sub.time_broken
        assert record.events[0] is sys.sub.failure
        assert record.events[0] is not original.failure


class TestSwapSystemTransient:

    def test_system(self):
        """Test systems swap transient"""
        sys = System("sys")
        sys.add_child(TransientSystem("sub"))
        swap_system(sys.sub, TransientSystem("sub"))
        sys.add_driver(EulerExplicit(time_interval=[0, 1], dt=0.1))
        sys.run_drivers()
        assert sys.sub.t1 == pytest.approx(1.0)

    def test_multi_system_and_event(self):
        """Test systems swap with several identical sub systems and transients"""
        sys = System("sys")
        sys.add_child(EventSystem("sys1"))
        sys.add_child(EventSystem("sys2"))

        sys.sys1.x_max = 0.4
        sys.sys2.x_max = 0.8

        swap_system(sys.sys1, EventSystem("sys1"))
        swap_system(sys.sys2, EventSystem("sys2"))

        driver = sys.add_driver(EulerExplicit(time_interval=[0, 1], dt=0.1))
        driver.set_scenario(
            values={
                'sys1.x': 't',
                'sys2.x': 't',
            },
        )
        sys.run_drivers()

        assert sys.sys1.broken
        assert sys.sys2.broken
        assert sys.sys1.time_broken == pytest.approx(sys.sys1.x_max)
        assert sys.sys2.time_broken == pytest.approx(sys.sys2.x_max)


class TestSwapVersions:

    def test_connectors(self, composite):
        """Test systems swap"""
        s = composite
        old_connectors = s.connectors()

        swap_system(s.bar, VersionB('bar'))
        new_connectors = s.connectors()
        
        assert set(old_connectors.keys()) == set(new_connectors.keys())

        for key, old_connector in old_connectors.items():
            new_connector = new_connectors[key]
            assert old_connector.source is new_connector.source
            assert old_connector.sink is new_connector.sink

    def test_run_once(self, composite: CompositeSystem):
        """Test run once"""
        s = composite
        assert isinstance(s.foo, VersionA)
        assert isinstance(s.bar, VersionA)

        s.run_once()

        assert s.x == 0.1
        assert s.p_in.a == 1.0
        assert s.p_in.b == 2.0
        assert np.array_equal(s.p_in.c, [0.5, 0.5, 0.5])
        assert s.foo.x == s.x
        assert s.foo.p_in.a == s.p_in.a
        assert s.foo.p_in.b == s.p_in.b
        assert np.array_equal(s.foo.p_in.c, s.p_in.c)
        assert np.array_equal(s.foo.p_out.c, [0.5, 0.5, 0.5])
        # Check `s.bar` outputs
        assert s.bar.y == 0.1
        assert s.bar.p_out.a == 1.0
        assert s.bar.p_out.b == 2.0
        assert np.array_equal(s.bar.p_out.c, [0.5, 0.5, 0.5])
        # Check `s` outputs
        assert s.y == 0.1
        assert s.p_out.a == pytest.approx(1, rel=1e-14)
        assert s.p_out.b == pytest.approx(2, rel=1e-14)
        assert s.p_out.c == pytest.approx([0.5, 0.5, 0.5], rel=1e-14)

        original = swap_system(s.bar, VersionB('bar'))
        assert isinstance(s.foo, VersionA)
        assert isinstance(s.bar, VersionB)

        s.run_once()

        assert s.x == 0.1
        assert s.p_in.a == 1.0
        assert s.p_in.b == 2.0
        assert np.array_equal(s.p_in.c, [0.5, 0.5, 0.5])
        assert s.foo.x == s.x
        assert s.foo.p_in.a == s.p_in.a
        assert s.foo.p_in.b == s.p_in.b
        assert np.array_equal(s.foo.p_in.c, s.p_in.c)
        assert np.array_equal(s.foo.p_out.c, [0.5, 0.5, 0.5])
        # Check `s.bar` outputs
        assert s.bar.y == pytest.approx(1.02, rel=1e-14)
        assert s.bar.p_out.a == pytest.approx(2.0, rel=1e-14)
        assert s.bar.p_out.b == pytest.approx(6.0, rel=1e-14)
        assert s.bar.p_out.c == pytest.approx([-0.25, -0.25, -0.25], rel=1e-14)
        # Check `s` outputs
        assert s.y == pytest.approx(1.02, rel=1e-14)
        assert s.p_out.a == pytest.approx(2.0, rel=1e-14)
        assert s.p_out.b == pytest.approx(6.0, rel=1e-14)
        assert s.p_out.c == pytest.approx([-0.25, -0.25, -0.25], rel=1e-14)

        swap_system(s.bar, original)
        assert isinstance(s.foo, VersionA)
        assert isinstance(s.bar, VersionA)

        s.run_once()

        assert s.x == 0.1
        assert s.p_in.a == 1.0
        assert s.p_in.b == 2.0
        assert np.array_equal(s.p_in.c, [0.5, 0.5, 0.5])
        assert s.foo.x == s.x
        assert s.foo.p_in.a == s.p_in.a
        assert s.foo.p_in.b == s.p_in.b
        assert np.array_equal(s.foo.p_in.c, s.p_in.c)
        assert np.array_equal(s.foo.p_out.c, [0.5, 0.5, 0.5])
        # Check `s.bar` outputs
        assert s.bar.y == 0.1
        assert s.bar.p_out.a == 1.0
        assert s.bar.p_out.b == 2.0
        assert np.array_equal(s.bar.p_out.c, [0.5, 0.5, 0.5])
        # Check `s` outputs
        assert s.y == 0.1
        assert s.p_out.a == pytest.approx(1, rel=1e-14)
        assert s.p_out.b == pytest.approx(2, rel=1e-14)
        assert s.p_out.c == pytest.approx([0.5, 0.5, 0.5], rel=1e-14)

    def test_copy_error(self, composite, caplog):
        """Test the handling of API mismatch and of uncopyable objects."""
        original = swap_system(composite.foo, VersionC('foo'))

        assert isinstance(original, VersionA)
        assert isinstance(composite.foo, VersionC)

        with caplog.at_level(logging.WARNING):
            swap_system(composite.foo, original)

        assert "Could not copy ['foo.extra_in', 'foo.extra_out', 'foo.x']" in caplog.text
        assert isinstance(composite.foo, VersionA)


def test_swap_system_parent_error(composite):
    with pytest.raises(ValueError, match="Cannot replace top system 'composite'"):
        swap_system(composite, System('bogus'))
    
    with pytest.raises(ValueError, match="System 'composite.bar' already belongs to a system tree"):
        swap_system(composite.foo, composite.bar)


def test_swap_system_rename(composite, caplog):
    with caplog.at_level(logging.INFO):
        swap_system(composite.foo, VersionB('bogus'))

    assert "New system 'bogus' renamed into 'foo' inside 'composite'" in caplog.text
    assert composite.foo.name == 'foo'


def test_swap_system_return(composite):
    head = composite
    assert isinstance(head.foo, VersionA)

    original = swap_system(head.foo, VersionB('foo'))

    assert isinstance(head.foo, VersionB)
    assert isinstance(original, VersionA)
    assert original.parent is None
