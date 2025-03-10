import pytest
import numpy as np
from numpy import pi, sin, inf, array, absolute
from contextlib import nullcontext as does_not_raise

from cosapp.multimode.event import (
    Event,
    FilteredEvent,
)
from cosapp.multimode.zeroCrossing import (
    ZeroCrossing,
    EventDirection,
)
from cosapp.systems import System
from cosapp.drivers import EulerExplicit
import cosapp.recorders as recorders


class UndefinedEventSystem(System):
    def setup(self):
        self.add_event("e")

    def compute(self) -> None:
        # Hack: manual event stepping just for testing
        self.e.tick()
        self.e.step()


class ExplosiveSystem(System):
    def setup(self):
        self.add_inward('x', 0.0)
        self.add_outward('y', -1.0)

        s1 = self.add_child(UndefinedEventSystem('s1'))
        s2 = self.add_child(UndefinedEventSystem('s2'))

        boom = self.add_event('boom', trigger="y > x")
        s1.e.trigger = boom
        s2.e.trigger = boom.filter("y < 0")

    def compute(self) -> None:
        self.boom.tick()
        self.y = self.x**3
        # Hack: manual event stepping to mimick cascade resolution
        self.boom.step()
        self.s1.e.step()
        self.s2.e.step()


class BeepSystem(System):
    def setup(self):
        self.add_inward('x', 0.0)
        self.add_inward('u', np.ones(3))
        self.add_outward('y', -1.0)

        self.add_event('beep', trigger="y > x")
        self.add_event('boom')


class DumDum(System):
    def setup(self, trigger_string):
        self.add_inward("a", 1.0)
        self.add_inward("b", 0.0)
        self.add_inward("c", -1)
        self.add_event("e", trigger = trigger_string)


@pytest.mark.parametrize("comp, expected_dir", [
    (">=", EventDirection.UP),
    ("==", EventDirection.UPDOWN),
    ("<=", EventDirection.DOWN),
    (">", EventDirection.UP),
    ("<", EventDirection.DOWN)
])
@pytest.mark.parametrize("lhs", ['a', '2 * a - b', 'c - a'])
@pytest.mark.parametrize("rhs", ['c - b', 'sin(a)', 'exp(b - a)', '0'])
def test_ZeroCrossing_from_comparison(comp, expected_dir, lhs, rhs):
    system = DumDum('a', trigger_string = f"{lhs} {comp} {rhs}")
    trigger = system.e.trigger
    assert isinstance(trigger, ZeroCrossing)
    assert trigger.direction == expected_dir
    assert trigger.expression == f"{lhs.strip()} - ({rhs.strip()})"


@pytest.mark.parametrize("wrong_trigger_string, exception", [
    ("a == b == c", ValueError),
    ("a <= b > c", ValueError),
    ("a + b - c", ValueError),
    ("ab - c < 1", NameError),
])
def test_ZeroCrossing_from_wrong_comparison(wrong_trigger_string, exception):
    """Test `ZeroCrossing.from_comparison` with erroneous expressions"""
    with pytest.raises(exception):
        DumDum('a', trigger_string = wrong_trigger_string)


@pytest.mark.parametrize("cmp", ZeroCrossing.operators().keys())
@pytest.mark.parametrize("lhs", [1, 2.3, '1', 'pi', 'cos(pi)'])
@pytest.mark.parametrize("rhs", [1, 2.3, '1', 'pi', 'cos(pi)'])
def test_ZeroCrossing_from_comparison_constant(lhs, cmp, rhs):
    """Test `ZeroCrossing.from_comparison` with erroneous expressions"""
    with pytest.raises(ValueError, match="constant"):
        DumDum('a', trigger_string = f"{lhs} {cmp} {rhs}")


# Simple system with `x` being a sinewave of period 1s
class SineWave(System):
    """Test system producing a sinewave"""
    def setup(self, delta):
        # Parameters
        self.add_inward('omega', 2.0 * pi, desc='Pulsation')
        self.add_inward('tau', delta, desc='Latency')
        self.add_inward('A', 1.0, desc='Amplitude')
        self.add_inward('x0', 0.0, desc='Offset')
        self.add_inward('x1', self.x0 + self.A / 2, desc='Threshold')
        # Time dependent variables
        self.add_outward('x', self.x0, desc='Sinewave')
        self.add_outward('z', False, desc='Event is present')
        self.add_outward('d', -inf, desc='Date of last event')
        self.add_outward('n', 0, desc='Event count')
        # Zero-crossing and event
        self.add_event('e', trigger=ZeroCrossing.updown("x - x1"))
        #TODO Later: self.add_event('e', trigger="x == x1")
        #self.add_inward('e', Event(owner=self,name='e', trigger=ZeroCrossing.updown("x - x1",self)))
        # Dummy transient variable
        self.add_transient('y', der='x')

    def compute(self):
        # dynamics
        self.x = self.x0 + self.A * sin(self.omega * self.time)
        # debugging
        # print(f'x({self.time:.2f}) = {self.x:.4f}')
        # detect zero-Xing and evaluate event
        self.e.step()
        # set `z` and `date` accordingly
        if self.e.present:
            self.z = True
            self.d = self.time
            self.n += 1
        else:
            self.z = False
        # advance logical time
        self.e.tick()
        # invariant on `t`
        assert (self.d < 0) or (absolute(self.x0 - self.x1 + self.A * sin(self.omega * self.d)) <= self.tau*self.omega)


@pytest.mark.parametrize("d", EventDirection)
def test_EventDirection(d):
    assert isinstance(d, EventDirection)


@pytest.mark.skip(reason = "To rewrite: event logic integrated to time drivers")
def test_Event_sinewave():
    timeStep = 0.05
    duration = 5.
    sinewave = SineWave('sinewave', delta=timeStep)
    driver = sinewave.add_driver(EulerExplicit())

    driver.time_interval = (0, duration)
    driver.dt = 0.05

    recorder = driver.add_recorder(recorders.DataFrameRecorder(includes=['x', 'y', 'z', 'd', 'n']), period=timeStep)

    driver.set_scenario(init = {}, values = {})

    sinewave.run_drivers()

    data = recorder.export_data()
    time = array(data['time'])
    traj = {
        'time': time,
        'x': array(data['x']),
        'y': array(data['y']),
        'z': array(data['z']),
        'd': array(data['d']),
        'n': array(data['n']),
    }
    # print(traj)

    assert traj['n'][len(time)-1] == 2*duration


def test_Event_trigger():
    top = ExplosiveSystem('top')
    assert isinstance(top.boom.trigger, ZeroCrossing)
    assert isinstance(top.s1.e.trigger, Event)
    assert isinstance(top.s2.e.trigger, FilteredEvent)
    assert top.s1.e.trigger is top.boom
    assert top.boom.is_primitive
    assert not top.s1.e.is_primitive
    assert not top.s2.e.is_primitive


@pytest.mark.parametrize("condition, expected", [
    ('x > 1', does_not_raise()),
    ('norm(u) < 1', does_not_raise()),
    ('norm(u) < x', does_not_raise()),
    ('True', does_not_raise()),
    (True, does_not_raise()),
    (False, pytest.warns(RuntimeWarning, match="filtered with unconditionally false expression")),
    ("0 > 1", pytest.warns(RuntimeWarning, match="filtered with unconditionally false expression")),
    ('x', pytest.raises(TypeError, match="must be a Boolean expression")),
    ('u', pytest.raises(TypeError, match="must be a Boolean expression")),
])
def test_Event_filter(condition, expected):
    s = BeepSystem('s')
    with expected:
        s.boom.trigger = s.beep.filter(condition)


def test_Event_filter_context():
    """Check filtered event based on a condition evaluated
    in a context other than that of the source event.
    """
    class Head(System):
        def setup(self):
            self.add_child(BeepSystem("sub"))
            self.add_inward("foo", 0.0)
            self.add_event("zap")
    
    head = Head("head")

    with pytest.raises(NameError, match="'foo'"):
        head.zap.trigger = head.sub.beep.filter("foo > 1")
    
    with does_not_raise():
        head.zap.trigger = head.sub.beep.filter("foo > 1", context=head)


def test_Event_final():
    dummy = System('dummy')
    foo = Event('foo', dummy)
    assert not foo.final
    foo.final = True
    assert foo.final
    foo.final = False
    assert not foo.final

    with pytest.raises(TypeError, match="final"):
        foo.final = 'False'


def test_Event_occurrence():
    """Test event occurrences"""
    # Case 1: `top.boom` occurs, but
    # filtered event `top.s2.e` does not
    top = ExplosiveSystem('top')
    top.x = 0.1
    top.run_once()
    assert not top.boom.present
    assert not top.s1.e.present
    assert not top.s2.e.present
    top.x = 1.1
    top.run_once()
    assert top.y >= 0
    assert top.boom.present
    assert top.s1.e.present
    assert not top.s2.e.present
    # Keep going; primitive event should not be triggered,
    # even though top.y > top.x
    top.x = 1.5
    top.run_once()
    assert not top.boom.present

    # Case 2: both `top.boom` and
    # filtered event `top.s2.e` occur
    top = ExplosiveSystem('top')
    top.x = -1.5
    top.run_once()
    assert not top.boom.present
    assert not top.s1.e.present
    assert not top.s2.e.present
    top.x = -0.8
    top.run_once()
    assert top.y < 0
    assert top.boom.present
    assert top.s1.e.present
    assert top.s2.e.present  # filtered event also occurs
    # Keep going; primitive event should not be triggered,
    # even though top.y > top.x
    top.x = -0.5
    top.run_once()
    assert top.y > top.x
    assert not top.boom.present


def test_Event_merge():
    """Test merged events"""
    class Bogus(System):
        def setup(self):
            self.add_inward('x', 0.0)
            self.add_outward('y', -1.0)

            self.add_child(UndefinedEventSystem('s1'))
            self.add_child(UndefinedEventSystem('s2'))

            self.add_event('foo')
            self.add_event('bar')

        def compute(self) -> None:
            self.foo.tick()
            self.bar.tick()
            self.y = self.x**3
            # print(f"x = {self.x}, y = {self.y}, cos(y) = {np.cos(self.y)}")
            # Hack: manual event stepping to mimick cascade resolution
            self.foo.step()
            self.bar.step()
            self.s1.e.step()
            self.s2.e.step()

    s = Bogus("s")
    s.foo.trigger = "x < y"
    s.bar.trigger = "x == cos(y)"
    s.s1.e.trigger = Event.merge(
        s.foo,  # event
        s.bar.filter("x > 0"),  # filtered event
    )
    s.s2.e.trigger = Event.merge(
        s.bar,  # event
        s.s1.e, # merged events
    )
    s.x = 0.5
    s.run_once()
    assert not s.foo.present
    assert not s.bar.present
    assert not s.s1.e.present
    assert not s.s2.e.present
    s.x = 0.95
    s.run_once()
    assert s.y >= 0
    assert not s.foo.present
    assert s.bar.present
    assert s.s1.e.present
    assert s.s2.e.present
    s.x = 1.1
    s.run_once()
    assert s.y >= 0
    assert s.foo.present
    assert not s.bar.present
    assert s.s1.e.present
    assert s.s2.e.present
