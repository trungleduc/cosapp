import pytest
import numpy as np
from contextlib import nullcontext as does_not_raise
from enum import Enum

from cosapp.base import System, Port
from cosapp.multimode.event import Event
from cosapp.ports.connectors import ConnectorError, BaseConnector
from cosapp.ports.mode_variable import ModeVariable
from cosapp.utils.testing import assert_keys, get_args
from cosapp.drivers import EulerExplicit
from cosapp.recorders import DataFrameRecorder


class BasicMultimodeSystem(System):
    def setup(self):
        self.add_inward_modevar("m_in", value=3., unit="m")
        self.add_outward_modevar("m_out", False, desc="System state")
        self.add_event("e_in", desc="My dummy inward event", trigger=None)


class ChildSystem(System):
    def setup(self):
        self.add_inward("my_data", value=3)
        self.add_inward("from_father")


class ChildMSystem(System):
    def setup(self):
        self.add_inward("my_data", value=3)
        self.add_inward("my_float_data", value=-1.)
        self.add_inward_modevar("my_mode_variable")


class ChildMTrigSystem(System):
    def setup(self):
        self.add_inward("a")
        self.add_inward("b")
        self.add_event("zeroxing", trigger="2. * a >= b")


class ParentSystem(System):
    def setup(self):
        self.add_outward("v", dtype=int, desc="My dummy output")


class ParentMSystem(System):
    def setup(self):
        self.add_event("e", desc="My dummy event")


def test_MultimodeSystem___init__(DummyFactory):
    s = DummyFactory("s",
        inwards = get_args("x", 1.0),
        outwards = get_args("y", 0.0),
        modevars_in = get_args("m_in", value=3.14, unit="m"),
        modevars_out = [
            get_args("m_out", False, desc="System state"),
            get_args("count", init=0, dtype=int),
        ],
        events = [
            get_args("e1", desc="My dummy event"),
            get_args("e2", trigger="x > y", final=True),
        ]
    )
    assert s.m_in == 3.14
    assert s.m_out == False
    assert s.count == 0
    assert s.e1.trigger is None
    assert not s.e1.final
    assert s.e2.is_primitive
    assert s.e2.final


def test_MultimodeSystem_children_from_System():
    p = ParentSystem("p")
    cs = ChildMSystem("c2")
    p.add_child(ChildMSystem("c1"))
    p.add_child(cs)

    assert_keys(p.children, "c1", "c2")
    assert p.children['c2'] is cs
    assert cs.parent is p
    assert p.parent is None
    assert list(p.exec_order) == ["c1", "c2"]

    assert isinstance(p.v,int)

    assert p.c1.my_data == 3
    assert p.c1.my_mode_variable == False
    assert cs.my_mode_variable == False


def test_MultimodeSystem_add_child():
    p = ParentMSystem("p")
    cs = ChildMSystem("c2")
    p.add_child(ChildSystem("c1"))
    p.add_child(cs)

    assert_keys(p.children, "c1", "c2")
    assert p.children['c2'] is cs
    assert cs.parent is p
    assert p.parent is None
    assert list(p.exec_order) == ["c1", "c2"]

    assert isinstance(p.e,Event)
    assert (not p.e.present)

    assert isinstance(p.c1.my_data,int)
    assert p.c1.my_data == 3
    assert isinstance(p.c2.my_float_data,float)
    assert cs.my_mode_variable == False


def test_MultimodeSystem_all_events():
    p = ParentMSystem("p")
    c1 = p.add_child(ChildSystem("c1"))
    c2 = p.add_child(ChildMSystem("c2"))
    cc1 = c1.add_child(BasicMultimodeSystem("yeah"))
    cc1.add_child(ParentMSystem("ccp"))
    c2.add_child(ParentMSystem("pcp"))
    p.add_child(ChildMTrigSystem("c3"))

    event_dict = {e.full_name():e for e in p.all_events()}
    assert len(event_dict) == 5
    expected = {
        "p.e" : p.e,
        "p.c1.yeah.e_in" : p.c1.yeah.e_in,
        "p.c1.yeah.ccp.e" : p.c1.yeah.ccp.e,
        "p.c2.pcp.e" : p.c2.pcp.e,
        "p.c3.zeroxing" : p.c3.zeroxing
    }
    assert event_dict == expected

    prim_event_dict = {e.full_name():e for e in p.all_events() if e.is_primitive}
    assert prim_event_dict == {"p.c3.zeroxing" : p.c3.zeroxing}

    nonprim_event_dict = {e.full_name():e for e in p.all_events() if not e.is_primitive}
    assert {**prim_event_dict, **nonprim_event_dict} == event_dict


class ContinuousVarSystem(System):
    def setup(self):
        self.add_inward("x_in_float", value=210., unit="degC")
        self.add_inward("x_in_int", value=1)
        self.add_inward("x_in_bool", value=True)
        self.add_inward("x_in_str", value="Hello.")
        self.add_outward("x_out_float", value=-210., unit="degC") # = -346 F
        self.add_outward("x_out_int", value=-1)
        self.add_outward("x_out_bool", value=False)
        self.add_outward("x_out_str", value="Fare thee well.")


class ChildMFloatIn(System):
    def setup(self):
        self.add_inward_modevar("mv", dtype=float, value=-350., unit="degF")
        self.add_inward_modevar("mv2", dtype=int)


class ChildMFloatOut(System):
    def setup(self):
        self.add_outward_modevar("mv", dtype=float, value=257., unit="degF")
        self.add_outward_modevar("mv2", dtype=int)


def test_MultimodeSystem_connect():
    top = System("top")
    cvs = top.add_child(ContinuousVarSystem("cvs"))
    c1_in = top.add_child(ChildMFloatIn("c1_in"))
    c2_in = top.add_child(ChildMFloatIn("c2_in"))
    c1_out = top.add_child(ChildMFloatOut("c1_out"))
    c2_out = top.add_child(ChildMFloatOut("c2_out"))

    top.connect(c1_in, c1_out, {'mv': 'mv'})
    top.run_once()
    assert c1_in.mv == pytest.approx(257, rel=1e-15)

    top.connect(c1_in, c2_in, {'mv2' : 'mv2'})
    # TODO Behavior in a run_once()?

    with pytest.raises(ConnectorError, match="as they are both outputs"):
        top.connect(c1_out, c2_out, {'mv2' : 'mv2'})

    # ModeVariable -> Variable
    top.connect(c2_out, cvs, {'mv' : 'x_in_float'})
    assert cvs.x_in_float == pytest.approx(210, rel=1e-15)
    top.run_once()
    assert c2_out.mv == pytest.approx(257, rel=1e-15)
    assert cvs.x_in_float == pytest.approx(125, rel=1e-15)

    # Variable -> ModeVariable: KO
    with pytest.raises(ConnectorError, match="Input mode variables cannot be connected to continuous time variables"):
        top.connect(cvs, c2_in, {'x_out_int' : 'mv2'})


@pytest.fixture
def mixed():
    class AbPort(Port):
        def setup(self):
            self.add_variable('a', 1.0)
            self.add_variable('b', np.zeros(3))

    class ContinuousTimeSystem(System):
        def setup(self):
            self.add_input(AbPort, 'p_in')
            self.add_output(AbPort, 'p_out')
            self.add_inward('x_in', 1.0)
            self.add_outward('x_out', 0.0)
    
    top = System('top')
    top.add_child(ContinuousTimeSystem('c1'))
    top.add_child(BasicMultimodeSystem('d1'))
    top.add_child(BasicMultimodeSystem('d2'))

    return top


@pytest.mark.parametrize("child1, child2, mapping, expected", [
    ('c1', 'd1', {'x_in': 'm_out'}, does_not_raise()),
    ('c1', 'd1', {'p_in.a': 'm_out'}, does_not_raise()),
    ('d1', 'd2', {'m_out': 'm_in'}, does_not_raise()),
    ('d1', 'd2', {'m_in': 'm_in'}, does_not_raise()),  # legal pulling
    (
        'c1', 'd1', {'x_out': 'm_in'},  # continuous.x_out -> discrete.m_in
        pytest.raises(
            ConnectorError,
            match="Input mode variables cannot be connected to continuous time variables",
        )
    ),
    (
        'c1', 'd1', {'p_out.a': 'm_in'},  # continuous.x_out -> discrete.m_in
        pytest.raises(
            ConnectorError,
            match="Input mode variables cannot be connected to continuous time variables",
        )
    ),
    (
        'd1', 'c1', {'m_in': 'x_in'},  # illegal pulling
        pytest.raises(
            ConnectorError,
            match="Input mode variables cannot be connected to continuous time variables",
        )
    ),
    (
        'c1', 'd1', {'x_in': 'm_in'},  # illegal pulling
        pytest.raises(
            ConnectorError,
            match="Input mode variables cannot be connected to continuous time variables",
        )
    ),
])
def test_MultimodeSystem_modevar_connection(mixed, child1, child2, mapping, expected):
    with expected:
        mixed.connect(mixed[child1], mixed[child2], mapping)


def test_MultimodeSystem_modevar_connection_in_in(mixed: System):
    """Input/input connections (thus causing pulling) involving
    mode variables and continuous time variables.

    This particular test checks that illegal input/input connectors
    do not create side effects (partial pulling) in parent system.
    """
    assert len(mixed.inwards) == 0
    assert len(mixed.modevars_in) == 0

    with pytest.raises(ConnectorError, match="Input mode variables cannot be connected to continuous time variables"):
        mixed.connect(mixed.d1, mixed.c1, {'m_in': 'x_in'})  # illegal pulling

    # Check that failed connection above
    # did not have any side effect in head system:
    assert len(mixed.inwards) == 0
    assert len(mixed.modevars_in) == 0

    with pytest.raises(ConnectorError, match="Input mode variables cannot be connected to continuous time variables"):
        mixed.connect(mixed.c1, mixed.d1, {'x_in': 'm_in'})  # illegal pulling


@pytest.mark.parametrize('pulling', [
    'm_in',
    'm_out',
    ['m_in', 'm_out'],
    {'m_in': 'm'},
    {'m_out': 'm'},
])
def test_MultimodeSystem_modevar_pulling(pulling):
    top = System('top')
    top.add_child(BasicMultimodeSystem('d'), pulling=pulling)
    mapping = BaseConnector.format_mapping(pulling)  # tested separately
    for varname in mapping.values():
        assert varname in top


def test_MultimodeSystem_modevar_pulling_attr():
    top = System('top')
    top.add_child(BasicMultimodeSystem('d'), pulling=['m_in', 'm_out'])

    m = top['modevars_in'].get_details('m_in')
    assert isinstance(m, ModeVariable)
    assert m.unit == 'm'
    assert m.description == ''

    m = top['modevars_out'].get_details('m_out')
    assert isinstance(m, ModeVariable)
    assert m.unit == ''
    assert m.description == 'System state'


def test_MultimodeSystem_mode_manager():
    """Test a pattern in which a mode manager transfers
    a mode variable to a sibling multimode system.

    The multimode system is expected to be in synch with
    the mode manager at each transition.
    """
    class ModeManager(System):
        def setup(self):
            self.add_inward('x', 0.0)
            self.add_event('pif', trigger="x > 1")
            self.add_event('paf', trigger="x < 1")
            self.add_event('zap', trigger="x > 2")
            self.add_outward_modevar('mode', init=0, dtype=int)

        def transition(self) -> None:
            if self.pif.present:
                self.mode = 1
            if self.paf.present:
                self.mode = 0
            if self.zap.present:
                self.mode = 2
    
    class MultimodeSystem(System):
        def setup(self):
            self.add_inward_modevar("mode", value=2)
            self.add_outward_modevar("state", init="get_state(mode)")

        @staticmethod
        def get_state(mode) -> float:
            if mode == 0:
                state = "A"
            elif mode == 1:
                state = "B"
            elif mode == 2:
                state = "C"
            else:
                raise ValueError
            return state

        def transition(self) -> None:
            self.state = self.get_state(self.mode)

    class TopSystem(System):
        def setup(self) -> None:
            self.add_child(ModeManager('manager'))
            self.add_child(MultimodeSystem('foo'))

            self.connect(self.manager, self.foo, 'mode')
    
    s = TopSystem('s')
    driver = s.add_driver(EulerExplicit(dt=0.1, time_interval=(0, 1)))
    driver.add_recorder(DataFrameRecorder(), period=driver.dt)
    driver.set_scenario(
        values={
            'manager.x': '3 * t',  # mode changes @ t=1/3 and 2/3
        }
    )

    s.run_drivers()

    data = driver.recorder.export_data()
    assert all(data['foo.mode'] == data['manager.mode'])
    assert all(list(map(s.foo.get_state, data['foo.mode'])) == data['foo.state'])


def test_MultimodeSystem_close_events():
    """Test event occurence within the time step directly following another event"""
    class TwoEventSystem(System):
        def setup(self) -> None:
            self.add_event('foo')
            self.add_event('bar')
    
    s = TwoEventSystem('s')

    driver = s.add_driver(EulerExplicit(dt=0.1, time_interval=(0, 0.3)))

    # Set triggers in two different time steps
    s.foo.trigger = "t == 0.12"
    s.bar.trigger = "t == 0.23"
    s.run_drivers()

    assert len(driver.recorded_events) == 2
    assert len(driver.recorded_events[0].events) == 1
    assert driver.recorded_events[0].time == 0.12
    assert driver.recorded_events[0].events[0] is s.foo 
    assert driver.recorded_events[1].time == 0.23
    assert len(driver.recorded_events[1].events) == 1
    assert driver.recorded_events[1].events[0] is s.bar 

    # Set both triggers within the same time step
    s.foo.trigger = "t == 0.12"
    s.bar.trigger = "t == 0.13"
    s.run_drivers()

    assert len(driver.recorded_events) == 2
    assert len(driver.recorded_events[0].events) == 1
    assert driver.recorded_events[0].time == 0.12
    assert driver.recorded_events[0].events[0] is s.foo
    assert driver.recorded_events[1].time == 0.13
    assert len(driver.recorded_events[1].events) == 1
    assert driver.recorded_events[1].events[0] is s.bar
