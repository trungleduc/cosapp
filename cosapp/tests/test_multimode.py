"""Various integration tests with multimode systems
"""
from __future__ import annotations
import pytest
import numpy

from cosapp.base import Port, System
from cosapp.drivers import EulerExplicit, RungeKutta, NonLinearSolver
from cosapp.recorders import DataFrameRecorder
from cosapp.multimode import PeriodicTrigger
from cosapp.utils import swap_system


class ElecPort(Port):
    def setup(self):
        self.add_variable('V', 0.0, unit='V')
        self.add_variable('I', 1.0, unit='A')


class Resistor(System):
    """Resistor component

    Attributes:
        R : float
            Resistance in Ohms
    """
    def setup(self, R=0):
        self.add_input(ElecPort, 'elec_in')
        self.add_output(ElecPort, 'elec_out')

        self.add_inward('R', abs(float(R)), unit='ohm', desc='Internal resistance')
        self.add_outward('deltaV')

    def compute(self):
        self.elec_out.I = self.elec_in.I
        self.elec_out.V = self.elec_in.V - (self.elec_out.I * self.R)
        self.deltaV = self.elec_in.V - self.elec_out.V


class Node(System):
    """System representing an electric circuit node with
    an arbitrary number of incoming and outgoing branches.

    Nodes provide an off-design problem ensuring potential equality
    and global current balance (see 'Unknowns' and 'Equations' below).

    Constructor arguments:
    ----------------------
    - n_in [int], optional: Number of incoming branches. Defaults to 1.
    - n_out [int], optional: Number of outgoing branches. Defaults to 1.

    Properties:
    -----------
    - n_in [int]: Number of incoming branches.
    - n_out [int]: Number of outgoing branches.
    - incoming: Tuple containing all `ElecPort` inputs.
    - outgoing: Tuple containing all `ElecPort` outputs.

    Unknowns:
    ---------
    - n_out current fractions (one per outgoing branch), if n_out > 1.

    Equations:
    ----------
    - (n_in - 1) potential equality conditions for incoming branches.
    - 1 total current balance equation, if n_out > 1.
    """
    def setup(self, n_in=1, n_out=1):
        """Node constructor.

        Arguments:
        -----------
        - n_in [int], optional: Number of incoming branches. Defaults to 1.
        - n_out [int], optional: Number of outgoing branches. Defaults to 1.
        """
        self.add_property('n_in', int(n_in))
        self.add_property('n_out', int(n_out))

        if min(self.n_in, self.n_out) < 1:
            raise ValueError("Node needs at least one incoming and one outgoing branch")

        self.add_property('incoming',
            tuple(
                self.add_input(ElecPort, f"elec_in{i}")
                for i in range(self.n_in)
            )
        )
        self.add_property('outgoing',
            tuple(
                self.add_output(ElecPort, f"elec_out{i}")
                for i in range(self.n_out)
            )
        )

        if self.n_out > 1:  # unnecessary otherwise
            self.add_inward('I_frac',
                value = numpy.full(self.n_out, 1.0 / self.n_out),
                desc = f"Current fractions distributed to outgoing branches",
                limits = (0, 1),
            )
            self.add_unknown('I_frac', lower_bound=0, upper_bound=1)
            self.add_equation('sum(I_frac) == 1', name='Current balance')

        for i in range(1, self.n_in):   # case where node is 'joiner'
            self.add_equation(f'elec_in{i}.V == elec_in0.V')

        self.add_outward('V', 0., unit='V', desc='Actual node voltage')
        self.add_outward('sum_I_in', 0., unit='A', desc='Sum of all incoming currents')
        self.add_outward('sum_I_out', 0., unit='A', desc='Sum of all outgoing currents')

    def compute(self):
        # Sum of incoming currents
        self.sum_I_in = I = sum(port.I for port in self.incoming)

        # Output voltage
        self.V = V = numpy.mean([port.V for port in self.incoming])

        # Current distribution
        try:
            I_frac = self.I_frac
        except AttributeError:
            I_frac = [1]
        for j, port in enumerate(self.outgoing):
            port.V = V
            port.I = I * I_frac[j]

        self.sum_I_out = I * sum(I_frac)

    @classmethod
    def make(cls, parent, name, incoming: list[ElecPort], outgoing: list[ElecPort], pulling=None) -> Node:
        """Factory method making appropriate connections with parent system"""
        node = cls(name, n_in=max(len(incoming), 1), n_out=max(len(outgoing), 1))
        parent.add_child(node, pulling=pulling)
        
        for branch_elec, node_elec in zip(incoming, node.incoming):
            parent.connect(branch_elec, node_elec)
        
        for branch_elec, node_elec in zip(outgoing, node.outgoing):
            parent.connect(branch_elec, node_elec)

        return node


class TwoWayCircuit(System):
    def setup(self):
        self.add_outward_modevar('upbranch', True)
        self.reconfig()

        self.add_outward('Requiv', 0.0)
        self.add_outward('deltaV', 0.0)
        self.add_event('switch', trigger='elec_in.V == elec_out.V')
    
    def compute(self):
        self.deltaV = self.elec_in.V - self.elec_out.V
        I = self.elec_in.I
        self.Requiv = self.deltaV / I if abs(I) > 0 else numpy.nan

    def transition(self):
        if self.switch.present:
            self.upbranch = not self.upbranch
            self.reconfig()

    def reconfig(self):
        for name in self.children:
            self.pop_child(name)
        pulled_ports = ['elec_in', 'elec_out']
        if self.upbranch:
            child = self.add_child(Resistor("R1", R=100), pulling=pulled_ports)
        else:
            child = self.add_child(Resistor("R2", R=500), pulling=pulled_ports)
        self.elec_out.V = child.elec_out.V = 0.0


class TwoWayCircuitWithEq(System):
    def setup(self):
        self.add_outward_modevar('upbranch', True)
        self.add_child(Resistor("R0", R=100), pulling=['elec_in', 'elec_out'])
        self.reconfig()

        self.add_outward('Requiv', 0.0)
        self.add_outward('deltaV', 0.0)
        self.add_event('switch', trigger='elec_in.V == elec_out.V')
    
    def compute(self):
        self.deltaV = self.elec_in.V - self.elec_out.V
        I = self.elec_in.I
        self.Requiv = self.deltaV / I if I != 0 else numpy.nan

    def transition(self):
        if self.switch.present:
            self.upbranch = not self.upbranch
            self.reconfig()

    def reconfig(self):
        problem = self.problem
        if not self.upbranch and problem.is_empty():
            self.add_unknown("R0.R").add_equation("R0.R == 500")
        else:
            problem.clear()
            self.R0.R = 100


@pytest.fixture
def case_TwoWayCircuit():
    circuit = TwoWayCircuit('circuit')
    driver = circuit.add_driver(
        EulerExplicit(time_interval=[0, 1], dt=0.1)
    )
    driver.add_recorder(
        DataFrameRecorder(includes=['elec_in.*', 'elec_out.V', 'Requiv', 'deltaV', 'upbranch']),
        period = 0.1,
    )
    solver = driver.add_child(NonLinearSolver('solver', tol=1e-9))
    solver.add_unknown('elec_in.I').add_equation("elec_out.V == 0")
    return circuit, driver


def test_TwoWayCircuit(case_TwoWayCircuit, caplog):
    """Test multimode system with system reconfiguration (new sub-system)"""
    circuit, driver = case_TwoWayCircuit
    omega = 6
    driver.set_scenario(
        values = {
            "elec_in.V": f"cos({omega} * t)",
        }
    )
    with caplog.at_level("INFO", logger="cosapp.drivers.time.base"):
        circuit.run_drivers()

    assert len(caplog.messages) == 2
    for message in caplog.messages:
        assert message.startswith("System structure changed during transition @t=")

    df = driver.recorder.export_data()
    # print("", df.drop(['Section', 'Status', 'Error code'], axis=1), sep="\n")

    for i, row in df.iterrows():
        I = row['elec_in.I']
        R = row['Requiv']
        context = f"row #{i}, {I = }, {list(circuit.exec_order)}"
        if I > 1e-12:
            assert R == pytest.approx(100), context
        elif I < -1e-12:
            assert R == pytest.approx(500), context
    
    assert [record.time for record in driver.recorded_events] == pytest.approx(
        [(2 * k + 1) * 0.5 * numpy.pi / omega for k in range(2)]
    )


@pytest.fixture
def case_TwoWayCircuitWithEq():
    circuit = TwoWayCircuitWithEq('circuit')
    driver = circuit.add_driver(
        EulerExplicit(time_interval=[0, 1], dt=0.1)
    )
    driver.add_recorder(
        DataFrameRecorder(includes=['elec_in.*', 'elec_out.V', 'Requiv', 'deltaV', 'upbranch']),
        period = 0.1,
    )
    solver = driver.add_child(NonLinearSolver('solver', tol=1e-9))
    solver.add_unknown('elec_in.I').add_equation("elec_out.V == 0")
    return circuit, driver


def test_TwoWayCircuitWithEq(case_TwoWayCircuitWithEq, caplog):
    """Test multimode system with system reconfiguration (intrinsic problem)"""
    circuit, driver = case_TwoWayCircuitWithEq
    omega = 6
    driver.set_scenario(
        values = {
            "elec_in.V": f"cos({omega} * t)",
        }
    )
    with caplog.at_level("INFO", logger="cosapp.drivers.time.base"):
        circuit.run_drivers()

    assert len(caplog.messages) == 2
    for message in caplog.messages:
        assert message.startswith("System structure changed during transition @t=")

    df = driver.recorder.export_data()
    # print("", df.drop(['Section', 'Status', 'Error code'], axis=1), sep="\n")

    for i, row in df.iterrows():
        I = row['elec_in.I']
        R = row['Requiv']
        context = f"row #{i}, {I = }"
        if I > 1e-12:
            assert R == pytest.approx(100), context
        elif I < -1e-12:
            assert R == pytest.approx(500), context
    
    assert [record.time for record in driver.recorded_events] == pytest.approx(
        [(2 * k + 1) * 0.5 * numpy.pi / omega for k in range(2)]
    )


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


def test_MultimodeSystem_event_init():
    """Test checking that the system is up-to-date when primary events are initialized.
    """
    class Kinematics(System):
        def setup(self):
            self.add_inward('v', numpy.zeros(2))
            self.add_transient('x', der='v')

    class Wall(System):
        def setup(self):
            self.add_inward('x', numpy.zeros(2), desc='Point on the wall')
            self.add_inward('n_dir', numpy.r_[1., 1.], desc='Normal direction')
            self.add_outward('n', numpy.r_[0., 0.], desc='Unit normal vector')

        def compute(self) -> None:
            self.n = self.n_dir / numpy.linalg.norm(self.n_dir)
        
        def distance(self, point) -> float:
            """Signed distance to wall"""
            return self.n.dot(numpy.asarray(point) - self.x)

    class Assembly(System):
        def setup(self):
            self.add_child(Kinematics('kinematics'), pulling=['x', 'v'])
            self.add_child(Wall('wall'))
            self.add_event('rebound', trigger='wall.distance(x) == 0')

        def transition(self) -> None:
            if self.rebound.present:
                n = self.wall.n
                v = self.v
                v -= (2 * v.dot(n)) * n

    s = Assembly('s')

    driver = s.add_driver(EulerExplicit('driver', time_interval=(0, 1), dt=1.0))
    driver.set_scenario(
        init={
            'x': [0., 0.],
        },
        values={
            'v': [1., 0.],
            'wall.x': [0.25, 0.],
            'wall.n_dir': [-1., 0.],
        },
        stop=s.rebound,
    )
    # Initialize the wall normal with an **incorrect** value.
    # If the system is not up-to-date, the actual distance to the wall
    # will be not be correctly initialized, and the rebound will be missed.
    s.wall.n[:] = [1., 0.]

    s.run_drivers()

    assert len(driver.recorded_events) == 1
    record = driver.recorded_events[-1]
    assert len(record.events) == 2
    assert record.events[0] is s.rebound
    assert record.events[1] is driver.scenario.stop
    assert record.time == pytest.approx(0.25, rel=1e-15)


def test_MultimodeSystem_transition_order():
    """Check mode initialisation and transition
    across a multimode system tree.
    """
    class ModeManager(System):
        def setup(self):
            self.add_inward('x', 0.0)
            self.add_event('pif', trigger="x > 1")
            self.add_event('paf', trigger="x < 1")
            self.add_outward_modevar('mode', init="0 if x < 1 else 1", dtype=int)

        def transition(self) -> None:
            if self.pif.present:
                self.mode = 1
            if self.paf.present:
                self.mode = 0
    
    class MultimodeSystem(System):
        def setup(self, shift=10):
            self.add_property('shift', shift)
            self.add_inward_modevar("m_in", value=2, dtype=int)
            self.add_outward_modevar("m_out", init=f"m_in + {shift}", dtype=int)

        def transition(self) -> None:
            self.m_out = self.m_in + self.shift
    
    class MultimodeAssembly(System):
        def setup(self):
            a = self.add_child(MultimodeSystem('a', shift=10), pulling='m_in')
            b = self.add_child(MultimodeSystem('b', shift=20), pulling='m_in')
            c = self.add_child(MultimodeSystem('c', shift=30), pulling='m_out')

            self.connect(b, c, {'m_out': 'm_in'})

    class TopSystem(System):
        def setup(self) -> None:
            self.add_child(ModeManager('manager'))
            self.add_child(MultimodeAssembly('foo'))

            self.connect(self.manager, self.foo, {'mode': 'm_in'})
    
    s = TopSystem('s')
    driver = s.add_driver(EulerExplicit(dt=0.1, time_interval=(0, 1)))
    driver.add_recorder(DataFrameRecorder(excludes='*.shift'), period=driver.dt)
    driver.set_scenario(
        values={
            "manager.x": "4 * t if t < 0.5 else 4 * (1 - t)",  # mode changes @ t=1/4 and 3/4
        }
    )

    s.run_drivers()

    data = driver.recorder.export_data()
    # data = data.drop(['Section', 'Status', 'Error code'], axis=1)
    # import pandas
    # pandas.set_option('display.width', 200)
    # pandas.set_option('display.max_columns', 200)
    # print("\n", data)

    expected_modes = numpy.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0])

    assert numpy.array_equal(data['foo.m_in'], expected_modes)
    assert numpy.array_equal(data['foo.m_in'], data['manager.mode'])
    assert numpy.array_equal(data['foo.a.m_in'], data['foo.m_in'])     # pulling
    assert numpy.array_equal(data['foo.b.m_in'], data['foo.m_in'])     # pulling
    assert numpy.array_equal(data['foo.c.m_in'], data['foo.b.m_out'])  # sibling connection
    assert numpy.array_equal(data['foo.c.m_out'], numpy.asarray(data['foo.c.m_in']) + s.foo.c.shift)
    assert numpy.array_equal(data['foo.a.m_out'], expected_modes + s.foo.a.shift)
    assert numpy.array_equal(data['foo.b.m_out'], expected_modes + s.foo.b.shift)
    assert numpy.array_equal(data['foo.c.m_out'], expected_modes + s.foo.b.shift + s.foo.c.shift)


def test_MultimodeSystem_loop_recomposition():
    """Check multimode system containing new loops after transition.
    """
    class System1(System):
        def setup(self):
            self.add_inward('x', 1.0)
            self.add_inward('y', 1.0)
            self.add_outward('z', 0.0)

        def compute(self):
            self.z = self.x + self.y

    class System2(System):
        def setup(self):
            self.add_inward('k', 1.0)
            self.add_inward('x', 1.0)
            self.add_outward('y', 0.0)

        def compute(self):
            self.y = self.k + self.x**2

    class MultimodeAssembly(System):
        def setup(self):
            a = self.add_child(System1('a'))
            b = self.add_child(System2('b'))
            c = self.add_child(System2('c'))
            
            self.add_event('click')

            self.connect(a, b, {'z': 'x', 'y': 'y'})  # loop
            self.connect(b, c, {'y': 'x'})

        def transition(self):
            if self.click.present:
                self.connect(self.a, self.c, {'x': 'y'})
    
    s = MultimodeAssembly('s')
    s.add_driver(NonLinearSolver('solver'))

    s.a.x = -2.0
    s.b.k = 1.0
    s.c.k = -2.0
    s.run_drivers()
    # print("", s.drivers['solver'].problem, sep="\n")
    assert s.drivers['solver'].results.success
    assert s.drivers['solver'].problem.shape == (1, 1)

    s.drivers.clear()
    driver = s.add_driver(EulerExplicit(dt=1.0, time_interval=(0, 1)))
    driver.add_child(NonLinearSolver('solver'))
    # driver.add_recorder(DataFrameRecorder(), period=1.0)

    s.click.trigger = "t == 0.4"

    try:
        s.run_drivers()
    except:
        raise
    finally:
        # data = driver.recorder.export_data()
        # data = data.drop(['Section', 'Status', 'Error code'], axis=1)
        # pandas.set_option('display.width', 200)
        # pandas.set_option('display.max_columns', 200)
        # print("", driver.solver.problem, "", data, sep="\n")
        problem = driver.solver.problem
        assert problem.shape == (2, 2)


def test_MultimodeSystem_event_init():
    """Test checking that the system is up-to-date when primary events are initialized.
    """
    class Kinematics(System):
        def setup(self):
            self.add_inward('v', numpy.zeros(2))
            self.add_transient('x', der='v')

    class Wall(System):
        def setup(self):
            self.add_inward('x', numpy.zeros(2), desc='Point on the wall')
            self.add_inward('n_dir', numpy.r_[1., 1.], desc='Normal direction')
            self.add_outward('n', numpy.r_[0., 0.], desc='Unit normal vector')

        def compute(self) -> None:
            self.n = self.n_dir / numpy.linalg.norm(self.n_dir)
        
        def distance(self, point) -> float:
            """Signed distance to wall"""
            return self.n.dot(numpy.asarray(point) - self.x)

    class Assembly(System):
        def setup(self):
            self.add_child(Kinematics('kinematics'), pulling=['x', 'v'])
            self.add_child(Wall('wall'))
            self.add_event('rebound', trigger='wall.distance(x) == 0')

        def transition(self) -> None:
            if self.rebound.present:
                n = self.wall.n
                v = self.v
                v -= (2 * v.dot(n)) * n

    s = Assembly('s')

    driver = s.add_driver(EulerExplicit('driver', time_interval=(0, 1), dt=1.0))
    driver.set_scenario(
        init={
            'x': [0., 0.],
        },
        values={
            'v': [1., 0.],
            'wall.x': [0.25, 0.],
            'wall.n_dir': [-1., 0.],
        },
        stop=s.rebound,
    )
    # Initialize the wall normal with an **incorrect** value.
    # If the system is not up-to-date, the actual distance to the wall
    # will be not be correctly initialized, and the rebound will be missed.
    s.wall.n[:] = [1., 0.]

    s.run_drivers()

    assert len(driver.recorded_events) == 1
    record = driver.recorded_events[-1]
    assert len(record.events) == 2
    assert record.events[0] is s.rebound
    assert record.events[1] is driver.scenario.stop
    assert record.time == pytest.approx(0.25, rel=1e-15)


@pytest.mark.parametrize("t0", [0.0, 2.4, 10.0])
@pytest.mark.parametrize("period", [0.033, 0.87, 1.123, 1.0, 2.0])
def test_MultimodeSystem_single_periodic_event(t0, period):
    """Test a system with a periodic event.
    """
    class PeriodicEventSystem(System):
        def setup(self, period=1.0):
            self.add_event('ping', trigger=PeriodicTrigger(period, t0=t0))

    s = PeriodicEventSystem('s', period=period)
    driver = s.add_driver(EulerExplicit(time_interval=(0, 10), dt=1.0))

    try:
        s.run_drivers()
    except RuntimeError:
        raise
    finally:
        t_end = driver.time_interval[1]
        expected_times = numpy.arange(t0 + period, t_end * (1.0 + 1e-14), period)
        event_times = numpy.array([record.time for record in driver.recorded_events])
        # print("", f"{event_times = }", f"{expected_times = }", sep="\n")
        assert event_times == pytest.approx(expected_times)


@pytest.mark.parametrize("t0", [0.0, 2.4, 10.0])
@pytest.mark.parametrize("period", [0.033, 0.87, 1.123, 1.0, 2.0])
def test_MultimodeSystem_multiple_periodic_events(t0, period):
    """Test a system with two periodic events, one with twice larger period than the other.
    Consequently, once every two records, one single event occurs (the faster one),
    and both occur the rest of the time.
    """
    class PeriodicEventSystem(System):
        def setup(self, period=1.0):
            self.add_event('fast', trigger=PeriodicTrigger(period, t0=t0))
            self.add_event('slow', trigger=PeriodicTrigger(2 * period, t0=t0))

    s = PeriodicEventSystem('s', period=period)
    driver = s.add_driver(EulerExplicit(time_interval=(0, 10), dt=1.0))

    s.run_drivers()

    t_end = driver.time_interval[1]
    expected_times = numpy.arange(t0 + period, t_end * (1.0 + 1e-14), period)
    event_times = numpy.array([record.time for record in driver.recorded_events])
    assert event_times == pytest.approx(expected_times)
    for record in driver.recorded_events[0::2]:
        assert record.events == [s.fast]
    for record in driver.recorded_events[1::2]:
        assert set(record.events) == {s.fast, s.slow}


class MultimodeOde(System):
    """Multimode ODE of the kind df/dt = df,
    with event `snap` (undefined by default).
    """
    def setup(self, varname="f"):
        self.add_inward(f"d{varname}", 0.0)
        self.add_transient(varname, der=f"d{varname}")

        self.add_event("snap")
        self.add_outward_modevar("snapped", init=False)

    def transition(self):
        if self.snap.present:
            self.snapped = True


class MultimodeOdeUv(MultimodeOde):
    """Extension of MultimodeOde with variables (u, du),
    plus output variable v = 2u, and event `cross`.
    """
    def setup(self):
        super().setup(varname="u")
        self.add_outward("v", 0.0)

        self.add_event("cross", trigger="u == v")
    
    def compute(self):
        self.v = 2 * self.u


@pytest.mark.parametrize("trigger", ["u == 0", "v == 0"])
def test_MultimodeSystem_nested_primary_events(trigger):
    """Test with nested primary events. One primary event, `s.ode_u.cross`,
    is triggered by a condition based either on an input (u == 0) or an output (v == 0).
    """
    class MultimodeAssembly(System):

        def setup(self):
            self.add_child(MultimodeOde("ode_x"), pulling={"f": "x", "df": "dx"})
            self.add_child(MultimodeOdeUv("ode_u"))

            self.add_inward("jump", 1.0)
            
            self.add_event("cross", trigger="x == ode_u.v")

        def transition(self):
            if self.ode_x.snap.present:
                self.ode_u.u += self.jump

    # Create assembly and simulation case
    s = MultimodeAssembly("s")

    driver = s.add_driver(EulerExplicit())
    driver.add_recorder(
        DataFrameRecorder(
            excludes=["*.snapped*", "ode_x.*", "jump"]
        ),
        period=0.1,
    )
    driver.set_scenario(
        init={"x": 0, "ode_u.u": -0.5},
        values={
            "dx": "1 if not ode_x.snapped else 0",
            "ode_u.du": "2 if not ode_u.snapped else -1",
        },
    )

    driver.time_interval = (0, 1)
    driver.dt = 0.1

    s.ode_u.cross.trigger = trigger  # u(t) = v(t) = 0
    s.ode_x.snap.trigger = "f > 0.55"
    s.ode_u.snap.trigger = s.ode_u.cross
    s.jump = 0.5

    s.run_drivers()

    event_records = driver.recorded_events
    assert len(event_records) == 3
    assert event_records[0].time == pytest.approx(0.25)
    assert event_records[1].time == pytest.approx(0.55)
    assert event_records[2].time == pytest.approx(0.75)
    assert event_records[0].events[0] is s.ode_u.cross
    assert event_records[1].events[0] is s.ode_x.snap
    assert event_records[2].events[0] is s.ode_u.cross
    assert set(event_records[0].events) == {s.ode_u.cross, s.ode_u.snap}
    assert set(event_records[1].events) == {s.ode_x.snap, s.ode_u.cross, s.ode_u.snap}
    assert set(event_records[2].events) == {s.ode_u.cross, s.ode_u.snap}


def test_MultimodeSystem_filter_context():
    """Integration test for a filtered event based on an expression evaluated
    in a context other than that of the event.
    Here, this feature is used on the stop condition of a time driver.
    """
    class Head(System):
        def setup(self):
            self.add_child(MultimodeOde("ode"))
            self.add_inward("x", 0.0)
            self.add_outward("y", 0.0)
        
        def compute(self):
            self.y = self.x + self.ode.f
    
    head = Head("head")
    ode: MultimodeOde = head.ode
    ode.snap.trigger = "f == 0.5"

    driver = head.add_driver(RungeKutta(order=3, time_interval=[0, 3], dt=0.01))
    driver.add_recorder(DataFrameRecorder(), period=0.1)
    driver.set_scenario(
        init={"ode.f": 0.0},
        values={
            "ode.df": "pi * cos(pi * t)",  # -> f(t) = sin(pi * t)
            "x": "t",  # -> y(t) = t + sin(pi * t)
        },
        stop=ode.snap.filter("y > 2", context=head),
    )
    head.run_drivers()

    records = driver.recorded_events
    assert len(records) == 3
    assert [record.time for record in records] == pytest.approx(numpy.r_[1.0, 5.0, 13.0] / 6)
    assert records[0].events == [head.ode.snap]
    assert records[1].events == [head.ode.snap]
    assert records[2].events == [head.ode.snap, driver.scenario.stop]
    assert head.y > 2.0


def test_MultimodeSystem_new_transients_1():
    """Test a transition which brings in new transient variables.
    Test #1: new transient variable x, with constant time derivative.
    """
    class MultimodeSystem(System):
        """System having a transient that add un sub system having a transient.in `transition`.
        """
        def setup(self):
            self.add_event("tada", trigger="time == 5.0")

        def transition(self):
            if self.tada.present:
                self.add_child(TransientSystem("sub"))

    class TransientSystem(System):
        """System having a transient.
        """
        def setup(self) -> None:
            self.add_inward("s", 1.0)
            self.add_inward("x", 0.0)

            self.add_transient("x", der="s")

    system = MultimodeSystem("system")
    system.add_driver(EulerExplicit(time_interval=[0, 10.0], dt=1.0))

    assert not hasattr(system, "sub")

    system.run_drivers()

    assert hasattr(system, "sub")
    assert system.sub.x == 5.0


def test_MultimodeSystem_new_transients_2():
    """Test a transition which brings in new transient variables.
    Test #2: swap from variables (x, y) to (x, z), where (x, z) are stacked as a single transient.
    """
    class MultimodeSystem(System):
        """System having a transient that add un sub system having a transient.in `transition`.
        """
        def setup(self):
            self.add_event("tada", trigger="time == 5.0")
            self.add_child(XyTransientSystem("sub"))

        def transition(self):
            if self.tada.present:
                swap_system(self.sub, XzTransientSystem("sub"))
                self.sub.s = -0.25
                self.sub.z = 1.0

    class XyTransientSystem(System):
        """System having a transient.
        """
        def setup(self) -> None:
            self.add_inward("s", 1.0)
            self.add_inward("x", 0.0)

            self.add_transient("x", der="s")
            self.add_transient("y", der="2 * s")

    class XzTransientSystem(System):
        """System having a transient.
        """
        def setup(self) -> None:
            self.add_inward("s", 1.0)
            self.add_inward("x", 0.0)

            self.add_transient("x", der="s")
            self.add_transient("z", der="x")

    system = MultimodeSystem("system")
    system.add_driver(RungeKutta(time_interval=[0, 10.0], dt=1.0))

    assert hasattr(system, "sub.x")
    assert hasattr(system, "sub.y")
    assert not hasattr(system, "sub.z")

    te = 5.23  # event time
    system.tada.trigger = f"t == {te}"

    system.run_drivers()

    assert hasattr(system, "sub.x")
    assert not hasattr(system, "sub.y")
    assert hasattr(system, "sub.z")

    t = system.time
    xe = te  # x @ t=te
    exact_x = xe - 0.25 * (t - te)
    exact_z = 1.0 + xe * (t - te) - 0.125 * (t - te)**2
    assert system.sub.x == pytest.approx(exact_x, rel=1e-14)
    assert system.sub.z == pytest.approx(exact_z, rel=1e-14)
