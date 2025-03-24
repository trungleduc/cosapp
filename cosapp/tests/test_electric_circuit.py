"""Regression test from tutorials"""
import pytest
import numpy as np

from cosapp.tests.library.systems import Resistor, Capacitor, Node, Source, Ground
from cosapp.systems import System
from cosapp.drivers import NonLinearSolver, RunSingleCase, CrankNicolson
from cosapp.recorders import DataFrameRecorder


class Circuit(System):
    """Simple resistor circuit

    --->--+--[R2]--+
          |        |
        [R1]     [R3]
          |        |
          +--------+---/ Ground
    """
    def setup(self):
        ground = self.add_child(Ground('ground', V=0.0))
        R1 = self.add_child(Resistor('R1', R=1000))
        R2 = self.add_child(Resistor('R2', R=500))
        R3 = self.add_child(Resistor('R3', R=250))
        
        Node.make(self, 'n1',
            incoming = [],
            outgoing = [R1, R2],
            pulling = {'I_in0': 'source'},
        )
        Node.make(self, 'n2',
            incoming = [R2],
            outgoing = [R3],
        )
        self.connect(ground.V_out, R1.V_out)
        self.connect(ground.V_out, R3.V_out)
        
        self.exec_order = ('ground', 'n1', 'n2', 'R1', 'R2', 'R3')

        # design methods
        self.add_inward('V1_design', 0.0)
        self.add_inward('V2_design', 0.0)
        self.add_design_method('R1').add_unknown('R1.R').add_equation('n1.V == V1_design')
        self.add_design_method('R2').add_unknown('R2.R').add_equation('n2.V == V2_design')


class RcCircuit(System):
    """Simple serial RC circuit"""
    def setup(self):
        r = self.add_child(Resistor('res'), pulling=['V_in', 'R'])
        c = self.add_child(Capacitor('capa'), pulling=['V_out', 'C', 'I'])

        Node.make(self, 'node', incoming=[r], outgoing=[c])


class RcSolution:
    """Solution of a serial RC circuit subjected to a unit voltage difference"""
    def __init__(self, R, C):
        self.R = R
        self.RC = R * C

    def V(self, t: float):
        return 1.0 - np.exp(-t / self.RC)

    def I(self, t: float):
        return (1.0 - self.V(t)) / self.R


class LegacyCircuit(System):
    """Legacy circuit model, as originally given in tutorials"""
    def setup(self):
        n1 = self.add_child(Node('n1', n_in=1, n_out=2), pulling={'I_in0': 'I_in'})
        n2 = self.add_child(Node('n2'))
        
        R1 = self.add_child(Resistor('R1', R=1000), pulling={'V_out': 'Vg'})
        R2 = self.add_child(Resistor('R2', R=500))
        R3 = self.add_child(Resistor('R3', R=250), pulling={'V_out': 'Vg'})
        
        self.connect(R1.V_in, n1.inwards, 'V')
        self.connect(R2.V_in, n1.inwards, 'V')
        self.connect(R1.I, n1.I_out0)
        self.connect(R2.I, n1.I_out1)
        
        self.connect(R2.V_out, n2.inwards, 'V')
        self.connect(R3.V_in, n2.inwards, 'V')
        self.connect(R2.I, n2.I_in0)
        self.connect(R3.I, n2.I_out0)
        
        self.exec_order = ('n1', 'n2', 'R1', 'R2', 'R3')

        # design methods
        self.add_inward('V1_design', 0.0)
        self.add_inward('V2_design', 0.0)
        self.add_design_method('R1').add_unknown('R1.R').add_equation('n1.V == V1_design')
        self.add_design_method('R2').add_unknown('R2.R').add_equation('n2.V == V2_design')


@pytest.fixture
def model():
    """Legacy model, as originally given in tutorials

     +----+--[R2]--+
     |    |        |
    (S)  [R1]     [R3]
     |    |        |
     +----+--------+---/ Ground
    """
    model = System('model')
    model.add_child(Source('source', I=0.1))
    model.add_child(Ground('ground', V=0.0))
    model.add_child(LegacyCircuit('circuit'))

    model.connect(model.source.I_out, model.circuit.I_in)
    model.connect(model.ground.V_out, model.circuit.Vg)

    model.circuit.R1.R = 1e3
    model.circuit.R2.R = 0.5e3
    model.circuit.R3.R = 0.25e3

    return model


@pytest.fixture
def circuit():
    circuit = Circuit('circuit')
    circuit.R1.R = 1e3
    circuit.R2.R = 0.5e3
    circuit.R3.R = 0.25e3
    circuit.source.I = 0.1
    return circuit


@pytest.fixture
def rc():
    rc = RcCircuit('rc')
    rc.C = 2e-3
    rc.R = 100.
    rc.V_in.V = 1.0
    rc.V_out.V = 0.0
    return rc


def test_LegacyCircuit_solve(model):
    model.add_driver(NonLinearSolver('solver'))
    model.run_drivers()

    assert model.source.I == 0.1
    assert model.circuit.R1.R == 1000
    assert model.circuit.R2.R == 500
    assert model.circuit.R3.R == 250
    assert model.circuit.R1.I.I == pytest.approx(0.04285714286)
    assert model.circuit.R2.I.I == pytest.approx(0.05714285714)
    assert model.circuit.R3.I.I == pytest.approx(0.05714285715)
    assert model.circuit.n1.V == pytest.approx(42.85714286)
    assert model.circuit.n2.V == pytest.approx(14.28571429)


def test_LegacyCircuit_design_multipoint(model):
    solver = model.add_driver(NonLinearSolver('solver', tol=1e-9))
    model.run_drivers()  # balance system

    # Add driver to set boundary conditions on point 1
    point1 = solver.add_child(RunSingleCase('point1'))
    point1.design.extend(model.circuit.design_methods['R2'])

    model.circuit.V2_design = 8.
    point1.set_values({
        'source.I': 0.08, 
        'ground.V': 0,
    })

    # Same as previous for a second point
    point2 = solver.add_child(RunSingleCase('point2')) 
    point2.design.extend(model.circuit.design_methods['R1'])

    model.circuit.V1_design = 50.
    point2.set_values({
        'source.I': 0.15,
        'ground.V': 0,
    })

    solver.add_recorder(
        DataFrameRecorder(includes=['*n?.V', '*R', 'source.I'], excludes='*R3*')
    )

    model.run_drivers()

    data = solver.recorder.export_data()
    assert data.at[0, 'circuit.n2.V'] == pytest.approx(8)
    assert data.at[1, 'circuit.n1.V'] == pytest.approx(50)
    assert model.circuit.R1.R == pytest.approx(5000 / 9)
    assert model.circuit.R2.R == pytest.approx(5250 / 9)


def test_Circuit_solve(circuit):
    circuit.add_driver(NonLinearSolver('solver'))
    circuit.run_drivers()

    assert circuit.source.I == 0.1
    assert circuit.R1.R == 1000
    assert circuit.R2.R == 500
    assert circuit.R3.R == 250
    assert circuit.R1.I.I == pytest.approx(0.04285714286)
    assert circuit.R2.I.I == pytest.approx(0.05714285714)
    assert circuit.R3.I.I == pytest.approx(0.05714285715)
    assert circuit.n1.V == pytest.approx(42.85714286)
    assert circuit.n2.V == pytest.approx(14.28571429)


def test_Circuit_design_multipoint(circuit):
    solver = circuit.add_driver(NonLinearSolver('solver', tol=1e-9))
    circuit.run_drivers()  # balance system

    # Add driver to set boundary conditions on point 1
    point1 = solver.add_child(RunSingleCase('point1'))
    point1.design.extend(circuit.design_methods['R2'])

    circuit.V2_design = 8.
    point1.set_values({
        'source.I': 0.08, 
        'ground.V': 0,
    })

    # Same as previous for a second point
    point2 = solver.add_child(RunSingleCase('point2')) 
    point2.design.extend(circuit.design_methods['R1'])

    circuit.V1_design = 50.
    point2.set_values({
        'source.I': 0.15,
        'ground.V': 0,
    })

    solver.add_recorder(
        DataFrameRecorder(includes=['n?.V', '*R', 'source.I'], excludes='*R3*')
    )

    circuit.run_drivers()

    data = solver.recorder.export_data()
    assert data.at[0, 'n2.V'] == pytest.approx(8)
    assert data.at[1, 'n1.V'] == pytest.approx(50)
    assert circuit.R1.R == pytest.approx(5000 / 9)
    assert circuit.R2.R == pytest.approx(5250 / 9)


def test_RcCircuit_solve_initial(rc):
    """Test initial equilibrium of RC circuit"""
    solver = rc.add_driver(NonLinearSolver('solver'))
    rc.run_drivers()

    assert solver.problem.shape == (2, 2)

    assert rc.I.I == pytest.approx(0.01, rel=1e-12)
    assert rc.node.V == pytest.approx(0.0, abs=1e-12)
    assert rc.capa.U == pytest.approx(0.0, abs=1e-12)
    assert rc.capa.dUdt == pytest.approx(5.0, rel=1e-12)
    assert rc.capa.deltaV == pytest.approx(0.0, abs=1e-12)


def test_RcCircuit_solve_transient(rc):
    """Test transient resolution of RC circuit"""
    driver = rc.add_driver(CrankNicolson('solver'))
    driver.time_interval = (0, 2)
    driver.dt = 1e-2
    driver.add_recorder(DataFrameRecorder())

    rc.run_drivers()

    assert driver._intrinsic_problem.shape == (2, 2)
    assert driver.problem.shape == (3, 2)

    data = driver.recorder.export_data()

    # Test initial values
    # Note: same values as in `test_RcCircuit_solve_initial`
    assert data["I.I"][0] == pytest.approx(0.01, rel=1e-12)
    assert data["node.V"][0] == pytest.approx(0.0, abs=1e-12)
    assert data["capa.U"][0] == pytest.approx(0.0, abs=1e-12)
    assert data["capa.dUdt"][0] == pytest.approx(5.0, rel=1e-12)
    assert data["capa.deltaV"][0] == pytest.approx(0.0, abs=1e-12)

    exact = RcSolution(rc.R, rc.C)

    time = np.asarray(data["time"])
    Vnum = np.asarray(data["node.V"])
    Inum = np.asarray(data["I.I"])

    assert Vnum == pytest.approx(exact.V(time), rel=1e-3)
    assert Inum == pytest.approx(exact.I(time), rel=5e-3)


def test_RcCircuit_stop(rc):
    """RC circuit resolution with a stop condition"""
    driver = rc.add_driver(CrankNicolson("solver"))
    driver.time_interval = (0, 2)
    driver.dt = 1e-2

    driver.set_scenario(
        stop = "capa.deltaV == res.deltaV",
    )

    rc.run_drivers()

    records = driver.recorded_events
    t_cross = np.log(2) * rc.R * rc.C

    assert len(records) == 1
    assert records[0].time == pytest.approx(t_cross, rel=1e-3)
    assert records[0].events == [driver.scenario.stop]

    assert rc.res.deltaV == pytest.approx(0.5, abs=1e-12)
    assert rc.capa.deltaV == pytest.approx(0.5, abs=1e-12)
    assert rc.capa.U == pytest.approx(0.5, abs=1e-12)
    assert rc.time == pytest.approx(t_cross, rel=1e-3)


def test_RcCircuit_transition():
    """RC circuit with a dummy event"""
    class MultimodeRc(RcCircuit):
        def setup(self):
            super().setup()
            self.add_event("tada")
    
    rc = MultimodeRc("rc")
    rc.tada.trigger = "res.deltaV == capa.deltaV"
    rc.C = 2e-3
    rc.R = 100.
    rc.V_in.V = 1.0
    rc.V_out.V = 0.0

    driver = rc.add_driver(CrankNicolson("solver"))
    driver.add_recorder(DataFrameRecorder(), period=0.1)
    driver.time_interval = (0, 1)
    driver.dt = 1e-2

    rc.run_drivers()

    records = driver.recorded_events
    t_cross = np.log(2) * rc.R * rc.C

    assert len(records) == 1
    assert records[0].time == pytest.approx(t_cross, rel=1e-3)
    assert records[0].events == [rc.tada]

    exact = RcSolution(rc.R, rc.C)

    data = driver.recorder.export_data()
    time = np.asarray(data["time"])
    Vnum = np.asarray(data["node.V"])
    Inum = np.asarray(data["I.I"])

    assert len(data) == 13
    assert Vnum == pytest.approx(exact.V(time), rel=1e-3)
    assert Inum == pytest.approx(exact.I(time), rel=5e-3)
