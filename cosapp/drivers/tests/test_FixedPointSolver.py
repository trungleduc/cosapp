import pytest
import numpy as np

from cosapp.drivers import FixedPointSolver
from cosapp.systems import System
from cosapp.recorders import DataFrameRecorder
from cosapp.utils.testing import pickle_roundtrip, are_same


class SystemA(System):
    def setup(self):
        self.add_inward('x', 1.0)
        self.add_outward('y', 0.0)

    def compute(self):
        self.y = np.cos(self.x)


class SystemB(System):
    def setup(self):
        self.add_inward('u', 1.0)
        self.add_outward('v', 0.0)

    def compute(self):
        self.v = 0.5 * self.u**2


class FixedPointSystem(System):
    """Cyclic assembly of two systems
    `a` and `b` exchanging inwards.

    `a.x` is the solution of `x = cos^2(x) / 2`
    (x ~ 0.41771479265994116, approximately)
    """

    def setup(self):
        a = self.add_child(SystemA('a'))
        b = self.add_child(SystemB('b'))
        self.connect(a, b, {'x': 'v', 'y': 'u'})
        self.exec_order = ('a', 'b')


def test_FixedPointSolver():
    s: System = FixedPointSystem('s')

    solver = s.add_driver(FixedPointSolver('solver', tol=1e-9))

    s.a.x = 1.0
    s.run_drivers()

    a, b = s.a, s.b
    assert a.x == pytest.approx(0.417714792)
    assert a.x == pytest.approx(0.5 * np.cos(a.x) ** 2)
    assert a.x == pytest.approx(b.v, abs=1e-9)
    assert solver.results.success
    assert solver.results.n_iter > 0


def test_FixedPointSolver_history():
    s: System = FixedPointSystem('s')

    solver = s.add_driver(FixedPointSolver('solver', tol=1e-9, history=True))
    solver.add_recorder(DataFrameRecorder(includes=['*', 'a.x - b.v', 'a.y - b.u']))

    s.a.x = 1.0
    s.run_drivers()

    a, b = s.a, s.b
    assert a.x == pytest.approx(0.417714792)
    assert a.x == pytest.approx(0.5 * np.cos(a.x) ** 2)
    assert a.x == pytest.approx(b.v, abs=1e-9)

    data = solver.recorder.export_data()
    # print("", data, sep="\n")
    assert solver.results.success
    assert solver.results.n_iter > 0
    assert len(data) == solver.results.n_iter + 1
    # assert data['a.x'].values[-1] == a.x
    assert data['a.y'].values[-1] == a.y
    assert data['b.v'].values[-1] == b.v
    assert data['a.x - b.v'].values[-1] == solver.results.r[-1]
    assert np.linalg.norm(data['a.y - b.u'], np.inf) == 0.0


@pytest.mark.parametrize("force_init", [True, False])
def test_FixedPointSolver_force_init(force_init):
    s: System = FixedPointSystem('s')

    solver = s.add_driver(FixedPointSolver('solver', tol=1e-9, force_init=force_init))
    solver.set_init({'a.x': 1.0})

    # First solver execution
    s.a.x = 1.0
    s.run_drivers()

    a, b = s.a, s.b
    assert a.x == pytest.approx(0.417714792)
    assert a.x == pytest.approx(0.5 * np.cos(a.x) ** 2)
    assert a.x == pytest.approx(b.v, abs=1e-9)
    assert solver.results.success
    assert solver.results.n_iter > 0

    # Second solver execution
    s.run_drivers()

    if force_init:
        # Solver had to iterate since initial value of a.x was enforced
        assert solver.results.n_iter > 0
    else:
        # No iterations expected since a.x is the computed solution
        assert solver.results.n_iter == 0


@pytest.mark.parametrize(
    "factor, n_iter",
    [
        (1.0, (20, 22)),
        (0.8, (8, 10)),
        (0.5, (16, 20)),
    ],
)
def test_FixedPointSolver_relaxation(factor, n_iter):
    s: System = FixedPointSystem('s')

    solver = s.add_driver(FixedPointSolver('solver', tol=1e-9, factor=factor))

    s.a.x = 1.0
    s.run_drivers()

    a, b = s.a, s.b
    assert a.x == pytest.approx(0.417714792)
    assert a.x == pytest.approx(0.5 * np.cos(a.x) ** 2)
    assert a.x == pytest.approx(b.v, abs=1e-9)

    assert solver.results.success
    assert solver.results.n_iter > 0
    if isinstance(n_iter, int):
        assert solver.results.n_iter == n_iter
    else:
        lower, upper = n_iter
        assert solver.results.n_iter >= lower
        assert solver.results.n_iter <= upper


class TestFixedPointSolverPickling:

    @pytest.fixture
    def system(self):
        s: System = FixedPointSystem('s')
        s.add_driver(FixedPointSolver('solver', tol=1e-9))
        return s

    def test_standalone(self):
        """Test pickle roundtrip."""

        solver = FixedPointSolver('solver', tol=1e-9)

        solver_copy = pickle_roundtrip(solver)
        assert solver_copy.options['tol'] == 1e-9

    def test_set_init(self, system):

        solver = system.drivers["solver"]
        solver.set_init({"a.x": 1.0})
        assert len(solver.initial_values) == 1

        new_s = pickle_roundtrip(system)
        assert are_same(new_s, system)

        new_solver = new_s.drivers["solver"]
        assert new_solver is not solver
        assert new_solver.owner is new_s
        assert new_solver.name == "solver"
        assert new_solver.initial_values["a.x"].context is new_s

        system.run_drivers()
        new_s = pickle_roundtrip(system)

        assert system.to_json() == new_s.to_json()
        assert are_same(new_s, system)

        new_solver = new_s.drivers["solver"]
        assert new_solver.results.success
        assert new_solver.results.n_iter == 21
        assert len(new_solver._loop_connectors) == 1

        connector = new_solver._loop_connectors[0]
        assert connector.source is new_s.b.outwards
        assert connector.sink is new_s.a.inwards
