import pytest
import numpy as np

from cosapp.drivers import FixedPointSolver
from cosapp.systems import System
from cosapp.recorders import DataFrameRecorder


@pytest.fixture
def FixedPointSystem():
    """Factory creating a cyclic assembly of
    two systems `a` and `b` exchanging inwards.

    `a.x` is the solution of `x = cos^2(x) / 2`
    (x ~ 0.41771479265994116, approximately)
    """
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

    def factory(name: str):
        top = System(name)
        a = top.add_child(SystemA('a'))
        b = top.add_child(SystemB('b'))
        top.connect(a, b, {'x': 'v', 'y': 'u'})
        top.exec_order = ('a', 'b')
        return top

    return factory


def test_FixedPointSolver(FixedPointSystem):
    s: System = FixedPointSystem('s')

    solver = s.add_driver(FixedPointSolver('solver', tol=1e-9))

    s.a.x = 1.0
    s.run_drivers()

    a, b = s.a, s.b
    assert a.x == pytest.approx(0.417714792)
    assert a.x == pytest.approx(0.5 * np.cos(a.x)**2)
    assert a.x == pytest.approx(b.v, abs=1e-9)
    assert solver.results.success
    assert solver.results.n_iter > 0


def test_FixedPointSolver_history(FixedPointSystem):
    s: System = FixedPointSystem('s')

    solver = s.add_driver(FixedPointSolver('solver', tol=1e-9, history=True))
    solver.add_recorder(DataFrameRecorder(includes=['*', 'a.x - b.v', 'a.y - b.u']))

    s.a.x = 1.0
    s.run_drivers()

    a, b = s.a, s.b
    assert a.x == pytest.approx(0.417714792)
    assert a.x == pytest.approx(0.5 * np.cos(a.x)**2)
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
def test_FixedPointSolver_force_init(FixedPointSystem, force_init):
    s: System = FixedPointSystem('s')

    solver = s.add_driver(FixedPointSolver('solver', tol=1e-9, force_init=force_init))
    solver.set_init({'a.x': 1.0})

    # First solver execution
    s.a.x = 1.0
    s.run_drivers()

    a, b = s.a, s.b
    assert a.x == pytest.approx(0.417714792)
    assert a.x == pytest.approx(0.5 * np.cos(a.x)**2)
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


@pytest.mark.parametrize("factor, n_iter", [
    (1.0, (20, 22)),
    (0.8, (8, 10)),
    (0.5, (16, 20)),
])
def test_FixedPointSolver_relaxation(FixedPointSystem, factor, n_iter):
    s: System = FixedPointSystem('s')

    solver = s.add_driver(FixedPointSolver('solver', tol=1e-9, factor=factor))

    s.a.x = 1.0
    s.run_drivers()

    a, b = s.a, s.b
    assert a.x == pytest.approx(0.417714792)
    assert a.x == pytest.approx(0.5 * np.cos(a.x)**2)
    assert a.x == pytest.approx(b.v, abs=1e-9)

    assert solver.results.success
    assert solver.results.n_iter > 0
    if isinstance(n_iter, int):
        assert solver.results.n_iter == n_iter
    else:
        lower, upper = n_iter
        assert solver.results.n_iter >= lower
        assert solver.results.n_iter <= upper
