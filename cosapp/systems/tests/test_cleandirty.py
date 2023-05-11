import pytest
import os, sys
from unittest import TestCase, mock
import numpy as np

from cosapp.base import System, Port, Scope
from cosapp.ports.connectors import CopyConnector
from cosapp.ports.port import PortType
from cosapp.drivers import RunOnce, RunSingleCase, NonLinearSolver
from cosapp.tests.library.systems import MultiplySystem2, Multiply1
from cosapp.tests.library.ports import FluidPort


class UnitTestCleanDirty(TestCase):

    class Foo:
        pass

    class TestPort(Port):
        def setup(self):
            self.add_variable('Pt', 101325., limits=(0., None))
            self.add_variable('W', 1., valid_range=(0., None))

    class FooPort(Port):
        def setup(self):
            self.add_variable('a', 1.)
            self.add_variable('foo', UnitTestCleanDirty.Foo())

        Connector = CopyConnector

    class SubSystem(System):
        def setup(self):
            self.add_input(UnitTestCleanDirty.TestPort, 'in_')
            self.add_output(UnitTestCleanDirty.TestPort, 'out')
            self.add_input(UnitTestCleanDirty.FooPort, 'f_in')
            self.add_output(UnitTestCleanDirty.FooPort, 'f_out')
            
            self.add_inward('sloss',
                0.95, valid_range=(0.8, 1.), invalid_comment='not valid',
                limits=(0., 1.), out_of_limits_comment='hasta la vista baby',
                desc='get down',
                scope=Scope.PROTECTED,
            )
            self.add_inward('useless_inward',0.)
            self.add_outward('tmp',
                valid_range=(1, 2), invalid_comment='not valid tmp',
                limits=(0, 3), out_of_limits_comment="I'll be back",
                desc='banana',
                scope=Scope.PROTECTED,
            )
            self.add_outward('dummy', 1.)
            self.add_equation('dummy == 0')

        def compute(self):
            p_in = self.in_
            self.out.set_values(
                W = p_in.W,
                Pt = p_in.Pt * self.sloss,
            )
            self.dummy /= 100

    class TopSystem(System):

        tags = ['cosapp', 'tester']

        def setup(self):
            self.add_inward('top_k')
            self.add_outward('top_tmp')

            self.add_child(UnitTestCleanDirty.SubSystem('sub'), pulling={'in_': 'in_', 'out': 'out'})

    class SystemWithSubSystems(System):
        def setup(self):
            self.add_inward('top_k')

            self.add_child(UnitTestCleanDirty.SubSystem('sub1'))
            self.add_child(UnitTestCleanDirty.SubSystem('sub2'))

            self.connect(self.sub2.in_, self.sub1.out)

            self.exec_order = ['sub1', 'sub2']

    class CyclicSystem(System):
        def setup(self):
            self.add_inward('top_k')

            self.add_child(UnitTestCleanDirty.SubSystem('sub1'))
            self.add_child(UnitTestCleanDirty.SubSystem('sub2'))

            self.connect(self.sub2.in_, self.sub1.out)
            self.connect(self.sub1.in_, self.sub2.out)

            self.exec_order = ['sub1', 'sub2']

    @classmethod
    def setUpClass(cls):
        cls.cdir = os.path.dirname(os.path.realpath(__file__))
        # Add path to allow System to find the component
        sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'library/systems'))

    @classmethod
    def tearDownClass(cls):
        # Undo path modification
        sys.path.pop()

    def test_inputs_are_clean(self):
        s = UnitTestCleanDirty.SystemWithSubSystems('test')
        s1 = s.sub1
        s2 = s.sub2

        clean_status = lambda direction=None: (
            s.is_clean(direction),
            s1.is_clean(direction),
            s2.is_clean(direction),
        )

        assert clean_status(PortType.IN) == (False,) * 3
        assert clean_status(PortType.OUT) == (False,) * 3

        s.run_once()
        assert clean_status(PortType.IN) == (True,) * 3
        assert clean_status(PortType.OUT) == (False,) * 3

        s.run_once()
        assert clean_status(PortType.IN) == (True,) * 3
        assert clean_status(PortType.OUT) == (True,) * 3

        s.top_k = 10.
        assert clean_status(PortType.IN) == (False, True, True)

        s.run_once()
        assert clean_status(PortType.IN) == (True,) * 3

        s.sub1.sloss = 0.96
        assert clean_status(PortType.IN) == (True, False, True)

        s.run_once()
        assert clean_status(PortType.IN) == (True,) * 3

        s.sub1.useless_inward = 1.
        assert clean_status(PortType.IN) == (True, False, True)
        s.run_once()
        assert clean_status(PortType.IN) == (True,) * 3

        s.sub2.sloss = 0.96
        assert clean_status(PortType.IN) == (True, True, False)
        s.run_once()
        assert clean_status(PortType.IN) == (True,) * 3

        s.sub2.f_in.foo = UnitTestCleanDirty.Foo()
        assert clean_status(PortType.IN) == (True, True, False)
        s.run_once()
        assert clean_status(PortType.IN) == (True,) * 3

        s.sub1.in_.W = 10.
        assert clean_status(PortType.IN) == (True, False, True)

        s.run_once()
        s2.add_child(UnitTestCleanDirty.SubSystem('ss1'))
        s.sub2.ss1.in_.W = 10.
        assert clean_status(PortType.IN) == (True, True, True)
        assert not s.sub2.ss1.is_clean(PortType.IN)

        # Test System with subsystem using pulling method
        s = UnitTestCleanDirty.TopSystem('s')
        s.run_once()
        assert s.is_clean(PortType.IN)
        assert s.sub.is_clean(PortType.IN)
        assert not s.is_clean(PortType.OUT)
        assert not s.sub.is_clean(PortType.OUT)

        s.run_once()
        assert s.is_clean(PortType.IN)
        assert s.sub.is_clean(PortType.IN)
        assert s.is_clean(PortType.OUT)
        assert s.sub.is_clean(PortType.OUT)

        s.in_.Pt = 10.
        assert not s.is_clean(PortType.IN)
        assert s.sub.is_clean(PortType.IN)
        assert s.is_clean(PortType.OUT)
        assert s.sub.is_clean(PortType.OUT)

        s.run_once()
        assert s.is_clean(PortType.IN)
        assert s.sub.is_clean(PortType.IN)
        assert not s.is_clean(PortType.OUT)
        assert not s.sub.is_clean(PortType.OUT)

        s.run_once()
        assert s.is_clean(PortType.IN)
        assert s.sub.is_clean(PortType.IN)
        assert s.is_clean(PortType.OUT)
        assert s.sub.is_clean(PortType.OUT)

    def test_dirty_after_add_child_1(self):
        """Check that adding a child sets parent status to dirty.
        Test 1: child added to head system.
        """
        s = UnitTestCleanDirty.SystemWithSubSystems('s')
        s.compute = mock.MagicMock(name='compute_s')
        s.sub1.compute = mock.MagicMock(name='compute_sub1')
        s.sub2.compute = mock.MagicMock(name='compute_sub2')

        clean_status = lambda direction=None: (
            s.is_clean(direction),
            s.sub1.is_clean(direction),
            s.sub2.is_clean(direction),
        )

        assert clean_status(PortType.IN) == (False,) * 3
        assert clean_status(PortType.OUT) == (False,) * 3
        assert len(s.compute.mock_calls) == 0
        assert len(s.sub1.compute.mock_calls) == 0
        assert len(s.sub2.compute.mock_calls) == 0

        s.run_once()
        assert clean_status(PortType.IN) == (True,) * 3
        assert clean_status(PortType.OUT) == (False,) * 3
        assert len(s.compute.mock_calls) == 1
        assert len(s.sub1.compute.mock_calls) == 1
        assert len(s.sub2.compute.mock_calls) == 1

        s.run_once()
        assert clean_status(PortType.IN) == (True,) * 3
        assert clean_status(PortType.OUT) == (True,) * 3
        assert len(s.compute.mock_calls) == 1
        assert len(s.sub1.compute.mock_calls) == 1
        assert len(s.sub2.compute.mock_calls) == 1

        s.add_child(System('empty'))
        assert clean_status(PortType.IN) == (True,) * 3
        assert clean_status(PortType.OUT) == (True,) * 3

        s.run_once()
        assert clean_status(PortType.IN) == (True,) * 3
        assert clean_status(PortType.OUT) == (True,) * 3
        # Check that top system has been reexecuted,
        # since new child `empty` is dirty:
        assert len(s.compute.mock_calls) == 2
        assert len(s.sub1.compute.mock_calls) == 1
        assert len(s.sub2.compute.mock_calls) == 1

    def test_dirty_after_add_child_2(self):
        """Same as test 1, with child added to a sub-system."""
        s = UnitTestCleanDirty.SystemWithSubSystems('s')
        s.compute = mock.MagicMock(name='compute_s')
        s.sub1.compute = mock.MagicMock(name='compute_sub1')
        s.sub2.compute = mock.MagicMock(name='compute_sub2')

        clean_status = lambda direction=None: (
            s.is_clean(direction),
            s.sub1.is_clean(direction),
            s.sub2.is_clean(direction),
        )

        assert clean_status(PortType.IN) == (False,) * 3
        assert clean_status(PortType.OUT) == (False,) * 3
        assert len(s.compute.mock_calls) == 0
        assert len(s.sub1.compute.mock_calls) == 0
        assert len(s.sub2.compute.mock_calls) == 0

        s.run_once()
        assert clean_status(PortType.IN) == (True,) * 3
        assert clean_status(PortType.OUT) == (False,) * 3
        assert len(s.compute.mock_calls) == 1
        assert len(s.sub1.compute.mock_calls) == 1
        assert len(s.sub2.compute.mock_calls) == 1

        s.sub2.add_child(System('empty'))
        assert clean_status(PortType.IN) == (True, True, True)
        assert clean_status(PortType.OUT) == (False,) * 3
        assert len(s.compute.mock_calls) == 1
        assert len(s.sub1.compute.mock_calls) == 1
        assert len(s.sub2.compute.mock_calls) == 1

        s.run_once()
        assert clean_status(PortType.IN) == (True,) * 3
        assert clean_status(PortType.OUT) == (False, True, False)
        assert len(s.compute.mock_calls) == 2
        assert len(s.sub1.compute.mock_calls) == 1
        assert len(s.sub2.compute.mock_calls) == 2

    def test_iterative_system(self):
        s = UnitTestCleanDirty.CyclicSystem('s')
        s1 = s.sub1
        s2 = s.sub2
        s.open_loops()

        clean_status = lambda direction=None: (
            s.is_clean(direction),
            s1.is_clean(direction),
            s2.is_clean(direction),
        )

        assert clean_status(PortType.IN) == (False, False, False)
        assert clean_status(PortType.OUT) == (False, False, False)

        s.run_once()
        assert clean_status(PortType.IN) == (True, True, True)
        assert clean_status(PortType.OUT) == (False, False, False)

        for connector in s.all_connectors():
            connector.transfer()
        # s1 remains clean, since loops have been open
        assert clean_status(PortType.IN) == (True, True, False)

        s.run_once()
        assert clean_status(PortType.IN) == (True,) * 3
        assert clean_status(PortType.OUT) == (False, True, False)

        s.sub2.sloss = 0.9
        assert clean_status(PortType.IN) == (True, True, False)
        
        s.run_once()
        assert clean_status(PortType.IN) == (True,) * 3
        assert clean_status(PortType.OUT) == (False, True, False)

        # Execute `s.sub1` only; status of `s` and `s.sub2` is not updated
        s.sub1.run_once()
        assert clean_status(PortType.IN) == (True,) * 3
        assert clean_status(PortType.OUT) == (False, True, False)

        # Execute `s.sub2` only; status of `s` and `s.sub1` is not updated
        s.sub2.run_once()
        assert clean_status(PortType.IN) == (True,) * 3
        assert clean_status(PortType.OUT) == (False, True, True)

        s.run_once()
        assert clean_status(PortType.IN) == (True,) * 3
        assert clean_status(PortType.OUT) == (True,) * 3


class IntegrationTestCleanDirty(TestCase):

    class TestMonitorInIn(System):
        def setup(self):
            self.add_child(Multiply1('mult1'))
            self.add_child(Multiply1('mult2'))

            self.connect(self.mult1.p_out, self.mult2.p_in)
            self.connect(self.mult2.inwards, self.mult1.inwards, 'K1')

            self.exec_order = ['mult1', 'mult2']

    def test_MultiplySystem2(self):
        s = MultiplySystem2('s')
        s1 = s.mult1
        s2 = s.mult2

        clean_status = lambda direction=None: (
            s.is_clean(direction),
            s1.is_clean(direction),
            s2.is_clean(direction),
        )

        s.compute = mock.MagicMock(name='compute_mult')
        s.mult1.compute = mock.MagicMock(name='compute_mult1')
        s.mult2.compute = mock.MagicMock(name='compute_mult2')
        s.call_clean_run = mock.MagicMock(name='call_clean_run')
        s.add_driver(RunOnce('run'))

        assert clean_status(PortType.IN) == (False,) * 3
        assert clean_status(PortType.OUT) == (False,) * 3
        assert len(s.compute.mock_calls) == 0
        assert len(s1.compute.mock_calls) == 0
        assert len(s2.compute.mock_calls) == 0

        s.run_drivers()
        assert clean_status(PortType.IN) == (True,) * 3
        assert clean_status(PortType.OUT) == (False,) * 3
        assert len(s.compute.mock_calls)== 1
        assert len(s1.compute.mock_calls)== 1
        assert len(s2.compute.mock_calls)== 1
        assert len(s.call_clean_run.mock_calls)== 1

        s.run_drivers()
        assert clean_status(PortType.IN) == (True,) * 3
        assert clean_status(PortType.OUT) == (True,) * 3
        assert len(s.compute.mock_calls) == 1
        assert len(s1.compute.mock_calls) == 1
        assert len(s2.compute.mock_calls) == 1
        assert len(s.call_clean_run.mock_calls) == 2

        s2.K1 = 50.
        assert clean_status(PortType.IN) == (True, True, False)
        assert clean_status(PortType.OUT) == (True,) * 3
        assert len(s.compute.mock_calls) == 1
        assert len(s1.compute.mock_calls) == 1
        assert len(s2.compute.mock_calls) == 1

        s.run_drivers()
        assert clean_status(PortType.IN) == (True,) * 3
        assert clean_status(PortType.OUT) == (True, True, False)
        assert len(s.compute.mock_calls) == 2
        assert len(s1.compute.mock_calls) == 1
        assert len(s2.compute.mock_calls) == 2
        assert len(s.call_clean_run.mock_calls) == 3

        s.run_drivers()
        assert clean_status(PortType.IN) == (True,) * 3
        assert clean_status(PortType.OUT) == (True,) * 3
        assert len(s.compute.mock_calls) == 2
        assert len(s1.compute.mock_calls) == 1
        assert len(s2.compute.mock_calls) == 2
        assert len(s.call_clean_run.mock_calls) == 4

    def test_MonitorInIn(self):
        s = IntegrationTestCleanDirty.TestMonitorInIn('mult')
        s1 = s.mult1
        s2 = s.mult2

        clean_status = lambda direction=None: (
            s.is_clean(direction),
            s1.is_clean(direction),
            s2.is_clean(direction),
        )

        s.compute = mock.MagicMock(name='compute_mult')
        s.mult1.compute = mock.MagicMock(name='compute_mult1')
        s.mult2.compute = mock.MagicMock(name='compute_mult2')
        s.call_clean_run = mock.MagicMock(name='call_clean_run')

        s.add_driver(RunOnce('run'))

        assert clean_status(PortType.IN) == (False,) * 3
        assert clean_status(PortType.OUT) == (False,) * 3
        assert len(s.compute.mock_calls) == 0
        assert len(s1.compute.mock_calls) == 0
        assert len(s2.compute.mock_calls) == 0

        s.run_drivers()
        s.run_drivers()
        assert clean_status(PortType.IN) == (True,) * 3
        assert clean_status(PortType.OUT) == (True,) * 3
        assert len(s.compute.mock_calls) == 1
        assert len(s1.compute.mock_calls) == 1
        assert len(s2.compute.mock_calls) == 1
        assert len(s.call_clean_run.mock_calls) == 2

        s2.K1 = 50.
        assert clean_status(PortType.IN) == (True, True, False)
        assert clean_status(PortType.OUT) == (True,) * 3
        assert len(s.compute.mock_calls) == 1
        assert len(s1.compute.mock_calls) == 1
        assert len(s2.compute.mock_calls) == 1

        s.run_drivers()
        assert clean_status(PortType.IN) == (True,) * 3
        assert clean_status(PortType.OUT) == (True, True, False)
        assert len(s.compute.mock_calls) == 2
        assert len(s1.compute.mock_calls) == 1
        assert len(s2.compute.mock_calls) == 2
        assert len(s.call_clean_run.mock_calls) == 3

        s.run_drivers()
        assert clean_status(PortType.IN) == (True,) * 3
        assert clean_status(PortType.OUT) == (True,) * 3
        assert len(s.compute.mock_calls) == 2
        assert len(s1.compute.mock_calls) == 1
        assert len(s2.compute.mock_calls) == 2
        assert len(s.call_clean_run.mock_calls) == 4

        s.mult1_K1 = 80.
        assert clean_status(PortType.IN) == (False, True, True)
        assert clean_status(PortType.OUT) == (True,) * 3
        assert len(s.compute.mock_calls) == 2
        assert len(s1.compute.mock_calls) == 1
        assert len(s2.compute.mock_calls) == 2

        s.run_drivers()
        assert clean_status(PortType.IN) == (True,) * 3
        assert clean_status(PortType.OUT) == (False,) * 3
        assert len(s.compute.mock_calls) == 3
        assert len(s1.compute.mock_calls) == 2
        assert len(s2.compute.mock_calls) == 3
        assert len(s.call_clean_run.mock_calls) == 5
        assert s1.K1 == 80
        assert s2.K1 == 80

        s.run_drivers()
        assert clean_status(PortType.IN) == (True,) * 3
        assert clean_status(PortType.OUT) == (True,) * 3
        assert len(s.compute.mock_calls) == 3
        assert len(s1.compute.mock_calls) == 2
        assert len(s2.compute.mock_calls) == 3
        assert len(s.call_clean_run.mock_calls) == 6


class NumpyArrayCleanDirtyTest(TestCase):
    """Issue #151: Clean/Dirty not working with numpy arrays"""
    class IssueSystem1(System):
        def setup(self):
            self.add_inward('array_var', np.r_[0., 1.])
            self.add_inward('float_var', 1.)

            self.add_outward('array_res', np.r_[0., 0.])
            self.add_outward('float_res', 0.)

        def compute(self):
            self.array_res = np.array(self.array_var)
            self.float_res = self.float_var

    class IssueSystem2(System):
        def setup(self):
            self.add_child(NumpyArrayCleanDirtyTest.IssueSystem1('sub'))

    def test_main_design(self):
        s = NumpyArrayCleanDirtyTest.IssueSystem2('s')
        solver = s.add_driver(NonLinearSolver('solver', tol=1e-6, max_iter=1))
        case = solver.add_child(RunSingleCase('case'))

        case.design.add_unknown(['sub.array_var[-1]']).add_equation('sub.array_res[-1] == 3')
        case.design.add_unknown(['sub.float_var']).add_equation('sub.float_res == 2')

        s.run_drivers()
        
        assert s.sub.array_var[-1] == pytest.approx(3, abs=1e-12)
        assert s.sub.float_var == pytest.approx(2, abs=1e-12)

    def test_main_offdesign(self):
        s = NumpyArrayCleanDirtyTest.IssueSystem2('s')
        solver = s.add_driver(NonLinearSolver('solver', tol=1e-6, max_iter=1))
        case = solver.add_child(RunSingleCase('case'))
        case.add_unknown(['sub.array_var[-1]']).add_equation('sub.array_res[-1] == 3')
        case.add_unknown(['sub.float_var']).add_equation('sub.float_res == 2')

        s.run_drivers()
        
        assert s.sub.array_var[-1] == pytest.approx(3, abs=1e-12)
        assert s.sub.float_var == pytest.approx(2, abs=1e-12)


class BendPressureLoss(System):
    """Geometry-based pressure loss model in a bend"""
    def setup(self):
        self.add_inward('d', 0.1, unit='m', desc="Duct diameter")
        self.add_inward('angle', 90.0, unit='deg', desc="Bend angle")

        self.add_outward('coef', 1e2, desc="Pressure loss coefficient")

    def compute(self):
        self.coef = 90e2 * self.d / np.fmod(self.angle, 360)  # bogus


class BendDuct(System):
    
    def setup(self):
        self.add_child(BendPressureLoss('lossmodel'), pulling=['d', 'angle'])

        self.add_input(FluidPort, 'f_in', {'Pt': 1.01325e5, 'W': 1.0})
        self.add_output(FluidPort, 'f_out', {'Pt': 1.01325e5, 'W': 1.0})

        self.add_outward('dPt', 0.0, unit='Pa', desc="Pressure loss")

    def compute(self):
        f_in: FluidPort = self.f_in
        f_out: FluidPort = self.f_out
        self.dPt = dPt = self.lossmodel.coef * f_in.W**2
        f_out.set_from(f_in)
        f_out.Pt -= dPt


def test_cleandirty_const_geom():
    duct = BendDuct('duct')
    duct.d = 0.05
    duct.angle = 60.0
    duct.f_in.Pt = 1.01325e5

    # Mock lossmodel and set loss coefficient
    duct.lossmodel.compute = mock.MagicMock()
    duct.lossmodel.coef = 100.0

    solver = duct.add_driver(NonLinearSolver('solver'))
    solver.add_unknown('f_in.W').add_equation('dPt == 200')

    duct.run_drivers()

    # Check that loss model has been computed only once
    duct.lossmodel.compute.assert_called_once()

    assert duct.dPt == pytest.approx(200, rel=1e-12)
    assert duct.f_in.W == pytest.approx(np.sqrt(2))


@pytest.mark.parametrize("pulling", [
    None,
    'f_in',
    'f_out',
    ['f_in', 'f_out'],
])
def test_cleandirty_const_geom_subsystem(pulling):
    """Same as previous, embedded within a parent system"""
    top = System('top')
    duct = top.add_child(BendDuct('duct'), pulling=pulling)
    duct.d = 0.05
    duct.angle = 60.0
    duct.f_in.Pt = 1.01325e5

    # Mock lossmodel and set loss coefficient
    duct.lossmodel.compute = mock.MagicMock()
    duct.lossmodel.coef = 100.0

    solver = top.add_driver(NonLinearSolver('solver'))
    solver.add_unknown('duct.f_in.W').add_equation('duct.dPt == 200')

    top.run_drivers()

    # Check that loss model has been computed only once
    duct.lossmodel.compute.assert_called_once()

    assert duct.dPt == pytest.approx(200, rel=1e-12)
    assert duct.f_in.W == pytest.approx(np.sqrt(2))


def test_cleandirty_const_geom_add_tree():
    """Same as `test_cleandirty_const_geom`, with a
    bogus loop inside pressure loss model, causing the
    latter to be recomputed at each iteration (side effect).
    """
    class Bogus(System):
        def setup(self) -> None:
            self.add_inward('x', 0.1)
            self.add_inward('y', 0.23)

            self.add_outward('z', 0.0)

        def compute(self):
            self.z = self.x + self.y**2

    class BendPressureLossWithLoop(System):
        """Geometry-based pressure loss model in a bend,
        with loop between two `Bogus` sub-systems.
        """
        def setup(self):
            foo = self.add_child(Bogus('foo'))
            bar = self.add_child(Bogus('bar'))
            self.add_inward('d', 0.1, unit='m', desc="Duct diameter")
            self.add_inward('angle', 90.0, unit='deg', desc="Bend angle")

            self.add_outward('coef', 1e2, desc="Pressure loss coefficient")

            self.connect(self, foo, {'d': 'x'})
            self.connect(foo, bar, {'z': 'x', 'y': 'z'})  # loop

        def compute(self):
            self.coef = 90e2 * self.d / np.fmod(self.angle, 360)  # bogus

    class BendDuct(System):
        
        def setup(self):
            self.add_child(BendPressureLossWithLoop('lossmodel'), pulling=['d', 'angle'])

            self.add_input(FluidPort, 'f_in', {'Pt': 1.01325e5, 'W': 1.0})
            self.add_output(FluidPort, 'f_out', {'Pt': 1.01325e5, 'W': 1.0})

            self.add_outward('dPt', 0.0, unit='Pa', desc="Pressure loss")

        def compute(self):
            f_in: FluidPort = self.f_in
            f_out: FluidPort = self.f_out
            self.dPt = dPt = self.lossmodel.coef * f_in.W**2
            f_out.set_from(f_in)
            f_out.Pt -= dPt

    duct = BendDuct('duct')
    duct.d = 0.05
    duct.angle = 60.0
    duct.f_in.Pt = 1.01325e5

    # Mock lossmodel and set loss coefficient
    duct.lossmodel.compute = mock.MagicMock()
    duct.lossmodel.coef = 100.0

    solver = duct.add_driver(NonLinearSolver('solver'))
    solver.add_unknown('f_in.W').add_equation('dPt == 200')

    duct.run_drivers()

    # Check that loss model has been computed several times during resolution,
    # since at least one of its sub-systems has been continuously recomputed
    # (`foo` and `bar` have, actually).
    assert duct.lossmodel.compute.call_count > 1
    assert duct.lossmodel.compute.call_count >= solver.results.fres_calls

    assert duct.dPt == pytest.approx(200, rel=1e-12)
    assert duct.f_in.W == pytest.approx(np.sqrt(2))
