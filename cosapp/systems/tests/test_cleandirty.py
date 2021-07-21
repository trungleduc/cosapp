import pytest
import os, sys
from unittest import TestCase, mock
import numpy as np

from cosapp.core.numerics.residues import Residue
from cosapp.ports.port import PortType, Port, Scope
from cosapp.systems import System
from cosapp.drivers import RunOnce, RunSingleCase, NonLinearSolver
from cosapp.tests.library.systems import MultiplySystem2, Multiply1


class UnitTestCleanDirty(TestCase):

    class TestPort(Port):
        def setup(self):
            self.add_variable('Pt', 101325., limits=(0., None))
            self.add_variable('W', 1., valid_range=(0., None))

    class SubSystem(System):
        def setup(self):
            self.add_input(UnitTestCleanDirty.TestPort, 'in_')
            self.add_output(UnitTestCleanDirty.TestPort, 'out')
            
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
            for name in self.out:
                self.out[name] = self.in_[name] * self.sloss
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

    class IterativeSystem(System):
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
        s.call_clean_run = mock.MagicMock(name='call_clean_run')
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
        assert clean_status(PortType.IN) == (False, False, True)

        s.run_once()
        assert clean_status(PortType.IN) == (True,) * 3

        s.sub1.useless_inward = 1.
        assert clean_status(PortType.IN) == (False, False, True)
        s.run_once()
        assert clean_status(PortType.IN) == (True,) * 3

        s.sub2.sloss = 0.96
        assert clean_status(PortType.IN) == (False, True, False)
        s.run_once()
        assert clean_status(PortType.IN) == (True,) * 3

        s.sub1.in_.W = 10.
        assert clean_status(PortType.IN) == (False, False, True)

        s.run_once()
        s2.add_child(UnitTestCleanDirty.SubSystem('ss1'))
        s.sub2.ss1.in_.W = 10.
        assert clean_status(PortType.IN) == (False, True, False)

        # Test System with subsystem using pulling method
        s = UnitTestCleanDirty.TopSystem('s')
        s.call_clean_run = mock.MagicMock(name='call_clean_run')
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

    def test_iterative_system(self):
        s = UnitTestCleanDirty.IterativeSystem('s')
        s.call_clean_run = mock.MagicMock(name='call_clean_run')
        s.open_loops()
        s1 = s.sub1
        s2 = s.sub2
        s1.call_clean_run = mock.MagicMock(name='call_clean_run')
        s2.call_clean_run = mock.MagicMock(name='call_clean_run')
        s.sub2_out_to_sub1_in_.call_clean_run = mock.MagicMock(name='call_clean_run')

        clean_status = lambda direction=None: (
            s.is_clean(direction),
            s1.is_clean(direction),
            s2.is_clean(direction),
            s.sub2_out_to_sub1_in_.is_clean(direction),
        )

        assert clean_status(PortType.IN) == (False, False, False, False)

        s.run_once()
        assert clean_status(PortType.IN) == (True, True, True, True)

        s.sub1_in_.W = 100.
        assert clean_status(PortType.IN) == (False, True, True, True)

        # Fake action run before running the iterative connector
        for c in filter(lambda c: "_to_sub2_out_to_sub1_in__" in c, s.connectors):
            s.connectors[c].transfer()
        assert clean_status(PortType.IN) == (False, True, True, False)

        residue = s.sub2_out_to_sub1_in_.residues['iterative_W']
        assert Residue.evaluate_residue(1., 0.95**2, residue.reference) == pytest.approx(residue.value, abs=1e-5)

        s.run_once()
        residue = s.sub2_out_to_sub1_in_.residues['iterative_W']
        assert s.sub2_out_to_sub1_in_.guess.W == 100
        assert Residue.evaluate_residue(100., 100 * 0.95**2, residue.reference) == pytest.approx(residue.value, abs=1e-5)

        s.sub1_in_.W = 10.
        s.run_once()
        residue = s.sub2_out_to_sub1_in_.residues['iterative_W']
        assert s.sub2_out_to_sub1_in_.guess.W == 10
        assert Residue.evaluate_residue(10., 10 * 0.95 * 0.95, residue.reference) == pytest.approx(residue.value, abs=1e-5)
        assert clean_status(PortType.IN) == (True,) * 4

        s.sub2.sloss = 0.9
        s.run_once()
        residue = s.sub2_out_to_sub1_in_.residues['iterative_W']
        assert s.sub2_out_to_sub1_in_.guess.W == 10
        assert Residue.evaluate_residue(10., 10 * 0.9 * 0.95, residue.reference) == pytest.approx(residue.value, abs=1e-5)
        assert clean_status(PortType.IN) == (True,) * 4
        assert clean_status(PortType.OUT) == (False,) * 4

        s1.run_once()
        assert clean_status(PortType.IN) == (True,) * 4
        assert clean_status(PortType.OUT) == (False, True, False, False)

        s2.run_once()
        assert clean_status(PortType.IN) == (True,) * 4
        assert clean_status(PortType.OUT) == (False, True, True, False)

        s.sub2_out_to_sub1_in_.run_once()
        assert clean_status(PortType.IN) == (True,) * 4
        assert clean_status(PortType.OUT) == (False, True, True, True)

        s.run_once()
        assert clean_status(PortType.IN) == (True,) * 4
        assert clean_status(PortType.OUT) == (True,) * 4


class IntegrationTestCleanDirty(TestCase):

    class TestMonitorInIn(System):
        def setup(self):
            self.add_child(Multiply1('mult1'))
            self.add_child(Multiply1('mult2'))

            self.connect(self.mult1.p_out, self.mult2.p_in)
            self.connect(self.mult2.inwards, self.mult1.inwards, 'K1')

            self.exec_order = ['mult1', 'mult2']

    def test_MultiplySystem2(self):
        s = MultiplySystem2('mult')
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
        assert clean_status(PortType.IN) == (False, True, False)
        assert clean_status(PortType.OUT) == (True,) * 3
        assert len(s.compute.mock_calls) == 1
        assert len(s1.compute.mock_calls) == 1
        assert len(s2.compute.mock_calls) == 1

        s.run_drivers()
        assert clean_status(PortType.IN) == (True,) * 3
        assert clean_status(PortType.OUT) == (False, True, False)
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
        assert clean_status(PortType.IN) == (False, True, False)
        assert clean_status(PortType.OUT) == (True,) * 3
        assert len(s.compute.mock_calls) == 1
        assert len(s1.compute.mock_calls) == 1
        assert len(s2.compute.mock_calls) == 1

        s.run_drivers()
        assert clean_status(PortType.IN) == (True,) * 3
        assert clean_status(PortType.OUT) == (False,) * 3
        assert len(s.compute.mock_calls) == 2
        assert len(s1.compute.mock_calls) == 2
        assert len(s2.compute.mock_calls) == 2
        assert len(s.call_clean_run.mock_calls) == 3

        s.run_drivers()
        assert clean_status(PortType.IN) == (True,) * 3
        assert clean_status(PortType.OUT) == (True,) * 3
        assert len(s.compute.mock_calls) == 2
        assert len(s1.compute.mock_calls) == 2
        assert len(s2.compute.mock_calls) == 2
        assert len(s.call_clean_run.mock_calls) == 4

        s.mult1_K1 = 80.
        assert clean_status(PortType.IN) == (False, True, True)
        assert clean_status(PortType.OUT) == (True,) * 3
        assert len(s.compute.mock_calls) == 2
        assert len(s1.compute.mock_calls) == 2
        assert len(s2.compute.mock_calls) == 2

        s.run_drivers()
        assert clean_status(PortType.IN) == (True,) * 3
        assert clean_status(PortType.OUT) == (False,) * 3
        assert len(s.compute.mock_calls) == 3
        assert len(s1.compute.mock_calls) == 3
        assert len(s2.compute.mock_calls) == 3
        assert len(s.call_clean_run.mock_calls) == 5
        assert s1.K1 == 80
        assert s2.K1 == 80

        s.run_drivers()
        assert clean_status(PortType.IN) == (True,) * 3
        assert clean_status(PortType.OUT) == (True,) * 3
        assert len(s.compute.mock_calls) == 3
        assert len(s1.compute.mock_calls) == 3
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

    def test_main(self):
        s = NumpyArrayCleanDirtyTest.IssueSystem2('s')
        solver = s.add_driver(NonLinearSolver('solver', tol=1e-6, max_iter=1))
        run = solver.add_child(RunSingleCase('run'))

        run.design.add_unknown(['sub.array_var[-1]']).add_equation('sub.array_res[-1] == 3')
        run.design.add_unknown(['sub.float_var']).add_equation('sub.float_res == 2')

        s.run_drivers()
        
        assert s.sub.array_var[-1] == pytest.approx(3, abs=1e-12)
        assert s.sub.float_var == pytest.approx(2, abs=1e-12)
