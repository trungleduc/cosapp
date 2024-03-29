import pytest

from unittest.mock import MagicMock

import numpy as np
import logging

from cosapp.systems import System
from cosapp.drivers import RunOnce
from cosapp.drivers import runonce
from cosapp.core.numerics.boundary import Boundary
from cosapp.recorders import DataFrameRecorder
from cosapp.tests.library.systems import Fan
from cosapp.utils.testing import DummySystemFactory, get_args, assert_keys, assert_all_type

# <codecell>

def test_RunOnce__initialize():
    d = RunOnce('runner')
    assert len(d.children) == 0
    assert len(d.initial_values) == 0
    assert len(d.solution) == 0
    assert isinstance(d.initial_values, dict)
    assert isinstance(d.solution, dict)


def test_RunOnce_get_init(ExtendedMultiply):
    # Test offdesign iteratives
    s = System('compute')
    s.add_child(ExtendedMultiply('mult'))
    d = RunOnce('compute')
    s.add_driver(d)
    
    assert len(d.get_init()) == 0

    def Dummy(name):
        return ExtendedMultiply(name, unknown=["K1"])

    s = System('compute')
    s.add_child(Dummy('mult'))
    d = RunOnce('compute')
    s.add_driver(d)
    assert np.array_equal(d.get_init(), [s.mult.K1])

    s.mult.K1 = 10
    d.set_init({'mult.K1': 33})
    assert_keys(d.initial_values, 'mult.K1')
    assert_all_type(d.initial_values, Boundary)
    assert s.mult.K1 == 33

    init_array = d.get_init()
    assert np.array_equal(init_array, [33])

    s.mult.K1 = 10
    assert_keys(d.initial_values, 'mult.K1')
    assert_all_type(d.initial_values, Boundary)
    init_array = d.get_init()
    assert np.array_equal(init_array, [33])
    d.solution['mult.K1'] = 11.
    init_array = d.get_init()
    assert np.array_equal(init_array, [11.])

    s.mult.K1 = 10
    init_array = d.get_init(force_init=True)
    assert np.array_equal(init_array, [33])

    d.set_init({'mult.K1': 32, 'mult.K2': 34})
    assert_keys(d.initial_values, 'mult.K1', 'mult.K2')
    assert_all_type(d.initial_values, Boundary)
    init_array = d.get_init()
    assert np.array_equal(init_array, [32])

    def Dummy(name):
        return ExtendedMultiply(name, unknown=["K1", "K2"])

    s = System('compute')
    s.add_child(Dummy('mult'))
    d = s.add_driver(RunOnce('compute'))
    s.mult.K1 = 10
    d.set_init({'mult.K1': 32, 'mult.K2': 34})
    
    d.setup_run()
    init_array = d.get_init()
    assert_keys(d.initial_values, 'mult.K1', 'mult.K2')
    assert_all_type(d.initial_values, Boundary)
    assert np.array_equal(init_array, [32, 34])

    d.solution['mult.K1'] = 22.
    d.solution['mult.K2'] = 42.
    init_array = d.get_init()
    assert np.array_equal(init_array, [22., 42.])


def test_RunOnce_set_init(ExtendedMultiply, hat_case):
    s = ExtendedMultiply('mult')
    d = RunOnce('compute')
    s.add_driver(d)

    d.set_init({'K1': 11.5})
    assert_keys(d.initial_values, 'K1')
    assert_all_type(d.initial_values, Boundary)
    value = d.initial_values['K1']
    assert value.default_value == 11.5
    assert value.mask is None

    s.run_drivers()
    assert s.K1 == 11.5

    with pytest.raises(TypeError):
        d.set_init({'inwards': 10.})

    d.set_init({'K1': 9.5})
    s.run_drivers()
    assert s.K1 == 9.5

    with pytest.raises(TypeError):
        d.set_init((s.K1, 10))

    with pytest.raises(AttributeError):
        d.set_init({'ImNotThere': 10.})

    d = RunOnce('compute')

    with pytest.raises(AttributeError, match="Driver '\w+' must be attached to a System"):
        d.set_init({'K1': 11.5})

    # Test vector variables
    s, case = hat_case(RunOnce)

    case.set_init({'in_.x': np.r_[-1., -2., -3.]})
    assert_keys(case.initial_values, 'in_.x')
    boundary = case.initial_values['in_.x']
    assert np.array_equal(boundary.value, [-1, -2, -3])
    assert np.array_equal(boundary.mask, [True, True, True])
    assert np.array_equal(s.in_.x, [-1, -2, -3])

    s, case = hat_case(RunOnce)
    case.set_init({'in_.x[0]': 42.})
    assert np.array_equal(s.in_.x, [42, -2, -3])

    s, case = hat_case(RunOnce)
    case.set_init({'in_.x[1:]': 24.})
    assert np.array_equal(s.in_.x, [42, 24, 24])

    s, case = hat_case(RunOnce)
    case.set_init({
        'in_.x[0]' : 22.,
        'in_.x[1:]': 33.,
        })
    assert np.array_equal(s.in_.x, [22, 33, 33])


def test_RunOnce_get_problem(ExtendedMultiply):
    s = ExtendedMultiply('mult')
    d = s.add_driver(RunOnce('runner'))
    assert d.get_problem().shape == (0, 0)

    def Dummy(name):
        return ExtendedMultiply(name, unknown=["K1"])

    s = Dummy('mult')
    d = s.add_driver(RunOnce('runner'))

    assert d.get_problem().shape == (1, 0)
    assert_keys(d.get_problem().unknowns, 'K1')

    f = Fan('fan')
    d = f.add_driver(RunOnce('runner'))
    assert d.get_problem().shape == (1, 2)
    assert_keys(d.get_problem().unknowns, 'gh')
    assert_keys(d.get_problem().residues, 'Wfan', 'PWfan')


def test_RunOnce_setup_run(caplog, ExtendedMultiply):
    caplog.clear()
    logging.disable(logging.NOTSET)  # enable all logging levels

    def warning_msg(name: str):
        return f"A mathematical problem on system {name!r} was detetected, but will not be solved by RunOnce driver"

    # Simple system
    Dummy = DummySystemFactory(
        "Dummy",
        inwards=get_args('x', 0.0),
        outwards=get_args('y', np.ones(2)),
    )
    s = Dummy('s')
    s.add_driver(RunOnce('runner'))

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=runonce.__name__):
        s.call_setup_run()
        assert len(caplog.text) == 0

    # System with intrinsic constraints
    DummyWithProblem = DummySystemFactory(
        "DummyWithProblem",
        inwards=get_args('x', 0.0),
        outwards=get_args('y', np.ones(2)),
        unknowns='x',
        equations='y == [0, 0]',
    )
    dummy = DummyWithProblem('dummy')
    dummy.add_driver(RunOnce('runner'))
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=runonce.__name__):
        dummy.call_setup_run()
        assert warning_msg('dummy') in caplog.text
        assert 'Problem:' not in caplog.text

    # Same with lower-level logger
    caplog.clear()
    debug_msg = '\n'.join([
        'Problem:',
        'Unknowns [1]',
        '  x = 0.0',
        'Equations [2]',
        '  y == [0, 0]',
    ])
    with caplog.at_level(logging.DEBUG, logger=runonce.__name__):
        dummy.call_setup_run()
        assert warning_msg('dummy') in caplog.text
        assert debug_msg in caplog.text

    # System with loops
    A = DummySystemFactory("A", inwards=get_args('x', 0.0), outwards=get_args('y', 0.0))
    B = DummySystemFactory("B", inwards=get_args('u', 0.0), outwards=get_args('v', 0.0))
    class Cyclic(System):
        def setup(self) -> None:
            a = self.add_child(A('a'))
            b = self.add_child(B('b'))
            self.connect(a, b, {'y': 'u', 'x': 'v'})
    
    cyclic = Cyclic('cyclic')
    cyclic.add_driver(RunOnce('runner'))
    caplog.clear()
    debug_msg = '\n'.join([
        'Problem:',
        'Unknowns [1]',
        '  a.x = 0.0',
        'Equations [1]',
        '  a.x == b.v (loop)',
    ])
    with caplog.at_level(logging.DEBUG, logger=runonce.__name__):
        cyclic.call_setup_run()
        assert warning_msg('cyclic') in caplog.text
        assert debug_msg in caplog.text

def test_RunOnce__precompute(ExtendedMultiply):
    s = ExtendedMultiply('mult')
    d = s.add_driver(RunOnce('runner'))
    d.solution['dummy'] = 1.
    d._precompute()
    assert len(d.solution) == 0


def test_RunOnce_compute(ExtendedMultiply):
    def Dummy(name):
        return ExtendedMultiply(name, unknown=["p_in.x"])

    s = Dummy('mult')

    d = s.add_driver(RunOnce('runner'))
    assert len(d.solution) == 0
    s.compute = MagicMock()
    s.run_drivers()
    assert_keys(d.solution, 'p_in.x')
    assert d.solution['p_in.x'] == 1
    s.compute.assert_called_once()


def test_RunOnce__postcompute(ExtendedMultiply):
    def Dummy(name):
        return ExtendedMultiply(name, unknown=["p_in.x"])
            
    s = Dummy('mult')

    d = s.add_driver(RunOnce('runner'))
    assert len(d.solution) == 0
    d._postcompute()
    assert_keys(d.solution, 'p_in.x')


def test_RunOnce_recorder():
    """Exposes bug reported in https://gitlab.com/cosapp/cosapp/-/issues/84"""
    class Bogus(System):
        def setup(self):
            self.add_inward("a")

    s = Bogus("s")
    run = s.add_driver(RunOnce('run'))
    rec = run.add_recorder(DataFrameRecorder(hold=True))

    s.a = 1.
    s.run_drivers()
    data = rec.export_data()
    assert len(data) == 1
    assert data['a'][0] == 1.

    s.a = 10.
    s.run_drivers()
    data = rec.export_data()
    assert len(data) == 2
    assert data['a'][0] == 1.
    assert data['a'][1] == 10.

    rec = run.add_recorder(DataFrameRecorder(hold=False))

    s.a = 20.
    s.run_drivers()
    data = rec.export_data()
    assert len(data) == 1
    assert data['a'][0] == 20.
