import pytest

from unittest.mock import MagicMock

import numpy as np
import logging

from cosapp.systems import System
from cosapp.drivers import RunOnce
from cosapp.drivers import runonce
from cosapp.core.numerics.boundary import Boundary
from cosapp.tests.library.ports import XPort
from cosapp.tests.library.systems import Fan, Mixer21, Strait1dLine
from cosapp.utils.testing import assert_keys, assert_all_type

# <codecell>

def test_RunOnce__initialize():
    d = RunOnce('runner')
    assert len(d.children) == 0
    assert len(d.initial_values) == 0
    assert len(d.solution) == 0
    assert isinstance(d.initial_values, dict)
    assert isinstance(d.solution, dict)


def test_RunOnce_get_init(DummyFactory):
    # Test offdesign iteratives
    s = System('compute')
    s.add_child(DummyFactory('mult'))
    d = RunOnce('compute')
    s.add_driver(d)
    
    assert len(d.get_init()) == 0

    def Dummy(name):
        return DummyFactory(name, unknown=["K1"])

    s = System('compute')
    s.add_child(Dummy('mult'))
    d = RunOnce('compute')
    s.add_driver(d)
    assert np.array_equal(d.get_init(), [s.mult.inwards.K1])

    s.mult.K1 = 10
    d.set_init({'mult.K1': 33})
    assert_keys(d.initial_values, 'mult.inwards.K1')
    assert_all_type(d.initial_values, Boundary)
    assert s.mult.K1 == 33

    init_array = d.get_init()
    assert np.array_equal(init_array, [33])

    s.mult.K1 = 10
    assert_keys(d.initial_values, 'mult.inwards.K1')
    assert_all_type(d.initial_values, Boundary)
    init_array = d.get_init()
    assert np.array_equal(init_array, [33])
    d.solution['mult.inwards.K1'] = 11.
    init_array = d.get_init()
    assert np.array_equal(init_array, [11.])

    s.mult.K1 = 10
    init_array = d.get_init(force_init=True)
    assert np.array_equal(init_array, [33])

    d.set_init({'mult.K1': 32, 'mult.K2': 34})
    assert_keys(d.initial_values, 'mult.inwards.K1', 'mult.inwards.K2')
    assert_all_type(d.initial_values, Boundary)
    init_array = d.get_init()
    assert np.array_equal(init_array, [32])

    def Dummy(name):
        return DummyFactory(name, unknown=["K1", "K2"])

    s = System('compute')
    s.add_child(Dummy('mult'))
    d = s.add_driver(RunOnce('compute'))
    s.mult.K1 = 10
    d.set_init({'mult.K1': 32, 'mult.K2': 34})
    
    d.setup_run()
    init_array = d.get_init()
    assert_keys(d.initial_values, 'mult.inwards.K1', 'mult.inwards.K2')
    assert_all_type(d.initial_values, Boundary)
    assert np.array_equal(init_array, [32, 34])

    d.solution['mult.inwards.K1'] = 22.
    d.solution['mult.inwards.K2'] = 42.
    init_array = d.get_init()
    assert np.array_equal(init_array, [22., 42.])


def test_RunOnce_set_init(DummyFactory, hat_case):
    s = DummyFactory('mult')
    d = RunOnce('compute')
    s.add_driver(d)

    d.set_init({'K1': 11.5})
    assert_keys(d.initial_values, 'inwards.K1')
    assert_all_type(d.initial_values, Boundary)
    value = d.initial_values['inwards.K1']
    assert value.default_value == 11.5
    assert value.mask is None

    s.run_drivers()
    assert s.inwards.K1 == 11.5

    with pytest.raises(TypeError):
        d.set_init({'inwards': 10.})

    d.set_init({'K1': 9.5})
    s.run_drivers()
    assert s.inwards.K1 == 9.5

    with pytest.raises(TypeError):
        d.set_init((s.inwards.K1, 10))

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


def test_RunOnce_get_problem(DummyFactory):
    s = DummyFactory('mult')
    d = s.add_driver(RunOnce('runner'))
    assert d.get_problem().shape == (0, 0)

    def Dummy(name):
        return DummyFactory(name, unknown=["K1"])

    s = Dummy('mult')
    d = s.add_driver(RunOnce('runner'))

    assert d.get_problem().shape == (1, 0)
    assert_keys(d.get_problem().unknowns, 'inwards.K1')

    f = Fan('fan')
    d = f.add_driver(RunOnce('runner'))
    assert d.get_problem().shape == (1, 2)
    assert_keys(d.get_problem().unknowns, 'inwards.gh')
    assert_keys(d.get_problem().residues, 'Wfan', 'PWfan')


def test_RunOnce_setup_run(caplog, DummyFactory):
    def Dummy(name):
        return DummyFactory(name, unknown=["p_in.x"])

    caplog.clear()
    logging.disable(logging.NOTSET)  # enable all logging levels

    s = Dummy('mult')
    s.add_driver(RunOnce('runner'))

    expected_msg = "Required iterations detected, not taken into account in RunOnce driver."

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=runonce.__name__):
        s.call_setup_run()
        assert expected_msg in caplog.text

    m = Mixer21('merger')
    m.add_driver(RunOnce('runner'))
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=runonce.__name__):
        m.call_setup_run()
        assert expected_msg in caplog.text


def test_RunOnce__precompute(DummyFactory):
    s = DummyFactory('mult')
    d = s.add_driver(RunOnce('runner'))
    d.solution['dummy'] = 1.
    d._precompute()
    assert len(d.solution) == 0


def test_RunOnce_compute(DummyFactory):
    def Dummy(name):
        return DummyFactory(name, unknown=["p_in.x"])

    s = Dummy('mult')

    d = s.add_driver(RunOnce('runner'))
    assert len(d.solution) == 0
    s.compute = MagicMock()
    s.run_drivers()
    assert_keys(d.solution, 'p_in.x')
    assert d.solution['p_in.x'] == 1
    s.compute.assert_called_once()


def test_RunOnce__postcompute(DummyFactory):
    def Dummy(name):
        return DummyFactory(name, unknown=["p_in.x"])
            
    s = Dummy('mult')

    d = s.add_driver(RunOnce('runner'))
    assert len(d.solution) == 0
    d._postcompute()
    assert_keys(d.solution, 'p_in.x')
