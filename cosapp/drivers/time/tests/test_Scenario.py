import pytest
import numpy as np

from cosapp.drivers import EulerExplicit
from cosapp.drivers.time.scenario import Scenario, Interpolator
from cosapp.core.eval_str import AssignString


@pytest.fixture(scope='function')
def case(two_tank_case):
    return two_tank_case(EulerExplicit, dt=0.1, time_interval=[0, 1])


@pytest.fixture(scope='function')
def scenario(case):
    driver = case[1]
    return Scenario('scenario', driver)


@pytest.fixture(scope='function')
def varying_mu_case(case):
    """Two-tank test case with exponentially decaying viscosity"""
    system, driver = case

    mu_0, mu_inf, tau = 1e-2, 2e-3, 200  # viscosity parameters
    mu = lambda t: mu_inf + (mu_0 - mu_inf) * np.exp(-t / tau)

    scenario = Scenario.make('scenario',
        driver = driver,
        init = {
            'tank1.height': '10 * pipe.D',
            'tank2.height': 'tank1.height + 1',
        },
        values = {
            'pipe.D': 0.3,
            'tank1.area': '5',
            'pipe.L': '50 * pipe.D',
            'pipe.mu': f'{mu_inf} + ({mu_0} - {mu_inf}) * exp(-t / {tau})',
        },
    )
    return system, driver, scenario, mu


def test_Scenario__init__(case):
    system, driver = case
    scenario = Scenario('scenario', driver)
    assert scenario.owner is driver
    assert scenario.context is system
    assert scenario.init_values == []
    assert scenario.case_values == []
    with pytest.raises(TypeError):
        scenario.owner = 'foo'


@pytest.mark.parametrize("init, expected", [
    (
        {'tank1.height': 1.2, 'tank2.height': 0.5},
        dict(content=['tank1.height = 1.2', 'tank2.height = 0.5'], n_const=2)
    ),
    (
        {'tank2.height': 0.5, 'tank1.height': 1.2},
        dict(content=['tank1.height = 1.2', 'tank2.height = 0.5'], n_const=2)
    ),
    (
        {'tank2.height': 'tank1.height + 1', 'tank1.height': 2, },
        dict(content=['tank1.height = 2', 'tank2.height = tank1.height + 1'], n_const=1)
    ),
    (
        {'tank1.height': '10 * pipe.D', 'tank2.height': 0.5, },
        dict(content=['tank2.height = 0.5', 'tank1.height = 10 * pipe.D'], n_const=1)
    ),
    (
        {'tank1.height': 'exp(1.5)', 'tank2.height': '10 * pipe.D', 'pipe.D': 0.25},
        dict(content=['pipe.D = 0.25', 'tank1.height = exp(1.5)', 'tank2.height = 10 * pipe.D'], n_const=2)
    ),
    (
        {'tank1.height': 'pipe.L / 2', 'tank2.height': '10 * pipe.D', 'pipe.D': 0.25},
        dict(content=['pipe.D = 0.25', 'tank1.height = pipe.L / 2', 'tank2.height = 10 * pipe.D'], n_const=1)
    ),
    (
        {'tank1.height': 'pipe.L / 2', 'tank2.height': '10 * pipe.D'},
        dict(content=['tank1.height = pipe.L / 2', 'tank2.height = 10 * pipe.D'], n_const=0)
    ),
    # Erroneous cases:
    ({'foo.bar': 2}, dict(error=AttributeError, match=r"'foo\.bar' is not known in system")),
    ({'pipe.k': 2}, dict(error=ValueError, match="Only variables in input ports can be used as boundaries")),
])
def test_Scenario_set_init(scenario, init, expected):
    error = expected.get('error', None)

    if error is None:
        scenario.set_init(init)
        assert all(isinstance(assignment, AssignString) for assignment in scenario.init_values)

        n_const = expected['n_const']
        content = expected['content']
        const = content[:n_const]
        nonconst = content[n_const:]
        init_conds = [str(assignment) for assignment in scenario.init_values]
        # First elements, expected to be constants init_conds 
        assert len(init_conds[:n_const]) == n_const
        assert set(init_conds[:n_const]) == set(const)
        assert all(assignment.constant for assignment in scenario.init_values[:n_const])
        # End init_conds (unordered, from dict entry)
        assert len(init_conds[n_const:]) == len(nonconst)
        assert set(init_conds[n_const:]) == set(nonconst)

    else:
        pattern = expected.get('match', None)
        with pytest.raises(error, match=pattern):
            scenario.set_init(init)


@pytest.mark.parametrize("values, expected", [
    (
        {'pipe.D': 0.3, 'pipe.L': '5'},
        dict(init=['pipe.L = 5', 'pipe.D = 0.3'], values=[],)
    ),
    (
        {'pipe.D': 0.3, 'pipe.L': '50 * pipe.D'},
        dict(init=['pipe.D = 0.3'], values=['pipe.L = 50 * pipe.D'],)
    ),
    (
        {
            'tank1.area': 2,
            'pipe.D': 0.3,
            'pipe.L': '50 * pipe.D',
            'pipe.mu': '{vinf} + ({v0} - {vinf}) * exp(-t / {tau})'.format(v0=1e-2, vinf=2e-3, tau=200),
        },
        dict(
            init=['pipe.D = 0.3', 'tank1.area = 2'],
            values=[
                'pipe.L = 50 * pipe.D',
                'pipe.mu = {vinf} + ({v0} - {vinf}) * exp(-t / {tau})'.format(v0=1e-2, vinf=2e-3, tau=200)
            ],
        )
    ),
    # Erroneous cases:
    ({'foo.bar': 2}, dict(error=AttributeError, match=r"'foo\.bar' is not known in system")),
    ({'pipe.k': 2}, dict(error=ValueError, match="Only variables in input ports can be used as boundaries")),
])
def test_Scenario_set_values(scenario, values, expected):
    error = expected.get('error', None)

    if error is None:
        scenario.set_values(values)
        assert all(isinstance(assignment, AssignString) for assignment in scenario.case_values)
        assert all(not assignment.constant for assignment in scenario.case_values)

        init_conds = [str(assignment) for assignment in scenario.init_values]
        time_conds = [str(assignment) for assignment in scenario.case_values]

        # Test time-dependent boundary conditions
        assert set(time_conds) == set(expected['values'])
        assert set(init_conds) == set(expected['init'])

    else:
        pattern = expected.get('match', None)
        with pytest.raises(error, match=pattern):
            scenario.set_values(values)


def test_Scenario_add_init(case):
    system, driver = case

    scenario = Scenario.make('scenario',
        driver = driver,
        values = {},
        init = {
            'tank1.height': '10 * pipe.D',
        },
    )
    assert len(scenario.init_values) == 1
    assert str(scenario.init_values[0]) == 'tank1.height = 10 * pipe.D'

    scenario.add_init({'tank2.height': 'tank1.height + 1'})
    assert len(scenario.init_values) == 2
    assert str(scenario.init_values[1]) == 'tank2.height = tank1.height + 1'

    scenario.add_init({'tank2.area': '1.3'})
    assert len(scenario.init_values) == 3
    assert str(scenario.init_values[0]) == 'tank2.area = 1.3'
    assert str(scenario.init_values[1]) == 'tank1.height = 10 * pipe.D'
    assert str(scenario.init_values[2]) == 'tank2.height = tank1.height + 1'


def test_Scenario_add_values(case):
    system, driver = case
    mu_0, mu_inf, tau = 1e-2, 2e-3, 200

    scenario = Scenario.make('scenario',
        driver = driver,
        values = {
            'tank1.area': '5',  # constant
            'pipe.D': 0.3,      # constant
            'pipe.mu': f'{mu_inf} + ({mu_0} - {mu_inf}) * exp(-t / {tau})',
        },
        init = {
            'tank1.height': '10 * pipe.D',
            'tank2.height': 'tank1.height + 1',
        },
    )

    assert len(scenario.init_values) == 4
    assert len(scenario.case_values) == 1

    init_conds = [str(assignment) for assignment in scenario.init_values]
    assert set(init_conds[:2]) == {'tank1.area = 5', 'pipe.D = 0.3'}
    assert set(init_conds[2:]) == {
        'tank1.height = 10 * pipe.D',
        'tank2.height = tank1.height + 1'
    }

    case_conds = [str(assignment) for assignment in scenario.case_values]
    assert set(case_conds) == {
            f'pipe.mu = {mu_inf} + ({mu_0} - {mu_inf}) * exp(-t / {tau})',
    }

    # Add non-const boundary condition
    scenario.add_values({'tank2.area': '10 * pipe.D**2'})
    assert len(scenario.init_values) == 4
    assert len(scenario.case_values) == 2
    assert str(scenario.case_values[-1]) == 'tank2.area = 10 * pipe.D**2'

    # Add constant boundary condition
    scenario.add_values({'pipe.L': 3.4})
    assert len(scenario.init_values) == 5
    assert len(scenario.case_values) == 2
    assert str(scenario.init_values[0]) == 'pipe.L = 3.4'


def test_Scenario_add_values_tabulated(case):
    system, driver = case
    mu_0, mu_inf, tau = 1e-2, 2e-3, 200

    scenario = Scenario.make('scenario',
        driver = driver,
        values = {
            'tank1.area': '5',  # constant
            'pipe.D': 0.3,      # constant
            'pipe.mu': Interpolator(data=[(0, mu_0), (1, mu_0), (1.1, mu_inf), (1000, mu_inf)]),
        },
        init = {
            'tank1.height': '10 * pipe.D',
            'tank2.height': 'tank1.height + 1',
        },
    )

    assert len(scenario.init_values) == 4
    assert len(scenario.case_values) == 1

    init_conds = [str(assignment) for assignment in scenario.init_values]
    assert set(init_conds[:2]) == {'tank1.area = 5', 'pipe.D = 0.3'}
    assert set(init_conds[2:]) == {
        'tank1.height = 10 * pipe.D',
        'tank2.height = tank1.height + 1'
    }

    case_conds = [str(assignment) for assignment in scenario.case_values]
    assert set(case_conds) == {
            'pipe.mu = Interpolator(t)',
    }


@pytest.mark.parametrize("case_data, expected", [
    (
        dict(
            init = {'tank1.height': '10 * pipe.D', 'tank2.height': 'tank1.height + 1'},
            values = {'pipe.D': 0.3, 'tank1.area': '5'},
        ),
        dict(
            init = dict(
                content=[
                    'pipe.D = 0.3', 'tank1.area = 5',  # const assignments passed as init conditions
                    'tank1.height = 10 * pipe.D', 'tank2.height = tank1.height + 1'
                ],
                n_const=2
            ),
            values = [],
        )
    ),
    (
        dict(
            init = {'tank1.height': '10 * pipe.D', 'tank2.height': 'tank1.height + 1'},
            values = {
                'pipe.D': 0.3,
                'tank1.area': '5',
                'pipe.L': '50 * pipe.D',
                'pipe.mu': '{vinf} + ({v0} - {vinf}) * exp(-t / {tau})'.format(v0=1e-2, vinf=2e-3, tau=200),
            },
        ),
        dict(
            init = dict(
                content=[
                    'pipe.D = 0.3', 'tank1.area = 5',  # const assignments passed as init conditions
                    'tank1.height = 10 * pipe.D', 'tank2.height = tank1.height + 1'
                ],
                n_const=2
            ),
            values = [
                'pipe.L = 50 * pipe.D',
                'pipe.mu = {vinf} + ({v0} - {vinf}) * exp(-t / {tau})'.format(v0=1e-2, vinf=2e-3, tau=200),
            ],
        )
    ),
])
def test_Scenario_factory(case, case_data, expected):
    system, driver = case
    scenario = Scenario.make('scenario',
        driver = driver,
        init = case_data['init'],
        values = case_data['values'],
    )

    assert scenario.owner is driver
    assert scenario.context is system
    assert isinstance(scenario, Scenario)
    assert all(isinstance(assignment, AssignString) for assignment in scenario.init_values)
    assert all(isinstance(assignment, AssignString) for assignment in scenario.case_values)
    assert all(assignment.eval_context is system for assignment in scenario.init_values)
    assert all(assignment.eval_context is system for assignment in scenario.case_values)

    init_conds = [str(assignment) for assignment in scenario.init_values]
    time_conds = [str(assignment) for assignment in scenario.case_values]

    # Test time-dependent boundary conditions
    assert set(time_conds) == set(expected['values'])

    # Test initial conditions
    content = expected['init']['content']
    n_const = expected['init']['n_const']
    const = content[:n_const]
    nonconst = content[n_const:]
    # First elements, expected to be constants init_conds 
    assert len(init_conds[:n_const]) == n_const
    assert set(init_conds[:n_const]) == set(const)
    assert all(assignment.constant for assignment in scenario.init_values[:n_const])
    # End init_conds (unordered, from dict entry)
    assert len(init_conds[n_const:]) == len(nonconst)
    assert set(init_conds[n_const:]) == set(nonconst)


def test_Scenario_apply_values(varying_mu_case, set_master_system):
    """Test methods `apply_init_values()` and `update_values()`"""
    system, driver, scenario, mu = varying_mu_case

    system.tank1.height = system.tank2.height = 1
    system.tank1.area = system.tank2.area = 1
    system.pipe.D = 1
    system.pipe.L = 1
    system.pipe.mu = 0.1

    scenario.apply_init_values()
    assert system.pipe.D == 0.3
    assert system.tank1.height == 3
    assert system.tank2.height == 4
    assert system.tank1.area == 5
    assert system.tank2.area == 1
    # Check that non-const boundary conditions have not been applied yet
    assert system.pipe.L == 1
    assert system.pipe.mu == 0.1

    # Apply time-dependent boundary conditions
    driver._set_time(0.)
    assert system.time == 0
    scenario.update_values()
    assert system.pipe.L == 15
    assert system.pipe.mu == mu(system.time)

    driver._set_time(10.)
    assert system.time == 10
    scenario.update_values()
    assert system.pipe.L == 15
    assert system.pipe.mu == mu(system.time)


def test_Scenario_clear_init(varying_mu_case):
    scenario = varying_mu_case[2]
    assert len(scenario.init_values) == 4
    assert len(scenario.case_values) == 2
    
    scenario.clear_init()
    assert len(scenario.init_values) == 0
    assert len(scenario.case_values) == 2


def test_Scenario_clear_values(varying_mu_case):
    scenario = varying_mu_case[2]
    assert len(scenario.init_values) == 4
    assert len(scenario.case_values) == 2
    
    scenario.clear_values()
    assert len(scenario.init_values) == 4
    assert len(scenario.case_values) == 0
