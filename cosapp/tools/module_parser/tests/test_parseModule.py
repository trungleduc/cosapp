import pytest
from typing import Dict, List, Any

from cosapp.base import System, Port
from cosapp.tools.module_parser.parseModule import (
    find_ports_and_systems,
    get_data_from_class,
    get_data_from_module,
)


def is_System(obj):
    return issubclass(obj, System)


def is_Port(obj):
    return issubclass(obj, Port)


def test_find_ports_and_systems_other_class_import():
    """Test that you can't get a class that doesn't inherit from System or Port"""
    from bogusModule import otherClass
    systemSet, portSet = find_ports_and_systems(otherClass)
    assert systemSet == set()
    assert portSet == set()


def test_find_ports_and_systems_single_system():
    """test that you can get a system"""
    import bogusModule.system as ts
    systemSet, portSet = find_ports_and_systems(ts)
    assert all(map(is_System, systemSet))
    assert all(map(is_Port, portSet))
    assert systemSet == {ts.BogusSingleSystem}
    assert portSet == set()


def test_find_ports_and_systems_single_port():
    """Test that you can get a port"""
    import bogusModule.port as tp
    systemSet, portSet = find_ports_and_systems(tp)
    assert all(map(is_System, systemSet))
    assert all(map(is_Port, portSet))
    assert systemSet == set()
    assert portSet == {tp.BogusSinglePort}


def test_find_ports_and_systems_file_with_several_classes():
    """Test that you can get all classes from one file"""
    import bogusModule.systemAndPort as tsp
    systemSet, portSet = find_ports_and_systems(tsp)
    assert all(map(is_System, systemSet))
    assert all(map(is_Port, portSet))
    assert systemSet == {tsp.BogusSystem, tsp.SystemWithKwargs}
    assert portSet == {tsp.BogusPort, tsp.AbPort, tsp.XyzPort}


def test_find_ports_and_systems_module_import():
    """Test that it's possible to get systems and ports from a full module"""
    import bogusModule
    systemSet, portSet = find_ports_and_systems(bogusModule)
    assert all(map(is_System, systemSet))
    assert all(map(is_Port, portSet))
    assert systemSet == {
        bogusModule.system.BogusSingleSystem,
        bogusModule.systemAndPort.BogusSystem,
        bogusModule.systemAndPort.SystemWithKwargs,
        bogusModule.systemWithChildPulling.BogusSystemChildPulling,
    }
    assert portSet == {
        bogusModule.port.BogusSinglePort,
        bogusModule.systemAndPort.BogusPort,
        bogusModule.systemAndPort.AbPort,
        bogusModule.systemAndPort.XyzPort,
    }


def test_find_ports_and_systems_not_module():
    """Test that an error is raised if the argument isn't a module"""
    with pytest.raises(TypeError):
        find_ports_and_systems('')


def test_get_data_from_class_system():
    from bogusModule.systemAndPort import BogusSystem
    data = get_data_from_class(BogusSystem)

    assert data == {
        'name': 'BogusSystem',
        'mod': 'bogusModule.systemAndPort',
        'className': 'BogusSystem',
        'pack': 'bogusModule',
        'desc': 'This is a markdown docstring,\nwith an indent',
        'inputs':
        [
            {
                'type': 'ExtensiblePort',
                'pack': 'bogusModule',
                'name': 'inwards',
                'desc': '|  |  |\n---|---\n  **foo** &#128274;&#128274; : 0 |',
                'variables':
                [
                    {
                        'name': 'foo',
                    },
                ]
            },
            {
                'type': 'AbPort',
                'pack': 'bogusModule',
                'name': 'p_in',
            },
        ],
        'outputs':
        [
            {
                'name': 'outwards',
                'type': 'ExtensiblePort',
                'pack': 'bogusModule',
                'desc': '|  |  |\n---|---\n  **bar**: 1 |',
                'variables':
                [
                    {
                        'name': 'bar',
                    },
                ],
            },
            {
                'type': 'XyzPort',
                'pack': 'bogusModule',
                'name': 'p_out',
            },
        ],
    }


def test_get_data_from_class_without_ports():
    from bogusModule.system import BogusSingleSystem
    data = get_data_from_class(BogusSingleSystem)

    assert data == {
        'name': 'BogusSingleSystem',
        'mod': 'bogusModule',
        'className': 'BogusSingleSystem',
        'pack': 'bogusModule',
    }


def test_get_data_from_class_port():
    from bogusModule.systemAndPort import XyzPort
    data = get_data_from_class(XyzPort)

    assert data == {
            "name": "XyzPort",
            "pack": "bogusModule",
            "variables":
            [
                {
                    "name": "x",
                    "desc": 'x var',
                },
                {
                    "name": "y",
                    'unit': 'K',
                },
                {
                    "name": "z",
                },
            ],
            "desc": '|  |  |\n---|---\n  **x**: 3 | x var\n  **y**: 2 K |\n  **z**: 1 |',
        }


def test_get_data_from_class_with_kwargs():
    from bogusModule.systemAndPort import SystemWithKwargs
    data = get_data_from_class(SystemWithKwargs, n=5, r=2)
    assert data == {
        'name': 'SystemWithKwargs',
        'mod': 'bogusModule.systemAndPort',
        'className': 'SystemWithKwargs',
        'pack': 'bogusModule',
        'inputs':
        [
            {
                'type': 'ExtensiblePort',
                'pack': 'bogusModule',
                'name': 'inwards',
                'desc': '|  |  |\n---|---\n  **v** &#128274;&#128274; : [1. 1. 1. 1. 1.] |\n  **r** &#128274;&#128274; : 2 |',
                'variables':
                [
                    {
                        'name': 'v',
                    },
                    {
                        'name': 'r',
                    },
                ],
            },
        ],
        'kwargs': {
            'n': 5,
            'r': 2,
        },
    }


def test_get_math_pb():
    from MathPbSystem import AssemblyWithMathPb
    mathPb = get_data_from_class(AssemblyWithMathPb)['mathProblem']
    assert mathPb['nUnknowns'] == 1
    assert len(mathPb['unknowns']) == 1
    assert mathPb['nEquations'] == 3
    assert len(mathPb['equations']) == 1
    

def sorted_by_name(list: List[Dict[str, Any]]):
    return sorted(list, key = lambda elem: elem['name'])


def test_get_data_from_module(expectedSystems, expectedPorts):
    import bogusModule
    ctor_config = dict(SystemWithKwargs=dict(n=2))
    moduleData = get_data_from_module(bogusModule, ctor_config)

    assert moduleData['name'] == 'bogusModule'
    assert moduleData['version'] == '0.0.0'
    assert sorted_by_name(moduleData['systems']) == sorted_by_name(expectedSystems)
    assert sorted_by_name(moduleData['ports']) == sorted_by_name(expectedPorts)


def test_get_data_from_module_with_opt_kwargs(expectedSystems, expectedPorts):
    import bogusModule
    ctor_config = dict(SystemWithKwargs=dict(n=2, r=1))
    moduleData = get_data_from_module(bogusModule, ctor_config)

    for system in expectedSystems:
        if system['name'] == 'SystemWithKwargs':
            system['inputs'][0]['variables'].append({
                'name': 'r',
            })
            system['inputs'][0]['desc'] += '\n  **r** &#128274;&#128274; : 1 |'
            system['kwargs']['r'] = 1 # add the optional argument to the expected result

    assert sorted_by_name(moduleData['systems']) == sorted_by_name(expectedSystems)
    assert sorted_by_name(moduleData['ports']) == sorted_by_name(expectedPorts)


def test_get_data_from_module_system_kwargs_with_alias():
    import bogusModule
    alias = 'TestAlias'
    ctor_config = dict(SystemWithKwargs=[dict(n=3, __alias__=alias), dict(n=2, r=1)])
    moduleData = get_data_from_module(bogusModule, ctor_config)

    expectedSystemWithAlias = {
        'name': alias,
        'className': 'SystemWithKwargs',
        'mod': 'bogusModule.systemAndPort',
        'pack': 'bogusModule',
        'inputs': [
            {
                'name': 'inwards',
                'type': 'ExtensiblePort',
                'pack': 'bogusModule',
                'desc': '|  |  |\n---|---\n  **v** &#128274;&#128274; : [1. 1. 1.] |',
                'variables':
                [{
                    'name': 'v',
                },],
            },
        ],
        'kwargs': { 'n': 3, },
    }

    expectedSystemWithoutAlias = {
        'name': 'SystemWithKwargs (n=2, r=1)',
        'className': 'SystemWithKwargs',
        'mod': 'bogusModule.systemAndPort',
        'pack': 'bogusModule',
        'inputs':
        [
            {
                'name': 'inwards',
                'type': 'ExtensiblePort',
                'pack': 'bogusModule',
                'desc': '|  |  |\n---|---\n  **v** &#128274;&#128274; : [1. 1.] |\n  **r** &#128274;&#128274; : 1 |',
                'variables': [
                    {
                        'name': 'v',
                    },
                    {
                        'name': 'r',
                    }
                ],
            },
        ],
        'kwargs': { 'n': 2, 'r': 1, },
    }

    c = 0
    for system in moduleData['systems']:
        if system['className'] == 'SystemWithKwargs':
            if system['name'] == alias:
                assert system == expectedSystemWithAlias
                c += 1
            else:
                assert system == expectedSystemWithoutAlias
                c += 1

    expectedSystems = [expectedSystemWithAlias, expectedSystemWithoutAlias]
    assert c == 2, f'{c} systems instead of {len(expectedSystems)}'


def test_get_data_from_module_includes_excludes(expectedSystems):
    import bogusModule
    includes = 'BogusSystem*'
    excludes = ['BogusSystem?*']
    moduleData = get_data_from_module(bogusModule, includes=includes, excludes=excludes)

    expectedSystems = [
        {
            'name': 'BogusSystem',
            'className': 'BogusSystem',
            'mod': 'bogusModule.systemAndPort',
            'pack': 'bogusModule',
            'desc': 'This is a markdown docstring,\nwith an indent',
            'inputs':
            [
                {
                    'name': 'inwards',
                    'type': 'ExtensiblePort',
                    'pack': 'bogusModule',
                    'desc': '|  |  |\n---|---\n  **foo** &#128274;&#128274; : 0 |',
                    'variables':
                    [{
                        'name': 'foo',
                    },],
                },
                {
                    'type': 'AbPort',
                    'pack': 'bogusModule',
                    'name': 'p_in'
                },
            ],
            'outputs':
            [
                {
                    'name': 'outwards',
                    'type': 'ExtensiblePort',
                    'pack': 'bogusModule',
                    'desc': '|  |  |\n---|---\n  **bar**: 1 |',
                    'variables':
                    [{
                        'name': 'bar',
                    },],
                },
                {
                    'type': 'XyzPort',
                    'pack': 'bogusModule',
                    'name': 'p_out'
                },
            ],
        },
    ]

    expectedPorts = [ # ports used by the system
        {
            "name": "XyzPort",
            "pack": "bogusModule",
            "variables":
            [
                {
                    "name": "x",
                    "desc": 'x var',
                },
                {
                    "name": "y",
                    'unit': 'K',
                },
                {
                    "name": "z",
                },
            ],
            "desc": '|  |  |\n---|---\n  **x**: 3 | x var\n  **y**: 2 K |\n  **z**: 1 |',
        },
        {
            "name": "AbPort",
            "pack": "bogusModule",
            "variables":
            [
                {
                    "name": "a",
                },
                {
                    "name": "b",
                },
            ],
            "desc": '|  |  |\n---|---\n  **a**: 1 |\n  **b**: [0. 0. 0.] |',
        },
    ]

    assert sorted_by_name(moduleData['systems']) == sorted_by_name(expectedSystems)
    assert sorted_by_name(moduleData['ports']) == sorted_by_name(expectedPorts)


def test_get_data_from_module_packageName():
    import bogusModule
    packageName = 'testPackageName'
    moduleData = get_data_from_module(bogusModule, package_name=packageName)

    assert moduleData['name'] == packageName
    for system in moduleData['systems']:
        assert system['pack'] == packageName
    for port in moduleData['ports']:
        assert port['pack'] == packageName


def test_get_data_from_module_child():
    import bogusModule.systemWithChildPulling as bg
    moduleData = get_data_from_module(bg)

    assert moduleData['name'] == bg.__name__
    assert sorted_by_name(moduleData['systems']) == sorted_by_name([
        {
            'name': 'BogusSystemChildPulling',
            'className': 'BogusSystemChildPulling',
            'mod': 'bogusModule.systemWithChildPulling',
            'pack': 'bogusModule.systemWithChildPulling',
            'inputs':
            [{
                'name': 'p_in_parent',
                'pack': 'bogusModule.systemWithChildPulling',
                'type': 'AbPort',
            },],
        },
        {
            'name': 'BogusSystem',
            'className': 'BogusSystem',
            'mod': 'bogusModule.systemAndPort',
            'pack': 'bogusModule.systemWithChildPulling',
            'desc': 'This is a markdown docstring,\nwith an indent',
            'inputs':
            [
                {
                    'name': 'inwards',
                    'type': 'ExtensiblePort',
                    'pack': 'bogusModule.systemWithChildPulling',
                    'desc': '|  |  |\n---|---\n  **foo** &#128274;&#128274; : 0 |',
                    'variables':
                    [{
                        'name': 'foo',
                    },],
                },
                {
                    'type': 'AbPort',
                    'pack': 'bogusModule.systemWithChildPulling',
                    'name': 'p_in'
                },
            ],
            'outputs':
            [
                {
                    'name': 'outwards',
                    'type': 'ExtensiblePort',
                    'pack': 'bogusModule.systemWithChildPulling',
                    'desc': '|  |  |\n---|---\n  **bar**: 1 |',
                    'variables':
                    [{
                        'name': 'bar',
                    },],
                },
                {
                    'type': 'XyzPort',
                    'pack': 'bogusModule.systemWithChildPulling',
                    'name': 'p_out'
                },
            ],
        },
    ])

    assert sorted_by_name(moduleData['ports']) == sorted_by_name([
        {
            'name': 'AbPort',
            'pack': 'bogusModule.systemWithChildPulling',
            'desc': '|  |  |\n---|---\n  **a**: 1 |\n  **b**: [0. 0. 0.] |',
            'variables': [
                {
                    'name': 'a',
                },
                {
                    'name': 'b',
                }
            ]
        },
        {
            "name": "XyzPort",
            "pack": "bogusModule.systemWithChildPulling",
            "desc": '|  |  |\n---|---\n  **x**: 3 | x var\n  **y**: 2 K |\n  **z**: 1 |',
            "variables":
            [
                {
                    "name": "x",
                    "desc": 'x var',
                },
                {
                    "name": "y",
                    'unit': 'K',
                },
                {
                    "name": "z",
                },
            ],
            
        }
    ])


@pytest.fixture
def expectedSystems():
    return [
        {
            'name': 'BogusSingleSystem',
            'className': 'BogusSingleSystem',
            'mod': 'bogusModule',
            'pack': 'bogusModule',
        },
        {
            'name': 'BogusSystem',
            'className': 'BogusSystem',
            'mod': 'bogusModule.systemAndPort',
            'pack': 'bogusModule',
            'desc': 'This is a markdown docstring,\nwith an indent',
            'inputs':
            [
                {
                    'name': 'inwards',
                    'type': 'ExtensiblePort',
                    'pack': 'bogusModule',
                    'desc': '|  |  |\n---|---\n  **foo** &#128274;&#128274; : 0 |',
                    'variables':
                    [{
                        'name': 'foo',
                    },],
                },
                {
                    'type': 'AbPort',
                    'pack': 'bogusModule',
                    'name': 'p_in'
                },
            ],
            'outputs':
            [
                {
                    'name': 'outwards',
                    'type': 'ExtensiblePort',
                    'pack': 'bogusModule',
                    'desc': '|  |  |\n---|---\n  **bar**: 1 |',
                    'variables':
                    [{
                        'name': 'bar',
                    },],
                },
                {
                    'type': 'XyzPort',
                    'pack': 'bogusModule',
                    'name': 'p_out'
                },
            ],
        },
        {
            'name': 'SystemWithKwargs',
            'className': 'SystemWithKwargs',
            'mod': 'bogusModule.systemAndPort',
            'pack': 'bogusModule',
            'inputs':
            [
                {
                    'name': 'inwards',
                    'type': 'ExtensiblePort',
                    'pack': 'bogusModule',
                    'desc': '|  |  |\n---|---\n  **v** &#128274;&#128274; : [1. 1.] |',
                    'variables':
                    [{
                        'name': 'v',
                    },],
                },
            ],
            'kwargs': { 'n': 2, },
        },
        {
            'name': 'BogusSystemChildPulling',
            'className': 'BogusSystemChildPulling',
            'mod': 'bogusModule.systemWithChildPulling',
            'pack': 'bogusModule',
            'inputs':
            [{
                'name': 'p_in_parent',
                'pack': 'bogusModule',
                'type': 'AbPort',
            },],
        },
    ]


@pytest.fixture
def expectedPorts():
    return [
        {
            "name": "XyzPort",
            "pack": "bogusModule",
            "variables":
            [
                {
                    "name": "x",
                    "desc": 'x var',
                },
                {
                    "name": "y",
                    'unit': 'K',
                },
                {
                    "name": "z",
                },
            ],
            "desc": '|  |  |\n---|---\n  **x**: 3 | x var\n  **y**: 2 K |\n  **z**: 1 |',
        },
        {
            "name": "AbPort",
            "pack": "bogusModule",
            "variables":
            [
                {
                    "name": "a",
                },
                {
                    "name": "b",
                },
            ],
            "desc": '|  |  |\n---|---\n  **a**: 1 |\n  **b**: [0. 0. 0.] |',
        },
        {
            "name": "BogusPort",
            "pack": "bogusModule",
            "variables": [],
            "desc": '|  |  |\n---|---',
        },
        {
            "name": "BogusSinglePort",
            "pack": "bogusModule",
            "variables": [],
            "desc": '|  |  |\n---|---',
        }
    ]