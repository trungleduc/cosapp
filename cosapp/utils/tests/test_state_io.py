import pytest

from cosapp.utils.state_io import get_state, set_state
from cosapp.tests.library.systems import ComplexTurbofan


@pytest.fixture
def turbofan():
    turbofan = ComplexTurbofan('turbofan')
    turbofan.run_once()
    return turbofan


@pytest.fixture
def turbofan_state_init():
    """Turbofan state before `run_once`"""
    state = {
        'ports': {},
        'children': {
            'atm': {
                'ports': {
                    'inwards': {
                        'Pt': 101325,
                        'Tt': 273.15,
                    },
                    'fl_out': {
                        'Pt': 101325,
                        'Tt': 273.15,
                    },
                },
            },
            'inlet': {
                'ports': {
                    'fl_in': {
                        'Tt': 273.15,
                        'Pt': 101325,
                    },
                    'W_in': {
                        'W': 200,
                    },
                    'fl_out': {
                        'Tt': 273.15,
                        'Pt': 101325.0,
                        'W': 0.0,
                    },
                },
            },
            'fanC': {
                'ports': {
                    'inwards': {
                        'gh': 0.1,
                    },
                    'fl_in': {
                        'Tt': 273.15,
                        'Pt': 101325.0,
                        'W': 0.0,
                    },
                    'mech_in': {
                        'XN': 100.0,
                        'PW': 0.0,
                    },
                    'fl_out': {
                        'Tt': 273.15,
                        'Pt': 101325.0,
                        'W': 0.0,
                    },
                },
                'children': {
                    'ductC': {
                        'ports': {
                            'fl_in': {
                                'Tt': 273.15,
                                'Pt': 101325.0,
                                'W': 0.0,
                            },
                            'fl_out': {
                                'Tt': 273.15,
                                'Pt': 101325.0,
                                'W': 0.0,
                            },
                        },
                        'children': {
                            'merger': {
                                'ports': {
                                    'fl1_in': {
                                        'Tt': 273.15,
                                        'Pt': 101325.0,
                                        'W': 0.0,
                                    },
                                    'fl2_in': {
                                        'Tt': 273.15,
                                        'Pt': 101325.0,
                                        'W': 0.0,
                                    },
                                    'fl_out': {
                                        'Tt': 273.15,
                                        'Pt': 101325.0,
                                        'W': 0.0,
                                    },
                                },
                            },
                            'duct': {
                                'ports': {
                                    'inwards': {
                                        'A': 1.0,
                                        'cst_loss': 1.0,
                                        'glp': 0.05,
                                    },
                                    'fl_in': {
                                        'Tt': 273.15,
                                        'Pt': 101325.0,
                                        'W': 0.0,
                                    },
                                    'outwards': {
                                        'PR': 1,
                                    },
                                    'fl_out': {
                                        'Tt': 273.15,
                                        'Pt': 101325.0,
                                        'W': 0.0,
                                    },
                                },
                            },
                            'bleed': {
                                'ports': {
                                    'inwards': {
                                        'split_ratio': 0.99,
                                    },
                                    'fl_in': {
                                        'Tt': 273.15, 
                                        'Pt': 101325.0, 
                                        'W': 0.0,
                                    },
                                    'fl1_out': {
                                        'Tt': 273.15,
                                        'Pt': 101325.0,
                                        'W': 0.0,
                                    },
                                    'fl2_out': {
                                        'Tt': 273.15,
                                        'Pt': 101325.0,
                                        'W': 0.0,
                                    },
                                },
                            },
                        },
                    },
                    'fan': {
                        'ports': {
                            'inwards': {
                                'gh': 0.1,
                            },
                            'fl_in': {
                                'Tt': 273.15,
                                'Pt': 101325.0,
                                'W': 0.0,
                            },
                            'mech_in': {
                                'XN': 100.0,
                                'PW': 0.0,
                            },
                            'outwards': {
                                'pcnr': 0.0,
                                'pr': 0.0,
                                'effis': 1.0,
                                'wr': 0.0,
                                'PWfan': 1000000.0,
                            },
                            'fl_out': {
                                'Tt': 273.15,
                                'Pt': 101325.0,
                                'W': 0.0,
                            },
                        },
                    },
                },
            },
            'merger': {
                'ports': {
                    'fl1_in': {
                        'Tt': 273.15,
                        'Pt': 101325.0,
                        'W': 0.0,
                    },
                    'fl2_in': {
                        'Tt': 273.15,
                        'Pt': 101325.0,
                        'W': 0.0,
                    },
                    'fl_out': {
                        'Tt': 273.15,
                        'Pt': 101325.0,
                        'W': 0.0,
                    },
                },
            },
            'duct': {
                'ports': {
                    'inwards': {
                        'A': 1.0,
                        'cst_loss': 0.98,
                        'glp': 0.05,
                    },
                    'fl_in': {
                        'Tt': 273.15,
                        'Pt': 101325.0,
                        'W': 0.0,
                    },
                    'outwards': {
                        'PR': 1,
                    },
                    'fl_out': {
                        'Tt': 273.15,
                        'Pt': 101325.0,
                        'W': 0.0,
                    },
                },
            },
            'bleed': {
                'ports': {
                    'inwards': {
                        'split_ratio': 0.99,
                    },
                    'fl_in': {
                        'Tt': 273.15,
                        'Pt': 101325.0,
                        'W': 0.0,
                    },
                    'fl1_out': {
                        'Tt': 273.15,
                        'Pt': 101325.0,
                        'W': 0.0,
                    },
                    'fl2_out': {
                        'Tt': 273.15,
                        'Pt': 101325.0,
                        'W': 0.0,
                    },
                },
            },
            'noz': {
                'ports': {
                    'inwards': {
                        'Acol': 0.4,
                        'Aexit': 0.5,
                    },
                    'fl_in': {
                        'Tt': 273.15,
                        'Pt': 101325.0,
                        'W': 0.0,
                    },
                    'outwards': {
                        'WRnozzle': 1,
                    },
                },
            },
        },
    }
    return state


@pytest.fixture
def turbofan_state():
    """Turbofan state after `run_once`"""
    state = {
        "ports": {},
        "children": {
            "atm": {
                "ports": {
                    "inwards": {
                        "Pt": 101325,
                        "Tt": 273.15,
                    },
                    "fl_out": {
                        "Tt": 273.15,
                        "Pt": 101325,
                    },
                }
            },
            "inlet": {
                "ports": {
                    "fl_in": {
                        "Tt": 273.15,
                        "Pt": 101325,
                    },
                    "W_in": {
                        "W": 200,
                    },
                    "fl_out": {
                        "Tt": 273.15,
                        "Pt": 100818.375,
                        "W": 200,
                    },
                },
            },
            "fanC": {
                "ports": {
                    "inwards": {
                        "gh": 0.1,
                    },
                    "fl_in": {
                        "Tt": 273.15,
                        "Pt": 100818.375,
                        "W": 200,
                    },
                    "mech_in": {
                        "XN": 100.0,
                        "PW": 0.0,
                    },
                    "fl_out": {
                        "Tt": 334.30312923003123,
                        "Pt": 204468.7879122772,
                        "W": 188.93525535420105,
                    }
                },
                "children": {
                    "ductC": {
                        "ports": {
                            "fl_in": {
                                "Tt": 273.15,
                                "Pt": 100818.375,
                                "W": 200,
                            },
                            "fl_out": {
                                "Tt": 273.15,
                                "Pt": 100818.375,
                                "W": 198.0,
                            },
                        },
                        "children": {
                            "merger": {
                                "ports": {
                                    "fl1_in": {
                                        "Tt": 273.15,
                                        "Pt": 100818.375,
                                        "W": 200,
                                    },
                                    "fl2_in": {
                                        "Tt": 273.15,
                                        "Pt": 101325.0,
                                        "W": 0.0,
                                    },
                                    "fl_out": {
                                        "Tt": 273.15,
                                        "Pt": 100818.375,
                                        "W": 200.0,
                                    },
                                },
                            },
                            "duct": {
                                "ports": {
                                    "inwards": {
                                        "A": 1.0,
                                        "cst_loss": 1.0,
                                        "glp": 0.05,
                                    },
                                    "fl_in": {
                                        "Tt": 273.15,
                                        "Pt": 100818.375,
                                        "W": 200.0,
                                    },
                                    "outwards": {
                                        "PR": 1.0,
                                    },
                                    "fl_out": {
                                        "Tt": 273.15,
                                        "Pt": 100818.375,
                                        "W": 200.0,
                                    },
                                },
                            },
                            "bleed": {
                                "ports": {
                                    "inwards": {
                                        "split_ratio": 0.99,
                                    },
                                    "fl_in": {
                                        "Tt": 273.15,
                                        "Pt": 100818.375,
                                        "W": 200.0,
                                    },
                                    "fl1_out": {
                                        "Tt": 273.15,
                                        "Pt": 100818.375,
                                        "W": 198.0,
                                    },
                                    "fl2_out": {
                                        "Tt": 273.15,
                                        "Pt": 100818.375,
                                        "W": 2.0000000000000018,
                                    },
                                },
                            },
                        },
                    },
                    "fan": {
                        "ports": {
                            "inwards": {
                                "gh": 0.1,
                            },
                            "fl_in": {
                                "Tt": 273.15,
                                "Pt": 100818.375,
                                "W": 198.0,
                            },
                            "mech_in": {
                                "XN": 100.0,
                                "PW": 0.0,
                            },
                            "outwards": {
                                "pcnr": 102.70904935462133,
                                "pr": 0.0,
                                "effis": 1.0,
                                "wr": 0.0,
                                "PWfan": 11600198.01513155,
                            },
                            "fl_out": {
                                "Tt": 334.30312923003123,
                                "Pt": 204468.7879122772,
                                "W": 188.93525535420105,
                            },
                        },
                    },
                },
            },
            "merger": {
                "ports": {
                    "fl1_in": {
                        "Tt": 334.30312923003123,
                        "Pt": 204468.7879122772,
                        "W": 188.93525535420105,
                    },
                    "fl2_in": {
                        "Tt": 273.15,
                        "Pt": 101325.0,
                        "W": 0.0,
                    },
                    "fl_out": {
                        "Tt": 334.30312923003123,
                        "Pt": 204468.7879122772,
                        "W": 188.93525535420105,
                    },
                },
            },
            "duct": {
                "ports": {
                    "inwards": {
                        "A": 1.0,
                        "cst_loss": 0.98,
                        "glp": 0.05,
                    },
                    "fl_in": {
                        "Tt": 334.30312923003123,
                        "Pt": 204468.7879122772,
                        "W": 188.93525535420105,
                    },
                    "outwards": {
                        "PR": 0.98
                    },
                    "fl_out": {
                        "Tt": 334.30312923003123,
                        "Pt": 200379.41215403166,
                        "W": 188.93525535420105,
                    },
                },
            },
            "bleed": {
                "ports": {
                    "inwards": {
                        "split_ratio": 0.99,
                    },
                    "fl_in": {
                        "Tt": 334.30312923003123,
                        "Pt": 200379.41215403166,
                        "W": 188.93525535420105,
                    },
                    "fl1_out": {
                        "Tt": 334.30312923003123,
                        "Pt": 200379.41215403166,
                        "W": 187.04590280065904,
                    },
                    "fl2_out": {
                        "Tt": 334.30312923003123,
                        "Pt": 200379.41215403166,
                        "W": 1.8893525535420121,
                    },
                },
            },
            "noz": {
                "ports": {
                    "inwards": {
                        "Acol": 0.4,
                        "Aexit": 0.5,
                    },
                    "fl_in": {
                        "Tt": 334.30312923003123,
                        "Pt": 200379.41215403166,
                        "W": 187.04590280065904,
                    },
                    "outwards": {
                        "WRnozzle": 101.87617310334723,
                    },
                },
            },
        },
    }
    return state


def test_get_state(turbofan: ComplexTurbofan, turbofan_state: dict):
    state = get_state(turbofan)

    assert set(state) == {'ports', 'children'}
    assert isinstance(state['ports'], dict)
    assert isinstance(state['children'], dict)

    assert set(state['ports']) == set()
    assert set(state['children']) == {
        'atm',
        'inlet',
        'fanC',
        'merger',
        'duct',
        'bleed',
        'noz',
    }
    assert set(state['children']['fanC']['ports']) == {
        'inwards',
        'mech_in',
        'fl_in',
        'fl_out',
    }

    assert state == turbofan_state


def test_set_state(turbofan: ComplexTurbofan, turbofan_state: dict):
    assert turbofan.inlet.W_in.W == 200.0
    assert turbofan.bleed.split_ratio == 0.99
    assert turbofan.noz.fl_in.W == pytest.approx(187.0459)
    assert turbofan.noz.WRnozzle == pytest.approx(101.87617)
    assert turbofan.bleed.fl2_out.W == pytest.approx(1.889352)
    assert turbofan.noz.fl_in.W == turbofan_state['children']['noz']['ports']['fl_in']['W']
    assert turbofan.noz.WRnozzle == turbofan_state['children']['noz']['ports']['outwards']['WRnozzle']
    assert turbofan.bleed.fl2_out.W == turbofan_state['children']['bleed']['ports']['fl2_out']['W']

    turbofan.inlet.W_in.W = 100.0
    turbofan.bleed.split_ratio = 0.985
    turbofan.run_once()

    assert turbofan.inlet.W_in.W == 100.0
    assert turbofan.bleed.split_ratio == 0.985
    assert turbofan.noz.fl_in.W == pytest.approx(187.962239)
    assert turbofan.noz.WRnozzle == pytest.approx(102.37526)
    assert turbofan.bleed.fl2_out.W == pytest.approx(2.8623691)

    set_state(turbofan, turbofan_state)
    assert turbofan.inlet.W_in.W == 200.0
    assert turbofan.bleed.split_ratio == 0.99
    assert turbofan.noz.fl_in.W == pytest.approx(187.0459)
    assert turbofan.noz.WRnozzle == pytest.approx(101.87617)
    assert turbofan.bleed.fl2_out.W == pytest.approx(1.889352)
