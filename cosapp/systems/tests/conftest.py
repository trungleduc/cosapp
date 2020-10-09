import sys
from pathlib import Path

import pytest
import numpy as np
from collections import OrderedDict

from cosapp.systems import System

import cosapp.tests as test

from io import StringIO

@pytest.fixture
def test_library():
    library_path = Path(test.__file__).parent / "library" / "systems"

    # Add path to allow System to find the component
    sys.path.append(str(library_path))
    try:
        yield library_path
    finally:
        # Undo path modification
        sys.path.remove(str(library_path))


@pytest.fixture
def test_data():
    return Path(test.__file__).parent / "data"



@pytest.fixture(scope="function")
def DummyFactory():
    """Factory creating a dummy system with custom attributes"""
    # mapping option / method
    # for example: `inputs` <-> `add_input`
    mapping = dict(
        (option, "add_" + option[:-1])
        for option in ("inputs", "outputs", "inwards", "outwards",
            "transients", "rates", "unknowns", "equations",
            "design_methods")
    )
    mapping["properties"] = "add_property"

    def Factory(name, **options):
        method_dict = OrderedDict(
            (mapping[option], options.pop(option))
            for option in list(options.keys()) if option in mapping
        )
        base = options.pop("base", System)
        class PrototypeSystem(base):
            def setup(self, **options):
                super().setup(**options)
                for method, values in method_dict.items():
                    if values is None:
                        continue
                    if not isinstance(values, list):
                        values = [values]
                    for args, kwargs in values:  # expects a list of (tuple, dict)
                        getattr(self, method)(*args, **kwargs)
        return PrototypeSystem(name, **options)
    return Factory



class FunkySystem(System):
    def setup(self):
        self.add_inward('m', 0.0)
        self.add_inward('v', np.zeros(3))
        self.add_outward('y')
        
        self.add_transient('x', der='v')
        self.add_transient('foo', der='m / y')

    def compute(self):
        self.y = np.exp(self.m)


@pytest.fixture(scope="function")
def funky():
    return FunkySystem('funky')



class GroovySystem(System):
    def setup(self):
        self.add_inward('bass', 0.0)
        self.add_inward('drums', np.zeros(3))

        self.add_child(FunkySystem('brass'))
        
        self.add_transient('F', der='bass')
        self.add_transient('G', der='drums + brass.x')
        self.add_rate('dB_dt', source='brass.x')


@pytest.fixture(scope="function")
def groovy():
    return GroovySystem('groovy')



class JazzySystem(GroovySystem):
    def setup(self):
        super().setup()
        self.add_rate('dH_dt', source='drums + brass.x')
        self.add_child(GroovySystem('sub'))


@pytest.fixture(scope="function")
def jazzy():
    return JazzySystem('jazzy')

@pytest.fixture()
def config():
    return StringIO(
        """{
        "$schema": "0-3-0/system.schema.json",
        "p1": {
            "class": "pressurelossvarious.PressureLossSys",
            "subsystems": {
            "p11": {
                "class": "pressurelossvarious.PressureLoss0D"
            },
            "p12": {
                "class": "pressurelossvarious.PressureLoss0D"
            }
            },
            "connections": [
                ["flnum_in", "p11.flnum_in"],
                ["p11.flnum_out", "p12.flnum_in"],
                ["p12.flnum_out", "flnum_out"]
            ],
            "exec_order": ["p11", "p12"]
        }}"""
    )