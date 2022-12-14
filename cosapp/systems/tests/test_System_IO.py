import pytest
import numpy as np
import json
from io import StringIO

from cosapp.utils.testing import assert_keys
from cosapp.tests.library.systems import AllTypesSystem
from cosapp.tests.library.ports import NumPort, V1dPort
from cosapp.ports.port import ExtensiblePort, ModeVarPort
from cosapp.systems import System


class SomeClass:
    """Regular, non-CoSApp class"""
    def __init__(self) -> None:
        self.x = np.linspace(-1, 1, 3)


class SystemWithProps(System):
    """CoSApp system class with various properties"""
    def setup(self) -> None:
        self.add_property('n', 3)
        self.add_property('g', 9.81)
        self.add_property('c', SomeClass())


class SystemWithNone(System):
    def setup(self) -> None:
        self.add_inward('x', None)


class SystemWithKwargs(System):
    def setup(self, n: int, r=None) -> None:
        self.add_property('n', n)
        self.add_inward('v', np.ones(n))
        if r is not None:
            self.add_inward('r', r)


def test_System_load(test_library):
    # Load super simple module
    config = StringIO(
        """{
            "$schema": "0-2-0/system.schema.json",
            "p1": {
            "class": "pressurelossvarious.PressureLoss0D"
            }
        }"""
    )
    s = System.load(config)

    assert s.__module__ == "pressurelossvarious"
    assert s.__class__.__qualname__ == "PressureLoss0D"
    assert s.name == "p1"
    assert s.parent is None
    assert len(s.children) == 0

    assert_keys(s.inputs, "inwards", "modevars_in", "flnum_in")
    assert isinstance(s.inputs["inwards"], ExtensiblePort)
    assert isinstance(s.inputs["modevars_in"], ModeVarPort)
    assert isinstance(s.inputs["flnum_in"], NumPort)

    assert_keys(s.outputs, "outwards", "modevars_out", "flnum_out")
    assert isinstance(s.outputs["outwards"], ExtensiblePort)
    assert isinstance(s.outputs["modevars_out"], ModeVarPort)
    assert isinstance(s.outputs["flnum_out"], NumPort)

    # Load simple module with boundaries
    config = StringIO(
        """{
            "$schema": "0-2-0/system.schema.json",
            "p1": {
            "class": "pressurelossvarious.PressureLoss0D",
            "inputs": {
                "flnum_in.Pt": 1000000.0,
                "flnum_in.W": 10.0
            }
            }}"""
    )
    s = System.load(config)

    assert s.__module__ == "pressurelossvarious"
    assert s.__class__.__qualname__ == "PressureLoss0D"
    assert s.name == "p1"
    assert s.parent is None
    assert len(s.children) == 0

    assert_keys(s.inputs, "inwards", "modevars_in", "flnum_in")
    assert isinstance(s.inputs["inwards"], ExtensiblePort)
    assert isinstance(s.inputs["modevars_in"], ModeVarPort)
    port = s.inputs["flnum_in"]
    assert isinstance(port, NumPort)
    assert port.Pt == 1e6
    assert port.W == 10

    assert_keys(s.outputs, "outwards", "modevars_out", "flnum_out")
    assert isinstance(s.outputs["outwards"], ExtensiblePort)
    assert isinstance(s.outputs["modevars_out"], ModeVarPort)
    assert isinstance(s.outputs["flnum_out"], NumPort)

    # Load simple module with properties
    config = StringIO(
        """{
            "$schema": "0-2-0/system.schema.json",
            "alltype": {
            "class": "vectors.AllTypesSystem",
            "properties": {
                "dimension": 3
            }
            }}"""
    )
    s = System.load(config)

    assert s.__module__ == "vectors"
    assert s.__class__.__qualname__ == "AllTypesSystem"
    assert s.name == "alltype"
    assert s.parent is None
    assert len(s.children) == 0

    assert s.properties ==  {"dimension": 3}
    assert s.dimension == 3

    assert_keys(s.inputs, "inwards", "modevars_in", "in_")
    assert isinstance(s.inputs["inwards"], ExtensiblePort)
    assert isinstance(s.inputs["modevars_in"], ModeVarPort)
    port = s.inputs["in_"]
    assert isinstance(port, V1dPort)

    assert_keys(s.outputs, "outwards", "modevars_out", "out")
    assert isinstance(s.outputs["outwards"], ExtensiblePort)
    assert isinstance(s.outputs["modevars_out"], ModeVarPort)
    assert isinstance(s.outputs["out"], V1dPort)

    # Load module in module - test for connector from submodule to top system
    #   Pushing port is the only possibility - pulling port is forbidden
    config = StringIO(
        """{
        "$schema": "0-2-0/system.schema.json",
        "p1": {
            "class": "pressurelossvarious.PressureLossSys",
            "subsystems": {
            "p11": {
                "class": "pressurelossvarious.PressureLoss0D"
            }
            },
            "connections": [
            ["flnum_in", "p11.flnum_in"],
            ["p11.flnum_out", "flnum_out"]
            ],
            "exec_order": ["p11"]
        }}"""
    )
    s = System.load(config)

    # check parent
    assert s.__module__ == "pressurelossvarious"
    assert s.__class__.__qualname__ == "PressureLossSys"
    assert s.name == "p1"
    assert s.parent is None
    assert_keys(s.children, "p11")
    for child in s.children.values():
        assert isinstance(child, System)
        assert child.parent is s
    assert list(s.exec_order) == ["p11"]

    assert_keys(s.inputs, "inwards", "modevars_in", "flnum_in")
    assert isinstance(s.inputs["inwards"], ExtensiblePort)
    assert isinstance(s.inputs["flnum_in"], NumPort)

    assert_keys(s.outputs, "outwards", "modevars_out", "flnum_out")
    assert isinstance(s.outputs["outwards"], ExtensiblePort)
    assert isinstance(s.outputs["flnum_out"], NumPort)

    # check child
    child = s.p11
    assert child.__module__ == "pressurelossvarious"
    assert child.__class__.__qualname__ == "PressureLoss0D"
    assert child.name == "p11"
    assert child.parent is s
    assert len(child.children) == 0  # too young, presumably

    assert_keys(child.inputs, "inwards", "modevars_in", "flnum_in")
    assert isinstance(child.inputs["inwards"], ExtensiblePort)
    assert isinstance(child.inputs["modevars_in"], ModeVarPort)
    assert isinstance(child.inputs["flnum_in"], NumPort)

    assert_keys(child.outputs, "outwards", "modevars_out", "flnum_out")
    assert isinstance(child.outputs["outwards"], ExtensiblePort)
    assert isinstance(child.outputs["modevars_out"], ModeVarPort)
    assert isinstance(child.outputs["flnum_out"], NumPort)

    connectors = s.connectors()
    assert set(connectors) == {
        'flnum_in -> p11.flnum_in',
        'p11.flnum_out -> flnum_out',
    }
    
    connector = connectors['flnum_in -> p11.flnum_in']
    assert connector.source is s.flnum_in
    assert connector.sink is child.flnum_in

    connector = connectors['p11.flnum_out -> flnum_out']
    assert connector.source is child.flnum_out
    assert connector.sink is s.flnum_out

    # Load 2 modules in module - test for connector from submodule to top system
    #   Pushing port is the only possibility - pulling port is forbidden
    config = StringIO(
        """{
        "$schema": "0-2-0/system.schema.json",
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
    s = System.load(config)

    # check parent
    assert s.__module__ == "pressurelossvarious"
    assert s.__class__.__qualname__ == "PressureLossSys"
    assert s.name == "p1"
    assert s.parent is None
    assert_keys(s.children, "p11", "p12")
    assert list(s.exec_order) == ["p11", "p12"]
    for child in s.children.values():
        assert isinstance(child, System)
        assert child.parent is s

    assert_keys(s.inputs, "inwards", "modevars_in", "flnum_in")
    assert isinstance(s.inputs["inwards"], ExtensiblePort)
    assert isinstance(s.inputs["modevars_in"], ModeVarPort)
    assert isinstance(s.inputs["flnum_in"], NumPort)

    assert_keys(s.outputs, "outwards", "modevars_out", "flnum_out")
    assert isinstance(s.outputs["outwards"], ExtensiblePort)
    assert isinstance(s.outputs["modevars_out"], ModeVarPort)
    assert isinstance(s.outputs["flnum_out"], NumPort)

    # check children
    child = s.p11
    assert child.__module__ == "pressurelossvarious"
    assert child.__class__.__qualname__ == "PressureLoss0D"
    assert child.name == "p11"
    assert child.parent is s
    assert len(child.children) == 0

    assert_keys(child.inputs, "inwards", "modevars_in", "flnum_in")
    assert isinstance(child.inputs["inwards"], ExtensiblePort)
    assert isinstance(child.inputs["modevars_in"], ModeVarPort)
    assert isinstance(child.inputs["flnum_in"], NumPort)

    assert_keys(child.outputs, "outwards", "modevars_out", "flnum_out")
    assert isinstance(child.outputs["outwards"], ExtensiblePort)
    assert isinstance(child.outputs["modevars_out"], ModeVarPort)
    assert isinstance(child.outputs["flnum_out"], NumPort)

    child = s.p12
    assert child.__module__ == "pressurelossvarious"
    assert child.__class__.__qualname__ == "PressureLoss0D"
    assert child.name == "p12"
    assert child.parent is s
    assert len(child.children) == 0

    assert_keys(child.inputs, "inwards", "modevars_in", "flnum_in")
    assert isinstance(child.inputs["inwards"], ExtensiblePort)
    assert isinstance(child.inputs["modevars_in"], ModeVarPort)
    assert isinstance(child.inputs["flnum_in"], NumPort)

    assert_keys(child.outputs, "outwards", "modevars_out", "flnum_out")
    assert isinstance(child.outputs["outwards"], ExtensiblePort)
    assert isinstance(child.outputs["modevars_out"], ModeVarPort)
    assert isinstance(child.outputs["flnum_out"], NumPort)

    # check connectors
    connectors = s.connectors()
    assert set(connectors) == {
        "flnum_in -> p11.flnum_in",
        "p11.flnum_out -> p12.flnum_in",
        "p12.flnum_out -> flnum_out",
    }

    connector = connectors["flnum_in -> p11.flnum_in"]
    assert connector.source is s.flnum_in
    assert connector.sink is s.p11.flnum_in

    connector = connectors["p11.flnum_out -> p12.flnum_in"]
    assert connector.source is s.p11.flnum_out
    assert connector.sink is s.p12.flnum_in

    connector = connectors["p12.flnum_out -> flnum_out"]
    assert connector.source is s.p12.flnum_out
    assert connector.sink is s.flnum_out


@pytest.mark.parametrize("name, expected_name", [
    (None, "p1"),
    ("s", "s"),
])
def test_System_load_rename(test_library, name, expected_name):
    """Test `System.load` with specified output name"""
    config = StringIO(
        """{
            "$schema": "0-2-0/system.schema.json",
            "p1": {
            "class": "pressurelossvarious.PressureLoss0D",
            "inputs": {
                "flnum_in.Pt": 1000000.0,
                "flnum_in.W": 10.0
            }
            }}"""
    )
    s = System.load(config, name=name)

    assert s.__module__ == "pressurelossvarious"
    assert s.__class__.__qualname__ == "PressureLoss0D"
    assert s.name == expected_name
    assert s.parent is None
    assert len(s.children) == 0

    assert set(s.inputs) == {"inwards", "modevars_in", "flnum_in"}
    assert isinstance(s.inputs["inwards"], ExtensiblePort)
    assert isinstance(s.inputs["modevars_in"], ModeVarPort)
    port = s.inputs["flnum_in"]
    assert isinstance(port, NumPort)
    assert port.Pt == 1e6
    assert port.W == 10

    assert set(s.outputs) == {"outwards", "modevars_out", "flnum_out"}
    assert isinstance(s.outputs["outwards"], ExtensiblePort)
    assert isinstance(s.outputs["modevars_out"], ModeVarPort)
    assert isinstance(s.outputs["flnum_out"], NumPort)


def test_System_load_from_dict(test_library):
    # Load super simple module
    d = {"p1": {"class": "pressurelossvarious.PressureLoss0D"}}
    name, param = d.popitem()
    s = System.load_from_dict(name, param)

    assert s.__module__ == "pressurelossvarious"
    assert s.__class__.__qualname__ == "PressureLoss0D"
    assert s.name == "p1"
    assert s.parent is None
    assert len(s.children) == 0
    assert len(s.exec_order) == 0

    assert_keys(s.inputs, "inwards", "modevars_in", "flnum_in")
    assert isinstance(s.inputs["inwards"], ExtensiblePort)
    assert isinstance(s.inputs["flnum_in"], NumPort)

    assert_keys(s.outputs, "outwards", "modevars_out", "flnum_out")
    assert isinstance(s.outputs["outwards"], ExtensiblePort)
    assert isinstance(s.outputs["flnum_out"], NumPort)

    # Load simple module with boundaries
    d = {
        "p1": {
            "class": "pressurelossvarious.PressureLoss0D",
            "inputs": {"flnum_in.Pt": 1000000.0, "flnum_in.W": 10.0},
        }
    }
    name, param = d.popitem()
    s = System.load_from_dict(name, param)

    assert s.__module__ == "pressurelossvarious"
    assert s.__class__.__qualname__ == "PressureLoss0D"
    assert s.name == "p1"
    assert s.parent is None
    assert len(s.children) == 0
    assert len(s.exec_order) == 0

    assert_keys(s.inputs, "inwards", "modevars_in", "flnum_in")
    assert isinstance(s.inputs["inwards"], ExtensiblePort)
    assert isinstance(s.inputs["flnum_in"], NumPort)
    assert s.flnum_in.Pt == 1e6
    assert s.flnum_in.W == 10

    assert_keys(s.outputs, "outwards", "modevars_out", "flnum_out")
    assert isinstance(s.outputs["outwards"], ExtensiblePort)
    assert isinstance(s.outputs["flnum_out"], NumPort)

    # Load module in module - test for connector from submodule to top system
    #   Pushing port is the only possibility - pulling port is forbidden
    d = {
        "p1": {
            "class": "pressurelossvarious.PressureLossSys",
            "subsystems": {"p11": {"class": "pressurelossvarious.PressureLoss0D"}},
            "connections": [
                ["flnum_in", "p11.flnum_in"],
                ["p11.flnum_out", "flnum_out"],
            ],
            "exec_order": ["p11"],
        }
    }
    name, param = d.popitem()
    s = System.load_from_dict(name, param)

    # check parent
    assert s.__module__ == "pressurelossvarious"
    assert s.__class__.__qualname__ == "PressureLossSys"
    assert s.name == "p1"
    assert s.parent is None
    assert_keys(s.children, "p11")
    assert list(s.exec_order) == ["p11"]
    for child in s.children.values():
        assert isinstance(child, System)
        assert child.parent is s

    assert_keys(s.inputs, "inwards", "modevars_in", "flnum_in")
    assert isinstance(s.inputs["inwards"], ExtensiblePort)
    assert isinstance(s.inputs["flnum_in"], NumPort)

    assert_keys(s.outputs, "outwards", "modevars_out", "flnum_out")
    assert isinstance(s.outputs["outwards"], ExtensiblePort)
    assert isinstance(s.outputs["flnum_out"], NumPort)

    # check child
    child = s.p11
    assert child.__module__ == "pressurelossvarious"
    assert child.__class__.__qualname__ == "PressureLoss0D"
    assert child.name == "p11"
    assert child.parent is s
    assert len(child.children) == 0

    assert_keys(child.inputs, "inwards", "modevars_in", "flnum_in")
    assert isinstance(child.inputs["inwards"], ExtensiblePort)
    assert isinstance(child.inputs["flnum_in"], NumPort)

    assert_keys(child.outputs, "outwards", "modevars_out", "flnum_out")
    assert isinstance(child.outputs["outwards"], ExtensiblePort)
    assert isinstance(child.outputs["flnum_out"], NumPort)

    # check connectors
    connectors = s.connectors()
    assert set(connectors) == {
        "flnum_in -> p11.flnum_in",
        "p11.flnum_out -> flnum_out",
    }

    connector = connectors["flnum_in -> p11.flnum_in"]
    assert connector.source is s.flnum_in
    assert connector.sink is s.p11.flnum_in

    connector = connectors["p11.flnum_out -> flnum_out"]
    assert connector.source is s.p11.flnum_out
    assert connector.sink is s.flnum_out

    # Load 2 modules in module - test for connector from submodule to top system
    #   Pushing port is the only possibility - pulling port is forbidden
    d = {
        "p1": {
            "class": "pressurelossvarious.PressureLossSys",
            "subsystems": {
                "p11": {"class": "pressurelossvarious.PressureLoss0D"},
                "p12": {"class": "pressurelossvarious.PressureLoss0D"},
            },
            "connections": [
                ["flnum_in", "p11.flnum_in"],
                ["p11.flnum_out", "p12.flnum_in"],
                ["p12.flnum_out", "flnum_out"],
            ],
            "exec_order": ["p11", "p12"],
        }
    }
    name, param = d.popitem()
    s = System.load_from_dict(name, param)

    # check parent
    assert s.__module__ == "pressurelossvarious"
    assert s.__class__.__qualname__ == "PressureLossSys"
    assert s.name == "p1"
    assert s.parent is None
    assert_keys(s.children, "p11", "p12")
    assert list(s.exec_order) == ["p11", "p12"]
    for child in s.children.values():
        assert isinstance(child, System)
        assert child.parent is s

    assert_keys(s.inputs, "inwards", "modevars_in", "flnum_in")
    assert isinstance(s.inputs["inwards"], ExtensiblePort)
    assert isinstance(s.inputs["flnum_in"], NumPort)

    assert_keys(s.outputs, "outwards", "modevars_out", "flnum_out")
    assert isinstance(s.outputs["outwards"], ExtensiblePort)
    assert isinstance(s.outputs["flnum_out"], NumPort)

    # check children
    child = s.p11
    assert child.__module__ == "pressurelossvarious"
    assert child.__class__.__qualname__ == "PressureLoss0D"
    assert child.name == "p11"
    assert child.parent is s
    assert len(child.children) == 0

    assert_keys(child.inputs, "inwards", "modevars_in", "flnum_in")
    assert isinstance(child.inputs["inwards"], ExtensiblePort)
    assert isinstance(child.inputs["flnum_in"], NumPort)

    assert_keys(child.outputs, "outwards", "modevars_out", "flnum_out")
    assert isinstance(child.outputs["outwards"], ExtensiblePort)
    assert isinstance(child.outputs["flnum_out"], NumPort)

    child = s.p12
    assert child.__module__ == "pressurelossvarious"
    assert child.__class__.__qualname__ == "PressureLoss0D"
    assert child.name == "p12"
    assert child.parent is s
    assert len(child.children) == 0

    assert_keys(child.inputs, "inwards", "modevars_in", "flnum_in")
    assert isinstance(child.inputs["inwards"], ExtensiblePort)
    assert isinstance(child.inputs["flnum_in"], NumPort)

    assert_keys(child.outputs, "outwards", "modevars_out", "flnum_out")
    assert isinstance(child.outputs["outwards"], ExtensiblePort)
    assert isinstance(child.outputs["flnum_out"], NumPort)

    # check connectors
    connectors = s.connectors()
    assert set(connectors) == {
        "flnum_in -> p11.flnum_in",
        "p11.flnum_out -> p12.flnum_in",
        "p12.flnum_out -> flnum_out",
    }

    connector = connectors["flnum_in -> p11.flnum_in"]
    assert connector.source is s.flnum_in
    assert connector.sink is s.p11.flnum_in

    connector = connectors["p11.flnum_out -> p12.flnum_in"]
    assert connector.source is s.p11.flnum_out
    assert connector.sink is s.p12.flnum_in

    connector = connectors["p12.flnum_out -> flnum_out"]
    assert connector.source is s.p12.flnum_out
    assert connector.sink is s.flnum_out

    # Erroneous cases
    d = {"p1": {"class": "pressurelossvarious"}}
    name, param = d.popitem()
    with pytest.raises(AttributeError):
        s = System.load_from_dict(name, param)

    d = {"p1": {"class": 1.0}}
    name, param = d.popitem()
    with pytest.raises(TypeError):
        s = System.load_from_dict(name, param)

    d = {"p1": {"class": "pressurelossvarious.xx"}}
    name, param = d.popitem()
    with pytest.raises(AttributeError):
        s = System.load_from_dict(name, param)

    d = {"p1": {"class": "pressurelossvarious.FalseSystem"}}
    name, param = d.popitem()
    with pytest.raises(AttributeError):
        s = System.load_from_dict(name, param)


def test_System_serialize_with_None(tmp_path):
    original = SystemWithNone('orig')
    filename = tmp_path/"original.json"

    with open(filename, "w") as fp:
        original.save(fp)

    loaded = System.load(filename, name='loaded')

    assert isinstance(loaded, type(original))
    assert loaded.name == 'loaded'
    assert loaded.x is None


def test_System_to_dict(test_library, config):

    s = System.load(config)

    d = s.to_dict()
    assert_keys(d, "p1")
    entry = d["p1"]
    assert isinstance(entry, dict)
    assert set(entry.keys()) == {
        "class", "inputs", "subsystems", "connections", "exec_order",
    }
    assert entry["class"] == "pressurelossvarious.PressureLossSys"
    assert entry["subsystems"]["p11"]["class"] == "pressurelossvarious.PressureLoss0D"
    assert entry["subsystems"]["p12"]["class"] == "pressurelossvarious.PressureLoss0D"
    assert set(entry["connections"]) == {
        ("p11.flnum_in", "flnum_in"),
        ("p12.flnum_in", "p11.flnum_out"),
        ("flnum_out", "p12.flnum_out"),
    }
    assert entry["exec_order"] == ["p11", "p12"]

    # Test partial connection
    config2 = StringIO(
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
                ["p11.flnum_in", "flnum_in"],
                ["p11.inwards", "inwards", {"K": "K11"}],
                ["p12.flnum_in", "p11.flnum_out"],
                ["flnum_out", "p12.flnum_out"],
                ["outwards", "p12.outwards", {"delta_p12": "delta_p"}]
            ],
            "exec_order": ["p11", "p12"]
        }}"""
    )
    s = System.load(config2)
    assert "delta_p12" in s.outwards

    d = s.to_dict()
    entry = d["p1"]["connections"]
    assert entry == [
        ('flnum_out', 'p12.flnum_out'),
        ('outwards', 'p12.outwards', {'delta_p12': 'delta_p'}),
        ('p11.flnum_in', 'flnum_in'),
        ('p11.inwards', 'inwards', {'K': 'K11'}),
        ('p12.flnum_in', 'p11.flnum_out'),
    ]


def test_System_to_dict_with_def(test_library, config):

    s = System.load(config)   
    d  = s._System__to_dict(True)
    assert_keys(d, "p1")
    entry = d["p1"]
    assert set(entry["inputs"].keys()) == {
        "flnum_in", "inwards", "modevars_in",
    }
    assert entry["inputs"]["inwards"] == {'K11': {'value': 100.0}}
    assert entry["inputs"]["modevars_in"] == {'__class__': 'ModeVarPort'}
    assert entry["inputs"]["flnum_in"]["__class__"] == 'NumPort'

    assert set(entry["outputs"].keys()) == {
        "flnum_out", "outwards", "modevars_out",
    }
    assert entry["outputs"]["outwards"] == {'delta_p12': {'value': 0.}}
    assert entry["outputs"]["modevars_out"] == {'__class__': 'ModeVarPort'}
    assert entry["outputs"]["flnum_out"]["__class__"] == 'NumPort'


def test_System_to_dict_with_port_def(test_library, config):

    s = System.load(config) 
    port_cls_data = {}  
    d  = s._System__to_dict(True, port_cls_data)
    assert_keys(port_cls_data, "NumPort")
    entry = port_cls_data["NumPort"]
    assert set(entry.keys()) == {"Pt", "W"}
    assert entry["Pt"] == {
        "value": 101325.0,
        "unit" : "Pa"
    }


def test_System_export_system(test_library, config):

    s = System.load(config) 
    d  = s.export_structure()
    assert set(d.keys()) == {"Ports", "Systems"}


def test_System_tojson(test_library):
    config = StringIO(
            """{
  "$schema": "0-3-0/system.schema.json",
  "p1": {
    "class": "pressurelossvarious.PressureLossSys",
    "connections": [
      [
        "flnum_out",
        "p12.flnum_out"
      ],
      [
        "p11.flnum_in",
        "flnum_in"
      ],
      [
        "p12.flnum_in",
        "p11.flnum_out"
      ]
    ],
    "exec_order": [
      "p11",
      "p12"
    ],
    "inputs": {
      "flnum_in.Pt": 101325.0,
      "flnum_in.W": 1.0,
      "inwards.K11": 100.0
    },
    "subsystems": {
      "p11": {
        "class": "pressurelossvarious.PressureLoss0D",
        "inputs": {
          "flnum_in.Pt": 101325.0,
          "flnum_in.W": 1.0,
          "inwards.K": 100.0
        }
      },
      "p12": {
        "class": "pressurelossvarious.PressureLoss0D",
        "inputs": {
          "flnum_in.Pt": 101325.0,
          "flnum_in.W": 1.0,
          "inwards.K": 100.0
        }
      }
    }
  }
}"""
    )
    s = System.load(config)
    config.seek(0)

    j = s.to_json(sort_keys=True)

    assert j == config.read()


def test_System_AllTypesSystem_serialization():
    original = AllTypesSystem("original")
    original.in_.x = np.r_[0.1, 0.2, 0.3]
    original.a = np.r_[0.3, 0.2, 0.1]
    original.b = np.r_[1.1, 2.2, 3.3]
    original.c = 2.5
    original.e = "John"

    data = original.to_dict()
    s = System.load_from_dict('loaded', data['original'])

    assert s.__module__ == "cosapp.tests.library.systems.vectors"
    assert s.__class__.__qualname__ == "AllTypesSystem"
    assert s.name == "loaded"
    assert s.parent is None
    assert len(s.children) == 0

    assert s.properties == {"dimension": 3}
    assert s.dimension == 3

    assert_keys(s.inputs, "inwards", "modevars_in", "in_")
    assert isinstance(s.inputs["inwards"], ExtensiblePort)
    port = s.inputs["in_"]
    assert isinstance(port, V1dPort)
    assert port.x == pytest.approx(original.in_.x, abs=0)
    assert s.a == pytest.approx(original.a, abs=0)
    assert s.b == pytest.approx(original.b, abs=0)
    assert s.c == original.c
    assert s.e == original.e

    assert_keys(s.outputs, "outwards", "modevars_out", "out")
    assert isinstance(s.outputs["outwards"], ExtensiblePort)
    assert isinstance(s.outputs["out"], V1dPort)


def test_System_property_to_json():
    """Test method `to_json` with a system containing properties.
    Related to https://gitlab.com/cosapp/cosapp/-/issues/95
    """
    original = SystemWithProps('original')

    # Serialization using dictionaries
    data = original.to_dict()

    assert set(data) == {original.name}
    assert set(data[original.name]) == {
        'class',
        'properties',
    }
    properties = data[original.name]['properties']
    assert set(properties) == {'n', 'g', 'c'}
    assert properties['n'] == original.n
    assert properties['g'] == original.g
    assert isinstance(properties['c'], SomeClass)
    assert np.array_equal(properties['c'].x, [-1, 0, 1])

    loaded = System.load_from_dict('loaded', data[original.name])

    assert isinstance(loaded, SystemWithProps)
    assert loaded.n == original.n
    assert loaded.g == original.g
    assert isinstance(loaded.c, SomeClass)
    assert np.array_equal(loaded.c.x, original.c.x)

    # Serialization using `System.to_json``
    jstr = original.to_json()
    other = json.loads(jstr)
    assert isinstance(other, dict)
    assert set(other) == {original.name, '$schema'}
    assert set(other[original.name]) == set(data[original.name])
    assert set(other[original.name]['properties']) == set(properties)


def test_System_property_serialization(tmp_path):
    """Test serialization of a system containing properties.
    """
    original = System('original')
    original.add_child(SystemWithProps('sub'))

    filepath = tmp_path / 'original.json'

    original.save(filepath)
    loaded = System.load(filepath, name='loaded')

    assert loaded.name == 'loaded'
    assert isinstance(loaded.sub, type(original.sub))
    assert loaded.sub.n == original.sub.n
    assert loaded.sub.g == original.sub.g
    assert isinstance(loaded.sub.c, SomeClass)
    assert np.array_equal(loaded.sub.c.x, original.sub.c.x)


def test_System_ctor_args_serialization(tmp_path):
    """Test serialization of systems containing ctor parameters.
    """
    original = System('original')
    original.add_child(SystemWithKwargs('s1', n=3))
    original.add_child(SystemWithKwargs('s2', n=6, r=0.1))
    assert original.s1.n == 3
    assert original.s2.n == 6
    assert original.s2.r == 0.1
    assert not hasattr(original.s1, 'r')
    original.s1.v[:] = np.arange(3, dtype=float)
    original.s2.v[:] = np.linspace(0, 1, original.s2.n)

    filepath = tmp_path / 'original.json'

    original.save(filepath)
    loaded = System.load(filepath, name='loaded')

    assert loaded.name == 'loaded'
    assert isinstance(loaded.s1, type(original.s1))
    assert isinstance(loaded.s2, type(original.s2))
    assert loaded.s1.n == 3
    assert loaded.s2.n == 6
    assert loaded.s2.r == 0.1
    assert not hasattr(loaded.s1, 'r')
    assert loaded.s1.v is not original.s1.v
    assert loaded.s2.v is not original.s2.v
    assert loaded.s1.v == pytest.approx([0, 1, 2], abs=0)
    assert loaded.s2.v == pytest.approx([0, 0.2, 0.4, 0.6, 0.8, 1], abs=1e-15)
