import json
import os
import pytest

import jsonschema

from cosapp import utils


class TestSystemSchema_0_3_0:
    def setup_method(self):
        self.curdir = os.path.dirname(os.path.realpath(utils.__file__))

        with open(os.path.join(self.curdir, "./0-3-0_system.schema.json")) as fp:
            self.schema = json.load(fp)

    def test_system_config(self):
        # Working test - data/system_config.json
        with open(os.path.join(self.curdir, "../tests/data/system_config.json")) as fp:
            test = json.load(fp)
        assert jsonschema.validate(test, self.schema) is None

        # Working test - data/system_config_ducts.json
        with open(
            os.path.join(self.curdir, "../tests/data/system_config_ducts.json")
        ) as fp:
            test = json.load(fp)
        assert jsonschema.validate(test, self.schema) is None

    def test_full_definition(self):
        # Minimal configuration file
        config = {"mySystem": {"class": "SuperSystem"}}
        assert jsonschema.validate(config, self.schema) is None

        # Configuration file - all keywords
        config = {
            "mySystem": {
                "class": "SuperSystem",
                "inputs": {},
                "connections": [],
                "subsystems": {},
                "exec_order": [],
            }
        }
        assert jsonschema.validate(config, self.schema) is None

        # Minimal configuration file with subsystem
        config = {
            "mySystem": {
                "class": "SuperSystem",
                "subsystems": {"mySubsystem": {"class": "Subsystem"}},
            }
        }
        assert jsonschema.validate(config, self.schema) is None

        # Unknown properties
        config = {
            "mySystem": {
                "class": "SuperSystem",
                "inputs": {},
                "connections": [],
                "subsystems": {},
                "myproperties": {},
            }
        }
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(config, self.schema)

    def test_name(self):
        # Invalid name
        config = {
            "": {"class": "SuperSystem"}  # Empty name not allowed
        }
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(config, self.schema)

        # Invalid name
        config = {
            "2mySystem": {  # First character must be a letter
                "class": "SuperSystem"
            }
        }
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(config, self.schema)

    def test_class(self):
        # Missing class
        config = {"mySystem": {}}
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(config, self.schema)

        # Invalid class name
        config["mySystem"]["class"] = "2SuperSystem"  # First character must be a letter
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(config, self.schema)

        # Multi level class name
        config["mySystem"]["class"] = "SuperSystem.MySystem"
        assert jsonschema.validate(config, self.schema) is None
        config["mySystem"]["class"] = "SuperSystem.MySystem.MyField"
        assert jsonschema.validate(config, self.schema) is None

    def test_inputs(self):
        # All possible boundaries type
        config = {
            "mySystem": {
                "class": "SuperSystem",
                "inputs": {
                    "integer": 42,
                    "boolean": True,
                    "float": 31416e-4,
                    "string": "Neo",
                    "array_int": [1, 2, 3],
                    "array_bool": [True, False, True],
                    "array_float": [3.1416, 2.72],
                    "array_str": ["Sarah", "O'Connor"],
                },
            }
        }
        assert jsonschema.validate(config, self.schema) is None

        # Boundary name should start with a letter (lower or upper)
        config["mySystem"]["inputs"] = {"2int": 42}
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(config, self.schema)

        # Not supported type
        config["mySystem"]["inputs"] = {"object": object()}
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(config, self.schema)

        # Not supported empty string
        config["inputs"] = {"empty_str": ""}
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(config, self.schema)

        # Not supported empty string in array
        config["mySystem"]["inputs"] = {"empty_str": ["Super", "", "man"]}
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(config, self.schema)

    def test_connections(self):
        config = {
            "mySystem": {
                "class": "SuperSystem",
                "connections": [["myinput", "module.myoutput"]],
            }
        }
        # if connections, subsystems must be defined
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(config, self.schema)
        
        config["mySystem"]["subsystems"] = {}
        assert jsonschema.validate(config, self.schema) is None

        # Port name should start with a letter (upper or lower)
        config["mySystem"]["connections"] = [["2myinput", "module.myoutput"]]
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(config, self.schema)

        # Port name should start with a letter (upper or lower)
        config["mySystem"]["connections"] = [["myinput", "2module.myoutput"]]
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(config, self.schema)

        # Port name should start with a letter (upper or lower)
        config["mySystem"]["connections"] = [["myinput", "module.2myoutput"]]
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(config, self.schema)

        # Connection value should be not empty string
        config["mySystem"]["connections"] = [["myinput", ""]]
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(config, self.schema)

        # Connection value cannot be to a submodule
        config["mySystem"]["connections"] = [["myinput", "module.subsubmodule.itsoutput"]]
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(config, self.schema)

        # Connection to the top module is not allowed
        config["mySystem"]["connections"] = [["myinput", "..itsoutput.T"]]
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(config, self.schema)

    def test_exec_order(self):
        # This is not possible to define a execution order if no subsystems are defined
        config = {
            "mySystem": {
                "class": "SuperSystem",
                "inputs": {},
                "connections": [],
                "exec_order": [],
            }
        }
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(config, self.schema)

        config["mySystem"]["subsystems"] = {
            "mysystem": {"class": "System1"}
        }
        assert jsonschema.validate(config, self.schema) is None

        config["mySystem"]["exec_order"].append("mysystem")
        assert jsonschema.validate(config, self.schema) is None
