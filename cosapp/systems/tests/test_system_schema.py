import json
import os
import unittest

import jsonschema

from cosapp import systems


class SystemSchemaTestCase(unittest.TestCase):
    def setUp(self):
        self.curdir = os.path.dirname(os.path.realpath(systems.__file__))

        with open(os.path.join(self.curdir, "./system.schema.json")) as fp:
            self.schema = json.load(fp)

    def test_system_config(self):
        # Working test - data/system_config.json
        with open(os.path.join(self.curdir, "../tests/data/system_config.json")) as fp:
            test = json.load(fp)
        self.assertIsNone(jsonschema.validate(test, self.schema))

        # Working test - data/system_config_ducts.json
        with open(
            os.path.join(self.curdir, "../tests/data/system_config_ducts.json")
        ) as fp:
            test = json.load(fp)
        self.assertIsNone(jsonschema.validate(test, self.schema))

    def test_full_definition(self):
        # Minimal configuration file
        config = {"mySystem": {"class": "SuperSystem"}}
        self.assertIsNone(jsonschema.validate(config, self.schema))

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
        self.assertIsNone(jsonschema.validate(config, self.schema))

        # Minimal configuration file with subsystem
        config = {
            "mySystem": {
                "class": "SuperSystem",
                "subsystems": {"mySubsystem": {"class": "Subsystem"}},
            }
        }
        self.assertIsNone(jsonschema.validate(config, self.schema))

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
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(config, self.schema)

    def test_name(self):
        # Invalid name
        config = {
            "": {"class": "SuperSystem"}  # Empty name not allowed
        }
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(config, self.schema)

        # Invalid name
        config = {
            "2mySystem": {  # First character must be a letter
                "class": "SuperSystem"
            }
        }
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(config, self.schema)

    def test_class(self):
        # Missing class
        config = {"mySystem": {}}
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(config, self.schema)

        # Invalid class name
        config["mySystem"]["class"] = "2SuperSystem"  # First character must be a letter
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(config, self.schema)

        # Multi level class name
        config["mySystem"]["class"] = "SuperSystem.MySystem"
        self.assertIsNone(jsonschema.validate(config, self.schema))
        config["mySystem"]["class"] = "SuperSystem.MySystem.MyField"
        self.assertIsNone(jsonschema.validate(config, self.schema))

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
        self.assertIsNone(jsonschema.validate(config, self.schema))

        # Boundary name should start with a letter (lower or upper)
        config["mySystem"]["inputs"] = {"2int": 42}
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(config, self.schema)

        # Not supported type
        config["mySystem"]["inputs"] = {"object": object()}
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(config, self.schema)

        # Not supported empty string
        config["inputs"] = {"empty_str": ""}
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(config, self.schema)

        # Not supported empty string in array
        config["mySystem"]["inputs"] = {"empty_str": ["Super", "", "man"]}
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(config, self.schema)

    def test_connections(self):
        config = {
            "mySystem": {
                "class": "SuperSystem",
                "connections": [["myinput", "module.myoutput"]],
            }
        }
        # if connections, subsystems must be defined
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(config, self.schema)
        
        config["mySystem"]["subsystems"] = {}
        self.assertIsNone(jsonschema.validate(config, self.schema))

        # Port name should start with a letter (upper or lower)
        config["mySystem"]["connections"] = [["2myinput", "module.myoutput"]]
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(config, self.schema)

        # Port name should start with a letter (upper or lower)
        config["mySystem"]["connections"] = [["myinput", "2module.myoutput"]]
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(config, self.schema)

        # Port name should start with a letter (upper or lower)
        config["mySystem"]["connections"] = [["myinput", "module.2myoutput"]]
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(config, self.schema)

        # Connection value should be not empty string
        config["mySystem"]["connections"] = [["myinput", ""]]
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(config, self.schema)

        # Connection value cannot be to a submodule
        config["mySystem"]["connections"] = [["myinput", "module.subsubmodule.itsoutput"]]
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(config, self.schema)

        # Connection to the top module is not allowed
        config["mySystem"]["connections"] = [["myinput", "..itsoutput.T"]]
        with self.assertRaises(jsonschema.ValidationError):
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
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(config, self.schema)

        config["mySystem"]["subsystems"] = {
            "mysystem": {"class": "System1"}
        }
        self.assertIsNone(jsonschema.validate(config, self.schema))

        config["mySystem"]["exec_order"].append("mysystem")
        self.assertIsNone(jsonschema.validate(config, self.schema))
