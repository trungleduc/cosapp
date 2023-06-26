import json
import logging
import os, re
from io import StringIO
from pathlib import Path
from tempfile import gettempdir
from unittest import mock

import pytest
from jsonschema import ValidationError
from typing import Callable

from cosapp.core import config
from cosapp.core.config import CoSAppConfiguration

import functools


here = os.path.dirname(os.path.abspath(config.__file__))


def patch_env(env: dict):
    """Decorator reversibly patching `os.environ`, with no side effect."""
    original = os.environ.copy()
    def deco(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            os.environ.update(env)
            try:
                return func(*args, **kwargs)
            finally:
                os.environ = original
        return wrapper
    return deco


def matcher(pattern: str) -> Callable[[str], bool]:
    """Utility function returning a match test function."""
    return lambda message: re.match(pattern, message)


@pytest.fixture
def fake_schema():
    with open(os.path.join(here, "configuration_schema.json")) as f:
        schema = f.read()

    class FakeSchema:
        def __init__(self, *args, **kwargs):
            self.f = StringIO(schema)

        def __enter__(self):
            return self.f

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.f.close()

    return FakeSchema


def test_CoSAppConfiguration__init__(fake_schema):
    fake_param_str = json.dumps(
        {
            "userid": "id1234",
            "roles": [
                ["scope1", "scope2"],
                ["scope3", "scope4"],
            ],
        }
    )
    fake_io = StringIO(fake_param_str)

    with mock.patch("builtins.open") as mock_open:

        def fake_open(file, mode="r", **kwargs):
            if str(file).endswith("configuration_schema.json"):
                return fake_schema()
            if "r" in mode:
                return fake_io
            else:
                return mock.DEFAULT

        mock_open.side_effect = fake_open  # Emulate access to configuration file

        config = CoSAppConfiguration()
        assert config.userid == "id1234"
        assert {"scope1", "scope2"} in config.roles
        assert {"scope3", "scope4"} in config.roles


def test_CoSAppConfiguration_userid():
    config = CoSAppConfiguration()
    with pytest.raises(AttributeError):
        config.userid = "r12345"


def test_CoSAppConfiguration_roles():
    config = CoSAppConfiguration()
    with pytest.raises(AttributeError):
        config.roles = [["r12345"]]


def test_CoSAppConfiguration_validate_file(fake_schema):
    # minimal
    fake_io = StringIO(json.dumps({"userid": "a", "roles": []}))

    with mock.patch("builtins.open") as mock_open:

        def fake_open(file, mode="r", **kwargs):
            if str(file).endswith("configuration_schema.json"):
                return fake_schema()
            if "r" in mode:
                return fake_io
            else:
                return mock.DEFAULT

        mock_open.side_effect = fake_open  # Emulate access to configuration file

        config = CoSAppConfiguration()
        assert config.userid == "a"
        assert len(config.roles) == 0


@pytest.mark.parametrize(
    "configuration",
    [
        ({"userid": "", "roles": [["role1"]]}),  # invalid userid
        ({"roles": [["role1"]]}),  # missing userid
        ({"userid": "abc"}),  # missing roles
        ({"userid": "abc", "roles": [["role1"]], "banana": 42}),  # unexpected keyword
        ({"userid": "abc", "roles": ["role1"]}),  # role is not an array
        ({"userid": "abc", "roles": [["role1", 1]]}),  # role contains none string element
        ({"userid": "abc", "roles": [["role1", "role2"], ["role1", "role2"]]}),  # duplicated role
        ({"userid": "abc", "roles": [["role1", "role2"], ["role3", "role3"]]}),  # duplicated tag in role
        ({"userid": "abc", "roles": [["role1", "role2"], ["role3", ""]]}),  # empty tag string
    ],
)
def test_CoSAppConfiguration_validate_file_error(fake_schema, configuration):
    fake_io = StringIO(json.dumps(configuration))

    with mock.patch("builtins.open") as mock_open:

        def fake_open(file, mode="r", **kwargs):
            if str(file).endswith("configuration_schema.json"):
                return fake_schema()
            if "r" in mode:
                return fake_io
            else:
                return mock.DEFAULT

        mock_open.side_effect = fake_open  # Emulate access to configuration file

        with pytest.raises(ValidationError):
            CoSAppConfiguration()


@mock.patch.object(CoSAppConfiguration, "update_userid", side_effect=OSError)
def test_CoSAppConfiguration_userid_oserror(fake_schema, caplog):
    """Check that default user ID 'UNKNOWN' is used when
    ID cannot be determined from environment variables.
    This may occur on systems where neither USER nor USERNAME
    environment variables are defined, for instance.

    In this test, method `update_userid` is patched to return `OSError`,
    on top of `open` returning `FileNotFoundError` when trying to access
    user config file.

    Related to: https://gitlab.com/cosapp/cosapp/-/issues/120
    """
    with mock.patch("builtins.open") as mock_open:

        def fake_open(file, mode="r", **kwargs):
            if str(file).endswith("configuration_schema.json"):
                return fake_schema()
            if "r" in mode:
                raise FileNotFoundError
            else:
                return mock.DEFAULT

        mock_open.side_effect = fake_open  # Emulate access to configuration file

        caplog.clear()
        with caplog.at_level(logging.WARNING):
            config = CoSAppConfiguration()
        assert config.userid == "UNKNOWN"
        assert config.roles == frozenset()
        warnings = [
            r.message for r in caplog.records 
            if r.levelno == logging.WARNING
        ]
        assert any(map(
            matcher("Configuration file `.*` cannot be opened"),
            warnings
        ))
        assert any(map(
            matcher("Unable to determine user ID"),
            warnings
        ))


@patch_env({"USER": "", "USERNAME": ""})
def test_CoSAppConfiguration_userid_invalid(fake_schema, caplog):
    """Check that default user ID 'UNKNOWN' is used when
    ID cannot be determined from environment variables.
    This may occur on systems where neither USER nor USERNAME
    environment variables are defined, for instance.

    In this test, `open` is patched to return `FileNotFoundError`
    when trying to access user config file, and environment
    variables USER and USERNAME are set to invalid values.

    Related to: https://gitlab.com/cosapp/cosapp/-/issues/120
    """
    with mock.patch("builtins.open") as mock_open:

        def fake_open(file, mode="r", **kwargs):
            if str(file).endswith("configuration_schema.json"):
                return fake_schema()
            if "r" in mode:
                raise FileNotFoundError
            else:
                return mock.DEFAULT

        mock_open.side_effect = fake_open  # Emulate access to configuration file

        caplog.clear()
        with caplog.at_level(logging.WARNING):
            config = CoSAppConfiguration()
        assert config.userid == "UNKNOWN"
        assert config.roles == frozenset()
        warnings = [
            r.message for r in caplog.records 
            if r.levelno == logging.WARNING
        ]
        assert any(map(
            matcher("Configuration file `.*` cannot be opened"),
            warnings
        ))
        assert any(map(
            matcher("Unable to determine user ID"),
            warnings
        ))


@patch_env({"USER": "", "USERNAME": ""})
def test_CoSAppConfiguration_update_userid_error(fake_schema):

    with mock.patch("builtins.open") as mock_open:

        def fake_open(file, mode="r", **kwargs):
            if str(file).endswith("configuration_schema.json"):
                return fake_schema()
            if "r" in mode:
                raise FileNotFoundError
            else:
                return mock.DEFAULT

        mock_open.side_effect = fake_open  # Emulate access to configuration file

        config = CoSAppConfiguration()
        config._userid = None

        with pytest.raises(OSError, match="Unable to determine user ID"):
            config.update_userid()


def test_CoSAppConfiguration_update_configuration(fake_schema):
    config = {
        "userid": "ab123",
        "roles": [
            ["role1"],
            ["foo", "bar"],
            ["led", "zep"],
        ],
    }
    fake_param_str = json.dumps(config)
    fake_io = StringIO(fake_param_str)

    with mock.patch("builtins.open") as mock_open:

        def fake_open(file, mode="r", **kwargs):
            if str(file).endswith("configuration_schema.json"):
                return fake_schema()
            if "r" in mode:
                return fake_io
            else:
                return mock.DEFAULT

        mock_open.side_effect = fake_open  # Emulate access to configuration file

        config = CoSAppConfiguration()
        
        # `open` called twice at construction: once to read the config file,
        # the other to read the validation schema, both is read mode.
        assert mock_open.call_count == 2
        call_args = list(map(
            lambda call: call.args,
            mock_open.call_args_list
        ))
        assert str(call_args[0][0]).endswith("cosapp_config.json")
        assert str(call_args[1][0]).endswith("configuration_schema.json")
        assert call_args[0][1] == "r"
        assert call_args[1][1] == "r"

        assert config.userid == "ab123"
        assert config.roles == {
            frozenset({"role1"}),
            frozenset({"foo", "bar"}),
            frozenset({"led", "zep"}),
        }
        assert isinstance(config.roles, frozenset)

        # Reset and update config
        mock_open.reset_mock()
        assert mock_open.call_count == 0
        config.update_configuration()
        assert mock_open.call_count == 1
        call_args = list(map(
            lambda call: call.args,
            mock_open.call_args_list
        ))
        assert str(call_args[0][0]).endswith("cosapp_config.json")
        assert call_args[0][1] == "w"
        assert mock.call.__enter__().write('"userid"') in mock_open.return_value.mock_calls


@patch_env({
    "COSAPP_CONFIG_DIR": str(
        Path(gettempdir()).joinpath("dummy_cosapp")
    ),
})
def test_CoSAppConfiguration_config_var_env(fake_schema):
    folder = Path(os.environ["COSAPP_CONFIG_DIR"])
    init = {"userid": "ab123", "roles": [["role1"]]}
    fake_param_str = json.dumps(init)
    fake_io = StringIO(fake_param_str)

    with mock.patch("builtins.open") as mock_open:

        def fake_open(file, mode="r", **kwargs):
            if str(file).endswith("configuration_schema.json"):
                return fake_schema()
            if "r" in mode:
                return fake_io
            else:
                return mock.DEFAULT

        mock_open.side_effect = fake_open  # Emulate access to configuration file

        CoSAppConfiguration()

        config_file = folder.joinpath(CoSAppConfiguration.CONFIG_FILE)
        for call in mock_open.calls:
            assert call in [mock.call(config_file, "r"), mock.call(config_file, "w")]


def test_CoSAppConfiguration___init__oserror(caplog):
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        with mock.patch("builtins.open") as mock_open:
            mock_open.side_effect = PermissionError

            CoSAppConfiguration()

    warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    is_match = matcher(
        "Configuration file `.*` cannot be opened; fall back to default"
    )
    assert any(map(is_match, warnings))


def test_CoSAppConfiguration_update_configuration_oserror(caplog):
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        config = CoSAppConfiguration()

        with mock.patch("builtins.open") as mock_open:
            mock_open.side_effect = PermissionError
            config.update_configuration()

    assert "Failed to save configuration locally." in caplog.text
