"""
Configuration of CoSApp.
"""
import os
import json

try:
    from json import JSONDecodeError
except ImportError:
    JSONDecodeError = ValueError
import logging
from pathlib import Path
from typing import FrozenSet, Union

import jsonschema

logger = logging.getLogger(__name__)


class CoSAppConfiguration:
    r"""Encapsulate CoSApp configuration parameters and its handlers.
    
    By default, CoSApp configuration folder is stored in 
    - `$HOME/.cosapp.d` (Linux)
    - `%USERPROFILE%\.cosapp.d` (MS Windows)

    This default configuration folder is overwritten by the environment
    variable ``COSAPP_CONFIG_DIR``.
    """

    COSAPP_CONFIG_DIR = ".cosapp.d"
    CONFIG_FILE = "cosapp_config.json"

    def __init__(self) -> None:
        """Constructor"""
        self._userid = ""
        self._roles: FrozenSet[FrozenSet[str]] = frozenset()

        self.__load_configuration()

    def __load_configuration(self) -> None:
        """Read configuration from file or generate the default.

        Raises
        ------
        OSError
            If the current platform is not recognized
        """
        fullpath = self.get_config_filename()
        ok = False

        if os.path.isfile(fullpath):  # Load local parameters - offline connection
            try:
                parameters = self.validate_file(fullpath)
            except (OSError, JSONDecodeError):
                ok = False
            else:
                ok = True
                self._userid = parameters["userid"]
                self._roles = frozenset(map(frozenset, parameters["roles"]))

        if not ok:
            logger.warn(
                f"Configuration file `{fullpath!s}` cannot be opened; fall back to default."
            )
            try:
                self.update_userid()
            except OSError:
                self._userid = unknwon_id = "UNKNOWN"
                self._roles = frozenset()
                logger.warn(
                    f"Unable to determine user ID; fall back to {unknwon_id!r}."
                )
            # Try to update user permission from official root
            self.update_configuration()

    @staticmethod
    def get_config_dir() -> Path:
        try:
            return Path(os.environ["COSAPP_CONFIG_DIR"])
        except KeyError:
            return Path.home().joinpath(CoSAppConfiguration.COSAPP_CONFIG_DIR)

    @classmethod
    def get_config_filename(cls) -> Path:
        config_dir = cls.get_config_dir()
        return config_dir.joinpath(cls.CONFIG_FILE)

    @property
    def userid(self) -> str:
        """str : User ID"""
        return self._userid

    @property
    def roles(self) -> FrozenSet[FrozenSet[str]]:
        """FrozenSet[FrozenSet[str]] : Roles assigned to user."""
        return self._roles

    @staticmethod
    def config_schema() -> dict:
        """Static method returning the JSON validation schema of the class."""
        path = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(path , "configuration_schema.json"), "r") as fp:
            config_schema = json.load(fp)
        return config_schema

    @classmethod
    def validate_file(cls, filename: Union[str, Path]) -> dict:
        """Validate the provided file against JSON schema for configuration file.

        Parameters
        ----------
        filename : str or Path
            Absolute path to the file to be tested.

        Returns
        -------
        dict
            The dictionary read in the validated file

        Raises
        ------
        `OSError` (and derived exceptions)
            If a problem occurs while opening the file.
        `jsonschema.exceptions.ValidationError`
            If the provided file does not conform to the JSON schema.
        """
        with open(filename, "r") as fp:
            params = json.load(fp)

        jsonschema.validate(params, cls.config_schema())

        return params

    def update_userid(self) -> None:
        """Update current user id.
        """
        candidates = iter(["USERNAME", "USER"])
        old_id = self._userid

        while not self._userid:
            try:
                self._userid = os.environ.get(next(candidates), "")
            except StopIteration:
                raise OSError("Unable to determine user ID.")
        
        if self._userid != old_id:
            logger.info(f"User ID changed from {old_id!r} to {self._userid!r}.")

    def update_configuration(self) -> None:
        """Update current user configuration file.
        """
        # TODO - request reference roles source here

        # Save back changes following the update
        parameters = {
            "userid": self.userid,
            "roles": list(map(list, self.roles)),
        }

        try:
            config_file = self.get_config_filename()
            config_dir = os.path.dirname(config_file)
            if not os.path.isdir(config_dir):
                os.makedirs(config_dir)

            with open(config_file, "w") as file:
                json.dump(parameters, file, sort_keys=True)

        except OSError:
            logger.warning("Failed to save configuration locally.")
