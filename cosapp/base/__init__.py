"""Module containing base classes for user-defined classes, as well as custom exceptions.
"""
from cosapp.ports import Port, Scope
from cosapp.systems import System
from cosapp.drivers import Driver
from cosapp.ports.connectors import BaseConnector
from cosapp.utils.surrogate_models import SurrogateModel

# Custom exceptions
from cosapp.ports import ConnectorError, ScopeError, UnitError
from cosapp.utils.options_dictionary import OptionError

__all__ = [
    "Port",
    "Scope",
    "System",
    "Driver",
    "BaseConnector",
    "SurrogateModel",
    "ConnectorError",
    "ScopeError",
    "UnitError",
    "OptionError",
]
