"""
Ports package of CoSApp.
"""
# This is a pure trick to have a public API similar to a project
# Ports must be part of the core as Module depends on them.
from .enum import PortType, Scope, Validity
from .exceptions import ScopeError
from .port import Port
from .units import UnitError, add_offset_unit, add_unit, convert_units

__all__ = [
    "Port",
    "PortType",
    "Scope",
    "ScopeError",
    "Validity",
    "UnitError",
    "add_unit",
    "add_offset_unit",
    "convert_units",
]
