"""
Ports package of CoSApp.
"""
# This is a pure trick to have a public API similar to a project
# Ports must be part of the core as Module depends on them.
from cosapp.ports.enum import PortType, Scope, Validity
from cosapp.ports.exceptions import ScopeError
from cosapp.ports.port import Port
from cosapp.ports.units import UnitError, add_offset_unit, add_unit, convert_units

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
