"""
Multimode package of CoSApp.
"""
from .event import Event, PeriodicTrigger, EventError
from .discreteStepper import DiscreteStepper

__all__ = [
    "Event",
    "EventError",
    "PeriodicTrigger",
    "DiscreteStepper",
]
