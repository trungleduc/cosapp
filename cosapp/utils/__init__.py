"""Classes and functions helper using third-party libraries to interact with CoSApp core.
"""
from .logging import LogLevel, set_log
from .helpers import partition
from .state_io import get_state, set_state
from .swap_system import swap_system

try:
    import pytest
except ModuleNotFoundError:
    pass
else:
    pytest.register_assert_rewrite("cosapp.utils.testing")

__all__ = [
    "LogLevel",
    "set_log",
    "partition",
    "get_state",
    "set_state",
    "swap_system",
]
