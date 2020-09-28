from numbers import Number
from typing import NoReturn

from cosapp.drivers.time.interfaces import ExplicitTimeDriver


class EulerExplicit(ExplicitTimeDriver):
    def __init__(self, name="", owner: "Optional[cosapp.systems.System]" = None, **options):
        """Initialization of the driver

        Parameters
        ----------
        owner : System, optional
            :py:class:`~cosapp.systems.system.System` to which this driver belongs; default None
        **options : Dict[str, Any]
            Optional keywords arguments; may contain time step and interval, with keys `dt` and `time_interval`
        """
        name = name or self.algo_info
        super().__init__(name, owner, **options)

    @property
    def algo_info(self) -> str:
        """str: Short description of driver's algorithm"""
        return "Explicit Euler time driver"

    def _update_transients(self, dt: Number) -> NoReturn:
        """
        Time integration of transient variables over time step `dt` by explicit Euler scheme.
        """
        for x in self._transients.values():
            x.value += x.d_dt * dt
