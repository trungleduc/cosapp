from numbers import Number

from cosapp.drivers.time.interfaces import ExplicitTimeDriver


class EulerExplicit(ExplicitTimeDriver):
    def __init__(self, name="Euler", owner: "Optional[cosapp.systems.System]" = None, **options):
        """Initialization of the driver

        Parameters
        ----------
        owner : System, optional
            :py:class:`~cosapp.systems.system.System` to which this driver belongs; default None
        **options : Dict[str, Any]
            Optional keywords arguments; may contain time step and interval, with keys `dt` and `time_interval`
        """
        super().__init__(name, owner, **options)

    def _update_transients(self, dt: Number) -> None:
        """
        Time integration of transient variables over time step `dt` by explicit Euler scheme.
        """
        for x in self._transients.values():
            x.value += x.d_dt * dt
