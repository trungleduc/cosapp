from numbers import Number
from typing import NoReturn

import numpy

from cosapp.drivers.time.interfaces import ExplicitTimeDriver
from cosapp.utils.helpers import check_arg


class RungeKutta(ExplicitTimeDriver):
    """
    Implementation of a few Runge-Kutta methods for time integration.
    https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods
    """

    __slots__ = ('__fracs', '__coefs', '__buffer', '__init_name')

    def __init__(self, name="", owner: "Optional[cosapp.systems.System]" = None, order: int = 2, **options):
        """Initialization of the driver

        Parameters
        ----------
        owner: System, optional
            :py:class:`~cosapp.systems.system.System` to which this driver belongs; default None
        order: int, optional
            Order of the numerical scheme
        **options: Dict[str, Any]
            Optional keywords arguments; may contain time step and interval, with keys `dt` and `time_interval`
        """
        check_arg(name, "name", str)
        super().__init__("temp", owner, **options)
        self.__init_name = self._name_check(name) if len(name) > 0 else name
        self.__fracs = None
        self.__coefs = None
        self.__buffer = None
        self.order = order

    @property
    def algo_info(self) -> str:
        """str: Short description of driver's algorithm"""
        return f"Explicit order-{self.order} Runge-Kutta time driver"

    @property
    def name(self) -> str:
        return self.__init_name or self.algo_info

    @property
    def order(self) -> int:
        """Numerical order of the time integration scheme"""
        return len(self.__coefs)

    @order.setter
    def order(self, order: int) -> NoReturn:
        check_arg(order, 'order', int, lambda n: 2 <= n <= 4)
        if order == 2:
            # Ralston's second-order scheme
            self.__fracs = numpy.r_[2 / 3]
            self.__coefs = numpy.r_[0.25, 0.75]
        elif order == 3:
            # Heun's third-oder scheme
            self.__fracs = numpy.r_[1, 2] / 3
            self.__coefs = numpy.r_[0.25, 0, 0.75]
        elif order == 4:
            # Runge-Kutta fourth-order scheme
            self.__fracs = numpy.r_[0.5, 0.5, 1]
            self.__coefs = numpy.r_[1, 2, 2, 1] / 6

    def _precompute(self) -> NoReturn:
        super()._precompute()
        # Create memory buffer to store nstages intermediate values of dx/dt, and x at t = tn
        nstages = len(self.__coefs)
        self.__buffer = { name : [x.value] * (nstages + 1) for name, x in self._transients.items() }

    def _update_transients(self, dt: Number) -> NoReturn:
        """
        Time integration of transient variables over time step `dt` by Runge-Kutta scheme.
        """
        transients = self._transients
        buffer = self.__buffer
        steps = self.__fracs * dt
        weights = self.__coefs * dt
        tn = self.time

        # Store unknown at t = tn
        for name, x in transients.items():
            buffer[name][-1] = x.value
        
        # Compute sub-step stages
        for stage, dts in enumerate(steps):
            for name, x in transients.items():
                buffer[name][stage] = dx_dt = x.d_dt
                x.value = buffer[name][-1] + dx_dt * dts
            self._set_time(tn + dts)

        # Compute weighted solution at time tn + dt
        for name, x in transients.items():
            buffer[name][-2] = x.d_dt
            new_x = buffer[name][-1]
            for stage, weight in enumerate(weights):
                new_x += weight * buffer[name][stage]
            x.value = new_x
