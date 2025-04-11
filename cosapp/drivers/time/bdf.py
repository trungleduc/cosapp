import numpy
from typing import Optional

from cosapp.drivers.time.implicit import ImplicitTimeDriver, System


def bdf_weights_1(steps: numpy.ndarray) -> list[float]:
    """BDF weights for y_{n}, y_{n+1}."""
    dt_inv = 1.0 / steps[0]
    return [-dt_inv, dt_inv]


def bdf_weights_2(steps: numpy.ndarray) -> list[float]:
    """BDF weights for y_{n}, y_{n+1}, y_{n+2}."""
    h1, h2 = steps[:2]
    hs = h1 + h2
    return [
        +h2 / (h1 * hs),
        -hs / (h1 * h2),
        (hs + h2) / (h2 * hs),
    ]


def bdf_weights_3(steps: numpy.ndarray) -> list[float]:
    """BDF weights for y_{n}, .., y_{n+3}."""
    h1, h2, h3 = steps[:3]
    hs = h1 + h2 + h3
    return [
        -h3 * (h2 + h3) / (h1 * hs * (h1 + h2)),
        +h3 * hs / (h1 * h2 * (h2 + h3)),
        -hs * (h2 + h3) / (h2 * h3 * (h1 + h2)),
        (hs * (h2 + h3 + h3) + h3 * (h2 + h3)) / (hs * h3 * (h2 + h3)),
    ]


def bdf_weights_4(steps: numpy.ndarray) -> list[float]:
    """BDF weights for y_{n}, .., y_{n+4}."""
    h1, h2, h3, h4 = steps[:4]
    s3 = h2 + h3 + h4
    hs = h1 + s3
    return [
        +h4 * s3 * (h3 + h4) / (h1 * hs * (h1 + h2) * (h1 + h2 + h3)),
        -h4 * hs * (h3 + h4) / (h1 * h2 * s3 * (h2 + h3)),
        +h4 * hs * s3 / (h2 * h3 * (h1 + h2) * (h3 + h4)),
        -hs * s3 * (h3 + h4) / (h3 * h4 * (h2 + h3) * (h1 + h2 + h3)),
        (hs * s3 * (h3 + h4 + h4) + h4 * (h3 + h4) * (s3 + hs)) / (hs * s3 * h4 * (h3 + h4)),
    ]


class BdfWeights:

    bdf_weight_funcs = {
        1: bdf_weights_1,
        2: bdf_weights_2,
        3: bdf_weights_3,
        4: bdf_weights_4,
    }
    __slots__ = ("order", "steps", "weights", "counter", "_iso_steps")

    def __init__(self, order: int):
        try:
            self.bdf_weight_funcs[order]
        except KeyError:
            raise ValueError(
                f"Invalid {order=}; must be in {sorted(self.bdf_weight_funcs)}"
            )
        self.order = order
        self.steps = numpy.zeros(order)
        self.weights = numpy.zeros(order + 1)
        self.reset()

    def reset(self) -> None:
        """Reset all steps and weights to zero."""
        self.counter = 0
        self.steps.fill(0.0)
        self.weights.fill(0.0)
        self._iso_steps = True

    def push_step(self, step: float) -> None:
        if step <= 0.0:
            raise ValueError("steps must be positive")
        steps = self.steps
        if not self._iso_steps or step != steps[-1]:
            self.counter = min(self.counter + 1, self.order)
            # Shift steps and take new value as last step
            steps[:-1] = steps[1:]
            self.__set_last_step(step)

    def replace_last_step(self, step: float) -> None:
        if step <= 0.0:
            raise ValueError("steps must be positive")
        if step != self.steps[-1]:
            self.__set_last_step(step)

    def __set_last_step(self, step: float):
        steps = self.steps
        steps[-1] = step
        self._iso_steps = all(steps[1:] == steps[0])
        self.__update_weights()

    def __update_weights(self) -> None:
        """Update weights from steps"""
        w = self.weights
        c = self.counter
        calc_weights = self.bdf_weight_funcs[c]
        n = c + 1
        w[:-n] = 0.0
        w[-n:] = calc_weights(self.steps[-c:])


class BdfIntegrator(ImplicitTimeDriver):
    """Second-order Crank-Nicolson implicit integrator."""

    __slots__ = (
        "_bdf",
        "_prev_yn",
        "_y_size",
        "_pre_transition_y",
        "_pre_transition_dy",
    )

    def __init__(self,
        name = "BDF time driver",
        order: int = 2,
        owner: Optional[System] = None,
        time_interval: Optional[tuple[float, float]] = None,
        dt: Optional[float] = None,
        record_dt: bool = False,
        **options,
    ):
        self._bdf = BdfWeights(order)
        self._y_size = 0
        self._prev_yn = numpy.empty(0)
        self._pre_transition_y = numpy.empty(0)
        self._pre_transition_dy = numpy.empty(0)
        super().__init__(name, owner, time_interval, dt, record_dt, **options)

    @property
    def order(self) -> int:
        """int: order of the BDF scheme"""
        return self._bdf.order

    @order.setter
    def order(self, order: int):
        if order != self._bdf.order:
            self._bdf = BdfWeights(order)

    def _initialize(self):
        """Initialization at time t=t0"""
        super()._initialize()
        y, dy = [], []
        time_problem = self._var_manager.problem
        for transient in time_problem.transients.values():
            y.extend(numpy.ravel(transient.value))
            dy.extend(numpy.ravel(transient.d_dt))
        y = numpy.array(y, dtype=float)
        self._prev_yn = numpy.tile(y, (self.order, 1))
        self._pre_transition_dy = numpy.array(dy, dtype=float)
        self._y_size = y.size

    def _update_transients(self, dt: float):
        """Integrate transient variable over time step `dt`."""
        self._bdf.push_step(dt)
        super()._update_transients(dt)
        # Push latest transients for next iteration
        prev_yn = self._prev_yn
        prev_yn[:-1] = prev_yn[1:]
        prev_yn[-1] = self._prev_x[-self._y_size:]

    def _pre_transition(self) -> None:
        """Pre-transition actions"""
        self._pre_transition_y, self._pre_transition_dy = self._transient_derivatives()
        super()._pre_transition()

    def _post_transition(self, dt: float) -> None:
        """Post-transition actions: check for discontinuities
        in transient values or their derivatives."""
        super()._post_transition(dt)
        y, dy = self._transient_derivatives()
        prev_y = self._pre_transition_y
        prev_dy = self._pre_transition_dy
        identical = numpy.array_equal(y, prev_y) and numpy.array_equal(dy, prev_dy)
        if identical:
            self._bdf.replace_last_step(dt)
        else:
            self._bdf.reset()
            self._prev_yn.fill(0.0)
        self._prev_yn[-1] = y
        self._pre_transition_dy = dy

    def _transient_derivatives(self):
        """Computes the transient unknown vector and its time derivative.
        """
        y, dy = [], []
        time_problem = self._var_manager.problem
        for transient in time_problem.transients.values():
            y.extend(numpy.ravel(transient.value))
            dy.extend(numpy.ravel(transient.d_dt))
        y = numpy.asarray(y, dtype=float)
        dy = numpy.asarray(dy, dtype=float)
        return y, dy

    def _time_residues(self, dt: float, current: bool):
        """Computes and returns the current- or next-time component
        of the transient problem residue vector.
        
        Parameters:
        -----------
        - dt [float]:
            Time step
        - current [bool]:
            If `True`, compute the current time (n) part of the residues.
            If `False`, compute the time (n + 1) part of the residues.
        """
        bdf = self._bdf

        if bdf.counter < 2 and bdf.order > 1:
            # Use Crank-Nicolson scheme for the first time step, when order > 1
            coeff = (0.5 if current else -0.5) * dt
            time_problem = self._var_manager.problem
            residues = []
            for transient in time_problem.transients.values():
                r = transient.value + coeff * numpy.ravel(transient.d_dt)
                residues.extend(numpy.ravel(r))

        elif not current:  # time (n + 1)
            coeff = bdf.weights[-1] * dt
            time_problem = self._var_manager.problem
            residues = []
            for transient in time_problem.transients.values():
                r = coeff * transient.value - dt * numpy.ravel(transient.d_dt)
                residues.extend(numpy.ravel(r))

        else:  # past times n, n - 1, ...
            n = bdf.counter
            prev_yn = self._prev_yn
            residues = numpy.zeros_like(prev_yn[0])
            for weight, prev_y in zip(bdf.weights[-n-1:-1], prev_yn[-n:]):
                residues -= (weight * dt) * prev_y

        return numpy.asarray(residues)
