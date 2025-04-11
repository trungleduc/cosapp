import numpy
from cosapp.drivers.time.implicit import ImplicitTimeDriver


class CrankNicolson(ImplicitTimeDriver):
    """Second-order Crank-Nicolson implicit integrator."""

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
        half_dt = (0.5 if current else -0.5) * dt
        time_problem = self._var_manager.problem
        residues = []
        for transient in time_problem.transients.values():
            r = transient.value + half_dt * numpy.ravel(transient.d_dt)
            residues.extend(numpy.ravel(r))
        return numpy.array(residues)
