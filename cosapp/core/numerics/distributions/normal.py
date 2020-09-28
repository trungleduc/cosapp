"""Basic class to define a variable distribution."""
from typing import Any, Dict, NoReturn, Union, Tuple, Optional

from scipy.optimize import root
from scipy.stats import norm
from .distribution import Distribution


class Normal(Distribution):
    """A class to define a gaussian distribution.

    Parameters
    ----------
    worst : float
        The parameter absolute worst value.
    best : float
        The parameter absolute best value.
    pworst : float, optional
        The worst value probability is the probability that the variable will be lower (if worst<best) or higher
        (if worst>best) than the worst value; default 0.15 (i.e. 15%).
    pbest : float, optional
        The best value probability is the probability that the variable will be higher (if worst<best) or lower
        (if worst>best) than the best value; default 0.15 (i.e. 15%).
    """

    def __init__(
        self,
        worst: float,
        best: float,
        pworst: Optional[float] = 0.15,
        pbest: Optional[float] = 0.15,
    ):
        self._rv = None  # type: scipy.stats.norm

        super(Normal, self).__init__(worst, best, pworst, pbest)

    def _set_distribution(self) -> NoReturn:
        """Set the probability distribution according the parameters."""
        if self.pworst + self.pbest > 1.0:
            raise ValueError(
                "Best and worst probabilities are incompatible: {!s}.".format(
                    self.__json__()
                )
            )

        pts = [self.worst, self.best]
        if self.worst > self.best:
            ppts = [(1.0 - self.pworst), self.pbest]
        else:
            ppts = [self.pworst, (1.0 - self.pbest)]

        if self._rv is None:
            x0 = [0.0, 1.0]
        else:
            params = self._rv.kwds  # return {"loc": #, "scale": #}
            x0 = [params["loc"], params["scale"]]

        def f(x):
            t = norm(loc=x[0], scale=x[1])
            ppf = t.ppf(ppts)
            return ppf - pts

        res = root(f, x0)
        if not res.success:
            raise ValueError(
                "Unable to fit normal distribution on {!s}.".format(self.__json__())
            )
        self._rv = norm(loc=res.x[0], scale=res.x[1])

    def draw(self, quantile: Optional[float] = None) -> float:
        """Generate a random number.

        If a quantile is given, generate the perturbation for that quantile.

        Parameters
        ----------
        quantile : Optional[float], optional
            Quantile for which the perturbation must be set; default None (i.e. random perturbation)
        
        Returns
        -------
        float
            The random number
        """
        return self._rv.rvs() if quantile is None else self._rv.ppf(quantile)
