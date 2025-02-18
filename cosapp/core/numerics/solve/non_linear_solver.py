import inspect
from abc import abstractmethod
from contextlib import contextmanager
from numbers import Number
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    Iterable,
    Generator,
)

import numpy as np
from scipy.linalg import LinAlgWarning
from scipy.optimize import OptimizeResult, root

from cosapp.core.numerics.basics import SolverResults
from cosapp.core.numerics.enum import NonLinearMethods
from cosapp.utils.json import jsonify
from cosapp.utils.logging import LogLevel, logging
from cosapp.utils.options_dictionary import HasCompositeOptions, HasOptions
from cosapp.utils.state_io import object__getstate__

from .jacobian import AbstractJacobianEvaluation, FfdJacobianEvaluation, JacobianStats
from .linear_solver import AbstractLinearSolver, DenseLUSolver

logger = logging.getLogger(__name__)

ConcreteSolver = TypeVar("ConcreteSolver", bound="AbstractNonLinearSolver")
RootFunction = Callable[[Sequence[float], Any], np.ndarray]


class AbstractNonLinearSolver(HasCompositeOptions):
    """Non linear solver abstract class."""

    def __init__(self) -> None:
        super().__init__()
        self._log_level: LogLevel = LogLevel.DEBUG

    @abstractmethod
    def solve(
        self,
        fun: RootFunction,
        x0: Sequence[float],
        args: Tuple[Any] = tuple(),
        callback=None,
    ) -> Union[SolverResults, OptimizeResult]:
        """Performs resolution of the non linear problem."""
        pass

    @abstractmethod
    def setup(self, size: int) -> None:
        """Performs setup of the non linear solver implementation."""
        pass

    @abstractmethod
    def teardown(self) -> None:
        """Performs clean-up of the non linear solver implementation."""
        pass

    def __getstate__(self) -> Dict[str, Any]:
        """Creates a state of the object.

        The state type does NOT match type specified in
        https://docs.python.org/3/library/pickle.html#object.__getstate__
        to allow custom serialization.

        Returns
        -------
        Dict[str, Any]:
            state
        """
        return object__getstate__(self)

    def __json__(self) -> Dict[str, Any]:
        """Creates a JSONable dictionary representation of the object.

        Break circular dependencies by removing some slots from the
        state.

        Returns
        -------
        Dict[str, Any]
            The dictionary
        """
        state = self.__getstate__().copy()

        return jsonify(state)

    @property
    def log_level(self) -> LogLevel:
        """Gets the log level."""
        return self._log_level

    @log_level.setter
    def log_level(self, value: LogLevel) -> None:
        """Sets the log level."""
        self._log_level = value


def get_kwargs():
    """Returns kwargs as a dictionary, using `inspect.currentframe`"""
    frame = inspect.currentframe().f_back
    keys, _, _, values = inspect.getargvalues(frame)
    kwargs = {
        key: values[key]
        for key in set(keys) - {"self"}
    }
    return kwargs


class GradientNLS(AbstractNonLinearSolver):
    """Gradient-based non linear solver abstract class."""

    def __init__(self, **options) -> None:
        super().__init__(**options)

    def setup(self, size: int) -> None:
        """Performs setup of the non linear solver implementation."""
        if self._jac:
            self._jac.setup(size)

    def teardown(self) -> None:
        """Performs clean-up of the non linear solver implementation."""
        if self._jac:
            self._jac.teardown()

    @contextmanager
    def _bound_fresidue(self, fresidues: Callable, args: Iterable) -> Generator:
        """Defines a context with residues function bound to the Jacobian
        evalution method."""
        if self._jac:
            self._jac.bind_residue_function(fresidues, args)
        yield
        if self._jac:
            self._jac.unbind_residue_function()


class ScipyRootSolver(GradientNLS):
    """Gradient-based non linear solver relying on `scipy.optimize.root`."""

    def __init__(
        self,
        method: NonLinearMethods = NonLinearMethods.POWELL,
        jac: Optional[AbstractJacobianEvaluation] = None,
        tol: Optional[float] = None,
        **options,
    ):
        self._method = method
        self._jac = jac
        self._tol = tol

        super().__init__(**options)

    def _alias_options(self) -> Dict[str, Any]:
        """Gets the options eventually aliased."""

        def alias(name):
            return self.__option_aliases.get(name, name)

        return {alias(key): val for key, val in self._options.items()}

    def _declare_options(self) -> None:
        """Declares options."""
        super()._declare_options()
        method = self._method
        options = self._options

        if method == NonLinearMethods.POWELL:
            self.__option_aliases = {
                "tol": "xtol",
                "max_eval": "maxfev",
            }
            options.declare(
                "tol",
                1.0e-7,
                dtype=float,
                desc="The calculation will terminate if the relative error between two consecutive iterations is at most tol.",
            )
            options.declare(
                "max_eval",
                0,
                dtype=int,
                desc="The maximum number of calls to the function. If zero, assumes 100 * (N + 1),"
                " where N is the number of elements in x0.",
            )
            options.declare(
                "eps",
                None,
                dtype=float,
                allow_none=True,
                desc="A suitable step length for the forward-difference approximation of the Jacobian (for fprime=None)."
                " If eps is less than machine precision u, it is assumed that the relative errors in the"
                " functions are of the order of u.",
            )
            options.declare(
                "factor",
                0.1,
                dtype=float,
                lower=0.1,
                upper=100.0,
                desc="A parameter determining the initial step bound factor * norm(diag * x). Should be in the interval [0.1, 100].",
            )

        elif method == NonLinearMethods.BROYDEN_GOOD:
            self.__option_aliases = {
                "tol": "fatol",
                "num_iter": "nit",
                "max_iter": "maxiter",
                "min_rel_step": "xtol",
                "min_abs_step": "xatol",
            }
            options.declare(
                "num_iter",
                100,
                dtype=int,
                desc="Number of iterations to perform. If omitted (default), iterate unit tolerance is met.",
            )
            options.declare(
                "max_iter",
                100,
                dtype=int,
                desc="Maximum number of iterations to make. If more are needed to meet convergence, NoConvergence is raised.",
            )
            options.declare(
                "disp",
                False,
                dtype=bool,
                desc="Print status to stdout on every iteration.",
            )
            options.declare(
                "tol",
                6e-6,
                dtype=float,
                desc="Absolute tolerance (in max-norm) for the residual. If omitted, default is 6e-6.",
            )
            options.declare(
                "line_search",
                "armijo",
                dtype=str,
                allow_none=True,
                desc="Which type of a line search to use to determine the step "
                "size in the direction given by the Jacobian approximation. Defaults to ‘armijo’.",
            )
            options.declare(
                "jac_options",
                {"reduction_method": "svd"},
                dtype=dict,
                allow_none=True,
                desc="Options for the respective Jacobian approximation. restart, simple or svd",
            )
        
        else:
            raise NotImplementedError("'NonLinearMethods' value is not handled.")
    
    def solve(self, fun, x0, args=(), **options) -> OptimizeResult:
        """Performs resolution of the non linear problem."""
        with self._bound_fresidue(fun, args):
            self.setup(x0.size)
            result = root(
                fun,
                x0,
                args=args,
                method=self._method.value,
                tol=self._options.get("tol", None),
                jac=self._jac,
                callback=options.pop("callback", None),
                options=self._alias_options(),
            )
            self.teardown()

        return result

    @AbstractNonLinearSolver.log_level.setter
    def log_level(self, value: LogLevel) -> None:
        """Sets the log level."""
        AbstractNonLinearSolver.log_level.fset(self, value)
        if self._jac:
            self._jac.log_level = value


class NewtonRaphsonSolver(GradientNLS):
    """Gradient-based non linear custom solver based on Newton-Raphson."""

    def __init__(
        self,
        tol="auto",
        factor=1.0,
        max_iter=100,
        eps=2 ** (-23),
        jac_update_tol=0.01,
        jac: Optional[FfdJacobianEvaluation] = None,
        linear_solver: Optional[DenseLUSolver] = None,
        history=False,
        tol_update_period=4,
        tol_to_noise_ratio=16,
        abs_step=np.inf,
        rel_step=np.inf,
        **kwargs,
    ):
        self._default_options = get_kwargs()

        self._tol = tol
        self._factor = factor
        self._max_iter = max_iter
        self._eps = eps
        self._jac_update_tol = jac_update_tol
        self._jac = jac
        self._linear_solver = linear_solver
        self._history = history
        self._tol_update_period = tol_update_period
        self._tol_to_noise_ratio = tol_to_noise_ratio
        self._abs_step = abs_step
        self._rel_step = rel_step

        super().__init__()

    def _declare_options(self) -> None:
        """Declares options."""
        super()._declare_options()
        options = self._options

        options.declare(
            "tol",
            "auto",
            dtype=(float, str),
            allow_none=True,
            desc="Absolute tolerance (in max-norm) for the residual.",
        )
        options.declare(
            "max_iter", 100, dtype=int, desc="The maximum number of iterations."
        )

        options.declare(
            "factor",
            1.0,
            dtype=Number,
            allow_none=True,
            lower=1e-3,
            upper=1.0,
            desc="A parameter determining the initial step bound factor * norm(diag * x). Should be in interval [0.001, 1].",
        )
        options.declare(
            "jac",
            lambda: FfdJacobianEvaluation(),
            dtype=AbstractJacobianEvaluation,
            allow_none=False,
            desc=".",
        )
        options.declare(
            "linear_solver",
            lambda: DenseLUSolver(),
            dtype=AbstractLinearSolver,
            allow_none=False,
            desc=".",
        )
        options.declare(
            "jac_update_tol",
            0.01,
            dtype=float,
            allow_none=False,
            lower=0,
            upper=1,
            desc="Tolerance level for partial Jacobian matrix update, based on nonlinearity estimation.",
        )
        options.declare(
            "recorder",
            None,
            allow_none=True,
            desc="A recorder to store solver intermediate results.",
        )
        options.declare(
            "lower_bound",
            None,
            dtype=np.ndarray,
            allow_none=True,
            desc="Min values for parameters iterated by solver.",
        )
        options.declare(
            "upper_bound",
            None,
            dtype=np.ndarray,
            allow_none=True,
            desc="Max values for parameters iterated by solver.",
        )
        options.declare(
            "abs_step",
            None,
            dtype=np.ndarray,
            allow_none=True,
            desc="Max absolute step for parameters iterated by solver.",
        )
        options.declare(
            "rel_step",
            None,
            dtype=np.ndarray,
            allow_none=True,
            desc="Max relative step for parameters iterated by solver.",
        )
        options.declare(
            "history",
            False,
            dtype=bool,
            allow_none=False,
            desc="Request saving the resolution trace.",
        )
        options.declare(
            "tol_update_period",
            4,
            dtype=int,
            lower=1,
            allow_none=False,
            desc="Tolerance update period, in iteration number, when tol='auto'.",
        )
        options.declare(
            "tol_to_noise_ratio",
            16,
            dtype=Number,
            lower=1.0,
            allow_none=False,
            desc="Tolerance-to-noise ratio, when tol='auto'.",
        )

    def _get_nested_objects_with_options(self) -> Iterable[HasOptions]:
        """Gets nested objects having options."""
        return (self._linear_solver, self._jac)

    def _set_options(self) -> None:
        """Sets options from the current state."""
        for key, val in self._options.items():
            setattr(self, f"_{key}", val)

    @AbstractNonLinearSolver.log_level.setter
    def log_level(self, value: LogLevel) -> None:
        """Sets the log level."""
        AbstractNonLinearSolver.log_level.fset(self, value)
        if self._jac:
            self._jac.log_level = value

    def solve(
        self,
        fresidues: RootFunction,
        x0: Sequence[float],
        args: Tuple[Union[float, str]] = tuple(),
        callback=None,
        **options,
    ) -> SolverResults:
        """Performs resolution of the non linear problem."""

        def check_numerical_features(name, default_value):
            value = getattr(self, f"_{name}")
            if value is None or np.shape(value) != x0.shape:
                setattr(self, f"_{name}", np.full_like(x0, default_value))

        check_numerical_features("lower_bound", -np.inf)
        check_numerical_features("upper_bound", np.inf)
        check_numerical_features("abs_step", np.inf)
        check_numerical_features("rel_step", np.inf)

        with self._bound_fresidue(fresidues, args):
            self.setup(x0.size)
            result = self._solve(fresidues, x0, args, callback=callback, **options)
            self.teardown()

        return result

    @staticmethod
    def _must_update_tol_auto(it: int, tol: float, iter_period: int) -> bool:
        """Computes if the automatic tolerance must be updated or not."""
        return it % iter_period == 0 or tol <= 0.0

    @staticmethod
    def _must_update_tol_default(*args) -> bool:
        """Defines a placeholder to always updating automatic tolerance."""
        return False

    def _auto_update_tolerance(
        self,
        prev_tol: float,
        it_solver: int,
        jac: np.ndarray,
        x: np.ndarray,
        epsilon: float,
        tol_to_noise: float,
    ) -> float:
        """Computes the updated tolance when using automatic mode."""
        norm_J = np.linalg.norm(jac, np.inf)
        norm_x = np.linalg.norm(x, np.inf)
        noise = epsilon * norm_J * norm_x
        tol = tol_to_noise * noise
        logger.log(
            LogLevel.FULL_DEBUG,
            f"iter #{it_solver}; {noise = }; {tol = }; |J| = {norm_J}; |x| = {norm_x}",
        )
        if tol == prev_tol:
            logger.log(
                self.log_level,
                f"Numerical saturation detected at iteration {it_solver}; {x = }",
            )

        return tol

    def _solve(
        self,
        fresidues: RootFunction,
        x0: Sequence[float],
        args: Tuple[Union[float, str]] = tuple(),
        compute_jacobian: Optional[bool] = False,
        callback=None,
        **options,
    ) -> SolverResults:
        """Performs resolution of the non linear problem.

        Uses a custom Newton-Raphson algorithm to solve `fresidues` starting with `x0`.

        Parameters
        ----------
        fresidues [callable[[Sequence[float], Any], np.ndarray[float]]]:
            Callable residues function
        x0 [sequence[float]]:
            Initial solution guess
        args [tuple[Any, ...]]:
            Additional arguments for `fun`
        options [dict, optional]:
            A dictionary of problem-dependent solver options.
        callback [callable, optional]:
            Optional callback function. It is called on every iteration as `callback(x, r)`,
            where x is the current solution and r the corresponding residual.

        Returns
        -------
        SolverResults
            The solution represented as a `SolverResults` object.
            See :py:class:`cosapp.core.numerics.basics.SolverResults` for details.
        """
        jac_update_tol = self._jac_update_tol
        factor_ref = factor = self._factor
        history = self._history
        tol = self._tol

        logger.debug("NR - Reference call")
        x = np.array(x0, dtype=np.float64).flatten()
        r = fresidues(x, *args)
        r_norm = np.linalg.norm(r, np.inf)
        results = SolverResults()

        logger.log(self.log_level, f"Initial residue: {r_norm}")
        logger.log(LogLevel.FULL_DEBUG, f"Initial relaxation factor {factor}.")

        if tol is None:
            tol = "auto"
        elif isinstance(tol, str):
            if tol != "auto":
                raise ValueError("Tolerance must be a float, None, or 'auto'.")
        elif not isinstance(tol, Number):
            raise TypeError(f"Tolerance must be a number, None, or 'auto'; got {tol!r}")

        auto_tol = tol == "auto"
        if auto_tol:
            tol = 0.0
            iter_period = self._tol_update_period
            tol_to_noise = self._tol_to_noise_ratio
            must_update_tol = self._must_update_tol_auto
        elif tol < 0:
            raise ValueError(f"Tolerance must be non-negative (got {tol})")
        else:
            iter_period = None
            tol_to_noise = None
            must_update_tol = self._must_update_tol_default

        l_solver = self._linear_solver
        if l_solver.need_jacobian:
            jac = self._linear_solver.jacobian
            reuse_jac = (
                self._linear_solver.has_valid_state(x.size) and not compute_jacobian
            )
            logger.debug("Reuse of previous Jacobian matrix")
        else:
            jac = None
            reuse_jac = True

        results.trace = trace = list()
        if history:
            record = {
                "x": x.copy(),
                "residues": r.copy(),
                "tol": tol,
            }
            trace.append(record)

        max_iter = self._max_iter

        dr = np.zeros_like(r)
        dx = np.full_like(r, np.nan)

        it_solver = 0
        rtol_x = 1e-14
        epsilon = np.finfo(np.float64).eps
        logger.log(
            LogLevel.FULL_DEBUG,
            "\t".join(
                [f"iter #{it_solver}", f"{tol = :.2e}", f"|R| = {r_norm}", f"{x = }"]
            ),
        )
        broyden_only = False
        res_index_to_update = set()
        jac_stats: JacobianStats = self._jac.get_stats()
        try:
            while r_norm > tol and it_solver < max_iter:
                logger.log(self.log_level, f"Iteration {it_solver}")
                if self._linear_solver.need_jacobian and not reuse_jac:
                    jac = self._jac(
                        x,
                        r0=r,
                        res_index_to_update=res_index_to_update,
                        jac=jac,
                        dx=dx,
                        dr=dr,
                        broyden_only=broyden_only,
                    )
                    jac_new_stats: JacobianStats = self._jac.get_stats()

                    if jac_stats != jac_new_stats:
                        self._linear_solver.setup(jac)
                        jac_stats = jac_new_stats
                else:
                    reuse_jac = False

                if must_update_tol(it_solver, tol, iter_period):
                    tol = self._auto_update_tolerance(
                        tol, it_solver, jac, x, epsilon, tol_to_noise
                    )

                dx = self._linear_solver.solve(r)
                it_solver += 1

                factor = self._update_relaxation_factor(factor, x, dx)

                if factor < factor_ref:
                    logger.debug(f"\trelaxation factor lowered to {factor}")

                r_norm_prev = r_norm
                x_prev = x.copy()
                dx *= factor
                x += dx

                new_r = fresidues(x, *args)
                r_norm = np.linalg.norm(new_r, np.inf)
                logger.log(self.log_level, f"Residue: {r_norm:.5g}")
                logger.log(
                    LogLevel.DEBUG,
                    "\t".join(
                        [
                            f"iter #{it_solver}",
                            f"{tol = :.2e}",
                            f"|R| = {r_norm}",
                            f"{x = }",
                        ]
                    ),
                )

                if np.allclose(x, x_prev, rtol=rtol_x, atol=0):
                    logger.log(
                        self.log_level,
                        f"Fixed point detected: {x = }, dx = {x - x_prev}",
                    )
                    break

                # Estimate non-linearity by comparing extrapolated and actual residues
                abs_r = np.abs(r)
                extrapolated_r = r + self._linear_solver.eval(dx)
                delta_vs_linear = np.abs(extrapolated_r - new_r)
                res_index_to_update = set(
                    np.argwhere(delta_vs_linear > jac_update_tol * abs_r).ravel()
                )
                res_index_to_update -= set(np.argwhere(abs_r < tol).ravel())

                r, dr = new_r, new_r - r

                if history:
                    record = {
                        "x": x.copy(),
                        "residues": r.copy(),
                        "tol": tol,
                    }
                    if not reuse_jac:
                        record["jac"] = jac.copy()
                    trace.append(record)

                if callback:
                    callback(x, r)

                if res_index_to_update:
                    factor = factor_ref
                    broyden_only = False
                elif r_norm != 0:
                    growth_rate = 1 + 0.1 * (1 - factor) * r_norm_prev / r_norm
                    factor = min(growth_rate * factor, 1)
                    broyden_only = True
                    logger.log(LogLevel.FULL_DEBUG, f"New relaxation factor {factor}.")

        except (ValueError, LinAlgWarning):
            is_nil = jac == 0
            zero_rows = np.argwhere(is_nil.all(axis=1)).flatten()
            zero_cols = np.argwhere(is_nil.all(axis=0)).flatten()
            results.success = False
            results.message = "Singular {}x{} Jacobian matrix".format(*jac.shape)
            results.jac_errors = {"unknowns": zero_cols, "residues": zero_rows}

        else:
            results.x = x
            results.fres_calls = it_solver
            results.success = bool(r_norm <= tol)

            jac_stats = self._jac.get_stats()
            results.jac_calls = jac_stats.full_updates

            status = "Converged" if results.success else "Not converged"
            message = (
                f"{status} ({r_norm:.4e}) in {it_solver} iterations,"
                f" {jac_stats.full_updates} complete, {jac_stats.partial_updates} partial Jacobian"
                f" and {jac_stats.broyden_updates} Broyden evaluation(s)"
                f" ({tol = :.1e})"
            )
            results.message = f"   -> {message}"

        finally:
            results.jac = jac
            results.tol = tol
            return results

    def _update_relaxation_factor(
        self, factor: float, x: np.ndarray, dx: np.ndarray
    ) -> float:
        """Computes relaxation factors from max absolute and relative steps."""
        with np.errstate(invalid="ignore", divide="ignore"):
            new_x = x + dx
            abs_x = np.abs(x)
            abs_dx = np.abs(dx)
            dx_max_abs = self._abs_step
            dx_max_rel = np.where(abs_x > 0, abs_x * self._rel_step, np.inf)
            dx_max = np.where(new_x > self._upper_bound, self._upper_bound - x, np.inf)
            dx_min = np.where(new_x < self._lower_bound, x - self._lower_bound, np.inf)

            factor_abs = np.where(abs_dx > dx_max_abs, dx_max_abs / abs_dx, 1).min()
            factor_rel = np.where(abs_dx > dx_max_rel, dx_max_rel / abs_dx, 1).min()
            factor_max = np.where(abs_dx > dx_max, dx_max / abs_dx, 1).min()
            factor_min = np.where(abs_dx > dx_min, dx_min / abs_dx, 1).min()

        return min(factor_rel, factor_abs, factor_max, factor_min, factor, 1)
