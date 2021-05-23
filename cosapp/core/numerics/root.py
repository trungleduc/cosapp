"""
Module provides dispatch to different non-linear algorithms.
"""
import logging
from typing import Callable, Sequence, Union, Tuple, Optional, Dict, Any, Iterable

import numpy
from scipy.linalg import lu_factor, lu_solve, LinAlgWarning
from scipy import optimize

from cosapp.core.numerics.enum import NonLinearMethods
from cosapp.core.numerics.basics import SolverResults
from cosapp.utils.logging import LogLevel

logger = logging.getLogger(__name__)


class NumericSolver:
    """Class defining root routine."""

    @classmethod
    def root(cls,
             fun: Callable[[Sequence[float], Union[float, str], bool], numpy.ndarray],
             x0: Sequence[float],
             args: Tuple[Union[float, str]] = (),
             method: NonLinearMethods = NonLinearMethods.POWELL,
             jac=None,
             tol=None,
             callback=None,
             options: Optional[Dict[str, Any]] = None) -> Union[SolverResults, optimize.OptimizeResult]:
        """Try to cancel residues produced by `fun` starting with x0 as initial solution.

        Parameters
        ----------
        fun : Callable[[Sequence[float], Union[float, str]], numpy.ndarray[float]]
            Callable residues function
        x0 : Sequence[float]
            Initial solution guess
        args : Tuple[Union[float, str], ...]
            Additional arguments for `fun` - first one being the time reference
        method : NonLinearMethods
            Algorithm to use
        jac : Callable[[Sequence[float], Union[float, str]], numpy.ndarray[float]], optional
            Callable providing the jacobian matrix; None if not provided (default: None)
        tol : float, optional
        callback : Callable, optional
        options : dict, optional
            A dictionary of solver options. E.g. tol or max_iter

        Returns
        -------
        SolverResults or scipy.optimize.OptimizeResult
            The solution represented as a OptimizeResult object. Important attributes are: x the solution array,
            success a Boolean flag indicating if the algorithm exited successfully and message which describes the
            cause of the termination. See :py:class:`cosapp.core.numerics.basics.SolverResults` for a description
            of other attributes.
        """

        if not isinstance(x0, Iterable):
            x0 = numpy.asarray([x0])

        if method == NonLinearMethods.NR:
            return cls._cosapp_root(fun, x0, args=args, options=options)
        else:
            if 'verbose' in options:
                options.pop('verbose')
            results = optimize.root(fun, x0, args=args, method=method.value, jac=jac, tol=tol, callback=callback,
                                    options=options)
            results.jac_lup = (None, None)
            if 'jac' in results:
                try:
                    results.jac_lup = lu_factor(results.jac, check_finite=False)
                except (ValueError, LinAlgWarning) as err:  # Silent LU decomposition failure
                    logger.debug(f"Silent error: {err}")

            return results

    @classmethod
    def _cosapp_root(cls,
                     fresidues: Callable[[Sequence[float], Union[float, str], bool], numpy.ndarray],
                     x0: Sequence[float],
                     args: Tuple[Union[float, str]] = (),
                     options: Optional[Dict[str, Any]] = None) -> SolverResults:
        """Customized Newton-Raphson algorithm to solve `fresidues` starting with `x0`.

        Parameters
        ----------
        fresidues : Callable[[Sequence[float], Union[float, str]], numpy.ndarray[float]]
            Callable residues function
        x0 : Sequence[float]
            Initial solution guess
        args : Tuple[Union[float, str], ...]
            Additional arguments for `fun` - first one being the time reference
        options : dict, optional
            A dictionary of solver options. E.g. tol or max_iter

        Options
        -------
        eps : float
            Relative perturbation to compute derivative by finite differences.
        tol : float
            Convergence tolerance criteria
        factor : float, optional
            Relaxation factor
        max_iter : int
            Maximum number of iterations
        jac_lup : (ndarray[float], ndarray[int]), optional
            LU decomposition of Jacobian given as tuple (LU, perm) to reuse as initial direction
        jac : ndarray, optional
            Jacobian to reuse for partial update
        compute_jacobian : bool
            Force to update the Jacobian matrix even if the provided one is useful
        lower_bound : numpy.ndarray
            Min values for parameters iterated by solver.
        upper_bound : numpy.ndarray
            Max values for parameters iterated by solver.
        abs_step : numpy.ndarray
            Max absolute step for parameters iterated by solver.
        rel_step : numpy.ndarray
            Max relative step for parameters iterated by solver.
        history : bool, optional
            Store the history trace of the resolution; default False.
        jac_update_tol : float, optional
            Tolerance level for Jacobian matrix update, based on nonlinearity estimation; default 0.01

        Returns
        -------
        SolverResults or scipy.optimize.OptimizeResult
            The solution represented as a OptimizeResult object. Important attributes are: x the solution array,
            success a Boolean flag indicating if the algorithm exited successfully and message which describes the
            cause of the termination. See :py:class:`cosapp.core.numerics.basics.SolverResults` for a description
            of other attributes.
        """
        verbose = options.get('verbose', False)
        log_level = LogLevel.INFO if verbose else LogLevel.DEBUG
        it_solver = 0

        jac_rel_perturbation = options['eps']
        factor_ref = factor = options['factor']
        jac = options.get('jac', None)
        if jac is not None:
            jac = numpy.asarray(jac)
        jac_lu, piv = options.get('jac_lup', (None, None))
        calc_jac = options.get('compute_jacobian', True) or jac is None or jac.shape[1] != x0.shape[0]
        jac_update_tol = options.get('jac_update_tol', 0.01)
        recorder = options['recorder']
        history = options.get('history', False)
        trace = list()

        def calc_jacobian(x: numpy.ndarray,
                          residues: numpy.ndarray,
                          it_jac: int,
                          it_p_jac: int,
                          jac: numpy.ndarray=None,
                          r_indices_to_update=set()) -> Tuple[numpy.ndarray, int, int]:

            n = x.size
            if jac is None or jac.shape != (n, n):
                new_jac = numpy.zeros((n, n), dtype=float)
                x_indices_to_update = set(range(n))
            else:
                new_jac = jac.copy()
                ncol = new_jac.shape[1]
                x_indices_to_update = set()
                for i in r_indices_to_update:
                    x_indices_to_update.update([j for j in range(ncol) if new_jac[i, j] != 0])

            x_copy = x.copy()

            for j in x_indices_to_update:
                delta = jac_rel_perturbation
                if abs(x_copy[j]) >= jac_rel_perturbation:
                    delta = x_copy[j] * jac_rel_perturbation
                x_copy[j] += delta
                logger.debug(f"Perturb unknown {j}")
                perturbation = fresidues(x_copy, *args)
                new_jac[:, j] = (perturbation - residues) * (1 / delta)
                x_copy[j] = x[j]

            n_partial = len(x_indices_to_update)
            if n_partial == n:
                logger.log(log_level, f'Jacobian matrix: full update')
                it_jac += 1
            elif x_indices_to_update:
                it_p_jac += 1
                logger.log(log_level, 
                    f'Jacobian matrix: {n_partial} over {n} derivative(s) updated')

            return new_jac, it_jac, it_p_jac

        logger.debug("NR - Reference call")
        x = numpy.array(x0, dtype=float)
        r = fresidues(x, *args)
        r_norm = numpy.linalg.norm(r, numpy.inf)
        logger.log(log_level, f"Initial residue: {r_norm}")
        logger.log(LogLevel.FULL_DEBUG, f"Initial relaxation factor {factor}.")
        results = SolverResults()

        def check_numerical_features(name, default_value):
            if name not in options or options[name] is None or numpy.shape(options[name]) != x.shape:
                options[name] = numpy.full_like(x, default_value)

        check_numerical_features('lower_bound', -numpy.inf)
        check_numerical_features('upper_bound', numpy.inf)
        check_numerical_features('abs_step', numpy.inf)
        check_numerical_features('rel_step', numpy.inf)

        if history:
            trace.append({"x": x.copy(), "residues": r.copy()})
        if recorder is not None:
            recorder.record_state(str(it_solver), 'ok', )

        if not calc_jac:
            logger.debug('Reuse of previous Jacobian matrix')

        tol, max_iter = options['tol'], options['max_iter']
        p_jac, p_jac_tries = options['partial_jac'], options['partial_jac_tries']
        it_jac, it_p_jac = 0, 0  # number of full and partial evalutions of the Jacobian
        n_broyden_update = 0
        dr = numpy.zeros_like(r)
        dx = numpy.full_like(r, numpy.nan)

        try:
            res_index_to_update = {}
            consecutive_p_jac = 0

            while r_norm > tol and it_solver < max_iter:
                logger.log(log_level, f'Iteration {it_solver}')

                if calc_jac:
                    if consecutive_p_jac > p_jac_tries or not p_jac:
                        res_index_to_update = range(x.size)
                        consecutive_p_jac = 0
                    else:
                        consecutive_p_jac += 1

                    jac, it_jac, it_p_jac = calc_jacobian(x, r, it_jac, it_p_jac,
                                                          jac, res_index_to_update)

                    jac_lu, piv = cls.__lu_factor(jac)  # may raise an exception

                elif it_solver > 0:
                    # Good Broyden update - source: https://nickcdryan.com/2017/09/16/broydens-method-in-python/
                    logger.log(log_level, f'Broyden update')
                    n_broyden_update += 1
                    corr = numpy.outer((dr - jac.dot(dx)), dx) / dx.dot(dx)
                    jac += corr
                    jac_lu, piv = cls.__lu_factor(jac)

                it_solver += 1

                dx = -lu_solve((jac_lu, piv), r)
                # Compute relaxation factors from max absolute and relative steps
                with numpy.errstate(invalid='ignore', divide='ignore'):
                    abs_x = numpy.abs(x)
                    abs_dx = numpy.abs(dx)
                    dx_max_abs = options['abs_step']
                    dx_max_rel = numpy.where(abs_x > 0, abs_x * options['rel_step'], numpy.inf)
                    factor_abs = numpy.where(abs_dx > dx_max_abs, dx_max_abs / abs_dx, 1).min()
                    factor_rel = numpy.where(abs_dx > dx_max_rel, dx_max_rel / abs_dx, 1).min()

                factor = min(factor_rel, factor_abs, factor, 1)

                if factor < factor_ref:
                    logger.debug(f"\trelaxation factor lowered to {factor}")

                r_norm_prev = r_norm
                x_prev = x.copy()
                dx *= factor
                x += dx

                # Force solution within admissible bounds
                # Note:
                #   This is clearly wrong, and does NOT belong in a nonlinear solver.
                #   The correct way to deal with such constraints is to use an optimizer.
                #   This part was only kept for non-regression purposes, but should be
                #   removed in a future revision.
                x = numpy.maximum(x, options['lower_bound'])
                x = numpy.minimum(x, options['upper_bound'])

                new_r = fresidues(x, *args)
                r_norm = numpy.linalg.norm(new_r, numpy.inf)
                logger.log(log_level, f'Residue: {r_norm:.5g}')

                if numpy.allclose(x, x_prev, rtol=1e-14):
                    logger.log(log_level, f"Fixed point detected: x = {x}, dx = {x - x_prev}")
                    break

                # Estimate non-linearity by comparing extrapolated and actual residues
                abs_r = numpy.abs(r)
                extrapolated_r = r + jac.dot(dx)
                delta_vs_linear = numpy.abs(extrapolated_r - new_r)
                res_index_to_update = set(
                    numpy.argwhere(delta_vs_linear > jac_update_tol * abs_r).ravel()
                )
                res_index_to_update -= set(numpy.argwhere(abs_r < tol).ravel())

                r, dr = new_r, new_r - r

                if history:
                    trace.append({"x": x.copy(), "residues": r.copy()})
                    if calc_jac:
                        trace[-1]["jac"] = jac.copy()
                if recorder is not None:
                    recorder.record_state(str(it_solver), 'ok', )

                if res_index_to_update:
                    factor = factor_ref
                    calc_jac = True
                elif r_norm != 0:
                    growth_rate = 1 + 0.1 * (1 - factor) * r_norm_prev / r_norm
                    factor = min(growth_rate * factor, 1)
                    calc_jac = False
                    logger.log(LogLevel.FULL_DEBUG, f"New relaxation factor {factor}.")

        except (ValueError, LinAlgWarning) as err:
            is_nil = (jac == 0)
            zero_rows = numpy.argwhere(is_nil.all(axis=1)).flatten()
            zero_cols = numpy.argwhere(is_nil.all(axis=0)).flatten()
            results.success = False
            results.message = 'Singular {}x{} Jacobian matrix'.format(*jac.shape)
            results.jac_errors = {'unknowns': zero_cols, 'residues': zero_rows}
            results.jac = jac
            return results

        results.x = x
        results.jac = jac
        results.jac_lup = (jac_lu, piv)
        results.jac_calls = it_jac + it_p_jac
        results.fres_calls = it_solver
        results.success = (r_norm <= tol)
        
        convergence = f'Converged' if results.success else 'Not converged'
        results.message = (
            f'   -> {convergence} ({r_norm:.4e}) in {it_solver} iterations,'
            f' {it_jac} complete, {it_p_jac} partial Jacobian and {n_broyden_update}'
            f' Broyden evaluation(s)')
        results.trace = trace

        return results


    @staticmethod
    def __lu_factor(matrix: numpy.ndarray):
        lu, piv = lu_factor(matrix, check_finite=True)
        min_diag = numpy.abs(lu.diagonal()).min()
        if min_diag < 1e-14:
            raise LinAlgWarning(
                f"Quasi-singular Jacobian matrix; min diag element of U matrix is {min_diag}"
            )
        return lu, piv


root = NumericSolver.root