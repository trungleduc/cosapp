import logging
from io import StringIO
from typing import (Any, Callable, Dict, List, NoReturn, Optional, Sequence,
                    Tuple, Union)

import numpy
import pandas

from cosapp.core.numerics.basics import SolverResults
from cosapp.core.numerics.enum import NonLinearMethods
from cosapp.core.numerics.root import root
from cosapp.drivers.abstractsolver import AbstractSolver
from cosapp.drivers.driver import Driver
from cosapp.drivers.iterativecase import IterativeCase
from cosapp.utils.helpers import check_arg
from cosapp.utils.logging import LogFormat, LogLevel

logger = logging.getLogger(__name__)


class NonLinearSolver(AbstractSolver):
    """Solve mathematical problem with algebraic variables.

    Attributes
    ----------
    compute_jacobian : bool
        Should the Jacobian matrix be computed? ; default True
    jac_lup : ndarray, optional
        LU decomposition of latest Jacobian matrix (if available, None otherwise)
    jac : ndarray, optional
        Latest Jacobian matrix computed (if available, None otherwise)
    """

    __slots__ = (
        '__method', '__option_aliases', '__trace',
        'compute_jacobian', 'jac_lup', 'jac'
    )

    def __init__(self, 
        name: str, 
        owner: "Optional[cosapp.systems.System]" = None, 
        method: Union[NonLinearMethods, str] = NonLinearMethods.NR, 
        **kwargs
    ) -> NoReturn:
        """Initialize a driver

        Parameters
        ----------
        name: str, optional
            Name of the `Module`
        owner : System, optional
            :py:class:`~cosapp.systems.system.System` to which this driver belong; default None
        method : Union[NonLinearMethods, str]
            Resolution method to use
        **kwargs : Dict[str, Any]
            Optional keywords arguments
        """
        super().__init__(name, owner, **kwargs)

        if isinstance(method, str):
            method = NonLinearMethods(method)
        self.__method = method
        self.__option_aliases = dict()
        self.__set_method(method, **kwargs)
        self.__trace: List[Dict[str, Any]] = list()

        self.compute_jacobian = True  # type: bool
            # desc='Should the Jacobian matrix be computed?'

        self.jac_lup = (None, None)  # type: Tuple[Optional[numpy.ndarray], Optional[numpy.ndarray]]
            # desc='LU decomposition of latest Jacobian matrix (if available).'
        self.jac = None  # type: Optional[numpy.ndarray]
            # desc='Latest computed Jacobian matrix (if available).'

        from cosapp.drivers.runsinglecase import RunSingleCase
        self.add_child(RunSingleCase(self._default_driver_name))

    @property
    def method(self) -> NonLinearMethods:
        """NonLinearMethods : Selected solver algorithm."""
        return self.__method

    def add_child(self, child: 'Driver', execution_index: Optional[int] = None) -> 'Driver':
        """Add a child `Driver` to the current `Driver`.

        When adding a child `Driver`, it is possible to specified its position in the execution order.

        Child `Port`, `inwards` and `outwards` can also be pulled at the parent level by providing either
        the name of the port/inward/outward or a list of them or the name mapping of the child element
        (dictionary keys) to the parent element (dictionary values).
        If the argument is not a dictionary, the name in the parent system will be the same as in the child.

        Parameters
        ----------
        child: Module
            `Module` to add to the current `Module`
        execution_index: int, optional
            Index of the execution order list at which the `Module` should be inserted;
            default is latest.

        Notes
        -----
        The added child will have its owner set to match the one of the current driver.
        """
        if len(self.children) == 1 and self._default_driver_name in self.children.keys():
            self.pop_child(self._default_driver_name)
        self.compute_jacobian = True
        return super().add_child(child, execution_index)

    def is_standalone(self) -> bool:
        """Is this Driver able to solve a system?

        Returns
        -------
        bool
            Ability to solve a system or not.
        """
        return True

    def resolution_method(self, fresidues: Callable[[Sequence[float], Union[float, str], bool], numpy.ndarray],
                          x0: Sequence[float],
                          args: Tuple[Union[float, str]] = (),
                          options: Optional[Dict[str, Any]] = None) -> SolverResults:

        if self.method == NonLinearMethods.NR:
            self.options.update(self._get_solver_limits())

        this_options = dict([(key, options[key]) for key in self.options if key in options])

        if self.method == NonLinearMethods.NR:
            this_options['compute_jacobian'] = self.compute_jacobian
            this_options['jac_lup'] = self.jac_lup
            this_options['jac'] = self.jac

        results = root(fresidues, x0, args=args, method=self.method, options=this_options)

        if results.jac_lup[0] is not None:
            self.jac_lup = results.jac_lup
            self.jac = results.jac
            self.compute_jacobian = False

        return results

    def _print_solution(self) -> NoReturn:  # TODO better returning a string
        """Print the solution in the log."""
        if self.options['verbose']:
            # TODO move it in MathematicalProblem
            logger.info('Parameters [{}]: '.format(len(self.solution)))
            for k, v in self.solution.items():
                logger.info('   # ' + str(k) + ': ' + str(v))

            for name, residue in self.problem.residues.items():
                value = numpy.asarray(residue.value)
                if value.ndim > 0:
                    msg = 'Residues [{}]: \n   # ({}'.format(len(self.problem.residues), name)
                    msg += '\n   #  {}'.format(" " * len(name)).join([", {:.5g}".format(v) for v in value])
                    msg += ')\n'
                else:
                    logger.info('   # ({}, {:.5g})'.format(name, value))

            tol = numpy.max(numpy.abs(self.problem.residues_vector))
            try:
                option = self.__option_aliases['tol']  # should never fail, in principle
                target = self.options[option]
            except:
                target = 0
            logger.debug(' # Current tolerance {} for target {}'.format(tol, target))

    def _precompute(self) -> NoReturn:
        """List all iteratives variables and get the initial values."""
        super()._precompute()

        for child in self.children.values():
            if isinstance(child, IterativeCase):
                self.problem.extend(child.get_problem(), copy=False)
                self.initial_values = numpy.append(self.initial_values, child.get_init(self.force_init))
            else:
                logger.warning('Including Driver "{}" without iteratives in Driver "{}" is not numerically advised.'
                                ''.format(child.name, self.name))

    def compute(self) -> NoReturn:
        """Run the resolution method to find free vars values that annul residues
        """

        # Reset status
        self.status = ''
        self.error_code = '0'

        try:
            self.problem.validate()
        except ArithmeticError as err:
            self.status = 'ERROR'
            self.error_code = '9'
            raise err

        if len(self.initial_values) > 0:
            self.solution = {}

            # compute first order
            results = self.resolution_method(
                self._fresidues,
                self.initial_values,
                options=self.options,
            )

            self.__trace = getattr(results, "trace", list())

            if results.success:
                self.status = ''
                self.error_code = '0'
                logger.info('solver : {}{}'.format(self.name, results.message))
            else:
                self.status = 'ERROR'
                self.error_code = '9'

                error_msg = f'The solver failed: {results.message}'

                error_desc = getattr(results, 'jac_errors', {})
                if error_desc:
                    if len(error_desc['unknowns']) > 0:
                        n_unknowns = len(error_desc['unknowns'])
                        unknown_idx = error_desc['unknowns']
                        unknown_names = [self.problem.unknowns_names[i] for i in unknown_idx]
                        error_msg += (f' \nThe {n_unknowns} following parameter(s) have '
                            f'no influence: {unknown_names} \n{unknown_idx}')


                    if len(error_desc['residues']) > 0:
                        n_equations = len(error_desc['residues'])
                        equation_names = [self.problem.residues_names[i] for i in error_desc['residues']]
                        error_msg += (f' \nThe {n_equations} following residue(s) are '
                            f'not influenced: {equation_names}')

                if self.parent is not None:
                    raise ArithmeticError(error_msg)
                else:
                    logger.error(error_msg)

            self.solution = dict([(key, unknown.default_value) for key, unknown in self.problem.unknowns.items()])
            self._print_solution()
        else:
            self.owner.run_children_drivers()
            logger.debug('No parameters/residues to solve. Fallback to children execution.')

        if self._recorder is not None:
            for child in self.children.values():
                child.run_once()
                self._recorder.record_state(child.name, self.status, self.error_code)

    def log_debug_message(
        self,
        handler: "HandlerWithContextFilters",
        record: logging.LogRecord,
        format: LogFormat = LogFormat.RAW
    ) -> bool:
        """Callback method on the driver to log more detailed information.
        
        This method will be called by the log handler when :py:meth:`~cosapp.utils.logging.LoggerContext.log_context`
        is active if the logging level is lower or equals to VERBOSE_LEVEL. It allows
        the object to send additional log message to help debugging a simulation.

        Parameters
        ----------
        handler : HandlerWithContextFilters
            Log handler on which additional message should be published.
        record : logging.LogRecord
            Log record
        format : LogFormat
            Format of the message

        Returns
        -------
        bool
            Should the provided record be logged?
        """
        message = record.getMessage()
        activate = getattr(record, "activate", None)
        emit_record = super().log_debug_message(handler, record, format)

        if message.endswith("call_setup_run") or message.endswith("call_clean_run"):
            emit_record = False

        elif activate == True:
            self.options["history"] = True
            emit_record = False

        elif activate == False:
            self.options["history"] = False
            emit_record = False

            message = ""
            for i, info in enumerate(self.__trace):
                if i == 0:
                    unknowns_trace = info["x"]
                    residues_trace = info["residues"]
                else:
                    unknowns_trace = numpy.vstack((unknowns_trace, info["x"]))
                    residues_trace = numpy.vstack((residues_trace, info["residues"]))

                if "jac" in info:
                    message += f"Iteration {i}\n"
                    container = StringIO()
                    numpy.savetxt(container, info["jac"], delimiter=",")
                    jacobian = container.getvalue()
                    unknowns = ",".join(self.problem.unknowns_names)
                    message += f"New Jacobian matrix:\n,{unknowns}\n"
                    for residue, line in zip(self.problem.residues_names, jacobian.splitlines()):
                        message += f"{residue},{line}\n"

            if len(self.__trace) > 0:
                if len(self.__trace) == 1:
                    size = self.problem.shape[0]
                    unknowns_trace = unknowns_trace.reshape((1, size))
                    residues_trace = residues_trace.reshape((1, size))
                unknowns_df = pandas.DataFrame(unknowns_trace, columns=self.problem.unknowns_names)
                container = StringIO()
                unknowns_df.to_csv(container, line_terminator="\n")
                message += f"Unknowns\n{container.getvalue()}\n"
                residues_df = pandas.DataFrame(residues_trace, columns=self.problem.residues_names)
                container = StringIO()
                residues_df.to_csv(container, line_terminator="\n")
                message += f"Residues\n{container.getvalue()}\n"

            handler.log(
                LogLevel.FULL_DEBUG,
                message,
                name=logger.name,
            )

        return emit_record

    def __set_method(self, method, **options):
        check_arg(method, 'method', NonLinearMethods)

        self.__method = method
        self.__option_aliases = dict()

        if self.__method == NonLinearMethods.NR:
            self.options.declare('tol', 1.0e-5, dtype=float, allow_none=True,
                desc='Absolute tolerance (in max-norm) for the residual.')
            self.options.declare('max_iter', 500, dtype=int,
                desc='The maximum number of iterations.')
            self.options.declare('eps', 1.0e-4, dtype=float, allow_none=True,
                desc='A suitable step length for the forward-difference approximation of the Jacobian (for fprime=None).'
                    ' If eps is smaller than machine precision u, it is assumed that the relative errors in the'
                    ' functions are of the order of u.')
            self.options.declare('factor', 1.0, dtype=float, allow_none=True, lower=1e-3, upper=1.0,
                desc='A parameter determining the initial step bound factor * norm(diag * x). Should be in the interval [0.1, 1].')
            self.options.declare('partial_jac', True, dtype=bool, allow_none=False,
                desc='Defines if partial Jacobian updates can be computed before a complete Jacobian matrix update.')
            self.options.declare('partial_jac_tries', 5, dtype=int, allow_none=False, lower=1, upper=5,
                desc='Defines how many partial Jacobian updates can be tried before a complete Jacobian matrix update.')
            self.options.declare('recorder', None, allow_none=True,
                desc='A recorder to store solver intermediate results.')
            self.options.declare('lower_bound', None, dtype=numpy.ndarray, allow_none=True,
                desc='Min values for parameters iterated by solver.')
            self.options.declare('upper_bound', None, dtype=numpy.ndarray, allow_none=True,
                desc='Max values for parameters iterated by solver.')
            self.options.declare('abs_step', None, dtype=numpy.ndarray, allow_none=True,
                desc='Max absolute step for parameters iterated by solver.')
            self.options.declare('rel_step', None, dtype=numpy.ndarray, allow_none=True,
                desc='Max relative step for parameters iterated by solver.')
            self.options.declare('history', False, dtype=bool, allow_none=False,
                desc='Request saving the resolution trace.')

        elif self.__method == NonLinearMethods.POWELL:
            self.__option_aliases = {
                'tol': 'xtol',
                'max_eval': 'maxfev',
            }

            self.options.declare('xtol', 1.0e-7, dtype=float,
                desc="The calculation will terminate if the relative error between two consecutive iterations is at most tol.")
            self.options.declare('maxfev', 0, dtype=int,
                desc='The maximum number of calls to the function. If zero, assumes 100 * (N + 1),'
                    ' where N is the number of elements in x0.')
            self.options.declare('eps', None, dtype=float, allow_none=True,
                desc='A suitable step length for the forward-difference approximation of the Jacobian (for fprime=None).'
                    ' If eps is less than machine precision u, it is assumed that the relative errors in the'
                    ' functions are of the order of u.')
            self.options.declare('factor', 0.1, dtype=float, lower=0.1, upper=100.,
                desc='A parameter determining the initial step bound factor * norm(diag * x). Should be in the interval [0.1, 100].')

        elif self.__method == NonLinearMethods.BROYDEN_GOOD:
            self.__option_aliases = {
                'tol': 'fatol',
                'num_iter': 'nit',
                'max_iter': 'maxiter',
                'min_rel_step': 'xtol',
                'min_abs_step': 'xatol',
            }

            self.options.declare('nit', 100, dtype=int,
                desc='Number of iterations to perform. If omitted (default), iterate unit tolerance is met.')
            self.options.declare('maxiter', 200, dtype=int,
                desc='Maximum number of iterations to make. If more are needed to meet convergence, NoConvergence is raised.')
            self.options.declare('disp', False, dtype=bool,
                desc='Print status to stdout on every iteration.')
            self.options.declare('fatol', 6e-6, dtype=float,
                desc='Absolute tolerance (in max-norm) for the residual. If omitted, default is 6e-6.')
            self.options.declare('line_search', 'armijo', dtype=str, allow_none=True,
                desc='Which type of a line search to use to determine the step '
                    'size in the direction given by the Jacobian approximation. Defaults to ‘armijo’.')
            self.options.declare('jac_options', {'reduction_method': 'svd'},
                dtype=dict, allow_none=True,
                desc='Options for the respective Jacobian approximation. restart, simple or svd')

        self._filter_options(options, self.__option_aliases)
