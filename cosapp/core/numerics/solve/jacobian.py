from abc import abstractmethod
from typing import Any, Dict, NamedTuple, Optional, Set, Callable, Iterable

import numpy

from cosapp.core.execution import (
    Batch,
    ExecutionPolicy,
    ExecutionType,
    FunctionCallBehavior as FunctionBehavior,
    Job,
    Pool,
    Task,
    TaskAction,
    ops,
)
from cosapp.utils.json import jsonify
from cosapp.utils.logging import LogLevel, logging
from cosapp.utils.options_dictionary import HasOptions
from cosapp.utils.state_io import object__getstate__

logger = logging.getLogger(__name__)


class JacobianStats(NamedTuple):
    """Statistics of a Jacobian evaluation."""

    partial_updates: int
    broyden_updates: int
    full_updates: int


class AbstractJacobianEvaluation(HasOptions):
    """Abstract base class for Jacobian evaluation."""

    def __init__(self):
        super().__init__()
        self._fun = None
        self._fargs = None
        self._log_level: LogLevel = LogLevel.DEBUG

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

    def bind_residue_function(self, f, args):
        """Binds the residue function and its arguments."""
        self._fun = f
        self._fargs = args

    def unbind_residue_function(self):
        """Unbinds the residue function and its arguments."""
        self._fun = None
        self._fargs = None

    def _fresidues(self, x: numpy.ndarray) -> numpy.ndarray:
        """Calls the residue function."""
        return self._fun(x, *self._fargs)

    @abstractmethod
    def __call__(self, x, *, pool: Optional[Pool] = None, **kwargs):
        """Performs a Jacobian evaluation."""
        pass

    @abstractmethod
    def get_stats(self) -> JacobianStats:
        """Gets Jacobian evaluation statistics."""
        pass

    @abstractmethod
    def reset_stats(self) -> None:
        """Resets Jacobian evaluation statistics."""
        pass


class FfdJacobianEvaluation(AbstractJacobianEvaluation):
    """Forward finite-difference Jacobian evaluation."""

    def __init__(
        self,
        eps: float = 2 ** (-16),
        partial_jac: bool = True,
        p_jac_tries: int = 10,
        execution_policy = ExecutionPolicy(
            workers_count=1,
            execution_type=ExecutionType.SINGLE_THREAD,
        ),
    ):
        super().__init__()

        self._eps = eps
        self._partial_jac = partial_jac
        self._partial_jac_tries = p_jac_tries
        self._execution_policy = execution_policy
        self._pool: Optional[Pool] = None

        self._x_copy = numpy.empty(0)

        self.reset_stats()

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
        state = super().__getstate__().copy()
        state.pop("_pool", None)
        return state

    def _declare_options(self) -> None:
        """Declares options."""
        super()._declare_options()
        options = self._options

        # Note: use a power of 2 for `eps`, to guaranty machine-precision accurate gradients in linear problems
        options.declare(
            "eps",
            2 ** (-16),
            dtype=float,
            allow_none=True,
            lower=2 ** (-30),
            desc="Relative step length for the forward-difference approximation of the Jacobian.",
        )
        options.declare(
            "partial_jac",
            True,
            dtype=bool,
            allow_none=False,
            desc="Defines if partial Jacobian updates can be computed before a complete Jacobian matrix update.",
        )
        options.declare(
            "partial_jac_tries",
            10,
            dtype=int,
            allow_none=False,
            lower=1,
            upper=10,
            desc="Defines how many partial Jacobian updates can be tried before a complete Jacobian matrix update.",
        )

    def _set_options(self) -> None:
        """Sets options from the current state."""
        for key, val in self._options.items():
            setattr(self, f"_{key}", val)

    def reset_stats(self) -> None:
        """Resets Jacobian evaluation statistics."""
        self._consecutive_p_jac_counter = 0
        self._partial_updates_counter = 0
        self._broyden_updates_counter = 0
        self._full_updates_counter = 0

    def get_stats(self) -> JacobianStats:
        """Gets Jacobian evaluation statistics."""
        return JacobianStats(
            self._partial_updates_counter,
            self._broyden_updates_counter,
            self._full_updates_counter,
        )

    @staticmethod
    def _set_worker_jac(res_size: int, unknown_size: int) -> numpy.ndarray:
        """Sets the default value of Jacobian matrix."""
        return numpy.zeros((res_size, unknown_size))

    def setup(self, size: int) -> None:
        """Performs setup of the evaluation method."""
        if self._execution_policy.is_sequential():
            self._sequential_setup(size)
        else:
            self._parallel_setup(size)

    def teardown(self) -> None:
        """Performs clean-up of the evaluation method."""
        if not self._execution_policy.is_sequential():
            self._pool.stop()

    def _sequential_setup(self, size: int) -> None:
        """Performs setup of the evaluation method for sequential execution."""
        self._x_copy = numpy.zeros(size)
        self._r0 = numpy.zeros(size)
        self._x_indices_to_update: list[int] = []
        self._size = size

        self.reset_stats()

    def _parallel_setup(self, size: int) -> None:
        """Performs setup of the evaluation method for parallel execution."""
        fresidue = self._fun
        if not callable(fresidue):
            raise RuntimeError(
                "Residue function and args must be bound before parallel setup."
            )
        self._size = size
        self._sequential_setup(size)

        def make_setup_job(rng: range):
            tasks = [
                Task(
                    TaskAction.FUNC_CALL,
                    FunctionBehavior.STORE_RETURNED_OBJECT,
                    (ops.return_arg, (fresidue,)),
                ),
                Task(
                    TaskAction.MEMOIZE,
                    data=0,
                ),
                Task(
                    TaskAction.FUNC_CALL,
                    FunctionBehavior.EXECUTE,
                    (self._set_worker_jac, (size, len(rng))),
                ),
                Task(
                    TaskAction.MEMOIZE,
                    data=1,
                ),
            ]
            return Job(tasks)

        self._pool = pool = Pool.from_policy(self._execution_policy)
        blocks = list(Batch.compute_blocks(size, pool._size, 1))
        setup_batch = Batch(map(make_setup_job, blocks))
        pool.start()
        pool.run_batch(setup_batch).join()

        def create_job(rng: range):
            return Job(
                Task(
                    TaskAction.FUNC_CALL,
                    FunctionBehavior.ARGS_IN_STORAGE | FunctionBehavior.RETURN_OBJECT,
                    (
                        self._compute_jacobian,
                        (
                            (0, 1),
                            self._x_copy,
                            self._r0,
                            rng,
                            self._x_indices_to_update,
                            self._eps,
                        ),
                    ),
                )
            )

        self._eval_batch = setup_batch.new_with_same_affinity(map(create_job, blocks))

    def __call__(
        self,
        x0: numpy.ndarray,
        *,
        r0: numpy.ndarray = None,
        jac: numpy.ndarray = None,
        res_index_to_update: Optional[Set[int]] = None,
        dx: Optional[numpy.ndarray] = None,
        dr: Optional[numpy.ndarray] = None,
        broyden_only: bool = False,
        **kwargs,
    ) -> numpy.ndarray:
        """Performs a Jacobian evaluation."""
        size = x0.size
        x_indices_to_update = self._x_indices_to_update

        if r0 is None:
            r0 = self._fresidues(x0)

        self._r0[:] = r0

        if (
            self._consecutive_p_jac_counter > self._partial_jac_tries
            or not self._partial_jac
        ):
            res_index_to_update = range(x0.size)
            self._consecutive_p_jac_counter = 0
        else:
            self._consecutive_p_jac_counter += 1

        if jac is None or jac.shape != (size, size):
            jac = numpy.zeros((size, size), dtype=float)
            x_indices_to_update.clear()
            x_indices_to_update.extend(range(size))
        
        elif broyden_only or (
            not res_index_to_update
            and res_index_to_update is not None
            and dx is not None
            and dr is not None
        ):
            self._broyden_update(jac, dx, dr)
            return jac
        
        else:
            unique_res_index_to_update = set()
            for i in res_index_to_update:
                unique_res_index_to_update.update(
                    j for j in range(size) if jac[i, j] != 0
                )
            x_indices_to_update.clear()
            x_indices_to_update.extend(unique_res_index_to_update)

        if self._execution_policy.is_sequential():
            self._sequential_evaluation(jac, x0, r0, x_indices_to_update)
        else:
            self._parallel_evaluation(jac, x0)

        self._update_counters()
        return jac

    def _sequential_evaluation(
        self,
        jac: numpy.ndarray,
        x0: numpy.ndarray,
        r0: numpy.ndarray,
        x_indices_to_update: Iterable[int],
    ) -> None:
        """Performs a Jacobian evaluation in sequential execution."""
        x_copy = x0.copy()

        for j in x_indices_to_update:
            delta = self._eps
            if abs(x_copy[j]) >= abs(self._eps):
                delta = x_copy[j] * self._eps
            x_copy[j] += delta
            logger.debug(f"Perturb unknown {j}")
            perturbation = self._fresidues(x_copy)
            jac[:, j] = (perturbation - r0) * (1 / delta)
            x_copy[j] = x0[j]

    @staticmethod
    def _compute_jacobian(
        fresidue: Callable,
        jac: numpy.ndarray,
        x: numpy.ndarray,
        r0: numpy.ndarray,
        rng: Iterable[int],
        x_indices_to_update: Iterable[int],
        eps: float,
    ) -> numpy.ndarray:
        """Evaluates Jacobian matrix in a subset of directions."""
        for idx, j in enumerate(rng):
            if j in x_indices_to_update:
                xj = x[j]
                delta = eps
                if abs(x[j]) >= abs(eps):
                    delta = x[j] * eps
                x[j] += delta
                logger.debug(f"Perturb unknown {j}")
                residue = fresidue(x)
                jac[:, idx] = (residue - r0) * (1 / delta)
                x[j] = xj

        return jac

    def _parallel_evaluation(self, jac: numpy.ndarray, x0: numpy.ndarray) -> None:
        """Performs a Jacobian evaluation in parallel execution."""
        self._x_copy[...] = x0
        eval_batch = self._eval_batch
        self._pool.run_batch(eval_batch).join()
        jac[:] = numpy.concatenate(
            [job.tasks[-1].result[1] for job in eval_batch.jobs], axis=1
        )

    def _broyden_update(self, jac: numpy.ndarray, dx: numpy.ndarray, dr: numpy.ndarray) -> None:
        """Updates the Jacobian matrix with a good Broyden method.

        Source: https://nickcdryan.com/2017/09/16/broydens-method-in-python/
        """
        # logger.log(log_level, f"Broyden update")
        self._broyden_updates_counter += 1
        corr = numpy.outer(dr - jac.dot(dx), dx) / dx.dot(dx)
        jac += corr

    def _update_counters(self) -> None:
        """Updates the internal counters."""
        n_partial = len(self._x_indices_to_update)
        if n_partial == self._size:
            logger.log(self._log_level, "Jacobian matrix: full update")
            self._full_updates_counter += 1
        elif self._x_indices_to_update:
            self._partial_updates_counter += 1
            logger.log(
                self._log_level,
                f"Jacobian matrix: {n_partial} over {self._size} derivative(s) updated",
            )
