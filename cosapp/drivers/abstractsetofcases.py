from __future__ import annotations

import abc
import copy
from typing import Any, Iterable, Optional, Dict, Tuple

from cosapp.drivers.driver import Driver, System
from cosapp.utils.logging import logging
from cosapp.core.execution import (
    Pool,
    ExecutionPolicy,
    ExecutionType,
    Task,
    Job,
    Batch,
    TaskAction,
    TaskResponseStatus,
    FunctionCallBehavior,
    ops,
)

logger = logging.getLogger(__name__)


# TODO
# [ ] Quid for vector variables
class AbstractSetOfCases(Driver):
    """
    This driver builds a set of cases from a list

    Parameters
    ----------
    name : str
        Name of the driver
    owner : System, optional
        :py:class:`~cosapp.systems.system.System` to which driver belongs; defaults to `None`
    **kwargs : Any
        Keyword arguments will be used to set driver options
    """

    __slots__ = ("cases", "_transients_variables", "_execution_policy")

    def __init__(
        self,
        name: str,
        owner: Optional[System] = None,
        execution_policy: ExecutionPolicy = ExecutionPolicy(workers_count=1, execution_type=ExecutionType.SINGLE_THREAD),
        **options
    ) -> None:
        """Initialize driver

        Parameters
        ----------
        name: str, optional
            Name of the `Module`
        owner: System, optional
            :py:class:`~cosapp.systems.system.System` to which driver belongs; defaults to `None`
        **kwargs:
            Additional keywords arguments forwarded to base class.
        """
        super().__init__(name, owner, **options)
        # TODO Fred - is this not too much oriented for MonteCarlo or DoE? What about a mission for which input
        # variables may not be the same on all points.
        self.cases = None  # type: Optional[Iterable[Any]]
            # desc="List of cases to be carried out."
        self._transients_variables = {} 
        self._execution_policy: ExecutionPolicy = execution_policy  # Execution policy to use for computation

    def _precase(self, index: int, case: Any):
        """Hook to be called before running each case.
        
        Parameters
        ----------
        case_idx : int
            Index of the case
        case : Any
            Parameters for this case
        """
        self.status = ""
        self.error_code = "0"

    @abc.abstractmethod
    def _build_cases(self) -> None:
        """Generator of cases."""
        pass

    def _postcase(self, index: int, case: Any):
        """Hook to be called after running each case.
        
        Parameters
        ----------
        case_idx : int
            Index of the case
        case : Any
            Parameters for this case
        """
        if (recorder := self._recorder):
            recorder.record_state(index, self.status, self.error_code)

    def setup_run(self):
        """Actions performed prior to the `Module.compute` call."""
        super().setup_run()
        self._build_cases()
        time_problem = self._owner.assembled_time_problem()
        self._transients_variables = {
            varname: copy.copy(transient.value)
            for varname, transient in time_problem.transients.items()
        }

    def run_children(self) -> None:
        """Runs all driver children.
        """
        for child in self.children.values():
            child.run_once()
            if len(child.status) > 0:
                self.status = child.status
            if child.error_code != "0":
                self.error_code = child.error_code

    def compute(self) -> None:
        """Contains the customized `Module` calculation, to execute after children.
        """
        exec_policy = self._execution_policy
        if exec_policy.is_sequential():
            self._compute_sequential(self)
        else:
            self._compute_parallel()

    @staticmethod
    def _compute_sequential(driver: AbstractSetOfCases) -> None:
        """Contains the customized `Module` calculation, to execute after children.
        """
        for case_idx, case in enumerate(driver.cases):
            if len(case) > 0:
                driver._precase(case_idx, case)
                driver.run_children()
                driver._postcase(case_idx, case)

    @staticmethod
    def _get_driver(system: System, name: str) -> AbstractSetOfCases:
        return system.drivers[name]

    @staticmethod
    def _prepare_recorders(
        driver: AbstractSetOfCases, exec_type: ExecutionType, chunk_id: int
    ) -> AbstractSetOfCases:
        for d in driver.tree():
            rec = d.recorder
            if rec:
                rec._enable_parallel_execution(exec_type, chunk_id)
                rec.clear()

        return driver
    
    @staticmethod
    def _modify_cases(driver: AbstractSetOfCases, rng) -> None:
        lrng = list(rng)
        start, stop = lrng[0], lrng[-1] + 1
        driver.cases = driver.cases[start:stop]
        return driver

    @staticmethod
    def _get_recorders_raw_data(driver: AbstractSetOfCases) -> Dict[str, Any]:
        """Gets recorders raw data.
        
        Returns
        -------
        Dict[str, Any]
            Recorders raw data dict, where keys are the drivers' names and values
            are their recorders' raw data 
        """
        def make_items(driver: Driver) -> Tuple[str, Any]:
            if (recorder := driver.recorder):
                value = recorder._raw_data
            else:
                value = None
            return (driver.name, value)

        return dict(map(make_items, driver.tree(downwards=True)))


    @staticmethod
    def _compute_and_return_results(driver: AbstractSetOfCases) -> Dict[str, Any]:
        """Computes the driver and returns recorders raw data.
        
        Returns
        -------
        Dict[str, Any]
            Recorders raw data dict, where keys are the drivers' names and values
            are their recorders' raw data 
        """
        AbstractSetOfCases._compute_sequential(driver)
        return AbstractSetOfCases._get_recorders_raw_data(driver)

    def _get_parallel_results(self, batch: Batch) -> Optional[Exception]:
        """Gets and dispatches results from parallel execution.
        
        Parameters
        ----------
        batch: Batch
            Batch of jobs from which the results must be gathered

        Returns
        -------
        Optional[Exception]
            Exceptions raised workers' side if any
        """
        mc_tree = list(self.tree(downwards=True))
        for job in batch.jobs:
            status, data = job.tasks[-1].result
            if status != TaskResponseStatus.OK:
                return data

            for driver in mc_tree:
                rec_data = data.get(driver.name, None)
                if rec_data:
                    driver.recorder._batch_record(rec_data)

    def _compute_parallel(self) -> None:
        """Computes the driver in parallel."""
        exec_policy = self._execution_policy
        pool = Pool.from_policy(exec_policy)

        def create_job(id_and_range: Tuple[int, range]):
            worker_id, index_range = id_and_range
            store_system = Task(
                TaskAction.FUNC_CALL,
                FunctionCallBehavior.STORE_RETURNED_OBJECT,
                (ops.return_arg, (self._owner,)),
            )
            get_mc = Task(
                TaskAction.FUNC_CALL,
                FunctionCallBehavior.CHAINED,
                (self._get_driver, (self.name,)),
            )
            prepare_recorders = Task(
                TaskAction.FUNC_CALL,
                FunctionCallBehavior.CHAINED,
                (self._prepare_recorders, (pool._type, worker_id)),
            )
            modify_cases = Task(
                TaskAction.FUNC_CALL,
                FunctionCallBehavior.CHAINED,
                (self._modify_cases, (index_range,)),
            )
            compute_and_return_results = Task(
                TaskAction.FUNC_CALL,
                FunctionCallBehavior.CHAINED | FunctionCallBehavior.RETURN_OBJECT,
                (self._compute_and_return_results, ()),
            )
            return Job([store_system, get_mc, prepare_recorders, modify_cases, compute_and_return_results])
        
        blocks = Batch.compute_blocks(len(self.cases), exec_policy.workers_count)
        batch = Batch(map(create_job, enumerate(blocks)))

        with pool.activate():
            pool.run_batch(batch)
            batch.join()
            error = self._get_parallel_results(batch)

        if error:
            raise error

    def run_once(self) -> None:
        """Run the driver once.
        
        Do not call children `run_once` method, this is handled by the derived drivers.
        """
        with self.log_context(" - run_once"):
            if self.is_active():
                self._precompute()

                logger.debug(f"Call {self.name}.compute_before()")
                self.compute_before()

                logger.debug(f"Call {self.name}.compute()")
                self._compute_calls += 1
                self.compute()

                self._postcompute()
                self.computed.emit()
            
            else:
                logger.debug(f"Skip {self.name} execution - Inactive")
