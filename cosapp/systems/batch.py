from __future__ import annotations
import warnings
import pandas
from typing import TYPE_CHECKING, Optional, Union, Sequence

from cosapp.utils.helpers import check_arg
from cosapp.utils.find_variables import (
    find_variable_names,
    natural_varname,
    SearchPattern,
)
from cosapp.core.execution import (
    Pool,
    Task,
    Job,
    Batch,
    TaskAction,
    TaskResponseStatus,
    FunctionCallBehavior,
    ExecutionPolicy,
    ExecutionType,
)
if TYPE_CHECKING:
    from cosapp.systems import System


class BatchRunner:
    """Batch execution framework for systems.

    Parameters
    ----------
    - system [System]:
        The system of interest.
    - output_varnames [Sequence[str]]:
        Variable names to be captured as inputs.
    - policiy [ExecutionPolicy]:
        Execution policy for batch runs.
    """
    def __init__(
        self,
        system: System,
        output_varnames: Sequence[str],
        policy: Optional[ExecutionPolicy] = None,
    ):
        from cosapp.systems import System
        check_arg(system, "system", System)
        self.__system = system
        self.__output_varnames: tuple[str] = tuple()
        self.__exec_policy = ExecutionPolicy(1, ExecutionType.SINGLE_THREAD)
        if output_varnames:
            self.output_varnames = output_varnames
        if policy:
            self.execution_policy = policy

    @classmethod
    def from_output_pattern(
        cls,
        system: System,
        includes: SearchPattern,
        excludes: SearchPattern,
        policy: Optional[ExecutionPolicy] = None,
    ) -> BatchRunner:
        """Factory constructing a `BatchRunner` instance from
        specific search patterns for output variables.
        """
        runner = cls(system, (), policy)
        runner.find_outputs(includes, excludes)
        return runner

    @property
    def system(self) -> System:
        """System executed by the runner"""
        return self.__system

    @property
    def output_varnames(self):
        """"tuple[str]: output variable names captured by runner"""
        return self.__output_varnames

    @output_varnames.setter
    def output_varnames(self, varnames: Sequence[str]):
        self.__output_varnames = tuple(varnames)

    def find_outputs(
        self,
        includes: SearchPattern = "*",
        excludes: SearchPattern = [],
    ) -> None:
        """Set output variable names matching inclusion and exclusion patterns"""
        self.output_varnames = find_variable_names(
            self.__system,
            includes=includes,
            excludes=excludes,
            inputs=False,
        )

    @property
    def execution_policy(self) -> ExecutionPolicy:
        """Default execution policy of the batch runner."""
        return self.__exec_policy

    @execution_policy.setter
    def execution_policy(self, execution_policy: ExecutionPolicy):
        check_arg(execution_policy, "execution_policy", ExecutionPolicy)
        self.__exec_policy = execution_policy

    def run(
        self,
        inputs: pandas.DataFrame,
        policy: Optional[ExecutionPolicy] = None,
    ) -> dict[str, list]:
        """Batch execution.

        Parameters
        ----------
        - inputs [pandas.DataFrame]:
            Dataframe containing batches of input values, referenced by variable names.
        - policy [ExecutionPolicy, optional]:
            Execution policy for the job. If not prescribed (default),
            uses the runner policy set at initialization.

        Returns
        -------
        dict[str, list]: output data, stored as a dictionary of the kind {varname, list[values]}.
        """
        if not policy:
            policy = self.__exec_policy
        else:
            check_arg(policy, "policy", ExecutionPolicy)

        system = self.__system
        output_varnames = self.__output_varnames

        if not output_varnames and not inputs.empty:
            warnings.warn(
                "No output variable names defined for batch run. "
                "Use `find_outputs` to set output variable names.",
                UserWarning,
            )
            outputs = {}

        elif policy.is_sequential() or inputs.empty:
            outputs = self._compute_outputs(system, inputs, output_varnames)

        else:
            def create_job(index_range: range):
                run = Task(
                    TaskAction.FUNC_CALL,
                    FunctionCallBehavior.RETURN_OBJECT,
                    (
                        self._compute_outputs,
                        (system, inputs.iloc[index_range], output_varnames),
                    ),
                )
                return Job(run)

            blocks = Batch.compute_blocks(len(inputs), policy.workers_count)
            batch = Batch(map(create_job, blocks))

            pool = Pool.from_policy(policy)
            outputs = self.empty_dataset(output_varnames)

            with pool.activate():
                pool.run_batch(batch)
                batch.join()
                error = self._gather_results(batch, outputs)

            if isinstance(error, Exception):
                raise error

        return outputs

    @staticmethod
    def empty_dataset(varnames: Sequence[str]) -> dict[str, list]:
        """Generate a dictionary mapping each variable name to an empty list."""
        return {name: [] for name in map(natural_varname, varnames)}

    @staticmethod
    def filter_headers(dataframe: pandas.DataFrame, inplace=True) -> pandas.DataFrame:
        """Apply `natural_varname` to dataframe column names"""
        mapping = {
            name: natural_varname(name)
            for name in dataframe.columns
        }
        dataframe.rename(mapping, axis=1, inplace=inplace)
        return dataframe

    @staticmethod
    def _compute_outputs(
        system: System,
        inputs: pandas.DataFrame,
        output_varnames: tuple[str],
    ):
        """Batch execution of `system` over input points.
        Used internally for single- and multi-processing executions.
        """
        outputs = BatchRunner.empty_dataset(output_varnames)
        for _, row in inputs.iterrows():
            for varname, value in row.items():
                setattr(system, varname, value)
            system.run_drivers()
            for varname, values in outputs.items():
                values.append(system[varname])
        return outputs

    @staticmethod
    def _gather_results(batch: Batch, outputs: dict[str, list]) -> Union[Exception, None]:
        """Gather results from batch processes into a single dataset `ouputs`.

        Returns
        -------
        `Exception`: First exception returned by a batch job, if any; else, `None` (normal execution).
        """
        for job in batch.jobs:
            status, data = job.tasks[-1].result
            if status != TaskResponseStatus.OK:
                return data  # should be an exception
            try:
                for varname, values in outputs.items():
                    values.extend(data[varname])
            except Exception as error:
                return error


def batch_run(
    system: System,
    inputs: pandas.DataFrame,
    output_varnames: Sequence[str],
    execution_policy: Optional[ExecutionPolicy] = None,
) -> dict[str, list]:
    """Convenience function to run a batch execution on a system.

    Parameters
    ----------
    - system [System]: The system to run.
    - inputs [pandas.DataFrame]: Dataframe containing input values.
    - output_varnames [Sequence[str]]: Variable names to capture as outputs.
    - execution_policy [Optional[ExecutionPolicy]]: Execution policy for the batch run.
        If not provided, defaults to single-thread execution.

    Returns
    -------
    dict[str, list]: Output data as a dictionary of lists.
    """
    runner = BatchRunner(system, output_varnames, policy=execution_policy)
    return runner.run(inputs)
