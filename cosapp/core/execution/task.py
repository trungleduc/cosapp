from __future__ import annotations
import weakref
from enum import IntEnum
from typing import (
    TYPE_CHECKING,
    Any, List, Tuple, Optional, Sequence,
    Union, Iterator, Generator,
)

from cosapp.utils.state_io import object__getstate__

if TYPE_CHECKING:
    from .worker import BaseWorker


class TaskAction(IntEnum):
    """Defines the task action."""
    FUNC_CALL = 0
    MEMOIZE = 1
    INTERRUPT = 2


class FunctionCallBehavior(IntEnum):
    """Defines a function call behavior."""
    EXECUTE = 0
    RETURN_OBJECT = 1
    STORE_RETURNED_OBJECT = 1 << 1
    ARGS_IN_STORAGE = 1 << 2
    CHAINED = 1 << 3


class TaskState(IntEnum):
    """Defines the task state."""
    CREATED = 0
    DISPATCHED = 1
    QUEUED = 2
    HANDLED = 3
    STARTED = 4
    FINISHED = 5


class TaskResponseStatus(IntEnum):
    """Defines the task response status code."""
    OK = 200
    CREATED = 201
    ACCEPTED = 202
    FUNCTION_CALL_RAISED = 400
    INVALID_STORAGE_ARGS = 401
    MISSING_STORED_OBJECT = 402
    INTERNAL_ERROR = 500


class TaskResultNotAvailableYet:
    """A dummy type representing a result not available yet."""
    ...


class Task:
    """An action to be executed by a worker."""

    __slots__ = (
        "_uid",
        "_action",
        "_options",
        "_data",
        "_result",
        "_state",
        "_worker",
        "_active",
        "_execution_count",
    )

    def __init__(self, action: TaskAction, options: int = 0, data: Any = ()):
        """Initializes an instance of a `Task`.

        Parameters
        ----------
        action_type : TaskActionType
            Type of action to perform
        options : int
            Options of the action
        data : Any
            Data to be passed to the worker (including function, and arguments)
        """
        self._uid: Optional[int] = None
        self._action: TaskAction = action
        self._options: int = options
        self._data: List[Any] = data
        self._result: Optional[Any] = None
        self._state: TaskState = TaskState.CREATED
        self._worker: weakref.ReferenceType[BaseWorker] = weakref.ref(lambda: None)
        self._active: bool = True
        self._execution_count: int = 0

    def __iter__(self):
        """Allows to unpack the object."""
        return iter((self._action, self._options, self._data))

    def __getstate__(self):
        """Defines custom `pickle` serialization."""
        _, slots = object__getstate__(self)
        slots.pop("_worker")
        return None, slots

    @property
    def result(self) -> Any:
        """Gets the result if available.

        This method will ask the worker to dequeue already available results.

        If the result is not yet available, the method will return a dummy
        type representing a result that is not available yet (non blocking).

        Returns
        -------
        Any
            Task result
        """
        if self._state == TaskState.FINISHED:
            return self._result

        worker = self._check_worker()
        worker.dequeue_results()

        if self._state != TaskState.FINISHED:
            return TaskResultNotAvailableYet

        return self._result

    @result.setter
    def result(self, value: Any) -> None:
        """Sets the result."""
        self._result = value

    def wait_for_result(self) -> Any:
        """Gets the result, waiting for it if not already available.

        This method will ask the worker to wait for results of a given
        task UID. The worker will dequeue incoming results and dispatch them
        to the tasks and will stop dequeuing when the expected result will
        be handled.

        The method will block if the worker has not yet processed the task.

        Returns
        -------
        Any
            Task result
        """
        if self._state == TaskState.FINISHED:
            return self._result

        worker = self._check_worker()
        worker.wait_for_results(self._uid)

        return self._result

    def join(self) -> None:
        """Joins this task."""
        self.wait_for_result()

    @property
    def state(self) -> TaskState:
        """Gets the state."""
        return self._state

    @state.setter
    def state(self, new_state: TaskState) -> None:
        """Sets the state."""
        self._state = new_state

    @property
    def worker(self) -> BaseWorker:
        """Gets the worker handling the task."""
        return self._worker()

    def _check_worker(self) -> BaseWorker:
        """Returns task worker, if any; otherwise, raises `RuntimeError`.
        For internal use only.
        """
        worker = self._worker()
        if worker is None:
            if self._state == TaskState.CREATED:
                raise RuntimeError(
                    "Task result should not be queried before being dispatched to a `Pool`"
                )
            else:
                raise RuntimeError(
                    "Result must be queried *before* stopping execution pool or deleting worker"
                )
        return worker

    @property
    def uid(self) -> int:
        """Gets the UID from the worker perspective."""
        return self._uid

    @uid.setter
    def uid(self, value: int) -> None:
        """Sets the UID from the worker perspective."""
        self._uid = value

    @property
    def active(self) -> int:
        """Gets whether the task is active or not."""
        return self._active

    @active.setter
    def active(self, value: bool) -> None:
        """Sets whether the task is active or not."""
        self._active = value

    def increment_execution_counter(self) -> None:
        """Increments the execution counter of this task."""
        self._execution_count += 1

    @property
    def execution_count(self) -> int:
        """Gets the execution counter of this task."""
        return self._execution_count


class Job:
    """A sequence of `Task`s to be executed by a worker.

    A job offers the guarantee to execute related tasks on the same worker.

    It does NOT transfer the result of a task to the next one by itself, but
    used with multiple tasks using `FunctionCallBehavior.CHAINED` this
    behavior can be achieved.
    """

    __slots__ = (
        "_tasks",
        "_worker",
    )

    def __init__(self, tasks: Union[Sequence[Task], Task]):
        """Initializes an instance of a `Job`.

        Parameters
        ----------
        tasks : Union[Sequence[Task], Task]
            Sequence of tasks grouped as a `Job`, or a single task
        """
        if isinstance(tasks, Task):
            self._tasks = [tasks]
        else:
            self._tasks = tasks

        self._worker: weakref.ReferenceType[BaseWorker] = weakref.ref(lambda: None)

    def __getstate__(self):
        """Defines custom `pickle` serialization."""
        _, slots = object__getstate__(self)
        slots.pop("_worker")
        return None, slots

    def __iter__(self):
        """Allows to iterate easily over the tasks."""
        return iter(self._tasks)

    def join(self) -> None:
        """Joins all tasks for this job."""
        self._tasks[-1].join()

    @property
    def tasks(self):
        """Gets the tasks."""
        return self._tasks

    @property
    def worker(self) -> Optional[BaseWorker]:
        """Gets the worker handling the job."""
        return self._worker()


class Batch:

    def __init__(
        self,
        jobs: Union[Sequence[Job], Job],
        block_min_size: int = 1,
    ):
        """Initializes an instance of a `Batch`.

        Parameters
        ----------
        jobs : Union[Sequence[Job], Job]
            Sequence of jobs grouped as a `Batch`, or a single job
        block_min_size: int
            The minimum size of a group of jobs to be dispatched on the same worker; default 1
        """
        self._jobs = self._make_jobs(jobs)
        self._size = len(self._jobs)
        self._block_min_size = max(block_min_size, 1)

    def new_with_same_affinity(self, jobs: Union[Sequence[Job], Job]):
        """Create and return a new batch of identical affinity, i.e.
        with jobs defined on the same worker pool.
        """
        jobs = self._make_jobs(jobs)

        if len(jobs) != len(self._jobs):
            raise RuntimeError(
                f"A new batch with same affinity must have the exact same jobs count"
                f"; {len(jobs)=} instead of {len(self._jobs)}."
            )

        for new_job, job in zip(jobs, self._jobs):
            new_job._worker = job._worker
            for task in new_job.tasks:
                task._worker = job._worker

        return Batch(jobs)

    @staticmethod
    def _make_jobs(jobs: Union[Sequence[Job], Job]) -> Tuple[Job]:
        if isinstance(jobs, Job):
            return (jobs,)
        else:
            return tuple(jobs)

    def __iter__(self) -> Iterator[Job]:
        """Allows to iterate easily over the jobs."""
        return iter(self._jobs)

    @property
    def jobs(self):
        """Returns all the jobs in the batch."""
        return self._jobs

    def get_blocks(self, block_count: int) -> Generator[range, None, None]:
        """
        Generates `block_count` index ranges covering the range of jobs in the batch.

        Parameters:
        -----------
        block_count [int]:
            Number of blocks (ranges) to generate.

        Returns:    
        --------
        Generator[range]
        """
        yield from self.compute_blocks(self._size, block_count, self._block_min_size)

    def join(self) -> None:
        """Joins all jobs for this batch."""
        for job in self._jobs:
            job.join()

    @staticmethod
    def compute_blocks(
        size: int,
        block_count: int,
        block_min_size: int = 1,
    ) -> Generator[range, None, None]:
        """
        Generates `block_count` index ranges covering range(`size`).

        Parameters:
        -----------
        size [int]:
            Extent of the total range to cover.
        block_count [int]:
            Number of blocks (ranges) to generate.
        block_min_size [int, optional]:
            Minimum size of individual blocks. Defaults to 1.

        Returns:    
        --------
        Generator[range]
        """
        if block_count > size:
            block_count = size

        if size < block_min_size * block_count:
            block_count = max(1, size // block_min_size)

        block_size, remainder = divmod(size, block_count)

        def start(index):
            return index * block_size + min(index, remainder)

        def end(index):
            return size if (index == block_count - 1) else start(index + 1)
        
        for i in range(block_count):
            yield range(start(i), end(i))
