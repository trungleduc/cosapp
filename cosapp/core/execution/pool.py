from __future__ import annotations

import multiprocessing as mp
import weakref
from dataclasses import dataclass
from enum import Enum
from typing import List, NamedTuple, Optional, Sequence, Union, Dict, Any, Generator
from contextlib import contextmanager

from cosapp.utils.state_io import object__getstate__

from .comms import PoolComms, WorkerComms
from .context import is_fork_available
from .task import Batch, Job, Task, TaskAction, TaskState
from .worker import BaseWorker, ForkServerWorker, ForkWorker, SpawnWorker


class ExecutionType(Enum):
    SINGLE_THREAD = 0
    MULTI_PROCESSING = 1
    MULTI_THREADING = 2

    def __json__(self):
        return {"value": self.value}


class WorkerStartMethodDetails(NamedTuple):
    context_type: mp.context.BaseContext
    worker_type: BaseWorker


class WorkerStartMethod(Enum):

    SPAWN = WorkerStartMethodDetails(
        context_type=mp.context.SpawnContext,
        worker_type=SpawnWorker,
    )

    if is_fork_available():
        FORK = WorkerStartMethodDetails(
            context_type=mp.context.ForkContext,
            worker_type=ForkWorker,
        )

        FORKSERVER = WorkerStartMethodDetails(
            context_type=mp.context.ForkServerContext,
            worker_type=ForkServerWorker,
        )
        AUTO = FORK
    else:
        AUTO = SPAWN

    def make_context(self) -> mp.context.BaseContext:
        return self.value.context_type()

    @property
    def worker_type(self) -> BaseWorker:
        return self.value.worker_type

    def __json__(self):
        if self == WorkerStartMethod.SPAWN:
            return {"value": "SPAWN"}
        elif self == WorkerStartMethod.FORK:
            return {"value": "FORK"}
        elif self == WorkerStartMethod.FORKSERVER:
            return {"value": "FORKSERVER"}
        elif self == WorkerStartMethod.AUTO:
            return {"value": "AUTO"}

        raise ValueError("`WorkerStartMethod` enum value is not handled")


@dataclass
class ExecutionPolicy:
    workers_count: int
    execution_type: ExecutionType
    start_method: WorkerStartMethod = WorkerStartMethod.AUTO

    def __json__(self) -> Dict[str, Any]:
        """Creates a JSONable dictionary representation of the object.

        Returns
        -------
        Dict[str, Any]
            The dictionary
        """
        return object__getstate__(self).copy()

    def is_sequential(self) -> bool:
        """Returns whether the policy is sequential or not."""
        return (
            self.workers_count == 1
            or self.execution_type == ExecutionType.SINGLE_THREAD
        )


class Pool:

    def __init__(
        self,
        workers_count: int,
        execution_type: ExecutionType,
        start_method: WorkerStartMethod = WorkerStartMethod.AUTO,
    ):
        self._size = size = workers_count
        self._type = execution_type
        self._worker_start_method = start_method

        self._workers: list[BaseWorker] = [None] * size
        self._context: mp.context.BaseContext = None
        self._worker_type = None
        self._set_worker_type_and_context()

        self._comms: PoolComms = PoolComms(self._context)
        self._workers_comms: List[WorkerComms] = [None] * size

        self._tasks_count = 0

    @staticmethod
    def from_policy(
        policy: ExecutionPolicy
    ) -> Pool:
        return Pool(policy.workers_count, policy.execution_type, policy.start_method)

    def _set_worker_type_and_context(self):

        if self._type == ExecutionType.MULTI_PROCESSING:
            method = self._worker_start_method
            self._context = method.make_context()
            self._worker_type = method.worker_type

        elif self._type == ExecutionType.MULTI_THREADING:
            raise NotImplementedError(f"Multithreading is not implemented yet")

        if self._context is None:
            raise ValueError("Invalid pool type or worker start method")

    def _create_worker(self, id: int) -> int:
        self._workers_comms[id] = worker_comms = WorkerComms(self._context, self._comms)
        self._workers[id] = self._worker_type(id, worker_comms)
        self._workers[id].start()

    def __del__(self):
        """Stops the pool when deleting this object."""
        self.stop()

    def start(self):
        """Starts the pools."""
        for id in range(self._size):
            self._create_worker(id)

    def get_worker(self, id: int):
        return self._workers[id]

    def run_task(self, task: Task, worker_id: Optional[int] = None) -> Task:
        """Dispatches a `Task` on a worker.

        Parameters
        ----------
        task : Task
            Task to run
        worker_id : Optional[int]
            ID of the worker that will handle the job; default None

        Returns
        -------
        Task
            Task to run
        """
        task.state = TaskState.CREATED

        if not task.active:
            return task

        if worker_id and not (0 <= worker_id < self._size):
            raise ValueError(
                f"Invalid worker ID {worker_id} for `Pool` with size {self._size}"
            )

        if worker_id:
            worker = self._workers[worker_id]
        elif task.worker:
            worker = task.worker
        else:  # TODO: improve the dispatch given the current workload of each worker
            worker = self._workers[self._tasks_count % self._size]

        task._worker = weakref.ref(worker)
        task.state = TaskState.DISPATCHED

        return worker.run_task(task)

    def run_job(self, job: Job, worker_id: Optional[int] = None) -> Job:
        """Dispatches a `Job` (a sequence of `Task`s) on a worker.

        Parameters
        ----------
        job : Job
            Job to run
        worker_id : Optional[int]
            ID of the worker that will handle the job; default None

        Returns
        -------
        Job
            Job to run
        """
        for task in job.tasks:
            task.state = TaskState.CREATED

        if worker_id and not (0 <= worker_id < self._size):
            raise ValueError(
                f"Invalid worker ID {worker_id} for `Pool` with size {self._size}"
            )

        if worker_id:
            worker = self._workers[worker_id]
        elif job.worker:
            worker = job.worker
        else:  # TODO: improve the dispatch given the current workload of each worker
            worker = self._workers[self._tasks_count % self._size]

        if worker is not job.worker:
            job._worker = weakref.ref(worker)
            for task in job.tasks:
                task._worker = weakref.ref(worker)

        for task in job.tasks:
            if task.active:
                task.state = TaskState.DISPATCHED
                worker.run_task(task)

        return job

    def run_batch(
        self, batch: Union[Batch, Sequence[Job], Job], min_block_size: int = 0
    ) -> Batch:
        """Dispatches a batch of `Job`s on workers.

        The jobs must be in a consistent state: either they are all newly created, or
        they all have already ran (and thus have affinity with the worker they ran on).

        Parameters
        ----------
        jobs : Union[Sequence[Job], Job]
            Jobs to run
        min_block_size : int
            The minimum number of jobs to be executed on the same worker; default 0

        Returns
        -------
        Sequence[Job]
            Jobs to run
        """
        if isinstance(batch, (Job, Sequence)):
            batch = Batch(batch, min_block_size)

        jobs_have_workers = [job.worker is not None for job in batch.jobs]

        if all(jobs_have_workers):
            for job in batch.jobs:
                self.run_job(job)
        elif not any(jobs_have_workers):
            for runner_id, block in enumerate(batch.get_blocks(self._size)):
                for idx in block:
                    self.run_job(batch.jobs[idx], runner_id)
        else:
            raise RuntimeError(
                "Invalid mix of new and already ran jobs in the same batch."
            )

        return batch

    def stop(self):
        """Stops the workers."""
        workers_to_stop = [
            (worker_id, worker)
            for worker_id, worker in enumerate(self._workers)
            if worker is not None and worker.is_alive()
        ]

        for worker_id, worker in workers_to_stop:
            worker.run_task(Task(TaskAction.INTERRUPT, None, ()))
        for worker_id, worker in workers_to_stop:
            worker.dequeue_results()
            worker.join()
            worker.close()
            self._workers[worker_id] = None

    @contextmanager
    def activate(self) -> Generator[None, Any, Any]:
        """Provides a context manager to start and stop the pool."""
        self.start()
        yield
        self.stop()

    @property
    def workers(self) -> List[BaseWorker]:
        """Gets the workers."""
        return self._workers
