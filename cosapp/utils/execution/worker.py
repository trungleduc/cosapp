import multiprocessing as mp
import signal
from queue import Queue
from typing import Any, Collection

from cosapp.utils.state_io import object__getstate__

from .comms import WorkerComms
from .context import is_fork_available, is_running_windows
from .task import (
    FunctionCallBehavior,
    Task, TaskState,
    TaskAction,
    TaskResponseStatus,
)


class StopWorkerError(Exception):
    """Exception used to kill a worker"""
    pass


class InterruptWorkerError(Exception):
    """Exception used to interrupt a worker"""
    pass


class BaseWorker:
    """Base class for all workers."""

    def __init__(self, id: int, comms: WorkerComms):
        """Initializes a worker object.

        Parameters
        ----------
        id : int
            Identifier
        comms: WorkerComms
            Communication gateway
        """
        super().__init__()

        self._id = id
        self._task_counter: int = 0
        self._storage = {}
        self._comms = comms
        self._received_tasks: Queue[Task] = Queue()

    def __getstate__(self):
        """Defines custom `pickle` serialization."""
        d = object__getstate__(self).copy()
        d.pop("_received_tasks")
        return d

    def _set_signal_handlers(self) -> None:
        """Sets signal handlers for graceful shutdown."""
        # TODO: review signal handlers

        if not is_running_windows():
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            signal.signal(signal.SIGHUP, self._on_kill_exit_gracefully)
            signal.signal(signal.SIGTERM, self._on_kill_exit_gracefully)
            signal.signal(signal.SIGUSR1, self._on_exception_exit_gracefully)

    def _on_kill_exit_gracefully(self, *_) -> None:
        # TODO: implement this method
        return

    def _on_exception_exit_gracefully(self, *_) -> None:
        # TODO: implement this method
        raise StopWorkerError

    def _on_exception_exit_gracefully_windows(self) -> None:
        # TODO: implement this method
        raise StopWorkerError

    def run(self) -> None:
        """Runs tasks received from parent process."""
        self._set_signal_handlers()

        previous_result: Any = None  # allow tasks chaining

        while True:
            action, opts, data = self._comms.get_task()
            self.task_done()

            if action == TaskAction.FUNC_CALL:
                store = opts & FunctionCallBehavior.STORE_RETURNED_OBJECT
                return_result = opts & FunctionCallBehavior.RETURN_OBJECT
                args_in_storage = opts & FunctionCallBehavior.ARGS_IN_STORAGE
                chained = opts & FunctionCallBehavior.CHAINED
                func, args = data

                storage_args = []
                if chained:
                    if isinstance(previous_result, (list, tuple)):
                        storage_args = previous_result
                    else:
                        storage_args = [previous_result]

                if args_in_storage:
                    largs = list(args)
                    try:
                        storage_ids = largs.pop(0)
                        storage_args.extend(self._storage[obj_id] for obj_id in storage_ids)
                    except KeyError as error:
                        self.add_result(TaskResponseStatus.MISSING_STORED_OBJECT, error)
                        continue
                    except Exception as error:
                        self.add_result(TaskResponseStatus.INVALID_STORAGE_ARGS, error)
                        continue
                else:
                    largs = args

                try:
                    result = previous_result = func(*storage_args, *largs)
                except Exception as error:
                    self.add_result(TaskResponseStatus.FUNCTION_CALL_RAISED, error)
                    continue
                else:
                    if store:
                        if isinstance(result, Collection):
                            result_ids = [id(res) for res in result]
                            for res in result:
                                self._storage[id(res)] = res
                        else:
                            result_ids = (id(result),)
                            self._storage[id(result)] = result

                        if return_result:
                            self.add_result(
                                TaskResponseStatus.OK,
                                (result_ids, result),
                            )
                        else:
                            self.add_result(TaskResponseStatus.OK, result_ids)
                    elif return_result:
                        self.add_result(TaskResponseStatus.OK, result)
                    else:
                        self.add_result(TaskResponseStatus.OK)
                        continue

            elif action == TaskAction.MEMOIZE:
                memo = data
                self._storage[memo] = previous_result
                self.add_result(TaskResponseStatus.OK)

            elif action == TaskAction.INTERRUPT:
                self.add_result(TaskResponseStatus.OK)
                break

    def task_done(self):
        """Marks a task as done."""
        self._comms.task_done()

    def add_result(self, status, data: Any = None):
        """Pushes a new result to the parent process."""
        self._comms.add_result((status, data))

    def run_task(self, task: Task):
        """Enqueues a task to be handled."""
        task.uid = self._task_counter
        self._task_counter += 1
        task.increment_execution_counter()

        self._comms.add_task(task)
        task.state = TaskState.QUEUED
        self._received_tasks.put(task)
        return task

    def _dequeue_task_result(self):
        """Dequeue latest result and update task."""
        result = self._comms.get_result()
        task = self._received_tasks.get_nowait()
        task.state = TaskState.FINISHED
        task.result = result
        return task

    def dequeue_results(self):
        """Dequeue available results and update tasks."""
        while not self._comms._results_queue.empty():
            self._dequeue_task_result()

    def wait_for_results(self, task_uid: int):
        """Dequeue available results and update tasks up to a given task ID."""
        while True:
            task = self._dequeue_task_result()
            if task is not None and task.uid == task_uid:
                break

    @property
    def uid(self) -> int:
        return self._id

    @property
    def task_count(self):
        """Gets the task count."""
        return self._task_counter


if is_fork_available():

    class ForkWorker(BaseWorker, mp.context.ForkProcess): ...

    class ForkServerWorker(BaseWorker, mp.context.ForkServerProcess): ...

else:
    class InvalidProcess:
        def __init__(self, *args, **kwargs):
            raise TypeError(f"Fork start method is not available on this platform")

    ForkWorker = InvalidProcess
    ForkServerWorker = InvalidProcess


class SpawnWorker(BaseWorker, mp.context.SpawnProcess): ...
