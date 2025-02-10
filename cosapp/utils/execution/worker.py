import multiprocessing as mp
import signal
import threading as mt
from queue import Queue
from typing import Any, Collection

from cosapp.utils.state_io import object__getstate__

from .comms import WorkerComms
from .context import is_fork_available, is_running_windows
from .task import (FunctionCallBehavior, Task, TaskActionType,
                   TaskResponseStatusCode, TaskState)


class StopWorker(Exception):
    """Exception used to kill a worker"""

    pass


class InterruptWorker(Exception):
    """Exception used to interrupt a worker"""

    pass


class AbstractWorker:
    """Interface class for all workers."""

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
        raise StopWorker

    def _on_exception_exit_gracefully_windows(self) -> None:
        # TODO: implement this method
        raise StopWorker

    def run(self) -> None:
        """Runs tasks received from parent process."""
        self._set_signal_handlers()

        loop: bool = True
        previous_result: Any = None  # allow tasks chaining
        while loop:
            action, opts, data = self._comms.get_task()
            self.task_done()

            apply = action == TaskActionType.FUNC_CALL
            memoize = action == TaskActionType.MEMOIZE
            stop = action == TaskActionType.INTERRUPTION

            if apply:
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
                        storage_args = [
                            previous_result,
                        ]

                if args_in_storage:
                    largs = list(args)
                    try:
                        storage_ids = largs.pop(0)
                        storage_args.extend(self._storage[obj_id] for obj_id in storage_ids)
                    except KeyError as e:
                        self.add_result(TaskResponseStatusCode.MISSING_STORED_OBJECT, e)
                        continue
                    except Exception as e:
                        self.add_result(TaskResponseStatusCode.INVALID_STORAGE_ARGS, e)
                        continue
                else:
                    largs = args

                try:
                    result = previous_result = func(*storage_args, *largs)
                except Exception as e:
                    self.add_result(TaskResponseStatusCode.FUNCTION_CALL_RAISED, e)
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
                                TaskResponseStatusCode.OK,
                                (result_ids, result),
                            )
                        else:
                            self.add_result(TaskResponseStatusCode.OK, result_ids)
                    elif return_result:
                        self.add_result(TaskResponseStatusCode.OK, result)
                    else:
                        self.add_result(TaskResponseStatusCode.OK)
                        continue

            if memoize:
                memo = data
                self._storage[memo] = previous_result
                self.add_result(TaskResponseStatusCode.OK)

            if stop:
                self.add_result(TaskResponseStatusCode.OK)
                loop = False

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
        r = self._comms.get_result()
        task = self._received_tasks.get_nowait()
        task.state = TaskState.FINISHED
        task.result = r

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


class InvalidProcess:
    def __init__(self, *args, **kwargs):
        raise TypeError(f"Fork start method is not available on this platform")


if is_fork_available():

    class ForkWorker(AbstractWorker, mp.context.ForkProcess): ...

    class ForkServerWorker(AbstractWorker, mp.context.ForkServerProcess): ...

else:
    ForkWorker = InvalidProcess
    ForkServerWorker = InvalidProcess


class SpawnWorker(AbstractWorker, mp.context.SpawnProcess): ...
