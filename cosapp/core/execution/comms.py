import ctypes
import multiprocessing as mp
import platform
import queue
from typing import Any, Optional

RUNNING_WINDOWS = platform.system() == "Windows"
RUNNING_MACOS = platform.system() == "Darwin"

from inspect import Traceback
from signal import SIG_IGN, SIGINT, Signals, getsignal
from signal import signal as signal_
from threading import current_thread, main_thread
from types import FrameType
from typing import Type

from .task import Task


class DelayedKeyboardInterrupt:

    def __init__(self) -> None:
        self.signal_received = None

    def __enter__(self) -> None:
        # When we're in a thread we can't use signal handling
        if current_thread() == main_thread():
            self.signal_received = False
            self.old_handler = signal_(SIGINT, self.handler)

    def handler(self, sig: Signals, frame: FrameType) -> None:
        self.signal_received = (sig, frame)

    def __exit__(self, exc_type: Type, exc_val: Exception, exc_tb: Traceback) -> None:
        if current_thread() == main_thread():
            signal_(SIGINT, self.old_handler)
            if self.signal_received:
                self.old_handler(*self.signal_received)


class DisableKeyboardInterruptSignal:

    def __enter__(self) -> None:
        if current_thread() == main_thread():
            # Prevent signal from propagating to child process
            self._handler = getsignal(SIGINT)
            ignore_keyboard_interrupt()

    def __exit__(self, exc_type: Type, exc_val: Exception, exc_tb: Traceback) -> None:
        if current_thread() == main_thread():
            # Restore signal
            signal_(SIGINT, self._handler)


def ignore_keyboard_interrupt():
    signal_(SIGINT, SIG_IGN)


class PoolComms:

    def __init__(self, ctx: mp.context.BaseContext) -> None:
        self.ctx = ctx
        self.exception_lock = self.ctx.Lock()
        self._exception_thrown = self.ctx.Event()


class WorkerComms:

    def __init__(self, ctx: mp.context.BaseContext, pool_comms: PoolComms) -> None:
        """
        :param ctx: Multiprocessing context
        :param n_jobs: Number of workers
        :param order_tasks: Whether to provide tasks to the workers in order, such that worker 0 will get chunk 0,
            worker 1 will get chunk 1, etc.
        """
        self.ctx = ctx

        self._task_queue: mp.JoinableQueue = self.ctx.JoinableQueue()
        self._worker_is_busy: mp.sharedctypes.Value = self.ctx.Value(
            ctypes.c_bool, False, lock=self.ctx.RLock()
        )

        self._results_queue: mp.JoinableQueue = self.ctx.JoinableQueue()
        self._results_stored: mp.sharedctypes.Value = self.ctx.Value(
            ctypes.c_uint32, False, lock=self.ctx.RLock()
        )
        self._results_popped: mp.sharedctypes.Value = self.ctx.Value(
            ctypes.c_uint32, False, lock=self.ctx.RLock()
        )

        self._pool_comms = pool_comms

    def add_task(self, task: Task) -> None:
        """Adds a task to the queue so a worker can process it.

        Parameters
        ----------
        task : Task
            Task to enqueue
        """
        with DelayedKeyboardInterrupt():
            self._task_queue.put(task, block=True)

    def get_task(self) -> Any:
        """Dequeues a task to execute."""
        while not self.exception_thrown():
            try:
                return self._task_queue.get(block=True, timeout=0.01)
            except queue.Empty:
                pass
        return None

    def task_done(self) -> None:
        """Marks the latest dequeued task as done."""
        self._task_queue.task_done()

    def add_result(self, result: Any) -> None:
        """Adds a result to the results queue.

        Parameters
        ----------
        result : Any
            Result to enqueue
        """
        self._results_queue.put(result)

    def get_result(self, block: bool = True, timeout: Optional[float] = None) -> Any:
        """Gets the next result from the results queue.
        
        Parameters
        ----------
        block : bool
            Whether to block (wait for results)
        timeout : Optiona[float]
            How long to wait for results in case ``block==True``

        Returns
        -------
        Any
            The next result from the queue
        """
        with DelayedKeyboardInterrupt():
            results = self._results_queue.get(block=block, timeout=timeout)
            self._results_queue.task_done()

            return results

    def exception_thrown(self) -> bool:
        """Whether an exception was thrown."""
        return self._pool_comms._exception_thrown.is_set()

    def wait_for_exception_thrown(self, timeout: Optional[float]) -> bool:
        """Waits until the exception thrown event is set
    
        Parameters
        ----------
        timeout : Optional[float]
            How long to wait before giving up

        Returns
        -------
        bool
            True when exception was thrown, False if timeout was reached
        """
        return self._pool_comms._exception_thrown.wait(timeout=timeout)
