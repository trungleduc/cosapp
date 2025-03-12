import sys
import weakref
from numbers import Number
from typing import Any

import pytest

from cosapp.base import System
from cosapp.core.execution import (
    Batch,
    ExecutionType,
    FunctionCallBehavior,
    Job,
    Pool,
    Task,
    TaskState,
    TaskAction,
    TaskResponseStatus,
    TaskResultNotAvailableYet,
    WorkerStartMethod,
    ops,
)
from cosapp.utils.testing import are_same


def _get_start_methods():
    if sys.platform == "win32":
        return (WorkerStartMethod.SPAWN, )

    return (WorkerStartMethod.FORK, WorkerStartMethod.SPAWN)

class S(System):
    def setup(self):
        self.add_inward("x", 1.0)
        self.add_inward("y", 1.2)


@pytest.fixture
def system():
    return S("s")


@pytest.fixture(scope="module")
def pool1(start_method: WorkerStartMethod = WorkerStartMethod.AUTO):
    p = Pool(1, ExecutionType.MULTI_PROCESSING, start_method)
    p.start()
    yield p

    p.stop()


@pytest.fixture(scope="module")
def pool2(start_method: WorkerStartMethod = WorkerStartMethod.AUTO):
    p = Pool(2, ExecutionType.MULTI_PROCESSING, start_method)
    p.start()
    yield p

    p.stop()


def add(rhs: Number, lhs: Number) -> Number:
    return rhs + lhs


def f_use_a_member(arg: Any) -> Any:
    return arg.a


class TestProcessPool:

    @pytest.mark.parametrize("start_method", _get_start_methods())
    def test_start_pool(self, start_method):
        """Test process pool start."""

        pool = Pool(2, ExecutionType.MULTI_PROCESSING, start_method)
        assert all([worker is None for worker in pool._workers])
        pool.start()
        assert all([worker.is_alive() for worker in pool._workers])


    @pytest.mark.parametrize("start_method", _get_start_methods())
    def test_stop_pool(self, start_method):
        """Test process pool stop."""

        pool = Pool(2, ExecutionType.MULTI_PROCESSING, start_method)
        pool.start()
        pool.stop()
        assert all([worker is None for worker in pool._workers])

    @pytest.mark.parametrize("pool1", _get_start_methods(), indirect=True)
    def test_store_returned_object(self, system, pool1):
        """Test capability to persist objects in workers' storage."""

        task = pool1.run_task(
            Task(
                TaskAction.FUNC_CALL,
                FunctionCallBehavior.STORE_RETURNED_OBJECT,
                (ops.return_arg, (system,)),
            )
        )
        assert task.result == TaskResultNotAvailableYet
        status, (storage_id,) = task.wait_for_result()
        assert isinstance(storage_id, int)
        assert status == TaskResponseStatus.OK
        assert task.state == TaskState.FINISHED

    @pytest.mark.parametrize("pool1", _get_start_methods(), indirect=True)
    def test_reuse_stored_object(self, system, pool1):
        """Test reuse of previously stored objects."""

        store = pool1.run_task(
            Task(
                TaskAction.FUNC_CALL,
                FunctionCallBehavior.STORE_RETURNED_OBJECT,
                (ops.return_arg, (system,)),
            )
        )
        status, data = store.wait_for_result()
        assert status == TaskResponseStatus.OK

        reuse = pool1.run_task(
            Task(
                TaskAction.FUNC_CALL,
                FunctionCallBehavior.RETURN_OBJECT
                | FunctionCallBehavior.ARGS_IN_STORAGE,
                (ops.return_arg, (data,)),
            )
        )
        status, data = reuse.wait_for_result()
        assert status == TaskResponseStatus.OK
        assert are_same(system, data)

    @pytest.mark.parametrize("pool1", _get_start_methods(), indirect=True)
    def test_invalid_reuse_stored_object(self, pool1):
        """Test reuse of previously stored objects."""
        invalid_object_id = 0
        reuse = pool1.run_task(
            Task(
                TaskAction.FUNC_CALL,
                FunctionCallBehavior.RETURN_OBJECT
                | FunctionCallBehavior.ARGS_IN_STORAGE,
                (ops.return_arg, ((invalid_object_id,),)),
            )
        )
        status, data = reuse.wait_for_result()
        assert status == TaskResponseStatus.MISSING_STORED_OBJECT
        assert isinstance(data, KeyError)

    @pytest.mark.parametrize("pool1", _get_start_methods(), indirect=True)
    def test_function_call_invalid_signature(self, pool1):
        """Test function call with invalid signature."""
        reuse = pool1.run_task(
            Task(
                TaskAction.FUNC_CALL,
                FunctionCallBehavior.EXECUTE,
                (ops.return_arg, ()),
            )
        )
        status, data = reuse.wait_for_result()
        assert status == TaskResponseStatus.FUNCTION_CALL_RAISED
        assert isinstance(data, TypeError)

    @pytest.mark.parametrize("pool1", _get_start_methods(), indirect=True)
    def test_function_call_bad_arguments(self, pool1):
        """Test function call with invalid signature."""
        reuse = pool1.run_task(
            Task(
                TaskAction.FUNC_CALL,
                FunctionCallBehavior.EXECUTE,
                (f_use_a_member, (None,)),
            )
        )
        status, data = reuse.wait_for_result()
        assert status == TaskResponseStatus.FUNCTION_CALL_RAISED
        assert isinstance(data, AttributeError)

    @pytest.mark.parametrize("pool1", _get_start_methods(), indirect=True)
    def test_function_store_and_return(self, pool1):
        """Test function call with stored and returned policy."""
        ref_obj = 12.1
        reuse = pool1.run_task(
            Task(
                TaskAction.FUNC_CALL,
                FunctionCallBehavior.STORE_RETURNED_OBJECT
                | FunctionCallBehavior.RETURN_OBJECT,
                (ops.return_arg, (ref_obj,)),
            )
        )
        status, ((storage_id,), returned_obj) = reuse.wait_for_result()
        assert status == TaskResponseStatus.OK
        assert isinstance(storage_id, int)
        assert returned_obj == ref_obj

    @pytest.mark.parametrize("pool2", _get_start_methods(), indirect=True)
    def test_worker_affinity(self, system, pool2):
        """Test worker affinity on a task ran multiple times."""
        task = Task(
            TaskAction.FUNC_CALL,
            FunctionCallBehavior.EXECUTE,
            (ops.return_arg, (system,)),
        )
        worker = pool2.workers[1]
        worker_initial_task_count = worker.task_count

        pool2.run_task(task, worker_id=1)
        assert task._uid == worker_initial_task_count
        assert worker.task_count == worker_initial_task_count + 1

        pool2.run_task(task)
        assert task._uid == worker_initial_task_count + 1
        assert worker.task_count == worker_initial_task_count + 2

    @pytest.mark.parametrize("pool1", _get_start_methods(), indirect=True)
    def test_chained_jobs(self, pool1):
        """Test dispatch of a `Job` with chained tasks."""
        ref_obj = 12.1
        task1, task2 = pool1.run_job(
            Job(
                [
                    Task(
                        TaskAction.FUNC_CALL,
                        FunctionCallBehavior.EXECUTE,
                        (ops.return_arg, (ref_obj,)),
                    ),
                    Task(
                        TaskAction.FUNC_CALL,
                        FunctionCallBehavior.RETURN_OBJECT
                        | FunctionCallBehavior.CHAINED,
                        (ops.return_arg, ()),
                    ),
                ]
            )
        )
        status, data = task1.wait_for_result()
        assert status == TaskResponseStatus.OK
        assert data is None

        status, data = task2.wait_for_result()
        assert status == TaskResponseStatus.OK
        assert isinstance(data, float)
        assert data == ref_obj

    @pytest.mark.parametrize("pool1", _get_start_methods(), indirect=True)
    def test_chained_jobs_with_partial_args(self, pool1):
        """Test dispatch of a `Job` with chained tasks and partial args."""
        ref_obj = 12.1
        task1, task2 = pool1.run_job(
            Job(
                [
                    Task(
                        TaskAction.FUNC_CALL,
                        FunctionCallBehavior.EXECUTE,
                        (ops.return_arg, (ref_obj,)),
                    ),
                    Task(
                        TaskAction.FUNC_CALL,
                        FunctionCallBehavior.RETURN_OBJECT
                        | FunctionCallBehavior.CHAINED,
                        (add, (10.0,)),
                    ),
                ]
            )
        )
        status, data = task1.wait_for_result()
        assert status == TaskResponseStatus.OK
        assert data is None

        status, data = task2.wait_for_result()
        assert status == TaskResponseStatus.OK
        assert isinstance(data, float)
        assert data == ref_obj + 10.0

    @pytest.mark.parametrize("pool2", _get_start_methods(), indirect=True)
    def test_run_batch(self, pool2):
        """Test dispatch of a 2 `Job`s as a batch."""
        ref_obj1 = [12.1]
        ref_obj2 = [13.1]

        (task1,), (task2,) = batch = pool2.run_batch(
            [
                Job(
                    Task(
                        TaskAction.FUNC_CALL,
                        FunctionCallBehavior.RETURN_OBJECT,
                        (ops.return_arg, (ref_obj1,)),
                    )
                ),
                Job(
                    Task(
                        TaskAction.FUNC_CALL,
                        FunctionCallBehavior.RETURN_OBJECT,
                        (ops.return_arg, (ref_obj2,)),
                    )
                ),
            ]
        )

        assert task1.execution_count == 1
        assert task2.execution_count == 1

        assert task1.worker.uid == 0
        status, data = task1.wait_for_result()
        assert status == TaskResponseStatus.OK
        assert data == ref_obj1

        assert task2.worker.uid == 1
        status, data = task2.wait_for_result()
        assert status == TaskResponseStatus.OK
        assert data == ref_obj2

        ref_obj1[0] = 15.1
        ref_obj2[0] = 16.1

        pool2.run_batch(batch)

        assert task1.worker.uid == 0
        status, data = task1.wait_for_result()
        assert status == TaskResponseStatus.OK
        assert data == ref_obj1
        assert task1.execution_count == 2

        assert task2.worker.uid == 1
        status, data = task2.wait_for_result()
        assert status == TaskResponseStatus.OK
        assert data == ref_obj2
        assert task2.execution_count == 2

    @pytest.mark.parametrize("pool2", _get_start_methods(), indirect=True)
    def test_run_batch_min_block_size(self, pool2):
        """Test dispatch of a 2 `Job`s as a batch on a pool of 2 workers, with
        minimum block size."""
        ref_obj1 = 12.1
        ref_obj2 = 13.1

        (task1,), (task2,) = pool2.run_batch(
            [
                Job(
                    Task(
                        TaskAction.FUNC_CALL,
                        FunctionCallBehavior.RETURN_OBJECT,
                        (ops.return_arg, (ref_obj1,)),
                    )
                ),
                Job(
                    Task(
                        TaskAction.FUNC_CALL,
                        FunctionCallBehavior.RETURN_OBJECT,
                        (ops.return_arg, (ref_obj2,)),
                    )
                ),
            ],
            min_block_size=2,
        )

        assert task1.worker.uid == 0
        status, data = task1.wait_for_result()
        assert status == TaskResponseStatus.OK
        assert data == ref_obj1

        assert task2.worker.uid == 0
        status, data = task2.wait_for_result()
        assert status == TaskResponseStatus.OK
        assert isinstance(data, float)
        assert data == ref_obj2

    @pytest.mark.parametrize("pool1", _get_start_methods(), indirect=True)
    def test_run_batch_small_pool(self, pool1):
        """Test dispatch of a 2 `Job`s as a batch on a pool of 1 worker."""
        ref_obj1 = 12.1
        ref_obj2 = 13.1

        (task1,), (task2,) = pool1.run_batch(
            [
                Job(
                    Task(
                        TaskAction.FUNC_CALL,
                        FunctionCallBehavior.RETURN_OBJECT,
                        (ops.return_arg, (ref_obj1,)),
                    )
                ),
                Job(
                    Task(
                        TaskAction.FUNC_CALL,
                        FunctionCallBehavior.RETURN_OBJECT,
                        (ops.return_arg, (ref_obj2,)),
                    )
                ),
            ]
        )

        assert task1.worker.uid == 0
        status, data = task1.wait_for_result()
        assert status == TaskResponseStatus.OK
        assert data == ref_obj1

        assert task2.worker.uid == 0
        status, data = task2.wait_for_result()
        assert status == TaskResponseStatus.OK
        assert isinstance(data, float)
        assert data == ref_obj2

    @pytest.mark.parametrize("pool1", _get_start_methods(), indirect=True)
    def test_run_batch_invalid_state(self, pool1):
        """Test dispatch of a batch with inconsistent states."""
        t1 = Task(
            TaskAction.FUNC_CALL,
            FunctionCallBehavior.RETURN_OBJECT,
            (ops.noop, ()),
        )
        job1 = Job(t1)
        job1._worker = weakref.ref(pool1.workers[0])

        with pytest.raises(RuntimeError):
            pool1.run_batch(
                [
                    job1,
                    Job(
                        Task(
                            TaskAction.FUNC_CALL,
                            FunctionCallBehavior.RETURN_OBJECT,
                            (ops.noop, ()),
                        )
                    ),
                ]
            )


def test_job_batch():
    jobs = [Job(Task(TaskAction.FUNC_CALL, ops.noop))]
    b = Batch(jobs * 10)
    assert [block for block in b.get_blocks(16)] == [range(i, i + 1) for i in range(10)]

    b = Batch(jobs * 10, block_min_size=2)
    assert [block for block in b.get_blocks(16)] == [
        range(i, i + 2) for i in range(0, 10, 2)
    ]

    b = Batch(jobs * 10, block_min_size=3)
    assert [block for block in b.get_blocks(16)] == [
        range(0, 4),
        range(4, 7),
        range(7, 10),
    ]

    b = Batch(jobs * 11, block_min_size=3)
    assert [block for block in b.get_blocks(16)] == [
        range(0, 4),
        range(4, 8),
        range(8, 11),
    ]

    b = Batch(jobs * 5, block_min_size=1)
    assert [block for block in b.get_blocks(3)] == [
        range(0, 2),
        range(2, 4),
        range(4, 5),
    ]

    b = Batch(jobs * 5, block_min_size=2)
    assert [block for block in b.get_blocks(3)] == [range(0, 3), range(3, 5)]
