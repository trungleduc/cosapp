import tempfile
import time
import os
from os import path, rename, remove
from unittest import mock
import pytest
from cosapp.tests.library.systems import IterativeNonLinear

watchdog = pytest.importorskip("watchdog")

from cosapp.tools.trigger import FileCreationHandler, FileModificationHandler

TIMEOUT = 0.05


def test_FileCreationHandler(tmp_path):
    s = IterativeNonLinear("mySystem")
    s.run_drivers = mock.MagicMock(name="run_drivers")

    t = FileCreationHandler(
        s, folder=str(tmp_path), patterns=["*.txt"], timeout=TIMEOUT
    )
    t.start()

    n_calls = 4
    for i in range(n_calls):
        f_name = tmp_path / f"test{i}.txt"
        with open(f_name, "ab"):
            os.utime(f_name)

    now = time.time()
    dt = 0.05 * TIMEOUT
    while s.run_drivers.call_count < n_calls and time.time() < (now + TIMEOUT):
        time.sleep(dt)
    t.stop()
    assert s.run_drivers.call_count == n_calls


def test_FileModificationHandler_rename(tmp_path):
    s = IterativeNonLinear("mySystem")
    s.run_drivers = mock.MagicMock(name="run_drivers")

    n_calls = 3
    t = FileModificationHandler(
        s, folder=str(tmp_path), patterns=["*.txt"], timeout=TIMEOUT
    )
    t.start()

    for i in range(n_calls):
        f_name = tmp_path / f"test{i}.txt"
        with open(f_name, "ab"):
            os.utime(f_name)
        f_name.rename(f_name.parent / f_name.name.replace("test", "tests"))

    now = time.time()
    dt = 0.05 * TIMEOUT
    while s.run_drivers.call_count < n_calls and time.time() < (now + TIMEOUT):
        time.sleep(dt)
    t.stop()
    assert s.run_drivers.call_count == n_calls


def test_FileModificationHandler_remove(tmp_path):
    s = IterativeNonLinear("mySystem")
    s.run_drivers = mock.MagicMock(name="run_drivers")

    n_calls = 5
    t = FileModificationHandler(
        s, folder=str(tmp_path), patterns=["*.txt"], timeout=TIMEOUT
    )
    t.start()

    for i in range(n_calls):
        f_name = tmp_path / f"test{i}.txt"
        with open(f_name, "ab"):
            os.utime(f_name)
        new_name = f_name.parent / f_name.name.replace("test", "tests")
        f_name.rename(new_name)
        new_name.unlink()

    now = time.time()
    dt = 0.05 * TIMEOUT
    while s.run_drivers.call_count < n_calls and time.time() < (now + TIMEOUT):
        time.sleep(dt)
    t.stop()
    assert s.run_drivers.call_count == n_calls


@pytest.mark.parametrize(
    "patterns,extensions,call_count",
    [
        (["*.txt", "*.tx"], ("txt", "tx", "txtx"), 2),
        (["*.*"], ("txt", "tx", "txtx"), 3),
        (["*.*"], ("txt", "txtx"), 2),
    ],
)
def test_FileCreationHandler_patterns(tmp_path, patterns, extensions, call_count):
    s = IterativeNonLinear("mySystem")
    s.run_drivers = mock.MagicMock(name="run_drivers")

    t = FileCreationHandler(s, folder=str(tmp_path), patterns=patterns, timeout=TIMEOUT)
    t.start()

    n_calls = len(extensions)
    for ext in extensions:
        f_name = tmp_path / f"test.{ext}"
        with open(f_name, "ab"):
            os.utime(f_name)

    now = time.time()
    dt = 0.05 * TIMEOUT
    while s.run_drivers.call_count < n_calls and time.time() < (now + TIMEOUT):
        time.sleep(dt)
    t.stop()
    assert s.run_drivers.call_count == call_count


def test_FileCreationHandler_compute(tmp_path):
    s = IterativeNonLinear("mySystem")
    s.run_drivers = mock.MagicMock(name="run_drivers")

    n_calls = 2
    
    t = FileCreationHandler(
        s, folder=str(tmp_path), patterns=["*.txt"], timeout=TIMEOUT
    )
    t.compute = mock.MagicMock(name="compute")
    t.start()

    for i in range(n_calls):
        f_name = tmp_path / f"test{i}.txt"
        with open(f_name, "ab"):
            os.utime(f_name)

    now = time.time()
    dt = 0.05 * TIMEOUT
    while s.run_drivers.call_count < n_calls and time.time() < (now + TIMEOUT):
        time.sleep(dt)
    t.stop()
    assert t.compute.call_count == n_calls
    s.run_drivers.assert_not_called()


def test_FileCreationHandler_reset(tmp_path):
    s = IterativeNonLinear("mySystem")
    s.run_drivers = mock.MagicMock(name="run_drivers")

    t = FileCreationHandler(
        s, folder=str(tmp_path), patterns=["*.txt"], timeout=TIMEOUT
    )
    t.reset = mock.MagicMock(name="reset")
    t.start()

    n_calls = 2
    for i in range(n_calls):
        f_name = tmp_path / f"test{i}.txt"
        time.sleep(TIMEOUT / 20.0)
        with open(f_name, "ab"):
            os.utime(f_name)

    now = time.time()
    dt = 0.05 * TIMEOUT
    while s.run_drivers.call_count < n_calls and time.time() < (now + TIMEOUT):
        time.sleep(dt)
    t.stop()
    assert t.reset.call_count == n_calls
    assert s.run_drivers.call_count == n_calls
