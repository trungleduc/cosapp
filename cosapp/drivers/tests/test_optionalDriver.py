from unittest.mock import MagicMock

import pytest

from cosapp.systems import System
from cosapp.drivers.optionaldriver import OptionalDriver


def test_OptionalDriver__initialize():
    d = OptionalDriver("test")
    assert d._active == None

    d = OptionalDriver("test", force=True)
    assert d._active == True

    d = OptionalDriver("test", force=False)
    assert d._active == False

def test_OptionalDriver_set_inhibited():
    assert OptionalDriver._OptionalDriver__inhibited == False
    OptionalDriver.set_inhibited(True)
    assert OptionalDriver._OptionalDriver__inhibited == True
    OptionalDriver.set_inhibited(False)
    assert OptionalDriver._OptionalDriver__inhibited == False

def test_OptionalDriver_is_active():
    d = OptionalDriver("test")
    assert d.is_active() == True
    OptionalDriver.set_inhibited(True)
    assert d.is_active() == False
    OptionalDriver.set_inhibited(False)
    assert d.is_active() == True

    d = OptionalDriver("test", force=True)
    assert d.is_active() == True
    OptionalDriver.set_inhibited(True)
    assert d.is_active() == True
    OptionalDriver.set_inhibited(False)
    assert d.is_active() == True

    d = OptionalDriver("test", force=False)
    assert d.is_active() == True
    OptionalDriver.set_inhibited(True)
    assert d.is_active() == False
    OptionalDriver.set_inhibited(False)
    assert d.is_active() == True

def test_OptionalDriver_compute():
    s = System("dummy")
    d = s.add_driver(OptionalDriver("test"))
    d.compute = MagicMock()
    d.run_once()
    d.compute.assert_called_once()
    d.compute = MagicMock()
    OptionalDriver.set_inhibited(True)
    d.run_once()
    d.compute.assert_not_called()
    d.compute = MagicMock()
    OptionalDriver.set_inhibited(False)
    d.run_once()
    d.compute.assert_called_once()

    s = System("dummy")
    d = s.add_driver(OptionalDriver("test", force=True))
    d.compute = MagicMock()
    d.run_once()
    d.compute.assert_called_once()
    d.compute = MagicMock()
    OptionalDriver.set_inhibited(True)
    d.run_once()
    d.compute.assert_called_once()
    d.compute = MagicMock()
    OptionalDriver.set_inhibited(False)
    d.run_once()
    d.compute.assert_called_once()

    s = System("dummy")
    d = s.add_driver(OptionalDriver("test", force=False))
    d.compute = MagicMock()
    d.run_once()
    d.compute.assert_called_once()
    d.compute = MagicMock()
    OptionalDriver.set_inhibited(True)
    d.run_once()
    d.compute.assert_not_called()
    d.compute = MagicMock()
    OptionalDriver.set_inhibited(False)
    d.run_once()
    d.compute.assert_called_once()
