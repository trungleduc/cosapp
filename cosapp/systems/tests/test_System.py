import pytest
from unittest import mock

import os
import sys
import logging
import re
from io import StringIO
from pathlib import Path
from collections import OrderedDict

import numpy as np

from cosapp.utils.testing import assert_keys, get_args
from cosapp.utils.logging import LogFormat, LogLevel
from cosapp.core.signal import Slot
from cosapp.core.connectors import Connector, ConnectorError
from cosapp.core.numerics.basics import MathematicalProblem
from cosapp.core.numerics.boundary import Unknown
from cosapp.core.numerics.residues import Residue
from cosapp.ports.port import ExtensiblePort, Port, PortType, Scope, Validity
from cosapp.ports.units import UnitError
from cosapp.drivers import Driver, RunOnce
from cosapp.systems import system as system_module
from cosapp.systems import System
from cosapp.systems.system import VariableReference, IterativeConnector

from cosapp.tests.library.systems import Multiply1
from cosapp.tests.library.systems.vectors import Strait1dLine
from cosapp.tests.library.ports import XPort


@pytest.fixture
def set_master_system():
    """Ensure the System class variable master is properly restored"""
    System._System__master_set = True
    try:
        yield
    finally:
        System._System__master_set = False


# Test partial connection
class DummyPort(Port):
    def setup(self):
        self.add_variable("a", 1)
        self.add_variable("b", 2)


class AnotherPort(Port):
    def setup(self):
        self.add_variable("aaaa", 1)


class System1(System):
    def setup(self):
        self.add_inward({"data1": 7.0, "a": 25.0, "b": 42.0, "aaaa": 1.0})
        self.add_outward({"local1": 11.0, "local2": 22.0, "local3": 33.0})


class System2(System):
    def setup(self):
        self.add_inward({"data1": 9.0, "data2": 11.0, "data3": 13.0})
        self.add_outward({"local1": 7.0, "a": 14.0, "b": 21.0})
        self.add_output(AnotherPort, "other")


class System3(System):
    def setup(self):
        self.add_input(DummyPort, "entry")
        self.add_output(DummyPort, "exit")


class VPort(Port):
    def setup(self):
        self.add_variable("v")


class PtWPort(Port):
    def setup(self):
        self.add_variable("Pt", 101325.0, unit="Pa", limits=(0.0, None))
        self.add_variable("W", 1.0, unit="kg/s", valid_range=(0.0, None))


class SubSystem(System):
    def setup(self):
        self.add_input(PtWPort, "in_")
        self.add_inward(
            "sloss",
            0.95,
            unit="m/s",
            dtype=float,
            valid_range=(0.8, 1.0),
            invalid_comment="not valid",
            limits=(0.0, 1.0),
            out_of_limits_comment="hasta la vista baby",
            desc="get down",
            scope=Scope.PROTECTED,
        )
        self.add_output(PtWPort, "out")
        self.add_outward(
            "tmp",
            unit="inch/lbm",
            dtype=(int, float, complex),
            valid_range=(1, 2),
            invalid_comment="not valid tmp",
            limits=(0, 3),
            out_of_limits_comment="I'll be back",
            desc="banana",
            scope=Scope.PROTECTED,
        )

        self.add_outward("dummy", 1.0)
        self.add_equation("dummy == 0")

    def compute(self):
        for name in self.out:
            self.out[name] = self.in_[name] * self.sloss
        self.dummy *= 1e-2


class TopSystem(System):
    def setup(self):
        self.add_inward("top_k")
        self.add_outward("top_tmp")
        self.add_property('const', 0.123)

        self.add_child(
            SubSystem("sub"), pulling={"in_": "in_", "out": "out"}
        )

@pytest.mark.parametrize("check_type", [True, False])
def test_System_set_master_for_master(caplog, check_type):
    m = Multiply1("m")

    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger=system_module.__name__):
        with System.set_master(repr(m), check_type) as is_master:
            assert is_master
            if check_type:
                with pytest.raises(TypeError):
                    m.p_in.x = "a"
            else:
                m.p_in.x = "a"
                assert m.p_in.x == "a"

    assert caplog.records[0].levelno == logging.DEBUG
    assert re.match(r"System <\w+ - [\w\.]+> is the execution master.", caplog.records[0].msg) is not None


@pytest.mark.parametrize("check_type", [True, False])
def test_System_set_master_for_non_master(caplog, set_master_system, check_type):
    m = Multiply1("m")

    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger=system_module.__name__):
        with System.set_master(repr(m), check_type) as is_master:
            assert not is_master
            with pytest.raises(TypeError):
                m.p_in.x = "a"

    assert len(caplog.records) == 0


def test_System_skip_type_checking():
    m = Multiply1("m")
    with pytest.raises(TypeError):
        m.p_in.x = "a"
    ExtensiblePort.set_type_checking(False)
    m.p_in.x = "a"
    m.K1 = 2
    ExtensiblePort.set_type_checking(True)
    pattern = r"Trying to set [\w\.]+ of type .*? with .*?"  # match in error message
    with pytest.raises(TypeError, match=pattern):
        m.run_once()

    m.add_driver(RunOnce("r"))
    # The following should not check for type
    m.run_drivers()
    assert isinstance(m.p_out.x, str)
    assert m.p_out.x == m.p_in.x * m.K1

    # Check that type checking kicks in again after run_driver
    with pytest.raises(TypeError):
        m.p_in.x = "a"
    m.K1 = 3  # Force a change to pass clean-dirty test
    with pytest.raises(TypeError, match=pattern):
        m.run_once()


def test_System_type_checking_sub_system(set_master_system):
    m = Multiply1("m")
    ExtensiblePort.set_type_checking(False)
    m.p_in.x = "a"  # Set bad value on purpose
    ExtensiblePort.set_type_checking(True)
    m.add_driver(RunOnce("r"))
    pattern = r"Trying to set [\w\.]+ of type .*? with .*?"  # match in error message

    m.K1 = 3  # Force a change to pass clean-dirty test
    # Calling a resolution on non-master sub systems does not change the type checking
    with pytest.raises(TypeError, match=pattern):
        m.run_drivers()



def test_System__init__():
    name = "test_system"
    s = System(name)
    assert s.name == name
    assert_keys(s.inputs, System.INWARDS)
    assert_keys(s.outputs, System.OUTWARDS)
    assert_keys(s.name2variable, System.INWARDS, System.OUTWARDS)
    assert len(s.children) == 0
    assert len(s.residues) == 0
    assert len(s.exec_order) == 0
    assert len(s.drivers) == 0
    assert s.parent is None


@pytest.mark.parametrize("name", ["Asystem", "ap1_ort_", "zsystem2"])
def test_System__init__ok(name):
    system = System(name)
    assert system.name == name


@pytest.mark.parametrize("name, error", [
    ("1system", ValueError),
    ("_system", ValueError),
    ("system-2", ValueError),
    ("system:2", ValueError),
    ("system.2", ValueError),
    (23, TypeError),
    (1.0, TypeError),
    (dict(a=True), TypeError),
    (list(), TypeError)
])
def test_System__init__bad_name(name, error):
    with pytest.raises(error):
        System(name)


def test_System___getattr__():
    s = TopSystem("test")
    assert s.sub is s.children["sub"]
    assert s.out is s.outputs["out"]
    assert s.in_ is s.inputs["in_"]
    assert s.in_.Pt is s.inputs["in_"].Pt
    assert s.sub.in_ is s.children["sub"].inputs["in_"]
    assert s.sub.out.W is s.children["sub"].outputs["out"].W
    assert s.sub.sloss is s.children["sub"].inputs["inwards"].sloss

    with pytest.raises(AttributeError):
        _ = s.sub.sloss1


def test_System___setattr__():
    s = TopSystem("test")
    s.out.Pt = 123456.0
    s.sub.inwards.sloss = 0.9
    assert s.out.Pt == 123456.0
    assert s.sub.inwards.sloss == 0.9

    s.sub.sloss = 0.95
    assert s.sub.sloss == 0.95

    # Forbid creating new attributes
    with pytest.raises(AttributeError):
        s.sub.sloss1 = 1.0


def test_System___contains__():
    s = TopSystem("test")
    assert "sub" in s
    assert "in_" in s
    assert "out" in s
    assert "in_.Pt" in s
    assert "const" in s

    assert "parent" not in s
    assert "inputs" not in s
    assert "outputs" not in s
    assert "children" not in s
    assert "name2variable" not in s


def test_System___getitem__():
    s = TopSystem("test")
    assert s["parent"] is s.parent
    assert s["sub"] is s.children["sub"]
    assert s["sub"] is s.sub
    assert s["out"] is s.outputs["out"]
    assert s["in_"] is s.inputs["in_"]
    assert s["in_.Pt"] == s.inputs["in_"].Pt
    assert s["sub.in_"] is s.children["sub"].inputs["in_"]
    assert s["sub.out.W"] == s.children["sub"].outputs["out"].W
    assert s["const"] == s.const
    assert s["const"] == 0.123
    assert s["name"] == s.name
    assert s["top_k"] == s.top_k


def test_System__setitem__():
    s = TopSystem("test")
    s["out.Pt"] = 123456.0
    s["sub.inwards.sloss"] = 0.9
    assert s.out.Pt == 123456.0
    assert s.sub.inwards.sloss == 0.9


def test_System__repr__():
    top = TopSystem("banana")

    assert repr(top) == "banana - TopSystem"
    assert repr(top.sub) == "sub - SubSystem"


def test_System_load_group():
    group = System("hat")
    group.add_child(System1("s1"))
    group.add_child(System2("s2"))
    group.add_child(System3("s3"))
    group.add_child(System3("s4"))

    group.connect(group.s1.inwards, group.s3.exit)
    group.connect(group.s2.outwards, group.s4.entry)
    group.connect(group.s1.inwards, group.s2.other, "aaaa")
    group.connect(group.s2.inwards, group.s1.outwards,
        {"data1": "local2", "data2": "local1"})

    src = group.to_json()
    config = StringIO(src)
    grp = System.load(config)
    assert grp.to_json() == src

    with pytest.raises(TypeError):
        System.load(1.0)

@mock.patch.object(System, 'setup_run')
@mock.patch.object(RunOnce, 'setup_run')
def test_System_call_setup_run(mock_RunOnce_setup_run, mock_System_setup_run):
    s = System('s')
    s.add_driver(RunOnce('run'))

    s.run_drivers()

    mock_System_setup_run.assert_called_once()
    mock_RunOnce_setup_run.assert_called_once()

    mock_System_setup_run.reset_mock()
    mock_RunOnce_setup_run.reset_mock()
    s.run_once()
    mock_System_setup_run.assert_called_once()
    mock_RunOnce_setup_run.assert_not_called()

@mock.patch.object(System, 'clean_run')
@mock.patch.object(RunOnce, 'clean_run')
def test_System_call_clean_run(mock_RunOnce_clean_run, mock_System_clean_run):
    s = System('s')
    s.add_driver(RunOnce('run'))

    s.run_drivers()

    mock_System_clean_run.assert_called_once()
    mock_RunOnce_clean_run.assert_called_once()

    mock_System_clean_run.reset_mock()
    mock_RunOnce_clean_run.reset_mock()
    s.run_once()
    mock_System_clean_run.assert_called_once()
    mock_RunOnce_clean_run.assert_not_called()


def test_System_convert_to():
    s = System("empty")

    with pytest.raises(TypeError, match="system is not part of a family"):
        s.convert_to("foobar")


def test_System_add_child():
    s = System("test")
    s2 = SubSystem("sub")
    res = s.add_child(s2)

    assert res is s2
    assert_keys(s.children, "sub")
    assert s.children['sub'] is s2
    assert s2.parent is s
    assert s.exec_order.last == s2.name

    to_check = {
        "sub": s2,
        "sub.in_": s2.inputs["in_"],
        "sub.out": s2.outputs["out"],
    }

    for key, obj in to_check.items():
        assert key in s.name2variable
        reference = s.name2variable[key]
        context = "key = {}".format(key)
        assert reference.mapping[reference.key] is obj, context

    s3 = SubSystem("sub2")
    assert s.add_child(s3, execution_index=0) is s3
    assert_keys(s.children, "sub", "sub2")
    assert s.exec_order.first == s3.name
    assert s.children['sub2'] is s3
    assert s3.parent is s

    to_check.update({
        "sub2": s3,
        "sub2.in_": s3.inputs["in_"],
        "sub2.out": s3.outputs["out"],
    })

    for key, obj in to_check.items():
        assert key in s.name2variable
        reference = s.name2variable[key]
        context = "key = {}".format(key)
        assert reference.mapping[reference.key] is obj, context


def test_System_add_child_pulling(caplog):
    caplog.set_level(logging.DEBUG)

    s = System("test")
    s2 = s.add_child(SubSystem("sub"), pulling={"in_": "entry", "out": "out"})

    records = list(filter(lambda record: record.levelno == logging.DEBUG, caplog.records))
    assert len(records) == 2
    pattern = r"Port \w+\.\w+ will be duplicated from \w+\.\w+ - including validation range and scope."
    for record in records:
        assert re.match(pattern, record.message)

    assert_keys(s.inputs, 'inwards', 'entry')
    entry = s.inputs['entry']
    assert entry is s.inputs['entry']  # check that assignment did not create a copy!
    assert entry is not s2.inputs["in_"]
    assert isinstance(entry, type(s2.inputs["in_"]))
    assert entry.direction is s2.inputs["in_"].direction
    assert entry is s.connectors["test_entry_to_sub_in_"].source

    assert_keys(s.outputs, 'outwards', 'out')
    s_out = s.outputs['out']
    assert s_out is s.outputs['out']  # check that assignment did not create a copy!
    assert s_out is not s2.outputs['out']
    assert isinstance(s_out, type(s2.outputs['out']))
    assert s_out.direction is s2.outputs['out'].direction
    assert s2.outputs["out"] is s.connectors["sub_out_to_test_out"].source

    # Use only str
    s = System("test")
    s2 = s.add_child(SubSystem("sub"), pulling="in_")
    assert_keys(s.inputs, 'in_', 'inwards')
    s_in = s.inputs['in_']
    assert s_in is s.inputs['in_']
    assert s_in is not s2.inputs['in_']
    assert isinstance(s_in, type(s2.inputs['in_']))
    assert s_in.direction is s2.inputs['in_'].direction
    assert s_in is s.connectors["test_in__to_sub_in_"].source

    # Use list of str
    s = System("test")
    s2 = s.add_child(SubSystem("sub"), pulling=["in_", "out"])
    assert_keys(s.inputs, 'in_', 'inwards')
    # Check pulled symbol 'in_'
    s_in = s.inputs['in_']
    assert s_in is s.inputs['in_']
    assert s_in is not s2.inputs['in_']
    assert isinstance(s_in, type(s2.inputs['in_']))
    assert s_in.direction is s2.inputs['in_'].direction
    assert s_in is s.connectors["test_in__to_sub_in_"].source
    # Check pulled symbol 'out'
    assert_keys(s.outputs, 'outwards', 'out')
    s_out = s.outputs['out']
    assert s_out is s.outputs['out']  # check that assignment did not create a copy!
    assert s_out is not s2.outputs['out']
    assert isinstance(s_out, type(s2.outputs['out']))
    assert s_out.direction is s2.outputs['out'].direction
    assert s2.outputs["out"] is s.connectors["sub_out_to_test_out"].source

    # Pulling from 2 children IN to same IN
    s = System("test")
    s.add_child(SubSystem("sub_a"), pulling={"in_": "entry"})
    s.add_child(SubSystem("sub_b"), pulling={"in_": "entry"})

    with pytest.raises(KeyError):
        s.add_child(SubSystem("sub_c"), pulling=["here"])

    assert_keys(s.inputs, 'entry', 'inwards')
    s_in = s.inputs['entry']
    assert s_in is s.inputs['entry']
    assert s_in is not s2.inputs['in_']
    assert isinstance(s_in, type(s2.inputs['in_']))
    assert s_in.direction is s2.inputs['in_'].direction
    assert s_in is s.connectors["test_entry_to_sub_a_in_"].source
    assert s_in is s.connectors["test_entry_to_sub_b_in_"].source

    # Pulling from 2 children OUT to same OUT
    s = System("test")
    s.add_child(SubSystem("sub_a"), pulling={"out": "out"})
    with pytest.raises(ConnectorError):
        s.add_child(SubSystem("sub_b"), pulling={"out": "out"})
    assert "sub_b" not in s.children
    with pytest.raises(ConnectorError):
        s.add_child(SubSystem("sub_b"), pulling={"in_": "out"})
    assert "sub_b" not in s.children

    # Pulling from 1 child IN and 1 child OUT to same IN
    s = System("test")
    s.add_child(SubSystem("sub_a"), pulling={"in_": "entry"})
    with pytest.raises(ConnectorError):
        s.add_child(SubSystem("sub_b"), pulling={"out": "entry"})
    assert "sub_b" not in s.children

    # Pulling inwards
    caplog.clear()
    s = System("test")
    s2a = s.add_child(SubSystem("sub_a"), pulling="sloss")

    records = list(filter(lambda record: record.levelno == logging.DEBUG, caplog.records))
    assert len(records) == 1
    assert re.match(
        r"inwards \w+\.\w+ will be duplicated from \w+\.\w+",
        records[-1].message)

    assert s.inwards['sloss'] == s2a['sloss']
    source = s2a.inwards.get_details("sloss")
    pulled = s.inwards.get_details("sloss")
    for attr in ["unit", "dtype", "description", "scope"]:
        assert getattr(pulled, attr) == getattr(source, attr)
    assert pulled.valid_range == (-np.inf, np.inf)
    assert pulled.valid_range == (-np.inf, np.inf)
    assert pulled.invalid_comment == ""
    assert pulled.limits == (-np.inf, np.inf)
    assert pulled.out_of_limits_comment == ""

    s = System("test")
    s2a = s.add_child(SubSystem("sub_a"), pulling=["sloss", "tmp"])
    assert s.inwards['sloss'] == s2a['sloss']
    assert s.outwards['tmp'] == s2a['tmp']

    s = System("test")
    s2a = s.add_child(SubSystem("sub_a"), pulling={"sloss": "a_sloss"})
    assert s.inwards['a_sloss'] == s2a['sloss']

    # Pulling all inwards
    s = System("test")
    s2a = s.add_child(SubSystem("sub_a"), pulling="inwards")
    assert s.inwards['sloss'] == s2a['sloss']

    # Pulling outwards
    caplog.clear()
    s = System("test")
    s2a = s.add_child(SubSystem("sub_a"), pulling="tmp")

    records = list(filter(lambda record: record.levelno == logging.DEBUG, caplog.records))
    assert len(records) == 1
    assert re.match(
        r"outwards \w+\.\w+ will be duplicated from \w+\.\w+", 
        records[-1].message)

    assert s.outwards['tmp'] == s2a['tmp']
    source = s2a.outwards.get_details("tmp")
    pulled = s.outwards.get_details("tmp")
    for attr in ["unit", "dtype", "description", "scope"]:
        assert getattr(pulled, attr) == getattr(source, attr)
    assert pulled.valid_range == (-np.inf, np.inf)
    assert pulled.valid_range == (-np.inf, np.inf)
    assert pulled.invalid_comment == ""
    assert pulled.limits == (-np.inf, np.inf)
    assert pulled.out_of_limits_comment == ""

    s = System("test")
    s2a = s.add_child(SubSystem("sub_a"), pulling={"tmp": "a_tmp"})
    assert s.outwards["a_tmp"] == s2a["tmp"]

    # Pulling all outwards
    s = System("test")
    s2a = s.add_child(SubSystem("sub_a"), pulling="outwards")
    assert s.outwards["tmp"] == s2a["tmp"]

    # Adding a child component with an already existing name
    s = TopSystem("test")
    with pytest.raises(ValueError):
        s.add_child(SubSystem("top_tmp"))

    # Adding Driver
    s = TopSystem("test")
    with pytest.raises(TypeError):
        s.add_child(Driver("dummy"))


@pytest.mark.parametrize("args", [
    PtWPort("p", PortType.IN),
    (System("sub"), "first"),
])
def test_System_add_child_TypeError(args):
    s = System("test")
    with pytest.raises(TypeError):
        s.add_child(*args)


def test_System_pop_child():
    s = System("test")
    s2 = SubSystem("sub")
    s3 = SubSystem("sub2")
    s.add_child(s2, pulling=["in_", "sloss", "tmp"])
    s.add_child(s3)
    s.connect(s2.out, s3.in_)
    assert_keys(s.children, "sub", "sub2")

    assert_keys(s.connectors,
        "test_in__to_sub_in_",
        "test_inwards_to_sub_inwards",
        "sub_outwards_to_test_outwards",
        "sub_out_to_sub2_in_")

    s.pop_child("sub")
    assert_keys(s.children, "sub2")
    assert s2.parent is None
    assert s2.name not in s.exec_order
    assert list(s.exec_order) == [s3.name]
    assert len(s.connectors) == 0
    keys = [
        "sub",
        "sub.in_",
        "sub.in_.Pt",
        "sub.in_.W",
        "sub.out",
        "sub.out.Pt",
        "sub.out.W",
    ]
    for key in keys:
        assert key not in s.name2variable


def test_System_add_port():
    s = System("test")

    s._add_port(VPort("port1", PortType.IN, {"v": 1}))
    assert "port1" in s.inputs
    assert "port1" not in s.outputs
    for key in ["port1", "port1.v"]:
        assert key in s

    s._add_port(VPort("port2", PortType.OUT, {"v": 1}))
    assert "port2" not in s.inputs
    assert "port2" in s.outputs
    for key in ["port2", "port2.v"]:
        assert key in s

    with pytest.raises(ValueError):
        s._add_port(VPort("port2", PortType.OUT, {"v": 1}))

    with pytest.raises(ValueError):
        s._add_port(VPort("port1", PortType.IN, {"v": 1}))

    class CustomPort(Port):
        def setup(self):
            self.add_variable("x")

    s = System("test")
    with pytest.raises(TypeError):
        s._add_port(CustomPort("port1", "in"))

    p = CustomPort("port1", PortType.IN)
    p._direction = "in"
    with pytest.raises(ValueError):
        s._add_port(p)


def test_System_add_input():
    class T(System):
        def setup(self):
            port = self.add_input(VPort, "port1", {"v": 1})
            assert port is self.inputs["port1"]

    s = T("test")
    assert "port1" in s.inputs
    assert "port1" not in s.outputs
    for key in ["port1", "port1.v"]:
        assert key in s

    with pytest.raises(AttributeError):
        s.add_input(VPort, "port1", {"v": 1})


def test_System_add_output():
    class T(System):
        def setup(self):
            port = self.add_output(VPort, "port2", {"v": 1})
            assert port is self.outputs["port2"]

    s = T("test")
    assert "port2" not in s.inputs
    assert "port2" in s.outputs
    for key in ["port2", "port2.v"]:
        assert key in s

    with pytest.raises(AttributeError):
        s.add_output(VPort, "port2", {"v": 1})


def test_System_add_data(DummyFactory):
    # Add one inwards
    s = DummyFactory("test", inwards=get_args("K", 2.0))
    assert "K" in s
    assert ".".join((System.INWARDS, "K")) in s
    assert s.K == 2.0

    with pytest.raises(AttributeError):
        s.add_inward("K", 2.0)

    # Add multiple inwards
    s = DummyFactory("test", inwards=get_args(
        {
            "K": 2.0,
            "switch": True,
            "r": {"value": 1, "scope": Scope.PUBLIC},
            "q": {"a": 1, "b": 2},
        }
        ))

    for name in ["K", "switch", "r", "q"]:
        assert name in s
        assert ".".join((System.INWARDS, name)) in s
    assert s.K == 2.0
    assert s.switch == True
    assert s.r == 1
    assert s[System.INWARDS].get_details("r").scope == Scope.PUBLIC
    assert s.q == {"a": 1, "b": 2}

    # Test variables attributes
    s = DummyFactory("test", inwards=get_args(
        "K", 2.0,
        unit="m",
        dtype=float,
        valid_range=(0.0, 5.0),
        limits=(-5.0, 10.0),
        desc="my little description.",
        scope=Scope.PRIVATE,
        ))
    assert_keys(s.inwards.get_details(), "K")
    details = s.inwards.get_details("K")
    assert details.unit == "m"
    assert details.dtype == float
    assert details.valid_range == (0.0, 5.0)
    assert details.limits == (-5.0, 10.0)
    assert details.description == "my little description."
    assert details.scope == Scope.PRIVATE


# @pytest.mark.parametrize("inputs, inwards, error", [
#     (get_args(System, "s"), None, TypeError),
#     (get_args(VPort("p", PortType.IN), "s"), None, TypeError),
#     (get_args(VPort, None), None, TypeError),
#     (get_args(VPort, "p", variables=[24, "a"]), None, TypeError),
#     (get_args(VPort, "b"), get_args("b"), ValueError),
# ])
# def test_System_add_input_error(DummyFactory, inputs, inwards, error):
#     with pytest.raises(error):
#         DummyFactory("dummy", inputs=inputs, inwards=inwards)


# @pytest.mark.parametrize("outputs, inwards, error", [
#     (get_args(System, "s"), None, TypeError),
#     (get_args(VPort("p", PortType.IN), "s"), None, TypeError),
#     (get_args(VPort, None), None, TypeError),
#     (get_args(VPort, "p", variables=[24, "a"]), None, TypeError),
#     (get_args(VPort, "b"), get_args("b"), ValueError),
# ])
# def test_System_add_output_error(DummyFactory, outputs, inwards, error):
#     with pytest.raises(error):
#         DummyFactory("dummy", outputs=outputs, inwards=inwards)


@pytest.mark.parametrize("port_kind", ["inputs", "outputs"])
@pytest.mark.parametrize("case_data, error", [
    # in values below, "io_port" key will be substituted by `port_kind`
    (
        dict(io_port = get_args(System, "s")),
        TypeError
    ),
    (
        dict(io_port = get_args(VPort("p", PortType.IN), "s")),
        TypeError
    ),
    (
        dict(io_port = get_args(VPort, None)),
        TypeError
    ),
    (
        dict(io_port = get_args(VPort, "p", variables=[24, "a"])),
        TypeError
    ),
])
def test_System_input_output_error(DummyFactory, port_kind, case_data, error):
    """Test add_input & add_output errors"""
    # swap keys "io_port" and `port_kind` (either "inputs or "outputs")
    ctor_data = case_data.copy()
    ctor_data[port_kind] = ctor_data.pop("io_port", None)
    with pytest.raises(error):
        DummyFactory("dummy", **ctor_data)


@pytest.mark.parametrize("data_kind", ["inwards", "outwards"])
@pytest.mark.parametrize("port_kind", ["inputs", "outputs"])
@pytest.mark.parametrize("case_data, match", [
    # in values below, "io_port" key will be substituted by `port_kind`
    # in values below, "io_data" key will be substituted by `data_kind`
    (
        dict(
            io_port = get_args(VPort, "foo"),
            io_data = get_args("foo"),
        ),
        "dummy.foo already exists"
    ),
])
def test_System_existing_name(DummyFactory, port_kind, data_kind, case_data, match):
    # swap keys "io_port" and `port_kind` (either "inputs or "outputs")
    # swap keys "io_data" and `data_kind` (either "inwards or "outwards")
    ctor_data = case_data.copy()
    ctor_data[port_kind] = ctor_data.pop("io_port", None)
    ctor_data[data_kind] = ctor_data.pop("io_data", None)
    with pytest.raises(ValueError, match=match):
        DummyFactory("dummy", **ctor_data)


@pytest.mark.parametrize("kind1", ["inwards", "outwards"])
@pytest.mark.parametrize("kind2", ["inwards", "outwards"])
def test_System_existing_data(DummyFactory, kind1, kind2):
    # data1 and data2 serve as either 'inwards' or 'outwards'
    data1 = get_args("foo", -2.5)
    data2 = get_args("foo", 3.14, dtype=float)
    ctor_data = dict()
    if kind1 == kind2:  # both 'inwards' or 'outwards'
        ctor_data[kind1] = [data1, data2]
    else:
        ctor_data[kind1] = data1
        ctor_data[kind2] = data2
    with pytest.raises(ValueError, match="dummy.foo already exists"):
        DummyFactory("dummy", **ctor_data)


@pytest.mark.parametrize("data_kind", ["inwards", "outwards"])
@pytest.mark.parametrize("case_data, expected", [
    # in values below, "io_data" key will be substituted by `data_kind`
    (
        dict(io_data = get_args(1.0)),
        dict(error=TypeError, match="argument 'definition'")
    ),
    (
        dict(io_data = get_args("s1", 3.14, unit=float)),
        dict(error=TypeError, match="'unit' should be str")
    ),
    (
        dict(io_data = get_args("s1", "tag", dtype=float)),
        dict(error=TypeError, match=r"Cannot set .*\.s1 of type float with a str")
    ),
    (
        dict(io_data = [get_args("s1", 0.1), get_args("s1", 0.1)]),
        dict(error=ValueError, match=r".*\.s1 already exists")
    ),
])
def test_System_inward_outward_error(DummyFactory, data_kind, case_data, expected):
    """Test add_inward & add_outward errors"""
    # swap keys "io_data" and `data_kind` (either "inwards or "outwards")
    ctor_data = case_data.copy()
    ctor_data[data_kind] = ctor_data.pop("io_data")
    error = expected['error']
    pattern = expected.get('match', None)
    with pytest.raises(error, match=pattern):
        DummyFactory("dummy", **ctor_data)


def test_System_add_locals(DummyFactory):
    # Add unique
    s = DummyFactory("dummy", outwards=get_args("r", 42.0))
    assert "r" in s
    assert ".".join((System.OUTWARDS, "r")) in s
    assert s.r == 42

    # Add multiple outwards
    s = DummyFactory("dummy", outwards=get_args(
        {
            "r": 42.0,
            "q": 12,
            "s": {"value": 1, "scope": Scope.PUBLIC},
            "x": {"a": 1, "b": 2},
        }
        ))

    for name in ["r", "q", "s", "x"]:
        assert name in s
        assert ".".join((System.OUTWARDS, name)) in s
    assert s.r == 42.0
    assert s.q == 12
    assert s.s == 1
    assert s[System.OUTWARDS].get_details("s").scope == Scope.PUBLIC
    assert s.x == {"a": 1, "b": 2}

    # Add multiple outwards with attributes
    s = DummyFactory("dummy", outwards=get_args(
        {"r": {"value": 42.0, "desc": "my value"}, "q": 12})
    )
    assert "r" in s
    assert "q" in s
    assert s.r == 42.0
    assert s.q == 12
    assert s.outwards.get_details("q").description == ""
    assert s.outwards.get_details("r").description == "my value"
    with pytest.raises(AttributeError):
        s.add_outward("a", 10.0)

    # Test outward attributes
    s = DummyFactory("dummy", outwards=get_args(
        "K", 2.0,
        unit="m",
        dtype=(int, float),
        valid_range=(0.0, 5.0),
        limits=(-5.0, 10.0),
        desc="my little description.",
        scope=Scope.PROTECTED,
        ))
    assert_keys(s.outwards.get_details(), "K")
    details = s.outwards.get_details("K")
    assert details.unit == "m"
    assert details.dtype == (int, float)
    assert details.valid_range == (0.0, 5.0)
    assert details.limits == (-5.0, 10.0)
    assert details.description == "my little description."
    assert details.scope == Scope.PROTECTED


@pytest.mark.parametrize("drivers, expected", [
    (RunOnce("run"), dict()),
    (RunOnce("run", verbose=1), dict(verbose=[1])),
    (RunOnce("run_", verbose=1), dict(verbose=[1])),
    ([RunOnce("run1"), RunOnce("run2", verbose=1)], dict(verbose=[0, 1])),
    ("run", dict(error=TypeError)),
    (["run"], dict(error=TypeError)),
    (0.123, dict(error=TypeError)),
])
def test_System_add_driver(drivers, expected):
    s = System("test")
    error = expected.get("error", None)
    if error is None:
        if not isinstance(drivers, (tuple, list)):
            drivers = [drivers]
        verbose = expected.get("verbose", [0] * len(drivers))
        for i, driver in enumerate(drivers):
            s.add_driver(driver)
            s_driver = s.drivers[driver.name]
            assert s_driver is s.drivers[driver.name]  # just to be sure!
            assert s_driver is driver
            assert s_driver.options["verbose"] == verbose[i]
        assert_keys(s.drivers, *(driver.name for driver in drivers))
    else:
        with pytest.raises(error):
            s.add_driver(drivers)


def test_System_is_running():
    class TSystem(System):
        def compute_before(self):
            assert self.is_running()

        def compute(self):
            assert self.is_running()

    m = TSystem("m")
    assert not m.is_running()
    m.run_once()
    assert not m.is_running()

    assert not m.is_running()
    m.run_children_drivers()
    assert not m.is_running()


def test_System_compute():
    s = SubSystem("s")
    p = PtWPort("p", PortType.OUT)
    assert s.in_.Pt == p.Pt
    assert s.in_.W == p.W
    assert s.out.Pt == p.Pt
    assert s.out.W == p.W
    assert s.residues["dummy == 0"].value == 1.0

    s.compute()
    assert s.in_.Pt == p.Pt
    assert s.in_.W == p.W
    assert s.out.Pt == s.sloss * p.Pt
    assert s.out.W == s.sloss * p.W
    assert s.residues["dummy == 0"].value == 1.0


def test_System_postcompute():
    s = TopSystem("test")

    p = PtWPort("p", PortType.OUT)
    assert s.out.Pt == p.Pt
    assert s.out.W == p.W

    s.sub.compute()
    assert s.sub.out.Pt != p.Pt
    assert s.sub.out.W != p.W

    s.sub._postcompute()

    assert len(s.residues) == 0
    assert s.out.Pt != s.sub.out.Pt
    assert s.out.W != s.sub.out.W
    s._postcompute()
    assert s.out.Pt == p.Pt
    assert s.out.W == p.W
    assert len(s.residues) == 0
    assert_keys(s.sub.residues, "dummy == 0")
    assert s.sub.residues["dummy == 0"].value == 0.01


def test_System_run_once():
    s = TopSystem("test")
    s.in_.Pt = 123456.0
    s.in_.W = 3.14

    p = PtWPort("p", PortType.OUT)
    assert s.sub.in_.Pt == p.Pt
    assert s.sub.in_.W == p.W
    assert s.sub.out.Pt == p.Pt
    assert s.sub.out.W == p.W
    assert s.sub.residues["dummy == 0"].value == 1.0
    assert s.out.Pt == p.Pt
    assert s.out.W == p.W
    assert len(s.residues) == 0

    s.run_once()

    assert s.sub.in_.Pt == s.in_.Pt
    assert s.sub.in_.W == s.in_.W
    assert s.sub.out.Pt == s.sub.sloss * s.in_.Pt
    assert s.sub.out.W == s.sub.sloss * s.in_.W
    assert s.sub.residues["dummy == 0"].value == 0.01
    assert s.out.Pt == s.sub.out.Pt
    assert s.out.W == s.sub.out.W
    assert len(s.residues) == 0


def test_System_computed():
    with mock.patch("cosapp.core.signal.signal.inspect"):
        fake_callback = mock.Mock(spec=lambda **kwargs: None)
        fake_callback.return_value = None

        s = TopSystem("test_a")
        _ = TopSystem("test_b")
        s.computed.connect(Slot(fake_callback))
        s.run_children_drivers()

        fake_callback.assert_called_once_with()


def test_System_append_name2variable():
    s = System("test")

    d = {"a": 1, "b": "hello world"}

    s.append_name2variable(
        [(key, VariableReference(context=s, mapping=d, key=key)) for key in d]
    )
    for key in d:
        reference = s.name2variable[key]
        assert reference.mapping[reference.key] is d[key]

    s2 = SubSystem("sub")
    s.add_child(s2)

    s2.append_name2variable(
        [(key, VariableReference(context=s2, mapping=d, key=key)) for key in d]
    )
    for key in d:
        abs_key = ".".join((s2.name, key))
        reference = s.name2variable[abs_key]
        assert reference.mapping[reference.key] is d[key]


def test_System_pop_name2variable():
    s = TopSystem("test")
    keys = ["out.Pt", "out.W"]
    for key in keys:
        assert key in s.name2variable

    s.pop_name2variable(keys)
    for key in keys:
        assert key not in s.name2variable

    s2 = s.sub
    for key in keys:
        abs_name = ".".join((s2.name, key))
        assert abs_name in s.name2variable

    s2.pop_name2variable(keys)
    for key in keys:
        abs_key = ".".join((s2.name, key))
        assert abs_key not in s.name2variable


def test_System_open_loops():
    class S(System):
        def setup(self):
            self.add_inward("a_in")
            self.add_inward("b_in")
            self.add_input(XPort, "entry")
            self.add_output(XPort, "exit")
            self.add_outward("a_out")
            self.add_outward("b_out")

        def compute(self):
            self.exit.x = self.entry.x * self.a_in + self.b_in
            self.a_out = self.entry.x * self.a_in
            self.b_out = self.b_in / self.a_in

    # Breaking link between Port
    s = System("group")
    a = s.add_child(S("a"))
    b = s.add_child(S("b"))
    s.connect(b.exit, a.entry)
    s.connect(a.exit, b.entry)

    s.open_loops()
    child = s.children["b_exit_to_a_entry"]
    assert isinstance(child, IterativeConnector)

    assert_keys(child.inputs, System.INWARDS,
        IterativeConnector.GUESS, IterativeConnector.RESULT)
    assert all(isinstance(obj, ExtensiblePort) for obj in child.inputs.values())
    assert len(child.inputs[System.INWARDS]) == 0
    c_input = child.inputs[IterativeConnector.GUESS]
    assert isinstance(c_input, Port)
    assert len(c_input) == len(a.entry)
    c_input = child.inputs[IterativeConnector.RESULT]
    assert isinstance(c_input, Port)
    assert len(c_input) == len(b.exit)

    connectors = s.connectors
    connector = connectors["group_a_entry_to_b_exit_to_a_entry_guess"]
    assert connector.source is s.inputs["a_entry"]
    connector = connectors["b_exit_to_b_exit_to_a_entry_result"]
    assert connector.source is b.outputs["exit"]
    connector = connectors["group_a_entry_to_a_entry"]
    assert connector.sink is a.inputs["entry"]

    # Breaking link between ExtensiblePort
    s = System("group")
    a = s.add_child(S("a"))
    b = s.add_child(S("b"))
    s.connect(a.inwards, b.outwards, {"a_in": "a_out"})
    s.connect(a.exit, b.entry)

    s.open_loops()
    child = s.children["b_outwards_to_a_inwards"]
    assert isinstance(child, IterativeConnector)
    assert_keys(child.inputs, System.INWARDS,
        IterativeConnector.GUESS, IterativeConnector.RESULT)
    assert all(isinstance(obj, ExtensiblePort) for obj in child.inputs.values())
    assert len(child.inputs[System.INWARDS]) == 0
    c_input = child.inputs[IterativeConnector.GUESS]
    assert len(c_input) == 1
    assert "a_in" in c_input
    c_input = child.inputs[IterativeConnector.RESULT]
    assert len(c_input) == 1
    assert "a_out" in c_input

    connectors = s.connectors
    connector = connectors["group_inwards_to_b_outwards_to_a_inwards_guess"]
    assert connector.source is s.inputs["inwards"]
    connector = connectors["b_outwards_to_b_outwards_to_a_inwards_result"]
    assert connector.source is b.outputs["outwards"]
    connector = connectors["group_inwards_to_a_inwards"]
    assert connector.sink is a.inputs["inwards"]
    assert "a_a_in" in s.inputs[System.INWARDS]

    # Breaking a link between a Port and an ExtensiblePort
    s = System("group")
    a = s.add_child(S("a"))
    b = s.add_child(S("b"))
    s.connect(a.inwards, b.exit, {"a_in": "x"})
    s.connect(a.exit, b.entry)

    s.open_loops()
    child = s.children["b_exit_to_a_inwards"]
    assert isinstance(child, IterativeConnector)
    assert_keys(child.inputs, System.INWARDS,
        IterativeConnector.GUESS, IterativeConnector.RESULT)
    assert all(isinstance(obj, ExtensiblePort) for obj in child.inputs.values())
    assert len(child[System.INWARDS]) == 0
    assert isinstance(child[IterativeConnector.RESULT], Port)
    c_input = child.inputs[IterativeConnector.GUESS]
    assert len(c_input) == 1
    assert "a_in" in c_input
    c_input = child.inputs[IterativeConnector.RESULT]
    assert len(c_input) == len(b.exit)

    connectors = s.connectors
    connector = connectors["group_inwards_to_b_exit_to_a_inwards_guess"]
    assert connector.source is s.inputs["inwards"]
    connector = connectors["b_exit_to_b_exit_to_a_inwards_result"]
    assert connector.source is b.outputs["exit"]
    connector = connectors["group_inwards_to_a_inwards"]
    assert connector.sink is a.inputs["inwards"]
    assert "a_a_in" in s.inputs[System.INWARDS]

    # Test forcing transfer
    s = System("group")
    a = s.add_child(S("a"))
    b = s.add_child(S("b"))
    s.connect(b.exit, a.entry)
    s.connect(a.exit, b.entry)
    b.exit.x = 123
    a.entry.x = 123
    s.open_loops()
    assert a.entry.x == b.exit.x == 123

    s = System("group")
    a = s.add_child(S("a"))
    b = s.add_child(S("b"))
    s.connect(b.exit, a.entry)
    s.connect(a.exit, b.entry)
    b.exit.x = 123
    a.entry.x = 1
    s.open_loops()
    assert a.entry.x == b.exit.x == 123

    # Test system depending on two others not already executed
    class SimpleSystem(System):
        def setup(self):
            self.add_inward("x")
            self.add_inward("y")
            self.add_outward("z")

    top = System("top")
    top.add_child(SimpleSystem("s1"))
    top.add_child(SimpleSystem("s2"))
    top.add_child(SimpleSystem("s3"))
    top.connect(top.s1.inwards, top.s2.outwards, {"x": "z"})
    top.connect(top.s1.inwards, top.s3.outwards, {"y": "z"})
    top.open_loops()

    assert "s2_outwards_to_s1_inwards" in top.children
    assert "s3_outwards_to_s1_inwards" in top.children
    assert "s1_x" in top.inputs[System.INWARDS]
    assert "s1_y" in top.inputs[System.INWARDS]
    connectors = top.connectors
    assert "top_inwards_to_s1_inwards" in connectors
    assert connectors["top_inwards_to_s1_inwards"].sink is top.s1.inputs[System.INWARDS]
    assert connectors["top_inwards_to_s1_inwards"].source is top.inputs[System.INWARDS]
    assert connectors["top_inwards_to_s1_inwards"].variable_mapping == {"x": "s1_x", "y": "s1_y"}

    # Test system depending on two others not already executed
    class SimpleSystem(System):
        def setup(self):
            self.add_inward("x")
            self.add_inward("y")
            self.add_outward("z")

    top = System("top")
    top.add_child(SimpleSystem("s1"))
    top.add_child(SimpleSystem("s2"))
    top.add_child(SimpleSystem("s3"))
    top.connect(top.s1.inwards, top.s2.outwards, {"x": "z"})
    top.connect(top.s1.inwards, top.s3.outwards, {"y": "z"})
    top.open_loops()

    assert "s2_outwards_to_s1_inwards" in top.children
    assert "s3_outwards_to_s1_inwards" in top.children
    assert "s1_x" in top.inputs[System.INWARDS]
    assert "s1_y" in top.inputs[System.INWARDS]
    connectors = top.connectors
    assert "top_inwards_to_s1_inwards" in connectors
    assert connectors["top_inwards_to_s1_inwards"].sink is top.s1.inputs[System.INWARDS]
    assert connectors["top_inwards_to_s1_inwards"].source is top.inputs[System.INWARDS]
    assert connectors["top_inwards_to_s1_inwards"].variable_mapping == {"x": "s1_x", "y": "s1_y"}

    # Test system depending on two others not already executed
    top = System("top")
    top.add_child(SimpleSystem("s1"))
    top.add_child(SimpleSystem("s2"))
    top.add_child(SimpleSystem("s3"))
    top.connect(top.s1.inwards, top.s2.outwards, {"x": "z"})
    top.connect(top.s1.inwards, top.s3.outwards, {"y": "z"})
    top.open_loops()

    assert "s2_outwards_to_s1_inwards" in top.children
    assert "s3_outwards_to_s1_inwards" in top.children
    assert "s1_x" in top.inputs[System.INWARDS]
    assert "s1_y" in top.inputs[System.INWARDS]
    connectors = top.connectors
    assert "top_inwards_to_s1_inwards" in connectors
    assert connectors["top_inwards_to_s1_inwards"].sink is top.s1.inputs[System.INWARDS]
    assert connectors["top_inwards_to_s1_inwards"].source is top.inputs[System.INWARDS]
    assert connectors["top_inwards_to_s1_inwards"].variable_mapping == {"x": "s1_x", "y": "s1_y"}


def test_System_close_loops():
    class S(System):
        def setup(self):
            self.add_inward("a_in")
            self.add_inward("b_in")
            self.add_input(XPort, "entry")
            self.add_output(XPort, "exit")
            self.add_outward("a_out")
            self.add_outward("b_out")

        def compute(self):
            self.exit.x = self.entry.x * self.a_in + self.b_in
            self.a_out = self.entry.x * self.a_in
            self.b_out = self.b_in / self.a_in

    # Breaking link between Port
    s = System("group")
    a = s.add_child(S("a"))
    b = s.add_child(S("b"))
    s.connect(b.exit, a.entry)
    s.connect(a.exit, b.entry)

    s.open_loops()
    s.close_loops()
    # Connection is restored
    connector = s.connectors["b_exit_to_a_entry"]
    assert connector.sink is s.a.entry
    assert connector.source is s.b.exit

    # Parent has no more pulled port
    assert "a_entry" not in s.inputs
    # Parent has no trace in name2variable
    for key in ("a_entry", "a_entry.x"):
        assert key not in s.name2variable

    # Breaking link between ExtensiblePort
    s = System("group")
    a = s.add_child(S("a"))
    b = s.add_child(S("b"))
    s.connect(a.inwards, b.outwards, {"a_in": "a_out"})
    s.connect(a.exit, b.entry)

    s.open_loops()
    s.close_loops()

    connector = s.connectors["b_outwards_to_a_inwards"]
    assert connector.sink is s.a.inwards
    assert connector.source is s.b.outwards
    assert connector.variable_mapping == {"a_in": "a_out"}

    for key in ("a_a_in", "inwards.a_a_in"):
        assert key not in s.name2variable
    assert "inwards" in s.name2variable

    # Breaking a link between a Port and an ExtensiblePort
    s = System("group")
    a = s.add_child(S("a"))
    b = s.add_child(S("b"))
    s.connect(a.inwards, b.exit, {"a_in": "x"})
    s.connect(a.exit, b.entry)

    s.open_loops()
    s.close_loops()

    connector = s.connectors["b_exit_to_a_inwards"]
    assert connector.sink is s.a.inwards
    assert connector.source is s.b.exit
    assert connector.variable_mapping == {"a_in": "x"}

    for key in ("a_a_in", "inwards.a_a_in"):
        assert key not in s.name2variable
    assert "inwards" in s.name2variable

    # Breaking a link between two Ports with mapping
    class YPort(Port):
        def setup(self):
            self.add_variable("y")

    class T(System):
        def setup(self):
            self.add_input(YPort, "entry")
            self.add_output(XPort, "exit")

    s = System("group")
    a = s.add_child(T("a"))
    b = s.add_child(S("b"))
    s.connect(b.exit, a.entry, {"x": "y"})
    s.connect(a.exit, b.entry)

    s.open_loops()
    s.close_loops()

    # Connection is restored
    connector = s.connectors["b_exit_to_a_entry"]
    assert connector.sink is s.a.entry
    assert connector.source is s.b.exit
    assert connector.variable_mapping == {"y": "x"}

    # Parent has no more pulled port
    assert "a_entry" not in s.inputs
    # Parent has no trace in name2variable
    for key in ("a_entry", "a_entry.y"):
        assert key not in s.name2variable

def test_System_clean_partial_inwards():
    # This test case comes from the following configuration
    # A system pull an inward to its parent, and a second one
    # is pulled by opening a loop. At closing, the wanted
    # pulled variable got removed.
    class Sub1(System):
        def setup(self):
            self.add_inward('user')
            self.add_inward('loop')
    
    class Sub2(System):
        def setup(self):
            self.add_outward('loop')

    top = System('top')
    top.add_child(Sub1('sink'), pulling="user")
    top.add_child(Sub2('source'))
    top.connect(top.source.outwards, top.sink.inwards, "loop")
    top.exec_order = ['sink', 'source']

    top.open_loops()
    assert 'sink_loop' in top
    assert 'inwards.sink_loop' in top
    assert 'user' in top
    assert 'inwards.user' in top
    top.close_loops()
    assert 'sink_loop' not in top
    assert 'inwards.sink_loop' not in top
    assert 'user' in top
    assert 'inwards.user' in top

def test_System_check():
    T = TopSystem("test")
    r = T.check()
    assert r == {
        "inwards.top_k": Validity.OK,
        "outwards.top_tmp": Validity.OK,
        "in_.Pt": Validity.OK,
        "in_.W": Validity.OK,
        "out.Pt": Validity.OK,
        "out.W": Validity.OK,
        "sub.in_.Pt": Validity.OK,
        "sub.in_.W": Validity.OK,
        "sub.out.Pt": Validity.OK,
        "sub.out.W": Validity.OK,
        "sub.inwards.sloss": Validity.OK,
        "sub.outwards.dummy": Validity.OK,
        "sub.outwards.tmp": Validity.OK,
    }

    T = TopSystem("test")
    T.sub.sloss = 0.7
    T.in_.Pt = -1.0
    r = T.check()
    assert r == {
        "inwards.top_k": Validity.OK,
        "outwards.top_tmp": Validity.OK,
        "in_.Pt": Validity.ERROR,
        "in_.W": Validity.OK,
        "out.Pt": Validity.OK,
        "out.W": Validity.OK,
        "sub.in_.Pt": Validity.OK,
        "sub.in_.W": Validity.OK,
        "sub.out.Pt": Validity.OK,
        "sub.out.W": Validity.OK,
        "sub.inwards.sloss": Validity.WARNING,
        "sub.outwards.dummy": Validity.OK,
        "sub.outwards.tmp": Validity.OK,
    }

    s = System("noname")
    with pytest.raises(AttributeError):
        s.check("a")


def test_System_add_unknowns(DummyFactory):
    m = DummyFactory("dummy", base=Multiply1,
        unknowns=get_args("K1", max_rel_step=0.01, lower_bound=-10.0),
    )

    unknown = m.get_unsolved_problem().unknowns["inwards.K1"]
    assert isinstance(unknown, Unknown)
    assert unknown.name == "inwards.K1"
    assert unknown.port == "inwards"
    assert unknown.max_rel_step == 0.01
    assert unknown.max_abs_step == np.inf
    assert unknown.lower_bound == -10
    assert unknown.upper_bound == np.inf
    assert unknown.mask is None

    m = DummyFactory("dummy", base=Multiply1,
        unknowns=get_args(
            "p_in.x",
            max_rel_step=0.1,
            max_abs_step=1e6,
            lower_bound=0.0,
            upper_bound=1e8,
        ),
    )

    unknown = m.get_unsolved_problem().unknowns["p_in.x"]
    assert isinstance(unknown, Unknown)
    assert unknown.name == "p_in.x"
    assert unknown.port == "p_in"
    assert unknown.max_rel_step == 0.1
    assert unknown.max_abs_step == 1e6
    assert unknown.lower_bound == 0
    assert unknown.upper_bound == 1e8
    assert unknown.mask is None

    with pytest.raises(ValueError, match="Only variables in input ports can be used as boundaries"):
        DummyFactory("dummy", base=Multiply1, unknowns=get_args("p_out.x"))

    with pytest.raises(AttributeError):
        DummyFactory("dummy", base=Multiply1, unknowns=get_args("foo"))

    # Test mask
    v = DummyFactory("v", base=Strait1dLine, unknowns=get_args("a"))
    
    unknown = v.get_unsolved_problem().unknowns["inwards.a"]
    assert np.array_equal(unknown.mask, [True, True, True])

    v = DummyFactory("v", base=Strait1dLine, unknowns=get_args("a[::2]"))
    
    unknown = v.get_unsolved_problem().unknowns["inwards.a"]
    assert np.array_equal(unknown.mask, [True, False, True])

    with pytest.raises(IndexError):
        DummyFactory("dummy", base=Strait1dLine, unknowns=get_args("a[[1, 3]]"))


def test_System_add_equation(DummyFactory):
    class ASyst(System):
        def setup(self):
            self.add_inward("x", 1.0)
            m = self.add_equation("x == 0", name="cancel_x")
            self.add_outward("math_problem", m)

    s = ASyst("s")
    assert s.math_problem is s._math
    with pytest.raises(AttributeError, match="`add_equation` cannot be called outside `setup`"):
        s.add_equation("x == 3.14")
    residues = s.residues
    assert_keys(residues, "cancel_x")
    for name, residue in residues.items():
        assert isinstance(residue, Residue)
        assert residue.name == name
    assert residues["cancel_x"].value == 1

    s = DummyFactory("dummy",
        inwards = [
            get_args("x", 1.0),
            get_args("y", 1.0),
            get_args("z", 1.0),
        ],
        equations = get_args([
            "x == 0",
            "y == 3",
            dict(equation="z == 5", name="test_r", reference=25.0),
        ]),
    )
    residues = s.residues
    assert_keys(residues, "x == 0", "y == 3", "test_r")
    for name, residue in residues.items():
        assert isinstance(residue, Residue)
        assert residue.name == name
    assert residues["x == 0"].value == 1
    assert residues["y == 3"].value == -2
    assert residues["test_r"].value == -4 / 25
    assert residues["test_r"].reference == 25


def test_System_add_design_method():
    class ASyst(System):
        def setup(self):
            m = self.add_design_method("method1")
            self.add_outward("math_problem", m)

    a = ASyst("a")
    with pytest.raises(AttributeError, match="`add_design_method` cannot be called outside `setup`"):
        a.add_design_method("methodX")
    assert isinstance(a.design("method1"), MathematicalProblem)
    assert a.design("method1") is a.math_problem


def test_System_design(DummyFactory):
    a = DummyFactory("a", design_methods=get_args("method1"))
    with pytest.raises(KeyError):
        a.design("method2")


@pytest.mark.parametrize("ctor_data, expected_data", [
    (dict(), dict()),
    (
        dict(
            inwards = [get_args("x", 22.0), get_args("y", 42.0)],
            equations = get_args("x == 1", name="dummy", reference="norm"),
        ),
        dict(residues = {"dummy": dict(value=2.1, reference=10)})
    ),
    (
        dict(
            inwards = [get_args("x", 22.0), get_args("y", 42.0)],
            unknowns = get_args("y"),
            equations = get_args("x == 1", name="dummy", reference="norm"),
        ),
        dict(
            unknowns = {"inwards.y": dict()},
            residues = {"dummy": dict(value=2.1)}
        )
    ),
    (
        dict(
            inwards = [get_args("x", 22.0), get_args("y", 42.0)],
            unknowns = get_args("x", 1.0, 0.1, -5, 40.0),
            equations = get_args("y == 1", name="dummy"),
        ),
        dict(
            n_residues = 1,
            n_unknowns = 1,
            unknowns = {"inwards.x": dict(
                max_abs_step = 1,
                max_rel_step = 0.1,
                lower_bound = -5,
                upper_bound = 40,
            )},
            residues = {"dummy": dict(value=41, reference=1)}
        )
    ),
])
def test_System_get_unsolved_problem(DummyFactory, ctor_data, expected_data):
    system = DummyFactory("test", **ctor_data)  # test object
    problem = system.get_unsolved_problem()
    assert isinstance(problem, MathematicalProblem)
    expected_unknowns = expected_data.get("unknowns", dict())
    expected_residues = expected_data.get("residues", dict())
    assert problem.shape == (len(expected_unknowns), len(expected_residues))
    # Test unknowns
    assert_keys(system.unknowns, *expected_unknowns.keys())
    assert_keys(problem.unknowns, *expected_unknowns.keys())
    for name, unknown in problem.unknowns.items():
        expected = { 
            # default values:
            "max_abs_step": np.inf,
            "max_rel_step": np.inf,
            "lower_bound": -np.inf,
            "upper_bound": np.inf,
        }
        expected.update(expected_unknowns[name])  # overwrite if present
        assert isinstance(unknown, Unknown)
        assert unknown is system._math.unknowns[name]
        assert unknown.name == name
        assert unknown.lower_bound == pytest.approx(expected['lower_bound'], rel=1e-14)
        assert unknown.upper_bound == pytest.approx(expected['upper_bound'], rel=1e-14)
        assert unknown.max_abs_step == pytest.approx(expected['max_abs_step'], rel=1e-14)
        assert unknown.max_rel_step == pytest.approx(expected['max_rel_step'], rel=1e-14)
    # Test equations/residues
    assert_keys(system.residues, *expected_residues.keys())
    assert_keys(problem.residues, *expected_residues.keys())
    for name, residue in problem.residues.items():
        expected = expected_residues[name]
        assert isinstance(residue, Residue)
        assert residue is system._math.residues[name]
        assert residue.name == name
        assert residue.value == pytest.approx(expected["value"], rel=1e-14)


def test_System_get_unsolved_problem_seq(DummyFactory):
    """Non-parametric version of `test_System_get_unsolved_system()`.
    Tests method `get_unsolved_problem` for systems with children."""
    a = DummyFactory("a",
        inwards = [get_args("x", 22.0), get_args("y", 42.0)],
        unknowns = get_args("x", 1.0, 0.1, -5, 40.0),
        equations = get_args("y == 1", name="dummy"),
    )
    problem = a.get_unsolved_problem()
    assert isinstance(problem, MathematicalProblem)
    assert len(problem.unknowns) == 1
    assert len(problem.residues) == 1

    T = DummyFactory("top")
    assert isinstance(T, System)
    T.add_child(a)
    problem = T.get_unsolved_problem()
    assert isinstance(problem, MathematicalProblem)
    assert len(problem.unknowns) == 1
    assert len(problem.residues) == 1
    # Test unknowns
    assert_keys(problem.unknowns, "a.inwards.x")
    unknown = problem.unknowns["a.inwards.x"]
    assert unknown is a._math.unknowns["inwards.x"]
    assert unknown.name == "inwards.x"
    assert unknown.context is a
    # Test residues/equations
    assert_keys(problem.residues, "a.(dummy)")
    residue = problem.residues["a.(dummy)"]
    assert residue is a._math.residues["dummy"]
    assert residue.name == "dummy"
    assert residue.context is a

    # Test that unknowns are suppressed by connection to a peer
    T = DummyFactory('top')
    b = DummyFactory('b', inwards=get_args('x'), outwards=get_args('y'))
    c = DummyFactory('c', base=b.__class__, unknowns=get_args('x'))
    T.add_child(b)
    T.add_child(c)
    T.connect(T.b.outwards, T.c.inwards, {"y": "x"})

    problem = T.get_unsolved_problem()
    assert problem.shape == (0, 0)

    # Test that unknowns are forwarded by connection to the parent
    T = DummyFactory('top')
    s = DummyFactory('sub',
        inwards=get_args('x'),
        outwards=get_args('y'),
        unknowns=get_args('x'),
        )
    T.add_child(s, pulling=["x"])

    problem = T.get_unsolved_problem()
    assert problem.shape == (1, 0)
    unknown = problem.unknowns["inwards.x"]
    assert unknown.context is T


def test_System_connect_ports(caplog, DummyFactory):
    caplog.set_level(logging.DEBUG)

    s1 = SubSystem("s1")
    s2 = SubSystem("s2")
    group = System("hat")
    group.add_child(s1)
    group.add_child(s2)

    assert len(group.connectors) == 0
    group.connect(s1.out, s2.in_)
    connectors = group.connectors
    assert all(isinstance(c, Connector) for c in connectors.values())
    assert_keys(connectors, "s1_out_to_s2_in_")
    connector = connectors["s1_out_to_s2_in_"]
    assert connector.source is s1.out
    assert connector.sink is s2.in_
    assert connector._mapping == {"Pt": "Pt", "W": "W"}
    assert connector._unit_conversions == {"Pt": (1, 0), "W": (1, 0)}

    # Partial connection
    caplog.clear()
    s1 = SubSystem("s1")
    s2 = SubSystem("s2")
    group = System("hat")
    group.add_child(s1)
    group.add_child(s2)
    group.connect(s1.out, s2.in_, "Pt")
    connectors = group.connectors
    assert all(isinstance(c, Connector) for c in connectors.values())
    assert_keys(connectors, "s1_out_to_s2_in_")
    connector = connectors["s1_out_to_s2_in_"]
    assert connector.source is s1.out
    assert connector.sink is s2.in_
    assert connector._mapping == {"Pt": "Pt"}
    assert connector._unit_conversions == {"Pt": (1, 0)}

    records = list(filter(lambda record: record.levelno == logging.DEBUG, caplog.records))
    assert len(records) == 1
    assert re.match(
        r"Partial connection between '\w+\.\w+' and '\w+\.\w+'\. "
        r"Variables \(\w+\.\w+, \w+\.\w+\) are not part of the mapping",
        records[-1].message)

    # Explicit full connection
    s1 = SubSystem("s1")
    s2 = SubSystem("s2")
    group = System("hat")
    group.add_child(s1)
    group.add_child(s2)
    group.connect(s1.out, s2.in_, {"Pt": "Pt", "W": "W"})
    connectors = group.connectors
    assert all(isinstance(c, Connector) for c in connectors.values())
    assert_keys(connectors, "s1_out_to_s2_in_")
    connector = connectors["s1_out_to_s2_in_"]
    assert connector.source is s1.out
    assert connector.sink is s2.in_
    assert connector._mapping == {"Pt": "Pt", "W": "W"}
    assert connector._unit_conversions == {"Pt": (1, 0), "W": (1, 0)}


    s1 = SubSystem("s1")
    group = DummyFactory("hat",
        inputs=get_args(PtWPort, "hat_in"),
        outputs=get_args(PtWPort, "hat_out"))
    group.add_child(s1)

    assert len(group.connectors) == 0
    assert len(s1.connectors) == 0

    group.connect(group.hat_in, s1.in_)
    connectors = group.connectors
    assert all(isinstance(c, Connector) for c in connectors.values())
    assert_keys(connectors, "hat_hat_in_to_s1_in_")
    connector = connectors["hat_hat_in_to_s1_in_"]
    assert connector.source is group.hat_in
    assert connector.sink is s1.in_
    assert connector._mapping == {"Pt": "Pt", "W": "W"}
    assert connector._unit_conversions == {"Pt": (1, 0), "W": (1, 0)}
    # Add new connection
    group.connect(s1.out, group.hat_out)
    connectors = group.connectors
    assert len(s1.connectors) == 0
    assert all(isinstance(c, Connector) for c in connectors.values())
    assert_keys(connectors, "hat_hat_in_to_s1_in_", "s1_out_to_hat_hat_out")
    connector = connectors["s1_out_to_hat_hat_out"]
    assert connector.source is s1.out
    assert connector.sink is group.hat_out
    assert connector._mapping == {"Pt": "Pt", "W": "W"}
    assert connector._unit_conversions == {"Pt": (1, 0), "W": (1, 0)}

    # Same as previous test, with different args order in group.connect(...)
    s1 = SubSystem("s1")
    group = DummyFactory("hat",
        inputs=get_args(PtWPort, "hat_in"),
        outputs=get_args(PtWPort, "hat_out"))
    group.add_child(s1)

    assert len(group.connectors) == 0
    assert len(s1.connectors) == 0

    group.connect(s1.in_, group.hat_in)
    connectors = group.connectors
    assert all(isinstance(c, Connector) for c in connectors.values())
    assert_keys(connectors, "hat_hat_in_to_s1_in_")
    connector = connectors["hat_hat_in_to_s1_in_"]
    assert connector.source is group.hat_in
    assert connector.sink is s1.in_
    assert connector._mapping == {"Pt": "Pt", "W": "W"}
    assert connector._unit_conversions == {"Pt": (1, 0), "W": (1, 0)}
    # Add new connection
    group.connect(group.hat_out, s1.out)
    connectors = group.connectors
    assert len(s1.connectors) == 0
    assert all(isinstance(c, Connector) for c in connectors.values())
    assert_keys(connectors, "hat_hat_in_to_s1_in_", "s1_out_to_hat_hat_out")
    connector = connectors["s1_out_to_hat_hat_out"]
    assert connector.source is s1.out
    assert connector.sink is group.hat_out
    assert connector._mapping == {"Pt": "Pt", "W": "W"}
    assert connector._unit_conversions == {"Pt": (1, 0), "W": (1, 0)}

    with pytest.raises(TypeError):
        group.connect(s1.out, group)

    with pytest.raises(TypeError):
        group.connect(s1, group.hat_in)

    g1 = System("group1")
    s1 = g1.add_child(SubSystem("s1"))
    g2 = System("group2")
    s2 = g2.add_child(SubSystem("s2"))

    top = System("top")
    top.add_child(g1)
    top.add_child(g2)

    pattern = r"Only ports belonging to direct children of '\w+' can be connected"

    with pytest.raises(ConnectorError, match=pattern):
        group.connect(s1.out, s2.in_)

    top = System("top")
    s1 = top.add_child(SubSystem("s1"))
    s2 = top.add_child(SubSystem("s2"))

    with pytest.raises(ConnectorError, match=pattern):
        s1.connect(s1.out, s2.in_)

    with pytest.raises(ConnectorError, match="Connecting ports of the same system is forbidden"):
        top.connect(s1.out, s1.in_)


def test_System_connect_hybrid(DummyFactory):
    # TODO Port to ExtensiblePort and ExtensiblePort to Port
    class CopyCatPort(Port):
        def setup(self):
            self.add_variable("Pt", 101325.0, unit="Pa")
            self.add_variable("W", 1.0, unit="kg/s")

    s1 = SubSystem("s1")
    s2 = DummyFactory("s2", inputs=get_args(CopyCatPort, "in_"))
    group = System("hat")
    group.add_child(s1)
    group.add_child(s2)
    group.connect(s2.in_, s1.out, {"Pt": "Pt", "W": "W"})

    connectors = group.connectors
    assert len(s1.connectors) == 0
    assert all(isinstance(c, Connector) for c in connectors.values())
    assert_keys(connectors, "s1_out_to_s2_in_")
    connector = connectors["s1_out_to_s2_in_"]
    assert connector.source is s1.out
    assert connector.sink is s2.in_
    assert connector._mapping == {"Pt": "Pt", "W": "W"}
    assert connector._unit_conversions == {"Pt": (1, 0), "W": (1, 0)}

    class CopyCatPort(Port):
        def setup(self):
            self.add_variable("P", 101325.0, unit="Pa")
            self.add_variable("Mfr", 1.0, unit="lbm/s")

    s1 = SubSystem("s1")
    s2 = DummyFactory("s2", inputs=get_args(CopyCatPort, "in_"))
    group = System("hat")
    group.add_child(s1)
    group.add_child(s2)
    group.connect(s2.in_, s1.out, {"P": "Pt", "Mfr": "W"})

    connectors = group.connectors
    assert len(s1.connectors) == 0
    assert all(isinstance(c, Connector) for c in connectors.values())
    assert_keys(connectors, "s1_out_to_s2_in_")
    connector = connectors["s1_out_to_s2_in_"]
    assert connector.source is s1.out
    assert connector.sink is s2.in_
    assert connector._mapping == {"P": "Pt", "Mfr": "W"}
    assert connector._unit_conversions == {"P": (1, 0), "Mfr": pytest.approx((2.2046226218487757, 0), abs=1e-14)}

    # Uncompatible unit
    class CopyCatPort(Port):
        def setup(self):
            self.add_variable("P", 101325.0, unit="N")
            self.add_variable("Mfr", 1.0, unit="kg/s")

    s1 = SubSystem("s1")
    s2 = DummyFactory("s2", inputs=get_args(CopyCatPort, "in_"))
    group = System("hat")
    group.add_child(s1)
    group.add_child(s2)
    with pytest.raises(UnitError):
        group.connect(s2.in_, s1.out, {"P": "Pt", "Mfr": "W"})


def test_System_connect_full():
    class DummyPort(Port):
        def setup(self):
            self.add_variable("a", 1)
            self.add_variable("b", 2)

    class System1(System):
        def setup(self):
            self.add_inward({"test": 7.0, "a": 25.0, "b": 42.0})
            self.add_outward({"local1": 11.0, "local2": 22.0, "local3": 33.0})

    class System2(System):
        def setup(self):
            self.add_inward({"data1": 9.0, "data2": 11.0, "data3": 13.0})
            self.add_outward({"test": 7.0, "a": 14.0, "b": 21.0})

    class System3(System):
        def setup(self):
            self.add_input(DummyPort, "entry")
            self.add_output(DummyPort, "exit")

    group = System("hat")
    group.add_child(System1("s1"))
    group.add_child(System2("s2"))
    group.add_child(System3("s3"))
    group.add_child(System3("s4"))

    group.connect(group.s1.inwards, group.s3.entry)
    connectors = group.connectors
    assert all(isinstance(c, Connector) for c in connectors.values())
    assert_keys(connectors, "hat_s3_entry_to_s3_entry", "hat_s3_entry_to_s1_inwards")
    # First connector:
    connector = connectors["hat_s3_entry_to_s3_entry"]
    assert connector.source is group.s3_entry
    assert connector.sink is group.s3.entry
    assert connector._mapping == {"a": "a", "b": "b"}
    assert connector._unit_conversions == {"a": (1, 0), "b": (1, 0)}
    # Second connector:
    connector = connectors["hat_s3_entry_to_s1_inwards"]
    assert connector.source is group.s3_entry
    assert connector.sink is group.s1.inwards
    assert connector._mapping == {"a": "a", "b": "b"}
    assert connector._unit_conversions == {"a": (1, 0), "b": (1, 0)}

    with pytest.raises(ConnectorError, match="already connected"):
        group.connect(group.s3.entry, group.s1.inwards)

    group = System("hat")
    group.add_child(System1("s1"))
    group.add_child(System2("s2"))
    group.add_child(System3("s3"))
    group.add_child(System3("s4"))

    group.connect(group.s1.inwards, group.s3.exit)
    connectors = group.connectors
    assert all(isinstance(c, Connector) for c in connectors.values())
    assert_keys(connectors, "s3_exit_to_s1_inwards")
    connector = connectors["s3_exit_to_s1_inwards"]
    assert connector.source is group.s3.exit
    assert connector.sink is group.s1.inwards
    assert connector._mapping == {"a": "a", "b": "b"}
    assert connector._unit_conversions == {"a": (1, 0), "b": (1, 0)}
    # Add new connection
    group.connect(group.s2.outwards, group.s4.entry)
    connectors = group.connectors
    assert all(isinstance(c, Connector) for c in connectors.values())
    assert_keys(connectors, "s3_exit_to_s1_inwards", "s2_outwards_to_s4_entry")
    connector = connectors["s2_outwards_to_s4_entry"]
    assert connector.source is group.s2.outwards
    assert connector.sink is group.s4.entry
    assert connector._mapping == {"a": "a", "b": "b"}
    assert connector._unit_conversions == {"a": (1, 0), "b": (1, 0)}

    with pytest.raises(ConnectorError, match=r"s1.inwards.a is already set by Connector\(s1.inwards <- s3.exit"):
        group.connect(group.s1.inwards, group.s2.outwards)

    group = System("hat")
    group.add_child(System1("s1"))
    group.add_child(System2("s2"))
    group.add_child(System3("s3"))
    group.add_child(System3("s4"))

    group.connect(group.s1.inwards, group.s2.outwards)
    connectors = group.connectors
    assert all(isinstance(c, Connector) for c in connectors.values())
    assert_keys(connectors, "s2_outwards_to_s1_inwards")
    connector = connectors["s2_outwards_to_s1_inwards"]
    assert connector.source is group.s2.outwards
    assert connector.sink is group.s1.inwards
    assert connector._mapping == {"a": "a", "b": "b", "test": "test"}


def test_System_connect_partial():
    class DummyPort(Port):
        def setup(self):
            self.add_variable("a", 1)
            self.add_variable("b", 2)

    class System1(System):
        def setup(self):
            self.add_inward({"data1": 3.0, "data2": 5.0, "data3": 7.0, "b": 9.0})
            self.add_outward(
                {"local1": 11.0, "local2": 22.0, "local3": 33.0, "a": 44}
            )

    class System2(System):
        def setup(self):
            self.add_inward({"d1": 9.0, "d2": 11.0, "d3": 13.0, "a": 17.0})
            self.add_outward({"l1": 7.0, "l2": 14.0, "l3": 21.0, "b": 28})

    class System3(System):
        def setup(self):
            self.add_input(DummyPort, "entry")
            self.add_output(DummyPort, "exit")

    group = System("hat")
    group.add_child(System1("s1"))
    group.add_child(System2("s2"))
    group.add_child(System3("s3"))
    group.add_child(System3("s4"))

    group.connect(group.s1.inwards, group.s3.entry, "b")
    connectors = group.connectors
    assert all(isinstance(c, Connector) for c in connectors.values())
    assert_keys(connectors, "hat_s3_entry_to_s3_entry", "hat_s3_entry_to_s1_inwards")
    # First connector:
    connector = connectors["hat_s3_entry_to_s3_entry"]
    assert connector.source is group.s3_entry
    assert connector.sink is group.s3.entry
    assert connector._mapping == {"b": "b"}
    assert connector._unit_conversions == {"b": (1, 0)}
    # Second connector:
    connector = connectors["hat_s3_entry_to_s1_inwards"]
    assert connector.source is group.s3_entry
    assert connector.sink is group.s1.inwards
    assert connector._mapping == {"b": "b"}
    assert connector._unit_conversions == {"b": (1, 0)}

    with pytest.raises(ConnectorError, match="already connected"):
        group.connect(group.s3.entry, group.s1.inwards, "b")

    group = System("hat")
    group.add_child(System1("s1"))
    group.add_child(System2("s2"))
    group.add_child(System3("s3"))
    group.add_child(System3("s4"))

    group.connect(group.s1.inwards, group.s3.exit, "b")
    connectors = group.connectors
    assert all(isinstance(c, Connector) for c in connectors.values())
    assert_keys(connectors, "s3_exit_to_s1_inwards")
    connector = connectors["s3_exit_to_s1_inwards"]
    assert connector.source is group.s3.exit
    assert connector.sink is group.s1.inwards
    assert connector._mapping == {"b": "b"}
    assert connector._unit_conversions == {"b": (1, 0)}
    # Add new connection
    group.connect(group.s3.exit, group.s4.entry, ["a", "b"])
    connectors = group.connectors
    assert all(isinstance(c, Connector) for c in connectors.values())
    assert_keys(connectors, "s3_exit_to_s1_inwards", "s3_exit_to_s4_entry")
    connector = connectors["s3_exit_to_s4_entry"]
    assert connector.source is group.s3.exit
    assert connector.sink is group.s4.entry
    assert connector._mapping == {"a": "a", "b": "b"}

    group = System("hat")
    group.add_child(System1("s1"))
    group.add_child(System2("s2"))
    group.add_child(System3("s3"))
    group.add_child(System3("s4"))

    group.connect(group.s1.inwards, group.s2.inwards, {"data1": "d1"})
    connectors = group.connectors
    assert all(isinstance(c, Connector) for c in connectors.values())
    assert_keys(connectors, "hat_inwards_to_s1_inwards", "hat_inwards_to_s2_inwards")
    # First connector
    connector = connectors["hat_inwards_to_s1_inwards"]
    assert connector.source is group.inwards
    assert connector.sink is group.s1.inwards
    assert connector._mapping == {"data1": "s2_d1"}
    # Second connector
    connector = connectors["hat_inwards_to_s2_inwards"]
    assert connector.source is group.inwards
    assert connector.sink is group.s2.inwards
    assert connector._mapping == {"d1": "s2_d1"}

    group = System("hat")
    group.add_child(System1("s1"))
    group.add_child(System2("s2"))
    group.add_child(System3("s3"))
    group.add_child(System3("s4"))

    group.connect(group.s3.exit, group.s4.entry, {"a": "a", "b": "b"})
    connectors = group.connectors
    assert all(isinstance(c, Connector) for c in connectors.values())
    assert_keys(connectors, "s3_exit_to_s4_entry")
    connector = connectors["s3_exit_to_s4_entry"]
    assert connector.source is group.s3.exit
    assert connector.sink is group.s4.entry
    assert connector._mapping == {"a": "a", "b": "b"}
    # Add new connection
    group.connect(group.s2.inwards, group.s1.outwards,
        {"d1": "local1", "d2": "local2"})
    connectors = group.connectors
    assert all(isinstance(c, Connector) for c in connectors.values())
    assert_keys(connectors, "s3_exit_to_s4_entry", "s1_outwards_to_s2_inwards")
    connector = connectors["s1_outwards_to_s2_inwards"]
    assert connector.source is group.s1.outwards
    assert connector.sink is group.s2.inwards
    assert connector._mapping == {"d1": "local1", "d2": "local2"}
    # Add new connection
    group.connect(group.s2.outwards, group.s1.inwards,
        {"l1": "data1", "l2": "data2"})
    connectors = group.connectors
    assert all(isinstance(c, Connector) for c in connectors.values())
    assert_keys(connectors,
        "s3_exit_to_s4_entry", "s1_outwards_to_s2_inwards", "s2_outwards_to_s1_inwards")
    connector = connectors["s2_outwards_to_s1_inwards"]
    assert connector.source is group.s2.outwards
    assert connector.sink is group.s1.inwards
    assert connector._mapping == {"data1": "l1", "data2": "l2"}

    g1 = System("group1")
    s1 = g1.add_child(System3("s1"))
    g2 = System("group2")
    s2 = g2.add_child(System3("s2"))
    top = System("top")
    top.add_child(g1)
    top.add_child(g2)

    with pytest.raises(ConnectorError, match="Only ports belonging to direct children of '.*' can be connected."):
        g1.connect(s1.entry, s2.exit)

    # Test connection extension
    group = System("hat")
    group.add_child(System1("s1"))
    group.add_child(System2("s2"))
    group.add_child(System3("s3"))
    group.add_child(System3("s4"))

    # First partial connection
    group.connect(group.s4.entry, group.s3.exit, "b")
    connectors = group.connectors
    assert all(isinstance(c, Connector) for c in connectors.values())
    assert_keys(connectors, "s3_exit_to_s4_entry")
    connector = connectors["s3_exit_to_s4_entry"]
    assert connector.source is group.s3.exit
    assert connector.sink is group.s4.entry
    assert connector._mapping == {"b": "b"}
    assert connector._unit_conversions == {"b": (1, 0)}
    # New connection extending the existing connector
    # Should not create any new connector
    group.connect(group.s4.entry, group.s3.exit, "a")
    connectors = group.connectors
    assert all(isinstance(c, Connector) for c in connectors.values())
    assert_keys(connectors, "s3_exit_to_s4_entry")
    connector = connectors["s3_exit_to_s4_entry"]
    assert connector.source is group.s3.exit
    assert connector.sink is group.s4.entry
    assert connector._mapping == {"a": "a", "b": "b"}
    assert connector._unit_conversions == {"a": (1, 0), "b": (1, 0)}


def test_System_add_property():
    class SystemWithProperty(System):
        def setup(self, foo=None):
            if foo is not None:
                self.add_property('foo', foo)

    a = SystemWithProperty("a", foo=0.123)
    b = SystemWithProperty("b")

    assert a.foo == 0.123

    with pytest.raises(AttributeError, match="has no attribute 'foo'"):
        b.foo

    with pytest.raises(AttributeError, match="can't set attribute"):
        a.foo = 3.14

    with pytest.raises(AttributeError, match="`add_property` cannot be called outside `setup`"):
        b.add_property("foo", 3.14)

    assert a.properties == {'foo': 0.123}
    assert b.properties == {}

    class SystemWithSize(System):
        def setup(self, x=[]):
            self.add_property('nx', len(x))
            self.add_property('dtype', type(x[0]) if len(x) > 0 else None)
            for i, value in enumerate(x):
                self.add_inward(f'x{i}', float(value))

    c = SystemWithSize("c", x=[0.1, 0.2])
    assert len(c.inwards) == 2
    for attr in ('nx', 'dtype', 'x0', 'x1'):
        assert attr in c
    assert c.nx == 2
    assert c.dtype is float
    assert c.x0 == 0.1
    assert c.x1 == 0.2

    c = SystemWithSize("c", x=[1, 2, 3])
    assert len(c.inwards) == 3
    for attr in ('nx', 'dtype', 'x0', 'x1', 'x2'):
        assert attr in c
    assert c.nx == 3
    assert c.dtype is int
    assert c.x0 == 1.0
    assert c.x1 == 2.0
    assert c.x2 == 3.0


@pytest.mark.parametrize("ctor_data, expected", [
    (dict(), dict()),
    (
        dict(
            inwards = [get_args("x", 22.0), get_args("y", 42.0)],
            properties = get_args("n", 12),
        ),
        dict(properties = {"n": 12})
    ),
    (
        dict(
            inwards = [get_args("x", 22.0), get_args("y", 42.0)],
            properties = [get_args("n", 12), get_args("Z", 3.2)],
        ),
        dict(properties = {"n": 12, "Z": 3.2})
    ),
    (
        OrderedDict(
            inwards = [get_args("x", 22.0), get_args("y", 42.0)],
            properties = get_args("x", 12),
        ),
        dict(error=ValueError, match="cannot add read-only property 'x'")
    ),
    (
        OrderedDict(
            inwards = [get_args("x", 22.0), get_args("y", 42.0)],
            properties = [get_args("N", 12), get_args("N", 2)],
        ),
        dict(error=ValueError, match="cannot add read-only property 'N'")
    ),
    (
        OrderedDict(
            properties = get_args("x", 12),
            inwards = [get_args("x", 22.0), get_args("y", 42.0)],
        ),
        dict(error=ValueError, match="cannot add inward 'x'")
    ),
])
def test_System_properties(DummyFactory, ctor_data, expected):
    error = expected.get('error', None)

    if error is None:
        system = DummyFactory("dummy", **ctor_data)  # test object
        assert system.properties == expected.get('properties', {})

    else:
        with pytest.raises(error, match=expected.get('match', None)):
            DummyFactory("dummy", **ctor_data)


def test_System_properties_safeview(DummyFactory):
    dummy = DummyFactory("dummy", 
        inwards = [get_args("x", 22.0), get_args("y", 42.0, desc="that's why")],
        properties = [get_args("Z", 3.2), get_args("n", 12)],
    )
    assert dummy.properties == {"n": 12, "Z": 3.2}

    props = dummy.properties
    assert props == {"n": 12, "Z": 3.2}

    with pytest.raises(TypeError, match="object does not support item assignment"):
        props['n'] = 4

    with pytest.raises(TypeError, match="object does not support item assignment"):
        props['led'] = 'zep'

    props = dummy.properties.copy()
    props['led'] = 'zep'
    assert props == {"n": 12, "Z": 3.2, "led": "zep"}
    assert dummy.properties == {"n": 12, "Z": 3.2}


@pytest.mark.parametrize("format", LogFormat)
@pytest.mark.parametrize("msg, kwargs, to_log, emitted", [
    ("zombie call_setup_run", dict(), False, None),
    ("useless start call_clean_run", dict(activate=True), False, None),
    (
        f"{System.CONTEXT_EXIT_MESSAGE} call_clean_run",
        dict(activate=False),
        False,
        dict(levelno=LogLevel.DEBUG, pattern=r"Compute calls for [\w\.]+: \d+")
    ),
    (
        "other message with activation",
        dict(activate=True),
        False,
        dict(levelno=LogLevel.FULL_DEBUG, pattern=r"#### \w+ - \w+ - Inputs")
    ),
    (
        "second message with deactivation",
        dict(activate=False),
        False, 
        dict(levelno=LogLevel.FULL_DEBUG, pattern=r"#### \w+ - \w+ - Outputs")
    ),
    ("common message", dict(), True, None),
])
def test_System_log_debug_message(format, msg, kwargs, to_log, emitted):
    handler = mock.MagicMock(level=LogLevel.DEBUG, log=mock.MagicMock())
    rec = logging.getLogRecordFactory()("log_test", LogLevel.INFO, __file__, 22, msg, (), None)
    for key, value in kwargs.items():
        setattr(rec, key, value)
    
    s = System("dummy")

    assert s.log_debug_message(handler, rec, format) == to_log

    if emitted:
        handler.log.assert_called_once()
        args = handler.log.call_args[0]
        assert args[0] == emitted["levelno"]
        assert re.match(emitted["pattern"], args[1]) is not None
    else:
        handler.log.assert_not_called()

