import pytest
from unittest import mock

import logging
import re
from io import StringIO
from collections import OrderedDict

import numpy as np
import math

from cosapp.utils.testing import assert_keys, get_args, no_exception
from cosapp.utils.logging import LogFormat, LogLevel
from cosapp.core.signal import Slot
from cosapp.core.connectors import BaseConnector, Connector, ConnectorError
from cosapp.core.numerics.basics import MathematicalProblem
from cosapp.core.numerics.boundary import Unknown
from cosapp.core.numerics.residues import Residue
from cosapp.ports.port import BasePort, Port, PortType, Scope, Validity
from cosapp.ports.units import UnitError
from cosapp.drivers import Driver, RunOnce, NonLinearSolver
from cosapp.systems import system as system_module
from cosapp.systems import System
from cosapp.systems.system import VariableReference

from cosapp.tests.library.systems import Multiply1
from cosapp.tests.library.systems.vectors import Strait1dLine


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


class EntryExit(System):
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
    BasePort.set_type_checking(False)
    m.p_in.x = "a"
    m.K1 = 2
    BasePort.set_type_checking(True)
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
    BasePort.set_type_checking(False)
    m.p_in.x = "a"  # Set bad value on purpose
    BasePort.set_type_checking(True)
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
    assert set(s.inputs) == {System.INWARDS, System.MODEVARS_IN}
    assert set(s.outputs) == {System.OUTWARDS, System.MODEVARS_OUT}
    assert set(s.name2variable) == {
        System.INWARDS,
        System.OUTWARDS,
        System.MODEVARS_IN,
        System.MODEVARS_OUT,
    }
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
    ("inwards", ValueError),
    ("outwards", ValueError),
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


def test_System___contains__(DummyFactory):
    top: System = DummyFactory("top",
        inwards = get_args('x', 1.0),
        outwards = get_args('y', 0.0),
        properties = get_args('const', 0.123),
        events = get_args('boom', trigger="y > x"),
    )
    sub: System = DummyFactory("sub",
        inputs = get_args(DummyPort, 'p_in'),
        outputs = get_args(DummyPort, 'p_out'),
        inward_modevars = get_args('m_in', True),
        outward_modevars = get_args('m_out', init=0, dtype=int),
    )
    top.add_child(sub, pulling=['p_in', 'p_out', 'm_out'])

    assert "x" in top
    assert "y" in top
    assert "sub" in top
    assert "p_in" in top
    assert "p_out" in top
    assert "p_in" in top.sub
    assert "p_out" in top.sub
    assert "p_in.a" in top
    assert "p_in.b" in top
    assert "sub.p_in" in top
    assert "sub.p_out" in top
    assert "sub.p_in.a" in top
    assert "sub.p_out.a" in top
    assert "p_out.a" in top
    assert "p_out.b" in top
    assert "const" in top
    assert "m_out" in top
    assert "m_in" not in top
    assert "sub.m_in" in top
    assert "m_in" in top.sub
    assert "boom" in top

    assert "parent" not in top
    assert "inputs" not in top
    assert "outputs" not in top
    assert "children" not in top
    assert "name2variable" not in top


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
    group.add_child(EntryExit("s3"))
    group.add_child(EntryExit("s4"))

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
    s = System("s")
    s2 = SubSystem("sub")
    res = s.add_child(s2)

    assert res is s2
    assert_keys(s.children, "sub")
    assert s.children['sub'] is s2
    assert s2.parent is s
    assert list(s.exec_order) == [s2.name]

    to_check = {
        "sub": s2,
        "sub.in_": s2.inputs["in_"],
        "sub.out": s2.outputs["out"],
    }

    for key, obj in to_check.items():
        assert key in s.name2variable
        reference = s.name2variable[key]
        assert reference.value is obj, f"key = {key}"

    s3 = SubSystem("sub2")
    assert s.add_child(s3, execution_index=0) is s3
    assert_keys(s.children, "sub", "sub2")
    assert list(s.exec_order) == [s3.name, s2.name]
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
        assert reference.value is obj, f"key = {key}"


def test_System_add_child_pulling(caplog):
    caplog.set_level(logging.DEBUG)

    s = System("s")
    s2 = s.add_child(SubSystem("sub"), pulling={"in_": "entry", "out": "out"})

    records = list(filter(lambda record: record.levelno == logging.DEBUG, caplog.records))
    assert len(records) == 2
    pattern = r"Port s\.\w+ has been duplicated from s\.sub\.\w+ - including validation range and scope."
    for record in records:
        assert re.match(pattern, record.message)

    assert_keys(s.inputs, 'inwards', 'modevars_in', 'entry')
    entry = s.inputs['entry']
    assert entry is s.inputs['entry']  # check that assignment did not create a copy!
    assert entry is not s2.inputs["in_"]
    assert isinstance(entry, type(s2.inputs["in_"]))
    assert entry.direction is s2.inputs["in_"].direction
    assert set(s.connectors) == {
        "entry -> sub.in_",
        "sub.out -> out",
    }
    assert entry is s.connectors["entry -> sub.in_"].source

    assert_keys(s.outputs, 'outwards', 'modevars_out', 'out')
    s_out = s.outputs['out']
    assert s_out is s.outputs['out']  # check that assignment did not create a copy!
    assert s_out is not s2.outputs['out']
    assert isinstance(s_out, type(s2.outputs['out']))
    assert s_out.direction is s2.outputs['out'].direction
    assert s2.outputs["out"] is s.connectors["sub.out -> out"].source

    # Use only str
    s = System("s")
    s2 = s.add_child(SubSystem("sub"), pulling="in_")
    assert_keys(s.inputs, 'in_', 'inwards', 'modevars_in')
    s_in = s.inputs['in_']
    assert s_in is s.inputs['in_']
    assert s_in is not s2.inputs['in_']
    assert isinstance(s_in, type(s2.inputs['in_']))
    assert s_in.direction is s2.inputs['in_'].direction
    assert s_in is s.connectors["in_ -> sub.in_"].source

    # Use list of str
    s = System("s")
    s2 = s.add_child(SubSystem("sub"), pulling=["in_", "out"])
    assert_keys(s.inputs, 'in_', 'inwards', 'modevars_in')
    # Check pulled symbol 'in_'
    s_in = s.inputs['in_']
    assert s_in is s.inputs['in_']
    assert s_in is not s2.inputs['in_']
    assert isinstance(s_in, type(s2.inputs['in_']))
    assert s_in.direction is s2.inputs['in_'].direction
    assert s_in is s.connectors["in_ -> sub.in_"].source
    # Check pulled symbol 'out'
    assert_keys(s.outputs, 'outwards', 'modevars_out', 'out')
    s_out = s.outputs['out']
    assert s_out is s.outputs['out']  # check that assignment did not create a copy!
    assert s_out is not s2.outputs['out']
    assert isinstance(s_out, type(s2.outputs['out']))
    assert s_out.direction is s2.outputs['out'].direction
    assert s2.outputs["out"] is s.connectors["sub.out -> out"].source

    # Pulling from 2 children IN to same IN
    s = System("s")
    s.add_child(SubSystem("sub_a"), pulling={"in_": "entry"})
    s.add_child(SubSystem("sub_b"), pulling={"in_": "entry"})

    with pytest.raises(KeyError):
        s.add_child(SubSystem("sub_c"), pulling=["here"])

    assert_keys(s.inputs, 'entry', 'inwards', 'modevars_in')
    s_in = s.inputs['entry']
    assert s_in is s.inputs['entry']
    assert s_in is not s2.inputs['in_']
    assert isinstance(s_in, type(s2.inputs['in_']))
    assert s_in.direction is s2.inputs['in_'].direction
    assert s_in is s.connectors["entry -> sub_a.in_"].source
    assert s_in is s.connectors["entry -> sub_b.in_"].source

    # Pulling from 2 children OUT to same OUT
    s = System("s")
    s.add_child(SubSystem("sub_a"), pulling={"out": "out"})
    with pytest.raises(ConnectorError):
        s.add_child(SubSystem("sub_b"), pulling={"out": "out"})
    assert "sub_b" not in s.children
    with pytest.raises(ConnectorError):
        s.add_child(SubSystem("sub_b"), pulling={"in_": "out"})
    assert "sub_b" not in s.children

    # Pulling from 1 child IN and 1 child OUT to same IN
    s = System("s")
    s.add_child(SubSystem("sub_a"), pulling={"in_": "entry"})
    with pytest.raises(ConnectorError):
        s.add_child(SubSystem("sub_b"), pulling={"out": "entry"})
    assert "sub_b" not in s.children

    # Pulling inwards
    caplog.clear()
    s = System("s")
    s2a = s.add_child(SubSystem("sub_a"), pulling="sloss")

    records = list(filter(lambda record: record.levelno == logging.DEBUG, caplog.records))
    assert len(records) == 1
    assert re.match(
        r"s\.\w+ has been duplicated from s\.\w+\.\w+",
        records[-1].message,
    )
    assert s.inwards['sloss'] == s2a['sloss']
    source = s2a.inwards.get_details("sloss")
    pulled = s.inwards.get_details("sloss")
    attribute_names = [
        "unit", "dtype", "description", "scope",
        "valid_range", "invalid_comment",
        "limits", "out_of_limits_comment",
    ]
    for attr in attribute_names:
        assert getattr(pulled, attr) == getattr(source, attr)

    s = System("s")
    s2a = s.add_child(SubSystem("sub_a"), pulling=["sloss", "tmp"])
    assert s.inwards['sloss'] == s2a['sloss']
    assert s.outwards['tmp'] == s2a['tmp']

    s = System("s")
    s2a = s.add_child(SubSystem("sub_a"), pulling={"sloss": "a_sloss"})
    assert s.inwards['a_sloss'] == s2a['sloss']

    # Pulling all inwards
    s = System("s")
    s2a = s.add_child(SubSystem("sub_a"), pulling="inwards")
    assert s.inwards['sloss'] == s2a['sloss']

    # Pulling outwards
    caplog.clear()
    s = System("s")
    s2a = s.add_child(SubSystem("sub_a"), pulling="tmp")

    records = list(filter(lambda record: record.levelno == logging.DEBUG, caplog.records))
    assert len(records) == 1
    assert re.match(
        r"s\.\w+ has been duplicated from s.\w+\.\w+", 
        records[-1].message,
    )
    assert s.outwards['tmp'] == s2a['tmp']
    source = s2a.outwards.get_details("tmp")
    pulled = s.outwards.get_details("tmp")
    for attr in attribute_names:
        assert getattr(pulled, attr) == getattr(source, attr)

    s = System("s")
    s2a = s.add_child(SubSystem("sub_a"), pulling={"tmp": "a_tmp"})
    assert s.outwards["a_tmp"] == s2a["tmp"]

    # Pulling all outwards
    s = System("s")
    s2a = s.add_child(SubSystem("sub_a"), pulling="outwards")
    assert s.outwards["tmp"] == s2a["tmp"]

    # Adding a child component with an already existing name
    s = TopSystem("s")
    with pytest.raises(ValueError):
        s.add_child(SubSystem("top_tmp"))

    # Adding Driver
    s = TopSystem("s")
    with pytest.raises(TypeError):
        s.add_child(Driver("dummy"))


@pytest.mark.parametrize("args", [
    PtWPort("p", PortType.IN),
    (System("sub"), "first"),
])
def test_System_add_child_TypeError(args):
    s = System("s")
    with pytest.raises(TypeError):
        s.add_child(*args)


def test_System_pop_child():
    s = System("s")
    s1 = SubSystem("sub1")
    s2 = SubSystem("sub2")
    s.add_child(s1, pulling=["in_", "sloss", "tmp"])
    s.add_child(s2)
    s.connect(s1.out, s2.in_)
    assert_keys(s.children, "sub1", "sub2")
    s.exec_order = ['sub2', 'sub1']

    assert set(s.connectors) == {
        "in_ -> sub1.in_",
        "inwards -> sub1.inwards",
        "sub1.outwards -> outwards",
        "sub1.out -> sub2.in_",
    }

    s.pop_child("sub1")
    assert_keys(s.children, "sub2")
    assert s1.parent is None
    assert s1.name not in s.exec_order
    assert list(s.exec_order) == [s2.name]
    assert len(s.connectors) == 0
    assert not any(key.startswith('sub1') for key in s.name2variable)


def test_System_add_port():
    s = System("s")

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

    s = System("s")
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
    s: System = DummyFactory("test", inwards=get_args("K", 2.0))
    assert "K" in s
    assert f"{System.INWARDS}.K" in s
    assert s.K == 2.0

    with pytest.raises(AttributeError):
        s.add_inward("K", 2.0)

    # Add multiple inwards
    s: System = DummyFactory("test", inwards=get_args(
        {
            "K": 2.0,
            "switch": True,
            "r": {"value": 1, "scope": Scope.PUBLIC},
            "q": {"a": 1, "b": 2},
        }
        ))

    for name in ["K", "switch", "r", "q"]:
        assert name in s
        assert f"{System.INWARDS}.{name}" in s
    assert s.K == 2.0
    assert s.switch == True
    assert s.r == 1
    assert s[System.INWARDS].get_details("r").scope == Scope.PUBLIC
    assert s.q == {"a": 1, "b": 2}

    # Test variables attributes
    s: System = DummyFactory("test", inwards=get_args(
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
    s: System = DummyFactory("dummy", outwards=get_args("r", 42.0))
    assert "r" in s
    assert f"{System.OUTWARDS}.r" in s
    assert s.r == 42

    # Add multiple outwards
    s: System = DummyFactory("dummy", outwards=get_args(
        {
            "r": 42.0,
            "q": 12,
            "s": {"value": 1, "scope": Scope.PUBLIC},
            "x": {"a": 1, "b": 2},
        }
        ))

    for name in ["r", "q", "s", "x"]:
        assert name in s
        assert f"{System.OUTWARDS}.{name}" in s
    assert s.r == 42.0
    assert s.q == 12
    assert s.s == 1
    assert s[System.OUTWARDS].get_details("s").scope == Scope.PUBLIC
    assert s.x == {"a": 1, "b": 2}

    # Add multiple outwards with attributes
    s: System = DummyFactory("dummy", outwards=get_args(
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
    s: System = DummyFactory("dummy", outwards=get_args(
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
    s = System("s")
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
    s = System("s")

    d = {"a": 1, "b": "hello world"}

    s.append_name2variable(
        [(key, VariableReference(context=s, mapping=d, key=key)) for key in d]
    )
    for key in d:
        reference = s.name2variable[key]
        assert reference.value is d[key]

    s2 = SubSystem("sub")
    s.add_child(s2)

    s2.append_name2variable(
        [(key, VariableReference(context=s2, mapping=d, key=key)) for key in d]
    )
    for key in d:
        abs_key = f"{s2.name}.{key}"
        reference = s.name2variable[abs_key]
        assert reference.value is d[key]


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
        abs_name = f"{s2.name}.{key}"
        assert abs_name in s.name2variable

    s2.pop_name2variable(keys)
    for key in keys:
        abs_key = f"{s2.name}.{key}"
        assert abs_key not in s.name2variable


def test_System_loops_1():
    """Test mathematical problem created by `open_loops`,
    and check that `close_loops` restores the initial configuration.

    Case: system with 2 sub-systems, each with ports and orphan vars.
    """
    class XvPort(Port):
        def setup(self):
            self.add_variable('x', 1.0)
            self.add_variable('v', np.ones(2))

    class SomeSystem(System):
        def setup(self):
            self.add_inward("a_in")
            self.add_inward("b_in")
            self.add_input(XvPort, "entry")
            self.add_output(XvPort, "exit")
            self.add_outward("a_out")
            self.add_outward("b_out")

        def compute(self):
            self.exit.x = self.entry.x * self.a_in + self.b_in
            self.a_out = self.entry.x * self.a_in
            self.b_out = self.b_in / self.a_in

    def make_case():
        s = System("top")
        a = s.add_child(SomeSystem("a"))
        b = s.add_child(SomeSystem("b"))
        return s, a, b

    # Case 1
    s, a, b = make_case()
    s.connect(b.exit, a.entry)
    s.connect(a.exit, b.entry)
    # Test initial config
    connectors = s.connectors
    assert set(connectors) == {
        'a.exit -> b.entry',
        'b.exit -> a.entry',
    }
    # Sanity check between `all_connectors()` and `connectors.values()`
    assert list(s.all_connectors()) == list(connectors.values())
    assert all(connector.is_active for connector in s.all_connectors())
    assert s.get_unsolved_problem().shape == (0, 0)

    s.open_loops()
    problem = s.get_unsolved_problem()
    assert problem.shape == (3, 3)
    assert set(problem.unknowns) == {
        'a.entry.x',
        'a.entry.v',
    }
    assert set(problem.residues) == {
        'a.entry.x == b.exit.x',
        'a.entry.v == b.exit.v',
    }
    connectors = s.connectors
    assert set(connectors) == {
        'a.exit -> b.entry',
        'b.exit -> a.entry',
    }
    assert connectors["a.exit -> b.entry"].is_active
    assert not connectors["b.exit -> a.entry"].is_active
    # Check that `close_loops` restores all connections
    s.close_loops()
    assert connectors["a.exit -> b.entry"].is_active
    assert connectors["b.exit -> a.entry"].is_active
    assert all(connector.is_active for connector in s.all_connectors())
    assert s.get_unsolved_problem().shape == (0, 0)

    # Case 2 - same as #1 with different exec order
    s, a, b = make_case()
    s.connect(b.exit, a.entry)
    s.connect(a.exit, b.entry)
    s.exec_order = ['b', 'a']

    problem = s.get_unsolved_problem()
    assert problem.shape == (0, 0)

    s.open_loops()
    problem = s.get_unsolved_problem()
    assert problem.shape == (3, 3)
    assert set(problem.unknowns) == {
        'b.entry.x',
        'b.entry.v',
    }
    assert set(problem.residues) == {
        'b.entry.x == a.exit.x',
        'b.entry.v == a.exit.v',
    }
    connectors = s.connectors
    assert set(connectors) == {
        'a.exit -> b.entry',
        'b.exit -> a.entry',
    }
    assert not connectors["a.exit -> b.entry"].is_active
    assert connectors["b.exit -> a.entry"].is_active
    # Check that `close_loops` restores all connections
    s.close_loops()
    assert all(connector.is_active for connector in s.all_connectors())
    assert s.get_unsolved_problem().shape == (0, 0)

    # Breaking link between ExtensiblePort (1)
    s, a, b = make_case()
    s.connect(a.inwards, b.outwards, {"a_in": "a_out"})
    s.connect(a.exit, b.entry)

    s.open_loops()
    problem = s.get_unsolved_problem()
    assert problem.shape == (1, 1)
    assert set(problem.unknowns) == {
        'a.a_in',
    }
    assert set(problem.residues) == {
        'a.a_in == b.a_out',
    }
    connectors = s.connectors
    assert set(connectors) == {
        'a.exit -> b.entry',
        'b.outwards -> a.inwards',
    }
    assert connectors["a.exit -> b.entry"].is_active
    assert not connectors["b.outwards -> a.inwards"].is_active
    # Check that `close_loops` restores all connections
    s.close_loops()
    assert all(connector.is_active for connector in s.all_connectors())
    assert s.get_unsolved_problem().shape == (0, 0)

    # Breaking link between ExtensiblePort (2)
    s, a, b = make_case()
    s.connect(a.inwards, b.outwards, {"a_in": "a_out"})
    s.connect(a.entry, b.exit)

    s.open_loops()
    problem = s.get_unsolved_problem()
    assert problem.shape == (4, 4)
    assert set(problem.unknowns) == {
        'a.entry.x',
        'a.entry.v',
        'a.a_in',
    }
    assert set(problem.residues) == {
        'a.entry.x == b.exit.x',
        'a.entry.v == b.exit.v',
        'a.a_in == b.a_out',
    }
    connectors = s.connectors
    assert set(connectors) == {
        'b.exit -> a.entry',
        'b.outwards -> a.inwards',
    }
    assert not connectors["b.outwards -> a.inwards"].is_active
    assert not connectors["b.exit -> a.entry"].is_active
    # Check that `close_loops` restores all connections
    s.close_loops()
    assert all(connector.is_active for connector in s.all_connectors())
    assert s.get_unsolved_problem().shape == (0, 0)


def test_System_loops_2():
    """Test mathematical problem created by `open_loops`,
    and check that `close_loops` restores the initial configuration.

    Case: system with 3 simple sub-systems.
    """
    # Test system depending on two others not already executed
    class Surface(System):
        """Interface for z = f(x, y)"""
        def setup(self):
            self.add_inward("x", 1.0)
            self.add_inward("y", 0.5)
            self.add_outward("z", 0.0)

    def make_case():
        top = System("top")
        top.add_child(Surface("s1"))
        top.add_child(Surface("s2"))
        top.add_child(Surface("s3"))
        return top

    # Test system depending on two others not already executed
    top = make_case()
    assert list(top.exec_order) == ['s1', 's2', 's3']
    top.connect(top.s2.outwards, top.s1.inwards, {"z": "x"})
    top.connect(top.s3.outwards, top.s1.inwards, {"z": "y"})

    top.open_loops()
    problem = top.get_unsolved_problem()
    assert problem.shape == (2, 2)
    assert set(problem.unknowns) == {
        's1.x',
        's1.y',
    }
    assert set(problem.residues) == {
        's1.x == s2.z',
        's1.y == s3.z',
    }
    connectors = top.connectors
    assert set(connectors) == {
        's2.outwards -> s1.inwards',
        's3.outwards -> s1.inwards',
    }
    assert not connectors["s2.outwards -> s1.inwards"].is_active
    assert not connectors["s2.outwards -> s1.inwards"].is_active
    # Check that `close_loops` restores all connections
    top.close_loops()
    assert all(connector.is_active for connector in top.all_connectors())
    assert top.get_unsolved_problem().shape == (0, 0)

    # Backward dependencies: information flow opposite to exec order
    # s1 <-- s2 <-- s3
    top = make_case()
    top.connect(top.s2.outwards, top.s1.inwards, {"z": "x"})
    top.connect(top.s3.outwards, top.s2.inwards, {"z": "y"})
    top.open_loops()
    problem = top.get_unsolved_problem()
    assert problem.shape == (2, 2)
    assert set(problem.unknowns) == {
        's1.x',
        's2.y',
    }
    assert set(problem.residues) == {
        's1.x == s2.z',
        's2.y == s3.z',
    }
    connectors = top.connectors
    assert set(connectors) == {
        's2.outwards -> s1.inwards',
        's3.outwards -> s2.inwards',
    }
    assert not connectors["s2.outwards -> s1.inwards"].is_active
    assert not connectors["s3.outwards -> s2.inwards"].is_active
    # Check that `close_loops` restores all connections
    top.close_loops()
    assert all(connector.is_active for connector in top.all_connectors())
    assert top.get_unsolved_problem().shape == (0, 0)

    # Same as previous, with s1 --> s3 connector
    top = make_case()
    top.connect(top.s1.outwards, top.s3.inwards, {"z": "x"})
    top.connect(top.s2.outwards, top.s1.inwards, {"z": "x"})
    top.connect(top.s3.outwards, top.s2.inwards, {"z": "y"})
    top.open_loops()
    problem = top.get_unsolved_problem()
    assert problem.shape == (2, 2)
    assert set(problem.unknowns) == {
        's1.x',
        's2.y',
    }
    assert set(problem.residues) == {
        's1.x == s2.z',
        's2.y == s3.z',
    }
    connectors = top.connectors
    assert set(connectors) == {
        's1.outwards -> s3.inwards',
        's2.outwards -> s1.inwards',
        's3.outwards -> s2.inwards',
    }
    assert connectors["s1.outwards -> s3.inwards"].is_active
    assert not connectors["s2.outwards -> s1.inwards"].is_active
    assert not connectors["s3.outwards -> s2.inwards"].is_active
    # Check that `close_loops` restores all connections
    top.close_loops()
    assert all(connector.is_active for connector in top.all_connectors())
    assert top.get_unsolved_problem().shape == (0, 0)


def test_System_loops_control_unknowns():
    """Test mathematical problem created by `open_loops`,
    with control over loop unknowns.
    """
    class A(System):
        def setup(self):
            self.add_inward('x', 2.0)
            self.add_outward('y', 0.0)

        def compute(self):
            self.y = self.x**2
    
    class B(System):
        def setup(self):
            self.add_inward('u', 3.0)
            self.add_outward('v', 0.0)

        def compute(self):
            self.v = self.u
    
    class Assembly(System):
        def setup(self):
            a = self.add_child(A('a'))
            b = self.add_child(B('b'))

            # Set inter-dependency between `a` and `b`
            # Solution is a.x = 0 or 1
            self.connect(a, b, {'y': 'u', 'x': 'v'})
            # Declare connected inputs as unknowns
            a.add_unknown('x', max_rel_step=0.5)  # in case `a.x` is ever used as an unknown
            b.add_unknown('u', max_abs_step=0.1)  # in case `b.u` is ever used as an unknown

    s = Assembly('s')
    # Check that assembled problem is empty, since
    # `s.a.x` and `s.b.u` are both connected to outputs
    assert s.get_unsolved_problem().shape == (0, 0)
    assert s.a.get_unsolved_problem().n_unknowns == 1
    assert s.b.get_unsolved_problem().n_unknowns == 1

    s.exec_order = ('a', 'b')
    s.open_loops()
    problem = s.get_unsolved_problem()
    assert problem.shape == (1, 1)
    assert set(problem.unknowns) == {'a.x'}
    unknown = problem.unknowns['a.x']
    assert unknown.max_abs_step == np.inf
    assert unknown.max_rel_step == 0.5
    s.close_loops()
    assert s.get_unsolved_problem().shape == (0, 0)
    # Solve problem
    s.add_driver(NonLinearSolver('solver', tol=1e-7))
    s.a.x = 10
    s.run_drivers()
    assert s.a.x == pytest.approx(1, abs=1e-6)

    s = Assembly('s')
    s.exec_order = ('b', 'a')
    s.open_loops()
    problem = s.get_unsolved_problem()
    assert problem.shape == (1, 1)
    assert set(problem.unknowns) == {'b.u'}
    unknown = problem.unknowns['b.u']
    assert unknown.max_abs_step == 0.1
    assert unknown.max_rel_step == np.inf
    s.close_loops()
    assert s.get_unsolved_problem().shape == (0, 0)
    # Solve problem
    s.add_driver(NonLinearSolver('solver', tol=1e-7))
    s.b.u = 3
    s.run_drivers()
    assert s.a.x == pytest.approx(1, abs=1e-6)


def test_System_loop_residue_reference():
    """Test normalization factor of residues created by `open_loops`.

    Detail on test case:
    -------------------
    At equilibrium, the two ends of the connector involved in the loop
    are nil (with solver accuracy `tol`). We test that if the connector is
    re-opened after convergence, the associated residue is not renormalized
    (otherwise, the normalization factor would be of the order of 1/tol).
    """
    class A(System):
        def setup(self):
            self.add_inward('x', 2.0)
            self.add_outward('y', 0.0)

        def compute(self):
            self.y = math.exp(self.x)
    
    class B(System):
        def setup(self):
            self.add_inward('x', 3.0)
            self.add_outward('y', 0.0)

        def compute(self):
            self.y = 1 - self.x
    
    class Assembly(System):
        def setup(self):
            a = self.add_child(A('a'))
            b = self.add_child(B('b'))

            # Set inter-dependency between `a` and `b`
            # Solution is a.x = 0 (or b.x = 1)
            self.connect(a, b, {'y': 'x', 'x': 'y'})

    s = Assembly('s')
    # Check that assembled problem is empty, since
    # `s.a.x` and `s.b.x` are both connected to outputs
    assert s.get_unsolved_problem().shape == (0, 0)

    s.exec_order = ('a', 'b')

    # Open loops *before* problem equilibration
    s.open_loops()
    problem = s.get_unsolved_problem()
    assert problem.shape == (1, 1)
    assert set(problem.unknowns) == {'a.x'}
    assert set(problem.residues) == {'a.x == b.y'}
    key = 'a.x == b.y'
    residue = problem.residues[key]
    assert residue.reference == 1.0, "Loop residue should not be normalized"
    s.close_loops()

    # Solve problem
    solver = s.add_driver(
        NonLinearSolver('solver', tol=1e-15, history=True)
    )
    s.a.x = 5.0
    s.run_drivers()
    # Check that both ends of loop connector are nil
    assert s.a.x == pytest.approx(0, abs=1e-15)
    assert s.b.y == pytest.approx(0, abs=1e-15)
    assert set(solver.problem.residues) == {key}
    residue = solver.problem.residues[key]
    assert residue.reference == 1.0, "Loop residue should not be normalized"
    assert len(solver.results.trace) > 1

    # Open loops *after* problem equilibration
    s.open_loops()
    problem = s.get_unsolved_problem()
    assert problem.shape == (1, 1)
    assert set(problem.unknowns) == {'a.x'}
    assert set(problem.residues) == {key}
    residue = problem.residues[key]
    assert residue.reference == 1.0, "Loop residue should not be normalized"
    s.close_loops()

    # Re-run solver
    s.run_drivers()
    # Check that solver did not iterate
    assert len(solver.results.trace) == 1


def test_System_is_input_var(DummyFactory):
    s: System = DummyFactory("dummy",
        inputs = get_args(PtWPort, 'flow_in'),
        outputs = get_args(PtWPort, 'flow_out'),
        inwards = [get_args('x', 1.0), get_args('y', np.zeros(4))],
        outwards = get_args("z", 42.0),
    )
    # Test `System.is_input_var`
    assert s.is_input_var('x')
    assert s.is_input_var('y')
    assert not s.is_input_var('z')

    assert s.is_input_var('flow_in.Pt')
    assert s.is_input_var('flow_in.W')
    assert not s.is_input_var('flow_in')  # does not apply to ports

    assert not s.is_input_var('flow_out.Pt')
    assert not s.is_input_var('flow_out.W')
    assert not s.is_input_var('flow_out')

    # Test `System.is_output_var`
    assert not s.is_output_var('x')
    assert not s.is_output_var('y')
    assert s.is_output_var('z')

    assert not s.is_output_var('flow_in.Pt')
    assert not s.is_output_var('flow_in.W')
    assert not s.is_output_var('flow_in')

    assert s.is_output_var('flow_out.Pt')
    assert s.is_output_var('flow_out.W')
    assert not s.is_output_var('flow_out')


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
    top.connect(top.source, top.sink, "loop")
    top.exec_order = ['sink', 'source']

    top.open_loops()
    problem = top.get_unsolved_problem()
    assert problem.shape == (1, 1)
    assert set(problem.unknowns) == {
        'sink.loop',
    }
    assert set(problem.residues) == {
        'sink.loop == source.loop',
    }
    connectors = top.connectors
    assert set(connectors) == {
        "inwards -> sink.inwards",  # pulling
        "source.outwards -> sink.inwards",
    }
    assert connectors["inwards -> sink.inwards"].is_active
    assert not connectors["source.outwards -> sink.inwards"].is_active
    top.close_loops()
    assert all(connector.is_active for connector in top.all_connectors())
    assert top.get_unsolved_problem().shape == (0, 0)


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
    m: System = DummyFactory("dummy", base=Multiply1,
        unknowns=get_args("K1", max_rel_step=0.01, lower_bound=-10.0),
    )
    unknown = m.get_unsolved_problem().unknowns["K1"]
    assert isinstance(unknown, Unknown)
    assert unknown.name == "K1"
    assert unknown.port is m.inwards
    assert unknown.max_rel_step == 0.01
    assert unknown.max_abs_step == np.inf
    assert unknown.lower_bound == -10
    assert unknown.upper_bound == np.inf
    assert unknown.mask is None

    m: System = DummyFactory("dummy", base=Multiply1,
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
    assert unknown.port is m.p_in
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
    v: System = DummyFactory("v", base=Strait1dLine, unknowns=get_args("a"))
    
    unknown = v.get_unsolved_problem().unknowns["a"]
    assert np.array_equal(unknown.mask, [True, True, True])

    v: System = DummyFactory("v", base=Strait1dLine, unknowns=get_args("a[::2]"))
    
    problem = v.get_unsolved_problem()
    assert set(problem.unknowns) == {"a[::2]"}
    unknown = problem.unknowns["a[::2]"]
    assert np.array_equal(unknown.mask, [True, False, True])

    with pytest.raises(IndexError):
        DummyFactory("dummy", base=Strait1dLine, unknowns=get_args("a[[1, 3]]"))


def test_System_add_equation(DummyFactory):
    class ASyst(System):
        def setup(self):
            self.add_inward("x", 1.0)
            m = self.add_equation("x == 0", name="cancel_x")
            self.add_property("math_problem", m)

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

    s: System = DummyFactory("dummy",
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


@pytest.mark.parametrize("args_kwargs, expected_name", [
    (get_args(), 'problem'),  # no args - default
    (get_args('foo'), 'foo'),
])
def test_System_new_problem(args_kwargs, expected_name):
    args, kwargs = args_kwargs
    s = System('s')
    p = s.new_problem(*args, **kwargs)
    assert isinstance(p, MathematicalProblem)
    assert p.context is s
    assert p.name == expected_name
    assert p.shape == (0, 0)


def test_System_add_design_method():
    class ASyst(System):
        def setup(self):
            m = self.add_design_method("method1")
            self.add_property("design_method", m)

    a = ASyst("a")
    with pytest.raises(AttributeError, match="`add_design_method` cannot be called outside `setup`"):
        a.add_design_method("methodX")
    assert isinstance(a.design("method1"), MathematicalProblem)
    assert a.design("method1") is a.design_method


def test_System_design(DummyFactory):
    a: System = DummyFactory("a", design_methods=[get_args("method1"), get_args("method2")])

    assert set(a.design_methods) == {'method1', 'method2'}
    for name, design_method in a.design_methods.items():
        assert isinstance(design_method, MathematicalProblem)
        assert a.design(name) is design_method

    with pytest.raises(KeyError):
        a.design("method3")


def test_System_add_target(DummyFactory):
    s: System = DummyFactory('s',
        inwards = [get_args('x', 1.0), get_args('y', 0.0)],
        outwards = get_args('z', 0.0),
        targets = get_args('z'),
    )
    s.z = 1.5

    offdesign = s.get_unsolved_problem()
    assert offdesign.shape == (0, 1)
    assert len(offdesign.residues) == 0
    assert len(offdesign.deferred_residues) == 1

    with pytest.raises(AttributeError, match="`add_target` cannot be called outside `setup`"):
        s.add_target('x')


def test_System_add_target_error(DummyFactory):
    """Checks that declaring a target on a non-existing variable raises `NameError`"""
    with pytest.raises(NameError, match="'foo' is not defined"):
        DummyFactory('oops',
            outwards = get_args('y', 0.0),
            inwards = get_args('x', 1.0),
            targets = get_args('foo'),
        )


def test_System_add_target_offdesign():
    """Use of `add_target` in off-design mode"""
    class SystemWithTarget(System):
        def setup(self):
            self.add_inward('x', 1.0)
            self.add_inward('y', 1.0)
            self.add_outward('z', 1.0)

            self.add_unknown('y').add_target('z')

        def compute(self):
            self.z = self.x * self.y**2

    s = SystemWithTarget('s')

    offdesign = s.get_unsolved_problem()
    assert offdesign.shape == (1, 1)
    assert len(offdesign.residues) == 0
    assert len(offdesign.deferred_residues) == 1

    solver = s.add_driver(NonLinearSolver('solver', tol=1e-9))

    s.x = 0.5
    s.y = 0.0
    s.z = 2.0  # dynamically set target
    s.run_drivers()
    assert s.z == pytest.approx(2)
    assert s.y == pytest.approx(2)
    assert set(solver.problem.residues) == {"z == 2.0"}
    assert solver.problem.residues["z == 2.0"].equation == "z == 2.0"
    # TODO: should be as below (with 'target' suffix)
    # assert set(solver.problem.residues) == {"z == 2.0 (target)"}
    # assert solver.problem.residues["z == 2.0 (target)"].equation == "z == 2.0"

    s.z = 4.0
    s.run_drivers()
    assert s.z == pytest.approx(4)
    assert s.y == pytest.approx(np.sqrt(8))

    s.x = 1.0
    s.z = 4.0
    s.run_drivers()
    assert s.z == pytest.approx(4)
    assert s.y == pytest.approx(2)


def test_System_add_target_design():
    """Use of `add_target` in a design method"""
    class SystemWithTarget(System):
        def setup(self):
            self.add_inward('x', 1.0)
            self.add_inward('y', 1.0)
            self.add_outward('z', 1.0)

            design = self.add_design_method('target_z')
            design.add_unknown('y').add_target('z')

        def compute(self):
            self.z = self.x * self.y**2

    s = SystemWithTarget('s')

    offdesign = s.get_unsolved_problem()
    assert offdesign.shape == (0, 0)

    solver = s.add_driver(NonLinearSolver('solver', tol=1e-9))
    solver.runner.design.extend(s.design('target_z'))

    s.x = 0.5
    s.y = 0.5
    s.z = 2.0  # set target
    s.run_drivers()
    assert s.z == pytest.approx(2)
    assert s.y == pytest.approx(2)
    assert solver.problem.shape == (1, 1)

    s.z = 4.0  # dynamically set new target
    s.run_drivers()
    assert s.z == pytest.approx(4)
    assert s.y == pytest.approx(np.sqrt(8))


def test_System_add_target_array():
    """Use of `add_target` in a design method, with an array variable"""
    class SystemWithTarget(System):
        def setup(self):
            self.add_inward('x', 1.0)
            self.add_inward('y', 1.0)
            self.add_outward('z', np.zeros(2))

            design = self.add_design_method('target_z')
            design.add_unknown(['x', 'y']).add_target('z')

        def compute(self):
            self.z[:] = [self.x * self.y**2, self.y / self.x]

    s = SystemWithTarget('s')

    offdesign = s.get_unsolved_problem()
    assert offdesign.shape == (0, 0)

    solver = s.add_driver(NonLinearSolver('solver', tol=1e-9, factor=0.5))
    solver.runner.design.extend(s.design('target_z'))

    s.x = 1.0
    s.y = 1.0
    s.z = np.r_[2.0, 4.0]  # set target
    s.run_drivers()
    assert s.x == pytest.approx(0.5)
    assert s.y == pytest.approx(2)
    assert s.z == pytest.approx([2, 4])
    assert solver.problem.shape == (2, 2)

    s.z = np.r_[-4.0, 0.25]  # dynamically set new target
    s.run_drivers()
    assert solver.problem.shape == (2, 2)
    assert s.x == pytest.approx(-4)
    assert s.y == pytest.approx(-1)
    assert s.z == pytest.approx([-4, 0.25])


def test_System_add_target_expression():
    """Use of `add_target` in a design method, with an evaluable expression"""
    class SystemWithTarget(System):
        def setup(self):
            self.add_inward('x', 1.0)
            self.add_inward('y', 1.0)
            self.add_outward('z', 1.0)

            design = self.add_design_method('target_z')
            design.add_unknown('y').add_target('abs(z)')

        def compute(self):
            self.z = self.x * self.y**2

    s = SystemWithTarget('s')

    offdesign = s.get_unsolved_problem()
    assert offdesign.shape == (0, 0)

    solver = s.add_driver(NonLinearSolver('solver', tol=1e-9))
    solver.runner.design.extend(s.design('target_z'))

    s.x = -0.5
    s.y = 0.5
    s.z = 2.0
    s.run_drivers()
    assert s.x == -0.5
    assert s.y == pytest.approx(2)
    assert s.z == pytest.approx(-2)

    s.z = 4.0
    s.run_drivers()
    assert s.x == -0.5
    assert s.y == pytest.approx(np.sqrt(8))
    assert s.z == pytest.approx(-4)

    s.x = 0.5
    s.z = -1.0
    s.run_drivers()
    assert s.x == 0.5
    assert s.y == pytest.approx(np.sqrt(2))
    assert s.z == pytest.approx(1)


def test_System_add_target_composite():
    """Use of `add_target` in a composite system"""
    class SubSystem(System):
        def setup(self):
            self.add_inward('x', 1.0)
            self.add_inward('y', 1.0)
            self.add_outward('z', 1.0)

        def compute(self):
            self.z = self.x * self.y**2

    class TopSystem(System):
        def setup(self):
            self.add_child(SubSystem('sub'))

            design = self.add_design_method('target_z')
            design.add_unknown('sub.y').add_target('sub.z')

    top = TopSystem('top')

    offdesign = top.get_unsolved_problem()
    assert offdesign.shape == (0, 0)

    solver = top.add_driver(NonLinearSolver('solver', tol=1e-9))
    solver.runner.design.extend(top.design('target_z'))

    top.sub.x = 0.5
    top.sub.y = 0.5
    top.sub.z = 2.0  # set target
    top.run_drivers()
    assert top.sub.z == pytest.approx(2)
    assert top.sub.y == pytest.approx(2)
    assert solver.problem.shape == (1, 1)

    top.sub.z = 4.0  # dynamically set new target
    top.run_drivers()
    assert top.sub.z == pytest.approx(4)
    assert top.sub.y == pytest.approx(np.sqrt(8))


def test_System_add_target_pulled_output_1():
    """Test involving a target set on a pulled output.
    Related to https://gitlab.com/cosapp/cosapp/-/issues/57
    """
    class SubSystem(System):
        def setup(self):
            self.add_inward('x', 0.0)
            self.add_outward('y', 0.0)
            
            self.add_unknown('x').add_target('y')
        
        def compute(self):
            self.y = 0.5 * self.x

    class TopSystem(System):
        def setup(self):
            self.add_child(SubSystem('sub'), pulling='y')

    top = TopSystem('top')
    solver = top.add_driver(NonLinearSolver('solver'))

    top.y = 3.14      # specify target at top level
    top.sub.y = -0.1  # set child variable to another value

    top.run_drivers()

    problem = solver.problem
    assert problem.shape == (1, 1)
    residue = list(problem.residues.values())[0]
    assert residue.name == "y == 3.14"
    assert top.y == pytest.approx(3.14)
    assert top.sub.x == pytest.approx(6.28)
    assert top.sub.y == pytest.approx(3.14)


def test_System_add_target_pulled_output_2():
    """Test involving a target set on a pulled output.
    Related to https://gitlab.com/cosapp/cosapp/-/issues/54
    """
    class SubSystem(System):
        """y = 0.5 * x**2"""
        def setup(self):
            self.add_inward('x', 1.0)
            self.add_outward('y', 0.0)
            
            self.add_unknown('x').add_target('y', weak=True)
        
        def compute(self):
            self.y = 0.5 * self.x**2

    class TopSystem(System):
        """y = 0.25 * x**4, by combination of two subsystems"""
        def setup(self):
            a = self.add_child(SubSystem('a'), pulling='x')
            b = self.add_child(SubSystem('b'), pulling='y')

            self.connect(a, b, {'y': 'x'})

    top = TopSystem('top')
    solver = top.add_driver(NonLinearSolver('solver', tol=1e-6))

    top.y = target = 3.14   # specify target at top level
    top.b.y = -0.1   # set child variable to another value

    top.run_drivers()

    problem = solver.problem
    assert problem.shape == (1, 1)
    residue = list(problem.residues.values())[0]
    assert residue.name == f"y == {target}"
    assert top.y == pytest.approx(target)
    assert top.x == pytest.approx((8 * target)**0.25)
    assert top.b.y == pytest.approx(target)


@pytest.mark.parametrize("weak", [True, False])
def test_System_add_target_weak(weak):
    """Use of `add_target` with `weak` option"""
    class SystemA(System):
        def setup(self):
            self.add_inward('x', 1.0)
            self.add_inward('y', 1.0)
            self.add_outward('z', 1.0)

            self.add_unknown('y').add_target('z', weak=weak)

        def compute(self):
            self.z = self.x * self.y**2

    class SystemB(System):
        def setup(self):
            self.add_inward('u', 0.0)
            self.add_outward('v', 0.0)

        def compute(self):
            self.v = 2 * self.u

    class TopSystem(System):
        def setup(self):
            a = self.add_child(SystemA('a'))
            b = self.add_child(SystemB('b'))

            self.connect(a.outwards, b.inwards, {'z': 'u'})

    top = TopSystem('top')

    offdesign = top.get_unsolved_problem()
    assert set(offdesign.unknowns) == {'a.y'}
    assert len(offdesign.residues) == 0

    if weak:
        # Weak target: residue is suppressed due to a.z -> b.u connection
        assert len(offdesign.deferred_residues) == 0
        assert offdesign.shape == (1, 0)

    else:
        # Strong target: residue is maintained despite connection
        assert len(offdesign.deferred_residues) == 1
        assert offdesign.shape == (1, 1)

        solver = top.add_driver(NonLinearSolver('solver', tol=1e-9))

        top.a.x = 0.5
        top.a.y = 0.5
        top.a.z = 2.0  # set target
        top.run_drivers()
        assert top.a.x == 0.5
        assert top.a.y == pytest.approx(2)
        assert top.a.z == pytest.approx(2)
        assert top.b.u == top.a.z
        assert solver.problem.shape == (1, 1)


# @pytest.mark.parametrize("weak", [True, False])
# def test_System_add_target_weak_in(weak):
#     """Use of `add_target` on an input with `weak` option"""
#     class SystemA(System):
#         def setup(self):
#             self.add_inward('x', 1.0)

#     class SystemB(System):
#         def setup(self):
#             self.add_inward('u', 0.0)
#             self.add_outward('v', 0.0)

#         def compute(self):
#             self.v = 2 * self.u

#     class TopSystem(System):
#         def setup(self):
#             a = self.add_child(SystemA('a'))
#             b = self.add_child(SystemB('b'))

#             self.connect(a, b, {'x': 'v'})
#             self.add_unknown('b.u').add_target('a.x', weak=weak)
#             self.exec_order = ['b', 'a']

#     top = TopSystem('top')

#     offdesign = top.get_unsolved_problem()
#     assert set(offdesign.unknowns) == {'b.u'}
#     assert len(offdesign.residues) == 0

#     if weak:
#         # Weak target: residue is suppressed due to b.v -> a.x connection
#         print(offdesign, offdesign.shape, sep="\n")
#         assert len(offdesign.deferred_residues) == 0
#         assert offdesign.shape == (1, 0)

#     else:
#         # Strong target: residue is maintained despite connection
#         assert len(offdesign.deferred_residues) == 1
#         assert offdesign.shape == (1, 1)

#         solver = top.add_driver(NonLinearSolver('solver', tol=1e-9))

#         top.b.u = 0.1
#         top.a.x = 0.5  # set target
#         top.run_drivers()
#         assert top.b.u == pytest.approx(0.25)
#         assert top.a.x == pytest.approx(0.5)
#         assert top.b.v == pytest.approx(0.5)
#         assert solver.problem.shape == (1, 1)


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
            unknowns = {"y": dict()},
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
            unknowns = {"x": dict(
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
    system: System = DummyFactory("test", **ctor_data)  # test object
    problem = system.get_unsolved_problem()
    assert isinstance(problem, MathematicalProblem)
    expected_unknowns = expected_data.get("unknowns", dict())
    expected_residues = expected_data.get("residues", dict())
    assert problem.shape == (len(expected_unknowns), len(expected_residues))
    # Test unknowns
    assert set(system.unknowns) == set(expected_unknowns)
    assert set(problem.unknowns) == set(expected_unknowns)
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
    assert set(system.residues) == set(expected_residues)
    assert set(problem.residues) == set(expected_residues)
    for name, residue in problem.residues.items():
        expected = expected_residues[name]
        assert isinstance(residue, Residue)
        assert residue is system._math.residues[name]
        assert residue.name == name
        assert residue.value == pytest.approx(expected["value"], rel=1e-14)


def test_System_get_unsolved_problem_seq(DummyFactory):
    """Non-parametric version of `test_System_get_unsolved_system()`.
    Tests method `get_unsolved_problem` for systems with children."""
    a: System = DummyFactory("a",
        inwards = [get_args("x", 22.0), get_args("y", 42.0)],
        unknowns = get_args("x", 1.0, 0.1, -5, 40.0),
        equations = get_args("y == 1", name="dummy"),
    )
    problem = a.get_unsolved_problem()
    assert isinstance(problem, MathematicalProblem)
    assert len(problem.unknowns) == 1
    assert len(problem.residues) == 1

    T: System = DummyFactory("top")
    assert isinstance(T, System)
    T.add_child(a)
    problem = T.get_unsolved_problem()
    assert isinstance(problem, MathematicalProblem)
    assert len(problem.unknowns) == 1
    assert len(problem.residues) == 1
    # Test unknowns
    assert set(problem.unknowns) == {"a.x"}
    unknown = problem.unknowns["a.x"]
    assert unknown is a._math.unknowns["x"]
    assert unknown.name == "x"
    assert unknown.context is a
    # Test residues/equations
    assert set(problem.residues) == {"a: dummy"}
    residue = problem.residues["a: dummy"]
    assert residue is a._math.residues["dummy"]
    assert residue.name == "dummy"
    assert residue.context is a

    # Test that unknowns are suppressed by connection to a peer
    T: System = DummyFactory('top')
    b: System = DummyFactory('b', inwards=get_args('x'), outwards=get_args('y'))
    c: System = DummyFactory('c', base=b.__class__, unknowns=get_args('x'))
    T.add_child(b)
    T.add_child(c)
    T.connect(T.b.outwards, T.c.inwards, {"y": "x"})

    problem = T.get_unsolved_problem()
    assert problem.shape == (0, 0)

    # Test that unknowns are forwarded by connection to the parent
    T: System = DummyFactory('top')
    s: System = DummyFactory('sub',
        inwards=get_args('x'),
        outwards=get_args('y'),
        unknowns=get_args('x'),
        )
    T.add_child(s, pulling=["x"])

    problem = T.get_unsolved_problem()
    assert problem.shape == (1, 0)
    unknown = problem.unknowns["x"]
    assert unknown.context is T


@pytest.mark.parametrize("direction", PortType)
def test_System_connect_orphan_ports(DummyFactory, direction):
    s: System = DummyFactory("s",
        inputs = get_args(PtWPort, "flow_in"),
        outputs = get_args(PtWPort, "flow_out"),
    )
    orphan = PtWPort("orphan", direction=direction)

    with pytest.raises(ValueError, match="Cannot connect orphan port"):
        s.connect(s.flow_in, orphan)

    with pytest.raises(ValueError, match="Cannot connect orphan port"):
        s.connect(s.flow_out, orphan)


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
    assert_keys(connectors, "s1.out -> s2.in_")
    connector = connectors["s1.out -> s2.in_"]
    assert connector.source is s1.out
    assert connector.sink is s2.in_
    assert connector.mapping == {"Pt": "Pt", "W": "W"}
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
    assert_keys(connectors, "s1.out -> s2.in_")
    connector = connectors["s1.out -> s2.in_"]
    assert connector.source is s1.out
    assert connector.sink is s2.in_
    assert connector.mapping == {"Pt": "Pt"}
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
    assert_keys(connectors, "s1.out -> s2.in_")
    connector = connectors["s1.out -> s2.in_"]
    assert connector.source is s1.out
    assert connector.sink is s2.in_
    assert connector.mapping == {"Pt": "Pt", "W": "W"}
    assert connector._unit_conversions == {"Pt": (1, 0), "W": (1, 0)}

    s1 = SubSystem("s1")
    group: System = DummyFactory("hat",
        inputs=get_args(PtWPort, "hat_in"),
        outputs=get_args(PtWPort, "hat_out"))
    group.add_child(s1)

    assert len(group.connectors) == 0
    assert len(s1.connectors) == 0

    group.connect(group.hat_in, s1.in_)
    connectors = group.connectors
    assert all(isinstance(c, Connector) for c in connectors.values())
    assert_keys(connectors, "hat_in -> s1.in_")
    connector = connectors["hat_in -> s1.in_"]
    assert connector.source is group.hat_in
    assert connector.sink is s1.in_
    assert connector.mapping == {"Pt": "Pt", "W": "W"}
    assert connector._unit_conversions == {"Pt": (1, 0), "W": (1, 0)}
    # Add new connection
    group.connect(s1.out, group.hat_out)
    connectors = group.connectors
    assert len(s1.connectors) == 0
    assert all(isinstance(c, Connector) for c in connectors.values())
    assert_keys(connectors, "hat_in -> s1.in_", "s1.out -> hat_out")
    connector = connectors["s1.out -> hat_out"]
    assert connector.source is s1.out
    assert connector.sink is group.hat_out
    assert connector.mapping == {"Pt": "Pt", "W": "W"}
    assert connector._unit_conversions == {"Pt": (1, 0), "W": (1, 0)}

    # Same as previous test, with different args order in group.connect(...)
    s1 = SubSystem("s1")
    group: System = DummyFactory("hat",
        inputs=get_args(PtWPort, "hat_in"),
        outputs=get_args(PtWPort, "hat_out"))
    group.add_child(s1)

    assert len(group.connectors) == 0
    assert len(s1.connectors) == 0

    group.connect(s1.in_, group.hat_in)
    connectors = group.connectors
    assert all(isinstance(c, Connector) for c in connectors.values())
    assert_keys(connectors, "hat_in -> s1.in_")
    connector = connectors["hat_in -> s1.in_"]
    assert connector.source is group.hat_in
    assert connector.sink is s1.in_
    assert connector.mapping == {"Pt": "Pt", "W": "W"}
    assert connector._unit_conversions == {"Pt": (1, 0), "W": (1, 0)}
    # Add new connection
    group.connect(group.hat_out, s1.out)
    connectors = group.connectors
    assert len(s1.connectors) == 0
    assert all(isinstance(c, Connector) for c in connectors.values())
    assert_keys(connectors, "hat_in -> s1.in_", "s1.out -> hat_out")
    connector = connectors["s1.out -> hat_out"]
    assert connector.source is s1.out
    assert connector.sink is group.hat_out
    assert connector.mapping == {"Pt": "Pt", "W": "W"}
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
    s2: System = DummyFactory("s2", inputs=get_args(CopyCatPort, "in_"))
    group = System("hat")
    group.add_child(s1)
    group.add_child(s2)
    group.connect(s2.in_, s1.out, {"Pt": "Pt", "W": "W"})

    connectors = group.connectors
    assert len(s1.connectors) == 0
    assert all(isinstance(c, Connector) for c in connectors.values())
    assert_keys(connectors, "s1.out -> s2.in_")
    connector = connectors["s1.out -> s2.in_"]
    assert connector.source is s1.out
    assert connector.sink is s2.in_
    assert connector.mapping == {"Pt": "Pt", "W": "W"}
    assert connector._unit_conversions == {"Pt": (1, 0), "W": (1, 0)}

    class CopyCatPort(Port):
        def setup(self):
            self.add_variable("P", 101325.0, unit="Pa")
            self.add_variable("Mfr", 1.0, unit="lbm/s")

    s1 = SubSystem("s1")
    s2: System = DummyFactory("s2", inputs=get_args(CopyCatPort, "in_"))
    group = System("hat")
    group.add_child(s1)
    group.add_child(s2)
    group.connect(s2.in_, s1.out, {"P": "Pt", "Mfr": "W"})

    connectors = group.connectors
    assert len(s1.connectors) == 0
    assert all(isinstance(c, Connector) for c in connectors.values())
    assert_keys(connectors, "s1.out -> s2.in_")
    connector = connectors["s1.out -> s2.in_"]
    assert connector.source is s1.out
    assert connector.sink is s2.in_
    assert connector.mapping == {"P": "Pt", "Mfr": "W"}
    assert connector._unit_conversions == {"P": (1, 0), "Mfr": pytest.approx((2.2046226218487757, 0), abs=1e-14)}

    # Uncompatible unit
    class CopyCatPort(Port):
        def setup(self):
            self.add_variable("P", 101325.0, unit="N")
            self.add_variable("Mfr", 1.0, unit="kg/s")

    s1 = SubSystem("s1")
    s2: System = DummyFactory("s2", inputs=get_args(CopyCatPort, "in_"))
    group = System("hat")
    group.add_child(s1)
    group.add_child(s2)
    with pytest.raises(UnitError):
        group.connect(s2.in_, s1.out, {"P": "Pt", "Mfr": "W"})


def test_System_connect_full():
    class System1(System):
        def setup(self):
            self.add_inward({"test": 7.0, "a": 25.0, "b": 42.0})
            self.add_outward({"local1": 11.0, "local2": 22.0, "local3": 33.0})

    class System2(System):
        def setup(self):
            self.add_inward({"data1": 9.0, "data2": 11.0, "data3": 13.0})
            self.add_outward({"test": 7.0, "a": 14.0, "b": 21.0})

    group = System("hat")
    group.add_child(System1("s1"))
    group.add_child(System2("s2"))
    group.add_child(EntryExit("s3"))
    group.add_child(EntryExit("s4"))

    group.connect(group.s1.inwards, group.s3.entry)
    connectors = group.connectors
    assert all(isinstance(c, Connector) for c in connectors.values())
    assert set(connectors) == {
        "s3_entry -> s3.entry",
        "s3_entry -> s1.inwards",
    }
    # First connector:
    connector = connectors["s3_entry -> s3.entry"]
    assert connector.source is group.s3_entry
    assert connector.sink is group.s3.entry
    assert connector.mapping == {"a": "a", "b": "b"}
    assert connector._unit_conversions == {"a": (1, 0), "b": (1, 0)}
    # Second connector:
    connector = connectors["s3_entry -> s1.inwards"]
    assert connector.source is group.s3_entry
    assert connector.sink is group.s1.inwards
    assert connector.mapping == {"a": "a", "b": "b"}
    assert connector._unit_conversions == {"a": (1, 0), "b": (1, 0)}

    with pytest.raises(ConnectorError, match="already connected"):
        group.connect(group.s3.entry, group.s1.inwards)

    group = System("hat")
    group.add_child(System1("s1"))
    group.add_child(System2("s2"))
    group.add_child(EntryExit("s3"))
    group.add_child(EntryExit("s4"))

    group.connect(group.s1.inwards, group.s3.exit)
    connectors = group.connectors
    assert all(isinstance(c, Connector) for c in connectors.values())
    assert_keys(connectors, "s3.exit -> s1.inwards")
    connector = connectors["s3.exit -> s1.inwards"]
    assert connector.source is group.s3.exit
    assert connector.sink is group.s1.inwards
    assert connector.mapping == {"a": "a", "b": "b"}
    assert connector._unit_conversions == {"a": (1, 0), "b": (1, 0)}
    # Add new connection
    group.connect(group.s2.outwards, group.s4.entry)
    connectors = group.connectors
    assert all(isinstance(c, Connector) for c in connectors.values())
    assert_keys(connectors, "s3.exit -> s1.inwards", "s2.outwards -> s4.entry")
    connector = connectors["s2.outwards -> s4.entry"]
    assert connector.source is group.s2.outwards
    assert connector.sink is group.s4.entry
    assert connector.mapping == {"a": "a", "b": "b"}
    assert connector._unit_conversions == {"a": (1, 0), "b": (1, 0)}

    with pytest.raises(ConnectorError, match=r"s1.inwards.{a, b} are already set by Connector\(s1.inwards <- s3.exit"):
        group.connect(group.s1.inwards, group.s2.outwards)

    group = System("hat")
    group.add_child(System1("s1"))
    group.add_child(System2("s2"))
    group.add_child(EntryExit("s3"))
    group.add_child(EntryExit("s4"))

    group.connect(group.s1.inwards, group.s2.outwards)
    connectors = group.connectors
    assert all(isinstance(c, Connector) for c in connectors.values())
    assert_keys(connectors, "s2.outwards -> s1.inwards")
    connector = connectors["s2.outwards -> s1.inwards"]
    assert connector.source is group.s2.outwards
    assert connector.sink is group.s1.inwards
    assert connector.mapping == {"a": "a", "b": "b", "test": "test"}


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

    group = System("hat")
    group.add_child(System1("s1"))
    group.add_child(System2("s2"))
    group.add_child(EntryExit("s3"))
    group.add_child(EntryExit("s4"))

    group.connect(group.s1.inwards, group.s3.entry, "b")
    connectors = group.connectors
    assert all(isinstance(c, Connector) for c in connectors.values())
    assert set(connectors) == {
        "s3_entry -> s3.entry",
        "s3_entry -> s1.inwards",
    }
    # First connector:
    connector = connectors["s3_entry -> s3.entry"]
    assert connector.source is group.s3_entry
    assert connector.sink is group.s3.entry
    assert connector.mapping == {"b": "b"}
    assert connector._unit_conversions == {"b": (1, 0)}
    # Second connector:
    connector = connectors["s3_entry -> s1.inwards"]
    assert connector.source is group.s3_entry
    assert connector.sink is group.s1.inwards
    assert connector.mapping == {"b": "b"}
    assert connector._unit_conversions == {"b": (1, 0)}

    with pytest.raises(ConnectorError, match="already connected"):
        group.connect(group.s3.entry, group.s1.inwards, "b")

    group = System("hat")
    group.add_child(System1("s1"))
    group.add_child(System2("s2"))
    group.add_child(EntryExit("s3"))
    group.add_child(EntryExit("s4"))

    group.connect(group.s1.inwards, group.s3.exit, "b")
    connectors = group.connectors
    assert all(isinstance(c, Connector) for c in connectors.values())
    assert set(connectors) == {"s3.exit -> s1.inwards"}
    connector = connectors["s3.exit -> s1.inwards"]
    assert connector.source is group.s3.exit
    assert connector.sink is group.s1.inwards
    assert connector.mapping == {"b": "b"}
    assert connector._unit_conversions == {"b": (1, 0)}
    # Add new connection
    group.connect(group.s3.exit, group.s4.entry, ["a", "b"])
    connectors = group.connectors
    assert all(isinstance(c, Connector) for c in connectors.values())
    assert set(connectors) == {
        "s3.exit -> s1.inwards",
        "s3.exit -> s4.entry",
    }
    connector = connectors["s3.exit -> s4.entry"]
    assert connector.source is group.s3.exit
    assert connector.sink is group.s4.entry
    assert connector.mapping == {"a": "a", "b": "b"}

    group = System("hat")
    group.add_child(System1("s1"))
    group.add_child(System2("s2"))
    group.add_child(EntryExit("s3"))
    group.add_child(EntryExit("s4"))

    group.connect(group.s1.inwards, group.s2.inwards, {"data1": "d1"})
    connectors = group.connectors
    assert all(isinstance(c, Connector) for c in connectors.values())
    assert set(connectors) == {
        "inwards -> s1.inwards",
        "inwards -> s2.inwards",
    }
    # First connector
    connector = connectors["inwards -> s1.inwards"]
    assert connector.source is group.inwards
    assert connector.sink is group.s1.inwards
    assert connector.mapping == {"data1": "s2_d1"}
    # Second connector
    connector = connectors["inwards -> s2.inwards"]
    assert connector.source is group.inwards
    assert connector.sink is group.s2.inwards
    assert connector.mapping == {"d1": "s2_d1"}

    group = System("hat")
    group.add_child(System1("s1"))
    group.add_child(System2("s2"))
    group.add_child(EntryExit("s3"))
    group.add_child(EntryExit("s4"))

    group.connect(group.s3.exit, group.s4.entry, {"a": "a", "b": "b"})
    connectors = group.connectors
    assert all(isinstance(c, Connector) for c in connectors.values())
    assert_keys(connectors, "s3.exit -> s4.entry")
    connector = connectors["s3.exit -> s4.entry"]
    assert connector.source is group.s3.exit
    assert connector.sink is group.s4.entry
    assert connector.mapping == {"a": "a", "b": "b"}
    # Add new connection
    group.connect(group.s2.inwards, group.s1.outwards,
        {"d1": "local1", "d2": "local2"})
    connectors = group.connectors
    assert all(isinstance(c, Connector) for c in connectors.values())
    assert set(connectors) == {
        "s3.exit -> s4.entry",
        "s1.outwards -> s2.inwards",
    }
    connector = connectors["s1.outwards -> s2.inwards"]
    assert connector.source is group.s1.outwards
    assert connector.sink is group.s2.inwards
    assert connector.mapping == {"d1": "local1", "d2": "local2"}
    # Add new connection
    group.connect(group.s2.outwards, group.s1.inwards,
        {"l1": "data1", "l2": "data2"})
    connectors = group.connectors
    assert all(isinstance(c, Connector) for c in connectors.values())
    assert set(connectors) == {
        "s3.exit -> s4.entry",
        "s1.outwards -> s2.inwards",
        "s2.outwards -> s1.inwards",
    }
    connector = connectors["s2.outwards -> s1.inwards"]
    assert connector.source is group.s2.outwards
    assert connector.sink is group.s1.inwards
    assert connector.mapping == {"data1": "l1", "data2": "l2"}

    g1 = System("group1")
    s1 = g1.add_child(EntryExit("s1"))
    g2 = System("group2")
    s2 = g2.add_child(EntryExit("s2"))
    top = System("top")
    top.add_child(g1)
    top.add_child(g2)

    with pytest.raises(ConnectorError, match="Only ports belonging to direct children of '.*' can be connected."):
        g1.connect(s1.entry, s2.exit)

    # Test connection extension
    group = System("hat")
    group.add_child(System1("s1"))
    group.add_child(System2("s2"))
    group.add_child(EntryExit("s3"))
    group.add_child(EntryExit("s4"))

    # First partial connection
    group.connect(group.s4.entry, group.s3.exit, "b")
    connectors = group.connectors
    assert all(isinstance(c, Connector) for c in connectors.values())
    assert_keys(connectors, "s3.exit -> s4.entry")
    connector = connectors["s3.exit -> s4.entry"]
    assert connector.source is group.s3.exit
    assert connector.sink is group.s4.entry
    assert connector.mapping == {"b": "b"}
    assert connector._unit_conversions == {"b": (1, 0)}
    # New connection extending the existing connector
    # Should not create any new connector
    group.connect(group.s4.entry, group.s3.exit, "a")
    connectors = group.connectors
    assert all(isinstance(c, Connector) for c in connectors.values())
    assert_keys(connectors, "s3.exit -> s4.entry")
    connector = connectors["s3.exit -> s4.entry"]
    assert connector.source is group.s3.exit
    assert connector.sink is group.s4.entry
    assert connector.mapping == {"a": "a", "b": "b"}
    assert connector._unit_conversions == {"a": (1, 0), "b": (1, 0)}


def test_System_connect_custom():
    top = System("top")
    s1 = top.add_child(SubSystem("s1"))
    s2 = top.add_child(SubSystem("s2"))

    with pytest.raises(TypeError, match="cls"):
        top.connect(s1.out, s2.in_, cls="Foo")

    class Foo:
        pass

    with pytest.raises(ValueError, match="cls"):
        top.connect(s1.out, s2.in_, cls=Foo)

    class PlainConnector(BaseConnector):
        """Simple assignment connector.
        """
        def transfer(self) -> None:
            source, sink = self.source, self.sink

            for target, origin in self.mapping.items():
                value = getattr(source, origin)
                setattr(sink, target, value)

    with no_exception():
        top.connect(s1.out, s2.in_, cls=PlainConnector)

    assert all(isinstance(c, BaseConnector) for c in top.all_connectors())
    connectors = top.connectors
    assert set(connectors) == {"s1.out -> s2.in_"}
    connector = connectors["s1.out -> s2.in_"]
    assert isinstance(connector, PlainConnector)
    assert connector.source is top.s1.out
    assert connector.sink is top.s2.in_
    assert connector.mapping == {'Pt': 'Pt', 'W': 'W'}

    # Test mapping extension
    s3 = top.add_child(SubSystem("s3"))
    top.connect(s2.out, s3.in_, 'Pt', cls=PlainConnector)
    connectors = top.connectors
    assert set(connectors) == {
        "s1.out -> s2.in_",
        "s2.out -> s3.in_",
    }
    connector = connectors["s2.out -> s3.in_"]
    assert isinstance(connector, PlainConnector)
    assert connector.mapping == {'Pt': 'Pt'}
    top.connect(s2.out, s3.in_, 'W', cls=PlainConnector)
    assert connector.mapping == {'Pt': 'Pt', 'W': 'W'}


def test_System_connect_systems():
    """Tests system/system connections"""
    top = System("top")
    s1 = top.add_child(SubSystem("s1"))
    s2 = top.add_child(SubSystem("s2"))
    s3 = top.add_child(EntryExit("s3"))

    with pytest.raises(TypeError, match="either two ports or two systems"):
        top.connect(s1, s2.out)

    with pytest.raises(TypeError, match="either two ports or two systems"):
        top.connect(s2.out, s1)

    with pytest.raises(ConnectorError, match="Full system connections are forbidden"):
        top.connect(s1, s2)

    with pytest.raises(TypeError, match="port2"):
        top.connect(s2, s3, {'out.Pt': 'entry'})

    top.connect(s1, s2, {'out': 'in_'})
    top.connect(s2, s3, {'out.Pt': 'entry.b', 'sloss': 'exit.a'})

    connectors = top.connectors
    assert set(connectors) == {
        "s1.out -> s2.in_",
        "s2.out -> s3.entry",
        "s3.exit -> s2.inwards",
    }
    assert connectors['s1.out -> s2.in_'].mapping == {'Pt': 'Pt', 'W': 'W'}
    assert connectors['s2.out -> s3.entry'].mapping == {'b': 'Pt'}
    assert connectors['s3.exit -> s2.inwards'].mapping == {'sloss': 'a'}

    # Complete existing connector
    top.connect(s2, s3, {'out.W': 'entry.a'})
    assert connectors['s2.out -> s3.entry'].mapping == {'b': 'Pt', 'a': 'W'}

    # Mapping requesting full connection between two sub-systems (forbidden)
    s4 = top.add_child(TopSystem('s4'))
    s5 = top.add_child(TopSystem('s5'))
    with pytest.raises(ConnectorError, match="Full system connections are forbidden"):
        top.connect(s4, s5, 'sub')

    # Mapping pointing to sub-system ports
    with pytest.raises(
        ConnectorError,
        match="Only ports belonging to direct children of 'top' can be connected",
    ):
        top.connect(s4, s5, {'sub.out': 'sub.in_'})

    # Mapping suggesting a pulling
    top = EntryExit('top')
    sub = top.add_child(EntryExit('sub'))
    top.connect(top, sub, ['entry', 'exit'])
    connectors = top.connectors
    assert set(connectors) == {
        'entry -> sub.entry',
        'sub.exit -> exit',
    }
    assert connectors['entry -> sub.entry'].mapping == {'a': 'a', 'b': 'b'}
    assert connectors['sub.exit -> exit'].mapping == {'a': 'a', 'b': 'b'}

    # Mapping suggesting partial pulling
    top = EntryExit('top')
    sub = top.add_child(EntryExit('sub'))
    top.connect(top, sub, ['entry.a', 'exit.b'])
    connectors = top.connectors
    assert set(connectors) == {
        'entry -> sub.entry',
        'sub.exit -> exit',
    }
    assert connectors['entry -> sub.entry'].mapping == {'a': 'a'}
    assert connectors['sub.exit -> exit'].mapping == {'b': 'b'}


def test_System_connect_port_connectors(caplog):
    """Tests connections with class-specific port connector
    """
    class AbcConnector(BaseConnector):
        """Connector for `AbcPort` objects
        """
        def __init__(self, name: str, sink: BasePort, source: BasePort, *args, **kwargs):
            super().__init__(name, sink, source, mapping=self.fixed_mapping())
        
        def transfer(self) -> None:
            source, sink = self.source, self.sink
            sink.a = source.b
            sink.b = -source.a

        @staticmethod
        def fixed_mapping():
            return dict(zip('ba', 'ab'))

    class AbPort(Port):
        """Port class with custom connector"""
        def setup(self):
            self.add_variable("a", 1.0)
            self.add_variable("b", 2.0)
        
        # Port-specific connector
        Connector = AbcConnector

    class XyPort(Port):
        def setup(self):
            self.add_variable("x", 1.0)
            self.add_variable("y", 2.0)

    class AbSystem(System):
        def setup(self):
            self.add_input(AbPort, 'p_in')
            self.add_output(AbPort, 'p_out')

        def compute(self):
            self.p_out.a = 2 * self.p_in.a
            self.p_out.b = 2 * self.p_in.b

    class XySystem(System):
        def setup(self):
            self.add_input(XyPort, 'p_in')
            self.add_outward('v', np.zeros(2))

        def compute(self):
            self.v = np.array([self.p_in.x, self.p_in.y])

    top = System("top")
    s1 = top.add_child(AbSystem("s1"))
    s2 = top.add_child(AbSystem("s2"))
    s3 = top.add_child(XySystem("s3"))

    caplog.clear()
    with caplog.at_level(logging.INFO):
        top.connect(s1.p_out, s2.p_in)
        top.connect(s2.p_out, s3.p_in, dict(zip('ab', 'xy')))

    assert len(caplog.records) == 1
    assert re.match(
        "'s1.p_out' and 's2.p_in' connected by.* `AbPort\.Connector`",
        caplog.records[0].message
    )

    connectors = top.connectors
    assert set(connectors) == {
        "s1.p_out -> s2.p_in",
        "s2.p_out -> s3.p_in",
    }
    assert isinstance(connectors['s1.p_out -> s2.p_in'], AbPort.Connector)
    assert isinstance(connectors['s2.p_out -> s3.p_in'], Connector)

    assert connectors['s1.p_out -> s2.p_in'].mapping == dict(zip('ba', 'ab'))
    assert connectors['s2.p_out -> s3.p_in'].mapping == dict(zip('xy', 'ab'))

    top.s1.p_in.a = 0.1
    top.s1.p_in.b = 0.25
    top.run_once()
    assert top.s2.p_in.a == pytest.approx(0.5, rel=1e-15)
    assert top.s2.p_in.b == pytest.approx(-0.2, rel=1e-15)
    assert top.s3.v == pytest.approx([1, -0.4], rel=1e-15)


def test_System_connect_empty():
    """Check that empty connectors are discarded"""
    top = System("top")
    s1 = top.add_child(SubSystem("s1"))
    s2 = top.add_child(EntryExit("s2"))
    s3 = top.add_child(EntryExit("s3"))

    with pytest.warns(UserWarning, match="empty connector"):
        top.connect(s1.out, s2.entry)
    assert len(top.connectors) == 0

    with pytest.warns(UserWarning, match="empty connector"):
        top.connect(s2.exit, s3.entry, [])
    assert len(top.connectors) == 0


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
        system: System = DummyFactory("dummy", **ctor_data)  # test object
        assert system.properties == expected.get('properties', {})

    else:
        with pytest.raises(error, match=expected.get('match', None)):
            DummyFactory("dummy", **ctor_data)


def test_System_properties_safeview(DummyFactory):
    dummy: System = DummyFactory("dummy", 
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


def test_System_add_event():
    class SystemWithEvent(System):
        def setup(self):
            self.add_event('beep')

    a = SystemWithEvent("a")
    b = System("b")

    assert hasattr(a, 'beep')
    assert not hasattr(b, 'beep')

    with pytest.raises(AttributeError, match="can't set attribute"):
        a.beep = 3.14

    with pytest.raises(AttributeError, match="has no attribute 'beep'"):
        b.beep

    with pytest.raises(AttributeError, match="`add_event` cannot be called outside `setup`"):
        b.add_event("boom")


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


@pytest.mark.parametrize("args_kwargs, expected", [
    # `args_kwargs` is a (tuple, dict) tuple
    (get_args(True), True),
    (get_args(1.25), 1.25),
    (get_args(1.25, unit='kg'), 1.25),
    (get_args(0.37, init='x + y'), 0.37),
    (get_args(init='x + y'), 1.5),  # init, but no value
])
def test_System_add_outward_modevar(DummyFactory, args_kwargs, expected):
    args, kwargs = args_kwargs
    s: System = DummyFactory("dummy",
        inwards = get_args('x', 1.0),
        outwards = get_args('y', 0.5),
        outward_modevars = get_args('a', *args, **kwargs),
    )
    assert "a" in s
    assert f"{System.MODEVARS_OUT}.a" in s
    assert s.a == expected


def test_System_add_outward_modevar_init(DummyFactory):
    s: System = DummyFactory("dummy",
        inwards = get_args('x', 1.0),
        outwards = get_args('y', 0.5),
        outward_modevars = [
            get_args('a', 0.1),  # value, no init
            get_args('b', init='x + y'),  # init, but no value
            get_args('c', 0.3, init='x + y'),  # value and init
        ],
    )
    assert s.a == 0.1
    assert s.b == 1.5
    assert s.c == 0.3
    port = s[System.MODEVARS_OUT]
    a = port.get_details('a')
    b = port.get_details('b')
    c = port.get_details('c')
    s.x = -1.5
    s.y = -0.2
    assert a.value == 0.1
    assert b.value == 1.5
    assert c.value == 0.3
    assert a.init_value() is None
    assert b.init_value() == -1.7
    assert c.init_value() == -1.7
