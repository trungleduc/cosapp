import pytest

from cosapp.ports import Port
from cosapp.systems import System

from ..markdown import system_to_md

# <codecell> Local test classes

class DummyPort(Port):
    def setup(self):
        self.add_variable("a", 1)
        self.add_variable("b", 2)

class AnotherPort(Port):
    def setup(self):
        self.add_variable("aaaa", 1)

class System2(System):
    def setup(self):
        self.add_inward({"data1": 9.0, "data2": 11.0, "data3": 13.0})
        self.add_outward({"local1": 7.0, "a": 14.0, "b": 21.0})
        self.add_output(AnotherPort, "other")

class System3(System):
    def setup(self):
        self.add_input(DummyPort, "entry")
        self.add_output(DummyPort, "exit")

class System4(System):
    def setup(self):
        self.add_inward("X", -1.0)
        self.add_equation("X == 1.")

class DynamicSystem2(System2):
    def setup(self):
        super().setup()
        self.add_transient("A", der="a")
        self.add_rate("db_dt", source="b")

class DynamicSystem3(System3):
    def setup(self):
        super().setup()
        self.add_transient("A", der="entry.a")
        self.add_transient("B", der="entry.b")
        self.add_rate("db_out", source="exit.b")

# <codecell>

# TODO('be sure the test is complete')
def test_markdown():
    s = System2("s")
    markdown = system_to_md(s)
    assert "### Child components" not in markdown
    assert "#### {}".format(System.INWARDS.capitalize()) in markdown
    assert "#### {}".format(System.OUTWARDS.capitalize()) in markdown
    assert "### Ports" in markdown
    assert "#### Inputs" not in markdown
    assert "#### Outputs" in markdown
    assert "#### Residues" not in markdown

    s = System3("s")
    markdown = system_to_md(s)
    assert "### Child components" not in markdown
    assert "#### {}".format(System.INWARDS.capitalize()) not in markdown
    assert "#### {}".format(System.OUTWARDS.capitalize()) not in markdown
    assert "### Ports" in markdown
    assert "#### Inputs" in markdown
    assert "#### Outputs" in markdown
    assert "#### Residues" not in markdown

    s = System4("s")
    markdown = system_to_md(s)
    assert "### Child components" not in markdown
    assert "#### {}".format(System.INWARDS.capitalize()) in markdown
    assert "#### {}".format(System.OUTWARDS.capitalize()) not in markdown
    assert "### Ports" not in markdown
    assert "#### Inputs" not in markdown
    assert "#### Outputs" not in markdown
    assert "### Residues" in markdown

    u = System("child")
    s.add_child(u)
    markdown = system_to_md(s)
    assert "### Child components" in markdown

    class NewS(System):
        tags = ["dev"]

    s = NewS("s")
    markdown = system_to_md(s)
    assert "**Tags**" in markdown
