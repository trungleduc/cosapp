import pytest

from cosapp.ports import Port, PortType
from cosapp.systems import System
import numpy
from ..markdown import system_to_md, port_to_md_table, port_to_md

# <codecell> Local test classes


class DummyPort(Port):
    def setup(self):
        self.add_variable("a", 1)
        self.add_variable("b", 2)


class AnotherPort(Port):
    def setup(self):
        self.add_variable("aaaa", 1)


class DummyPortWithDesc(Port):
    def setup(self):
        self.add_variable("foo", 1, desc="Foo variable")
        self.add_variable("bar", 2, desc="Bar variable")


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

# TODO("be sure the test is complete")
def test_system_to_markdown():
    s = System("s")
    markdown = system_to_md(s)
    assert "### Child components" not in markdown
    assert "### Inputs" not in markdown
    assert "### Outputs" not in markdown
    assert "### Residues" not in markdown
    assert f"`{System.INWARDS}`: ExtensiblePort" not in markdown
    assert f"`{System.OUTWARDS}`: ExtensiblePort" not in markdown

    s = System2("s")
    markdown = system_to_md(s)
    assert "### Child components" not in markdown
    assert "### Inputs" in markdown
    assert "### Outputs" in markdown
    assert "### Residues" not in markdown
    assert f"`{System.INWARDS}`: ExtensiblePort" in markdown
    assert f"`{System.OUTWARDS}`: ExtensiblePort" in markdown

    s = System3("s")
    markdown = system_to_md(s)
    assert "### Child components" not in markdown
    assert "### Inputs" in markdown
    assert "### Outputs" in markdown
    assert "### Residues" not in markdown
    assert f"`{System.INWARDS}`: ExtensiblePort" not in markdown
    assert f"`{System.OUTWARDS}`: ExtensiblePort" not in markdown

    s = System4("s")
    markdown = system_to_md(s)
    assert "### Child components" not in markdown
    assert "### Inputs" in markdown
    assert "### Outputs" not in markdown
    assert "### Residues" in markdown
    assert f"`{System.INWARDS}`: ExtensiblePort" in markdown
    assert f"`{System.OUTWARDS}`: ExtensiblePort" not in markdown

    u = System("child")
    s.add_child(u)
    markdown = system_to_md(s)
    assert "### Child components" in markdown

    class NewS(System):
        tags = ["dev"]

    s = NewS("s")
    markdown = system_to_md(s)
    assert "**Tags**" in markdown


@pytest.mark.parametrize(
    "PortCls, expected",
    [
        (DummyPort, "  **a**: 1 |\n  **b**: 2 |"),
        (DummyPortWithDesc, "  **foo**: 1 | Foo variable\n  **bar**: 2 | Bar variable"),
    ],
)
def test_port_to_md(PortCls: Port, expected):
    p = PortCls("p", PortType.IN)
    ret = port_to_md(p)
    header = "<div style='margin-left:25px'>\n\n<!-- -->|<!-- --> \n---|---\n"
    footer = "\n</div>\n"
    assert ret == f"{header}{expected}{footer}"


@pytest.mark.parametrize(
    "PortCls, expected",
    [
        (
            DummyPort,
            [
                "<div style='margin-left:25px'>",
                "",
                "<!-- -->|<!-- --> ",
                "---|---",
                "  **a**: 1 |",
                "  **b**: 2 |",
                "</div>",
                "",
            ],
        ),
        (
            DummyPortWithDesc,
            [
                "<div style='margin-left:25px'>",
                "",
                "<!-- -->|<!-- --> ",
                "---|---",
                "  **foo**: 1 | Foo variable",
                "  **bar**: 2 | Bar variable",
                "</div>",
                "",
            ],
        ),
    ],
)
def test_port_to_md_table(PortCls: Port, expected):
    p = PortCls("p", PortType.IN)
    ret = port_to_md_table(p)
    print(ret)
    assert ret == expected
