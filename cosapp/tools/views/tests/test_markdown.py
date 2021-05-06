from _pytest.fixtures import reorder_items
import pytest

from cosapp.ports import Port, PortType
from cosapp.systems import System
from ..markdown import system_to_md, port_to_md_table, port_to_md, table_css

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


@pytest.fixture(scope='function')
def table_header():
    return f"{table_css()}\n\n<!-- -->|<!-- -->\n---|---\n"

# <codecell>

def test_table_css():
    assert table_css() == (
        r"<div class='cosapp-port-table' style='margin-left: 25px; margin-top: -12px'>"
        r"<style type='text/css'>"
        r".cosapp-port-table >table >thead{display: none}"
        r".cosapp-port-table tbody tr{background: white!important}"
        r".cosapp-port-table tbody tr:hover{background: #e1f5fe!important}"
        r"</style>"
    )


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


@pytest.mark.parametrize("direction", PortType)
@pytest.mark.parametrize(
    "PortCls, expected",
    [
        (DummyPort, "  **a**: 1 |\n  **b**: 2 |"),
        (DummyPortWithDesc, "  **foo**: 1 | Foo variable\n  **bar**: 2 | Bar variable"),
    ],
)
def test_port_to_md(PortCls: Port, direction, table_header, expected):
    p = PortCls("p", direction)
    header = f"`p`: {PortCls.__name__}\n\n{table_header}"
    footer = "\n</div>\n"
    assert port_to_md(p) == f"{header}{expected}{footer}"


@pytest.mark.parametrize("direction", PortType)
@pytest.mark.parametrize("PortCls, expected", [
    (
        DummyPort,
        [
            "`p`: DummyPort",
            "",
            table_css(),
            "",
            "<!-- -->|<!-- -->",
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
            "`p`: DummyPortWithDesc",
            "",
            table_css(),
            "",
            "<!-- -->|<!-- -->",
            "---|---",
            "  **foo**: 1 | Foo variable",
            "  **bar**: 2 | Bar variable",
            "</div>",
            "",
        ],
    ),
])
def test_port_to_md_table(PortCls: Port, direction, expected):
    p = PortCls("p", direction)
    assert port_to_md_table(p) == expected
