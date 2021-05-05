"""Mardown viewers for Module and Port."""
from cosapp.ports.port import ExtensiblePort
from cosapp.systems import System
from typing import List


def port_to_md(port: ExtensiblePort) -> str:
    """Returns the representation of this port variables in Markdown format.

    Parameters
    ----------
    port : ExtensiblePort
        Port to describe

    Returns
    -------
    str
        Markdown formatted representation
    """
    return "\n".join(port_to_md_table(port))


def table_css() -> str:
    """Local override of Jupyter Lab table CSS"""
    return "".join([
        r"<div class='cosapp-port-table' style='margin-left: 25px; margin-top: -12px'>",
        r"<style type='text/css'>",
        r".cosapp-port-table >table >thead{display: none}",  # suppress empty table header
        r".cosapp-port-table tbody tr{background: white!important}",  # override even/odd coloring
        r".cosapp-port-table tbody tr:hover{background: #e1f5fe!important}",  # set hover color
        r"</style>",
    ])


def port_to_md_table(port: ExtensiblePort, contextual=True) -> List[str]:
    """Returns the representation of this port variables in as a Markdown table.

    Parameters
    ----------
    port : ExtensiblePort
        Port to describe

    Returns
    -------
    List[str]
        List of Markdown strings to represent the table of variables
    """
    name = port.contextual_name if contextual else port.name
    doc = []
    doc.append(f"`{name}`: {type(port).__name__}")
    # Local override of Jupyter Lab table CSS
    doc.extend(["", table_css()])
    doc.extend(["", "<!-- -->|<!-- -->", "---|---"])
    doc.extend(f"  {value._repr_markdown_()}" for value in port.get_details().values())
    doc.extend(["</div>", ""])
    return doc


def system_to_md(system: System) -> str:
    """Returns the representation of a system in Markdown format.

    Parameters
    ----------
    system : System
        System to describe

    Returns
    -------
    str
        Markdown formatted representation
    """
    doc = list()
    if len(system.tags) > 0:
        doc.extend(["", f"**Tags**: {list(system.tags)!s}", ""])

    if len(system.children) > 0:
        doc.extend(["", "### Child components", ""])
        doc.extend(
            f"- `{name}`: {type(child).__name__}"
            for name, child in system.children.items()
        )

    def dump_port_data(header, port_dict):
        if all(len(port) == 0 for port in port_dict.values()):
            return
        doc.extend(["", f"### {header.title()}", ""])
        for port in port_dict.values():
            if len(port) > 0:
                port_doc = port_to_md_table(port, contextual=False)
                port_doc[0] = f"- {port_doc[0]}"
                doc.extend(port_doc)

    dump_port_data("inputs", system.inputs)
    dump_port_data("outputs", system.outputs)

    if hasattr(system, "residues") and len(system.residues) > 0:
        doc.extend(["", "### Residues", ""])
        doc.append(", ".join(f"`{key}`" for key in system.residues))

    return "\n".join(doc)
