"""Build a VISjs view of a System."""
import os
from typing import Dict, List, Optional, Tuple, Union

from cosapp.ports.port import ExtensiblePort
from cosapp.drivers import Driver
from cosapp.systems import System


def to_visjs(system: System, filename: str, embeddable: bool = False):
    """Export a system as HTML using vis.JS library.

    Parameters
    ----------
    system : System
        System to export
    filename : str
        Filename to write to
    embeddable: bool, optional
        Is the HTML to be embedded in an existing page? Default: False
    """
    # Optional dependencies
    try:
        from jinja2 import Environment, PackageLoader
    except ImportError:
        raise ImportError("jinja2 needs to be installed to export to vis JS.")

    env = Environment(
        loader=PackageLoader("cosapp.tools", "templates"),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template("system_repr.html")

    # This is inspired from openmdao problem_viewer

    html_begin_tags = """<!DOCTYPE html>
<html>
<head>
<meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
<title>{{title}}</title>
</head>
<body>\n
"""
    html_end_tags = """
</body>
</html>
"""

    from cosapp.tools import templates

    ressource_dir = os.path.dirname(os.path.abspath(templates.__file__))
    libs_dir = os.path.join(ressource_dir, "lib")

    # Grab the ressources
    with open(os.path.join(libs_dir, "vis.min.js"), "rb") as f:
        vis_js = f.read().decode("utf-8")
    with open(os.path.join(libs_dir, "vis.min.css"), "rb") as f:
        vis_css = f.read().decode("utf-8")

    def to_visJS(system: System, focus_system: Optional[System] = None) -> Dict:
        """Convert the `System` in a dictionary with 4 keys: title, nodes, edges and groups.

        The title is a string with the Module name.
        The nodes is a list of dictionaries defining node for vis JS library.
        The edges is a list of dictionaries defining edge for vis JS library.
        The groups is a list of string defining the nodes clusters for vis JS library.

        Parameters
        ----------
        focus_system : System or None
            System on which the visualization should focus by default

        Returns
        -------
        dict
            Dictionary containing elements to draw the `System` using visJS library
        """

        if system.parent is not None:
            center = system if focus_system is None else focus_system
            return to_visJS(system.parent, center)

        def get_component2id(
            system: System,
            cmpt2id: Dict[Union[System, ExtensiblePort], int],
            node_id: int,
            edges: List[Dict],
        ) -> Tuple[Dict[Union[System, ExtensiblePort], int], int, List[Dict]]:
            """Set the mapping between the `Module` and their unique id.

            Parameters
            ----------
            system : Module
                The parent `Module` for which id must be set
            cmpt2id : Dict[Union[Module, ExtensiblePort], int]
                Mapping of `Module` or port and its id
            node_id : int
                Latest used id
            edges : List[Dict]
                List of connection edges

            Returns
            -------
            tuple (List[Dict], Dict[Union['Module', 'ExtensiblePort'], int], int)
                The first is the mapping between each involved system or port and an unique id,
                the second is the latest used id and
                the third index is the list of edges (one per connector).
            """

            def get_driver2id(driver, cmpt2id, node_id, edges):
                if driver not in cmpt2id:
                    cmpt2id[driver] = node_id
                    node_id = node_id + 1

                previous_child = None
                for child in driver.children.values():
                    cmpt2id, node_id, edges = get_driver2id(
                        child, cmpt2id, node_id, edges
                    )
                    if (
                        previous_child is not None
                    ):  # Connection between same level driver
                        edges.append(
                            {
                                "from": cmpt2id[previous_child],
                                "to": cmpt2id[child],
                                "arrows": "to",
                                "title": "",
                                "hidden": True,
                                "physics": False,
                            }
                        )
                    previous_child = child
                    # Connection with parent driver
                    edges.append(
                        {
                            "from": cmpt2id[driver],
                            "to": cmpt2id[child],
                            "title": "",
                            "hidden": True,
                            "physics": False,
                        }
                    )

                return cmpt2id, node_id, edges

            def connectors_to_visJS(
                system: System,
                component_id: Optional[Dict[Union[System, ExtensiblePort], int]] = None,
                node_id: int = 1,
            ) -> Tuple[List[Dict], Dict[Union[System, ExtensiblePort], int], int]:
                """Build the dictionary the list of edges.

                Parameters
                ----------
                system : System
                    System owning the connectors
                component_id : Dict[Union[System, Port], int], optional
                    Mapping of `System` or `Port` and its unique id; default None
                node_id : int, optional
                    Latest used id; default 1

                Returns
                -------
                tuple (List[Dict], Dict[Union['System', 'ExtensiblePort'], int], int)
                    The first index is the list of edges (one per connector),
                    the second is the mapping between each involved system or port and an unique id and
                    the third is the latest used id.
                """
                edges = list()

                for connection in system.connectors.values():
                    supplier = connection.source.owner
                    if len(supplier.children):  # Insert port as node
                        supplier = connection.source

                    if supplier not in component_id:
                        component_id[supplier] = node_id
                        node_id = node_id + 1

                    target = connection.sink.owner
                    if len(target.children):  # Insert port as node
                        target = connection.sink

                    if target not in component_id:
                        component_id[target] = node_id
                        node_id = node_id + 1

                    edge = {
                        "from": component_id[supplier],
                        "to": component_id[target],
                        "arrows": "to",
                        "title": "",
                    }
                    if set(connection.variable_mapping.values()) == set(
                        connection.variable_mapping
                    ):
                        edge["title"] = str(list(connection.variable_mapping))
                    else:
                        edge["title"] = str(connection.variable_mapping)

                    edges.append(edge)

                    # if supplier or target is top-system, set edge.length = 0
                    if (
                        connection.source.owner is connection.sink.owner.parent
                        or connection.source.owner.parent is connection.sink.owner
                    ):
                        edges[-1]["length"] = 0

                return edges, component_id, node_id

            for driver in system.drivers.values():
                cmpt2id, node_id, edges = get_driver2id(driver, cmpt2id, node_id, edges)

            for cpt in system.children.values():
                if cpt not in cmpt2id:
                    cmpt2id[cpt] = node_id
                    node_id = node_id + 1

                if len(cpt.children):
                    cmpt2id, node_id, edges = get_component2id(
                        cpt, cmpt2id, node_id, edges
                    )

            local_edges, cmpt2id, node_id = connectors_to_visJS(system, cmpt2id, node_id)
            edges.extend(local_edges)

            return cmpt2id, node_id, edges

        cmpt2id = dict()
        id = 1
        edges = list()
        cmpt2id, id_init, edges = get_component2id(system, cmpt2id, id, edges)

        nodes = list()
        # Add hidden node for the top system
        nodes.append(
            dict(
                label="{}".format(system.name),
                title="{} - {}".format(system.name, system.__class__.__name__),
                id=id_init,
                level=0,
                hidden=True,
                physics=False,
                group=".{}".format(system.name),
            )
        )

        # Add edges from uppest level to its children but deactivate physics on
        # them. The goal is to have them available for the hierarchical view but
        # not playing any role in the default layout.
        for child in system.children.values():
            edges.append(
                {
                    "from": id_init,
                    "to": cmpt2id[child],
                    "hidden": True,
                    "physics": False,
                }
            )

        def get_fullname(system: System) -> str:
            name = [system.name]
            parent = system
            while parent.parent is not None:
                parent = parent.parent
                name.insert(0, parent.name)

            return ".".join(name)

        groups = list()
        for c, id in cmpt2id.items():
            if isinstance(c, ExtensiblePort):
                port_name = c.name

                ref = dict(
                    title="{}.{} - {}".format(
                        c.owner.name, port_name, c.__class__.__name__
                    ),
                    id=id,
                    group="{}".format(get_fullname(c.owner)),
                    mass=2,
                )

                # Add edge to bind to central owner node
                edge = {
                    "from": id,
                    "to": cmpt2id[c.owner] if c.owner in cmpt2id else id_init,
                    "hidden": True,
                }
                if id_init in (edge["from"], edge["to"]):
                    edge["physics"] = False
                edges.append(edge)

            elif isinstance(c, Driver):
                # Add driver node
                ref = dict(
                    label=type(c).__name__,
                    title="{} - {}".format(c.name, type(c).__name__),
                    id=id,
                    level=-0.7,
                    shape="box",
                    hidden=True,
                    physics=False,
                    group="{}".format(get_fullname(c.owner)),
                )

                # Add edge to Module owner if top driver
                if c.parent is None:
                    edges.append(
                        {
                            "from": cmpt2id[c.owner] if c.owner in cmpt2id else id_init,
                            "to": cmpt2id[c],
                            "dashes": True,
                            "hidden": True,
                            "physics": False,
                        }
                    )
                else:

                    def driver_depth(driver: Driver, depth: int = 0) -> int:
                        if driver.parent is None:
                            return depth
                        else:
                            return driver_depth(driver.parent, depth + 1)

                    ref["level"] += 0.2 * driver_depth(c)

            else:
                ref = dict(
                    label="{}".format(c.name),
                    title="{}.{} - {}".format(
                        c.parent.name, c.name, c.__class__.__name__
                    ),
                    id=id,
                )

                if len(c.children) > 0:  # Component containing component => Cluster
                    ref["hidden"] = True

                    # Add edge to force binding to central system node
                    for child in c.children.values():
                        edges.append({"from": id, "to": cmpt2id[child], "hidden": True})

                ref["group"] = "{}".format(get_fullname(c.parent))

            if "level" not in ref:
                ref["level"] = 0
            ref["level"] += ref["group"].count(".") + 1  # Set hierarchical level

            nodes.append(ref)
            if "group" in ref and ref["group"] not in groups:
                groups.append(ref["group"])

        # Sort the groups by system to ensure that levels are clustered bottom-up
        groups.sort(reverse=True)

        data = {"title": system.name, "nodes": nodes, "edges": edges, "groups": groups}

        if focus_system is not None:
            data["center_id"] = cmpt2id[focus_system]

        return data

    elements = to_visJS(system)
    elements["visJS"] = vis_js
    elements["visCSS"] = vis_css

    rendered_html = template.render(elements)
    if not embeddable:
        html_begin_tags = html_begin_tags.replace("{{title}}", elements["title"])
        rendered_html = html_begin_tags + rendered_html + html_end_tags

    _, ext = os.path.splitext(filename)
    if ext != ".html":
        filename = os.extsep.join((filename, "html"))

    with open(filename, mode="w", encoding="utf-8") as f:
        f.write(rendered_html)
