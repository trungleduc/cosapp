"""Build a VISjs view of a System."""

import os
from typing import Dict, List, Optional, Tuple, Union

from cosapp.drivers import Driver
from cosapp.ports.port import BasePort, ExtensiblePort
from cosapp.systems import System
from cosapp.tools import templates

from .baseRenderer import BaseRenderer


class VisJsRenderer(BaseRenderer):
    """
    Utility class to export a system as HTML using vis.JS library.

    Parameters
    ----------
    system : System
        System to export
    embeddable: bool, optional
        Is the HTML to be embedded in an existing page? Default: False
    """

    __globals = None

    def __init__(self, system: System, embeddable=False):
        super().__init__(system, embeddable)

    @staticmethod
    def get_system_label(system: System):
        label = system.name
        lmax = 7
        if len(label) > lmax:
            dots = "\u2026"  # single character
            label = label[:lmax-1] + dots
        return label

    def get_system_title(self, system: System):
        head = self.system
        if system is head:
            name = head.name
        else:
            name = head.get_path_to_child(system)
        return f"{name} - {system.__class__.__name__}"

    def get_system_group(self, system: System):
        head = self.system
        if system is head:
            name = f".{head.name}"
        else:
            path = head.get_path_to_child(system.parent)
            name = f"{head.name}.{path}" if path else head.name
        return name

    def get_data(self) -> Dict:
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
        head = self.system

        def get_component2id(
            system: System,
            cmpt2id: Dict[Union[System, BasePort], int],
            node_id: int,
            edges: List[Dict],
        ) -> Tuple[Dict[Union[System, BasePort], int], int, List[Dict]]:
            """Set the mapping between the `Module` and their unique id.

            Parameters
            ----------
            system : Module
                The parent `Module` for which id must be set
            cmpt2id : Dict[Union[Module, BasePort], int]
                Mapping of `Module` or port and its id
            node_id : int
                Latest used id
            edges : List[Dict]
                List of connection edges

            Returns
            -------
            tuple (List[Dict], Dict[Union["Module", "BasePort"], int], int)
                The first is the mapping between each involved system or port and an unique id,
                the second is the latest used id and
                the third index is the list of edges (one per connector).
            """

            def get_driver2id(driver: Driver, cmpt2id: dict, node_id: int, edges: List[dict]):
                if driver not in cmpt2id:
                    cmpt2id[driver] = node_id
                    node_id += 1

                previous_driver = None
                for subdriver in driver.children.values():
                    cmpt2id, node_id, edges = get_driver2id(
                        subdriver, cmpt2id, node_id, edges
                    )
                    # Connection between same level drivers
                    if previous_driver is not None: 
                        edges.append(
                            {
                                "from": cmpt2id[previous_driver],
                                "to": cmpt2id[subdriver],
                                "arrows": "to",
                                "title": "",
                                "hidden": True,
                                "physics": False,
                            }
                        )
                    previous_driver = subdriver
                    # Connection with parent driver
                    edges.append(
                        {
                            "from": cmpt2id[driver],
                            "to": cmpt2id[subdriver],
                            "title": "",
                            "hidden": True,
                            "physics": False,
                        }
                    )

                return cmpt2id, node_id, edges

            def connectors_to_visJS(
                system: System,
                component_id: Optional[Dict[Union[System, BasePort], int]] = None,
                node_id: int = 1,
            ) -> Tuple[List[Dict], Dict[Union[System, BasePort], int], int]:
                """Build the dictionary the list of edges.

                Parameters
                ----------
                system : System
                    System owning the connectors
                component_id : Dict[Union[System, Port], int], optional
                    Mapping of `System` or `Port` and its unique id; default None
                node_id : int, optional
                    Latest used id; default is 1.

                Returns
                -------
                tuple (List[Dict], Dict[System | BasePort, int], int)
                    First element is the mapping between each involved system/port and a unique id.
                    second element is the latest id used;
                    Third element is the list of edges (one per connector);
                """
                edges = list()

                for connection in system.all_connectors():
                    supplier = connection.source.owner
                    if supplier.children:  # Insert port as node
                        supplier = connection.source

                    if supplier not in component_id:
                        component_id[supplier] = node_id
                        node_id += 1

                    target = connection.sink.owner
                    if target.children:  # Insert port as node
                        target = connection.sink

                    if target not in component_id:
                        component_id[target] = node_id
                        node_id += 1

                    edge = {
                        "from": component_id[supplier],
                        "to": component_id[target],
                        "arrows": "to",
                        "title": "",
                    }
                    if connection.is_mirror():
                        edge["title"] = type(connection.source).__name__
                    else:
                        edge["title"] = f"[{connection.pretty_mapping()}]"

                    # if supplier or target is top-system, set edge.length = 0
                    if (
                        connection.source.owner is connection.sink.owner.parent
                        or connection.source.owner.parent is connection.sink.owner
                    ):
                        edge["length"] = 0

                    edges.append(edge)

                return edges, component_id, node_id

            for driver in system.drivers.values():
                cmpt2id, node_id, edges = get_driver2id(driver, cmpt2id, node_id, edges)

            for child in system.children.values():
                if child not in cmpt2id:
                    cmpt2id[child] = node_id
                    node_id += 1

                if child.children:
                    cmpt2id, node_id, edges = get_component2id(child, cmpt2id, node_id, edges)

            local_edges, cmpt2id, node_id = connectors_to_visJS(system, cmpt2id, node_id)
            edges.extend(local_edges)

            return cmpt2id, node_id, edges

        cmpt2id = dict()
        id = 1
        edges = list()
        cmpt2id, id_init, edges = get_component2id(head, cmpt2id, id, edges)

        nodes = list()
        # Add hidden node for the top system
        nodes.append(
            dict(
                name=head.name,
                label=self.get_system_label(head),
                title=self.get_system_title(head),
                id=id_init,
                level=0,
                hidden=True,
                physics=False,
                group=f".{head.name}",
            )
        )

        # Add edges from uppest level to its children but deactivate physics on
        # them. The goal is to have them available for the hierarchical view but
        # not playing any role in the default layout.
        for child in head.children.values():
            edges.append(
                {
                    "from": id_init,
                    "to": cmpt2id[child],
                    "hidden": True,
                    "physics": False,
                }
            )

        groups = list()

        for c, id in cmpt2id.items():

            if isinstance(c, BasePort):
                ref = dict(
                    title=f"{c.full_name(trim_root=True)}",
                    id=id,
                    group=head.get_path_to_child(c.owner, trim_top=False),
                    mass=2,
                )
                if not isinstance(c, ExtensiblePort):
                    ref["title"] += f" - {type(c).__name__}"

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
                    title=f"{c.name} - {type(c).__name__}",
                    id=id,
                    level=-0.7,
                    shape="box",
                    hidden=True,
                    physics=False,
                    group=head.get_path_to_child(c.owner, trim_top=False),
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
                    driver_depth = len(list(c.path_to_root())) - 1
                    ref["level"] += 0.2 * driver_depth

            else:
                # c is a System
                ref = dict(
                    name=c.name,
                    label=self.get_system_label(c),
                    title=self.get_system_title(c),
                    id=id,
                    widthConstraint={'maximum': '10px'},
                )

                if c.children:  # Component containing component => Cluster
                    ref["hidden"] = True

                    # Add edge to force binding to central system node
                    for child in c.children.values():
                        edges.append({"from": id, "to": cmpt2id[child], "hidden": True})

                ref["group"] = self.get_system_group(c)

            group = ref["group"]
            ref.setdefault("level", 0)
            ref["level"] += group.count(".") + 1  # Set hierarchical level

            nodes.append(ref)
            if group not in groups:
                groups.append(group)

        # Sort the groups by system to ensure that levels are clustered bottom-up
        groups.sort(reverse=True)

        data = {
            "title": self.get_system_label(head),
            "nodes": nodes,
            "edges": edges,
            "groups": groups,
        }

        return data

    def html_content(self) -> str:
        """Returns vis.JS HTML content of renderer"s system as a character string."""
        elements = self.get_data()
        common = self.get_globals()
        elements["visJS"] = common["visJS"]
        elements["visCSS"] = common["visCSS"]
        html_begin_tags = common["html_begin_tags"]
        html_end_tags = common["html_end_tags"]
        template = common["template"]
        
        content = template.render(elements)
        if not self.embeddable:
            html_begin_tags = html_begin_tags.replace("{{title}}", elements["title"])
            content = html_begin_tags + content + html_end_tags

        return content

    @classmethod
    def html_resources(cls) -> Dict[str, str]:
        """
        Return a dictionary of two keys.

        - `visJS` key holds the content of vis.js library.
        - `visCSS` key holds the css style of plot.

        Returns
        -------
        Dict[str, Dict]
            The necessary resources to render Jinja template
        """
        ressource_dir = os.path.dirname(os.path.abspath(templates.__file__))
        libs_dir = os.path.join(ressource_dir, "lib")

        with open(os.path.join(libs_dir, "vis.min.js"), "rb") as f:
            vis_js = f.read().decode("utf-8")
        with open(os.path.join(libs_dir, "vis.min.css"), "rb") as f:
            vis_css = f.read().decode("utf-8")

        return {"visJS": vis_js, "visCSS": vis_css}
    
    @classmethod
    def html_template(cls):
        """
        Return the Jinja template used to create HTML file.

        Returns
        -------
        Template
        """
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
        return template
    
    @classmethod
    def get_globals(cls):
        cg = cls.__globals
        if cg is None:
            cg = super().get_globals()
        return cg


def visjs_html(system: System, embeddable=False) -> str:
    """Returns vis.JS HTML content of `system`.

    Parameters
    ----------
    system : System
        System to export

    embeddable: bool, optional
        Is the HTML to be embedded in an existing page? Default: False
    """
    renderer = VisJsRenderer(system, embeddable)
    return renderer.html_content()


def to_visjs(system: System, filename: str, embeddable=False) -> None:
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
    ext = os.path.splitext(filename)[1]
    html_ext = f"{os.extsep}html"
    if ext != html_ext:
        filename += html_ext

    rendered_html = visjs_html(system, embeddable)
    with open(filename, mode="w", encoding="utf-8") as f:
        f.write(rendered_html)
