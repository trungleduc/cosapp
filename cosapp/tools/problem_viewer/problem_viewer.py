import os
import json

from typing import Any, Dict, Union

from collections import OrderedDict
import logging
import base64

from cosapp.systems import System
from cosapp.ports.port import BasePort, ExtensiblePort

from cosapp.utils.helpers import check_arg

logger = logging.getLogger(__name__)


def _get_tree_dict(
    obj: Union[System, BasePort],
    include_orphan_vars=True
) -> Dict[str, Any]:
    """Get a dictionary representation of the system hierarchy.
    """
    def init_tree(obj, subtype):
        return OrderedDict(
            name = obj.name,
            type = 'subsystem',
            subsystem_type = subtype,
        )

    def port_tree(port):
        tree_dict = init_tree(port, 'component')
        tree_dict['children'] = children = list()

        if include_orphan_vars or not isinstance(port, ExtensiblePort):
            dtype = 'param' if port.is_input else 'unknown'
            children.extend(
                map(lambda n: OrderedDict(name=n, type=dtype), iter(port))
            )
        return tree_dict

    def system_tree(system):
        tree_dict = init_tree(system, 'group')
        tree_dict['children'] = children = list()

        def add_ports(ports):
            nonlocal children
            children.extend(port_tree(port)
                for port in filter(lambda p: len(p) > 0, ports)
            )

        add_ports(system.inputs.values())

        children.extend(system_tree(child)
            for child in system.children.values()
        )
        # outputs come after child list for a matter of visualization
        add_ports(system.outputs.values())
        
        return tree_dict

    if isinstance(obj, System):
        return system_tree(obj)
    elif isinstance(obj, BasePort):
        return port_tree(obj)
    else:
        raise TypeError(f"Object must be a port or a system; got {type(obj)}.")


def _get_connections(system: System, include_orphan_vars: bool) -> Dict[str, str]:
    """Get a dictionary representation of the system connections. Structure is {'_in': '_out'}."""
    connections = dict()

    def contextual_name(port: BasePort):
        return port.name if port.owner is system else port.contextual_name

    def add_prefix(name: str, varname: str):
        prefix = f"{name}."
        return varname if varname.startswith(prefix) else f"{prefix}{varname}"

    # Gather connections for current system
    for connector in system.all_connectors():
        if not include_orphan_vars and isinstance(connector.sink, ExtensiblePort):
            continue
        sink_name = contextual_name(connector.sink)
        source_name = contextual_name(connector.source)
        for target, origin in connector.mapping.items():
            connections[f"{sink_name}.{target}"] = f"{source_name}.{origin}"

    # Recursively gather children's connections
    for name, child in system.children.items():
        child_connections = _get_connections(child, include_orphan_vars)
        for k, v in child_connections.items():
            connections[add_prefix(name, k)] = add_prefix(name, v)

    return connections


def _get_viewer_data(system: System, include_orphan_vars: bool) -> Dict[str, str]:
    """Get the data needed by the N2 viewer as a dictionary."""
    if isinstance(system, System):
        head = system
    else:
        raise TypeError('get_model_viewer_data only accepts System')

    if not include_orphan_vars:
        logger.warning('The system may contain inwards or outwards.')

    tree_data =_get_tree_dict(head, include_orphan_vars)
    connections = _get_connections(head, include_orphan_vars)
    data_dict = {
        'tree': tree_data,
        'connections_list' : [
            dict(src=origin, tgt=target)
            for target, origin in connections.items()
            if origin is not None
        ],
    }
    return data_dict


def view_model(
    problem_or_filename,
    outfile='n2.html',
    show_browser=True,
    embeddable=False,
    draw_potential_connections=False,
    include_orphan_vars=True,
) -> None:
    """
    Generates an HTML file containing a tree viewer. Optionally pops up a web browser to
    view the file.

    Parameters
    ----------
    problem_or_filename : A System
        System : The System for the desired tree.

    outfile : str, optional
        The name of the final output file

    show_browser : bool, optional
        If True, pop up the system default web browser to view the generated html file.
        Defaults to True.

    embeddable : bool, optional
        If True, gives a single HTML file that doesn't have the <html>, <DOCTYPE>, <body>
        and <head> tags. If False, gives a single, standalone HTML file for viewing.

    draw_potential_connections : bool, optional
        If true, allows connections to be drawn on the N2 that do not currently exist
        in the model. Defaults to True.

    include_orphan_vars : bool, optional
    If True, display inwards and outwards on the N2 diagram. Defaults to True.
    """
    def add_html_tags(content: str) -> str:
        """Format `content` into standalone HTML content."""
        html_begin_tags = """<!DOCTYPE html>
<html>
    <head>
        <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
    </head>
    <body>\n"""
        html_end_tags = """
    </body>
</html>"""
        return html_begin_tags + content + html_end_tags

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    viz_dir = os.path.join(cur_dir, "visualization")
    lib_dir = os.path.join(viz_dir, "libs")
    src_dir = os.path.join(viz_dir, "src")
    sty_dir = os.path.join(viz_dir, "style")

    def get_content(path: str, filename: str, binary=False) -> str:
        fullpath = os.path.join(path, filename)
        if binary:
            with open(fullpath, "rb") as fp:
                return str(base64.b64encode(fp.read()).decode("ascii"))
        else:
            with open(fullpath, "r") as fp:
                return fp.read()

    def get_lib(filename: str, binary=False) -> str:
        return get_content(lib_dir, filename, binary)

    def get_src(filename: str, binary=False) -> str:
        return get_content(src_dir, filename, binary)

    def get_style(filename: str, binary=False) -> str:
        return get_content(sty_dir, filename, binary)

    # Get index.html
    index = get_content(viz_dir, "index.html")

    # Get model viewer data
    data = _get_viewer_data(problem_or_filename, include_orphan_vars)
    model_viewer_data = f"var modelData = {json.dumps(data)}"

    # Put all style and JS into index
    index = index.replace("{{awesomplete_style}}", get_style("awesomplete.css"))
    index = index.replace("{{partition_tree_style}}", get_style("partition_tree.css"))
    index = index.replace("{{fontello}}", get_style("fontello.woff", binary=True))
    index = index.replace("{{d3_lib}}", get_lib("d3.v4.min.js"))
    index = index.replace("{{awesomplete_lib}}", get_lib("awesomplete.js"))
    index = index.replace("{{vk_beautify_lib}}", get_lib("vkBeautify.js"))
    index = index.replace("{{model_data}}", model_viewer_data)
    index = index.replace("{{constants_lib}}", get_src("constants.js"))
    index = index.replace("{{modal_lib}}", get_src("modal.js"))
    index = index.replace("{{svg_lib}}", get_src("svg.js"))
    index = index.replace("{{search_lib}}", get_src("search.js"))
    index = index.replace("{{legend_lib}}", get_src("legend.js"))
    index = index.replace("{{draw_lib}}", get_src("draw.js"))
    index = index.replace("{{ptn2_lib}}", get_src("ptN2.js"))
    index = index.replace(
        "{{draw_potential_connections}}",
        'true' if draw_potential_connections else 'false',
    )

    if not embeddable:
        index = add_html_tags(index)

    with open(outfile, 'w') as fp:
        fp.write(index)

    # Open in browser, if required
    if show_browser:
        from cosapp.tools.problem_viewer.webview import webview
        webview(outfile)
