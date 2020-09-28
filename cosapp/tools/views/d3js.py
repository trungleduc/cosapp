"""Build a D3js view of a System."""
import os
from typing import NoReturn

from cosapp.systems import System
from cosapp.tools import templates


def to_d3(system: System, show: bool = True, size: int = 300):
    """Returns the representation of this system in HTML format.

    Returns
    -------
    str
        HTML formatted representation
    """
    # Optional dependencies
    try:
        from jinja2 import Environment, PackageLoader
    except ImportError:
        raise ImportError("jinja2 needs to be installed to export to d3 JS.")

    def build_d3_repr(
        system: System, filename: str, embeddable: bool = False, size: int = 600
    ) -> NoReturn:
        env = Environment(
            loader=PackageLoader("cosapp.tools", "templates"),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        template = env.get_template("system_d3.html")

        # This is inspired from openmdao problem_viewer
        html_begin_tags = """<!DOCTYPE html>
<html>
<head>
<meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
<title>{{title}}</title>
</head>
<body>
<div class="flexdiv">
"""
        html_end_tags = """
</div>
</body>
</html>
"""

        ressource_dir = os.path.dirname(os.path.abspath(templates.__file__))
        libs_dir = os.path.join(ressource_dir, "lib")
        src_dir = os.path.join(ressource_dir, "src")

        # Grab the resources
        with open(os.path.join(libs_dir, "d3.min.js"), "r", encoding="utf-8") as f:
            d3_js = f.read()
        with open(os.path.join(src_dir, "d3_draw.js"), "r", encoding="utf-8") as f:
            draw_js = f.read()
        with open(os.path.join(src_dir, "d3_styles.css"), "r", encoding="utf-8") as f:
            d3_styles = f.read()

        def build_d3_json(
            system: System, parent_name: str = None, nlevels: int = 0
        ) -> dict:

            if parent_name is None:
                full_name = system.name
            else:
                full_name = ".".join((parent_name, system.name))

            if nlevels == 0:  # Number of System levels
                def get_level(syst) -> int:
                    if len(syst.children) == 0:
                        return 1
                    else:
                        return 1 + max(map(get_level, syst.children.values()))

                nlevels = get_level(system)

            level = len(full_name.split("."))

            rtn = {"name": system.name, "full_name": full_name}
            if system.children:
                rtn["children"] = [
                    build_d3_json(system[child], parent_name=full_name, nlevels=nlevels)
                    for child in system.exec_order
                ]
            else:
                rtn["size"] = (nlevels / level) ** 5

            return rtn

        elements = {
            "title": "my viewer",
            "model_data": "var modelData = {!s}".format(build_d3_json(system)),
            "d3_styles": d3_styles,
            "d3JS": d3_js,
            "draw": draw_js,
        }

        rendered_html = template.render(elements)
        if not embeddable:
            html_begin_tags = html_begin_tags.replace("{{title}}", elements["title"])
            rendered_html = html_begin_tags + rendered_html + html_end_tags

        _, ext = os.path.splitext(filename)
        if ext != ".html":
            filename = os.extsep.join((filename, "html"))

        with open(filename, mode="w", encoding="utf-8") as f:
            f.write(rendered_html)

    temp_name = ".".join((system.name, "html"))
    build_d3_repr(system, temp_name)

    if show:
        # Optional dependencies
        try:
            from IPython.display import IFrame
        except ImportError:
            raise ImportError("IPython needs to be installed to display the d3 figure.")
        return IFrame(temp_name, "99.9%", "435px")
    else:
        return None

