"""
Tool to print the description of a CoSApp object.
"""
from typing import Union, Any

from cosapp.core.module import Module
from cosapp.ports.enum import PortType
from cosapp.ports.port import ExtensiblePort, Port
from cosapp.drivers.driver import Driver
from cosapp.systems.system import System


class DocDisplay:
    """Helper class to print nicely information about CoSApp classes.
    """

    # TODO unit tests + improve to have more common code with cosapp.notebook.display_doc & get_list_out/inputs

    def __init__(self, object_type: Union[type, Any]):
        """DocDisplay constructor.

        Documentation can only be built for object of type :py:class:`~cosapp.drivers.driver.Driver`,
        :py:class:`~cosapp.ports.port.Port` or :py:class:`~cosapp.systems.system.System`.

        Parameters
        ----------
        object_type : Any
            Class type of the object to display.
        """
        if not (
            isinstance(object_type, (ExtensiblePort, Module))
            or issubclass(object_type, (ExtensiblePort, Module))
        ):
            raise TypeError("Only Driver, Port and System are supported for display.")

        if isinstance(object_type, type):  # Instantiate an object
            if issubclass(object_type, Port):
                object_type = object_type("dummy", PortType.OUT)
            else:
                object_type = object_type("dummy")

        self._obj = object_type

    def __repr__(self) -> str:
        """Returns the documentation of the class given to this constructor
        object.

        Returns
        -------
        str
            Documentation string.
        """
        return self._repr_markdown_()

    def _repr_markdown_(self) -> str:
        """Returns the documentation of the class given to this constructor
        object.

        Returns
        -------
        str
            Documentation markdown formatted string.
        """
        object_type = type(self._obj)

        indent = 0
        stripped_doc = ["Class: {}".format(object_type.__name__), ""]
        if object_type.__doc__:
            stripped_doc.extend(["  Documentation", ""])
            for l in object_type.__doc__.split("\n"):
                if indent == 0:
                    stripped_l = l.lstrip()
                    if len(stripped_l):
                        indent = len(l) - len(stripped_l)
                else:
                    stripped_l = l[indent:]
                stripped_doc.append(stripped_l)

        if isinstance(self._obj, Module):
            if isinstance(self._obj, System):
                self._obj.run_once()
            stripped_doc.append(self._obj._repr_markdown_())

        elif isinstance(self._obj, Port):
            if len(self._obj):
                stripped_doc.extend(["", "  Variables", ""])
                stripped_doc.append(self._obj._repr_markdown_())

        return "\n".join(stripped_doc)

    @staticmethod
    def display_doc(object_type: Union[type, Any]) -> "DocDisplay":
        """Display information for `Driver`, `System` or `Port` in a Jupyter notebook.

        Parameters
        ----------
        object_type : Any
            Object of interest
        """
        return DocDisplay(object_type)


try:
    # If cosapp.notebook package is available use it
    from cosapp.notebook import display_doc
except ImportError:
    display_doc = DocDisplay.display_doc
