"""
Tool to print the description of a CoSApp object.
"""
from typing import Union

from cosapp.core.module import Module
from cosapp.ports.enum import PortType
from cosapp.ports.port import BasePort


class DocDisplay:
    """Helper class to print nicely information about CoSApp classes."""

    # TODO unit tests

    def __init__(self, obj: Union[type, BasePort, Module]):
        """DocDisplay constructor.

        Documentation can only be built for object of type :py:class:`~cosapp.drivers.driver.Driver`,
        :py:class:`~cosapp.ports.port.Port` or :py:class:`~cosapp.systems.system.System`.

        Parameters
        ----------
        obj: BasePort or Module class, or instance thereof
            Class type of the object to display, or instance of such class.
        """
        supported = (BasePort, Module)
        if not (isinstance(obj, supported) or issubclass(obj, supported)):
            raise TypeError("Only Driver, Port and System are supported for display.")

        if isinstance(obj, type):  # Instantiate an object
            if issubclass(obj, BasePort):
                obj = obj("dummy", PortType.OUT)
            else:
                obj = obj("dummy")

        self._obj = obj

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
        obj = self._obj
        obj_type = type(self._obj)

        indent = 0
        doc = [f"## Class: {obj_type.__name__}", ""]
        if obj_type.__doc__:
            doc.extend(["### Documentation", ""])
            for line in obj_type.__doc__.split("\n"):
                if indent == 0:
                    stripped = line.lstrip()
                    if len(stripped) > 0:
                        indent = len(line) - len(stripped)
                else:
                    stripped = line[indent:]
                doc.append(stripped)

        if isinstance(obj, Module):
            doc.append(obj._repr_markdown_())

        elif isinstance(obj, Port):
            if len(obj) > 0:
                doc.extend(["", "###  Variables", ""])
                doc.append(obj._repr_markdown_())

        return "\n".join(doc)

    @classmethod
    def display_doc(cls, obj: Union[type, BasePort, Module]) -> "DocDisplay":
        """Display information for `Driver`, `System` or `Port` in a Jupyter notebook.

        Parameters
        ----------
        obj: Any
            Object of interest
        """
        return cls(obj)


display_doc = DocDisplay.display_doc
