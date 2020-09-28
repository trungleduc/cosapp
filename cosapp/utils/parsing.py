from typing import Optional, Tuple
from collections.abc import Collection

import numpy

from .helpers import check_arg


# TODO should we reintegrate it as Boundary method as it is now the only used place
def get_indices(
    syst: "cosapp.core.module.Module", name: str
) -> Tuple[str, Optional[numpy.ndarray]]:
    """Decompose a variable specification into its name and its mask.

    Parameters
    ----------
    syst : Module
        Module to which belong the variable
    name : str
        Variable specification

    Returns
    -------
    Tuple[str, Optional[numpy.ndarray[bool]]]
        Tuple (variable name, variable mask or None)
    """
    if "[" in name:  # User want to use a part of a sequence
        var_name, selector = name.split("[", maxsplit=1)
        selector = "[" + selector
    else:
        var_name = name
        selector = None

    if var_name not in syst:
        raise AttributeError(
            "Variable {} is not known in system {}.".format(var_name, syst.name)
        )

    value = syst[var_name]
    if selector:
        # Check value is an array
        if not (
            isinstance(value, numpy.ndarray)
            and numpy.issubdtype(value.dtype, numpy.number)
            and value.size > 1
        ):
            raise TypeError(
                "Only non-empty numpy array can be partially selected; got '{}.{}'.".format(
                    syst.name, name
                )
            )
        mask = numpy.zeros(value.shape, dtype=bool)
        # Set the mask using the selector
        try:
            exec("mask{}=True".format(selector), {}, {"mask": mask})
        except (SyntaxError, IndexError) as err:
            raise type(err)(
                'Selection "{}" for variable "{}.{}" is not valid: {}'
                "".format(selector, syst.name, var_name, str(err))
            )
    else:
        mask = None
        if isinstance(value, Collection) and not isinstance(value, str):
            mask = numpy.ones(numpy.asarray(value).shape, dtype=bool)

    return var_name, mask
