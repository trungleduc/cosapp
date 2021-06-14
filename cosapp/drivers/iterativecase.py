import logging
import numpy

from cosapp.drivers.runonce import RunOnce
from cosapp.core.variableref import VariableReference
from cosapp.core.numerics.basics import MathematicalProblem
from cosapp.core.numerics.boundary import Unknown
from cosapp.utils.graph_analysis import get_free_inputs

from typing import Union, Dict, Optional

logger = logging.getLogger(__name__)


class IterativeCase(RunOnce):
    """Abstract interface to children cases for a :py:class:`~cosapp.drivers.abstractsolver.AbstractSolver`.

    Parameters
    ----------
    name: str, optional
        Name of the `Module`
    owner : System, optional
        :py:class:`~cosapp.systems.system.System` to which this driver belong; default None
    **kwargs : Dict[str, Any]
        Optional keywords arguments
    """

    __slots__ = ('design', '_input_mapping')

    def __init__(
        self, name: str, owner: "Optional[cosapp.systems.System]" = None, **kwargs
    ) -> None:
        """Initialize a driver

        Parameters
        ----------
        name: str, optional
            Name of the `Module`
        owner : System, optional
            :py:class:`~cosapp.systems.system.System` to which this driver belong; default None
        **kwargs : Dict[str, Any]
            Optional keywords arguments
        """
        super().__init__(name, owner, **kwargs)

        self.design = MathematicalProblem(self.name, self.owner)  # type: MathematicalProblem
            # desc="Additional mathematical problem to solve for the case.",
        self._input_mapping = dict()  # type: Dict[str, VariableReference]

    @RunOnce.owner.setter
    def owner(self, value: "Optional[cosapp.systems.System]") -> None:
        # Trick to call super setter (see: https://bugs.python.org/issue14965)
        if self.owner is not value:
            if self.owner is not None:
                logger.warning(
                    f"System owner of Driver {self.name!r} has changed. Optimization equations have been cleared."
                )
            self.design = MathematicalProblem(self.design.name, value)
        super(IterativeCase, IterativeCase).owner.__set__(self, value)

    def set_iteratives(self, x: numpy.ndarray) -> int:
        """Set iteratives from the vector x.

        Parameters
        ----------
        x : numpy.ndarray
            The non-consumed new iteratives vector

        Returns
        -------
        int
            The number of value consumed.
        """
        counter = 0
        for name, unknown in self.get_problem().unknowns.items():
            if unknown.mask is None:
                unknown.set_default_value(x[counter])
                counter += 1
            else:
                n = numpy.count_nonzero(unknown.mask)
                unknown.set_default_value(x[counter : counter + n])
                counter += n
            # Set all design variables at once
            if name in self.design.unknowns:
                # Set the variable to the new x
                if not numpy.array_equal(unknown.value, unknown.default_value):
                    unknown.set_to_default()

        return counter

    def setup_run(self):
        """Method called once before starting any simulation."""
        super().setup_run()
        self._input_mapping = get_free_inputs(self.owner)

    def _postcompute(self) -> None:
        """Actions to carry out after the :py:meth:`~cosapp.drivers.runonce.RunOnce.compute` method call.

        This gathers the residues for this point and undo the variable status changes
        """
        # Request the residues of the current case to be updated
        for residue in self.get_problem().residues.values():
            residue.update()

        super()._postcompute()

    # TODO Fred do we need a get_problem here or better integration between IterativeCase -> RunOptim & RunSingleCase

    def get_free_unknown(self, unknown: Unknown, name: Optional[str]=None) -> Union[Unknown, None]:
        """Checks if `unknown` is aliased by pulling.
        If so, returns alias unknown; else, returns original unknown.
        """
        if name is None:
            name = unknown.name
        try:
            alias = self._input_mapping[name]
        except KeyError:
            logger.warning(f"Skip connected unknown {name!r}")
            return None
        if alias.mapping is not unknown.port:
            if alias.context is not self.owner:
                alias_name = f"{alias.mapping.contextual_name}.{alias.key}"
                contextual_name = f"{unknown.context.name}.{unknown.name}"
                logger.warning(
                    f"Unknown {contextual_name!r} is aliased by {alias_name!r}"
                    f", defined outside the context of {self.owner.name!r}"
                    f"; it is likely to be overwritten after the computation."
                )
            else:
                alias_name = f"{alias.mapping.name}.{alias.key}"
                logger.info(
                    f"Replace unknown {name!r} by {alias_name!r}"
                )
                unknown = unknown.transfer(alias.context, alias_name)
        return unknown
