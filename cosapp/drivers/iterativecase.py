import logging
from typing import NoReturn

import numpy

from cosapp.drivers.runonce import RunOnce
from cosapp.core.numerics.basics import MathematicalProblem

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

    __slots__ = ('design')

    def __init__(
        self, name: str, owner: "Optional[cosapp.systems.System]" = None, **kwargs
    ) -> NoReturn:
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

    @RunOnce.owner.setter
    def owner(self, value: "Optional[cosapp.systems.System]") -> NoReturn:
        # Trick to call super setter (see: https://bugs.python.org/issue14965)
        if self.owner is not value:
            if self.owner is not None:
                logger.warning(
                    "System owner of Driver '{}' has changed. Optimization equations have been cleared.".format(
                        self.name
                    )
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

    def _postcompute(self) -> NoReturn:
        """Actions to carry out after the :py:meth:`~cosapp.drivers.runonce.RunOnce.compute` method call.

        This gathers the residues for this point and undo the variable status changes
        """
        # Request the residues of the current case to be updated
        for residue in self.get_problem().residues.values():
            residue.update()

        super()._postcompute()

    # TODO Fred do we need a get_problem here or better integration between IterativeCase -> RunOptim & RunSingleCase
