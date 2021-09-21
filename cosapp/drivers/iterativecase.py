import abc
import logging
from typing import Optional

from cosapp.drivers.runonce import RunOnce

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

    __slots__ = ()

    def __init__(self,
        name: str,
        owner: "Optional[cosapp.systems.System]" = None,
        **kwargs
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
        self.reset_problem()

    @RunOnce.owner.setter
    def owner(self, system: "Optional[cosapp.systems.System]") -> None:
        # Trick to call super setter (see: https://bugs.python.org/issue14965)
        defined = self.owner is not None
        changed = self.owner is not system
        cls = IterativeCase
        super(cls, cls).owner.__set__(self, system)
        if changed:
            if defined:
                logger.warning(
                    f"System owner of Driver {self.name!r} has changed. Mathematical problem has been cleared."
                )
            self.reset_problem()

    @abc.abstractmethod
    def reset_problem(self) -> None:
        """Reset mathematical problem(s) defined on case."""
        pass

    def _postcompute(self) -> None:
        """Actions to carry out after the :py:meth:`~cosapp.drivers.runonce.RunOnce.compute` method call.

        This gathers the residues for this point and undo the variable status changes
        """
        # Request the residues of the current case to be updated
        for residue in self.get_problem().residues.values():
            residue.update()

        super()._postcompute()

    # TODO Fred do we need a get_problem here or better integration between IterativeCase -> RunOptim & RunSingleCase
