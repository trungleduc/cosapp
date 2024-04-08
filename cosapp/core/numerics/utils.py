from __future__ import annotations
from cosapp.core.numerics.boundary import Boundary
from cosapp.core.numerics.residues import Residue
from typing import TYPE_CHECKING, Optional, Callable, Union, Iterable, TypeVar
if TYPE_CHECKING:
    from cosapp.systems import System


AnyBoundary = TypeVar("AnyBoundary", bound=Boundary)


class RelativePathFinder:
    """Path finder in a system tree relative to a reference system.

    Parameters
    ----------
    - reference [System]: root of the system tree of interest.
    """
    def __init__(self, reference: System) -> None:
        self.reference = reference
        self.sys_paths: dict[int, str] = dict()

    def get_path(self, system: System) -> str:
        """Return the relative path of `system` with respect to the reference.

        Parameters
        ----------
        - system [System]: the system to be found

        Returns
        -------
        path [str]: path from the reference system.

        Raises
        ------
        `ValueError` if `system` does not belong to the reference tree.
        """
        key = id(system)
        sys_paths = self.sys_paths
        try:
            path = sys_paths[key]
        except KeyError:
            sys_paths[key] = path = self.reference.get_path_to_child(system)
        return path


class TransferHelper:
    """Utility class to transfer unknowns from one context (system) to another.
    """
    def __init__(self, destination: System) -> None:
        self.destination = destination

    @property
    def destination(self) -> System:
        """System: destination system."""
        return self.path_finder.reference

    @destination.setter
    def destination(self, system: System):
        self.path_finder = RelativePathFinder(system)

    def transfer_unknowns(
        self,
        unknowns: Iterable[AnyBoundary],
        copy = True,
        name_wrapper: Optional[Callable[[str], str]] = None,
    ) -> dict[str, AnyBoundary]:
        """Transfer `unknowns` in the context of the destination system,
        as a dictionary indexed by unknown names.

        Parameters
        ----------
        - unknowns [Iterable[Boundary]]:
            Iterable collection of unknowns to be transfered.
        - copy [bool, optional]:
            Should the objects be copied; default is `True`.
        - name_wrapper [callable, optional]:
            Wrapping function for dictionary keys.

        Returns
        -------
        MathematicalProblem
            The resulting mathematical system
        """
        return self._transfer(
            unknowns,
            key_formatter=self.variable_key_format,
            name_wrapper=name_wrapper,
            copy=copy,
        )

    def transfer_residues(
        self,
        residues: Iterable[Residue],
        copy = True,
        name_wrapper: Optional[Callable[[str], str]] = None,
    ) -> dict[str, Boundary]:
        """Transfer `unknowns` in the context of the destination system,
        as a dictionary indexed by unknown names.

        Parameters
        ----------
        - unknowns [Iterable[Boundary]]:
            Iterable collection of unknowns to be transfered.
        - copy [bool, optional]:
            Should the objects be copied; default is `True`.
        - name_wrapper [callable[str] -> str, optional]:
            Wrapping function to be mapped on dictionary keys.

        Returns
        -------
        MathematicalProblem
            The resulting mathematical system
        """
        if name_wrapper is None:
            name_wrapper = lambda key: key

        def res_key_format(path: str, eqname: str):
            return f"{path}: {eqname}" if path else eqname

        return self._transfer(
            residues,
            key_formatter=res_key_format,
            name_wrapper=name_wrapper,
            copy=copy,
        )

    def _transfer(
        self,
        collection: Union[Iterable[Residue], Iterable[Residue]],
        key_formatter: Callable[[str, str], str],
        name_wrapper: Optional[Callable[[str], str]] = None,
        copy = True,
    ) -> dict[str, Boundary]:
        """Transfer `unknowns` in the context of the destination system,
        as a dictionary indexed by unknown names.

        Parameters
        ----------
        - unknowns [Iterable[Boundary]]:
            Iterable collection of unknowns to be transfered.
        - copy [bool, optional]:
            Should the objects be copied; default is `True`.
        - name_wrapper [callable, optional]:
            Wrapping function for dictionary keys.

        Returns
        -------
        MathematicalProblem
            The resulting mathematical system
        """
        if name_wrapper is None:
            name_wrapper = lambda key: key

        get = (lambda obj: obj.copy()) if copy else (lambda obj: obj)

        get_path = self.path_finder.get_path

        def make_key(obj: Union[Boundary, Residue]) -> str:
            """Generic key formatter for unknowns and residues."""
            path = get_path(obj.context)
            key = key_formatter(path, obj.name)
            return name_wrapper(key)

        output: dict[str, Boundary] = dict()

        for obj in collection:
            key = make_key(obj)
            output[key] = get(obj)

        return output

    @staticmethod
    def variable_key_format(path: str, varname: str) -> str:
        """Key formatter for variables/unknowns."""
        return f"{path}.{varname}" if path else varname
