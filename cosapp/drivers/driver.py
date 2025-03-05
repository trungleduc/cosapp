"""
Classes driving simulation on CoSApp :py:class:`~cosapp.systems.system.System`.
"""
import json
import logging
import time
from typing import Optional, TypeVar, Union, List, Dict, Any, Callable

from cosapp.patterns.visitor import Visitor
from cosapp.core.module import Module
from cosapp.systems import System
from cosapp.recorders.recorder import BaseRecorder
from cosapp.utils.options_dictionary import HasCompositeOptions, OptionsDictionary
from cosapp.utils.naming import NameChecker, CommonPorts
from cosapp.utils.helpers import check_arg
from cosapp.utils.json import jsonify

logger = logging.getLogger(__name__)

AnyDriver = TypeVar("AnyDriver", bound="Driver")
AnyRecorder = TypeVar("AnyRecorder", bound=BaseRecorder)


class Driver(Module, HasCompositeOptions):
    """Abstract base class for all systems drivers.

    Parameters
    ----------
    name : str
        Name of the driver
    owner : System, optional
        :py:class:`~cosapp.systems.system.System` to which this driver belongs; default None
    **kwargs : Any
        Keyword arguments used to set driver options

    Attributes
    ----------
    name : str
        Name of the driver
    parent : Driver, optional
        Top driver containing this driver; default None.
    owner : System
        :py:class:`~cosapp.systems.system.System` to which this driver belong.
    children : OrderedDict[str, Driver]
        Drivers belonging to this one.

    options : OptionsDictionary
      |  Options for the current driver:
      |  **verbose** : int, {0, 1}
      |      Verbosity level of the driver; default 0 (i.e. minimal information)

    solution :  List[Tuple[str, float]]
        List of (name, value) for the iteratives when a solution is reached
    """

    __slots__ = ('_owner', '_recorder', '_options', 'start_time', 'status', 'error_code')

    _name_check = NameChecker(
        pattern = r"^[A-Za-z][\w\s@-]*[\w]?$",
        message = "Driver name must start with a letter, and contain only alphanumerics + {'_', '@', ' ', '-'}",
        excluded = CommonPorts.names(),
    )

    def __init__(
        self,
        name: str,
        owner: Optional[System] = None,
        **options
    ) -> None:
        """Initialize driver

        Parameters
        ----------
        - name [str]:
            Driver name
        - owner [System, optional]:
            :py:class:`~cosapp.systems.system.System` to which driver belongs; defaults to `None`.
        - **kwargs:
            Optional keywords arguments.
        """
        super().__init__(name)
        HasCompositeOptions.__init__(self)

        self._owner: Optional[System] = None
        self._recorder: Optional[BaseRecorder] = None
        self.owner = owner

        self.start_time = 0.0  # type: float
            # unit="s",
            # desc="Absolute time at which the Driver execution started.",
        self.status = ""  # type: str
            #desc="Status of the driver."
            # TODO Fred what are the status? Enum, any str?
        self.error_code = "0"  # type: str
            # desc="Error code during the execution."
            # TODO Fred what is the code? ESI?

        self._init_options(options)
        if options:
            raise RuntimeError(
                f"Unknown option(s) {list(options.keys())!r} for {type(self).__name__}"
                f"; available options are: {self.available_options(0)}."
            )

    def _declare_options(self) -> None:
        """Declares options."""
        super()._declare_options()
        self._options.declare(
            "verbose",
            default=0,
            dtype=int,
            lower=0,
            upper=1,
            desc="Verbosity level of the driver",
        )

    @property
    def options(self) -> OptionsDictionary:
        """Gets options."""
        return self._options

    def __getstate__(self) -> tuple[None, Dict[str, Any]]:
        """Creates a state of the object.

        The state type depend on the object, see
        https://docs.python.org/3/library/pickle.html#object.__getstate__
        for further details.
        
        Returns
        -------
        tuple[None, Dict[str, Any]]:
            state
        """
        _, slots = super().__getstate__()

        for slot in ("setup_ran", "computed", "clean_ran"):
            slots.pop(slot)

        return None, slots

    def __setstate__(self, state: tuple[None, Dict[str, Any]]) -> None:
        """Sets the object from a provided state.

        Parameters
        ----------
        state : tuple[None, Dict[str, Any]]
            State
        """
        _, slots = state

        for name, value in slots.items():
            setattr(self, name, value)

    def __reduce_ex__(self, _: Any) -> tuple[Callable, tuple, dict]:
        """Defines how to serialize/deserialize the object.
        
        Parameters
        ----------
        _ : Any
            Protocol used

        Returns
        -------
        tuple[Callable, tuple, dict]
            A tuple of the reconstruction method, the arguments to pass to
            this method, and the state of the object
        """
        state = self.__getstate__()
        return self._from_state, (self.name,), state

    @classmethod
    def _from_state(cls, name):
        return cls(name)

    @classmethod
    def _slots_not_jsonified(cls) -> tuple[str]:
        """Returns slots that must not be JSONified."""
        return ("_owner", "parent")

    def __json__(self) -> Dict[str, Any]:
        """Creates a JSONable dictionary representation of the object.
        
        Break circular dependencies by removing some slots from the 
        state.

        Returns
        -------
        Dict[str, Any]
            The dictionary
        """
        _, slots = self.__getstate__()
        for s in self._slots_not_jsonified():
            slots.pop(s)

        return jsonify(slots)
    
    def to_json(self,*,
        indent: Optional[int] = None,
        sort_keys: bool = True)->str:
        return json.dumps(self.__json__(), indent=indent, sort_keys=sort_keys)

    def available_options(self, level=1) -> Union[List[str], Dict[str, Any], None]:
        """Prints out the driver options.
        
        Parameters
        ----------
        level: int, optional
            Level of detail displayed:
            0: show option names only;
            1: show options as a (name, value) dictionary (default);
            2: print out a detailed description of the options.
        
        Returns
        -------
        list[str] | dict[str, str] | None, depending on level.
        """
        if level == 0:
            return list(self.options)
        elif level == 1:
            return dict(self.options.items())
        else:
            print(self.options.__str__(width=128))

    def accept(self, visitor: Visitor) -> None:
        """Specifies course of action when visited by `visitor`"""
        visitor.visit_driver(self)

    def __repr__(self) -> str:
        context = "alone" if self.owner is None else f"on System {self.owner.name!r}"
        return f"{self.name} ({context}) - {self.__class__.__name__}"

    @property
    def owner(self) -> System:
        """System: System owning the driver and its children."""
        return self._owner

    @owner.setter
    def owner(self, system: Optional[System]) -> None:
        self._set_owner(system)

    def _set_owner(self, system: Union[System, None]) -> bool:
        """Owner setter as a protected method, to be used by derived classes.
        This prevents from calling base class `owner.setter`, which can be tricky.

        Parameters
        ----------
        system: System | None
            Owner system (or None).

        Returns
        -------
        changed [bool]:
            `True` if owner has changed, `False` otherwise.
        """
        if system is not None:
            check_arg(system, 'owner', System)
        
        changed = system is not self._owner
        self._owner: Optional[System] = system
        if self._recorder is not None:
            self._recorder.watched_object = system
        for child in self.children.values():
            child.owner = system

        return changed

    def check_owner_attr(self, item: str) -> None:
        if item not in self.owner:
            raise AttributeError(f"{item!r} not found in System {self.owner.name!r}")

    @property
    def recorder(self) -> Optional[BaseRecorder]:
        """BaseRecorder or None : Recorder attached to this `Driver`."""
        return self._recorder

    def is_standalone(self) -> bool:
        """Is this Driver able to solve a system?

        Returns
        -------
        bool
            Ability to solve a system or not.
        """
        for driver in self.children.values():
            if driver.is_standalone():
                return True

        return False

    def _set_children_active_status(self, active_status : bool) -> None:
        self._active = active_status
        for child in self.children.values():
            child._set_children_active_status(active_status)

    def setup_run(self) -> None:
        """Set execution order and start the recorder."""
        if self.owner is None:
            raise AttributeError(f"Driver {self.name!r} has no owner system.")

    def _precompute(self) -> None:
        """Set execution order and start the recorder."""
        self.start_time = time.time()

        if self.owner.parent is None and self.parent is None:
            logger.info(" " + "-" * 60)
            logger.info(f" # Starting driver {self.name!r} on {self.owner.name!r}")

        if self._recorder is not None:
            self._recorder.start()

    def _postcompute(self) -> None:
        """Actions performed after the `Module.compute` call."""
        if self._recorder is not None:
            self._recorder.exit()  # TODO Fred Better in clean_run ?

        if self.owner.parent is None and self.parent is None:
            logger.info(
                " # Ending driver {!r} on {!r} in {} seconds\n".format(
                    self.name, self.owner.name, round(time.time() - self.start_time, 3)
                )
            )

    def add_child(self, child: AnyDriver, execution_index: Optional[int]=None, desc="") -> AnyDriver:
        """Add a child `Driver` to the current `Driver`.

        When adding a child `Driver`, it is possible to specified its position in the execution
        order.

        Parameters
        ----------
        - child: Driver
            `Driver` to add to the current `Driver`
        - execution_index: int, optional
            Index of the execution order list at which the `Module` should be inserted;
            default latest.
        - desc [str, optional]:
            Sub-driver description in the context of its parent driver.

        Returns
        -------
        `child`

        Notes
        -----
        The added child will have its owner set to that of current driver.
        """
        check_arg(child, 'child', Driver)
        child.owner = self.owner

        return super().add_child(child, execution_index, desc)

    def add_driver(self, child: AnyDriver, execution_index: Optional[int]=None, desc="") -> AnyDriver:
        """Alias for :py:meth:`~cosapp.drivers.driver.Driver.add_child`."""
        return self.add_child(child, execution_index, desc)

    def add_recorder(self, recorder: AnyRecorder) -> AnyRecorder:
        check_arg(recorder, 'recorder', BaseRecorder)

        self._recorder = recorder
        self._recorder.watched_object = self.owner
        self._recorder._owner = self

        return self.recorder
