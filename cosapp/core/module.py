"""
Basic class handling model tree structure.
"""
import abc
import collections
import logging
import re
from typing import Any, Dict, List, NoReturn, Optional, Union

from cosapp.core.signal import Signal
from cosapp.utils.naming import NameChecker
from cosapp.utils.helpers import check_arg
from cosapp.utils.logging import LoggerContext, LogFormat, LogLevel
from cosapp.utils.orderedset import OrderedSet

logger = logging.getLogger(__name__)


class Module(LoggerContext, metaclass=abc.ABCMeta):
    # class Module does not inherit directly from abc.ABC, due to 
    # a bug in Python 3.6 preventing the use of __weakref__ in slots.
    # Ref: https://bugs.python.org/issue30463
    # Here, use of abc.ABCMeta makes the class effectively abstract,
    # by activating decorators of the kind @abc.abstractclass, etc.
    """
    A class to describe generic properties and functions of a component that can be single or
    made of child `Module`.

    Parameters
    ----------
    name: str
        Name of the `Module`

    Attributes
    ----------
    name : str
        `Module` name
    inputs : :obj:`collections.OrderedDict` of :obj:`ExtensiblePort`
        Dictionary of `ExtensiblePort` containing the values needed to compute the `Module`
    outputs : Dict[ExtensiblePort]
        Dictionary of `ExtensiblePort` containing the values computed by the `Module`
    children : Dict[str, Module]
        Child `Module` of this `Module`
    parent : Module
        Parent `Module` of this `Module`; None if there is no parent.
    exec_order : OrderedSet[str]
        Execution order in which the child `Module` should be solved.

    _active : bool
        If False, the `Module` will not execute its `run_once` method
    _is_clean : dict[bool]
        Reflects the status of the inputs and outputs. `clean` status means the group of ports was not updated since
        last computation of the `Module`
    _compute_calls: int
        Store if the number of times :py:meth:`~cosapp.core.module.Module.compute` was called (due to inhibition of clean status)

    Signals
    -------
    setup_ran : Signal()
        Signal emitted after :py:meth:`~cosapp.core.module.Module.call_setup_run` execution
    computed : Signal()
        Signal emitted after the :py:meth:`~cosapp.core.module.Module.compute` stack (= after :py:meth:`~cosapp.core.module.Module._post_compute`) execution
    clean_ran : Signal()
        Signal emitted after the :py:meth:`~cosapp.core.module.Module.call_clean_run` execution
    """

    __slots__ = (
        '__weakref__', '_name', 'children', 'parent', '__exec_order', '_active', '_compute_calls',
        'setup_ran', 'computed', 'clean_ran'
    )

    _name_check = NameChecker()

    def __init__(self, name: str):
        """`Module` constructor

        Parameters
        ----------
        name: str, optional
            Name of the `Module`
        """
        self._name = self._name_check(name)  # type: str
        self.children = collections.OrderedDict()
        self.parent = None  # type: Optional[Module]
        self.__exec_order = OrderedSet()  # type: OrderedSet[str]
        self._active = True  # type: bool

        self._compute_calls = 0  # type: int

        # Signals
        self.setup_ran = Signal(name="cosapp.core.module.Module.setup_ran")
        self.computed = Signal(name="cosapp.core.module.Module.computed")
        self.clean_ran = Signal(name="cosapp.core.module.Module.clean_ran")

    def __getattr__(self, name: str) -> Any:
        try:  # Faster than testing
            return self.children[name]
        except KeyError:
            return super().__getattribute__(name)

    @property
    def contextual_name(self) -> str:
        """str : Name of the module relative to the root one."""
        return "" if self.parent is None else self._get_fullname(skip_root=True)

    @property
    def compute_calls(self) -> int:
        """int : Number of calls to the compute method at last execution."""
        return self._compute_calls

    @property
    def name(self) -> str:
        """str : `Module` identifier."""
        return self._name

    @name.setter
    def name(self, name: str) -> NoReturn:
        self._name = self._name_check(name)

    @property
    def exec_order(self) -> "OrderedSet[str]":
        return self.__exec_order

    @exec_order.setter
    def exec_order(self, iterable) -> NoReturn:
        new_set = OrderedSet(iterable)
        if not all(isinstance(elem, str) for elem in new_set):
            raise TypeError(f"All elements of {self.name}.exec_order must be strings")
        self.__exec_order = new_set

    @property
    def size(self) -> int:
        """int : System size (= number of children + 1)."""
        size = 1
        for child in self.children.values():
            size += child.size
        return size

    def _get_fullname(self, skip_root: bool = False) -> str:
        """Get the fullname up to the root Module.
        
        Parameters
        ----------
        skip_root : bool
            Should the root Module name be skipped.

        Returns
        -------
        str
            The module fullname
        """
        current = self
        fullname = [current.name]
        while current.parent is not None:
            current = current.parent
            fullname.append(current.name)
        if skip_root:
            fullname.pop()
        return ".".join(reversed(fullname))

    def call_setup_run(self):
        """Execute `setup_run` recursively on all modules."""
        with self.log_context(" - call_setup_run"):
            logger.debug(f"Call {self.name}.setup_run")
            self._compute_calls = 0  # Reset the counter
            self.setup_run()
            for child in self.children.values():
                child.call_setup_run()
            self.setup_ran.emit()

    def setup_run(self):
        """Method called once before starting any simulation."""
        pass  # pragma: no cover

    def call_clean_run(self):
        """Execute `clean_run` recursively on all modules."""
        with self.log_context(" - call_clean_run"):
            logger.debug(f"Call {self.name}.clean_run")
            self.clean_run()
            for child in self.children.values():
                child.call_clean_run()
            self.clean_ran.emit()

    def clean_run(self):
        """Method called once after any simulation."""
        pass  # pragma: no cover

    def get_path_to_child(self, from_child: "Module") -> str:
        """Get the roots of the current `Module`.

        Returns
        -------
        Roots
            The root name of the `Module`
        """
        s = from_child
        path = list()
        while s is not self:
            path.insert(0, s.name)
            s = s.parent
            if s is None:
                raise ValueError(
                    f"Module '{from_child.name}' is not a child of module '{self.name}'."
                )
        return ".".join(path)

    def add_child(self,
        child: "Module",
        execution_index: Optional[int] = None
    ) -> "Module":
        """Add a child `Module` to the current `Module`.

        When adding a child `Module`, it is possible to specified its position in the execution
        order.

        Parameters
        ----------
        child: Module
            `Module` to add to the current `Module`
        execution_index: int, optional
            Index of the execution order list at which the `Module` should be inserted;
            default latest.
        """
        # Type validation
        if not isinstance(child, Module):
            raise TypeError(
                "Argument 'child' should be of type Module; got {}.".format(
                    type(child).__name__
                )
            )
        if not isinstance(execution_index, (type(None), int)):
            raise TypeError(
                "Argument 'execution_index' should be of type int; got {}.".format(
                    type(execution_index).__name__
                )
            )

        if child.name in self.children:
            raise ValueError(
                "{} {!s} cannot be added, as Module already contains an object with the same name"
                "".format(type(child).__qualname__, child.name)
            )

        child.parent = self
        children = self.children
        children[child.name] = child

        if execution_index is None:
            self.__exec_order.append(child.name)
        else:
            self.__exec_order.insert(execution_index, child.name)

        return child

    def pop_child(self, name: str) -> Optional["Module"]:
        """Remove the `Module` called `name` from the current top `Module`.

        Parameters
        ----------
        name: str
            Name of the `Module` to remove

        Returns
        -------
        `Module` or None
            The removed `Module` or None if no match found
        """
        children = self.children

        if name not in children:
            message = "Component {} is not a children of {}.".format(name, self)
            logger.error(message)
            raise AttributeError(message)

        child = children.pop(name)
        child.parent = None

        self.__exec_order.remove(name)

        return child

    def _set_execution_order(self):
        # TODO doc + unit test

        # if self.execution_algorithm != ExecutionOrdering.MANUAL:
        #     pass  # TODO implement alternative
        # Warning the execution order should not be set for standalone sub-multisystems. They
        # should take care of them self.

        # Check if all the children are listed in execution order list. Warn if not
        for key, component in self.children.items():
            if key not in self.exec_order:
                logger.info(
                    f"Missing module '{key}' has been appended to the execution order list."
                )
                self.exec_order.add(key)
            component._set_execution_order()

        if len(self.children) != len(self.exec_order):
            for c in self.exec_order:
                if c not in self.children:
                    logger.info(
                        f"Module {c} in execution order list without component matching. The entry will be removed."
                    )
                    self.exec_order.discard(c)

    def _precompute(self) -> NoReturn:
        """Actions performed prior to the `Module.compute` call."""
        pass

    def compute_before(self) -> NoReturn:
        """Contains the customized `Module` calculation, to execute before children."""
        pass

    def compute(self) -> NoReturn:
        """Contains the customized `Module` calculation, to execute after children."""
        pass

    def _postcompute(self) -> NoReturn:
        """Actions performed after the `Module.compute` call."""
        pass

    def run_once(self) -> NoReturn:
        """Run the module once.

        Execute the model of this `Module` and its children in the execution order.

        Notes
        -----

        The driver are not executed when calling this method; only the physical model.
        """
        with self.log_context(" - run_once"):
            if self.is_active():
                self._precompute()

                logger.debug(f"Call {self.name}.compute_before")
                self.compute_before()

                for child in self.exec_order:
                    logger.debug(f"Call {self.name}.{child}.run_once")
                    self.children[child].run_once()

                logger.debug(f"Call {self.name}.compute")
                self._compute_calls += 1
                self.compute()

                self._postcompute()
                self.computed.emit()
            else:
                logger.debug(f"Skip {self.name} execution - Inactive")

    def is_active(self) -> bool:
        """Is this Module execution activated?

        Returns
        -------
        bool
            Activation status
        """
        return self._active

    @abc.abstractmethod
    def is_standalone(self) -> bool:
        """Is this Module able to solve itself?

        Returns
        -------
        bool
            Ability to solve the module or not.
        """
        pass

    def log_debug_message(
        self,
        handler: "HandlerWithContextFilters",
        record: logging.LogRecord,
        format: LogFormat = LogFormat.RAW,
    ) -> bool:
        """Callback method on the module to log more detailed information.
        
        This method will be called by the log handler when :py:meth:`~cosapp.utils.logging.LoggerContext.log_context`
        is active if the logging level is lower or equals to VERBOSE_LEVEL. It allows
        the object to send additional log message to help debugging a simulation.

        .. note::
            logger.log method cannot be used here. Use handler.handle(record)

        Parameters
        ----------
        handler : HandlerWithContextFilters
            Log handler on which additional message should be published.
        record : logging.LogRecord
            Log record
        format : LogFormat
            Format of the message

        Returns
        -------
        bool
            Should the provided record be logged?
        """
        message = record.getMessage()
        emit_record = super().log_debug_message(handler, record, format)

        if message.endswith("call_setup_run"):
            emit_record = False
        elif message.endswith("call_clean_run"):
            emit_record = False
            activate = getattr(record, "activate", None)
            if activate == False:
                # Display the number of system execution
                msg = f"Compute calls for {self._get_fullname()}: {self.compute_calls}"
                handler.log(LogLevel.DEBUG, msg, name=logger.name)

        return emit_record
