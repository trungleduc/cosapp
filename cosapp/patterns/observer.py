import abc
import inspect
import weakref
from typing import Dict, Any
from cosapp.utils.state_io import object__getstate__
class Observer(abc.ABC):
    """Generic interface for observers"""
    def __init__(self, subject=None):
        self._subject = None
        if subject is not None:
            self.observe(subject)

    @abc.abstractmethod
    def _update(self, *args, **kwargs) -> None:
        """Action to perform when notified by observed subject"""
        pass

    def observe(self, subject) -> None:
        """Sign in as observer of subject"""
        self.quit()
        subject.add(self)
        self._subject = weakref.ref(subject)

    def quit(self) -> None:
        """Quit observing whoever observer is currently observing"""
        if self._subject is not None:
            self._subject().remove(self)
        self._subject = None

    def observes(self, subject=None) -> bool:
        """Bool: does observer observe `subject`?
        If `subject` is None, returns True if observer observes anyone, False otherwise."""
        observing = (self._subject is not None)  # does observer observe anyone?
        if subject is None:
            return observing
        return observing and (self._subject() is subject)

    def __del__(self) -> None:
        try:
            object.__getattribute__(self, "_subject")
        except AttributeError:
            pass  # Quit should be skipped if `_subject` is already deleted
        else:
            self.quit()


class Subject:
    """
    Prototype of subject for Observer objects.
    The philosophy is that observers are responsible for signing in or out.
    Therefore, the subject should not unilaterally add or remove observers, except when it is cleared.
    """
    __obs_type = Observer

    def __new__(cls, obs_type=Observer):
        """
        Constructor of class `Subject`, accepting observers of type `Observer`.
        Derived classes can be constructed with a more specialized
        definition of allowed observer types, by specifying `obs_type`.
        """
        cls.__obs_type = obs_type if inspect.isclass(obs_type) else type(obs_type)
        return super().__new__(cls)

    def __init__(self):
        self._observers = weakref.WeakSet()

    def __getstate__(self) -> Dict[str, Any]:
        """Creates a state of the object.

        The state type depend on the object, see
        https://docs.python.org/3/library/pickle.html#object.__getstate__
        for further details.

        Returns
        -------
        Dict[str, Any]:
            state
        """
        state = object__getstate__(self).copy()
        state.update({"_observers": (self._observers.data, self._observers._pending_removals, self._observers._iterating)})
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Sets the object from a provided state.

        Parameters
        ----------
        state : Dict[str, Any]
            State
        """
        self.__dict__.update(state)
        data, pending_removals, iterating = state.pop("_observers")
        observer = weakref.WeakSet()
        observer.data = data
        observer._pending_removals = pending_removals
        observer._iterating = iterating

        self.__dict__.update({"_observers": observer})

    def __json__(self) -> Dict[str, Any]:
        """Creates a JSONable dictionary representation of the object.
        
        Returns
        -------
        Dict[str, Any]
            The dictionary
        """
        d = self.__dict__.copy()
        d.pop("_observers")
        return d

    @classmethod
    def observer_type(cls) -> type:
        """Returns the type of observers allowed to observe Subject"""
        return cls.__obs_type
    
    def notify(self, *args, **kwargs) -> None:
        """Notify observers that they must update"""
        for observer in self._observers:
            observer._update(*args, **kwargs)
    
    def add(self, observer):
        """
        Add an observer to the list of observers.
        Invoked by outside observers when they sign in; should not be called by self directly.
        """
        otype = self.observer_type()
        if not isinstance(observer, otype):
            cls_name = self.__class__.__name__
            raise TypeError(f"{cls_name} can only be observed by objects of type {otype.__name__}")
        self._observers.add(observer)

    def remove(self, observer) -> None:
        self._observers.remove(observer)

    def clear(self) -> None:
        """Force all observers to quit"""
        for observer in self._observers.copy():
            observer.quit()

    @property
    def n_observers(self) -> int:
        """int: number of observers of current subject"""
        return len(self._observers)

    def __del__(self) -> None:
        self.clear()
